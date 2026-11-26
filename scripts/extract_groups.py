#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_groups.py -- Route B: per-group FBX splitter for Shogun take FBXs.

Splits a full-scene Shogun 1.13 FBX (subjects + props + System +
Unlabeled_Markers) into per-group FBX files, each containing one subject,
its assigned props, and shared furniture (props that go into every group).

Output files are BYTE-EXACT subsets of the source: every kept node is copied
verbatim (payloads untouched); only container end-offsets are re-based.
No re-solve, no re-label, no value loss.

Binary FBX 7.7 layout (verified against Shogun 1.13 exports):
  file  = 23B magic + int32 version(7700) + top-level nodes + zero tail
  node  = int64 end_off | int64 nprops | int64 proplen | u8 namelen | name
          | props(proplen bytes) | children (up to end_off)
  term  = node with all-zero header
  prop  = 1 type byte + value:
            C=bool(1B) I=int32 L=int64 Y=int16 D=double F=float
            S=int32 len + bytes   R=int32 len + bytes
            d/f/i/l = arrays: int32 count + int32 encoding + int32 bytelen
                       + blob (zlib if encoding==1)

Commands:
  analyze <fbx>                          per-branch motion stats (JSON)
  infer   <data_root> -o grouping.json [--prefix G01-]
                                          data-driven grouping from all takes
  export  <fbx> -g grouping.json [--only NAME]
  batch   <data_root> -g grouping.json [--prefix G01-]
                                          export all <prefix>*.fbx takes
  verify  <src.fbx> <out.fbx>            integrity check of a split output

Grouping JSON:
  {
    "furniture": ["desk", ...],              # included in EVERY group
    "groups": {"lzh": ["white_mouse", ...],
               "zfy": ["white_teaport", ...]},
    "per_take": {"G01-INFO_X_TEA_IN02":
                     {"lzh": [...], "zfy": [...]}}   # optional override
  }
Props listed nowhere are excluded from every group file (a warning is
emitted), so the config is the single source of truth for membership.
"""
import argparse
import json
import os
import struct
import sys
import zlib
from bisect import bisect_left
from collections import defaultdict, deque

MAGIC = b'Kaydara FBX Binary  \x00\x1A\x00'
VARIANCE_EPS = 1e-4  # cm; below this a curve counts as constant


# --------------------------------------------------------------------------
# raw binary FBX 7.7 parsing
# --------------------------------------------------------------------------

class Node:
    __slots__ = ('name', 'nstart', 'off', 'nprops', 'proplen', 'cstart',
                 'children', 'uid')

    def __init__(self, name, nstart, off, nprops, proplen, cstart):
        self.name = name
        self.nstart = nstart
        self.off = off
        self.nprops = nprops
        self.proplen = proplen
        self.cstart = cstart
        self.children = []
        self.uid = None


def parse_node(buf, p):
    off, nprops, proplen = struct.unpack_from('<qqq', buf, p)
    namelen = buf[p + 24]
    if off == 0 and nprops == 0 and proplen == 0 and namelen == 0:
        return None
    name = buf[p + 25:p + 25 + namelen]
    body = p + 25 + namelen
    return Node(name, p, off, nprops, proplen, body + proplen)


def parse_tree(buf, start, end):
    nodes = []
    p = start
    while p + 25 <= end:
        n = parse_node(buf, p)
        if n is None:
            break
        n.children = parse_tree(buf, n.cstart, n.off)
        nodes.append(n)
        p = n.off
    return nodes


def parse_file(path):
    with open(path, 'rb') as stream:
        buf = stream.read()
    if buf[:len(MAGIC)] != MAGIC:
        raise ValueError(f'not a binary FBX: {path}')
    version = struct.unpack_from('<I', buf, len(MAGIC))[0]
    tops = parse_tree(buf, len(MAGIC) + 4, len(buf))
    return buf, version, tops


def clean_name(b):
    return b.split(b'\x00')[0].decode('utf-8', 'replace')


# --------------------------------------------------------------------------
# property decoding
# --------------------------------------------------------------------------

_SCALAR = {0x43: ('<?', 1), 0x49: ('<i', 4), 0x4C: ('<q', 8),
           0x59: ('<h', 2), 0x44: ('<d', 8), 0x46: ('<f', 4)}
_ARRAY = {0x64: ('<d', 8), 0x66: ('<f', 4), 0x69: ('<i', 4),
          0x6C: ('<q', 8), 0x43: ('<?', 1)}


def decode_props(buf, node):
    """Decode a node's property payload -> [(typechar, value)]."""
    vals = []
    pr = buf[node.cstart - node.proplen:node.cstart]
    i = 0
    while i < len(pr):
        t = pr[i]
        i += 1
        if t in _SCALAR:
            fmt, size = _SCALAR[t]
            vals.append((chr(t), struct.unpack_from(fmt, pr, i)[0]))
            i += size
        elif t == 0x53 or t == 0x52:  # S str / R bytes
            n = struct.unpack_from('<i', pr, i)[0]
            i += 4
            raw = pr[i:i + n]
            i += n
            vals.append((chr(t), raw.decode('utf-8', 'replace')
                         if t == 0x53 else raw))
        elif t in _ARRAY:
            fmt, size = _ARRAY[t]
            count = struct.unpack_from('<i', pr, i)[0]
            enc = struct.unpack_from('<i', pr, i + 4)[0]
            blen = struct.unpack_from('<i', pr, i + 8)[0]
            i += 12
            blob = pr[i:i + blen]
            i += blen
            if enc:
                blob = zlib.decompress(blob)
            vals.append((chr(t), [struct.unpack_from(fmt, blob, j * size)[0]
                                  for j in range(count)]))
        else:
            raise ValueError(f'unknown property type 0x{t:02x} in '
                             f'{node.name!r}')
    return vals


# --------------------------------------------------------------------------
# scene graph
# --------------------------------------------------------------------------

class Graph:
    pass


def build_graph(buf, tops):
    g = Graph()
    g.tops = {clean_name(t.name): t for t in tops}
    obj_children = g.tops['Objects'].children
    for c in obj_children:
        pr = decode_props(buf, c)
        c.uid = pr[0][1]  # every object node starts with int64 uid
    g.obj_children = obj_children
    g.uid2node = {c.uid: c for c in obj_children}

    g.model_names = {}
    for c in obj_children:
        if c.name == b'Model':
            pr = decode_props(buf, c)
            g.model_names[c.uid] = (clean_name(pr[1][1].encode())
                                    if isinstance(pr[1][1], str)
                                    else clean_name(pr[1][1]), pr[2][1])

    conns = []
    for c in g.tops['Connections'].children:
        pr = decode_props(buf, c)
        kind = pr[0][1]
        if kind == 'OO':
            conns.append(('OO', pr[1][1], pr[2][1]))
        else:
            conns.append(('OP', pr[1][1], pr[2][1],
                          pr[3][1] if len(pr) > 3 else ''))
    g.conns = conns
    child2 = defaultdict(list)
    for conn in conns:
        if conn[0] == 'OO':
            child2[conn[2]].append(conn[1])
    g.child2 = child2

    g.roots = {}
    for a in child2.get(0, []):
        if a in g.uid2node and g.uid2node[a].name == b'Model':
            nm = g.model_names[a][0]
            g.roots[nm] = a

    ac_uids = {c.uid for c in obj_children if c.name == b'AnimationCurve'}
    acn_uids = {c.uid for c in obj_children
                if c.name == b'AnimationCurveNode'}
    model_uids = {u for u in g.model_names}
    g.ac_uids = ac_uids
    g.acn_uids = acn_uids

    g.acn_name = {c.uid: clean_name(c.name) for c in obj_children
                  if c.name == b'AnimationCurveNode'}
    g.model2curves = defaultdict(dict)   # model_uid -> {(attr, chan): ac_uid}
    g.ac2acn = {}                        # ac_uid -> (acn_uid, chan)
    acn2acs = defaultdict(dict)          # acn_uid -> {chan: ac_uid}
    for conn in conns:
        if conn[0] == 'OP' and conn[1] in ac_uids and conn[2] in acn_uids:
            g.ac2acn[conn[1]] = (conn[2], conn[3])
            acn2acs[conn[2]][conn[3]] = conn[1]
    for conn in conns:
        if conn[0] != 'OP':
            continue
        src, dst, member = conn[1], conn[2], conn[3]
        if src in acn_uids and dst in model_uids:
            for chan, ac in acn2acs.get(src, {}).items():
                g.model2curves[dst][(member, chan)] = ac
    return g


def branch_uids(g, root_uid, include_root=True):
    seen = {root_uid} if include_root else set()
    q = deque(g.child2.get(root_uid, []))
    while q:
        n = q.popleft()
        if n not in seen:
            seen.add(n)
            q.extend(g.child2.get(n, []))
    return seen


def keep_set_for_group(g, members):
    keep = set()
    for m in members:
        if m not in g.roots:
            raise KeyError(f'subject/prop not found in FBX roots: {m!r}')
        keep |= branch_uids(g, g.roots[m])
    # pull in animation data: OP connections point animation -> model
    # (AC -> ACN -> Model), so a kept model must pull in its ACN/ACs.
    changed = True
    while changed:
        changed = False
        for conn in g.conns:
            if conn[0] == 'OP' and conn[2] in keep and conn[1] not in keep:
                keep.add(conn[1])
                changed = True
    keep &= set(g.uid2node)
    # AnimationStack/AnimationLayer form a separate OO branch
    # (stack -> layer -> every AnimationCurveNode) with no OP links into
    # models, so the closure above never reaches them.  three.js builds
    # its animation clips from exactly those nodes (OO children of the
    # stack/layer), so keep them in every group file.
    for c in g.obj_children:
        if c.name in (b'AnimationStack', b'AnimationLayer'):
            keep.add(c.uid)
    return keep


def curve_values(buf, g, ac_uid):
    node = g.uid2node[ac_uid]
    kt = kv = None
    for ch in node.children:
        if ch.name == b'KeyTime' and ch.proplen:
            pr = decode_props(buf, ch)
            kt = pr[0][1]
        elif ch.name == b'KeyValueFloat' and ch.proplen:
            pr = decode_props(buf, ch)
            kv = pr[0][1]
    return kt, kv


def model_t_points(buf, g, model_uid):
    """World-frame (time, x, y, z) samples from Lcl Translation curves."""
    pts = []
    curves = g.model2curves.get(model_uid, {})
    tcurves = {}
    for (attr, chan), ac in curves.items():
        if attr == 'Lcl Translation' and chan in ('d|X', 'd|Y', 'd|Z'):
            tcurves[chan] = ac
    if len(tcurves) < 3:
        return None
    data = {ch: curve_values(buf, g, ac) for ch, ac in tcurves.items()}
    times = data['d|X'][0]
    for ch in ('d|Y', 'd|Z'):
        if data[ch][0] is None or len(data[ch][0]) != len(times):
            return None
    for i, t in enumerate(times):
        pts.append((t, data['d|X'][1][i], data['d|Y'][1][i],
                    data['d|Z'][1][i]))
    return pts or None


# --------------------------------------------------------------------------
# analysis / inference
# --------------------------------------------------------------------------

def branch_stats(buf, g, root_name):
    root = g.roots[root_name]
    br = branch_uids(g, root)
    models = [u for u in br if u in g.model_names]
    markers = [u for u in models if g.model_names[u][1] == 'Marker']
    limbs = [u for u in models if g.model_names[u][1] == 'LimbNode']
    nulls = [u for u in models if g.model_names[u][1] not in
             ('Marker', 'LimbNode')]
    varying = 0
    anchor_uid, anchor_range = None, 0.0
    hips_pts = None
    for m in models:
        curves = g.model2curves.get(m, {})
        node_vary = False
        t_range = 0.0
        for (attr, chan), ac in curves.items():
            if ac not in g.ac_uids:
                continue
            kt, kv = curve_values(buf, g, ac)
            if kv:
                rng = max(kv) - min(kv)
                if rng > VARIANCE_EPS:
                    node_vary = True
                    if attr == 'Lcl Translation':
                        t_range = max(t_range, rng)
        if node_vary:
            varying += 1
        if t_range > anchor_range:
            anchor_range = t_range
            anchor_uid = m
        nm = g.model_names[m][0]
        if root_name in ('lzh', 'zfy') and nm.endswith('Hips'):
            hips_pts = model_t_points(buf, g, m)
    return {
        'root': root_name,
        'nodes': len(br),
        'models': len(models),
        'markers': len(markers),
        'limbs': len(limbs),
        'animated': varying,
        'moving': anchor_range > VARIANCE_EPS,
        'anchor_range_cm': round(anchor_range, 2),
        'hips': hips_pts,
        'anchor_uid': anchor_uid,
    }


def anchor_points(buf, g, root_name):
    br = branch_uids(g, g.roots[root_name])
    models = [u for u in br if u in g.model_names]
    best, best_rng = None, 0.0
    for m in models:
        nm = g.model_names[m][0]
        if nm == 'Hips' or nm.endswith(':Hips'):
            pts = model_t_points(buf, g, m)
            if pts:
                return pts, _total_range(pts)
    for m in models:
        pts = model_t_points(buf, g, m)
        if not pts:
            continue
        rng = _total_range(pts)
        if rng > best_rng:
            best_rng, best = rng, pts
    return best, best_rng


def _total_range(pts):
    return (max(p[1] for p in pts) - min(p[1] for p in pts)
            + max(p[2] for p in pts) - min(p[2] for p in pts)
            + max(p[3] for p in pts) - min(p[3] for p in pts))



def find_takes(data_root, prefix='G01-'):
    takes = []
    for dirpath, _dirs, files in os.walk(data_root):
        for f in files:
            if f.endswith('.fbx') and f.startswith(prefix) and \
               not f.endswith('.bak') and \
               os.path.exists(os.path.join(dirpath, f[:-4] + '.mcp')):
                takes.append(os.path.join(dirpath, f))
    return sorted(takes)


MOTION_THRESH_CM = 1.0     # per-take displacement above this = the prop moved
FURNITURE_MAX_CM = 2.0     # max displacement over ALL takes <= this = furniture *candidate* (reference only)
FRAME_V_THRESH_CM = 0.1    # per-frame displacement marking a "moving" frame


def motion_frame_dist(pts, subj_pts, v_thresh=FRAME_V_THRESH_CM):
    """Median prop-to-subject distance over frames where the prop moves."""
    if not pts or not subj_pts:
        return None
    st = [p[0] for p in subj_pts]
    ds = []
    for i in range(len(pts) - 1):
        dx = pts[i + 1][1] - pts[i][1]
        dy = pts[i + 1][2] - pts[i][2]
        dz = pts[i + 1][3] - pts[i][3]
        if (dx * dx + dy * dy + dz * dz) ** 0.5 <= v_thresh:
            continue
        j = min(bisect_left(st, pts[i][0]), len(st) - 1)
        _, bx, by, bz = subj_pts[j]
        ds.append(((pts[i][1] - bx) ** 2 + (pts[i][2] - by) ** 2
                   + (pts[i][3] - bz) ** 2) ** 0.5)
    if not ds:
        return None
    ds.sort()
    return ds[len(ds) // 2]


def infer_grouping(data_root, out_path, thresh=None, verbose=True,
                   prefix='G01-'):
    thresh = MOTION_THRESH_CM if thresh is None else thresh
    takes = find_takes(data_root, prefix)
    if verbose:
        print(f'infer: {len(takes)} takes, motion threshold {thresh} cm',
              file=sys.stderr)
    prop_stats = defaultdict(lambda: {'ranges': {}, 'take_votes': [],
                                      'dists': defaultdict(list)})
    for take in takes:
        base = os.path.basename(take)[:-4]
        if verbose:
            print(f'  {base}', file=sys.stderr, flush=True)
        buf, ver, tops = parse_file(take)
        g = build_graph(buf, tops)
        subjects, props = [], []
        for r in g.roots:
            if r in ('System', 'Unlabeled_Markers'):
                continue
            br = branch_uids(g, g.roots[r])
            has_hips = any(g.model_names.get(u, ('',))[0].endswith('Hips')
                           for u in br if u in g.model_names)
            (subjects if has_hips else props).append(r)
        subj_pts = {}
        for s in subjects:
            pts, _rng = anchor_points(buf, g, s)
            subj_pts[s] = pts
        for p in props:
            pts, rng = anchor_points(buf, g, p)
            prop_stats[p]['ranges'][base] = round(rng, 2)
            if rng <= thresh or not pts:
                continue
            # over the frames the prop actually moves, who is it next to?
            best_s, best_d = None, None
            for s in subjects:
                d = motion_frame_dist(pts, subj_pts[s])
                if d is None:
                    continue
                prop_stats[p]['dists'][s].append(round(d, 1))
                if best_d is None or d < best_d:
                    best_s, best_d = s, d
            if best_s:
                prop_stats[p]['take_votes'].append((base, best_s))
    diag = {}
    for p in sorted(prop_stats):
        st = prop_stats[p]
        rs = list(st['ranges'].values())
        diag[p] = {
            'max_range_cm': max(rs) if rs else 0,
            'take_votes': st['take_votes'],
            'motion_dists': {s: {'n': len(v),
                                 'median_cm': sorted(v)[len(v) // 2]}
                             for s, v in st['dists'].items()},
            'per_take': st['ranges'],
        }
    with open(out_path + '.diag.json', 'w', encoding='utf-8') as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)
    # furniture is NOT inferred: the config's "furniture" list (user-written)
    # is the single source of truth.  We only draft "groups" from votes and
    # report still props as *candidates* so the user can copy them in.
    groups = defaultdict(list)
    furniture_candidates, unassigned = [], []
    for p in sorted(prop_stats):
        st = prop_stats[p]
        rs = list(st['ranges'].values())
        if not rs or max(rs) <= FURNITURE_MAX_CM:
            furniture_candidates.append(p)
            continue
        if not st['take_votes']:
            unassigned.append(p)
            continue
        tally = defaultdict(int)
        for _base, s in st['take_votes']:
            tally[s] += 1
        owner = min(tally, key=lambda s: (-tally[s],
                    sorted(st['dists'][s])[len(st['dists'][s]) // 2]
                    if s in st['dists'] else 1e9, s))
        groups[owner].append(p)
    table = {
        'furniture': [],
        'groups': {k: sorted(v) for k, v in sorted(groups.items())},
        'per_take': {},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f'infer: wrote {out_path} '
              f'("furniture" left empty - fill it in by hand)', file=sys.stderr)
        print(f'infer: still props, furniture candidates (not in table): '
              f'{furniture_candidates}', file=sys.stderr)
        if unassigned:
            print(f'infer: unassigned (moved, but no ownership vote): '
                  f'{unassigned}', file=sys.stderr)
        print(json.dumps(table, indent=2, ensure_ascii=False))
    return table


# --------------------------------------------------------------------------
# splicing (byte-exact subset)
# --------------------------------------------------------------------------

def emit_node(buf, out, node, children=None):
    start = len(out)
    out += struct.pack('<q', 0)  # end_off placeholder
    out += buf[node.nstart + 8:node.cstart]  # header rest + props, verbatim
    sub = (node.children if children is None else children)
    for c in sub:
        emit_node(buf, out, c)
    if sub:
        out += bytes(25)
    struct.pack_into('<q', out, start, len(out))


def splice(buf, tops, keep):
    out = bytearray()
    out += buf[:27]  # magic + version
    objs_top = None
    conns_top = None
    for t in tops:
        nm = clean_name(t.name)
        if nm == 'Objects':
            objs_top = t
        elif nm == 'Connections':
            conns_top = t
    kept_objs = [c for c in objs_top.children if c.uid in keep]
    kept_conns = []
    for c in conns_top.children:
        pr = decode_props(buf, c)
        kind = pr[0][1]
        if kind == 'OO':
            child, parent = pr[1][1], pr[2][1]
            if child in keep and (parent == 0 or parent in keep):
                kept_conns.append(c)
        else:
            src, dst = pr[1][1], pr[2][1]
            if src in keep and dst in keep:
                kept_conns.append(c)
    for t in tops:
        nm = clean_name(t.name)
        if nm == 'Objects':
            emit_node(buf, out, t, kept_objs)
        elif nm == 'Connections':
            emit_node(buf, out, t, kept_conns)
        else:
            emit_node(buf, out, t)
    last = max(t.off for t in tops)
    out += buf[last:]  # zero tail (terminator + padding), verbatim
    return bytes(out), len(kept_objs), len(kept_conns)


def group_members(table, take_base, group):
    per = table.get('per_take', {}).get(take_base, {})
    if group in per:
        props = list(per[group])
    else:
        props = list(table.get('groups', {}).get(group, []))
    return [group] + props + list(table.get('furniture', []))


def export_one(fbx_path, table, out_dir=None, only=None):
    base = os.path.basename(fbx_path)[:-4]
    buf, ver, tops = parse_file(fbx_path)
    g = build_graph(buf, tops)
    out_dir = out_dir or os.path.dirname(fbx_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    written = []
    groups = [x for x in table.get('groups', {})
              if only is None or x == only]
    group_names = set(table.get('groups', {}))
    for pt in table.get('per_take', {}).values():
        group_names |= set(pt)
    all_props = set(table.get('furniture', [])) | group_names
    for gr in table.get('groups', {}):
        all_props |= set(table['groups'][gr])
    for pt in table.get('per_take', {}).values():
        for gr in pt:
            all_props |= set(pt[gr])
    unassigned = [r for r in g.roots
                  if r not in ('System', 'Unlabeled_Markers')
                  and r not in all_props]
    if unassigned:
        print(f'  warn: roots listed nowhere in config, excluded from all '
              f'groups: {unassigned}', file=sys.stderr)
    for gr in groups:
        members = group_members(table, base, gr)
        keep = keep_set_for_group(g, members)
        data, n_obj, n_conn = splice(buf, tops, keep)
        out_path = os.path.join(out_dir, f'{base}.{gr}.fbx')
        with open(out_path, 'wb') as f:
            f.write(data)
        written.append((out_path, len(data), n_obj, n_conn, members))
        print(f'  {gr}: {n_obj} objects, {n_conn} connections, '
              f'{len(data) / 1048576:.2f} MB -> {out_path}', file=sys.stderr)
    return written


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def verify(src_path, out_path, verbose=True):
    sbuf, sv, st = parse_file(src_path)
    obuf, ov, ot = parse_file(out_path)
    g_s = build_graph(sbuf, st)
    g_o = build_graph(obuf, ot)
    s_uids = set(g_s.uid2node)
    o_uids = set(g_o.uid2node)
    problems = []
    if not o_uids <= s_uids:
        problems.append(f'out has {len(o_uids - s_uids)} uids not in src')
    # payload byte-identity for every kept node
    bad_payload = 0
    for u, on in g_o.uid2node.items():
        sn = g_s.uid2node[u]
        if on.nprops != sn.nprops or on.proplen != sn.proplen or \
           on.name != sn.name:
            bad_payload += 1
            continue
        if sbuf[sn.nstart + 8:sn.cstart] != obuf[on.nstart + 8:on.cstart]:
            bad_payload += 1
    if bad_payload:
        problems.append(f'{bad_payload} nodes with altered header/name/payload')
    # connection integrity
    o_set = set()
    dangling = 0
    for c in g_o.tops['Connections'].children:
        pr = decode_props(obuf, c)
        if pr[0][1] == 'OO':
            if pr[1][1] not in o_uids or (pr[2][1] != 0 and pr[2][1] not in o_uids):
                dangling += 1
            o_set.add(('OO', pr[1][1], pr[2][1]))
        else:
            if pr[1][1] not in o_uids or pr[2][1] not in o_uids:
                dangling += 1
            o_set.add(('OP', pr[1][1], pr[2][1],
                       pr[3][1] if len(pr) > 3 else ''))
    if dangling:
        problems.append(f'{dangling} dangling connections')
    # exact connection subset: every src conn with endpoints in out must be kept
    s_set = set()
    for c in g_s.tops['Connections'].children:
        pr = decode_props(sbuf, c)
        if pr[0][1] == 'OO':
            s_set.add(('OO', pr[1][1], pr[2][1]))
        else:
            s_set.add(('OP', pr[1][1], pr[2][1],
                       pr[3][1] if len(pr) > 3 else ''))
    expected = set()
    for conn in s_set:
        if conn[0] == 'OO':
            _, a, b = conn
            if a in o_uids and (b == 0 or b in o_uids):
                expected.add(conn)
        else:
            _, a, b, m = conn
            if a in o_uids and b in o_uids:
                expected.add(conn)
    if o_set != expected:
        problems.append(f'connection set mismatch: '
                        f'missing={len(expected - o_set)} extra={len(o_set - expected)}')
    # curve value spot-check (10 random kept curves)
    import random
    random.seed(42)
    acs = [u for u in o_uids if u in g_o.ac_uids]
    for u in random.sample(acs, min(10, len(acs))):
        a = curve_values(sbuf, g_s, u)
        b = curve_values(obuf, g_o, u)
        if a != b:
            problems.append(f'curve {u} values differ')
    # top-level structure
    if [clean_name(t.name) for t in st] != [clean_name(t.name) for t in ot]:
        problems.append('top-level node list differs')
    if verbose:
        print(f'verify {os.path.basename(out_path)}:')
        print(f'  src objects: {len(s_uids)} | out objects: {len(o_uids)}')
        print(f'  connections: src {len(s_set)} -> out {len(o_set)}')
        if problems:
            for p in problems:
                print(f'  FAIL: {p}')
            return False
        print(f'  PASS: byte-exact subset, refs intact, curves identical')
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('analyze')
    p.add_argument('fbx')

    p = sub.add_parser('infer')
    p.add_argument('data_root')
    p.add_argument('-o', '--out', default='grouping_inferred.json')
    p.add_argument('--prefix', default='G01-',
                   help="take file name prefix (Shogun session name, default 'G01-')")

    p = sub.add_parser('export')
    p.add_argument('fbx')
    p.add_argument('-g', '--grouping', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--only', default=None)

    p = sub.add_parser('batch')
    p.add_argument('data_root')
    p.add_argument('-g', '--grouping', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--prefix', default='G01-',
                   help="take file name prefix (Shogun session name, default 'G01-')")

    p = sub.add_parser('verify')
    p.add_argument('src')
    p.add_argument('out')

    a = ap.parse_args()

    if a.cmd == 'analyze':
        buf, ver, tops = parse_file(a.fbx)
        g = build_graph(buf, tops)
        stats = {}
        for r in sorted(g.roots):
            stats[r] = branch_stats(buf, g, r)
        print(json.dumps(stats, indent=2))
    elif a.cmd == 'infer':
        infer_grouping(a.data_root, a.out, prefix=a.prefix)
    elif a.cmd == 'export':
        table = json.load(open(a.grouping, encoding='utf-8'))
        export_one(a.fbx, table, a.out, a.only)
    elif a.cmd == 'batch':
        table = json.load(open(a.grouping, encoding='utf-8'))
        takes = find_takes(a.data_root, a.prefix)
        print(f'batch: {len(takes)} takes', file=sys.stderr)
        ok = 0
        for take in takes:
            base = os.path.basename(take)[:-4]
            print(f'{base}', file=sys.stderr, flush=True)
            try:
                export_one(take, table, a.out)
                ok += 1
            except Exception as e:
                print(f'  ERROR: {e}', file=sys.stderr)
        print(f'batch done: {ok}/{len(takes)} ok', file=sys.stderr)
    elif a.cmd == 'verify':
        if not verify(a.src, a.out):
            sys.exit(1)


if __name__ == '__main__':
    main()

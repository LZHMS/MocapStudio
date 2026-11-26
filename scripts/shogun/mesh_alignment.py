"""Read-only, name-gated rigid calibration for Shogun binary FBX 7700.

Asset-world hole coordinates are millimetres. The returned rigid transform maps
these coordinates into the animated bone's local millimetres, without scaling
or reflection. Geometry is never used to search outside the named candidates.
Only XYZ Euler transforms with identity ancestors are evaluated; unsupported
FBX transform stacks are reported rather than approximated.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import struct
import re
from pathlib import Path

import numpy as np
from mesh_fbx import parse_checked

_TICKS_PER_SECOND = 46186158000
_AMBIGUITY_MM = 0.01
_ROOT_SUFFIX = ':Labeling:root'


def _normalized(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Names must be nonempty strings')
    return value.strip().casefold()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def _point(value):
    if isinstance(value, dict):
        value = [value[axis] for axis in 'xyz']
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Hole point must contain three finite coordinates')
    return point


def _holes(metadata):
    records = metadata.get('holes')
    if not isinstance(records, list) or not records:
        raise ValueError('Calibration requires a nonempty holes list')
    holes, identifiers = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Each hole must be an object')
        identifier = record.get('hole_id', record.get('marker', record.get('id')))
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('Hole identifiers must be nonempty and unique')
        identifiers.add(identifier)
        alternatives = []
        for key, scale in (('center_mm', 1.), ('mouth_mm', 1.),
                           ('point_mm', 1.), ('center_m', 1000.), ('mouth_m', 1000.)):
            if key in record:
                alternatives.append(_point(record[key]) * scale)
        if not alternatives:
            raise ValueError(f'No supported hole coordinates for {identifier}')
        if any(not np.allclose(alternatives[0], point, rtol=1e-6, atol=1e-4)
               for point in alternatives[1:]):
            raise ValueError(f'Contradictory millimetre/metre coordinates for {identifier}')
        holes.append({'id': identifier, 'point_mm': alternatives[0].tolist()})
    return holes


def load_catalog(asset_roots: Path | str | list[Path | str]) -> dict:
    """Discover original GLBs with valid adjacent calibration JSON, read-only.

    Accepts one root or multiple semicolon-separated/listed directories. Metadata
    historical paths are ignored; numeric calibration variants (e.g. camera_1)
    can share their base clean GLB.
    """
    if isinstance(asset_roots, (list, tuple)):
        raw_roots = asset_roots
    elif isinstance(asset_roots, Path):
        raw_roots = [asset_roots]
    else:
        raw_roots = str(asset_roots).split(';')
    roots = [Path(r).expanduser().resolve(strict=True) for r in raw_roots if str(r).strip()]
    if not roots:
        raise ValueError('At least one asset root directory is required')
    catalog, sources = {}, {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f'Asset root is not a directory: {root}')
        for path in sorted(root.rglob('*.json')):
            try:
                metadata = _json(path)
            except Exception:
                continue
            if not isinstance(metadata, dict) or 'holes' not in metadata:
                continue
            stem = path.stem.removesuffix('_track_print')
            if stem.casefold().endswith(('_track', '_track_print')):
                continue
            glb = path.with_name(stem + '.glb')
            if not glb.is_file():
                base_glb = path.with_name(re.sub(r'_\d+$', '', stem) + '.glb')
                if base_glb.is_file():
                    glb = base_glb
            if not glb.is_file():
                continue
            try:
                holes = _holes(metadata)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Invalid calibration {path}: {exc}') from exc
            identifier = (path.parent / stem).relative_to(root).as_posix()
            source = (glb.resolve(), path.resolve())
            if source not in sources:
                sources[source] = {
                    'id': identifier, 'glb': str(source[0]),
                    'calibration_json': str(source[1]), 'holes': holes,
                }
            catalog[identifier] = sources[source]
            if '/' in identifier:
                sub = identifier.split('/', 1)[1]
                if sub not in catalog:
                    catalog[sub] = catalog[identifier]
    return catalog


def load_mapping(path: Path) -> dict:
    """Read schema version 1; candidate existence is checked against each catalog."""
    data = _json(path)
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Mapping requires schema_version: 1')
    if not isinstance(data.get('props'), dict):
        raise ValueError('Mapping requires a props object')
    result = {}
    for name, entry in data['props'].items():
        key = _normalized(name)
        if key in result:
            raise ValueError(f'Duplicate normalized prop name: {key}')
        if not isinstance(entry, dict):
            raise ValueError(f'{name}: mapping entry must be an object')
        candidates = entry.get('candidates')
        if (not isinstance(candidates, list) or not candidates
                or any(not isinstance(item, str) or not item for item in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError(f'{name}: candidates must be nonempty unique asset IDs')
        copied = {'candidates': list(candidates)}
        if 'marker_map' in entry:
            marker_map = entry['marker_map']
            if (not isinstance(marker_map, dict) or not marker_map
                    or any(marker not in ('M_1', 'M_2', 'M_3', 'M_4')
                           or not isinstance(hole, str) or not hole
                           for marker, hole in marker_map.items())
                    or len(set(marker_map.values())) != len(marker_map)):
                raise ValueError(f'{name}: marker_map must map unique M_1..M_4 to hole IDs')
            copied['marker_map'] = dict(marker_map)
        if 'bone_offset_mm' in entry:
            bone_offset = entry['bone_offset_mm']
            if (not isinstance(bone_offset, list) or len(bone_offset) != 3
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bone_offset)):
                raise ValueError(f'{name}: bone_offset_mm must be a list of 3 finite numbers')
            copied['bone_offset_mm'] = [float(v) for v in bone_offset]
        result[key] = copied
    return result


def _reader():
    path = Path(__file__).resolve().parents[1] / 'extract_groups.py'
    spec = importlib.util.spec_from_file_location('_mesh_alignment_fbx_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PropError(ValueError):
    def __init__(self, reason, details):
        super().__init__(details)
        self.reason = reason
        self.details = details


def _properties(reader, buf, node):
    result = {}
    for group in node.children:
        if group.name == b'Properties70':
            for child in group.children:
                values = reader.decode_props(buf, child)
                if len(values) >= 4:
                    result[values[0][1]] = [value for _, value in values[4:]]
    return result


def _vector(properties, key, default):
    value = np.asarray(properties.get(key, default), dtype=float)
    if value.shape != (len(default),) or not np.isfinite(value).all():
        raise _PropError('UNSUPPORTED_TRANSFORM', f'Invalid transform property {key}')
    return value


def _check_transform(reader, buf, graph, uid, *, identity=False):
    properties = _properties(reader, buf, graph.uid2node[uid])
    for attr in ('PreRotation', 'PostRotation', 'RotationPivot', 'ScalingPivot',
                 'RotationOffset', 'ScalingOffset'):
        if not np.allclose(_vector(properties, attr, [0., 0., 0.]), 0., atol=1e-10):
            raise _PropError('UNSUPPORTED_TRANSFORM', f'{uid}: nonzero {attr}')
    if not np.allclose(_vector(properties, 'Lcl Scaling', [1., 1., 1.]), 1., atol=1e-10):
        raise _PropError('UNSUPPORTED_TRANSFORM', f'{uid}: nonidentity scaling')
    if properties.get('RotationOrder', [0]) != [0]:
        raise _PropError('UNSUPPORTED_TRANSFORM', f'{uid}: non-XYZ rotation order')
    if properties.get('InheritType', [0]) not in ([0], [1], [2]):
        raise _PropError('UNSUPPORTED_TRANSFORM', f'{uid}: unsupported inheritance')
    attrs = {attr for attr, _ in graph.model2curves.get(uid, {})}
    if attrs - {'Lcl Translation', 'Lcl Rotation'}:
        raise _PropError('UNSUPPORTED_TRANSFORM', f'{uid}: animated non-rigid transform')
    if identity and (attrs or any(not np.allclose(_vector(properties, attr, [0., 0., 0.]),
                                                 0., atol=1e-10)
                                for attr in ('Lcl Translation', 'Lcl Rotation'))):
        raise _PropError('UNSUPPORTED_TRANSFORM', f'{uid}: nonidentity/animated ancestor')


def _curves(reader, buf, graph, uid, attr):
    channels = graph.model2curves.get(uid, {})
    nodes = [conn[1] for conn in graph.conns
             if conn[0] == 'OP' and conn[2] == uid and conn[3] == attr
             and conn[1] in graph.acn_uids]
    if len(nodes) != 1:
        raise _PropError('INVALID_CURVES', f'{uid}: missing or layered/duplicate {attr}')
    channel_counts = {}
    for conn in graph.conns:
        if conn[0] == 'OP' and conn[2] == nodes[0] and conn[1] in graph.ac_uids:
            channel_counts[conn[3]] = channel_counts.get(conn[3], 0) + 1
    if any(channel_counts.get('d|' + axis) != 1 for axis in 'XYZ'):
        raise _PropError('INVALID_CURVES', f'{uid}: missing/duplicate {attr} channels')
    records = []
    for axis in 'XYZ':
        curve = channels.get((attr, 'd|' + axis))
        if curve is None:
            raise _PropError('INVALID_CURVES', f'{uid}: missing {attr} {axis}')
        times, values = reader.curve_values(buf, graph, curve)
        if times is None or values is None or not times or len(times) != len(values):
            raise _PropError('INVALID_CURVES', f'{uid}: missing/unequal curve arrays')
        times = np.asarray(times, dtype=np.int64)
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
            raise _PropError('INVALID_CURVES', f'{uid}: nonfinite values or unordered key times')
        if records and not np.array_equal(times, records[0][0]):
            raise _PropError('INVALID_CURVES', f'{uid}: unequal XYZ key times')
        records.append((times, values))
    return records[0][0], np.stack([record[1] for record in records], axis=1)


def _xyz_rotations(euler):
    # FBX XYZ means Rz @ Ry @ Rx acting on column vectors.
    angles = np.deg2rad(euler)
    cx, cy, cz = np.cos(angles).T
    sx, sy, sz = np.sin(angles).T
    result = np.empty((len(euler), 3, 3))
    result[:, 0, 0] = cz * cy
    result[:, 0, 1] = cz * sy * sx - sz * cx
    result[:, 0, 2] = cz * sy * cx + sz * sx
    result[:, 1, 0] = sz * cy
    result[:, 1, 1] = sz * sy * sx + cz * cx
    result[:, 1, 2] = sz * sy * cx - cz * sx
    result[:, 2, 0] = -sy
    result[:, 2, 1] = cy * sx
    result[:, 2, 2] = cy * cx
    return result


def _fit(points, reference):
    center_a, center_b = points.mean(axis=0), reference.mean(axis=0)
    u, _, vt = np.linalg.svd((points - center_a).T @ (reference - center_b))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = center_b - rotation @ center_a
    errors = np.linalg.norm(points @ rotation.T + translation - reference, axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), rotation, translation


def _stats(errors):
    return {'rms_mm': float(np.sqrt(np.mean(errors ** 2))),
            'p99_mm': float(np.quantile(errors, .99)), 'max_mm': float(errors.max())}


def _candidates(prop, catalog, mapping):
    key = _normalized(prop)
    if key in mapping:
        entry = mapping[key]
        identifiers = entry['candidates']
        marker_map = entry.get('marker_map', {})
        bone_offset = entry.get('bone_offset_mm', [0.0, 0.0, 0.0])
    else:
        identifiers = [identifier for identifier in catalog if identifier.casefold() == key]
        if not identifiers:
            # Distinct source assets with the same stem remain ambiguous.
            identifiers = [identifier for identifier in catalog
                           if identifier.rsplit('/', 1)[-1].casefold() == key]
        marker_map, bone_offset = {}, [0.0, 0.0, 0.0]
    # Catalog lookup aliases share one canonical source ID. Plans and extraction
    # requests must use that same ID, and aliases cannot compete as separate fits.
    candidates = list(dict.fromkeys(catalog[identifier]['id'] if identifier in catalog
                                    else identifier for identifier in identifiers))
    return candidates, marker_map, bone_offset


def _find_stable_reference_window(local: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Find a stable 4-5 frame window in the middle of the sequence to compute reference pose."""
    F = len(local)
    if F <= window_size:
        return np.median(local, axis=0)
    start_f = max(0, int(F * 0.15))
    end_f = min(F, int(F * 0.85))
    if end_f - start_f < window_size:
        start_f, end_f = 0, F
    best_var = 1e9
    best_start = start_f
    found = False
    for f in range(start_f, end_f - window_size + 1):
        sub = local[f:f + window_size]
        if not np.isfinite(sub).all():
            continue
        dists = [np.std(np.linalg.norm(sub[:, i] - sub[:, j], axis=1))
                 for i in range(4) for j in range(i + 1, 4)]
        var = float(np.mean(dists))
        if var < best_var:
            best_var = var
            best_start = f
            found = True
            if var < 0.05:
                break
    if found:
        return np.median(local[best_start:best_start + window_size], axis=0)
    return np.median(local[:min(120, F)], axis=0)

def _analyze_prop(reader, buf, graph, prop, roots, named, parents, catalog,
                  candidate_ids, marker_map, bone_offset, native_unit_mm, max_rms_mm,
                  max_outlier_multiplier: float = 3.0):
    prefix = prop + ':'
    branches = set(roots)
    for uid, (name, _) in graph.model_names.items():
        if name.startswith(prefix):
            branches.update(reader.branch_uids(graph, uid))
    if any((uid in graph.model_names and graph.model_names[uid][1] == 'Mesh')
           or (uid in graph.uid2node and graph.uid2node[uid].name == b'Geometry')
           for uid in branches):
        raise _PropError('ALREADY_HAS_MESH', 'Existing mesh in this prop branch')
    if len(roots) != 1:
        raise _PropError('MISSING_ROOT' if not roots else 'DUPLICATE_ROOT',
                         'Requires one exact <prop>:Labeling:root model')
    bone_uid = roots[0]
    marker_names = [prefix + 'M_' + str(index) for index in range(1, 5)]
    if any(len(named.get(name, [])) != 1 for name in marker_names):
        raise _PropError('MISSING_MARKERS', 'Requires exactly one of each M_1..M_4')
    extra = [name for name in named if name.startswith(prefix + 'M_')
             and name not in marker_names]
    if extra:
        raise _PropError('MISSING_MARKERS', 'Unexpected marker names: ' + ', '.join(sorted(extra)))
    marker_uids = [named[name][0] for name in marker_names]
    if any(graph.model_names[uid][1] != 'Marker' for uid in marker_uids):
        raise _PropError('MISSING_MARKERS', 'M_1..M_4 must be Marker models')
    for uid in [bone_uid] + marker_uids:
        _check_transform(reader, buf, graph, uid)
        if len(parents.get(uid, [])) != 1:
            raise _PropError('UNSUPPORTED_TRANSFORM', f'{uid}: missing or multiple model parents')
    bone_parent = parents[bone_uid][0]
    if any(parents[uid][0] != bone_parent for uid in marker_uids):
        raise _PropError('UNSUPPORTED_TRANSFORM', 'Markers and bone must share an identity parent')
    seen = {bone_uid}
    ancestor = bone_parent
    while ancestor:
        if ancestor in seen or ancestor not in graph.model_names:
            raise _PropError('UNSUPPORTED_TRANSFORM', 'Invalid model ancestry')
        seen.add(ancestor)
        _check_transform(reader, buf, graph, ancestor, identity=True)
        if len(parents.get(ancestor, [])) != 1:
            raise _PropError('UNSUPPORTED_TRANSFORM', 'Missing or multiple ancestor parents')
        ancestor = parents[ancestor][0]
    times, translation = _curves(reader, buf, graph, bone_uid, 'Lcl Translation')
    rotation_times, euler = _curves(reader, buf, graph, bone_uid, 'Lcl Rotation')
    if not np.array_equal(times, rotation_times):
        raise _PropError('INVALID_CURVES', 'Bone translation/rotation key times differ')
    rotations = _xyz_rotations(euler)
    world = []
    for uid in marker_uids:
        marker_times, xyz = _curves(reader, buf, graph, uid, 'Lcl Translation')
        if not np.array_equal(times, marker_times):
            raise _PropError('INVALID_CURVES', 'Marker/bone key times differ')
        world.append(xyz)
    world = np.stack(world, axis=1)
    local = np.einsum('nji,nkj->nki', rotations,
                      (world - translation[:, None, :]) * native_unit_mm)
    reference = _find_stable_reference_window(local, window_size=5)
    calibration_count = min(120, len(times))
    if np.linalg.matrix_rank(reference - reference.mean(axis=0), tol=.1) < 2:
        raise _PropError('DEGENERATE_LAYOUT', 'Observed markers are collinear/coincident')
    ranked, invalid = [], []
    for asset_id in candidate_ids:
        asset = catalog.get(asset_id)
        if asset is None:
            continue
        try:
            holes = asset['holes']
            hole_ids = [hole['id'] for hole in holes]
            points = np.asarray([_point(hole['point_mm']) for hole in holes])
            if (len(holes) != 4 or len(set(hole_ids)) != 4
                    or np.linalg.matrix_rank(points - points.mean(axis=0), tol=.1) < 2):
                raise ValueError('Requires four unique non-collinear holes')
            if any(marker not in ('M_1', 'M_2', 'M_3', 'M_4') or hole not in hole_ids
                   for marker, hole in marker_map.items()):
                raise ValueError('marker_map references an unknown marker or hole')
            if len(set(marker_map.values())) != len(marker_map):
                raise ValueError('marker_map assigns one hole to multiple markers')
            fits = []
            for permutation in itertools.permutations(range(4)):
                if any(hole_ids[permutation[int(marker[2:]) - 1]] != hole
                       for marker, hole in marker_map.items()):
                    continue
                rms, rotation, offset = _fit(points[list(permutation)], reference)
                fits.append((rms, permutation, rotation, offset))
            fits.sort(key=lambda item: (item[0], item[1]))
            if not fits:
                raise ValueError('No permutation satisfies marker_map')
            ranked.append((fits[0][0], asset_id, fits, points, hole_ids))
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append({'asset_id': asset_id, 'error': str(exc)})
    if not ranked:
        raise _PropError('INVALID_CALIBRATION', invalid)
    ranked.sort(key=lambda item: (item[0], item[1]))
    rms, asset_id, fits, points, hole_ids = ranked[0]
    if rms > max_rms_mm:
        raise _PropError('FIT_REJECTED', {'best_asset_id': asset_id,
                                        'calibration_fit_rms_mm': rms,
                                        'max_rms_mm': max_rms_mm})
    near_assets = [item for item in ranked if item[0] <= rms + _AMBIGUITY_MM]
    if len(near_assets) > 1:
        raise _PropError('AMBIGUOUS_ASSET', {'candidates': [
            {'asset_id': item[1], 'rms_mm': item[0]} for item in near_assets]})
    near_orientations = [item for item in fits if item[0] <= rms + _AMBIGUITY_MM]
    if len(near_orientations) > 1:
        raise _PropError('AMBIGUOUS_ORIENTATION', {
            'asset_id': asset_id, 'equivalent_permutations': len(near_orientations),
            'details': 'Configure marker_map from measured marker/hole identities; no upright prior applied'})
    _, permutation, rotation, offset = fits[0]
    predicted = points[list(permutation)] @ rotation.T + offset
    errors = np.linalg.norm(local - predicted[None, :, :], axis=2)
    full = _stats(errors)
    if full['rms_mm'] > max_rms_mm or full['max_mm'] > max_rms_mm * max_outlier_multiplier:
        raise _PropError('FULL_SEQUENCE_RESIDUAL', {'asset_id': asset_id,
                                                  'full_sequence': full,
                                                  'max_rms_mm': max_rms_mm})
    return {
        'prop': prop, 'bone_uid': bone_uid, 'asset_id': asset_id,
        'asset_to_bone_rotation': rotation.tolist(),
        'asset_to_bone_translation_mm': (offset + np.asarray(bone_offset, dtype=float)).tolist(),
        'bone_offset_mm': list(bone_offset),
        'marker_mapping': [{'marker': marker_names[index], 'hole': hole_ids[hole]}
                           for index, hole in enumerate(permutation)],
        'calibration_fit_rms_mm': rms, 'calibration_frames': calibration_count,
        'full_sequence': full,
        'held_out_frames': (_stats(errors[calibration_count:])
                            if len(times) > calibration_count else None),
        'held_out_frame_count': max(0, len(times) - calibration_count),
        'frame_count': len(times),
        'runtime_samples': [
            {'source_frame_index': int(index),
             'time_seconds': float(times[index] / _TICKS_PER_SECOND),
             'bone_translation_native': translation[index].tolist(),
             'bone_rotation_matrix': rotations[index].tolist(),
             'marker_world_native': world[index].tolist()}
            for index in np.unique(np.linspace(0, len(times) - 1, min(13, len(times)), dtype=int))],
    }


def analyze_fbx(source: Path, catalog: dict, mapping: dict, *, max_rms_mm: float = 2.0,
                max_outlier_multiplier: float = 3.0) -> dict:
    """Return a JSON-serializable attachment plan without modifying any input.

    File-format/global-unit errors raise ValueError. Individual prop failures
    are deterministic skipped_props records; an empty scene is a valid plan.
    """
    if not math.isfinite(max_rms_mm) or max_rms_mm <= 0:
        raise ValueError('max_rms_mm must be positive and finite')
    source = Path(source).resolve()
    # Check the header before the repository reader interprets 64-bit node offsets.
    with source.open('rb') as stream:
        header = stream.read(27)
    if header[:23] != b'Kaydara FBX Binary  \x00\x1a\x00' or len(header) != 27:
        raise ValueError('Expected binary FBX 7700')
    version = struct.unpack_from('<I', header, 23)[0]
    if version != 7700:
        raise ValueError(f'Unsupported FBX version {version}; expected 7700')
    reader = _reader()
    buf = source.read_bytes()
    tops = parse_checked(reader, buf, allow_unterminated_children=True)
    graph = reader.build_graph(buf, tops)
    settings = graph.tops.get('GlobalSettings')
    properties = _properties(reader, buf, settings) if settings else {}
    unit = properties.get('UnitScaleFactor')
    if (not isinstance(unit, list) or len(unit) != 1
            or not isinstance(unit[0], (int, float)) or not math.isfinite(unit[0]) or unit[0] <= 0):
        raise ValueError('Missing/invalid FBX UnitScaleFactor; refusing to guess units')
    native_unit_mm = float(unit[0] * 10.)
    plan = {'source_fbx': str(source), 'source_sha256': hashlib.sha256(buf).hexdigest(),
            'fbx_version': version, 'native_unit_mm': native_unit_mm,
            'matched_props': [], 'skipped_props': [], 'detected_prop_count': 0,
            'warnings': []}
    roots, named, parents = {}, {}, {}
    for uid, (name, kind) in graph.model_names.items():
        named.setdefault(name, []).append(uid)
        if name.endswith(_ROOT_SUFFIX):
            roots.setdefault(name[:-len(_ROOT_SUFFIX)], []).append(uid)
        elif (kind == 'Marker' and ':' in name and name.rsplit(':', 1)[0]
              and name.rsplit(':', 1)[-1] in ('M_1', 'M_2', 'M_3', 'M_4')):
            roots.setdefault(name.rsplit(':', 1)[0], [])
    for connection in graph.conns:
        if connection[0] == 'OO' and connection[1] in graph.model_names:
            parents.setdefault(connection[1], []).append(connection[2])
    plan['detected_prop_count'] = len(roots)
    for prop in sorted(roots, key=lambda value: (value.casefold(), value)):
        candidate_ids, marker_map, bone_offset = _candidates(prop, catalog, mapping)
        missing = [identifier for identifier in candidate_ids if identifier not in catalog]
        if missing:
            plan['warnings'].append({'code': 'MISSING_CATALOG_CANDIDATES', 'prop': prop,
                                     'asset_ids': missing})
        try:
            if _normalized(prop) not in mapping and len(candidate_ids) > 1:
                raise _PropError('AMBIGUOUS_ASSET', {
                    'candidates': candidate_ids,
                    'details': 'Exact asset stem/ID is not unique; configure explicit candidates'})
            # Already-meshed and missing-root branches are still detected when no asset exists.
            if not any(identifier in catalog for identifier in candidate_ids):
                branches = set()
                for uid, (name, _) in graph.model_names.items():
                    if name.startswith(prop + ':'):
                        branches.update(reader.branch_uids(graph, uid))
                if any(graph.uid2node[uid].name == b'Geometry'
                       or graph.model_names.get(uid, ('', ''))[1] == 'Mesh'
                       for uid in branches if uid in graph.uid2node):
                    raise _PropError('ALREADY_HAS_MESH', 'Existing mesh in this prop branch')
                raise _PropError('NO_MODEL', {'requested_candidates': candidate_ids,
                                            'details': 'No exact name mapping or usable original asset'})
            record = _analyze_prop(reader, buf, graph, prop, roots[prop], named, parents,
                                   catalog, candidate_ids, marker_map, bone_offset, native_unit_mm, max_rms_mm,
                                   max_outlier_multiplier=max_outlier_multiplier)
            plan['matched_props'].append(record)
            if record['held_out_frame_count'] == 0:
                plan['warnings'].append({'code': 'NO_HELD_OUT_FRAMES', 'prop': prop,
                                         'details': 'All available frames used for calibration; full-sequence residual still checked'})
        except _PropError as exc:
            plan['skipped_props'].append({'prop': prop, 'reason': exc.reason, 'details': exc.details})
    if plan['matched_props']:
        plan['warnings'].append({
            'code': 'HOLE_REFERENCE_NOT_SPHERE_CENTER',
            'details': 'JSON hole mouth/center is the nominal asset-world reference, not a measured reflective sphere center. No seating/radius offset is invented; low rigid RMS cannot exclude a common offset.'})
    return plan

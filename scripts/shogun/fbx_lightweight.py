#!/usr/bin/env python3
"""Losslessly trim unlabeled markers and optical cameras into a separate FBX."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct


try:
    from . import mesh_fbx as fbx
except ImportError:
    import mesh_fbx as fbx


_ANIMATION = {b"AnimationCurve", b"AnimationCurveNode"}




def _index(parser, buf, tops):
    top = {node.name.decode("ascii"): node for node in tops}
    for name in ("Objects", "Connections", "Definitions", "GlobalSettings"):
        if name not in top:
            raise ValueError(f"Missing FBX section: {name}")
    objects, models = {}, {}
    for node in top["Objects"].children:
        values = [value for _, value in parser.decode_props(buf, node)]
        if not values or not isinstance(values[0], int) or values[0] == 0 or values[0] in objects:
            raise ValueError("Invalid or duplicate object UID")
        uid = values[0]
        objects[uid] = node
        if node.name == b"Model":
            if len(values) != 3:
                raise ValueError("Unsupported Model header")
            models[uid] = {"name": values[1].split("\x00", 1)[0], "kind": values[2]}
    connections = []
    for node in top["Connections"].children:
        values = tuple(value for _, value in parser.decode_props(buf, node))
        if (len(values) not in (3, 4) or values[0] not in ("OO", "OP")
                or not isinstance(values[1], int) or not isinstance(values[2], int)):
            raise ValueError("Unsupported connection record")
        connections.append((node, values))
    return top, objects, models, connections


def _prune(objects, models, connections, *, drop_unused_labeling=False, drop_solver_trees=False):
    children = defaultdict(list)
    parents = defaultdict(list)
    for _, link in connections:
        if link[0] == "OO" and link[1] in models and link[2] in models:
            children[link[2]].append(link[1])
            parents[link[1]].append(link[2])
    removed_models = {uid for uid, model in models.items() if model["kind"] == "Camera"}
    branch = [uid for uid, model in models.items()
              if model["name"].rsplit(":", 1)[-1] == "Unlabeled_Markers"]
    visited = set()
    while branch:
        uid = branch.pop()
        if uid in visited:
            continue
        visited.add(uid)
        if models[uid]["kind"] in ("Mesh", "LimbNode"):
            raise ValueError("Unlabeled_Markers contains a mesh/bone; cannot delete its parent hierarchy")
        removed_models.add(uid)
        branch.extend(children[uid])
    # Vicon exports also carry unlabeled markers as loose roots named M_1…;
    # the container branch alone is not enough.
    removed_models.update(uid for uid, model in models.items()
                          if model["kind"] == "Marker" and re.fullmatch(r"M_\d+", model["name"]))
    if drop_solver_trees:
        # The solved Character plus its markers are the usable data;
        # auxiliary Labeling*/Solving skeleton trees and loose, unattributed
        # marker-bone trees (template presets like a bare ARIEL, duplicating
        # the actor as extra stick figures) are dropped. A tree is only dropped
        # when it carries no subject-prefixed markers and no Meshes.
        stack_roots = [uid for uid, model in models.items()
                       if re.fullmatch(r"Labeling(?:_\d+)?|Solving", model["name"])]
        for root_uid in stack_roots:
            if root_uid in removed_models:
                continue
            subtree, stack = set(), [root_uid]
            blocked = False
            while stack:
                uid = stack.pop()
                if uid in subtree:
                    continue
                if models[uid]["kind"] in ("Marker", "Mesh"):
                    blocked = True
                    break
                subtree.add(uid)
                stack.extend(children[uid])
            if not blocked:
                removed_models.update(subtree)
        prefixed = any(model["kind"] == "Marker" and ":" in model["name"] for model in models.values())
        if prefixed:
            for uid, model in models.items():
                if uid in removed_models or parents[uid] or ":" in model["name"] or model["kind"] == "Camera":
                    continue
                subtree, stack = set(), [uid]
                blocked = False
                while stack:
                    node = stack.pop()
                    if node in subtree:
                        continue
                    subtree.add(node)
                    if ":" in models[node]["name"] or models[node]["kind"] == "Mesh":
                        blocked = True
                        break
                    stack.extend(children[node])
                if not blocked and any(models[node]["kind"] == "Marker" for node in subtree):
                    removed_models.update(subtree)
    if drop_unused_labeling:
        labeling = {uid for uid, model in models.items()
                    if model["kind"] in ("Null", "LimbNode")
                    and "Labeling" in model["name"].split(":")[1:]}
        # Keep ancestors of every surviving non-Labeling model. In particular,
        # object meshes use Labeling rigid roots, not interchangeable Solving roots.
        needed = set(models) - removed_models - labeling
        pending = list(needed)
        while pending:
            for parent in parents[pending.pop()]:
                if parent in labeling and parent not in needed:
                    needed.add(parent)
                    pending.append(parent)
        removed_models.update(labeling - needed)
    # Never remove a parent from beneath a retained human/mesh/null model.
    for uid in removed_models:
        if any(child not in removed_models for child in children[uid]):
            raise ValueError(f"Deleted subtree contains retained model children: {uid}")
    removed = set(removed_models)
    dependencies = {uid for uid, node in objects.items()
                    if node.name in _ANIMATION or node.name == b"NodeAttribute"}
    # Only dependencies of deleted objects are candidates; unrelated orphans stay.
    # Layer membership is organizational, not an animation usage edge.
    usage = []
    for _, link in connections:
        source, target = link[1:3]
        if source not in dependencies or target not in objects:
            continue
        if (objects[source].name == b"AnimationCurveNode" and link[0] == "OO"
                and objects[target].name == b"AnimationLayer"):
            continue
        usage.append((source, target))
    affected = set(removed)
    changed = True
    while changed:
        changed = False
        for source, target in usage:
            if target in affected and source not in affected:
                affected.add(source)
                changed = True
    live = set(objects) - affected
    changed = True
    while changed:
        changed = False
        for source, target in usage:
            if target in live and source not in live:
                live.add(source)
                changed = True
    removed.update((affected & dependencies) - live)
    keep = set(objects) - removed
    kept_links = [(node, link) for node, link in connections
                  if link[1] in keep and (link[2] == 0 or link[2] in keep)]
    dangling = [link for _, link in connections
                if link[1] not in objects or (link[2] != 0 and link[2] not in objects)]
    return keep, kept_links, removed_models, dangling


def _definition_updates(parser, buf, top, objects, keep):
    counts = Counter(objects[uid].name.decode("ascii") for uid in keep)
    counts["GlobalSettings"] += 1
    replacements, additions, seen, old = {}, [], set(), {}

    def replace(node, count):
        props = parser.decode_props(buf, node)
        if len(props) != 1 or props[0][0] not in ("I", "L"):
            raise ValueError("Unsupported Definitions Count")
        replacements[node.nstart] = (fbx.integer(count, props[0][0]), 1)

    definitions = top["Definitions"]
    for node in definitions.children:
        if node.name != b"ObjectType":
            continue
        name = parser.decode_props(buf, node)[0][1]
        if name in seen:
            raise ValueError("Duplicate Definitions ObjectType")
        seen.add(name)
        count_node = fbx.child_named(node, "Count")
        old[name] = parser.decode_props(buf, count_node)[0][1]
        replace(count_node, counts[name])
    for name in sorted(counts.keys() - seen):
        additions.append(fbx.node("ObjectType", fbx.string(name),
                                  children=[fbx.node("Count", fbx.integer(counts[name]))]))
    replace(fbx.child_named(definitions, "Count"), sum(counts.values()))
    return replacements, additions, dict(counts), old


def _emit(buf, out, node, replacements, filtered, additions):
    """Emit original payloads with relocated offsets and ORIGINAL-tail handling."""
    if isinstance(node, fbx.AddedNode):
        fbx.emit_node(buf, out, node, {}, {})
        return
    start = len(out)
    replacement = replacements.get(node.nstart)
    if replacement is None:
        out.extend(bytes(8))
        out.extend(memoryview(buf)[node.nstart + 8:node.cstart])
    else:
        payload, count = replacement
        out.extend(struct.pack("<QQQB", 0, count, len(payload), len(node.name)))
        out.extend(node.name)
        out.extend(payload)
    children = filtered.get(node.nstart, node.children)
    extras = additions.get(node.nstart, [])
    for child in children:
        _emit(buf, out, child, replacements, filtered, additions)
    for child in extras:
        _emit(buf, out, child, replacements, filtered, additions)
    tail = fbx.tail_bytes(buf, node)
    if tail and tail != fbx.NULL_RECORD:
        raise ValueError("Unexpected non-null node tail")
    # A removed last child must never be copied as part of the source tail.
    if children or extras or tail:
        out.extend(fbx.NULL_RECORD)
    struct.pack_into("<Q", out, start, len(out))


def _document(buf, tops, footer, replacements, filtered, additions):
    out = bytearray(buf[:27])
    for node in tops:
        _emit(buf, out, node, replacements, filtered, additions)
    out.extend(footer.prefix)
    out.extend(bytes((-len(out)) % 16))
    out.extend(footer.suffix)
    return out


def _semantic_fingerprint(buf, node):
    return fbx.fingerprint(buf, node, normalize_missing_child_terminators=True)


def _verify(parser, source, source_tops, source_index, output, keep, links,
            expected_counts):
    tops = fbx.parse_checked(parser, output)
    top, objects, models, output_links = _index(parser, output, tops)
    source_top, source_objects, source_models, _ = source_index
    if set(objects) != keep:
        raise ValueError("Serialized object UID set differs from pruning plan")
    if [link for _, link in output_links] != [link for _, link in links]:
        raise ValueError("Serialized connection payloads differ from retained links")
    if any(link[1] not in keep or (link[2] != 0 and link[2] not in keep) for _, link in output_links):
        raise ValueError("Output has dangling connections")
    if any(model["kind"] == "Camera" or model["name"].rsplit(":", 1)[-1] == "Unlabeled_Markers"
           for model in models.values()):
        raise ValueError("Optical Camera or Unlabeled_Markers Model survived")
    for uid in keep:
        if _semantic_fingerprint(source, source_objects[uid]) != _semantic_fingerprint(output, objects[uid]):
            raise ValueError(f"Retained object payload changed: {uid}")
    if [node.name for node in source_tops] != [node.name for node in tops]:
        raise ValueError("Top-level section roster changed")
    for name, original in source_top.items():
        if name not in ("Objects", "Connections", "Definitions"):
            if _semantic_fingerprint(source, original) != _semantic_fingerprint(output, top[name]):
                raise ValueError(f"Top-level settings/metadata changed: {name}")
    counts, definition_nodes = {}, {}
    for node in top["Definitions"].children:
        if node.name == b"ObjectType":
            name = parser.decode_props(output, node)[0][1]
            counts[name] = parser.decode_props(output, fbx.child_named(node, "Count"))[0][1]
            definition_nodes[name] = node
    if any(counts.get(name) != count for name, count in expected_counts.items()):
        raise ValueError("Definitions counts disagree with actual objects")
    if any(count != expected_counts.get(name, 0) for name, count in counts.items()):
        raise ValueError("Stale Definitions count survived")
    if parser.decode_props(output, fbx.child_named(top["Definitions"], "Count"))[0][1] != sum(expected_counts.values()):
        raise ValueError("Definitions total disagrees with actual objects")
    for original in source_top["Definitions"].children:
        if original.name != b"ObjectType":
            continue
        name = parser.decode_props(source, original)[0][1]
        old_templates = [child for child in original.children if child.name != b"Count"]
        new_templates = [child for child in definition_nodes[name].children if child.name != b"Count"]
        if [_semantic_fingerprint(source, n) for n in old_templates] != [_semantic_fingerprint(output, n) for n in new_templates]:
            raise ValueError("Definitions property template changed")
    before_footer = fbx.footer_parts(source, source_tops, 7700)
    after_footer = fbx.footer_parts(output, tops, 7700)
    if before_footer.prefix != after_footer.prefix or before_footer.suffix != after_footer.suffix:
        raise ValueError("Footer nonpadding bytes changed")
    if (tops[-1].off + len(after_footer.prefix) + after_footer.padding_size) % 16:
        raise ValueError("Footer alignment is invalid")
    return {"retained_object_payloads_exact": True,
            "retained_animation_payloads_exact": True,
            "retained_object_uid_roster_exact": True,
            "retained_connections_exact": True,
            "dangling_connections": 0, "definitions_and_templates_valid": True,
            "clip_bounds_frame_rate_units_and_root_settings_exact": True,
            "footer_nonpadding_exact_and_alignment_valid": True,
            "retained_object_uids": sorted(keep),
            "retained_animation_object_uids": sorted(
                uid for uid in keep if objects[uid].name in _ANIMATION),
            "models": len(models),
            "marker_models": sum(m["kind"] == "Marker" for m in models.values()),
            "bones": sum(m["kind"] == "LimbNode" for m in models.values()),
            "meshes": sum(m["kind"] == "Mesh" for m in models.values())}


def _marker_counts(models, links):
    # Shogun exports vary: some files link markers as (marker, character)
    # reversed pairs, some parent them under Model:false container Nodes.
    # Attribute each Marker to its nearest Model ancestor over the UNDIRECTED
    # OO graph (breadth-first, smallest uid on layer ties). This is a
    # statistics pass validated as before==after, so deterministic attribution
    # is what matters.
    adjacency = defaultdict(set)
    for _, link in links:
        if link[0] == "OO":
            adjacency[link[1]].add(link[2])
            adjacency[link[2]].add(link[1])
    counts = Counter()
    for uid, model in models.items():
        if model["kind"] != "Marker":
            continue
        layer, seen = [uid], {uid}
        found = None
        while layer and found is None:
            layer = [node for uid_at in layer for node in adjacency.get(uid_at, ()) if node not in seen]
            seen.update(layer)
            candidates = sorted(node for node in layer if node in models)
            if candidates:
                found = candidates[0]
        if found is None:
            raise ValueError(f"Labeled Marker has no Model ancestor: {uid}")
        counts[models[found]["name"]] += 1
    return dict(sorted(counts.items()))


def lighten_file(source, destination, *, report_path,
                 source_quality_status="NEEDS_REVIEW", drop_unused_labeling=False, drop_solver_trees=False):
    """Validate and atomically publish new files; never overwrite any existing path."""
    source_path = Path(source).resolve(strict=True)
    protected = [source_path, Path(__file__).resolve(), fbx.PARSER_PATH.resolve()]
    destination = fbx.checked_output(destination, protected, overwrite=False)
    report_path = fbx.checked_output(report_path, protected + [destination], overwrite=False)
    paths = [destination, report_path]
    source_bytes = source_path.read_bytes()
    parser = fbx.load_parser()
    tops = fbx.parse_checked(parser, source_bytes, allow_unterminated_children=True)
    footer = fbx.footer_parts(source_bytes, tops, 7700)
    source_index = _index(parser, source_bytes, tops)
    top, objects, all_models, source_links = source_index
    keep, links, removed_models, dangling = _prune(
        objects, all_models, source_links, drop_unused_labeling=drop_unused_labeling,
        drop_solver_trees=drop_solver_trees)
    replacements, definitions, counts, old_counts = _definition_updates(parser, source_bytes, top, objects, keep)
    filtered = {top["Objects"].nstart: [node for uid, node in objects.items() if uid in keep],
                top["Connections"].nstart: [node for node, _ in links]}
    additions = {top["Definitions"].nstart: definitions}
    output = _document(source_bytes, tops, footer, replacements, filtered, additions)
    checks = _verify(parser, source_bytes, tops, source_index, output, keep, links, counts)
    models = {uid: model for uid, model in all_models.items() if uid in keep}
    marker_counts = _marker_counts(models, links)
    report = dict(passed=True, source_fbx=str(source_path), output_fbx=str(destination),
                  source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  output_sha256=hashlib.sha256(output).hexdigest(),
                  source_bytes=len(source_bytes), output_bytes=len(output),
                  size_reduction_percent=100. * (1. - len(output) / len(source_bytes)),
                  source_quality_status=source_quality_status,
                  output_quality_status=source_quality_status,
                  physical_motion_repaired=False, retained_animation_payloads_exact=True,
                  drop_unused_labeling=drop_unused_labeling,
                  retained_labeling_anchors=[
                      dict(uid=uid, name=model["name"], kind=model["kind"])
                      for uid, model in models.items()
                      if model["kind"] in ("Null", "LimbNode")
                      and "Labeling" in model["name"].split(":")[1:]],
                  fbx_version=7700,
                  original_model_count=len(all_models), retained_model_count=len(models),
                  original_model_counts=dict(Counter(m["kind"] for m in all_models.values())),
                  retained_model_counts=dict(Counter(m["kind"] for m in models.values())),
                  removed_models=[dict(uid=uid, name=all_models[uid]["name"],
                                       kind=all_models[uid]["kind"])
                                  for uid in sorted(removed_models)],
                  retained_marker_counts_by_subject=marker_counts,
                  original_object_counts=dict(Counter(node.name.decode("ascii") for node in objects.values())),
                  retained_object_counts=counts, original_definitions_counts=old_counts,
                  inherited_dangling_connections_removed=len(dangling),
                  inherited_dangling_connections=[list(link) for link in dangling],
                  original_clock_and_frame_rate_preserved=True,
                  source_footer_layout=footer.layout, checks=checks)
    payloads = [output, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")]
    staged = []
    try:
        for path, payload in zip(paths, payloads):
            staged.append(fbx.stage_output(path, payload))
        _verify(parser, source_bytes, tops, source_index, staged[0].read_bytes(),
                keep, links, counts)
        for path in paths:
            fbx.checked_output(path, [source_path] + [other for other in paths if other != path], overwrite=False)
        # Report is published last, making it the completion record.
        for index in (0, 1):
            fbx.publish_output(staged[index], paths[index], overwrite=False)
    finally:
        for temporary in staged:
            if temporary.exists():
                temporary.unlink()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-quality-status", default="NEEDS_REVIEW")
    parser.add_argument("--drop-unused-labeling", action="store_true",
                        help="Remove unused Labeling skeletons; preserve mesh/Marker ancestor anchors")
    args = parser.parse_args(argv)
    result = lighten_file(args.source, args.destination, report_path=args.report,
                          source_quality_status=args.source_quality_status,
                          drop_unused_labeling=args.drop_unused_labeling)
    print(json.dumps({key: result[key] for key in ("passed", "output_fbx", "source_bytes", "output_bytes",
                                                  "size_reduction_percent")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Graft baked static Blender binary FBX assets into a capture's FBX7700.

Only object UIDs and connection endpoints are remapped for added assets. Geometry
layers, material properties and embedded Video.Content bytes stay opaque/exact.
Explicit original Mesh corrections affect only local translation/XYZ rotation,
Geometry.Vertices or a sole LayerElementNormal.Normals array. Array counts and
topology remain fixed; normals are never recomputed or normalized. Producers
remain responsible for normal correctness.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

import mesh_fbx as fbx


_ALLOWED = {b"Geometry", b"Model", b"Material", b"Texture", b"Video"}
_SCALAR_SIZE = {ord("Y"): 2, ord("C"): 1, ord("I"): 4, ord("F"): 4,
                ord("D"): 8, ord("L"): 8}


def _raw_properties(buf, original):
    """Split encoded properties, without decoding/copying vertex or UV arrays."""
    view = memoryview(buf)
    cursor, end = original.cstart - original.proplen, original.cstart
    result = []
    for _ in range(original.nprops):
        start = cursor
        if cursor >= end:
            raise ValueError("Truncated FBX property list")
        kind = buf[cursor]
        cursor += 1
        if kind in _SCALAR_SIZE:
            cursor += _SCALAR_SIZE[kind]
        elif kind in (ord("S"), ord("R")):
            if cursor + 4 > end:
                raise ValueError("Truncated FBX string/raw property")
            cursor += 4 + struct.unpack_from("<I", buf, cursor)[0]
        elif kind in b"dfilbc":
            if cursor + 12 > end:
                raise ValueError("Truncated FBX array property")
            cursor += 12 + struct.unpack_from("<I", buf, cursor + 8)[0]
        else:
            raise ValueError(f"Unsupported FBX property type: {chr(kind)!r}")
        if cursor > end:
            raise ValueError("FBX property exceeds its node bounds")
        result.append(view[start:cursor])
    if cursor != end:
        raise ValueError("FBX property count does not match payload length")
    return tuple(result)


def _clone(buf, original, replacements=None):
    properties = list(_raw_properties(buf, original))
    for index, value in (replacements or {}).items():
        properties[index] = value
    return fbx.AddedNode(original.name, tuple(properties),
                         [_clone(buf, child) for child in original.children],
                         bool(fbx.tail_bytes(buf, original)))


def _property_nodes(parser, buf, model):
    scopes = [child for child in model.children if child.name == b"Properties70"]
    if len(scopes) > 1:
        raise ValueError("Duplicate Properties70 scope")
    entries = {}
    if scopes:
        for child in scopes[0].children:
            if child.name != b"P":
                continue
            values = parser.decode_props(buf, child)
            if len(values) < 4 or values[0][0] != "S":
                raise ValueError("Malformed Properties70 entry")
            name = values[0][1]
            if name in entries:
                raise ValueError(f"Duplicate model property: {name}")
            entries[name] = (child, [value for _, value in values[4:]])
    return scopes[0] if scopes else None, entries


def _identity_model(parser, buf, model):
    _, properties = _property_nodes(parser, buf, model)
    defaults = {"Lcl Translation": (0, 0, 0), "Lcl Rotation": (0, 0, 0),
                "Lcl Scaling": (1, 1, 1), "GeometricTranslation": (0, 0, 0),
                "GeometricRotation": (0, 0, 0), "GeometricScaling": (1, 1, 1),
                "RotationOffset": (0, 0, 0), "RotationPivot": (0, 0, 0),
                "ScalingOffset": (0, 0, 0), "ScalingPivot": (0, 0, 0),
                "PreRotation": (0, 0, 0), "PostRotation": (0, 0, 0)}
    for name, default in defaults.items():
        if name in properties and properties[name][1] != list(default):
            raise ValueError(f"Static mesh must be baked to identity: {name}")


def _snippet_tops(parser, buf):
    """Read Blender's 7400 headers, or 7500+ headers, into shared Node objects."""
    if len(buf) < 27 or buf[:23] != parser.MAGIC:
        raise ValueError("Static snippet is not binary FBX")
    version = struct.unpack_from("<I", buf, 23)[0]
    if version not in (7400, 7500, 7600, 7700):
        raise ValueError(f"Unsupported static snippet FBX version: {version}")
    header_size = 13 if version == 7400 else 25
    header_format = "<IIIB" if version == 7400 else "<QQQB"
    null = bytes(header_size)

    def scan(start, end, *, top=False):
        nodes, cursor = [], start
        while cursor < end:
            if cursor + header_size > end:
                raise ValueError("Truncated static snippet node header")
            if buf[cursor:cursor + header_size] == null:
                if not top and cursor + header_size != end:
                    raise ValueError("Invalid static snippet child terminator")
                return nodes
            off, nprops, proplen, namelen = struct.unpack_from(header_format, buf, cursor)
            cstart = cursor + header_size + namelen + proplen
            if not cursor < cstart <= off <= end:
                raise ValueError("Invalid static snippet node bounds")
            current = parser.Node(bytes(buf[cursor + header_size:cursor + header_size + namelen]),
                                  cursor, off, nprops, proplen, cstart)
            if cstart < off:
                current.children = scan(cstart, off)
            nodes.append(current)
            cursor = off
        if nodes and not top:
            raise ValueError("Missing static snippet child terminator")
        return nodes

    tops = scan(27, len(buf), top=True)
    if not tops or len({item.name for item in tops}) != len(tops):
        raise ValueError("Missing or duplicate static snippet sections")
    tail = buf[tops[-1].off:]
    suffix = bytes(4) + struct.pack("<I", version) + bytes(120) + fbx.FOOTER_END_MAGIC
    if tail and not (tail == null or
                     (header_size <= len(tail) <= header_size + 15 and not any(tail))):
        padding = len(tail) - header_size - 16 - len(suffix)
        if (not 0 <= padding <= 16 or tail[:header_size] != null
                or tail[header_size + 16:header_size + 16 + padding] != bytes(padding)
                or tail[header_size + 16 + padding:] != suffix):
            raise ValueError("Truncated or unsupported static snippet footer")
    return tops


def _load_graph(parser, buf, *, source=False):
    tops = (fbx.parse_checked(parser, buf, allow_unterminated_children=True) if source
            else _snippet_tops(parser, buf))
    if source:
        fbx.footer_parts(buf, tops, 7700)
    required = {b"Objects", b"Connections"}
    if source:
        required |= {b"Definitions", b"GlobalSettings"}
    if not required <= {top.name for top in tops}:
        raise ValueError("Missing required FBX sections")
    graph = parser.build_graph(buf, tops)
    if len(graph.uid2node) != len(graph.obj_children) or 0 in graph.uid2node:
        raise ValueError("Duplicate or zero object UIDs")
    for obj in graph.obj_children:
        props = _raw_properties(buf, obj)
        if not props or props[0][0] != ord("L"):
            raise ValueError("FBX object UID must be signed int64")
    return tops, graph


def _embedded_video(buf, video):
    content = fbx.child_named(video, "Content")
    props = _raw_properties(buf, content)
    if len(props) != 1 or props[0][0] != ord("R") or len(props[0]) <= 5:
        raise ValueError("Video must contain nonempty embedded raw image bytes, not a path")
    image = props[0][5:]
    return dict(byte_count=len(image), sha256=hashlib.sha256(image).hexdigest())


def _snippet(parser, path, parent, name, prop, allocate):
    buf = path.read_bytes()
    _, graph = _load_graph(parser, buf)
    for obj in graph.obj_children:
        if obj.name not in _ALLOWED:
            raise ValueError(f"Static snippet contains unsupported object type: {obj.name!r}")
        values = parser.decode_props(buf, obj)
        if len(values) != 3:
            raise ValueError("Static snippet object needs UID, name and subtype")
        if obj.name in (b"Model", b"Geometry") and values[2][1] != "Mesh":
            raise ValueError("Static snippet may contain only Mesh models/geometries")
    models = [obj for obj in graph.obj_children if obj.name == b"Model"]
    geometries = [obj for obj in graph.obj_children if obj.name == b"Geometry"]
    if len(models) != 1 or not geometries:
        raise ValueError("Each static snippet must contain one Mesh model and real geometry")
    model = models[0]
    _identity_model(parser, buf, model)
    links = fbx.connection_values(parser, buf, graph)
    outgoing = Counter()
    root_count = 0
    for link in links:
        if len(link) not in (3, 4) or link[0] not in ("OO", "OP"):
            raise ValueError("Unsupported static snippet connection")
        child, target = graph.uid2node.get(link[1]), graph.uid2node.get(link[2])
        if child is None:
            raise ValueError("Dangling static snippet connection")
        kinds = (child.name, target.name if target is not None else None)
        if link == ("OO", model.uid, 0):
            root_count += 1
        elif link[0] == "OO" and len(link) == 3 and kinds in {
                (b"Geometry", b"Model"), (b"Material", b"Model"), (b"Video", b"Texture")}:
            pass
        elif (link[0] == "OP" and len(link) == 4 and kinds == (b"Texture", b"Material")
              and isinstance(link[3], str) and link[3]):
            pass
        else:
            raise ValueError(f"Unsupported static snippet link: {link!r}")
        outgoing[link[1]] += 1
    if root_count != 1 or any(not outgoing[obj.uid] for obj in graph.obj_children):
        raise ValueError("Snippet must have one root link and no disconnected asset objects")
    textures = [obj for obj in graph.obj_children if obj.name == b"Texture"]
    videos = [obj for obj in graph.obj_children if obj.name == b"Video"]
    images = {video.uid: _embedded_video(buf, video) for video in videos}
    for texture in textures:
        media = [link[1] for link in links if link[0] == "OO" and link[2] == texture.uid
                 and graph.uid2node[link[1]].name == b"Video"]
        if len(media) != 1:
            raise ValueError("Each texture needs one genuinely embedded Video")
    for geometry in geometries:
        _identity_model(parser, buf, geometry)
        vertices = fbx.array_data(buf, fbx.child_named(geometry, "Vertices"), "d")
        polygons = fbx.array_data(buf, fbx.child_named(geometry, "PolygonVertexIndex"), "i")
        if vertices.size < 9 or vertices.size % 3 or not polygons.size:
            raise ValueError("Static snippet contains empty/malformed geometry")
        if textures:
            layers = [child for child in geometry.children if child.name == b"LayerElementUV"]
            if not layers or any(not fbx.array_data(buf, fbx.child_named(layer, "UV"), "d").size
                                 for layer in layers):
                raise ValueError("Textured geometry requires nonempty UV layers")
    remap = {obj.uid: allocate() for obj in graph.obj_children}
    objects = [_clone(buf, obj, {0: fbx.integer(remap[obj.uid], "L"),
                                **({1: fbx.string(name + "\x00\x01Model")} if obj is model else {})})
               for obj in graph.obj_children]
    connections, expected_links = [], []
    for original, link in zip(graph.tops["Connections"].children, links):
        target = parent if link[2] == 0 else remap[link[2]]
        connections.append(_clone(buf, original, {1: fbx.integer(remap[link[1]], "L"),
                                                  2: fbx.integer(target, "L")}))
        expected_links.append((link[0], remap[link[1]], target, *link[3:]))
    report = dict(name=name, prop=prop, source_fbx=str(path), source_sha256=hashlib.sha256(buf).hexdigest(),
                  parent_uid=parent, model_uid=remap[model.uid],
                  geometry_uids=[remap[obj.uid] for obj in geometries],
                  material_uids=[remap[obj.uid] for obj in graph.obj_children if obj.name == b"Material"],
                  texture_uids=[remap[obj.uid] for obj in textures],
                  embedded_images=[dict(video_uid=remap[uid], **image) for uid, image in images.items()],
                  uid_remap={str(uid): new for uid, new in remap.items()})
    return objects, connections, expected_links, report


def _override_plan(parser, buf, graph, overrides):
    replacements, extras, reports = {}, {}, []
    if overrides is None:
        return replacements, extras, reports
    if not isinstance(overrides, dict):
        raise ValueError("model_transform_overrides must be a UID-keyed dictionary")
    for uid, values in overrides.items():
        if isinstance(uid, bool) or not isinstance(uid, int):
            raise ValueError("Transform override UID must be an integer")
        model = graph.uid2node.get(uid)
        if model is None or graph.model_names.get(uid, (None, None))[1] != "Mesh":
            raise ValueError("Transform overrides are allowed only on existing Mesh models")
        if any(link[2] == uid and link[1] in graph.uid2node
               and graph.uid2node[link[1]].name in (b"AnimationCurve", b"AnimationCurveNode")
               for link in graph.conns):
            raise ValueError("Cannot override a Mesh model with its own animated curves")
        if not isinstance(values, dict) or set(values) != {"translation", "rotation_degrees"}:
            raise ValueError("Override requires only translation and rotation_degrees")
        desired = {"Lcl Translation": fbx.vector3(values["translation"], "override.translation"),
                   "Lcl Rotation": fbx.vector3(values["rotation_degrees"], "override.rotation_degrees")}
        scope, entries = _property_nodes(parser, buf, model)
        if "RotationOrder" in entries and entries["RotationOrder"][1] != [0]:
            raise ValueError("Mesh pose override requires XYZ RotationOrder")
        added = []
        before = {}
        for name, vector in desired.items():
            before[name] = entries[name][1] if name in entries else [0., 0., 0.]
            if name in entries:
                original = entries[name][0]
                props = _raw_properties(buf, original)
                if len(props) != 7:
                    raise ValueError(f"Expected three scalar components for {name}")
                replacements[original.nstart] = (b"".join((*props[:4], *(fbx.real(v) for v in vector))), 7)
            else:
                added.append(fbx.property70(name, name, [fbx.real(v) for v in vector]))
        if added:
            if scope is None:
                extras[model.nstart] = [fbx.node("Properties70", children=added)]
            else:
                extras[scope.nstart] = added
        reports.append(dict(uid=uid, name=graph.model_names[uid][0], before=before,
                            after={key: list(vector) for key, vector in desired.items()},
                            changed_properties=list(desired), rotation_order="XYZ (degrees)",
                            translation_units="source native"))
    return replacements, extras, reports


def _geometry_override_plan(parser, buf, graph, overrides, *, normals=False):
    replacements, reports = {}, []
    if overrides is None:
        return replacements, reports
    noun = "normal" if normals else "vertex"
    array_name = "Normals" if normals else "Vertices"
    if not isinstance(overrides, dict):
        raise ValueError(f"geometry_{noun}_overrides must be a UID-keyed dictionary")
    for uid, values in overrides.items():
        if isinstance(uid, bool) or not isinstance(uid, int):
            raise ValueError("Geometry override UID must be an integer")
        geometry = graph.uid2node.get(uid)
        if geometry is None or geometry.name != b"Geometry":
            raise ValueError("Array overrides require an original Mesh Geometry UID")
        properties = parser.decode_props(buf, geometry)
        if len(properties) != 3 or properties[2] != ("S", "Mesh"):
            raise ValueError("Array overrides require an original Mesh Geometry UID")
        scope = geometry
        if normals:
            layers = [child for child in geometry.children if child.name == b"LayerElementNormal"]
            if len(layers) != 1:
                raise ValueError("Normal overrides require exactly one original LayerElementNormal")
            scope = layers[0]
        arrays = [child for child in scope.children if child.name == array_name.encode("ascii")]
        if len(arrays) != 1:
            raise ValueError(f"Original Mesh Geometry must have exactly one {array_name} array")
        original = arrays[0]
        raw = _raw_properties(buf, original)
        if len(raw) != 1 or raw[0][0] not in b"df":
            raise ValueError(f"Original {array_name} must be a single floating-point array")
        kind = chr(raw[0][0])
        before = fbx.array_data(buf, original, kind)
        if not before.size or before.size % 3 or not np.isfinite(before).all():
            raise ValueError(f"Original {array_name} must contain finite XYZ vectors")
        before = before.reshape(-1, 3)
        vectors = fbx.numeric_array(values, f"geometry_{noun}_overrides values", columns=3)
        if vectors.shape != before.shape:
            raise ValueError(f"Geometry override must retain the exact original {noun} count")
        with np.errstate(over="ignore", invalid="ignore"):
            vectors = np.asarray(vectors, dtype=before.dtype)
        if not np.isfinite(vectors).all():
            raise ValueError(f"Geometry override {noun} values exceed the original array precision")
        replacements[original.nstart] = (fbx.array(vectors, kind), 1)
        record = dict(uid=uid, array_type=kind)
        record.update({
            f"old_{noun}_count": len(before), f"new_{noun}_count": len(vectors),
            f"changed_{noun}_count": int(np.count_nonzero(np.any(before != vectors, axis=1))),
        })
        if normals:
            record["normal_policy"] = "Stored vectors; no recomputation or normalization"
        else:
            record["position_units"] = "source native"
        reports.append(record)
    return replacements, reports


def _expected_fingerprint(parser, source, original, replacements, extras):
    encoded = bytearray(source[:27])
    fbx.emit_node(source, encoded, original, replacements, extras)
    parsed = fbx.parse_checked(parser, encoded)
    return fbx.fingerprint(encoded, parsed[0])


def _added_fingerprint(parser, added):
    encoded = bytearray(parser.MAGIC + struct.pack("<I", 7700))
    fbx.emit_node(b"", encoded, added, {}, {})
    return fbx.fingerprint(encoded, fbx.parse_checked(parser, encoded)[0])


def graft_static_assets(source: Path, destination: Path, assets: list[dict], *,
                        verification_path: Path, model_transform_overrides=None,
                        geometry_vertex_overrides=None, geometry_normal_overrides=None) -> dict:
    """Validate, graft, verify and atomically publish new FBX/report artifacts.

    Asset vertices are baked in native parent-local units, or native world units
    for parent_uid=0 (the FBX world/root, not an original bone). Other parents
    must be original bone/root/null Models. No axis/unit conversion or animation
    evaluation occurs here. Existing files, input aliases and inherited dangling
    IDs are never overwritten/reassigned. An empty asset list requires a nonempty
    explicit Mesh pose, Geometry Vertices or Geometry Normals override map.
    Geometry overrides retain array counts, topology and normal indices/mapping.
    Normals are never recomputed or normalized.
    """
    source_path = Path(source).resolve(strict=True)
    if not isinstance(assets, list) or (
            not assets and not model_transform_overrides and not geometry_vertex_overrides
            and not geometry_normal_overrides):
        raise ValueError("Supply static assets or explicit original Mesh pose/Geometry array overrides")
    paths = []
    for asset in assets:
        if not isinstance(asset, dict) or set(asset) != {"fbx", "parent_uid", "name", "prop"}:
            raise ValueError("Each asset requires only fbx, parent_uid, name, prop")
        paths.append(Path(asset["fbx"]).resolve(strict=True))
    protected = [source_path, *paths, Path(__file__).resolve(), Path(fbx.__file__).resolve(),
                 fbx.PARSER_PATH.resolve()]
    output_path = fbx.checked_output(destination, protected, overwrite=False)
    report_path = fbx.checked_output(verification_path, protected + [output_path], overwrite=False)
    source_bytes = source_path.read_bytes()
    parser = fbx.load_parser()
    source_tops, graph = _load_graph(parser, source_bytes, source=True)
    footer = fbx.footer_parts(source_bytes, source_tops, 7700)
    used = set(graph.uid2node) | {0}
    for connection in graph.conns:
        used.update(connection[1:3])
    documents = graph.tops.get("Documents")
    if documents is not None:
        for document in documents.children:
            if document.name == b"Document":
                used.add(parser.decode_props(source_bytes, document)[0][1])
    next_uid = max(used) + 1

    def allocate():
        nonlocal next_uid
        if next_uid > 9223372036854775807:
            raise ValueError("No available positive signed int64 object UID")
        result = next_uid
        next_uid += 1
        return result

    names = {value[0] for value in graph.model_names.values()}
    objects, connections, expected_links, asset_reports = [], [], [], []
    for asset, path in zip(assets, paths):
        name = fbx.text(asset["name"], "asset.name")
        prop = fbx.text(asset["prop"], "asset.prop")
        if name in names:
            raise ValueError(f"Mesh model name is not unique: {name}")
        names.add(name)
        parent = asset["parent_uid"]
        if isinstance(parent, bool) or not isinstance(parent, int):
            raise ValueError("asset.parent_uid must identify world/root 0 or an original bone Model")
        if parent != 0 and graph.model_names.get(parent, (None, None))[1] not in ("LimbNode", "Root", "Null"):
            raise ValueError("Static asset parent must be world/root 0 or an original bone/root/null Model")
        added, links, decoded, details = _snippet(parser, path, parent, name, prop, allocate)
        objects.extend(added)
        connections.extend(links)
        expected_links.extend(decoded)
        asset_reports.append(details)
    replacements, definitions, counts, original_counts = fbx.definition_updates(
        parser, source_bytes, graph, objects)
    override_replacements, override_extras, override_reports = _override_plan(
        parser, source_bytes, graph, model_transform_overrides)
    replacements.update(override_replacements)
    vertex_replacements, vertex_reports = _geometry_override_plan(
        parser, source_bytes, graph, geometry_vertex_overrides)
    replacements.update(vertex_replacements)
    normal_replacements, normal_reports = _geometry_override_plan(
        parser, source_bytes, graph, geometry_normal_overrides, normals=True)
    replacements.update(normal_replacements)
    extras = {graph.tops["Objects"].nstart: objects, graph.tops["Connections"].nstart: connections,
              graph.tops["Definitions"].nstart: definitions, **override_extras}
    output = bytearray(source_bytes[:27])
    for original in source_tops:
        fbx.emit_node(source_bytes, output, original, replacements, extras)
    output.extend(footer.prefix)
    output.extend(bytes((-len(output)) % 16))
    output.extend(footer.suffix)
    metadata = dict(source_fbx=str(source_path), output_fbx=str(output_path), verification_path=str(report_path),
                    source_sha256=hashlib.sha256(source_bytes).hexdigest(), output_sha256=hashlib.sha256(output).hexdigest(),
                    source_bytes=len(source_bytes), output_bytes=len(output), fbx_version=7700,
                    original_definitions_counts=original_counts,
                    coordinate_policy="Baked native parent-local/world static snippets; only explicit original Mesh pose and Geometry Vertices/Normals overrides")
    # Reuse the engine's source/animation/footer/Definitions integrity verifier;
    # replace its generated-untextured-mesh checks with opaque static asset checks.
    report = fbx.verify(parser, source_bytes, source_tops, graph, output, [], counts, metadata)
    output_tops = fbx.parse_checked(parser, output)
    output_graph = parser.build_graph(output, output_tops)
    overridden = {item["uid"] for item in override_reports}
    vertex_overridden = {item["uid"] for item in vertex_reports}
    normal_overridden = {item["uid"] for item in normal_reports}
    geometry_overridden = vertex_overridden | normal_overridden
    expected_replacements = {**override_replacements, **vertex_replacements, **normal_replacements}
    for item in report["original_objects"]:
        item["intentional_transform_override"] = item["uid"] in overridden
        item["intentional_geometry_vertex_override"] = item["uid"] in vertex_overridden
        item["intentional_geometry_normal_override"] = item["uid"] in normal_overridden
        item["matches_expected"] = item["unchanged"]
        if item["uid"] in overridden | geometry_overridden:
            expected = _expected_fingerprint(parser, source_bytes, graph.uid2node[item["uid"]],
                                             expected_replacements, override_extras)
            item["expected_sha256"] = expected.hex()
            item["matches_expected"] = expected.hex() == item["output_sha256"]
    original_records = {item["uid"]: item for item in report["original_objects"]}
    for records, normals in ((vertex_reports, False), (normal_reports, True)):
        for item in records:
            original = graph.uid2node[item["uid"]]
            copied = output_graph.uid2node[item["uid"]]
            array_name = "Normals" if normals else "Vertices"
            if normals:
                original = fbx.child_named(original, "LayerElementNormal")
                copied = fbx.child_named(copied, "LayerElementNormal")
            original_array = fbx.child_named(original, array_name)
            copied_array = fbx.child_named(copied, array_name)
            item["checks"] = {
                array_name.lower() + "_match_override": _expected_fingerprint(
                    parser, source_bytes, original_array, expected_replacements, {})
                == fbx.fingerprint(output, copied_array),
                "non_overridden_geometry_payloads_preserved": original_records[item["uid"]]["matches_expected"],
            }
            item["passed"] = all(item["checks"].values())
    allocated = {uid for asset in asset_reports for uid in asset["uid_remap"].values()}
    object_ok = {}
    for obj in objects:
        uid = struct.unpack_from("<q", obj.properties[0], 1)[0]
        copied = output_graph.uid2node.get(uid)
        object_ok[uid] = copied is not None and _added_fingerprint(parser, obj) == fbx.fingerprint(output, copied)
    decoded = fbx.connection_values(parser, output, output_graph)
    for asset in asset_reports:
        model = output_graph.uid2node[asset["model_uid"]]
        parents = [link[2] for link in decoded if link[0] == "OO" and link[1] == asset["model_uid"]]
        image_ok = all(_embedded_video(output, output_graph.uid2node[image["video_uid"]]) ==
                       {key: image[key] for key in ("byte_count", "sha256")} for image in asset["embedded_images"])
        asset["checks"] = dict(remapped_payloads_uv_materials_textures_exact=all(
            object_ok[uid] for uid in asset["uid_remap"].values()), embedded_image_bytes_exact=image_ok,
            exact_requested_parent=parents == [asset["parent_uid"]],
            unique_mesh_name=output_graph.model_names[model.uid][0] == asset["name"])
        _identity_model(parser, output, model)
        asset["checks"]["model_identity"] = True
        asset["passed"] = all(asset["checks"].values())
    checks = report["checks"]
    checks.pop("all_original_nested_payloads_and_existing_tails_preserved")
    checks["all_original_payloads_preserved_except_explicit_mesh_pose_and_geometry_arrays"] = all(
        item["matches_expected"] for item in report["original_objects"])
    checks["geometry_vertex_overrides_valid"] = all(item["passed"] for item in vertex_reports)
    checks["geometry_normal_overrides_valid"] = all(item["passed"] for item in normal_reports)
    checks["original_animation_curves_exact"] = all(item["unchanged"] for item in report["original_objects"]
                                                    if item["type"] == "AnimationCurve")
    checks["original_bones_markers_and_source_meshes_preserved"] = all(
        item["unchanged"] for item in report["original_objects"] if item["type"] == "Model"
        and item["uid"] not in overridden)
    checks["exact_expected_uid_set"] = set(output_graph.uid2node) == set(graph.uid2node) | allocated
    checks["added_connections_exact"] = decoded[len(graph.tops["Connections"].children):] == expected_links
    checks["all_meshes_valid"] = all(asset["passed"] for asset in asset_reports)
    checks["definitions_templates_preserved_except_counts"] = _expected_fingerprint(
        parser, source_bytes, graph.tops["Definitions"], replacements, extras) == fbx.fingerprint(
            output, output_graph.tops["Definitions"])
    checks["original_container_properties_preserved"] = all(
        source_bytes[graph.tops[name].nstart + 8:graph.tops[name].cstart] ==
        output[output_graph.tops[name].nstart + 8:output_graph.tops[name].cstart]
        for name in ("Objects", "Connections"))
    report.update(schema_version=1, mesh_count=len(assets), new_object_count=len(allocated),
                  meshes=asset_reports, model_transform_overrides=override_reports,
                  geometry_vertex_overrides=vertex_reports, geometry_normal_overrides=normal_reports,
                  verification_scope="Binary static graft integrity, embedded images and explicit Mesh pose/Geometry Vertices/Normals changes; not physical-fit or normal correctness",
                  passed=all(checks.values()))
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("Integrity verification failed; no outputs published: " + ", ".join(failed))
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    staged = []
    try:
        staged.append(fbx.stage_output(output_path, output))
        staged.append(fbx.stage_output(report_path, report_bytes))
        fbx.checked_output(output_path, protected, overwrite=False)
        fbx.checked_output(report_path, protected + [output_path], overwrite=False)
        fbx.publish_output(staged[1], report_path, overwrite=False)
        fbx.publish_output(staged[0], output_path, overwrite=False)
    finally:
        for temporary in staged:
            if temporary.exists():
                temporary.unlink()
    return report

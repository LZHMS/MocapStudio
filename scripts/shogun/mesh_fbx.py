#!/usr/bin/env python3
"""Incrementally graft real bone-local meshes into binary FBX7700 documents.

Original properties, animation, object/connection order and non-padding footer
bytes are conserved. Definitions counts and relocated node end offsets change;
missing legacy child terminators are added without changing property payloads.
Verification proves scene conservation and serialization, not physical fit.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import zlib

import numpy as np

PARSER_PATH = Path(__file__).resolve().parents[1] / "extract_groups.py"
NULL_RECORD = bytes(25)
FOOTER_MAGIC = bytes.fromhex("fabca90ed4c8d667b070fa8318fb2b7d")
FOOTER_END_MAGIC = bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b")
_ARRAY_DTYPES = {"d": "<f8", "f": "<f4", "i": "<i4", "l": "<i8"}


@dataclass
class AddedNode:
    name: bytes
    properties: tuple = ()
    children: list = field(default_factory=list)
    terminated: bool = False


@dataclass(frozen=True)
class FooterParts:
    prefix: bytes
    suffix: bytes
    padding_size: int
    layout: str
    terminator_present: bool = True

    @property
    def footer_present(self):
        return self.layout not in ("nodes_only", "terminator_only")


def string(value):
    raw = value.encode("utf-8")
    return b"S" + struct.pack("<I", len(raw)) + raw


def integer(value, kind="I"):
    return kind.encode("ascii") + struct.pack("<i" if kind == "I" else "<q", value)


def real(value):
    return b"D" + struct.pack("<d", value)


def array(values, kind):
    """Encode one contiguous little-endian array without Python scalar expansion."""
    values = np.ascontiguousarray(values, dtype=_ARRAY_DTYPES[kind]).reshape(-1)
    raw = memoryview(values).cast("B")
    compressed = zlib.compress(raw)
    encoding, body = (1, compressed) if len(compressed) < len(raw) else (0, raw)
    return b"".join((kind.encode("ascii") + struct.pack("<III", values.size, encoding, len(body)), body))


def node(name, *properties, children=None, terminated=False):
    return AddedNode(name.encode("ascii"), tuple(properties),
                     list(children) if children is not None else [], terminated)


def property70(name, kind, values, *, label="", flags="A"):
    return node("P", string(name), string(kind), string(label), string(flags), *values)


def load_parser():
    spec = importlib.util.spec_from_file_location("mesh_graft_fbx_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import FBX parser: {PARSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def checked_output(path, protected, *, overwrite):
    candidate = Path(path).resolve()
    for original in protected:
        if candidate == original or (candidate.exists() and original.exists()
                                     and os.path.samefile(candidate, original)):
            raise ValueError(f"Refusing to overwrite a protected input/output: {candidate}")
    if candidate.exists():
        if not candidate.is_file():
            raise ValueError(f"Output is not a regular file: {candidate}")
        if not overwrite:
            raise FileExistsError(f"Output exists; explicit overwrite=True required: {candidate}")
    return candidate


def finite_number(value, label):
    if isinstance(value, (bool, np.bool_, np.complexfloating)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def numeric_array(value, label, *, columns=None, integer_only=False):
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a rectangular numeric array") from exc
    allowed = "iu" if integer_only else "iuf"
    if result.dtype.kind not in allowed:
        raise ValueError(f"{label} must contain {'integer' if integer_only else 'real numeric'} values")
    if columns is not None:
        if result.ndim != 2 or result.shape[1] != columns:
            raise ValueError(f"{label} must have shape N x {columns}")
    elif result.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if not integer_only and not np.isfinite(result).all():
        raise ValueError(f"{label} must contain finite values")
    return result


def vector3(value, label):
    result = numeric_array(value, label)
    if result.size != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    return tuple(float(v) for v in result)


def text(value, label):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty string without NUL characters")
    return value


def validate_mesh(mesh, graph, names):
    if not isinstance(mesh, dict):
        raise ValueError("Each mesh must be an object")
    name = text(mesh.get("name"), "mesh.name")
    if name in names:
        raise ValueError(f"Duplicate mesh name: {name}")
    names.add(name)
    prop = text(mesh.get("prop"), f"{name}.prop")
    parent = mesh.get("parent_uid")
    if isinstance(parent, (bool, np.bool_)) or not isinstance(parent, (int, np.integer)):
        raise ValueError(f"{name}: parent_uid must identify an original Model")
    parent = int(parent)
    if parent not in graph.uid2node or graph.uid2node[parent].name != b"Model":
        raise ValueError(f"{name}: parent_uid must identify an original Model")
    vertices = numeric_array(mesh.get("vertices"), f"{name}.vertices", columns=3)
    if not 3 <= len(vertices) <= 2147483647:
        raise ValueError(f"{name}: require 3..2147483647 vertices")
    vertices = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(vertices).all():
        raise ValueError(f"{name}: vertices exceed FBX double precision range")
    triangles = numeric_array(mesh.get("triangles"), f"{name}.triangles", columns=3, integer_only=True)
    if not len(triangles) or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError(f"{name}: triangles must contain valid vertex indices")
    if (np.any(triangles[:, 0] == triangles[:, 1]) or np.any(triangles[:, 1] == triangles[:, 2])
            or np.any(triangles[:, 0] == triangles[:, 2])):
        raise ValueError(f"{name}: triangle has repeated vertex indices")
    triangles = np.asarray(triangles, dtype=np.int32)
    normals = mesh.get("normals")
    if normals is not None:
        normals = numeric_array(normals, f"{name}.normals", columns=3)
        if normals.shape != vertices.shape:
            raise ValueError(f"{name}: normals must be per-vertex")
        normals = np.asarray(normals, dtype=np.float64)
        if not np.all(np.abs(np.linalg.norm(normals, axis=1) - 1.0) <= .001):
            raise ValueError(f"{name}: supplied normals must already be normalized")
    vertex_colors = mesh.get("vertex_colors")
    if vertex_colors is not None:
        vertex_colors = numeric_array(vertex_colors, f"{name}.vertex_colors", columns=4)
        if len(vertex_colors) != len(vertices) or np.any(vertex_colors < 0) or np.any(vertex_colors > 1):
            raise ValueError(f"{name}: vertex_colors must contain per-vertex RGBA in [0,1]")
        vertex_colors = np.asarray(vertex_colors, dtype=np.float64)
    materials = mesh.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError(f"{name}: supply at least one material")
    parsed_materials = []
    for material in materials:
        if not isinstance(material, dict):
            raise ValueError(f"{name}: each material must be an object")
        material_name = text(material.get("name"), f"{name}.material.name")
        color = vector3(material.get("diffuse_rgb"), f"{name}.diffuse_rgb")
        opacity = finite_number(material.get("opacity"), f"{name}.opacity")
        if material.get("color_space") != "sRGB":
            raise ValueError(f"{name}: material color_space must be 'sRGB'")
        if not all(0 <= v <= 1 for v in color) or not 0 <= opacity <= 1:
            raise ValueError(f"{name}: diffuse_rgb and opacity must be in [0,1]")
        parsed_materials.append(dict(name=material_name, diffuse_rgb=color, opacity=opacity,
                                     color_space="sRGB"))
    indices = numeric_array(mesh.get("triangle_material_indices"), f"{name}.triangle_material_indices",
                            integer_only=True)
    if len(indices) != len(triangles) or np.any(indices < 0) or np.any(indices >= len(materials)):
        raise ValueError(f"{name}: material indices must cover every triangle with a valid material")
    return dict(name=name, prop=prop, parent_uid=parent, vertices=vertices, triangles=triangles,
                normals=normals, vertex_colors=vertex_colors, materials=parsed_materials,
                triangle_material_indices=np.asarray(indices, dtype=np.int32))


def mesh_normals(mesh):
    if mesh["normals"] is not None:
        return "ByVertice", mesh["normals"].reshape(-1)
    vertices, triangles = mesh["vertices"], mesh["triangles"]
    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                       vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError(f"{mesh['name']}: cannot compute normals for degenerate triangles")
    normals /= lengths[:, None]
    return "ByPolygonVertex", np.repeat(normals, 3, axis=0).reshape(-1)


def polygon_indices(mesh):
    result = mesh["triangles"].copy()
    result[:, 2] = -result[:, 2] - 1
    return result.reshape(-1)


def create_mesh_objects(mesh, allocate):
    geometry_uid, model_uid = allocate(), allocate()
    material_uids = [allocate() for _ in mesh["materials"]]
    mesh.update(geometry_uid=geometry_uid, model_uid=model_uid, material_uids=material_uids)
    mapping, normals = mesh_normals(mesh)
    geometry = node("Geometry", integer(geometry_uid, "L"), string(mesh["name"] + "\x00\x01Geometry"),
                    string("Mesh"), children=[
        node("GeometryVersion", integer(124)),
        node("Vertices", array(mesh["vertices"], "d")),
        node("PolygonVertexIndex", array(polygon_indices(mesh), "i")),
        node("LayerElementNormal", integer(0), children=[
            node("Version", integer(101)), node("Name", string("")),
            node("MappingInformationType", string(mapping)),
            node("ReferenceInformationType", string("Direct")), node("Normals", array(normals, "d")),
        ]),
        node("LayerElementMaterial", integer(0), children=[
            node("Version", integer(101)), node("Name", string("")),
            node("MappingInformationType", string("ByPolygon")),
            node("ReferenceInformationType", string("IndexToDirect")),
            node("Materials", array(mesh["triangle_material_indices"], "i")),
        ]),
        node("Layer", integer(0), children=[
            node("Version", integer(100)),
            node("LayerElement", children=[node("Type", string("LayerElementNormal")),
                                           node("TypedIndex", integer(0))]),
            node("LayerElement", children=[node("Type", string("LayerElementMaterial")),
                                           node("TypedIndex", integer(0))]),
        ]),
    ])
    if mesh["vertex_colors"] is not None:
        geometry.children.insert(-1, node("LayerElementColor", integer(0), children=[
            node("Version", integer(101)), node("Name", string("Color")),
            node("MappingInformationType", string("ByVertice")),
            node("ReferenceInformationType", string("Direct")),
            node("Colors", array(mesh["vertex_colors"], "d")),
        ]))
        geometry.children[-1].children.append(node("LayerElement", children=[
            node("Type", string("LayerElementColor")), node("TypedIndex", integer(0)),
        ]))
    transforms = []
    for name, kind, value in (
        ("Lcl Translation", "Lcl Translation", (0., 0., 0.)),
        ("Lcl Rotation", "Lcl Rotation", (0., 0., 0.)),
        ("Lcl Scaling", "Lcl Scaling", (1., 1., 1.)),
        ("GeometricTranslation", "Vector3D", (0., 0., 0.)),
        ("GeometricRotation", "Vector3D", (0., 0., 0.)),
        ("GeometricScaling", "Vector3D", (1., 1., 1.)),
    ):
        transforms.append(property70(name, kind, [real(v) for v in value]))
    model = node("Model", integer(model_uid, "L"), string(mesh["name"] + "\x00\x01Model"),
                 string("Mesh"), children=[
        node("Version", integer(232)), node("Properties70", children=transforms),
        node("Shading", b"C\x01"), node("Culling", string("CullingOff")),
    ])
    objects = [geometry, model]
    connections = [node("C", string("OO"), integer(geometry_uid, "L"), integer(model_uid, "L")),
                   node("C", string("OO"), integer(model_uid, "L"), integer(mesh["parent_uid"], "L"))]
    for uid, material in zip(material_uids, mesh["materials"]):
        opacity = material["opacity"]
        objects.append(node("Material", integer(uid, "L"),
                            string(material["name"] + "\x00\x01Material"), string(""), children=[
            node("Version", integer(102)), node("ShadingModel", string("lambert")),
            node("MultiLayer", integer(0)), node("Properties70", children=[
                property70("DiffuseColor", "Color", [real(v) for v in material["diffuse_rgb"]]),
                property70("DiffuseFactor", "Number", [real(1.)]),
                property70("Opacity", "Number", [real(opacity)]),
                property70("TransparentColor", "Color", [real(1.)] * 3),
                property70("TransparencyFactor", "Number", [real(1. - opacity)]),
            ]),
        ]))
        connections.append(node("C", string("OO"), integer(uid, "L"), integer(model_uid, "L")))
    return objects, connections


def tail_bytes(buf, original):
    end = original.children[-1].off if original.children else original.cstart
    return buf[end:original.off]


def emit_node(buf, out, original, replacements, extra_children):
    start = len(out)
    if isinstance(original, AddedNode):
        properties = b"".join(original.properties)
        out.extend(struct.pack("<QQQB", 0, len(original.properties), len(properties), len(original.name)))
        out.extend(original.name)
        out.extend(properties)
        for child in original.children:
            emit_node(buf, out, child, replacements, extra_children)
        if original.children or original.terminated:
            out.extend(NULL_RECORD)
    else:
        replacement = replacements.get(original.nstart)
        if replacement is None:
            out.extend(bytes(8))
            out.extend(buf[original.nstart + 8:original.cstart])
        else:
            properties, count = replacement
            out.extend(struct.pack("<QQQB", 0, count, len(properties), len(original.name)))
            out.extend(original.name)
            out.extend(properties)
        for child in original.children:
            emit_node(buf, out, child, replacements, extra_children)
        added = extra_children.get(original.nstart, [])
        for child in added:
            emit_node(buf, out, child, replacements, extra_children)
        tail = tail_bytes(buf, original)
        out.extend(tail if tail else (NULL_RECORD if original.children or added else b""))
    struct.pack_into("<Q", out, start, len(out))


def parse_checked(parser, buf, *, allow_unterminated_children=False):
    """Bound every offset; legacy scopes may end exactly at their last child."""
    if len(buf) < 27 or buf[:23] != parser.MAGIC or struct.unpack_from("<I", buf, 23)[0] != 7700:
        raise ValueError("Only binary FBX version 7700 is supported")

    def subtree(start, end, *, top=False):
        nodes, cursor = [], start
        while cursor < end:
            if cursor + 25 > end:
                raise ValueError(f"Truncated FBX node header at {cursor}")
            if buf[cursor:cursor + 25] == NULL_RECORD:
                if not top and cursor + 25 != end:
                    raise ValueError(f"Invalid FBX null tail at {cursor}")
                return nodes
            off, nprops, proplen, namelen = struct.unpack_from("<QQQB", buf, cursor)
            cstart = cursor + 25 + namelen + proplen
            if not cursor < cstart <= off <= end:
                raise ValueError(f"Invalid FBX node bounds at {cursor}")
            current = parser.Node(bytes(buf[cursor + 25:cursor + 25 + namelen]), cursor,
                                  off, nprops, proplen, cstart)
            if cstart < off:
                current.children = subtree(cstart, off)
            nodes.append(current)
            cursor = off
        if nodes and not top and not allow_unterminated_children:
            raise ValueError(f"Missing FBX child null terminator at {end}")
        return nodes

    tops = subtree(27, len(buf), top=True)
    if not tops:
        raise ValueError("Source has no FBX nodes")
    if len({n.name for n in tops}) != len(tops):
        raise ValueError("Duplicate FBX top-level sections")
    return tops


def footer_parts(buf, tops, version):
    """Recognize complete tails only; never scan backward past unknown bytes.

    The 16 bytes after the root terminator are an opaque file identifier, not a
    fixed magic value (native Shogun exports vary). The splitter also copies old
    padding verbatim after relocating nodes. Accept 0..15 zero alignment bytes,
    but require the complete versioned suffix at that exact boundary.

    Footerless writers may end exactly at the last complete node, or after a
    complete root terminator and up to 15 zero padding bytes. Partial terminators
    and partial/opaque footers are errors. A wholly absent footer cannot prove
    why it was omitted; this distinction is recorded, not treated as recovery
    from a truncated node or a partially written footer.
    """
    end = tops[-1].off
    tail = buf[end:]
    prefix = NULL_RECORD + FOOTER_MAGIC
    suffix = bytes(4) + struct.pack("<I", version) + bytes(120) + FOOTER_END_MAGIC
    if not tail:
        return FooterParts(prefix, suffix, 0, "nodes_only", False)
    if tail[:25] != NULL_RECORD:
        raise ValueError("Unsupported source footer layout: incomplete root terminator")
    if 25 <= len(tail) <= 40 and not any(tail):
        return FooterParts(prefix, suffix, len(tail) - 25, "terminator_only")
    padding_size = len(tail) - 41 - len(suffix)
    if (not 0 <= padding_size <= 16
            or tail[41:41 + padding_size] != bytes(padding_size)
            or tail[41 + padding_size:] != suffix):
        raise ValueError("Unsupported source footer layout; refusing truncated or opaque bytes")
    prefix = bytes(tail[:41])
    layout = "standard_footer" if prefix[25:] == FOOTER_MAGIC else "variable_identifier_footer"
    return FooterParts(prefix, bytes(tail[41 + padding_size:]), padding_size, layout)


def child_named(original, name):
    matches = [c for c in original.children if c.name == name.encode("ascii")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} under {original.name!r}")
    return matches[0]


def definition_updates(parser, buf, graph, additions):
    definitions = graph.tops["Definitions"]
    counts = Counter(obj.name.decode("ascii") for obj in graph.obj_children)
    counts.update(obj.name.decode("ascii") for obj in additions)
    counts["GlobalSettings"] += 1
    replacements, appended, seen, original_counts = {}, [], set(), {}

    def replace_count(count_node, value):
        props = parser.decode_props(buf, count_node)
        if len(props) != 1 or props[0][0] not in ("I", "L"):
            raise ValueError("Unsupported Definitions Count property")
        replacements[count_node.nstart] = (integer(value, props[0][0]), 1)

    for object_type in definitions.children:
        if object_type.name != b"ObjectType":
            continue
        name = parser.decode_props(buf, object_type)[0][1]
        if name in seen:
            raise ValueError(f"Duplicate Definitions ObjectType: {name}")
        seen.add(name)
        count_node = child_named(object_type, "Count")
        original_counts[name] = parser.decode_props(buf, count_node)[0][1]
        replace_count(count_node, counts[name])
    for name in sorted(counts.keys() - seen):
        appended.append(node("ObjectType", string(name), children=[node("Count", integer(counts[name]))]))
    replace_count(child_named(definitions, "Count"), sum(counts.values()))
    return replacements, appended, dict(counts), original_counts


def fingerprint(buf, original, *, normalize_missing_child_terminators=False):
    """Hash raw nested payloads and tails, with explicit legacy normalization."""
    digest = hashlib.sha256()
    header_payload = memoryview(buf)[original.nstart + 8:original.cstart]
    digest.update(struct.pack("<Q", len(header_payload)))
    digest.update(header_payload)
    digest.update(struct.pack("<Q", len(original.children)))
    for child in original.children:
        digest.update(fingerprint(buf, child,
                                  normalize_missing_child_terminators=normalize_missing_child_terminators))
    tail = tail_bytes(buf, original)
    if normalize_missing_child_terminators and original.children and not tail:
        tail = NULL_RECORD
    digest.update(struct.pack("<Q", len(tail)))
    digest.update(tail)
    return digest.digest()


def array_data(buf, original, kind):
    """Read an array directly into NumPy; do not inflate it into Python lists."""
    start = original.cstart - original.proplen
    if original.nprops != 1 or original.proplen < 13 or buf[start] != ord(kind):
        raise ValueError(f"Expected a single {kind} array under {original.name!r}")
    count, encoding, size = struct.unpack_from("<III", buf, start + 1)
    if 13 + size != original.proplen or encoding not in (0, 1):
        raise ValueError(f"Malformed FBX array under {original.name!r}")
    body = memoryview(buf)[start + 13:original.cstart]
    raw = zlib.decompress(body) if encoding else body
    dtype = np.dtype(_ARRAY_DTYPES[kind])
    if len(raw) != count * dtype.itemsize:
        raise ValueError(f"FBX array length mismatch under {original.name!r}")
    return np.frombuffer(raw, dtype=dtype)


def connection_values(parser, buf, graph):
    return [tuple(value for _, value in parser.decode_props(buf, c))
            for c in graph.tops["Connections"].children]


def dangling_connections(connections, uids):
    return [c for c in connections
            if len(c) < 3 or c[1] not in uids or (c[2] != 0 and c[2] not in uids)]


def properties70(parser, buf, original):
    return {parser.decode_props(buf, p)[0][1]: [v for _, v in parser.decode_props(buf, p)[4:]]
            for p in child_named(original, "Properties70").children}


def verify(parser, source, source_tops, source_graph, output, meshes, expected_counts, metadata):
    output_tops = parse_checked(parser, output)
    output_graph = parser.build_graph(output, output_tops)
    source_uids, output_uids = set(source_graph.uid2node), set(output_graph.uid2node)
    allocated = {m["geometry_uid"] for m in meshes} | {m["model_uid"] for m in meshes}
    allocated.update(uid for m in meshes for uid in m["material_uids"])
    objects = []
    for original in source_graph.obj_children:
        copied = output_graph.uid2node.get(original.uid)
        before = fingerprint(source, original, normalize_missing_child_terminators=True).hex()
        after = fingerprint(output, copied).hex() if copied is not None else None
        objects.append(dict(uid=original.uid, type=original.name.decode("ascii"),
                            source_sha256=before, output_sha256=after, unchanged=before == after))
    old_connections = source_graph.tops["Connections"].children
    new_connections = output_graph.tops["Connections"].children
    connections_ok = len(new_connections) >= len(old_connections) and all(
        fingerprint(source, original, normalize_missing_child_terminators=True) == fingerprint(output, copied)
        for original, copied in zip(old_connections, new_connections))
    decoded_connections = connection_values(parser, output, output_graph)
    source_connections = connection_values(parser, source, source_graph)
    dangling = dangling_connections(decoded_connections, output_uids)
    source_dangling = dangling_connections(source_connections, source_uids)
    new_dangling = list((Counter(dangling) - Counter(source_dangling)).elements())
    untouched_tops = {}
    for original in source_tops:
        name = original.name.decode("ascii")
        if name not in ("Objects", "Connections", "Definitions"):
            copied = output_graph.tops.get(name)
            untouched_tops[name] = copied is not None and fingerprint(
                source, original, normalize_missing_child_terminators=True) == fingerprint(output, copied)
    actual_counts = Counter(obj.name.decode("ascii") for obj in output_graph.obj_children)
    actual_counts["GlobalSettings"] += 1
    declared_counts = {}
    for entry in output_graph.tops["Definitions"].children:
        if entry.name == b"ObjectType":
            name = parser.decode_props(output, entry)[0][1]
            declared_counts[name] = parser.decode_props(output, child_named(entry, "Count"))[0][1]
    total = parser.decode_props(output, child_named(output_graph.tops["Definitions"], "Count"))[0][1]
    definitions_ok = (all(declared_counts.get(k) == v for k, v in actual_counts.items())
                      and all(actual_counts.get(k, 0) == v for k, v in declared_counts.items())
                      and total == sum(actual_counts.values()) and dict(actual_counts) == expected_counts)
    mesh_reports, expected_added_connections = [], []
    for mesh in meshes:
        geometry = output_graph.uid2node[mesh["geometry_uid"]]
        model = output_graph.uid2node[mesh["model_uid"]]
        vertices = array_data(output, child_named(geometry, "Vertices"), "d")
        polygons = array_data(output, child_named(geometry, "PolygonVertexIndex"), "i")
        material_layer = child_named(geometry, "LayerElementMaterial")
        material_indices = array_data(output, child_named(material_layer, "Materials"), "i")
        normal_layer = child_named(geometry, "LayerElementNormal")
        normal_mapping, expected_normals = mesh_normals(mesh)
        transforms = properties70(parser, output, model)
        identity = all(transforms.get(k) == list(v) for k, v in {
            "Lcl Translation": (0, 0, 0), "Lcl Rotation": (0, 0, 0), "Lcl Scaling": (1, 1, 1),
            "GeometricTranslation": (0, 0, 0), "GeometricRotation": (0, 0, 0), "GeometricScaling": (1, 1, 1),
        }.items())
        expected_connections = [("OO", mesh["geometry_uid"], mesh["model_uid"]),
                                ("OO", mesh["model_uid"], mesh["parent_uid"])]
        expected_connections.extend(("OO", uid, mesh["model_uid"]) for uid in mesh["material_uids"])
        expected_added_connections.extend(expected_connections)
        model_parents = [c[2] for c in decoded_connections if c[0] == "OO" and c[1] == mesh["model_uid"]]
        material_refs = [c[1] for c in decoded_connections if c[0] == "OO" and c[2] == mesh["model_uid"]
                         and c[1] in output_graph.uid2node and output_graph.uid2node[c[1]].name == b"Material"]
        material_values_ok = True
        for uid, material in zip(mesh["material_uids"], mesh["materials"]):
            mat = output_graph.uid2node[uid]
            properties = properties70(parser, output, mat)
            material_values_ok &= (mat.name == b"Material" and properties.get("DiffuseColor") == list(material["diffuse_rgb"])
                                   and properties.get("DiffuseFactor") == [1.]
                                   and properties.get("Opacity") == [material["opacity"]]
                                   and properties.get("TransparencyFactor") == [1. - material["opacity"]]
                                   and parser.decode_props(output, child_named(mat, "ShadingModel"))[0][1] == "lambert")
        checks = dict(
            vertices_match_payload=bool(np.array_equal(vertices, mesh["vertices"].reshape(-1))),
            triangles_match_payload=bool(np.array_equal(polygons, polygon_indices(mesh))),
            normals_match_payload_or_face_normals=bool(
                np.array_equal(array_data(output, child_named(normal_layer, "Normals"), "d"), expected_normals)
                and parser.decode_props(output, child_named(normal_layer, "MappingInformationType"))[0][1] == normal_mapping
                and parser.decode_props(output, child_named(normal_layer, "ReferenceInformationType"))[0][1] == "Direct"),
            per_face_material_indices_match=bool(np.array_equal(material_indices, mesh["triangle_material_indices"])
                and parser.decode_props(output, child_named(material_layer, "MappingInformationType"))[0][1] == "ByPolygon"
                and parser.decode_props(output, child_named(material_layer, "ReferenceInformationType"))[0][1] == "IndexToDirect"),
            model_identity=identity, exact_original_parent=model_parents == [mesh["parent_uid"]],
            material_references_valid=material_refs == mesh["material_uids"], material_values_match_payload=material_values_ok,
            geometry_type_valid=geometry.name == b"Geometry" and parser.decode_props(output, geometry)[2][1] == "Mesh",
            model_type_valid=model.name == b"Model" and parser.decode_props(output, model)[2][1] == "Mesh")
        if mesh["vertex_colors"] is not None:
            color_layer = child_named(geometry, "LayerElementColor")
            layer = child_named(geometry, "Layer")
            layer_elements = [
                (parser.decode_props(output, child_named(entry, "Type"))[0][1],
                 parser.decode_props(output, child_named(entry, "TypedIndex"))[0][1])
                for entry in layer.children if entry.name == b"LayerElement"
            ]
            checks["vertex_colors_match_payload"] = bool(
                np.array_equal(array_data(output, child_named(color_layer, "Colors"), "d"),
                               mesh["vertex_colors"].reshape(-1))
                and parser.decode_props(output, child_named(color_layer, "MappingInformationType"))[0][1] == "ByVertice"
                and parser.decode_props(output, child_named(color_layer, "ReferenceInformationType"))[0][1] == "Direct"
                and ("LayerElementColor", 0) in layer_elements)
        mesh_reports.append(dict(name=mesh["name"], prop=mesh["prop"], parent_uid=mesh["parent_uid"],
                                 model_uid=mesh["model_uid"], geometry_uid=mesh["geometry_uid"], material_uids=mesh["material_uids"],
                                 vertex_count=vertices.size // 3, triangle_count=polygons.size // 3,
                                 normal_mapping=normal_mapping, material_count=len(mesh["material_uids"]),
                                 checks=checks, passed=all(checks.values())))
    source_footer = footer_parts(source, source_tops, 7700)
    output_footer = footer_parts(output, output_tops, 7700)
    footer_ok = (source_footer.prefix == output_footer.prefix
                 and source_footer.suffix == output_footer.suffix)
    output_footer_valid = (output_footer.footer_present
                           and output_footer.padding_size == (-(output_tops[-1].off + 41)) % 16)
    normalization = []
    if not source_footer.terminator_present:
        normalization.append("added_top_level_terminator")
    if not source_footer.footer_present:
        normalization.append("added_standard_footer")
    if (source_footer.padding_size != output_footer.padding_size
            or source_footer.padding_size != (-(source_tops[-1].off + 41)) % 16):
        normalization.append("realigned_footer_padding")
    def missing_child_terminators(nodes):
        return sum(int(bool(n.children) and not tail_bytes(source, n))
                   + missing_child_terminators(n.children) for n in nodes)

    checks = dict(original_uid_set_retained=source_uids <= output_uids,
                  exact_expected_uid_set=output_uids == source_uids | allocated,
                  unique_object_uids=len(output_graph.obj_children) == len(output_uids),
                  original_object_order_unchanged=[n.uid for n in output_graph.obj_children[:len(objects)]] == [n.uid for n in source_graph.obj_children],
                  all_original_nested_payloads_and_existing_tails_preserved=all(o["unchanged"] for o in objects),
                  original_connections_byte_equivalent_and_ordered=connections_ok,
                  added_connections_exact=decoded_connections[len(old_connections):] == expected_added_connections,
                  no_new_dangling_references=not new_dangling,
                  inherited_dangling_references_unchanged=Counter(dangling) == Counter(source_dangling),
                  global_settings_unchanged=untouched_tops["GlobalSettings"],
                  untouched_top_level_nodes_unchanged=all(untouched_tops.values()),
                  top_level_order_unchanged=[t.name for t in source_tops] == [t.name for t in output_tops],
                  definitions_counts_correct=definitions_ok, node_offsets_and_null_terminators_valid=True,
                  footer_nonpadding_bytes_preserved=footer_ok, canonical_output_footer_valid=output_footer_valid,
                  all_meshes_valid=all(m["passed"] for m in mesh_reports))
    return dict(schema_version=1, verification_scope="Binary scene integrity only; not animated physical-fit quality",
                **metadata, passed=all(checks.values()), checks=checks,
                mesh_count=len(meshes), source_object_count=len(source_uids), output_object_count=len(output_uids),
                new_object_count=len(allocated), original_objects=objects,
                original_connection_count=len(old_connections), output_connection_count=len(new_connections),
                inherited_dangling_reference_count=len(source_dangling),
                new_dangling_references=[list(c) for c in new_dangling],
                inherited_dangling_references=[list(c) for c in source_dangling], layout_problems=[],
                untouched_top_level_nodes=untouched_tops,
                node_normalization=dict(added_child_terminators=missing_child_terminators(source_tops),
                                        source_child_bounds_checked=True, output_child_terminators_required=True),
                fingerprint_basis="Raw nested properties/order and existing tails; only missing child terminators normalized",
                definitions=dict(actual_counts=dict(actual_counts), declared_counts=declared_counts, total=total),
                footer=dict(source_format=source_footer.layout, output_format=output_footer.layout,
                            source_tail_offset=source_tops[-1].off,
                            source_tail_bytes=len(source) - source_tops[-1].off,
                            source_terminator_present=source_footer.terminator_present,
                            source_footer_present=source_footer.footer_present,
                            source_identifier_hex=source_footer.prefix[25:].hex() if source_footer.footer_present else None,
                            output_identifier_hex=output_footer.prefix[25:].hex(),
                            source_padding_bytes=source_footer.padding_size,
                            output_padding_bytes=output_footer.padding_size,
                            normalization=normalization, nonpadding_bytes_preserved=footer_ok,
                            output_alignment_valid=output_footer_valid), meshes=mesh_reports)


def stage_output(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        return Path(temporary)
    except BaseException:
        os.unlink(temporary)
        raise


def publish_output(temporary, path, *, overwrite):
    if overwrite:
        os.replace(temporary, path)
    elif os.name == "nt":
        # Windows rename is atomic and fails if the destination already exists.
        os.rename(temporary, path)
    else:
        # POSIX rename replaces existing files; linking is atomic create-if-absent.
        os.link(temporary, path)
        temporary.unlink()


def graft_file(source: Path, destination: Path, meshes: list[dict], *,
               verification_path: Path, overwrite: bool = False) -> dict:
    """Validate, graft, verify, then atomically publish FBX and JSON report.

    All vertices/normals are already in native bone-local units. Inputs are never
    modified. Existing destination/report files require explicit ``overwrite``;
    neither path may alias the source or one another, including hard links.
    Complete versioned footers retain their opaque identifier; copied alignment
    padding is normalized. Exact node-boundary EOF or a complete zero root
    terminator may omit the footer, which is explicitly reported and synthesized.
    Partial nodes, terminators, footers and unknown tail bytes are rejected.
    Legacy child scopes ending exactly at their last child are accepted with
    checked bounds; output scopes always receive the required null terminators.
    Inherited dangling connections are preserved and reported separately.
    """
    source_path = Path(source).resolve(strict=True)
    protected = [source_path, Path(__file__).resolve(), PARSER_PATH.resolve()]
    output_path = checked_output(destination, protected, overwrite=overwrite)
    report_path = checked_output(verification_path, protected + [output_path], overwrite=overwrite)
    source_bytes = source_path.read_bytes()
    parser = load_parser()
    source_tops = parse_checked(parser, source_bytes, allow_unterminated_children=True)
    source_footer = footer_parts(source_bytes, source_tops, 7700)
    top_names = {n.name for n in source_tops}
    for required in ("Objects", "Connections", "Definitions", "GlobalSettings"):
        if required.encode("ascii") not in top_names:
            raise ValueError(f"Missing required top-level section: {required}")
    graph = parser.build_graph(source_bytes, source_tops)
    if len(graph.uid2node) != len(graph.obj_children):
        raise ValueError("Source contains duplicate object UIDs")
    if not isinstance(meshes, list) or not meshes:
        raise ValueError("Supply at least one real mesh")
    names = set()
    mesh_payloads = [validate_mesh(mesh, graph, names) for mesh in meshes]
    used_uids = set(graph.uid2node) | {0}
    for connection in graph.conns:
        used_uids.update(connection[1:3])
    documents = graph.tops.get("Documents")
    if documents is not None:
        for document in documents.children:
            if document.name == b"Document":
                used_uids.add(parser.decode_props(source_bytes, document)[0][1])
    next_uid = max(used_uids) + 1

    def allocate():
        nonlocal next_uid
        if next_uid > 9223372036854775807:
            raise ValueError("No available positive signed int64 object UID")
        result = next_uid
        next_uid += 1
        return result

    additions, connections = [], []
    for mesh in mesh_payloads:
        objects, links = create_mesh_objects(mesh, allocate)
        additions.extend(objects)
        connections.extend(links)
    replacements, definitions, expected_counts, original_counts = definition_updates(
        parser, source_bytes, graph, additions)
    extras = {graph.tops["Objects"].nstart: additions,
              graph.tops["Connections"].nstart: connections,
              graph.tops["Definitions"].nstart: definitions}
    output = bytearray(source_bytes[:27])
    for original in source_tops:
        emit_node(source_bytes, output, original, replacements, extras)
    output.extend(source_footer.prefix)
    output.extend(bytes((-len(output)) % 16))
    output.extend(source_footer.suffix)
    metadata = dict(source_fbx=str(source_path), output_fbx=str(output_path), verification_path=str(report_path),
                    source_sha256=hashlib.sha256(source_bytes).hexdigest(), output_sha256=hashlib.sha256(output).hexdigest(),
                    source_bytes=len(source_bytes), output_bytes=len(output), fbx_version=7700,
                    original_definitions_counts=original_counts,
                    coordinate_policy="Payload vertices/normals already native bone-local; no axis, unit or fit transform applied")
    report = verify(parser, source_bytes, source_tops, graph, output, mesh_payloads, expected_counts, metadata)
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise ValueError("Integrity verification failed; no outputs published: " + ", ".join(failed))
    # Stage both complete artifacts before publishing either. Recheck aliases and
    # overwrite authorization immediately before publication, too.
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    staged = []
    try:
        staged.append(stage_output(output_path, output))
        staged.append(stage_output(report_path, report_bytes))
        checked_output(output_path, protected, overwrite=overwrite)
        checked_output(report_path, protected + [output_path], overwrite=overwrite)
        publish_output(staged[1], report_path, overwrite=overwrite)
        publish_output(staged[0], output_path, overwrite=overwrite)
    finally:
        for temporary in staged:
            if temporary.exists():
                temporary.unlink()
    return report

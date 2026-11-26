"""Mesh graft regressions using small real binary FBX7700 documents, no SDK."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

import mesh_fbx


def _string(value):
    raw = value.encode("utf-8")
    return b"S" + struct.pack("<I", len(raw)) + raw


def _int(value, kind="L"):
    return kind.encode() + struct.pack("<q" if kind == "L" else "<i", value)


def _double(value):
    return b"D" + struct.pack("<d", value)


def _array(kind, values):
    formats = {"l": "q", "f": "f", "d": "d", "i": "i"}
    raw = struct.pack("<" + formats[kind] * len(values), *values)
    compressed = zlib.compress(raw)
    return kind.encode() + struct.pack("<III", len(values), 1, len(compressed)) + compressed


def _node(name, properties=(), children=(), *, terminated=False):
    return name.encode(), list(properties), list(children), terminated


def _emit(record, offset):
    name, properties, children, terminated = record
    payload = b"".join(properties)
    size = 25 + len(name) + len(payload)
    encoded_children = bytearray()
    for child in children:
        encoded_children.extend(_emit(child, offset + size + len(encoded_children)))
    if children or terminated:
        encoded_children.extend(bytes(25))
    end = offset + size + len(encoded_children)
    return (struct.pack("<QQQB", end, len(properties), len(payload), len(name))
            + name + payload + encoded_children)


def _property(name, kind, values):
    return _node("P", [_string(name), _string(kind), _string(""), _string("A"), *values])


def _write_scene(path, *, version=7700, dangling=True):
    """An animated prop, marker sibling, material, raw payload and take metadata."""
    def obj(kind, uid, name, subtype, children=()):
        return _node(kind, [_int(uid), _string(name + "\x00\x01" + kind), _string(subtype)], children)

    def conn(kind, child, parent, member=None):
        props = [_string(kind), _int(child), _int(parent)]
        if member is not None:
            props.append(_string(member))
        return _node("C", props)

    objects = [
        obj("Model", 10, "cup:Labeling:root", "LimbNode", [
            _node("Version", [_int(232, "I")]),
            _node("Properties70", children=[
                _property("Lcl Translation", "Lcl Translation", [_double(v) for v in (1., 2., 3.)]),
                _property("Lcl Rotation", "Lcl Rotation", [_double(v) for v in (0., 0., 0.)]),
                _property("Lcl Scaling", "Lcl Scaling", [_double(v) for v in (1., 1., 1.)]),
                _property("UserMetadata", "KString", [_string("untouched é marker data")]),
            ]),
            _node("OpaquePayload", [b"R" + struct.pack("<I", 7) + b"\x00\xff\x00data"], terminated=True),
        ]),
        obj("Model", 11, "cup:M_1", "Null", [
            _node("Properties70", children=[
                _property("Lcl Translation", "Lcl Translation", [_double(v) for v in (4., 5., 6.)]),
            ]),
        ]),
        obj("AnimationStack", 20, "Take 001", "", [
            _node("Properties70", children=[
                _property("LocalStart", "KTime", [_int(0)]),
                _property("LocalStop", "KTime", [_int(92372316000)]),
            ]),
        ]),
        obj("AnimationLayer", 21, "BaseLayer", ""),
        obj("AnimationCurveNode", 22, "T", "", [
            _node("Properties70", children=[_property("d|X", "Number", [_double(1.)])]),
        ]),
        obj("AnimationCurve", 23, "TX", "", [
            _node("Default", [_double(1.)]),
            _node("KeyVer", [_int(4008, "I")]),
            _node("KeyTime", [_array("l", [0, 46186158000, 92372316000])]),
            _node("KeyValueFloat", [_array("f", [1., 4., -2.])]),
            _node("KeyAttrFlags", [_array("i", [24836, 24836, 24836])]),
            _node("KeyAttrDataFloat", [_array("f", [0.] * 12)]),
            _node("KeyAttrRefCount", [_array("i", [1, 1, 1])]),
        ]),
        obj("Material", 24, "OriginalMaterial", "", [
            _node("Properties70", children=[
                _property("DiffuseColor", "Color", [_double(v) for v in (.1, .2, .3)]),
            ]),
        ]),
    ]
    connections = [conn("OO", 10, 0), conn("OO", 11, 0), conn("OO", 21, 20),
                   conn("OO", 22, 21), conn("OP", 22, 10, "Lcl Translation"),
                   conn("OP", 23, 22, "d|X")]
    if dangling:
        # These IDs must stay unassigned, or grafting would silently repair the source.
        connections += [conn("OP", 9000, 10, "InheritedMissing"), conn("OO", 11, 9001),
                        conn("OP", 9000, 10, "InheritedMissing")]
    roots = [
        _node("FBXHeaderExtension", children=[_node("FBXVersion", [_int(7700, "I")])]),
        _node("GlobalSettings", children=[_node("Properties70", children=[
            _property("UnitScaleFactor", "double", [_double(1.)]),
            _property("UpAxis", "int", [_int(2, "I")]),
        ])]),
        _node("Documents", [_int(1, "I")], [
            _node("Document", [_int(950), _string("Scene"), _string("Scene")], [
                _node("RootNode", [_int(0)]),
            ]),
        ]),
        _node("Definitions", children=[
            _node("Version", [_int(100, "I")]), _node("Count", [_int(999, "I")]),
            _node("ObjectType", [_string("Model")], [
                _node("Count", [_int(999, "I")]),
                _node("PropertyTemplate", [_string("FbxNode")], [
                    _node("Properties70", children=[_property("Visibility", "Visibility", [_double(1.)])]),
                ]),
            ]),
            _node("ObjectType", [_string("UnusedType")], [_node("Count", [_int(7, "I")])]),
        ]),
        _node("Objects", children=objects), _node("Connections", children=connections),
        _node("Takes", children=[_node("Current", [_string("Take 001")])]),
    ]
    result = bytearray(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", version))
    for root in roots:
        result.extend(_emit(root, len(result)))
    result.extend(bytes(25))
    result.extend(bytes.fromhex("fabca90ed4c8d667b070fa8318fb2b7d"))
    result.extend(bytes((-len(result)) % 16))
    result.extend(bytes(4) + struct.pack("<I", version) + bytes(120)
                  + bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b"))
    path.write_bytes(result)
    return bytes(result)


def _mesh(*, numpy=True, normals=True):
    result = dict(
        name="cup_body", prop="cup", parent_uid=10,
        vertices=[[.25, -1., 0.], [2.25, -1., 0.], [2.25, 1., 0.], [.25, 1., 0.]],
        triangles=[[0, 1, 2], [0, 2, 3]],
        materials=[dict(name="Body cream", diffuse_rgb=[.91, .82, .61], opacity=1., color_space="sRGB"),
                   dict(name="Dark rim", diffuse_rgb=[.12, .17, .23], opacity=.75, color_space="sRGB")],
        triangle_material_indices=[0, 1],
    )
    if normals:
        result["normals"] = [[0., 0., 1.]] * 4
    if numpy:
        result["vertices"] = np.asarray(result["vertices"], dtype=np.float64)
        result["triangles"] = np.asarray(result["triangles"], dtype=np.int32)
        result["triangle_material_indices"] = np.asarray(result["triangle_material_indices"], dtype=np.int64)
        if normals:
            result["normals"] = np.asarray(result["normals"], dtype=np.float32)
    return result


class MeshFbxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.fbx"
        self.output = self.root / "meshed.fbx"
        self.report = self.root / "verification.json"
        self.original = _write_scene(self.source)
        self.parser = mesh_fbx.load_parser()

    def graft(self, meshes=None, **kwargs):
        return mesh_fbx.graft_file(self.source, self.output, [_mesh()] if meshes is None else meshes,
                                   verification_path=self.report, **kwargs)

    def graph(self, path):
        buf, version, tops = self.parser.parse_file(path)
        self.assertEqual(version, 7700)
        return buf, self.parser.build_graph(buf, tops)

    def test_numpy_geometry_material_regions_and_original_payloads(self):
        payload = _mesh()
        saved = {key: value.copy() for key, value in payload.items() if isinstance(value, np.ndarray)}
        for value in saved:
            payload[value].flags.writeable = False
        result = self.graft([payload])
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(self.source.read_bytes(), self.original)
        for key, expected in saved.items():
            np.testing.assert_array_equal(payload[key], expected)
            self.assertFalse(payload[key].flags.writeable)
        self.assertNotIn("geometry_uid", payload)
        self.assertEqual(json.loads(self.report.read_text(encoding="utf-8")), result)
        self.assertEqual(result["output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(result["source_sha256"], hashlib.sha256(self.original).hexdigest())
        before, source_graph = self.graph(self.source)
        after, output_graph = self.graph(self.output)
        for uid, original in source_graph.uid2node.items():
            self.assertEqual(mesh_fbx.fingerprint(before, original),
                             mesh_fbx.fingerprint(after, output_graph.uid2node[uid]))
        added = result["meshes"][0]
        geometry = output_graph.uid2node[added["geometry_uid"]]
        model = output_graph.uid2node[added["model_uid"]]
        self.assertEqual(self.parser.decode_props(after, model)[2][1], "Mesh")
        np.testing.assert_array_equal(
            mesh_fbx.array_data(after, mesh_fbx.child_named(geometry, "Vertices"), "d").reshape(-1, 3),
            saved["vertices"])
        np.testing.assert_array_equal(
            mesh_fbx.array_data(after, mesh_fbx.child_named(geometry, "PolygonVertexIndex"), "i"),
            [0, 1, -3, 0, 2, -4])
        material_layer = mesh_fbx.child_named(geometry, "LayerElementMaterial")
        np.testing.assert_array_equal(
            mesh_fbx.array_data(after, mesh_fbx.child_named(material_layer, "Materials"), "i"), [0, 1])
        links = [c[1] for c in output_graph.conns if c[0] == "OO" and c[2] == added["model_uid"]
                 and output_graph.uid2node[c[1]].name == b"Material"]
        self.assertEqual(links, added["material_uids"])
        for uid, expected in zip(links, payload["materials"]):
            material = output_graph.uid2node[uid]
            props = mesh_fbx.properties70(self.parser, after, material)
            self.assertEqual(props["DiffuseColor"], expected["diffuse_rgb"])
            self.assertEqual(props["Opacity"], [expected["opacity"]])
        self.assertEqual(result["definitions"]["actual_counts"]["Model"], 3)
        self.assertEqual(result["definitions"]["declared_counts"]["UnusedType"], 0)
        self.assertEqual(result["definitions"]["total"], result["output_object_count"] + 1)

    def test_vertex_rgba_survives_export_with_direct_vertex_mapping(self):
        payload = _mesh()
        payload["vertex_colors"] = np.array([
            [1., 0., 0., 1.], [0., 1., 0., .5],
            [0., 0., 1., .25], [.1, .2, .3, 0.],
        ])
        result = self.graft([payload])
        after, graph = self.graph(self.output)
        geometry = graph.uid2node[result["meshes"][0]["geometry_uid"]]
        colors = mesh_fbx.child_named(geometry, "LayerElementColor")
        np.testing.assert_array_equal(
            mesh_fbx.array_data(after, mesh_fbx.child_named(colors, "Colors"), "d").reshape(-1, 4),
            payload["vertex_colors"])
        self.assertEqual(self.parser.decode_props(
            after, mesh_fbx.child_named(colors, "MappingInformationType"))[0][1], "ByVertice")
        self.assertEqual(self.parser.decode_props(
            after, mesh_fbx.child_named(colors, "ReferenceInformationType"))[0][1], "Direct")
        layer = mesh_fbx.child_named(geometry, "Layer")
        self.assertIn("LayerElementColor", [
            self.parser.decode_props(after, mesh_fbx.child_named(entry, "Type"))[0][1]
            for entry in layer.children if entry.name == b"LayerElement"])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_invalid_vertex_colors_do_not_publish(self):
        for colors in (np.ones((3, 4)), np.full((4, 4), 1.1),
                       np.full((4, 4), -.1), np.full((4, 4), np.nan)):
            with self.subTest(colors=colors):
                payload = _mesh()
                payload["vertex_colors"] = colors
                with self.assertRaises(ValueError):
                    self.graft([payload])
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())

    def test_legacy_split_scopes_gain_terminators_without_changing_animation(self):
        _, _, tops = self.parser.parse_file(self.source)
        graph = self.parser.build_graph(self.original, tops)
        legacy, _, _ = self.parser.splice(self.original, tops, set(graph.uid2node) | {9999})
        self.source.write_bytes(legacy)
        with self.assertRaises(ValueError):
            mesh_fbx.parse_checked(self.parser, legacy)
        result = self.graft()
        self.assertTrue(result["passed"])
        self.assertEqual(self.source.read_bytes(), legacy)
        before, source_graph = self.graph(self.source)
        after, output_graph = self.graph(self.output)
        curve_uid = source_graph.model2curves[10][("Lcl Translation", "d|X")]
        old_curve, new_curve = source_graph.uid2node[curve_uid], output_graph.uid2node[curve_uid]
        for name in ("KeyTime", "KeyValueFloat"):
            old = mesh_fbx.child_named(old_curve, name)
            new = mesh_fbx.child_named(new_curve, name)
            self.assertEqual(before[old.nstart + 8:old.cstart], after[new.nstart + 8:new.cstart])
        self.assertEqual(after[new_curve.off - 25:new_curve.off], bytes(25))

    def test_legacy_normalization_still_rejects_a_child_crossing_its_parent(self):
        _, _, tops = self.parser.parse_file(self.source)
        graph = self.parser.build_graph(self.original, tops)
        legacy, _, _ = self.parser.splice(self.original, tops, set(graph.uid2node) | {9999})
        nodes = mesh_fbx.parse_checked(self.parser, legacy, allow_unterminated_children=True)
        broken = bytearray(legacy)
        parent = next(n for n in nodes if n.children)
        struct.pack_into("<Q", broken, parent.nstart, parent.children[-1].off - 1)
        self.source.write_bytes(broken)
        with self.assertRaises(ValueError):
            self.graft()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_consumer_sees_same_animation_on_original_parent_and_child_mesh(self):
        result = self.graft()
        before, source_graph = self.graph(self.source)
        after, output_graph = self.graph(self.output)
        # Resolve through consumer-facing OP and OO links, not just report flags.
        self.assertEqual(source_graph.model2curves[10], output_graph.model2curves[10])
        curve_uid = output_graph.model2curves[10][("Lcl Translation", "d|X")]
        def sample(buf, graph, seconds):
            curve = graph.uid2node[curve_uid]
            times = np.asarray(self.parser.decode_props(buf, mesh_fbx.child_named(curve, "KeyTime"))[0][1]) / 46186158000
            values = self.parser.decode_props(buf, mesh_fbx.child_named(curve, "KeyValueFloat"))[0][1]
            return np.interp(seconds, times, values)
        child = result["meshes"][0]
        parents = [c[2] for c in output_graph.conns if c[0] == "OO" and c[1] == child["model_uid"]]
        self.assertEqual(parents, [10])
        identity = mesh_fbx.properties70(self.parser, after, output_graph.uid2node[child["model_uid"]])
        self.assertEqual(identity["Lcl Translation"], [0., 0., 0.])
        self.assertEqual(identity["Lcl Rotation"], [0., 0., 0.])
        self.assertEqual(identity["Lcl Scaling"], [1., 1., 1.])
        for time in (0., .25, 1., 1.75, 2.):
            self.assertEqual(sample(before, source_graph, time), sample(after, output_graph, time))
            self.assertEqual(sample(before, source_graph, time) + .25,
                             sample(after, output_graph, time) + _mesh()["vertices"][0, 0])
        self.assertEqual(mesh_fbx.fingerprint(before, source_graph.tops["Takes"]),
                         mesh_fbx.fingerprint(after, output_graph.tops["Takes"]))

    def test_inherited_dangling_references_remain_separate_and_reserved(self):
        result = self.graft()
        self.assertEqual(result["inherited_dangling_reference_count"], 3)
        self.assertEqual(result["inherited_dangling_references"],
                         [["OP", 9000, 10, "InheritedMissing"], ["OO", 11, 9001],
                          ["OP", 9000, 10, "InheritedMissing"]])
        self.assertEqual(result["new_dangling_references"], [])
        self.assertTrue(result["checks"]["inherited_dangling_references_unchanged"])
        for mesh in result["meshes"]:
            self.assertGreater(mesh["geometry_uid"], 9001)
            self.assertGreater(mesh["model_uid"], 9001)
            self.assertTrue(all(uid > 9001 for uid in mesh["material_uids"]))

    def test_clean_source_preserves_document_uid_and_has_no_inherited_defects(self):
        self.original = _write_scene(self.source, dangling=False)
        result = self.graft()
        self.assertTrue(result["passed"])
        self.assertEqual(result["inherited_dangling_reference_count"], 0)
        self.assertEqual(result["inherited_dangling_references"], [])
        self.assertEqual(result["new_dangling_references"], [])
        self.assertGreater(result["meshes"][0]["geometry_uid"], 950)
        self.assertEqual(self.source.read_bytes(), self.original)
        before, source_graph = self.graph(self.source)
        after, output_graph = self.graph(self.output)
        original_type = next(n for n in source_graph.tops["Definitions"].children
                             if n.name == b"ObjectType"
                             and self.parser.decode_props(before, n)[0][1] == "Model")
        output_type = next(n for n in output_graph.tops["Definitions"].children
                           if n.name == b"ObjectType"
                           and self.parser.decode_props(after, n)[0][1] == "Model")
        self.assertEqual(
            mesh_fbx.fingerprint(before, mesh_fbx.child_named(original_type, "PropertyTemplate")),
            mesh_fbx.fingerprint(after, mesh_fbx.child_named(output_type, "PropertyTemplate")))

    def test_list_payload_face_normals_and_multiple_parts(self):
        first = _mesh(numpy=False, normals=False)
        second = _mesh(numpy=False, normals=False)
        second["name"] = "cup_handle"
        result = self.graft([first, second])
        self.assertTrue(result["passed"])
        self.assertEqual(result["mesh_count"], 2)
        self.assertEqual([m["normal_mapping"] for m in result["meshes"]], ["ByPolygonVertex"] * 2)
        buf, graph = self.graph(self.output)
        for record in result["meshes"]:
            layer = mesh_fbx.child_named(graph.uid2node[record["geometry_uid"]], "LayerElementNormal")
            normals = mesh_fbx.array_data(buf, mesh_fbx.child_named(layer, "Normals"), "d").reshape(-1, 3)
            np.testing.assert_array_equal(normals, [[0., 0., 1.]] * 6)
        self.assertEqual(first, _mesh(numpy=False, normals=False))

    def test_standard_footer_remains_complete_and_preserves_identifier(self):
        result = self.graft()
        footer = result["footer"]
        self.assertEqual(footer["source_format"], "standard_footer")
        self.assertEqual(footer["output_format"], "standard_footer")
        self.assertTrue(footer["source_footer_present"])
        self.assertTrue(footer["source_terminator_present"])
        self.assertEqual(footer["source_identifier_hex"], "fabca90ed4c8d667b070fa8318fb2b7d")
        self.assertEqual(footer["output_identifier_hex"], footer["source_identifier_hex"])
        self.assertTrue(result["checks"]["canonical_output_footer_valid"])
        self.assertEqual(self.output.read_bytes()[-144:], self.original[-144:])

    def test_native_group6_variable_identifier_is_not_a_fixed_magic(self):
        # Exact identifier observed after the complete last node of the native
        # group6 export; the remaining suffix is the same versioned FBX footer.
        identifier = bytes.fromhex("fabca903d9c8d76cba72fd8816f92c76")
        end = self.parser.parse_file(self.source)[2][-1].off
        padding = (-(end + 41)) % 16
        data = self.original[:end] + bytes(25) + identifier + bytes(padding) + self.original[-144:]
        self.source.write_bytes(data)
        result = self.graft()
        self.assertTrue(result["passed"])
        self.assertEqual(result["footer"]["source_format"], "variable_identifier_footer")
        self.assertEqual(result["footer"]["source_identifier_hex"], identifier.hex())
        self.assertEqual(result["footer"]["output_identifier_hex"], identifier.hex())
        self.assertTrue(result["footer"]["nonpadding_bytes_preserved"])
        self.assertNotIn("added_standard_footer", result["footer"]["normalization"])
        before, source_graph = self.graph(self.source)
        after, output_graph = self.graph(self.output)
        for uid, original in source_graph.uid2node.items():
            self.assertEqual(mesh_fbx.fingerprint(before, original),
                             mesh_fbx.fingerprint(after, output_graph.uid2node[uid]))
        self.assertEqual(mesh_fbx.connection_values(self.parser, before, source_graph),
                         mesh_fbx.connection_values(self.parser, after, output_graph)[
                             :len(source_graph.tops["Connections"].children)])
        self.assertEqual(self.source.read_bytes(), data)

    def test_all_copied_footer_padding_widths_are_normalized(self):
        end = self.parser.parse_file(self.source)[2][-1].off
        prefix = self.original[end:end + 41]
        suffix = self.original[-144:]
        for padding in range(16):
            with self.subTest(padding=padding):
                data = self.original[:end] + prefix + bytes(padding) + suffix
                self.source.write_bytes(data)
                result = self.graft(overwrite=True)
                output = self.output.read_bytes()
                output_end = self.parser.parse_file(self.output)[2][-1].off
                expected_padding = (-(output_end + 41)) % 16
                self.assertEqual(output[output_end:], prefix + bytes(expected_padding) + suffix)
                self.assertEqual(result["footer"]["source_padding_bytes"], padding)
                self.assertEqual(result["footer"]["output_padding_bytes"], expected_padding)
                self.assertTrue(result["footer"]["output_alignment_valid"])
                if padding != (-(end + 41)) % 16 or padding != expected_padding:
                    self.assertIn("realigned_footer_padding", result["footer"]["normalization"])
                self.assertEqual(self.source.read_bytes(), data)

    def test_footerless_exact_node_and_complete_terminator_boundaries(self):
        end = self.parser.parse_file(self.source)[2][-1].off
        for tail_length in (0, *range(25, 41)):
            with self.subTest(tail_length=tail_length):
                data = self.original[:end] + bytes(tail_length)
                self.source.write_bytes(data)
                result = self.graft(overwrite=True)
                footer = result["footer"]
                self.assertEqual(footer["source_format"], "nodes_only" if not tail_length else "terminator_only")
                self.assertFalse(footer["source_footer_present"])
                self.assertIsNone(footer["source_identifier_hex"])
                self.assertEqual(footer["source_terminator_present"], bool(tail_length))
                self.assertIn("added_standard_footer", footer["normalization"])
                self.assertEqual("added_top_level_terminator" in footer["normalization"], not tail_length)
                output = self.output.read_bytes()
                output_end = self.parser.parse_file(self.output)[2][-1].off
                expected_prefix = bytes(25) + bytes.fromhex("fabca90ed4c8d667b070fa8318fb2b7d")
                self.assertEqual(output[output_end:output_end + 41], expected_prefix)
                self.assertEqual(output[-144:], self.original[-144:])
                self.assertTrue(result["checks"]["canonical_output_footer_valid"])
                self.assertTrue(result["checks"]["original_connections_byte_equivalent_and_ordered"])
                self.assertEqual(self.source.read_bytes(), data)

    def test_every_partial_terminator_and_footer_boundary_is_rejected(self):
        end = self.parser.parse_file(self.source)[2][-1].off
        tail = self.original[end:]
        # EOF exactly after a complete root terminator is indistinguishable
        # from an intentionally footerless writer. No other partial prefix of
        # this nonzero-identifier footer is a supported footerless layout.
        for cut in range(1, len(tail)):
            if cut == 25:
                continue
            with self.subTest(cut=cut):
                data = self.original[:end] + tail[:cut]
                self.source.write_bytes(data)
                with self.assertRaises(ValueError):
                    self.graft()
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())
                self.assertEqual(self.source.read_bytes(), data)

    def test_partial_final_node_is_not_mistaken_for_footerless_output(self):
        end = self.parser.parse_file(self.source)[2][-1].off
        for removed in range(1, 26):
            with self.subTest(removed=removed):
                data = self.original[:end - removed]
                self.source.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "node bounds"):
                    self.graft()
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())

    def test_unknown_tail_bytes_and_malformed_suffixes_are_not_discarded(self):
        end = self.parser.parse_file(self.source)[2][-1].off
        prefix = self.original[end:end + 41]
        suffix = self.original[-144:]
        malformed = [
            bytes(41),  # Beyond the bounded footerless padding allowance.
            bytes(24) + b"x",  # Broken root terminator.
            prefix + bytes(16) + suffix,  # Not an alignment pad.
            prefix + b"x" + suffix,  # Opaque padding.
            prefix + suffix + b"x",  # Opaque data after the final signature.
            prefix + b"x" + suffix[1:],  # Nonzero reserved word.
            prefix + suffix[:4] + struct.pack("<I", 7400) + suffix[8:],
            prefix + suffix[:8] + b"x" + suffix[9:],  # Nonzero reserved block.
            prefix + suffix[:-1] + bytes([suffix[-1] ^ 1]),
        ]
        for index, tail in enumerate(malformed):
            with self.subTest(index=index):
                self.source.write_bytes(self.original[:end] + tail)
                with self.assertRaises(ValueError):
                    self.graft()
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())

    def test_source_aliases_and_output_report_collisions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            mesh_fbx.graft_file(self.source, self.source, [_mesh()], verification_path=self.report, overwrite=True)
        with self.assertRaisesRegex(ValueError, "protected"):
            mesh_fbx.graft_file(self.source, self.output, [_mesh()], verification_path=self.source, overwrite=True)
        with self.assertRaisesRegex(ValueError, "protected"):
            mesh_fbx.graft_file(self.source, self.output, [_mesh()], verification_path=self.output, overwrite=True)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_hard_links_cannot_overwrite_source_or_alias_outputs(self):
        alias = self.root / "alias.fbx"
        try:
            os.link(self.source, alias)
        except OSError as exc:
            self.skipTest(f"Filesystem does not support hard links: {exc}")
        for destination, report in ((alias, self.report), (self.output, alias)):
            with self.assertRaisesRegex(ValueError, "protected"):
                mesh_fbx.graft_file(self.source, destination, [_mesh()], verification_path=report, overwrite=True)
        self.output.write_bytes(b"old output")
        os.link(self.output, self.report)
        with self.assertRaisesRegex(ValueError, "protected"):
            self.graft(overwrite=True)
        self.assertEqual(self.output.read_bytes(), b"old output")
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_existing_outputs_require_explicit_overwrite(self):
        self.output.write_bytes(b"old output")
        with self.assertRaises(FileExistsError):
            self.graft()
        self.assertEqual(self.output.read_bytes(), b"old output")
        self.assertFalse(self.report.exists())
        self.output.unlink()
        self.report.write_bytes(b"old report")
        with self.assertRaises(FileExistsError):
            self.graft()
        self.assertEqual(self.report.read_bytes(), b"old report")
        self.assertFalse(self.output.exists())
        self.output.write_bytes(b"old output")
        self.assertTrue(self.graft(overwrite=True)["passed"])
        self.assertTrue(self.output.read_bytes().startswith(self.parser.MAGIC))
        self.assertTrue(json.loads(self.report.read_text(encoding="utf-8"))["passed"])

    def test_verification_failure_publishes_neither_artifact(self):
        original_verify = mesh_fbx.verify
        def failed_verification(*args, **kwargs):
            result = original_verify(*args, **kwargs)
            result["checks"]["all_meshes_valid"] = False
            result["passed"] = False
            return result
        self.output.write_bytes(b"old output")
        self.report.write_bytes(b"old report")
        with patch.object(mesh_fbx, "verify", side_effect=failed_verification):
            with self.assertRaisesRegex(ValueError, "Integrity verification failed"):
                self.graft(overwrite=True)
        self.assertEqual(self.output.read_bytes(), b"old output")
        self.assertEqual(self.report.read_bytes(), b"old report")
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_new_dangling_connection_is_rejected_before_publication(self):
        original_create = mesh_fbx.create_mesh_objects
        def broken_mesh(*args):
            objects, connections = original_create(*args)
            connections.append(mesh_fbx.node("C", mesh_fbx.string("OO"),
                                              mesh_fbx.integer(8888, "L"), mesh_fbx.integer(10, "L")))
            return objects, connections
        with patch.object(mesh_fbx, "create_mesh_objects", side_effect=broken_mesh):
            with self.assertRaisesRegex(ValueError, "no_new_dangling_references"):
                self.graft()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_atomic_staging_failure_does_not_publish_destination(self):
        original_stage = mesh_fbx.stage_output
        def fail_report(path, content):
            if path == self.report:
                raise OSError("report staging unavailable")
            return original_stage(path, content)
        with patch.object(mesh_fbx, "stage_output", side_effect=fail_report):
            with self.assertRaisesRegex(OSError, "staging unavailable"):
                self.graft()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_no_overwrite_publication_does_not_clobber_racing_writer(self):
        original_publish = mesh_fbx.publish_output
        def race(temporary, path, *, overwrite):
            if path == self.output:
                path.write_bytes(b"racing output")
            return original_publish(temporary, path, overwrite=overwrite)
        with patch.object(mesh_fbx, "publish_output", side_effect=race):
            with self.assertRaises(FileExistsError):
                self.graft()
        self.assertEqual(self.output.read_bytes(), b"racing output")
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_rejects_invalid_geometry_materials_and_parents_without_outputs(self):
        mutations = [
            lambda m: m.update(parent_uid=23),
            lambda m: m.update(parent_uid=999),
            lambda m: m.update(vertices=[[float("nan"), 0., 0.]] * 4),
            lambda m: m.update(triangles=[[0, 1, 10]]),
            lambda m: m.update(triangles=[[0, 1, 1]]),
            lambda m: m.update(triangles=[[0., 1., 2.]]),
            lambda m: m.update(triangle_material_indices=[0]),
            lambda m: m.update(triangle_material_indices=[0, 2]),
            lambda m: m.update(normals=[[0., 0., 2.]] * 4),
            lambda m: m["materials"][0].update(color_space="linear"),
            lambda m: m["materials"][0].update(opacity=float("nan")),
            lambda m: m["materials"][0].update(diffuse_rgb=[1.1, 0., 0.]),
            lambda m: m.update(vertices=[[0., 0., 0.]] * 4, normals=None),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                payload = _mesh(numpy=False)
                mutate(payload)
                with self.assertRaises(ValueError):
                    self.graft([payload])
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())
        payload = _mesh()
        with self.assertRaisesRegex(ValueError, "Duplicate mesh name"):
            self.graft([payload, copy.deepcopy(payload)])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_unsupported_version_bad_offsets_and_opaque_footer_are_explicit(self):
        for corrupt, message in (
            (lambda b: struct.pack_into("<I", b, 23, 7400), "7700"),
            (lambda b: struct.pack_into("<Q", b, 27, 27), "node bounds"),
            (lambda b: b.__setitem__(-1, b[-1] ^ 1), "footer layout"),
        ):
            with self.subTest(message=message):
                data = bytearray(self.original)
                corrupt(data)
                self.source.write_bytes(data)
                with self.assertRaisesRegex(ValueError, message):
                    self.graft()
                self.assertEqual(self.source.read_bytes(), bytes(data))
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())


if __name__ == "__main__":
    unittest.main()

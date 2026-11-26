"""Real binary FBX regressions for opaque textured static grafts; no Blender/SDK."""
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import mesh_fbx as fbx
from mesh_fbx_static import graft_static_assets
from test_mesh_fbx import _write_scene


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=")


def _object(kind, uid, name, subtype="", children=()):
    return fbx.node(kind, fbx.integer(uid, "L"), fbx.string(name + "\x00\x01" + kind),
                    fbx.string(subtype), children=children)


def _link(kind, child, parent, member=None):
    props = [fbx.string(kind), fbx.integer(child, "L"), fbx.integer(parent, "L")]
    if member is not None:
        props.append(fbx.string(member))
    return fbx.node("C", *props)


def _vector(name, values):
    return fbx.property70(name, name, [fbx.real(value) for value in values])


def _emit(record, offset, version):
    """A small independent 7400/7700 fixture writer with real absolute offsets."""
    header = 13 if version == 7400 else 25
    payload = b"".join(record.properties)
    own_size = header + len(record.name) + len(payload)
    children = bytearray()
    for child in record.children:
        children.extend(_emit(child, offset + own_size + len(children), version))
    if record.children or record.terminated:
        children.extend(bytes(header))
    end = offset + own_size + len(children)
    return (struct.pack("<IIIB" if version == 7400 else "<QQQB", end,
                        len(record.properties), len(payload), len(record.name))
            + record.name + payload + children)


def _write_snippet(path, *, version=7400, translation=(0., 0., 0.), extra_object=None,
                   embedded=True, geometric_translation=(0., 0., 0.), model_kind="Mesh",
                   omit_uv=False, extra_link=None):
    geometry = _object("Geometry", 10, "FurnitureGeometry", "Mesh", [
        fbx.node("Vertices", fbx.array(np.array([10., 11., 12., 1., 0., 0., 0., 1., 0.]), "d")),
        fbx.node("PolygonVertexIndex", fbx.array(np.array([0, 1, -3]), "i")),
        fbx.node("LayerElementNormal", fbx.integer(0), children=[
            fbx.node("MappingInformationType", fbx.string("ByPolygonVertex")),
            fbx.node("ReferenceInformationType", fbx.string("Direct")),
            fbx.node("Normals", fbx.array(np.array([0., 0., 1.] * 3), "d")),
        ]),
        fbx.node("LayerElementMaterial", fbx.integer(0), children=[
            fbx.node("MappingInformationType", fbx.string("AllSame")),
            fbx.node("ReferenceInformationType", fbx.string("IndexToDirect")),
            fbx.node("Materials", fbx.array(np.array([0]), "i")),
        ]),
    ])
    if not omit_uv:
        geometry.children.extend([
            fbx.node("LayerElementUV", fbx.integer(0), children=[
                fbx.node("Name", fbx.string("UVMap")),
                fbx.node("MappingInformationType", fbx.string("ByPolygonVertex")),
                fbx.node("ReferenceInformationType", fbx.string("IndexToDirect")),
                fbx.node("UV", fbx.array(np.array([10., 11., 1., 0., 0., 1.]), "d")),
                fbx.node("UVIndex", fbx.array(np.array([0, 1, 2]), "i")),
            ]),
            fbx.node("Layer", fbx.integer(0), children=[fbx.node("LayerElement", children=[
                fbx.node("Type", fbx.string("LayerElementUV")), fbx.node("TypedIndex", fbx.integer(0)),
            ])]),
        ])
    objects = [geometry, _object("Model", 11, "Furniture", model_kind, [
        fbx.node("Properties70", children=[
            _vector("Lcl Translation", translation), _vector("Lcl Rotation", (0., 0., 0.)),
            _vector("Lcl Scaling", (1., 1., 1.)),
            _vector("GeometricTranslation", geometric_translation),
        ]),
    ]), _object("Material", 12, "TexturedMaterial", children=[
        fbx.node("ShadingModel", fbx.string("phong")),
        fbx.node("Properties70", children=[
            fbx.property70("DiffuseColor", "Color", [fbx.real(v) for v in (.5, .4, .3)]),
        ]),
    ]), _object("Texture", 13, "Albedo", children=[
        fbx.node("Type", fbx.string("TextureVideoClip")),
        fbx.node("Media", fbx.string("FurnitureImage")),
        fbx.node("FileName", fbx.string("not-present-on-disk.png")),
        fbx.node("RelativeFilename", fbx.string("not-present-on-disk.png")),
        fbx.node("Properties70", children=[fbx.property70("UVSet", "KString", [fbx.string("UVMap")])]),
    ]), _object("Video", 14, "FurnitureImage", "Clip", [
        fbx.node("Type", fbx.string("Clip")),
        fbx.node("Filename", fbx.string("not-present-on-disk.png")),
        fbx.node("RelativeFilename", fbx.string("not-present-on-disk.png")),
        fbx.node("Content", (b"R" + struct.pack("<I", len(_PNG)) + _PNG) if embedded
                 else fbx.string("not-present-on-disk.png")),
    ])]
    if extra_object is not None:
        objects.append(extra_object)
    links = [_link("OO", 10, 11), _link("OO", 11, 0), _link("OO", 12, 11),
             _link("OP", 13, 12, "DiffuseColor"), _link("OO", 14, 13)]
    if extra_link is not None:
        links.append(extra_link)
    # Full Blender-style top sections are accepted, but never merged into source.
    roots = [fbx.node("FBXHeaderExtension", children=[fbx.node("FBXVersion", fbx.integer(version))]),
             fbx.node("GlobalSettings", children=[fbx.node("Properties70", children=[
                 fbx.property70("UnitScaleFactor", "double", [fbx.real(99.)]),
             ])]),
             fbx.node("Definitions", children=[fbx.node("Count", fbx.integer(len(objects)))]),
             fbx.node("Objects", children=objects), fbx.node("Connections", children=links)]
    buf = bytearray(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", version))
    for record in roots:
        buf.extend(_emit(record, len(buf), version))
    buf.extend(bytes(13 if version == 7400 else 25))
    buf.extend(fbx.FOOTER_MAGIC)
    buf.extend(bytes((-len(buf)) % 16))
    buf.extend(bytes(4) + struct.pack("<I", version) + bytes(120) + fbx.FOOTER_END_MAGIC)
    path.write_bytes(buf)
    return bytes(buf)


def _add_original_mesh(path, *, omit_transforms=False, animated=False, rotation_order=0,
                       textured=False, geometry_kind="Mesh", duplicate_vertices=False,
                       normal_layers=1, normal_kind="d", duplicate_normals=False, omit_normals=False):
    parser = fbx.load_parser()
    buf = path.read_bytes()
    tops = fbx.parse_checked(parser, buf)
    graph = parser.build_graph(buf, tops)
    properties = [_vector("Lcl Scaling", (2., 3., 4.)),
                  fbx.property70("UserData", "KString", [fbx.string("preserve me")])]
    if rotation_order:
        properties.append(fbx.property70("RotationOrder", "enum", [fbx.integer(rotation_order)]))
    if not omit_transforms:
        properties[:0] = [_vector("Lcl Translation", (7., 8., 9.)),
                          _vector("Lcl Rotation", (1., 2., 3.))]
    mesh = _object("Model", 30, "ExistingCupMesh", "Mesh", [
        fbx.node("Properties70", children=properties), fbx.node("Culling", fbx.string("CullingOff")),
    ])
    geometry = _object("Geometry", 31, "ExistingCupGeometry", geometry_kind, [
        fbx.node("Vertices", fbx.array(np.array([0., 0., 0., 1., 0., 0., 0., 1., 0.]), "d")),
        fbx.node("PolygonVertexIndex", fbx.array(np.array([0, 1, -3]), "i")),
        fbx.node("LayerElementNormal", fbx.integer(0), children=[
            fbx.node("MappingInformationType", fbx.string("ByPolygonVertex")),
            fbx.node("ReferenceInformationType", fbx.string("IndexToDirect")),
            fbx.node("Normals", fbx.array(np.array([0., 0., 1.] * 3), normal_kind)),
            fbx.node("NormalsIndex", fbx.array(np.array([0, 1, 2]), "i")),
        ]),
        fbx.node("LayerElementUV", fbx.integer(0), children=[
            fbx.node("MappingInformationType", fbx.string("ByPolygonVertex")),
            fbx.node("ReferenceInformationType", fbx.string("IndexToDirect")),
            fbx.node("UV", fbx.array(np.array([0., 0., 1., 0., 0., 1.]), "d")),
            fbx.node("UVIndex", fbx.array(np.array([0, 1, 2]), "i")),
        ]),
    ])
    normal_layer = next(child for child in geometry.children if child.name == b"LayerElementNormal")
    if duplicate_normals:
        normal_layer.children.append(normal_layer.children[2])
    if omit_normals:
        normal_layer.children = [child for child in normal_layer.children if child.name != b"Normals"]
    index = geometry.children.index(normal_layer)
    geometry.children[index:index + 1] = [normal_layer] * normal_layers
    if duplicate_vertices:
        geometry.children.append(geometry.children[0])
    links = [_link("OO", 30, 10), _link("OO", 31, 30), _link("OO", 24, 30)]
    if animated:
        links.append(_link("OP", 22, 30, "Lcl Rotation"))
    objects = [mesh, geometry]
    if textured:
        objects.extend([
            _object("Texture", 32, "OriginalTexture", children=[
                fbx.node("Type", fbx.string("TextureVideoClip")),
                fbx.node("Media", fbx.string("OriginalImage")),
            ]),
            _object("Video", 33, "OriginalImage", "Clip", [
                fbx.node("Content", b"R" + struct.pack("<I", len(_PNG)) + _PNG),
            ]),
        ])
        links.extend([_link("OP", 32, 24, "DiffuseColor"), _link("OO", 33, 32)])
    extras = {graph.tops["Objects"].nstart: objects, graph.tops["Connections"].nstart: links}
    output = bytearray(buf[:27])
    for original in tops:
        fbx.emit_node(buf, output, original, {}, extras)
    footer = fbx.footer_parts(buf, tops, 7700)
    output.extend(footer.prefix)
    output.extend(bytes((-len(output)) % 16))
    output.extend(footer.suffix)
    path.write_bytes(output)
    return bytes(output)


class StaticFbxTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.fbx"
        self.snippet = self.root / "furniture.fbx"
        self.output = self.root / "output.fbx"
        self.report_path = self.root / "report.json"
        _write_scene(self.source)
        self.original = _add_original_mesh(self.source)
        self.snippet_bytes = _write_snippet(self.snippet)
        self.parser = fbx.load_parser()

    def graft(self, *, assets=None, **kwargs):
        if assets is None:
            assets = [dict(fbx=self.snippet, parent_uid=10, name="DeskMesh", prop="desk")]
        return graft_static_assets(self.source, self.output, assets,
                                   verification_path=self.report_path, **kwargs)

    def graph(self, path):
        buf = path.read_bytes()
        tops = fbx.parse_checked(self.parser, buf)
        return buf, self.parser.build_graph(buf, tops)

    def assert_unpublished(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report_path.exists())

    def test_blender7400_textures_uvs_embedded_images_and_animation_survive(self):
        result = self.graft()
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.snippet.read_bytes(), self.snippet_bytes)
        self.assertEqual(json.loads(self.report_path.read_text(encoding="utf-8")), result)
        before, source_graph = self.graph(self.source)
        after, graph = self.graph(self.output)
        self.assertEqual(struct.unpack_from("<I", after, 23)[0], 7700)
        for uid, obj in source_graph.uid2node.items():
            self.assertEqual(fbx.fingerprint(before, obj), fbx.fingerprint(after, graph.uid2node[uid]))
        self.assertEqual(fbx.footer_parts(before, fbx.parse_checked(self.parser, before), 7700).prefix,
                         fbx.footer_parts(after, fbx.parse_checked(self.parser, after), 7700).prefix)
        asset = result["meshes"][0]
        remap = asset["uid_remap"]
        self.assertTrue(all(uid > 9001 for uid in remap.values()))
        geometry = graph.uid2node[remap["10"]]
        np.testing.assert_array_equal(fbx.array_data(after, fbx.child_named(geometry, "Vertices"), "d"),
                                      [10., 11., 12., 1., 0., 0., 0., 1., 0.])
        uv = fbx.child_named(geometry, "LayerElementUV")
        np.testing.assert_array_equal(fbx.array_data(after, fbx.child_named(uv, "UV"), "d"),
                                      [10., 11., 1., 0., 0., 1.])
        np.testing.assert_array_equal(fbx.array_data(after, fbx.child_named(uv, "UVIndex"), "i"), [0, 1, 2])
        image = graph.uid2node[remap["14"]]
        self.assertEqual(self.parser.decode_props(after, fbx.child_named(image, "Content")), [("R", _PNG)])
        self.assertEqual(asset["embedded_images"][0]["sha256"], hashlib.sha256(_PNG).hexdigest())
        self.assertIn(("OO", remap["14"], remap["13"]), graph.conns)
        self.assertIn(("OP", remap["13"], remap["12"], "DiffuseColor"), graph.conns)
        self.assertIn(("OO", remap["11"], 10), graph.conns)
        self.assertNotIn(("OO", remap["11"], 0), graph.conns)
        self.assertEqual(result["definitions"]["actual_counts"]["Video"], 1)
        self.assertEqual(result["inherited_dangling_reference_count"], 3)

    def test_world_root_textured_architecture_retains_native_vertices_and_source(self):
        result = self.graft(assets=[
            dict(fbx=self.snippet, parent_uid=0, name="RoomArchitecture", prop="room")])
        self.assertTrue(result["passed"])
        asset = result["meshes"][0]
        self.assertEqual(asset["parent_uid"], 0)
        self.assertTrue(asset["checks"]["exact_requested_parent"])
        self.assertNotIn("exact_original_bone_parent", asset["checks"])
        after, graph = self.graph(self.output)
        self.assertNotIn(0, graph.uid2node)
        self.assertIn(("OO", asset["model_uid"], 0), graph.conns)
        geometry = graph.uid2node[asset["geometry_uids"][0]]
        np.testing.assert_array_equal(
            fbx.array_data(after, fbx.child_named(geometry, "Vertices"), "d"),
            [10., 11., 12., 1., 0., 0., 0., 1., 0.])
        self.assertTrue(asset["checks"]["embedded_image_bytes_exact"])
        self.assertTrue(all(item["unchanged"] for item in result["original_objects"]))
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.snippet.read_bytes(), self.snippet_bytes)

    def test_geometry_only_correction_preserves_every_other_payload_and_connection(self):
        _write_scene(self.source)
        original_bytes = _add_original_mesh(self.source, textured=True)
        vertices = np.array([[0., 0., 0.], [10., 0., 0.], [0., 12., 0.]])
        result = self.graft(assets=[], geometry_vertex_overrides={31: vertices})
        self.assertTrue(result["passed"])
        self.assertEqual(result["mesh_count"], 0)
        self.assertEqual(result["new_object_count"], 0)
        before, old = self.graph(self.source)
        after, new = self.graph(self.output)
        self.assertEqual(set(old.uid2node), set(new.uid2node))
        self.assertEqual(old.conns, new.conns)
        for uid, obj in old.uid2node.items():
            if uid != 31:
                self.assertEqual(fbx.fingerprint(before, obj), fbx.fingerprint(after, new.uid2node[uid]))
        old_geometry, new_geometry = old.uid2node[31], new.uid2node[31]
        self.assertEqual([child.name for child in old_geometry.children],
                         [child.name for child in new_geometry.children])
        for child, copied in zip(old_geometry.children, new_geometry.children):
            if child.name != b"Vertices":
                self.assertEqual(fbx.fingerprint(before, child), fbx.fingerprint(after, copied))
        np.testing.assert_array_equal(
            fbx.array_data(after, fbx.child_named(new_geometry, "Vertices"), "d"), vertices.reshape(-1))
        self.assertEqual(self.parser.decode_props(after, fbx.child_named(new.uid2node[33], "Content")),
                         [("R", _PNG)])
        record = result["geometry_vertex_overrides"][0]
        self.assertEqual({key: record[key] for key in (
            "uid", "old_vertex_count", "new_vertex_count", "changed_vertex_count")},
            dict(uid=31, old_vertex_count=3, new_vertex_count=3, changed_vertex_count=2))
        self.assertTrue(record["checks"]["non_overridden_geometry_payloads_preserved"])
        self.assertTrue(record["checks"]["vertices_match_override"])
        item = next(item for item in result["original_objects"] if item["uid"] == 31)
        self.assertFalse(item["unchanged"])
        self.assertTrue(item["intentional_geometry_vertex_override"])
        self.assertTrue(item["matches_expected"])
        self.assertTrue(result["checks"]["original_animation_curves_exact"])
        self.assertTrue(result["checks"]["original_connections_byte_equivalent_and_ordered"])
        self.assertEqual(self.source.read_bytes(), original_bytes)

    def test_normal_only_correction_preserves_vertices_indices_mapping_and_other_objects(self):
        _write_scene(self.source)
        original_bytes = _add_original_mesh(self.source, textured=True)
        normals = np.array([[2., 0., 0.], [0., -3., 0.], [0., 0., 0.]])
        result = self.graft(assets=[], geometry_normal_overrides={31: normals})
        self.assertTrue(result["passed"])
        self.assertEqual(result["mesh_count"], 0)
        self.assertEqual(result["new_object_count"], 0)
        self.assertEqual(result["geometry_vertex_overrides"], [])
        before, old = self.graph(self.source)
        after, new = self.graph(self.output)
        self.assertEqual(set(old.uid2node), set(new.uid2node))
        self.assertEqual(old.conns, new.conns)
        for uid, obj in old.uid2node.items():
            if uid != 31:
                self.assertEqual(fbx.fingerprint(before, obj), fbx.fingerprint(after, new.uid2node[uid]))
        for child, copied in zip(old.uid2node[31].children, new.uid2node[31].children):
            if child.name != b"LayerElementNormal":
                self.assertEqual(fbx.fingerprint(before, child), fbx.fingerprint(after, copied))
            else:
                for member, new_member in zip(child.children, copied.children):
                    if member.name != b"Normals":
                        self.assertEqual(fbx.fingerprint(before, member), fbx.fingerprint(after, new_member))
                np.testing.assert_array_equal(
                    fbx.array_data(after, fbx.child_named(copied, "Normals"), "d"), normals.reshape(-1))
        record = result["geometry_normal_overrides"][0]
        self.assertEqual({key: record[key] for key in (
            "uid", "old_normal_count", "new_normal_count", "changed_normal_count")},
            dict(uid=31, old_normal_count=3, new_normal_count=3, changed_normal_count=3))
        self.assertTrue(record["checks"]["normals_match_override"])
        self.assertTrue(record["checks"]["non_overridden_geometry_payloads_preserved"])
        item = next(item for item in result["original_objects"] if item["uid"] == 31)
        self.assertFalse(item["unchanged"])
        self.assertFalse(item["intentional_geometry_vertex_override"])
        self.assertTrue(item["intentional_geometry_normal_override"])
        self.assertTrue(item["matches_expected"])
        self.assertTrue(result["checks"]["original_animation_curves_exact"])
        self.assertEqual(self.source.read_bytes(), original_bytes)

    def test_geometry_and_pose_overrides_can_accompany_world_root_asset(self):
        result = self.graft(
            assets=[dict(fbx=self.snippet, parent_uid=0, name="Room", prop="room")],
            model_transform_overrides={30: {
                "translation": [1., 2., 3.], "rotation_degrees": [4., 5., 6.]}},
            geometry_vertex_overrides={31: [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]},
            geometry_normal_overrides={31: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]})
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["model_transform_overrides"]), 1)
        self.assertEqual(len(result["geometry_vertex_overrides"]), 1)
        self.assertEqual(len(result["geometry_normal_overrides"]), 1)
        self.assertTrue(result["checks"]["geometry_normal_overrides_valid"])
        self.assertTrue(all(item["matches_expected"] for item in result["original_objects"]))

    def test_malformed_geometry_overrides_do_not_publish(self):
        valid = [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]
        invalid_maps = [
            [], [(31, valid)], "31", {True: valid}, {"31": valid},
            {0: valid}, {999: valid}, {30: valid}, {10: valid}, {23: valid},
            {31: None}, {31: [0.] * 9}, {31: [[0., 0.]] * 3},
            {31: [[0., 0., 0.]] * 2}, {31: [[0., 0., 0.]] * 4},
            {31: [[0., 0., 0.], [1., 2.], [0., 1., 0.]]},
            {31: [[float("nan"), 0., 0.]] * 3},
            {31: [[float("inf"), 0., 0.]] * 3},
            {31: [["1", "2", "3"]] * 3},
            {31: [[True, False, True]] * 3},
            {31: [[1j, 0, 0]] * 3},
        ]
        for keyword in ("geometry_vertex_overrides", "geometry_normal_overrides"):
            for override in invalid_maps:
                with self.subTest(keyword=keyword, override=override), self.assertRaises(ValueError):
                    self.graft(**{keyword: override})
                self.assert_unpublished()
        for overrides in ({}, {"model_transform_overrides": {}},
                          {"geometry_vertex_overrides": {}},
                          {"geometry_normal_overrides": {}},
                          {"model_transform_overrides": {}, "geometry_vertex_overrides": {},
                           "geometry_normal_overrides": {}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.graft(assets=[], **overrides)
            self.assert_unpublished()

    def test_geometry_only_alias_and_existing_output_rejections_preserve_source(self):
        override = {31: [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]}
        for destination, report in ((self.source, self.report_path),
                                    (self.output, self.source), (self.output, self.output)):
            with self.subTest(destination=destination, report=report), self.assertRaises(ValueError):
                graft_static_assets(self.source, destination, [], verification_path=report,
                                    geometry_vertex_overrides=override)
            self.assert_unpublished()
        self.output.write_bytes(b"keep existing output")
        with self.assertRaises(FileExistsError):
            self.graft(assets=[], geometry_vertex_overrides=override)
        self.assertEqual(self.output.read_bytes(), b"keep existing output")
        self.assertFalse(self.report_path.exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_invalid_static_parents_are_not_treated_as_world_root(self):
        for parent in (False, True, "0", -1, 30, 31, 999):
            with self.subTest(parent=parent), self.assertRaises(ValueError):
                self.graft(assets=[dict(fbx=self.snippet, parent_uid=parent, name="Room", prop="room")])
            self.assert_unpublished()

    def test_nonmesh_or_duplicate_original_vertices_cannot_be_overridden(self):
        for options in ({"geometry_kind": "Shape"}, {"duplicate_vertices": True}):
            with self.subTest(options=options):
                _write_scene(self.source)
                _add_original_mesh(self.source, **options)
                with self.assertRaises(ValueError):
                    self.graft(assets=[], geometry_vertex_overrides={
                        31: [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]})
                self.assert_unpublished()

    def test_missing_multiple_or_malformed_original_normal_arrays_are_rejected(self):
        for options in ({"normal_layers": 0}, {"normal_layers": 2}, {"normal_kind": "i"},
                        {"duplicate_normals": True}, {"omit_normals": True}, {"geometry_kind": "Shape"}):
            with self.subTest(options=options):
                _write_scene(self.source)
                _add_original_mesh(self.source, **options)
                with self.assertRaises(ValueError):
                    self.graft(assets=[], geometry_normal_overrides={31: [[1., 0., 0.]] * 3})
                self.assert_unpublished()

    def test_array_whitelist_does_not_hide_unexpected_normal_metadata_or_uv_changes(self):
        before, graph = self.graph(self.source)
        geometry = graph.uid2node[31]
        real_emit = fbx.emit_node
        cases = (
            ("LayerElementNormal", "Normals", "d", False),
            ("LayerElementUV", "UV", "d", False),
            ("LayerElementNormal", "NormalsIndex", "i", True),
            ("LayerElementUV", "UV", "d", True),
        )
        for layer_name, array_name, kind, normal_override in cases:
            with self.subTest(array=array_name, normal_override=normal_override):
                array = fbx.child_named(fbx.child_named(geometry, layer_name), array_name)
                values = fbx.array_data(before, array, kind).copy()
                values[0] += 1
                unexpected = (fbx.array(values, kind), 1)

                def corrupt_nonwhitelisted_payload(buf, output, original, replacements, extras):
                    if original.name == b"Objects":
                        replacements = {**replacements, array.nstart: unexpected}
                    return real_emit(buf, output, original, replacements, extras)

                normal_overrides = {31: [[1., 0., 0.]] * 3} if normal_override else None
                with patch.object(fbx, "emit_node", side_effect=corrupt_nonwhitelisted_payload):
                    with self.assertRaisesRegex(ValueError, "Integrity verification failed"):
                        self.graft(assets=[],
                            geometry_vertex_overrides={31: [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]},
                            geometry_normal_overrides=normal_overrides)
                self.assert_unpublished()

    def test_three_independent_snippets_allocate_unique_uids_and_names(self):
        result = self.graft(assets=[dict(fbx=self.snippet, parent_uid=10, name=name, prop=name)
                                    for name in ("desk", "orangechair", "whitechair")])
        self.assertTrue(result["passed"])
        self.assertEqual(result["new_object_count"], 15)
        self.assertEqual(len({item["model_uid"] for item in result["meshes"]}), 3)
        self.assertEqual(result["definitions"]["declared_counts"]["Video"], 3)

    def test7700_snippet_is_also_supported(self):
        _write_snippet(self.snippet, version=7700)
        self.assertTrue(self.graft()["passed"])

    def test_mesh_pose_override_preserves_scale_mesh_marker_bone_and_curve_payloads(self):
        result = self.graft(model_transform_overrides={30: {
            "translation": [0.1, -0.2, 0.3], "rotation_degrees": [10., 20., -30.]}})
        self.assertTrue(result["passed"])
        before, old = self.graph(self.source)
        after, new = self.graph(self.output)
        for uid, obj in old.uid2node.items():
            if uid != 30:
                self.assertEqual(fbx.fingerprint(before, obj), fbx.fingerprint(after, new.uid2node[uid]))
        props = fbx.properties70(self.parser, after, new.uid2node[30])
        self.assertEqual(props["Lcl Translation"], [.1, -.2, .3])
        self.assertEqual(props["Lcl Rotation"], [10., 20., -30.])
        self.assertEqual(props["Lcl Scaling"], [2., 3., 4.])
        self.assertEqual(props["UserData"], ["preserve me"])
        self.assertEqual(result["model_transform_overrides"][0]["uid"], 30)
        self.assertTrue(result["checks"]["original_animation_curves_exact"])
        item = next(item for item in result["original_objects"] if item["uid"] == 30)
        self.assertFalse(item["unchanged"])
        self.assertTrue(item["matches_expected"])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_pose_only_correction_retains_every_object_and_existing_geometry(self):
        result = graft_static_assets(self.source, self.output, [],
            verification_path=self.report_path, model_transform_overrides={30: {
                "translation": [0., 0., -.7125], "rotation_degrees": [12., 0., 0.]}})
        before, old = self.graph(self.source)
        after, new = self.graph(self.output)
        self.assertEqual(set(old.uid2node), set(new.uid2node))
        self.assertEqual(old.conns, new.conns)
        for uid, obj in old.uid2node.items():
            if uid != 30:
                self.assertEqual(fbx.fingerprint(before, obj), fbx.fingerprint(after, new.uid2node[uid]))
        properties = fbx.properties70(self.parser, after, new.uid2node[30])
        self.assertEqual(properties["Lcl Translation"], [0., 0., -.7125])
        self.assertEqual(properties["Lcl Rotation"], [12., 0., 0.])
        self.assertEqual(properties["Lcl Scaling"], [2., 3., 4.])
        self.assertTrue(result["checks"]["original_animation_curves_exact"])

    def test_empty_assets_without_a_correction_does_not_publish(self):
        with self.assertRaises(ValueError):
            graft_static_assets(self.source, self.output, [], verification_path=self.report_path)
        self.assert_unpublished()

    def test_nonxyz_mesh_rotation_order_is_not_silently_reinterpreted(self):
        _write_scene(self.source)
        _add_original_mesh(self.source, rotation_order=1)
        with self.assertRaisesRegex(ValueError, "XYZ RotationOrder"):
            self.graft(model_transform_overrides={30: {
                "translation": [0, 0, 0], "rotation_degrees": [10, 20, 30]}})
        self.assert_unpublished()

    def test_missing_default_local_pose_properties_can_be_explicitly_set(self):
        _write_scene(self.source)
        _add_original_mesh(self.source, omit_transforms=True)
        self.assertTrue(self.graft(model_transform_overrides={30: {
            "translation": [1, 2, 3], "rotation_degrees": [4, 5, 6]}})["passed"])

    def test_bone_marker_or_animated_mesh_override_is_rejected(self):
        for uid in (10, 11, 31, 999):
            with self.subTest(uid=uid), self.assertRaises(ValueError):
                self.graft(model_transform_overrides={uid: {
                    "translation": [1, 2, 3], "rotation_degrees": [4, 5, 6]}})
            self.assert_unpublished()
        _write_scene(self.source)
        _add_original_mesh(self.source, animated=True)
        with self.assertRaisesRegex(ValueError, "animated curves"):
            self.graft(model_transform_overrides={30: {
                "translation": [1, 2, 3], "rotation_degrees": [4, 5, 6]}})
        self.assert_unpublished()

    def test_unsupported_animation_skin_camera_and_nonidentity_models_rejected(self):
        for kind in ("AnimationCurve", "Deformer", "NodeAttribute", "Pose"):
            with self.subTest(kind=kind):
                _write_snippet(self.snippet, extra_object=_object(kind, 99, "Unsupported"))
                with self.assertRaisesRegex(ValueError, "unsupported object"):
                    self.graft()
                self.assert_unpublished()
        for kwargs in ({"translation": (1., 0., 0.)},
                       {"geometric_translation": (0., 1., 0.)}, {"model_kind": "LimbNode"}):
            with self.subTest(kwargs=kwargs):
                _write_snippet(self.snippet, **kwargs)
                with self.assertRaises(ValueError):
                    self.graft()
                self.assert_unpublished()

    def test_external_image_missing_uv_and_new_dangling_links_rejected(self):
        for kwargs in ({"embedded": False}, {"omit_uv": True},
                       {"extra_link": _link("OO", 999, 11)}):
            with self.subTest(kwargs=kwargs):
                _write_snippet(self.snippet, **kwargs)
                with self.assertRaises(ValueError):
                    self.graft()
                self.assert_unpublished()

    def test_invalid_names_inputs_and_override_values_are_not_published(self):
        for name in ("ExistingCupMesh", "", "bad\x00name"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.graft(assets=[dict(fbx=self.snippet, parent_uid=10, name=name, prop="desk")])
            self.assert_unpublished()
        for override in ({30: {"translation": [float("nan"), 0, 0], "rotation_degrees": [0, 0, 0]}},
                         {30: {"translation": [0, 0, 0], "rotation_degrees": [0, 0, 0], "scale": [1, 1, 1]}},
                         {True: {"translation": [0, 0, 0], "rotation_degrees": [0, 0, 0]}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.graft(model_transform_overrides=override)
            self.assert_unpublished()

    def test_no_overwrite_or_source_snippet_report_aliases(self):
        for destination, report in ((self.source, self.report_path), (self.snippet, self.report_path),
                                    (self.output, self.source), (self.output, self.snippet),
                                    (self.output, self.output)):
            with self.subTest(destination=destination, report=report), self.assertRaises(ValueError):
                graft_static_assets(self.source, destination, [dict(
                    fbx=self.snippet, parent_uid=10, name="desk", prop="desk")], verification_path=report)
        self.output.write_bytes(b"existing output")
        with self.assertRaises(FileExistsError):
            self.graft()
        self.assertEqual(self.output.read_bytes(), b"existing output")
        self.output.unlink()
        self.report_path.write_bytes(b"existing report")
        with self.assertRaises(FileExistsError):
            self.graft()
        self.assertEqual(self.report_path.read_bytes(), b"existing report")
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_hardlink_input_alias_rejected_when_supported(self):
        alias = self.root / "alias.fbx"
        try:
            os.link(self.source, alias)
        except OSError as error:
            self.skipTest(f"Hard links unavailable: {error}")
        with self.assertRaises(ValueError):
            graft_static_assets(self.source, alias, [dict(
                fbx=self.snippet, parent_uid=10, name="desk", prop="desk")], verification_path=self.report_path)

    def test_failed_staging_cleans_temporary_files_without_publication(self):
        real_stage = fbx.stage_output
        calls = []

        def fail_second(path, content):
            if calls:
                raise OSError("report staging failure")
            temporary = real_stage(path, content)
            calls.append(temporary)
            return temporary

        with patch.object(fbx, "stage_output", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "staging failure"):
                self.graft()
        self.assert_unpublished()
        self.assertTrue(all(not path.exists() for path in calls))


if __name__ == "__main__":
    unittest.main()

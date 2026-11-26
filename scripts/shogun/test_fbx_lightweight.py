"""Real binary7700 lossless trimming regressions, without mocked verification."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import fbx_lightweight as light
import mesh_fbx as fbx
from test_mesh_fbx_static import _emit, _link, _object, _vector


TICKS = (385269534, 770539069, 1155808601, 1541078139, 1926347670)


def _compressed(code, fmt, values):
    raw = struct.pack("<" + fmt * len(values), *values)
    payload = zlib.compress(raw, 9)
    return code.encode("ascii") + struct.pack("<III", len(values), 1, len(payload)) + payload


def _curve(uid):
    return _object("AnimationCurve", uid, "curve", children=[
        fbx.node("Default", fbx.real(-123.456789012345)),
        fbx.node("KeyVer", fbx.integer(4009)),
        fbx.node("KeyTime", _compressed("l", "q", TICKS)),
        fbx.node("KeyValueFloat", _compressed("f", "f", (0., 1.25, -7., 3.5, 4.))),
        fbx.node("KeyAttrFlags", _compressed("i", "i", (24836, 8456))),
        fbx.node("KeyAttrDataFloat", _compressed("f", "f", (.125, -.25, 17., 42., 3., 4., 5., 6.))),
        # Preserve even stale source metadata: trimming must not repair tangents.
        fbx.node("KeyAttrRefCount", _compressed("i", "i", (2, 1))),
        fbx.node("OpaqueCurveData", b"R" + struct.pack("<I", 8) + b"\x00opaque\xff"),
    ])


def _write_source(path, *, shared=False, malformed=None, labeling=False):
    def model(uid, name, kind):
        return _object("Model", uid, name, kind, [
            fbx.node("Properties70", children=[
                _vector("Lcl Translation", (0., 0., 0.)),
                _vector("Lcl Rotation", (0., 0., 0.)),
                _vector("Lcl Scaling", (1., 1., 1.)),
            ]),
        ])

    objects = [
        model(10, "human", "Null"), model(11, "human:root", "LimbNode"),
        model(12, "System:room", "Mesh"), model(13, "Unlabeled_Markers", "Null"),
        model(14, "Unlabeled_Markers:_1", "Marker"),
        _object("NodeAttribute", 15, "unlabeled", "Marker"),
        _object("NodeAttribute", 16, "shared", "Marker"),
        _object("Geometry", 17, "roomGeometry", "Mesh", [
            fbx.node("Vertices", _compressed("d", "d", (0., 0., 0., 1., 0., 0., 0., 1., 0.))),
            fbx.node("PolygonVertexIndex", _compressed("i", "i", (0, 1, -3))),
        ]),
        _object("Material", 18, "material", children=[fbx.node("Opaque", fbx.string("keep material"))]),
        _object("Texture", 19, "texture", children=[fbx.node("RelativeFilename", fbx.string("textures/asset.png"))]),
        _object("Video", 20, "video", children=[fbx.node("Content", b"R" + struct.pack("<I", 8) + b"\x89PNGtest")]),
        _object("AnimationStack", 21, "original clip", children=[fbx.node("Properties70", children=[
            fbx.property70("LocalStart", "KTime", [fbx.integer(TICKS[0], "L")]),
            fbx.property70("LocalStop", "KTime", [fbx.integer(TICKS[-1], "L")]),
        ])]),
        _object("AnimationLayer", 22, "BaseLayer"),
        _object("AnimationCurveNode", 23, "HumanT", children=[fbx.node("Properties70", children=[
            fbx.property70("d|X", "Number", [fbx.real(987.6543210123)])])]),
        _curve(24),
        _object("AnimationCurveNode", 25, "UnlabeledT"), _curve(26),
        _object("AnimationCurveNode", 27, "MaterialOpacity"), _curve(28),
        _object("Deformer", 29, "opaque deformer", "Cluster", [fbx.node("Opaque", fbx.string("untouched"))]),
        model(30, "human:LASI", "Marker"), model(31, "camera_1", "Null"),
        model(32, "camera_1:M_1", "Marker"), model(33, "System", "Null"),
        model(34, "System:OpticalCamera", "Camera"),
        _object("NodeAttribute", 35, "optical", "Camera"),
        _object("AnimationCurveNode", 36, "CameraT"), _curve(37),
        _object("AnimationCurveNode", 38, "PropT"), _curve(39),
        model(40, "camera_1:root", "LimbNode"), model(41, "camera_1:body", "Mesh"),
        model(42, "Unlabeled_Markers:group", "Null"),
        model(43, "Unlabeled_Markers:group:_2", "Marker"),
        _object("AnimationCurveNode", 44, "NestedUnlabeledT"), _curve(45),
    ]
    links = [
        _link("OO", 10, 0), _link("OO", 11, 10), _link("OO", 30, 10),
        _link("OO", 33, 0), _link("OO", 12, 33), _link("OO", 34, 33),
        _link("OO", 35, 34), _link("OO", 31, 0), _link("OO", 32, 31),
        _link("OO", 40, 31), _link("OO", 41, 56 if labeling else 40),
        _link("OO", 13, 0), _link("OO", 14, 13), _link("OO", 42, 13), _link("OO", 43, 42),
        _link("OO", 15, 14), _link("OO", 16, 14), _link("OO", 16, 30),
        _link("OO", 17, 12), _link("OO", 18, 12), _link("OP", 19, 18, "DiffuseColor"),
        _link("OO", 20, 19), _link("OO", 22, 21),
        _link("OO", 23, 22), _link("OP", 23, 30, "Lcl Translation"), _link("OP", 24, 23, "d|X"),
        _link("OO", 25, 22), _link("OP", 25, 14, "Lcl Translation"), _link("OP", 26, 25, "d|X"),
        _link("OO", 27, 22), _link("OP", 27, 18, "TransparencyFactor"), _link("OP", 28, 27, "d|X"),
        _link("OO", 36, 22), _link("OP", 36, 34, "Lcl Translation"), _link("OP", 37, 36, "d|X"),
        _link("OO", 38, 22), _link("OP", 38, 32, "Lcl Translation"), _link("OP", 39, 38, "d|X"),
        _link("OO", 44, 22), _link("OP", 44, 43, "Lcl Translation"), _link("OP", 45, 44, "d|X"),
        _link("OO", 123456, 10),
    ]
    if shared:
        links += [_link("OP", 23, 14, "Lcl Translation"),
                  _link("OP", 24, 25, "d|Y"), _link("OP", 23, 18, "Opacity")]
    if malformed is not None:
        objects.append(model(90, "must_not_disappear", malformed))
        links.append(_link("OO", 90, 42))
    if labeling:
        objects += [
            model(50, "human:Labeling", "Null"),
            model(51, "human:Labeling:Position", "Null"),
            model(52, "human:Labeling:Hips", "LimbNode"),
            model(53, "human:Labeling:Hand", "LimbNode"),
            model(54, "human:Solving:Hips", "LimbNode"),
            model(55, "camera_1:Labeling", "Null"),
            model(56, "camera_1:Labeling:root", "LimbNode"),
            model(57, "camera_1:Solving:root", "LimbNode"),
            model(58, "human:Labeling_notes:root", "LimbNode"),
            _object("AnimationCurveNode", 60, "LabelHipsT"), _curve(61),
            _object("AnimationCurveNode", 62, "LabelHandT"), _curve(63),
            _object("AnimationCurveNode", 64, "SolveHipsT"), _curve(65),
            _object("AnimationCurveNode", 66, "MeshAnchorT"), _curve(67),
            model(68, "camera_1:Labeling:unused", "LimbNode"),
            _object("AnimationCurveNode", 69, "UnusedLabelT"), _curve(70),
        ]
        links += [
            _link("OO", 50, 10), _link("OO", 51, 50), _link("OO", 52, 51),
            _link("OO", 53, 52), _link("OO", 54, 10), _link("OO", 58, 10),
            _link("OO", 55, 31), _link("OO", 56, 55), _link("OO", 57, 31),
            _link("OO", 68, 55),
            _link("OO", 60, 22), _link("OP", 60, 52, "Lcl Translation"), _link("OP", 61, 60, "d|X"),
            _link("OO", 62, 22), _link("OP", 62, 53, "Lcl Translation"), _link("OP", 63, 62, "d|X"),
            _link("OO", 64, 22), _link("OP", 64, 54, "Lcl Translation"), _link("OP", 65, 64, "d|X"),
            _link("OO", 66, 22), _link("OP", 66, 56, "Lcl Translation"), _link("OP", 67, 66, "d|X"),
            _link("OO", 69, 22), _link("OP", 69, 68, "Lcl Translation"), _link("OP", 70, 69, "d|X"),
            # A shared channel still drives a retained physical-camera Marker.
            _link("OP", 38, 52, "Lcl Translation"),
        ]
    roots = [
        fbx.node("GlobalSettings", children=[fbx.node("Properties70", children=[
            fbx.property70("UnitScaleFactor", "double", [fbx.real(.1)]),
            fbx.property70("TimeMode", "enum", [fbx.integer(18)]),
            fbx.property70("CustomFrameRate", "double", [fbx.real(119.88011988011988)]),
        ])]),
        fbx.node("Definitions", children=[fbx.node("Count", fbx.integer(999)),
            fbx.node("ObjectType", fbx.string("Model"), children=[fbx.node("Count", fbx.integer(999)),
                fbx.node("PropertyTemplate", fbx.string("FbxNode"), children=[fbx.node("Opaque", fbx.string("keep template"))])]),
            fbx.node("ObjectType", fbx.string("UnusedType"), children=[fbx.node("Count", fbx.integer(44))]),
        ]),
        fbx.node("Objects", children=objects), fbx.node("Connections", children=links),
        fbx.node("Takes", children=[fbx.node("Current", fbx.string("original clip"))]),
    ]
    out = bytearray(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7700))
    for root in roots:
        out.extend(_emit(root, len(out), 7700))
    out.extend(fbx.NULL_RECORD + b"unique-file-id!!")
    out.extend(bytes((-len(out)) % 16))
    out.extend(bytes(4) + struct.pack("<I", 7700) + bytes(120) + fbx.FOOTER_END_MAGIC)
    path.write_bytes(out)
    return bytes(out)


class LightweightFbxTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "full.fbx"
        self.output = self.root / "display.fbx"
        self.report = self.root / "report.json"
        self.parser = fbx.load_parser()

    def read(self, path):
        buf = path.read_bytes()
        tops = fbx.parse_checked(self.parser, buf)
        top = {node.name.decode("ascii"): node for node in tops}
        objects = {self.values(buf, node)[0]: node for node in top["Objects"].children}
        links = [self.values(buf, node) for node in top["Connections"].children]
        return buf, tops, top, objects, links

    def values(self, buf, node):
        return tuple(value for _, value in self.parser.decode_props(buf, node))

    def payload(self, buf, node):
        """Compare encoded properties recursively, independently of relocated offsets."""
        return (node.name, bytes(buf[node.cstart - node.proplen:node.cstart]),
                tuple(self.payload(buf, child) for child in node.children),
                fbx.tail_bytes(buf, node))

    def test_named_marker_motion_and_scene_are_lossless(self):
        original = _write_source(self.source)
        source, source_tops, source_top, before, original_links = self.read(self.source)
        report = light.lighten_file(self.source, self.output, report_path=self.report)
        out, tops, top, objects, links = self.read(self.output)
        removed = {13, 14, 15, 25, 26, 34, 35, 36, 37, 42, 43, 44, 45}
        self.assertEqual(set(objects), set(before) - removed)
        self.assertEqual(links, [link for link in original_links
                                if link[1] in objects and (link[2] == 0 or link[2] in objects)])
        for uid, node in objects.items():
            with self.subTest(uid=uid):
                self.assertEqual(self.payload(source, before[uid]), self.payload(out, node))
        for uid in (24, 28, 39):
            self.assertEqual(fbx.array_data(out, fbx.child_named(objects[uid], "KeyTime"), "l").tolist(), list(TICKS))
            self.assertEqual(fbx.array_data(out, fbx.child_named(objects[uid], "KeyAttrFlags"), "i").tolist(), [24836, 8456])
            self.assertEqual(fbx.array_data(out, fbx.child_named(objects[uid], "KeyAttrRefCount"), "i").tolist(), [2, 1])
        for name in ("GlobalSettings", "Takes"):
            self.assertEqual(self.payload(source, source_top[name]), self.payload(out, top[name]))
        definitions = top["Definitions"]
        self.assertEqual(self.values(out, fbx.child_named(definitions, "Count")), (len(objects) + 1,))
        counts = {}
        for node in objects.values():
            name = node.name.decode("ascii")
            counts[name] = counts.get(name, 0) + 1
        counts["GlobalSettings"] = 1
        actual_counts = {}
        for node in definitions.children:
            if node.name == b"ObjectType":
                name = self.values(out, node)[0]
                actual_counts[name] = self.values(out, fbx.child_named(node, "Count"))[0]
                if name == "Model":
                    old_type = next(item for item in source_top["Definitions"].children
                                    if item.name == b"ObjectType" and self.values(source, item) == ("Model",))
                    self.assertEqual(self.payload(source, fbx.child_named(old_type, "PropertyTemplate")),
                                     self.payload(out, fbx.child_named(node, "PropertyTemplate")))
        self.assertEqual({name: count for name, count in actual_counts.items() if count}, counts)
        self.assertEqual(fbx.footer_parts(source, source_tops, 7700).prefix,
                         fbx.footer_parts(out, tops, 7700).prefix)
        self.assertEqual(source[-144:], out[-144:])
        self.assertTrue(report["passed"])
        self.assertTrue(report["retained_animation_payloads_exact"])
        self.assertFalse(report["physical_motion_repaired"])
        self.assertEqual(report["source_quality_status"], "NEEDS_REVIEW")
        self.assertEqual(report["output_quality_status"], "NEEDS_REVIEW")
        self.assertEqual(report["original_model_counts"]["Marker"], 4)
        self.assertEqual(report["retained_model_counts"]["Marker"], 2)
        self.assertEqual(report["retained_marker_counts_by_subject"], {"human": 1, "camera_1": 1})
        self.assertEqual({item["uid"] for item in report["removed_models"]}, {13, 14, 34, 42, 43})
        self.assertEqual(report["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(report["output_sha256"], hashlib.sha256(out).hexdigest())
        self.assertEqual(json.loads(self.report.read_text(encoding="utf-8")), report)
        self.assertEqual(self.source.read_bytes(), original)

    def test_shared_curve_and_node_attribute_keep_only_live_branch_links(self):
        _write_source(self.source, shared=True)
        source, _, _, before, original_links = self.read(self.source)
        light.lighten_file(self.source, self.output, report_path=self.report)
        out, _, _, objects, links = self.read(self.output)
        for uid in (16, 23, 24):
            self.assertEqual(self.payload(source, before[uid]), self.payload(out, objects[uid]))
        for uid in (15, 25, 26):
            self.assertNotIn(uid, objects)
        self.assertIn(("OO", 16, 30), links)
        self.assertIn(("OP", 24, 23, "d|X"), links)
        self.assertIn(("OP", 23, 18, "Opacity"), links)
        self.assertEqual(links, [link for link in original_links
                                if link[1] in objects and (link[2] == 0 or link[2] in objects)])

    def test_drop_unused_labeling_keeps_mesh_anchors_and_shared_marker_motion(self):
        original = _write_source(self.source, labeling=True)
        source, _, _, before, original_links = self.read(self.source)
        report = light.lighten_file(self.source, self.output, report_path=self.report,
                                    drop_unused_labeling=True)
        out, _, _, objects, links = self.read(self.output)
        removed = {13, 14, 15, 25, 26, 34, 35, 36, 37, 42, 43, 44, 45,
                   50, 51, 52, 53, 60, 61, 62, 63, 68, 69, 70}
        self.assertEqual(set(objects), set(before) - removed)
        for uid, node in objects.items():
            self.assertEqual(self.payload(source, before[uid]), self.payload(out, node))
        self.assertEqual(links, [link for link in original_links
                                if link[1] in objects and (link[2] == 0 or link[2] in objects)])
        self.assertIn(("OO", 41, 56), links)
        self.assertIn(("OP", 38, 32, "Lcl Translation"), links)
        self.assertNotIn(("OP", 38, 52, "Lcl Translation"), links)
        self.assertEqual({item["uid"] for item in report["retained_labeling_anchors"]}, {55, 56})
        self.assertEqual(report["retained_marker_counts_by_subject"], {"human": 1, "camera_1": 1})
        self.assertEqual(self.source.read_bytes(), original)

    def test_labeling_removal_is_explicit_opt_in(self):
        _write_source(self.source, labeling=True)
        light.lighten_file(self.source, self.output, report_path=self.report)
        out, _, _, objects, _ = self.read(self.output)
        self.assertEqual({self.values(out, objects[uid])[1].split("\x00")[0]
                          for uid in (50, 52, 54, 56)},
                         {"human:Labeling", "human:Labeling:Hips",
                          "human:Solving:Hips", "camera_1:Labeling:root"})

    def test_deleted_parent_with_retained_mesh_or_bone_fails_atomically(self):
        for kind in ("Mesh", "LimbNode"):
            with self.subTest(kind=kind):
                original = _write_source(self.source, malformed=kind)
                with self.assertRaises(ValueError):
                    light.lighten_file(self.source, self.output, report_path=self.report)
                self.assertEqual(self.source.read_bytes(), original)
                self.assertEqual({path.name for path in self.root.iterdir()}, {self.source.name})

    def test_alias_and_existing_paths_are_never_overwritten(self):
        original = _write_source(self.source)
        for output, report in ((self.source, self.report), (self.output, self.source),
                               (self.output, self.output)):
            with self.subTest(output=output, report=report), self.assertRaises(ValueError):
                light.lighten_file(self.source, output, report_path=report)
        alias = self.root / "source-hardlink.fbx"
        try:
            os.link(self.source, alias)
        except OSError:
            pass
        else:
            for output, report in ((alias, self.report), (self.output, alias)):
                with self.subTest(output=output, report=report), self.assertRaises(ValueError):
                    light.lighten_file(self.source, output, report_path=report)
            self.assertEqual(alias.read_bytes(), original)
            alias.unlink()
        for target, sentinel in ((self.output, b"user-owned output"), (self.report, b"user-owned report")):
            with self.subTest(target=target):
                target.write_bytes(sentinel)
                with self.assertRaises(FileExistsError):
                    light.lighten_file(self.source, self.output, report_path=self.report)
                self.assertEqual(target.read_bytes(), sentinel)
                self.assertFalse((self.report if target == self.output else self.output).exists())
                target.unlink()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual({path.name for path in self.root.iterdir()}, {self.source.name})


if __name__ == "__main__":
    unittest.main()

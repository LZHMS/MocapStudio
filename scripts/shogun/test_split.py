"""Subject split regression tests using small real binary FBX documents."""
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zlib

import pipeline_split
from pipeline_split import fbx, split_subject_fbx


def _string(value):
    raw = value.encode("utf-8")
    return b"S" + struct.pack("<I", len(raw)) + raw


def _integer(value):
    return b"L" + struct.pack("<q", value)


def _array(kind, values):
    formats = {"l": "q", "f": "f", "d": "d", "i": "i"}
    raw = struct.pack("<" + formats[kind] * len(values), *values)
    compressed = zlib.compress(raw)
    return kind.encode() + struct.pack("<III", len(values), 1, len(compressed)) + compressed


def _node(name, properties=(), children=()):
    return name.encode(), list(properties), list(children)


def _emit(node, offset):
    name, properties, children = node
    payload = b"".join(properties)
    size = 25 + len(name) + len(payload)
    encoded_children = bytearray()
    for child in children:
        encoded_children.extend(_emit(child, offset + size + len(encoded_children)))
    # FBX >=7500 uses 25-byte null node records. A leaf needs no sentinel.
    if children:
        encoded_children.extend(bytes(25))
    end = offset + size + len(encoded_children)
    return (struct.pack("<QQQB", end, len(properties), len(payload), len(name))
            + name + payload + encoded_children)


def _write_scene(path, humans=("Alice", "Bob"), props=("cup", "table"), version=7700):
    objects = []
    connections = []
    records = {}
    next_uid = 100

    def object_node(kind, name, subtype, children=()):
        nonlocal next_uid
        next_uid += 1
        uid = next_uid
        objects.append(_node(kind, [_integer(uid), _string(name + "\x00\x01" + kind),
                                    _string(subtype)], children))
        return uid

    def connect(kind, src, dst, member=None):
        properties = [_string(kind), _integer(src), _integer(dst)]
        if member is not None:
            properties.append(_string(member))
        connections.append(_node("C", properties))

    stack = object_node("AnimationStack", "Take 001", "", [
        _node("Properties70", children=[
            _node("P", [_string("LocalStart"), _string("KTime"), _string("Time"),
                        _string(""), _integer(123456789)]),
            _node("P", [_string("LocalStop"), _string("KTime"), _string("Time"),
                        _string(""), _integer(234567890)])])])
    layer = object_node("AnimationLayer", "BaseLayer", "")
    connect("OO", layer, stack)
    for index, name in enumerate((*humans, *props, "System", "Unlabeled_Markers")):
        root = object_node("Model", name, "Null")
        connect("OO", root, 0)
        child = object_node("Model", name + "_Hips" if name in humans else name + "_Mesh",
                            "LimbNode" if name in humans else "Mesh")
        connect("OO", child, root)
        geometry = object_node("Geometry", name + "_Geometry", "Mesh", [
            _node("Vertices", [_array("d", [index + .125, -2.5, 3.75])]),
            _node("PolygonVertexIndex", [_array("i", [0, 0, -1])])])
        connect("OO", geometry, child)
        curves = []
        # More than verify's ten sampled curves in a human output.
        for channel in range(12):
            acn = object_node("AnimationCurveNode", name + f"_T{channel}", "")
            curve = object_node("AnimationCurve", name + f"_Curve{channel}", "", [
                _node("KeyTime", [_array("l", [123456789, 170000000, 234567890])]),
                _node("KeyValueFloat", [_array("f", [index + .25, channel + 4.5, -7.0])]),
                _node("KeyAttrFlags", [_array("i", [24836, 24836, 24836])])])
            connect("OO", acn, layer)
            connect("OP", acn, child, "Lcl Translation")
            connect("OP", curve, acn, "d|X")
            curves.append(curve)
        records[name] = {"root": root, "child": child,
                         "geometry": geometry, "curves": curves}
    nodes = [
        _node("FBXHeaderExtension", children=[_node("FBXVersion", [_integer(version)])]),
        _node("GlobalSettings", children=[_node("Properties70", children=[
            _node("P", [_string("TimeMode"), _string("enum"), _string(""),
                        _string(""), _integer(6)]),
            _node("P", [_string("UpAxis"), _string("int"), _string("Integer"),
                        _string(""), _integer(1)])])]),
        _node("Objects", children=objects),
        _node("Connections", children=connections),
        _node("Takes", children=[_node("Current", [_string("Take 001")])]),
    ]
    data = bytearray(fbx.MAGIC + struct.pack("<I", version))
    for node in nodes:
        data.extend(_emit(node, len(data)))
    data.extend(bytes(25))
    path.write_bytes(data)
    return records


def _read_scene(path):
    buf, version, tops = fbx.parse_file(path)
    return buf, version, tops, fbx.build_graph(buf, tops)


def _raw_tree(buf, node):
    return (buf[node.nstart + 8:node.cstart],
            tuple(_raw_tree(buf, child) for child in node.children))


class SubjectSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "fullscene.fbx"
        self.records = _write_scene(self.source)

    def split(self, humans=None, props=None, include_props=True, output_dir=None):
        return split_subject_fbx(self.source, humans if humans is not None else ["Alice", "Bob"],
                                 props if props is not None else ["cup", "table"],
                                 output_dir or self.root / "subjects", include_props)

    def assert_payloads_and_animation(self, entry):
        out = self.root / entry["relative_path"]
        sbuf, version, st, source_graph = _read_scene(self.source)
        obuf, out_version, ot, output_graph = _read_scene(out)
        self.assertEqual(version, out_version)
        self.assertEqual(set(output_graph.roots), set(entry["members"]))
        expected_curves = set()
        for member in entry["members"]:
            expected_curves.update(self.records[member]["curves"])
            geometry = self.records[member]["geometry"]
            self.assertEqual(_raw_tree(sbuf, source_graph.uid2node[geometry]),
                             _raw_tree(obuf, output_graph.uid2node[geometry]))
        self.assertEqual(output_graph.ac_uids, expected_curves)
        for curve in expected_curves:
            self.assertEqual(fbx.curve_values(sbuf, source_graph, curve),
                             fbx.curve_values(obuf, output_graph, curve))
            self.assertEqual(_raw_tree(sbuf, source_graph.uid2node[curve]),
                             _raw_tree(obuf, output_graph.uid2node[curve]))
        self.assertEqual([_raw_tree(sbuf, node) for node in st
                          if node.name not in (b"Objects", b"Connections")],
                         [_raw_tree(obuf, node) for node in ot
                          if node.name not in (b"Objects", b"Connections")])
        self.assertTrue(fbx.verify(self.source, out, verbose=False))
        self.assertTrue(entry["verification"]["passed"])
        self.assertEqual(entry["verification"]["dangling_links"], 0)

    def test_two_humans_each_keep_all_shared_prop_context_and_animation(self):
        original = self.source.read_bytes()
        entries = self.split()
        self.assertEqual([entry["subject"] for entry in entries], ["Alice", "Bob"])
        for entry in entries:
            self.assertEqual(entry["members"], [entry["subject"], "cup", "table"])
            self.assert_payloads_and_animation(entry)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(len(list((self.root / "subjects").rglob("*.fbx"))), 2)
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_mp_excludes_every_prop_and_other_human(self):
        for entry in self.split(include_props=False):
            self.assertEqual(entry["members"], [entry["subject"]])
            self.assert_payloads_and_animation(entry)
            graph = _read_scene(self.root / entry["relative_path"])[3]
            for excluded in set(self.records) - {entry["subject"]}:
                self.assertNotIn(self.records[excluded]["root"], graph.uid2node)
                self.assertNotIn(self.records[excluded]["geometry"], graph.uid2node)
                self.assertTrue(set(self.records[excluded]["curves"]).isdisjoint(graph.ac_uids))

    def test_exact_names_not_nearest_match_and_missing_props_fail_before_writing(self):
        for humans, props in [(["Alic"], ["cup"]), (["alice"], ["cup"]),
                              (["Alice"], ["missing-cup"]),
                              (["Alice", "missing-person"], ["cup"])]:
            with self.subTest(humans=humans, props=props):
                with self.assertRaisesRegex(ValueError, "not found"):
                    self.split(humans=humans, props=props)
                self.assertFalse((self.root / "subjects").exists())
        with self.assertRaisesRegex(ValueError, "not found"):
            self.split(props=["missing"], include_props=False)

    def test_safe_deterministic_folders_for_case_slashes_unicode_and_windows_names(self):
        names = ["A", "a", "A/B", "A\\B", "CON", "NUL.", "trailing. ",
                 "\u4eba\u7269", "..", "%2F", "x" * 400]
        self.records = _write_scene(self.source, humans=names, props=())
        entries = self.split(humans=names, props=[], include_props=False)
        folders = [Path(entry["relative_path"]).parent.name for entry in entries]
        self.assertEqual(len(set(folder.casefold() for folder in folders)), len(names))
        for entry, folder in zip(entries, folders):
            self.assertEqual(Path(entry["relative_path"]).parts, ("subjects", folder, "sequence.fbx"))
            self.assertLess(len(folder), 256)
            self.assertFalse(any(char in folder for char in '/\\:*?"<>|'))
            self.assertFalse(folder.endswith((".", " ")))
            self.assertEqual(folder, pipeline_split._subject_folder(entry["subject"]))
        alternate = self.root / "alternate"
        single = self.split(humans=["A"], props=[], include_props=False, output_dir=alternate)
        self.assertEqual(Path(single[0]["relative_path"]).parent.name, folders[0])

    def test_existing_outputs_are_never_overwritten(self):
        entries = self.split()
        target = self.root / entries[0]["relative_path"]
        target.write_bytes(b"preexisting user data")
        with self.assertRaises(FileExistsError):
            self.split()
        self.assertEqual(target.read_bytes(), b"preexisting user data")

    def test_case_colliding_existing_path_is_not_reused(self):
        folder = pipeline_split._subject_folder("Alice")
        existing = self.root / "subjects" / folder.swapcase()
        existing.mkdir(parents=True)
        target = existing / "sequence.fbx"
        target.write_bytes(b"user data")
        with self.assertRaises(FileExistsError):
            self.split(humans=["Alice"])
        self.assertEqual(target.read_bytes(), b"user data")

    def test_source_filename_and_staging_boundaries_are_protected(self):
        self.source = self.root / "sequence.fbx"
        _write_scene(self.source)
        original = self.source.read_bytes()
        for destination in [self.root, self.root.parent / (self.root.name + "-sibling"),
                            self.root / ".." / (self.root.name + "-sibling")]:
            with self.subTest(destination=destination):
                with self.assertRaises(ValueError):
                    self.split(output_dir=destination)
        self.assertEqual(self.source.read_bytes(), original)
        self.assert_payloads_and_animation(self.split()[0])
        self.assertEqual(self.source.read_bytes(), original)

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.root.parent) as real_outside:
            link = self.root / "linked-subjects"
            try:
                link.symlink_to(real_outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(ValueError):
                self.split(output_dir=link)

    def test_forbidden_duplicate_and_overlapping_rosters_fail(self):
        for humans, props in [(["System"], []), (["Alice"], ["Unlabeled_Markers"]),
                              (["Alice", "Alice"], []), (["Alice"], ["cup", "cup"]),
                              (["Alice"], ["Alice"]), ([], [])]:
            with self.subTest(humans=humans, props=props):
                with self.assertRaises(ValueError):
                    self.split(humans=humans, props=props)

    def test_7500_supported_and_old_binary_version_rejected(self):
        self.records = _write_scene(self.source, version=7500)
        self.assert_payloads_and_animation(self.split(humans=["Alice"])[0])
        _write_scene(self.source, version=7400)
        with self.assertRaisesRegex(ValueError, ">= 7500"):
            self.split(output_dir=self.root / "old")

    def test_false_verification_is_fatal_and_removes_created_outputs(self):
        with patch.object(fbx, "verify", side_effect=[True, False]):
            with self.assertRaisesRegex(ValueError, "verification failed"):
                self.split()
        self.assertFalse(list((self.root / "subjects").rglob("*.fbx")))

    def test_nested_geometry_corruption_is_fatal_even_if_existing_verify_passes(self):
        original_splice = fbx.splice

        def corrupt(buf, tops, keep):
            data, objects, connections = original_splice(buf, tops, keep)
            parsed = fbx.parse_tree(data, 27, len(data))
            graph = fbx.build_graph(data, parsed)
            geometry = graph.uid2node[self.records["Alice"]["geometry"]]
            vertices = geometry.children[0]
            damaged = bytearray(data)
            # Corrupt nested array content, not its parent object header. The
            # splitter must check this even beyond existing verify's scope.
            damaged[vertices.cstart - 1] ^= 1
            return bytes(damaged), objects, connections

        with patch.object(fbx, "splice", side_effect=corrupt), \
                patch.object(fbx, "verify", return_value=True):
            with self.assertRaisesRegex(ValueError, "nested object payload"):
                self.split(humans=["Alice"])
        self.assertFalse(list((self.root / "subjects").rglob("*.fbx")))


if __name__ == "__main__":
    unittest.main()

"""Roster-analysis regressions for sequence_inspect: synthetic FBX/MCP fixtures
plus report compaction. No real datasets, no Shogun."""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import mesh_fbx as fbx
import sequence_inspect as si
from test_mesh_fbx_static import _emit, _link, _object

_VERSION = 7700
_T = 46_186_158_000


def _write_fbx(path: Path) -> None:
    models = [
        _object("Model", 10, "scene:tianyongxin", "Null"),
        _object("Model", 11, "scene:tianyongxin:Hips", "LimbNode"),
        _object("Model", 12, "scene:tianyongxin:Spine", "LimbNode"),
        _object("Model", 13, "scene:tianyongxin:Head", "LimbNode"),
        _object("Model", 14, "scene:tianyongxin:LeftArm", "LimbNode"),
        _object("Model", 15, "scene:tianyongxin:Head:_1", "Marker"),
        _object("Model", 20, "scene:desk", "Null"),
        _object("Model", 21, "scene:desk:Labeling:root", "LimbNode"),
        _object("Model", 22, "scene:desk:Labeling:root:desk_mesh", "Mesh"),
        _object("Model", 23, "scene:desk:Labeling:root:_1", "Marker"),
        _object("Model", 30, "Unlabeled_Markers", "Null"),
        _object("Model", 31, "Unlabeled_Markers:_1", "Marker"),
    ]
    stack = _object("AnimationStack", 40, "Take", "Stack", children=[
        fbx.node("Properties70", children=[
            fbx.property70("LocalStart", "KTime", [fbx.integer(0, "K")], label="Time", flags=""),
            fbx.property70("LocalStop", "KTime", [fbx.integer(3 * _T, "K")], label="Time", flags=""),
        ]),
    ])
    objects = fbx.node("Objects", children=[*models, stack])
    connections = fbx.node("Connections", children=[
        _link("OO", 11, 10), _link("OO", 12, 11), _link("OO", 13, 12), _link("OO", 14, 12),
        _link("OO", 15, 13), _link("OO", 21, 20), _link("OO", 22, 21), _link("OO", 23, 21),
        _link("OO", 31, 30),
    ])
    settings = fbx.node("GlobalSettings", children=[
        fbx.node("Properties70", children=[
            fbx.property70("TimeMode", "enum", [fbx.integer(2, "I")]),
        ]),
    ])
    out = bytearray(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", _VERSION))
    for top in (objects, connections, settings):
        out.extend(_emit(top, len(out), _VERSION))
    path.write_bytes(bytes(out))


def _write_mcp(path: Path) -> None:
    def record(name: str, bone: str = "Hips") -> bytes:
        return struct.pack("<I", len(name)) + name.encode("ascii") + b"\x1f\x6a\xe7\x34\x44\x00\x00\x00" \
            + struct.pack("<I", len(bone)) + bone.encode("ascii")
    payload = bytearray(b"ViconMCPFile" + bytes(64))
    payload += (record("sunjian") + bytes(300) + record("yanjie") + bytes(120) + record("sunjian")
                + record("desk", "root") + bytes(80) + record("white_trash", "root") + record("desk", "root"))
    # Near-miss noise: a length-prefixed word that is not followed by a bone name.
    payload += struct.pack("<I", 5) + b"frame" + bytes(8) + struct.pack("<I", 5) + b"Other"
    path.write_bytes(bytes(payload))


class FbxRosterTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.path = Path(self.workspace.name) / "sequence.fbx"
        _write_fbx(self.path)

    def tearDown(self):
        self.workspace.cleanup()

    def test_roster_classification_and_counts(self):
        parsed = si.parse_fbx_roster(self.path)
        self.assertEqual([row["name"] for row in parsed["subjects"]], ["tianyongxin"])
        subject = parsed["subjects"][0]
        self.assertEqual((subject["bones"], subject["markers"]), (4, 1))
        self.assertEqual([row["name"] for row in parsed["objects"]], ["desk"])
        desk = parsed["objects"][0]
        self.assertEqual((desk["bones"], desk["markers"], desk["meshes"]), (1, 1, 1))
        self.assertEqual([row["name"] for row in parsed["unlisted_branches"]], ["Unlabeled_Markers"])
        self.assertEqual(parsed["stats"]["model_kinds"]["Marker"], 3)

    def test_parse_step_projects_fbx_roster(self):
        out = si.parse_step(self.path)
        self.assertEqual(out["kind"], "fbx")
        self.assertEqual([row["name"] for row in out["subjects"]], ["tianyongxin"])
        self.assertEqual([row["name"] for row in out["objects"]], ["desk"])
        self.assertEqual(out["stats"]["unlabeled_markers"], 1)
        self.assertEqual(out["stats"]["labeled_markers"], 2)
        self.assertEqual(out["frame_rate_hz"], 100.0)

    def test_animation_span_and_frame_rate(self):
        parsed = si.parse_fbx_roster(self.path)
        self.assertEqual(parsed["frame_rate_hz"], 100.0)
        self.assertEqual(len(parsed["animation"]), 1)
        stack = parsed["animation"][0]
        self.assertEqual(stack["name"], "Take")
        self.assertEqual(stack["duration_seconds"], 3.0)
        self.assertEqual(stack["approx_frames"], 300)

    def test_rejects_non_7700(self):
        self.path.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7500) + bytes(40))
        with self.assertRaises(ValueError):
            si.parse_fbx_roster(self.path)


class FamilyTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name)

    def tearDown(self):
        self.workspace.cleanup()

    def test_family_detection(self):
        self.assertEqual(si.dataset_family(self.root / "01_data_raw" / "group10" / "HOI" / "t" / "a.mcp"), "HOI")
        self.assertEqual(si.dataset_family(self.root / "01_data_raw" / "group1" / "MP" / "t" / "a.mcp"), "MP")
        self.assertEqual(si.dataset_family(self.root / "01_data_raw" / "body" / "actor001" / "C47" / "a.mcp"), "BODY")
        self.assertIsNone(si.dataset_family(self.root / "03_exports" / "group15" / "a.fbx"))

    def test_parse_step_flags_human_only_families(self):
        for family, folder in (("MP", "group1/MP"), ("BODY", "body/actor001")):
            path = self.root / "01_data_raw" / folder / "take" / "a.mcp"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_mcp(path)
            out = si.parse_step(path)
            self.assertEqual(out["dataset_family"], family)
            self.assertIn("human_only_family", out["objects_policy"])
        path = self.root / "01_data_raw" / "group10" / "HOI" / "take" / "a.mcp"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_mcp(path)
        out = si.parse_step(path)
        self.assertEqual(out["dataset_family"], "HOI")
        self.assertNotIn("objects_policy", out)


class McpScanTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.path = Path(self.workspace.name) / "take.mcp"
        _write_mcp(self.path)

    def tearDown(self):
        self.workspace.cleanup()

    def test_subject_records_deduplicated(self):
        found = si.scan_mcp_subjects(self.path)
        self.assertEqual([row["name"] for row in found], ["sunjian", "yanjie"])
        self.assertEqual(found[0]["occurrences"], 2)
        self.assertEqual(found[1]["occurrences"], 1)

    def test_roster_separates_humans_from_rigid_bodies(self):
        roster = si.scan_mcp_roster(self.path)
        self.assertEqual([row["name"] for row in roster["subject"]], ["sunjian", "yanjie"])
        self.assertEqual([row["name"] for row in roster["rigid"]], ["desk", "white_trash"])
        self.assertEqual(roster["rigid"][0]["occurrences"], 2)

    def test_parse_step_projects_mcp_roster(self):
        out = si.parse_step(self.path)
        self.assertEqual(out["kind"], "mcp")
        self.assertEqual([row["name"] for row in out["subjects"]], ["sunjian", "yanjie"])
        self.assertEqual([row["name"] for row in out["rigid_bodies"]], ["desk", "white_trash"])
        self.assertIn("note", out)


class ReportCompactionTests(unittest.TestCase):
    def test_bounded_projection(self):
        report = {"status": "NEEDS_REVIEW", "subjects": [
            {"name": "sunjian", "rig_class": "HUMAN_RIG",
             "marker_frames": {f"M{i}": i for i in range(100)},
             "core_unavailable_intervals": [f"{i} {i + 2}" for i in range(50)]},
        ], "props": [{"name": "desk"}]}
        summary = si.summarize_report(report)
        subject = summary["subjects"][0]
        self.assertEqual(subject["name"], "sunjian")
        self.assertIn("…keys_omitted", subject["marker_frames"])
        self.assertLessEqual(len(subject["core_unavailable_intervals"]), 13)

    def test_non_report_inputs_return_none(self):
        self.assertIsNone(si.summarize_report(None))
        self.assertIsNone(si.summarize_report("text"))


    def test_mcp_capture_clock_frequency_and_timestamp(self):
        import mmap, struct
        from pathlib import Path
        from sequence_inspect import scan_mcp_clock
        blob = bytearray()
        for name, value in ((b'Input.ClockFrequency', b'120.0'), (b'Input.ClockUseTimestamp', b'true')):
            blob += struct.pack('<I', len(name)) + name + struct.pack('<I', len(value)) + value
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'take.mcp'
            path.write_bytes(bytes(blob))
            with open(path, 'rb') as handle:
                view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    self.assertEqual(scan_mcp_clock(view), {'frequency_hz': 120.0, 'timestamped': True})
                finally:
                    view.close()

    def test_mcp_capture_window_wall_clock(self):
        import struct
        from datetime import datetime, timezone
        from sequence_inspect import scan_mcp_capture_window
        date_ms = 1793675461253
        base = 1342932215625
        blob = bytearray()
        for name, value in ((b'Date', b'1793675461253'), (b'LiveGlobalStartTick', b'1342932215625')):
            blob += struct.pack('<I', len(name)) + name + struct.pack('<I', len(value)) + value
        blob += bytes(64)
        for index in range(10):
            blob += struct.pack('<Q', base + (100 + index) * 1_125_000)  # frame ticks at 120fps
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'take.mcp'
            path.write_bytes(bytes(blob))
            window = scan_mcp_capture_window(path, 10)
            self.assertIsNotNone(window)
            start = datetime.fromisoformat(window['capture_start'])
            end = datetime.fromisoformat(window['capture_end'])
            expected_open = datetime.fromtimestamp(date_ms / 1000)  # naive local, same frame as capture_start
            self.assertAlmostEqual((start - expected_open).total_seconds(), (100 * 1_125_000) / 135e6, places=3)
            # end = exact start + frame budget at the capture clock (default 120fps)
            self.assertAlmostEqual((end - start).total_seconds(), 10 / 120, places=3)


if __name__ == "__main__":
    unittest.main()

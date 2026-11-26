"""Consumer-visible routing and containment boundaries for sequence-local runs."""
from pathlib import Path
import tempfile
import unittest

from pipeline_layout import (artifact_roots, bucket_entries, in_run, initialize_layout,
                             is_generated, load_layout, report_paths, report_relative, take_paths)


class SequenceLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.raw = Path(self.temporary.name) / "MocapData"
        self.relative = Path("group1/HOI/INFO_X_TEA_CL01/G01-INFO_X_TEA_CL01.mcp")
        self.source = self.raw / self.relative
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"original MCP")
        self.control = self.raw / "_processing" / "main"

    def configure(self):
        return initialize_layout(self.control, self.raw)

    def test_planning_does_not_write_and_keeps_raw_and_versions_separate(self):
        planned = take_paths(self.control, self.relative, data_root=self.raw)
        self.assertEqual(planned["report_path"], self.source.parent / "processed/main" / self.source.name / "report.json")
        self.assertFalse(self.control.exists())
        self.assertFalse(planned["take_root"].exists())
        self.configure()
        second = self.raw / "_processing" / "repair"
        initialize_layout(second, self.raw)
        another_take = self.relative.with_name("G01-INFO_X_TEA_CL02.mcp")
        self.assertNotEqual(take_paths(self.control, self.relative)["destination"],
                            take_paths(second, self.relative)["destination"])
        self.assertNotEqual(take_paths(self.control, self.relative)["take_root"],
                            take_paths(self.control, another_take)["take_root"])
        self.assertEqual(self.source.read_bytes(), b"original MCP")

    def test_canonical_scan_excludes_other_versions_and_diagnostic_reports(self):
        self.configure()
        paths = take_paths(self.control, self.relative)
        paths["report_path"].parent.mkdir(parents=True)
        paths["report_path"].write_text("{}", encoding="utf-8")
        diagnostic = paths["failed_root"] / "attempt" / "report.json"
        diagnostic.parent.mkdir(parents=True)
        diagnostic.write_text("{}", encoding="utf-8")
        other = self.raw / "_processing" / "repair"
        initialize_layout(other, self.raw)
        other_report = take_paths(other, self.relative)["report_path"]
        other_report.parent.mkdir(parents=True)
        other_report.write_text("{}", encoding="utf-8")
        self.assertEqual(list(report_paths(self.control)), [paths["report_path"]])
        self.assertEqual(report_relative(self.control, paths["report_path"]), self.relative)
        for invalid in (other_report, diagnostic, self.source):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                report_relative(self.control, invalid)

    def test_scope_authorizes_only_control_and_exact_version_take_descendants(self):
        self.configure()
        paths = take_paths(self.control, self.relative)
        self.assertTrue(in_run(self.control, paths["destination"] / "sequence.fbx"))
        self.assertTrue(in_run(self.control, self.control / "repair_plan.json"))
        self.assertFalse(in_run(self.control, self.source))
        self.assertFalse(in_run(self.control, self.source.parent / "processed/main"))
        self.assertFalse(in_run(self.control, self.source.parent / "processed/other" / self.source.name / "artifacts/file.fbx"))
        self.assertFalse(in_run(self.control, self.source.parent / "processed/main/not-a-take/artifacts/file.fbx"))
        self.assertEqual(artifact_roots(self.control, self.relative),
                         (paths["destination"], paths["review_destination"]))

    def test_attempt_discovery_includes_only_this_version(self):
        self.configure()
        paths = take_paths(self.control, self.relative)
        attempt = paths["work_root"] / "job-attempt"
        attempt.mkdir(parents=True)
        unassigned = self.control / ".work" / "unassigned"
        unassigned.mkdir(parents=True)
        sibling = self.source.parent / "processed/other" / self.source.name / ".work/foreign"
        sibling.mkdir(parents=True)
        self.assertEqual(set(bucket_entries(self.control, ".work")), {attempt, unassigned})

    def test_control_root_and_identity_traversal_are_rejected(self):
        for bad in (self.raw, self.source.parent / "output", self.raw.parent):
            with self.subTest(control=bad), self.assertRaises(ValueError):
                initialize_layout(bad, self.raw)
        self.configure()
        for bad in ("../take.mcp", str(self.source), "group1/HOI/seq/processed/take.mcp"):
            with self.subTest(relative=bad), self.assertRaises(ValueError):
                take_paths(self.control, bad)
        with self.assertRaises(ValueError):
            initialize_layout(self.control, self.raw, version_key="other")

    def test_generated_subtrees_are_excluded_without_excluding_raw_sequences(self):
        self.assertFalse(is_generated(self.source, self.raw))
        self.assertTrue(is_generated(self.source.parent / "processed/main/take.mcp", self.raw))
        self.assertTrue(is_generated(self.control / "copy.mcp", self.raw))

    def test_generic_output_layout_is_still_independent(self):
        output = self.raw.parent / "generic"
        paths = take_paths(output, self.relative)
        self.assertIsNone(load_layout(output))
        self.assertEqual(paths["report_path"], output / "takes" / self.relative / "report.json")
        self.assertEqual(paths["review_destination"], output / "_needs_review" / self.relative / "artifacts")
        self.assertFalse(in_run(output, self.source))


if __name__ == "__main__":
    unittest.main()

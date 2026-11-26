import json
from pathlib import Path
import tempfile
import unittest

from pipeline_migration_inventory import raw_identity, supplemental_plan


class SupplementalInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.raw = self.base / "MocapData"
        self.relative = Path("group1/HOI/seq/G01-seq.mcp")
        self.input = self.raw / self.relative
        self.input.parent.mkdir(parents=True)
        self.input.write_bytes(b"raw")
        self.source = self.base / "processed"
        self.canonical = self.source / "takes" / self.relative / "report.json"
        self.save(self.canonical, {"input_path": str(self.input), "relative_path": self.relative.as_posix()})
        self.artifact = self.canonical.parent / "artifacts" / "G01-seq.fbx"
        self.artifact.parent.mkdir()
        self.artifact.write_bytes(b"fbx")
        self.controls = self.raw / "_processing"

    def save(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_standalone_repair_routes_only_with_source_evidence(self):
        repair = self.base / "repair"
        self.save(repair / "repair_audit.json", {"source": str(self.artifact), "status": "REJECTED"})
        (repair / "candidate.vdf").write_bytes(b"rejected candidate")
        unknown = self.base / "G01-seq-filename-only"
        unknown.mkdir()
        (unknown / "G01-seq.fbx").write_bytes(b"not evidence")
        plan = supplemental_plan(self.raw, [self.source, repair, unknown], self.controls)
        moves = {row["source"]: row["target"] for row in plan["moves"]}
        self.assertEqual(Path(moves[str(repair)]), self.input.parent / "processed/repair" / self.input.name / "diagnostics")
        self.assertNotIn(str(unknown), moves)
        self.assertTrue((repair / "candidate.vdf").is_file())
        self.assertEqual(self.input.read_bytes(), b"raw")

    def test_mesh_scene_and_subject_outputs_preserve_separate_destinations(self):
        mesh = self.base / "mesh"
        subject = self.artifact.parent / "subjects" / "actor" / "sequence.fbx"
        subject.parent.mkdir(parents=True)
        subject.write_bytes(b"subject")
        scene_out, subject_out = mesh / "scene_with_mesh.fbx", mesh / "subject_with_mesh.fbx"
        self.save(mesh / "mesh_batch_report.json", {"files": [
            {"input": str(self.artifact), "output": str(scene_out), "status": "PARTIAL"},
            {"input": str(subject), "output": str(subject_out), "status": "SUCCESS"}]})
        scene_out.write_bytes(b"scene mesh")
        subject_out.write_bytes(b"subject mesh")
        plan = supplemental_plan(self.raw, [self.source, mesh], self.controls)
        moved = {Path(row["source"]): Path(row["target"]) for row in plan["moves"]}
        self.assertEqual(moved[scene_out].parent.name, "artifacts")
        self.assertEqual(moved[subject_out].parent.parts[-3:], ("artifacts", "subjects", "actor"))
        self.assertEqual(plan["mesh_reports"][0]["files"][0]["status"], "PARTIAL")
        self.assertFalse(plan["mesh_reports"][0]["tracking_quality_changed"])

    def test_deleted_raw_and_foreign_source_cannot_acquire_an_identity(self):
        self.assertEqual(raw_identity(self.artifact, self.raw, [self.source]), self.relative)
        self.assertIsNone(raw_identity(self.artifact, self.raw, [self.base / "different"]))
        self.input.unlink()
        self.assertIsNone(raw_identity(self.artifact, self.raw, [self.source]))

    def test_failed_checkpoint_uses_control_input_not_candidate_filename(self):
        stage = self.source / "_failed" / "attempt"
        self.save(stage / ".control/result.json", {"input_path": str(self.input)})
        candidate = stage / "unrelated-name.vdf"
        candidate.write_bytes(b"checkpoint")
        self.assertEqual(raw_identity(candidate, self.raw, [self.source]), self.relative)

    def test_legacy_missing_observations_do_not_erase_explicit_diagnostic_identity(self):
        diagnosis = self.base / "diagnostic"
        self.save(diagnosis / "diagnostic_job.json",
                  {"input_path": str(self.input), "points": [float("nan"), 1.0]})
        plan = supplemental_plan(self.raw, [self.source, diagnosis], self.controls)
        self.assertIn(str(diagnosis), {item["source"] for item in plan["moves"]})

    def test_failed_mesh_attempt_without_output_retains_audit_not_a_fake_artifact(self):
        mesh = self.base / "failed_mesh"
        self.save(mesh / "mesh_batch_report.json",
                  {"files": [{"input": str(self.artifact), "output": None, "status": "FAILED"}]})
        plan = supplemental_plan(self.raw, [self.source, mesh], self.controls)
        self.assertEqual(plan["mesh_reports"], [])
        self.assertTrue(any("no output" in warning for warning in plan["warnings"]))


if __name__ == "__main__":
    unittest.main()

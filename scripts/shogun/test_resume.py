"""Unit tests for pipeline_resume.select_resume_jobs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pipeline_layout import initialize_layout, take_paths
from pipeline_resume import select_resume_jobs


class ResumeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.takes = self.output / "takes"
        self.needs_review = self.output / "_needs_review"
        self.work = self.output / ".work"

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_job(self, name: str, input_kind: str = "mcp", preflight_errors=None):
        job_id = f"job_{name}"
        input_path = str(self.root / f"{name}.{input_kind}")
        dest = self.takes / name / "artifacts"
        rev_dest = self.needs_review / name / "artifacts"
        rep_path = self.takes / name / "report.json"
        return {
            "job_id": job_id,
            "input_path": input_path,
            "input_kind": input_kind,
            "relative_path": name,
            "destination": str(dest),
            "review_destination": str(rev_dest),
            "report_path": str(rep_path),
            "log_path": str(self.output / "logs" / name / "processing.log"),
            "native_log_path": str(self.output / "logs" / name / "shogun.log"),
            "preflight_errors": preflight_errors or [],
        }

    def write_artifacts(self, directory: Path, stem: str, kind: str = "mcp") -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        roles = ["vdf", "fbx", "vsk"] if kind == "mcp" else ["vdf", "fbx"]
        artifacts = {}
        for role in roles:
            file_path = directory / f"{stem}.{role}"
            file_path.write_bytes(b"dummy_payload")
            artifacts[role] = str(file_path)
        return artifacts

    def local_job(self, name="take.mcp"):
        data = self.root / "raw"
        data.mkdir(exist_ok=True)
        initialize_layout(self.output, data, "local-version")
        relative = f"group1/HOI/sequence/{name}"
        paths = take_paths(self.output, relative)
        job = self.make_job(name)
        job.update(relative_path=relative, input_path=str(data / relative),
                   layout_data_root=str(data),
                   **{field: str(paths[field]) for field in
                      ("destination", "review_destination", "report_path")})
        job["log_path"] = str(paths["log_dir"] / "processing.log")
        job["native_log_path"] = str(paths["log_dir"] / "shogun.log")
        return job, paths

    def complete_report(self, job, artifacts, status="SUCCESS"):
        path = Path(job["report_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "status": status, "job_id": job["job_id"], "input_path": job["input_path"],
            "artifacts": artifacts,
        }), encoding="utf-8")

    def complete_stage(self, job, stage, *, input_path=None):
        artifacts = self.write_artifacts(stage, "take")
        control = stage / ".control/result.json"
        control.parent.mkdir()
        control.write_text(json.dumps({
            "status": "SUCCESS", "job_id": job["job_id"],
            "input_path": input_path or job["input_path"],
            "artifacts": {role: Path(value).name for role, value in artifacts.items()},
        }), encoding="utf-8")

    def test_local_success_and_review_reports_are_preserved_read_only(self):
        for name, status, field in (("first.mcp", "SUCCESS", "destination"),
                                    ("second.mcp", "NEEDS_REVIEW", "review_destination")):
            job, paths = self.local_job(name)
            self.complete_report(job, self.write_artifacts(paths[field], "take"), status)
            before = Path(job["report_path"]).read_bytes()
            result = select_resume_jobs([job], self.output)
            self.assertEqual(result["preserved"], [job])
            self.assertEqual(result["pending"], [])
            self.assertEqual(Path(job["report_path"]).read_bytes(), before)

    def test_local_stage_recovery_uses_only_own_take_workspace_and_input_identity(self):
        job, paths = self.local_job()
        valid = paths["work_root"] / (job["job_id"] + "-valid")
        self.complete_stage(job, valid)
        self.complete_stage(job, self.work / (job["job_id"] + "-centralized"))
        other = take_paths(self.output, "group1/HOI/sequence/other.mcp")
        self.complete_stage(job, other["work_root"] / (job["job_id"] + "-borrowed"))
        wrong = paths["work_root"] / (job["job_id"] + "-wrong-input")
        self.complete_stage(job, wrong, input_path=str(self.root / "different.mcp"))
        result = select_resume_jobs([job], self.output)
        self.assertEqual(result["recoverable"], [(job, valid)])
        self.assertEqual(result["pending"], [])

    def test_local_artifacts_cannot_borrow_raw_neighbor_version_or_control_files(self):
        job, paths = self.local_job()
        artifacts = self.write_artifacts(paths["destination"], "take")
        disallowed = [
            self.root / "raw/group1/HOI/sequence/raw.vdf",
            take_paths(self.output, "group1/HOI/sequence/other.mcp")["destination"] / "take.vdf",
            paths["take_root"].parent.parent / "other-version/take.mcp/artifacts/take.vdf",
            self.output / "unrelated.vdf",
        ]
        for path in disallowed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not this take's artifact")
            with self.subTest(path=path):
                self.complete_report(job, {**artifacts, "vdf": str(path)})
                with self.assertRaisesRegex(ValueError, "different take"):
                    select_resume_jobs([job], self.output)

    def test_local_subject_export_and_report_must_match_exact_take(self):
        job, paths = self.local_job()
        artifacts = self.write_artifacts(paths["destination"], "take")
        self.complete_report(job, artifacts)
        report_path = Path(job["report_path"])
        report = json.loads(report_path.read_bytes())
        other = take_paths(self.output, "group1/HOI/sequence/other.mcp")
        foreign = self.write_artifacts(other["destination"], "other")["fbx"]
        report["subject_exports"] = [{"path": foreign}]
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Subject export belongs to a different take"):
            select_resume_jobs([job], self.output)
        job["report_path"] = str(other["report_path"])
        with self.assertRaisesRegex(ValueError, "Canonical report location"):
            select_resume_jobs([job], self.output)

    def test_local_failed_report_is_queued_for_retry_without_losing_evidence(self):
        job, paths = self.local_job()
        self.complete_report(job, {}, "FAILED")
        before = paths["report_path"].read_bytes()
        result = select_resume_jobs([job], self.output)
        self.assertEqual(result["pending"], [job])
        self.assertEqual(result["retry_reports"][0]["report_path"], job["report_path"])
        self.assertEqual(paths["report_path"].read_bytes(), before)

    def test_local_symlink_artifact_escape_cannot_be_preserved(self):
        job, paths = self.local_job()
        artifacts = self.write_artifacts(paths["destination"], "take")
        original = Path(artifacts["vdf"])
        external = self.root / "external.vdf"
        external.write_bytes(b"external native data")
        original.unlink()
        try:
            original.symlink_to(external)
        except OSError as error:
            self.skipTest(f"Host does not permit test symlinks: {error}")
        self.complete_report(job, artifacts)
        with self.assertRaisesRegex(ValueError, "different take"):
            select_resume_jobs([job], self.output)

    def test_valid_success_and_needs_review_are_preserved(self):
        job1 = self.make_job("take1", "mcp")
        job2 = self.make_job("take2", "vdf")

        art1 = self.write_artifacts(Path(job1["destination"]), "take1", "mcp")
        Path(job1["report_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job1["report_path"]).write_text(json.dumps({
            "status": "SUCCESS", "job_id": job1["job_id"], "input_path": job1["input_path"],
            "artifacts": art1
        }), encoding="utf-8")

        art2 = self.write_artifacts(Path(job2["review_destination"]), "take2", "vdf")
        Path(job2["report_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job2["report_path"]).write_text(json.dumps({
            "status": "NEEDS_REVIEW", "job_id": job2["job_id"], "input_path": job2["input_path"],
            "artifacts": art2
        }), encoding="utf-8")

        res = select_resume_jobs([job1, job2], self.output)
        self.assertEqual(len(res["preserved"]), 2)
        self.assertEqual(len(res["pending"]), 0)
        self.assertEqual(len(res["retry_reports"]), 0)
        self.assertEqual(len(res["recoverable"]), 0)

    def test_failed_report_is_retried_and_queued_for_archive(self):
        job = self.make_job("take1", "mcp")
        Path(job["report_path"]).parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "status": "FAILED", "job_id": job["job_id"], "input_path": job["input_path"],
            "error": "Timeout"
        }
        Path(job["report_path"]).write_text(json.dumps(report_data), encoding="utf-8")

        res = select_resume_jobs([job], self.output)
        self.assertEqual(len(res["preserved"]), 0)
        self.assertEqual(len(res["pending"]), 1)
        self.assertEqual(len(res["retry_reports"]), 1)
        self.assertEqual(res["retry_reports"][0]["job_id"], job["job_id"])
        self.assertEqual(res["retry_reports"][0]["report_data"]["status"], "FAILED")

    def test_unprocessed_job_is_pending(self):
        job = self.make_job("take1", "mcp")
        res = select_resume_jobs([job], self.output)
        self.assertEqual(len(res["preserved"]), 0)
        self.assertEqual(len(res["pending"]), 1)
        self.assertEqual(res["pending"][0]["job_id"], job["job_id"])

    def test_empty_destination_dir_does_not_block_pending(self):
        job = self.make_job("take1", "mcp")
        Path(job["destination"]).mkdir(parents=True, exist_ok=True)
        res = select_resume_jobs([job], self.output)
        self.assertEqual(len(res["pending"]), 1)

    def test_unreported_nonempty_files_in_destination_raises_error(self):
        job = self.make_job("take1", "mcp")
        Path(job["destination"]).mkdir(parents=True, exist_ok=True)
        (Path(job["destination"]) / "take1.vdf").write_bytes(b"orphan")
        with self.assertRaisesRegex(ValueError, "Unreported non-empty output files exist"):
            select_resume_jobs([job], self.output)

    def test_missing_referenced_artifact_raises_error(self):
        job = self.make_job("take1", "mcp")
        art = self.write_artifacts(Path(job["destination"]), "take1", "mcp")
        # Delete one artifact
        Path(art["vsk"]).unlink()
        Path(job["report_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["report_path"]).write_text(json.dumps({
            "status": "SUCCESS", "job_id": job["job_id"], "input_path": job["input_path"],
            "artifacts": art
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Canonical report references missing or empty artifact"):
            select_resume_jobs([job], self.output)

    def test_identity_mismatch_raises_error(self):
        job = self.make_job("take1", "mcp")
        art = self.write_artifacts(Path(job["destination"]), "take1", "mcp")
        Path(job["report_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["report_path"]).write_text(json.dumps({
            "status": "SUCCESS", "job_id": "different_id", "input_path": job["input_path"],
            "artifacts": art
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Job ID mismatch"):
            select_resume_jobs([job], self.output)

    def test_recoverable_stage_found_when_report_missing(self):
        job = self.make_job("take1", "mcp")
        stage = self.work / f"{job['job_id']}-uuid"
        ctrl = stage / ".control"
        ctrl.mkdir(parents=True, exist_ok=True)
        (stage / "take1.vdf").write_bytes(b"vdf_data")
        (stage / "take1.fbx").write_bytes(b"fbx_data")
        (stage / "take1.vsk").write_bytes(b"vsk_data")
        (ctrl / "result.json").write_text(json.dumps({
            "status": "SUCCESS", "job_id": job["job_id"], "input_path": job["input_path"],
            "artifacts": {"vdf": "take1.vdf", "fbx": "take1.fbx", "vsk": "take1.vsk"}
        }), encoding="utf-8")

        res = select_resume_jobs([job], self.output)
        self.assertEqual(len(res["recoverable"]), 1)
        self.assertEqual(res["recoverable"][0][0]["job_id"], job["job_id"])
        self.assertEqual(res["recoverable"][0][1], stage)

    def test_competing_recoverable_stages_raises_error(self):
        job = self.make_job("take1", "mcp")
        for suffix in ("1", "2"):
            stage = self.work / f"{job['job_id']}-uuid{suffix}"
            ctrl = stage / ".control"
            ctrl.mkdir(parents=True, exist_ok=True)
            (stage / "take1.vdf").write_bytes(b"vdf_data")
            (stage / "take1.fbx").write_bytes(b"fbx_data")
            (stage / "take1.vsk").write_bytes(b"vsk_data")
            (ctrl / "result.json").write_text(json.dumps({
                "status": "SUCCESS", "job_id": job["job_id"], "input_path": job["input_path"],
                "artifacts": {"vdf": "take1.vdf", "fbx": "take1.fbx", "vsk": "take1.vsk"}
            }), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Multiple competing complete stages"):
            select_resume_jobs([job], self.output)


if __name__ == "__main__":
    unittest.main()

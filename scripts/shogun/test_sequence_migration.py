"""Filesystem-only migration behavior; never imports or launches the Shogun SDK."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pipeline_layout import load_layout, take_paths
import step0_organize_sequences as migration


class SequenceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "raw"
        self.data.mkdir()
        self.control = self.data / "_processing"
        self.source = self.root / "batch"
        self.source.mkdir()
        self.relative = Path("group1/HOI/take/A.mcp")
        self.raw = self.data / self.relative
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(b"original MCP\x00\xff")
        self.raw.with_suffix(".x2d").write_bytes(b"original X2D")
        self.raw.with_suffix(".enf").write_bytes(b"capture bookkeeping")

    def write_json(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
        return path

    def report(self, source=None, relative=None, *, review=False, status=None, extra=None):
        source = source or self.source
        relative = relative or self.relative
        artifact = source / ("_needs_review" if review else "takes") / relative / "artifacts" / "native.vdf"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"native VDF bytes\x00\x80")
        log = source / "logs" / relative / "processing.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"historical path: {artifact}\n", encoding="utf-8")
        document = {"schema_version": 1, "input_path": str(self.data / relative),
                    "relative_path": relative.as_posix(),
                    "job_id": hashlib.sha256(relative.as_posix().casefold().encode()).hexdigest()[:16],
                    "status": status or ("NEEDS_REVIEW" if review else "SUCCESS"),
                    "quality": {"basis": "post_processing", "usable": not review},
                    "artifacts": {"vdf": str(artifact)}, "logs": {"processing": str(log)}}
        document.update(extra or {})
        path = self.write_json(source / "takes" / relative / "report.json", document)
        return path, artifact, document

    def paths(self, source=None):
        source = source or self.source
        return take_paths(self.control / source.name, self.relative, data_root=self.data)

    def plan(self, sources=None, **kwargs):
        return migration.plan_migration(self.data, sources or [self.source], self.control, **kwargs)

    def test_artifact_from_other_version_of_same_take_is_remapped(self):
        repair = self.source / "_repair_runs" / "repair_1"
        report, _, _ = self.report()
        _, repair_artifact, _ = self.report(source=repair, review=True)
        document = json.loads(report.read_text(encoding="utf-8"))
        document["artifacts"]["vdf"] = str(repair_artifact)
        self.write_json(report, document)
        plan = self.plan()
        self.assertTrue(any("another version of the same take" in w for w in plan["warnings"]))
        plan = migration.apply_migration(plan)
        moved_report = self.paths()["report_path"]
        self.assertTrue(moved_report.is_file())
        moved = json.loads(moved_report.read_text(encoding="utf-8"))
        repair_take = take_paths(self.control / self.source.name / "_repair_runs" / "repair_1",
                                 self.relative, data_root=self.data)
        self.assertEqual(moved["artifacts"]["vdf"],
                         str(repair_take["review_destination"] / "native.vdf"))
        self.assertTrue((repair_take["review_destination"] / "native.vdf").is_file())


    def test_default_dry_run_is_exact_and_writes_nothing(self):
        report, artifact, _ = self.report(review=True)
        before = {str(path): migration._fingerprint(path) for path in self.root.rglob("*") if path.is_file()}
        plan = migration.migrate(self.data, [self.source], self.control)
        after = {str(path): migration._fingerprint(path) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(self.control.exists())
        self.assertEqual(plan["phase"], "planned")
        mapping = plan["path_mapping"]
        self.assertEqual(migration._mapped(report, mapping), self.paths()["report_path"])
        self.assertEqual(migration._mapped(artifact, mapping), self.paths()["review_destination"] / artifact.name)
        self.assertTrue(any(item["source"] == str(report) and item["migrated_sha256"]
                            for item in plan["json_files"]))

    def test_apply_preserves_raw_binary_and_historical_log_bytes(self):
        report, artifact, document = self.report(review=True)
        original_report = report.read_bytes()
        original_binary = artifact.read_bytes()
        original_identity = migration._fingerprint(artifact)
        log = Path(document["logs"]["processing"])
        original_log = log.read_bytes()
        baseline = migration._raw_baseline(self.data)
        plan = migration.apply_migration(self.plan())
        paths = self.paths()
        self.assertEqual(migration._raw_baseline(self.data), baseline)
        self.assertFalse(self.source.exists())
        self.assertEqual((paths["review_destination"] / artifact.name).read_bytes(), original_binary)
        self.assertEqual(migration._fingerprint(paths["review_destination"] / artifact.name), original_identity)
        self.assertEqual((paths["log_dir"] / log.name).read_bytes(), original_log)
        current = json.loads(paths["report_path"].read_text())
        self.assertEqual(current["status"], "NEEDS_REVIEW")
        self.assertEqual(current["quality"], document["quality"])
        self.assertEqual(current["artifacts"]["vdf"], str(paths["review_destination"] / artifact.name))
        self.assertEqual(load_layout(self.control / self.source.name)["version_key"], self.source.name)
        record_index = next(index for index, row in enumerate(plan["json_files"]) if row["source"] == str(report))
        self.assertEqual(migration._archive_path(Path(plan["journal"]), record_index).read_bytes(), original_report)
        self.assertEqual(plan["phase"], "complete")

    def test_failed_work_recheck_and_resume_history_use_explicit_identity(self):
        report, _, document = self.report(status="FAILED")
        failed = self.source / "_failed" / "not-a-job-nickname"
        self.write_json(failed / ".control" / "result.json",
                        {"job_id": document["job_id"], "input_path": str(self.raw), "status": "FAILED"})
        checkpoint = failed / "checkpoint.vdf"
        checkpoint.write_bytes(b"failed attempt native checkpoint")
        work = self.source / ".work" / "random-attempt"
        self.write_json(work / ".control" / "session-ready.json", {"input_path": str(self.raw)})
        unknown = self.source / ".work" / (document["job_id"] + "-looks-owned")
        unknown.mkdir(parents=True)
        (unknown / "partial.vdf").write_bytes(b"must not guess ownership")
        recheck = self.source / ".quality_recheck_attempts" / "independent-attempt"
        self.write_json(recheck / "attempt.json", {"report": str(report), "phase": "aborted"})
        history = self.source / "_resume_history" / "old-run" / "takes" / self.relative / "report.json"
        self.write_json(history, document)
        plan = migration.apply_migration(self.plan())
        paths = self.paths()
        self.assertEqual((paths["failed_root"] / failed.name / checkpoint.name).read_bytes(), b"failed attempt native checkpoint")
        self.assertTrue((paths["work_root"] / work.name / ".control" / "session-ready.json").is_file())
        attempt = json.loads((paths["recheck_root"] / recheck.name / "attempt.json").read_text())
        self.assertEqual(attempt["report"], str(paths["report_path"]))
        self.assertEqual(attempt["phase"], "aborted")
        self.assertTrue((paths["history_root"] / history.parent.relative_to(self.source / "_resume_history") / "report.json").is_file())
        self.assertTrue((self.control / self.source.name / ".work" / unknown.name / "partial.vdf").is_file())
        self.assertTrue(any("Unknown job identity" in warning for warning in plan["warnings"]))

    def test_nested_repair_lineage_and_independent_report_hashes_follow_json(self):
        report, checkpoint, document = self.report(review=True)
        old_hash = migration._digest(report)
        checkpoint_fingerprint = {"size": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns}
        repair = self.source / "_repair_runs" / "repair-version"
        child, _, _ = self.report(repair, extra={
            "source_output": str(self.source), "source_report": str(report), "source_report_sha256": old_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "processing_source": {"path": str(checkpoint), "checkpoint_fingerprint": checkpoint_fingerprint,
                                  "completion_evidence": {"source_report": str(report), "source_report_sha256": old_hash}}})
        independent = self.write_json(self.source / "recheck_summary.json",
                                      {"rows": [{"report": str(report), "report_sha256": old_hash}], "usable": False})
        batch = self.write_json(self.source / "batch_report.json", {"jobs": [document], "source_report_sha256": old_hash})
        plan = migration.apply_migration(self.plan())
        new_report = self.paths()["report_path"]
        new_hash = migration._digest(new_report)
        self.assertNotEqual(old_hash, new_hash)
        repair_control = self.control / self.source.name / "_repair_runs" / repair.name
        child_new = take_paths(repair_control, self.relative, data_root=self.data)["report_path"]
        current = json.loads(child_new.read_text())
        self.assertEqual(current["source_report_sha256"], new_hash)
        self.assertEqual(current["processing_source"]["completion_evidence"]["source_report_sha256"], new_hash)
        self.assertEqual(current["checkpoint_fingerprint"], checkpoint_fingerprint)
        self.assertEqual(current["processing_source"]["checkpoint_fingerprint"], checkpoint_fingerprint)
        self.assertEqual(json.loads((self.control / self.source.name / independent.name).read_text())["rows"][0]["report_sha256"], new_hash)
        current_batch = json.loads((self.control / self.source.name / batch.name).read_text())
        self.assertEqual(current_batch["jobs"][0]["artifacts"]["vdf"], str(self.paths()["review_destination"] / checkpoint.name))
        self.assertEqual(current_batch["source_report_sha256"], new_hash)
        self.assertTrue(any(row["source"] == str(child) for row in plan["json_files"]))

    def test_legacy_failed_input_only_report_stays_canonical_without_synthesized_fields(self):
        document = {"status": "FAILED", "input_path": str(self.raw),
                    "job_id": "legacy-preflight", "artifacts": {},
                    "error": "Worker exited without a result report",
                    "expected_subjects": ["actor"]}
        report = self.write_json(self.source / "takes" / self.relative / "report.json", document)
        original = report.read_bytes()
        failed = self.source / "_failed" / "legacy-worker"
        self.write_json(failed / "result.json", {"job_id": document["job_id"]})
        (failed / "checkpoint.vdf").write_bytes(b"retain failed worker evidence")
        plan = migration.apply_migration(self.plan())
        row = next(row for row in plan["canonical_reports"] if row["source"] == str(report))
        self.assertEqual(row["relative_path"], self.relative.as_posix())
        self.assertEqual(row["status"], "FAILED")
        self.assertEqual(row["identity_evidence"], "failed_input_path_and_take_location")
        self.assertEqual(json.loads(self.paths()["report_path"].read_text()), document)
        self.assertEqual((self.paths()["failed_root"] / failed.name / "checkpoint.vdf").read_bytes(),
                         b"retain failed worker evidence")
        index = next(index for index, row in enumerate(plan["json_files"]) if row["source"] == str(report))
        self.assertEqual(migration._archive_path(Path(plan["journal"]), index).read_bytes(), original)
        self.assertTrue(any(str(report) in warning and "retained canonical" in warning
                            for warning in plan["warnings"]))
        self.assertEqual(migration.migrate(self.data, [self.source], self.control, apply=True)["phase"], "complete")

    def test_legacy_failed_input_identity_must_match_complete_take_location(self):
        other = self.raw.with_name("other.mcp")
        other.write_bytes(b"different raw capture")
        report = self.write_json(self.source / "takes" / self.relative / "report.json",
                                 {"status": "FAILED", "input_path": str(other), "artifacts": {}})
        with self.assertRaises(migration.MigrationError) as raised:
            self.plan()
        self.assertIn(str(report), str(raised.exception))
        self.assertIn("location/identity mismatch", str(raised.exception))
        self.assertFalse(self.control.exists())
        self.assertTrue(report.is_file())

    def test_missing_relative_path_for_nonfailed_report_reports_exact_file_and_cause(self):
        report = self.write_json(self.source / "takes" / self.relative / "report.json",
                                 {"status": "SUCCESS", "input_path": str(self.raw), "artifacts": {}})
        with self.assertRaises(migration.MigrationError) as raised:
            self.plan()
        self.assertIn(str(report), str(raised.exception))
        self.assertIn("Canonical report is missing relative_path", str(raised.exception))
        self.assertFalse(self.control.exists())

    def test_legacy_failed_exception_never_accepts_invalid_present_relative_path(self):
        report = self.source / "takes" / self.relative / "report.json"
        for value in (None, "", 0, False, [], {}):
            with self.subTest(relative_path=value):
                self.write_json(report, {"status": "FAILED", "input_path": str(self.raw),
                                         "relative_path": value, "artifacts": {}})
                with self.assertRaises(migration.MigrationError) as raised:
                    self.plan()
                self.assertIn(str(report), str(raised.exception))
                self.assertFalse(self.control.exists())

    def test_existing_take_target_refuses_before_any_mutation(self):
        report, _, _ = self.report()
        target = self.paths()["take_root"]
        target.mkdir(parents=True)
        sentinel = target / "user.bin"
        sentinel.write_bytes(b"do not overwrite")
        with self.assertRaises(migration.MigrationError):
            self.plan()
        self.assertTrue(report.is_file())
        self.assertEqual(sentinel.read_bytes(), b"do not overwrite")
        self.assertFalse(self.control.exists())

    def test_duplicate_version_basename_refuses(self):
        self.report()
        other = self.root / "other" / self.source.name
        self.report(other)
        with self.assertRaisesRegex(migration.MigrationError, "collision"):
            self.plan([self.source, other])
        self.assertFalse(self.control.exists())

    def test_malicious_input_path_and_relative_escape_refuse(self):
        report, _, document = self.report()
        outside = self.root / "outside.mcp"
        outside.write_bytes(b"unrelated raw")
        document["input_path"] = str(outside)
        self.write_json(report, document)
        with self.assertRaises(migration.MigrationError):
            self.plan()
        document["input_path"] = str(self.raw)
        document["relative_path"] = "../outside.mcp"
        self.write_json(report, document)
        with self.assertRaises(migration.MigrationError):
            self.plan()
        self.assertFalse(self.control.exists())

    def test_deleted_historical_raw_is_unassigned_not_a_neighbor(self):
        self.report()
        deleted = self.relative.with_name("deleted-timestamp.mcp")
        old, artifact, _ = self.report(relative=deleted)
        plan = migration.apply_migration(self.plan())
        target = self.control / self.source.name / "unassigned" / "takes" / deleted
        self.assertTrue((target / "report.json").is_file())
        self.assertEqual((target / "artifacts" / artifact.name).read_bytes(), b"native VDF bytes\x00\x80")
        self.assertFalse(take_paths(self.control / self.source.name, deleted, data_root=self.data)["take_root"].exists())
        self.assertTrue(any("Deleted raw identity" in warning for warning in plan["warnings"]))
        self.assertFalse(any(row["source"] == str(old) for row in plan["canonical_reports"]))

    def test_interrupted_atomic_move_resumes_and_completed_resume_is_idempotent(self):
        report, _, _ = self.report()
        original = migration._rename_no_replace
        interrupted = False

        def rename_then_interrupt(source, target):
            nonlocal interrupted
            original(source, target)
            if not interrupted and source == report.parent:
                interrupted = True
                raise RuntimeError("simulated power loss after rename before journal stage save")

        plan = self.plan()
        with mock.patch.object(migration, "_rename_no_replace", side_effect=rename_then_interrupt):
            with self.assertRaisesRegex(RuntimeError, "simulated power loss"):
                migration.apply_migration(plan)
        self.assertTrue(Path(plan["journal"]).is_file())
        self.assertEqual(json.loads(Path(plan["journal"]).read_text())["phase"], "moving")
        resumed = migration.migrate(self.data, [self.source], self.control, apply=True)
        self.assertEqual(resumed["phase"], "complete")
        snapshot = self.paths()["report_path"].read_bytes()
        again = migration.migrate(self.data, [self.source], self.control, apply=True)
        self.assertEqual(again["phase"], "complete")
        self.assertEqual(self.paths()["report_path"].read_bytes(), snapshot)

    def test_take_with_attempts_and_history_resumes_after_every_directory_move(self):
        _, _, document = self.report(status="FAILED")
        attempts = []
        for bucket, destination in migration.BUCKETS.items():
            attempt = self.source / bucket / "explicit-attempt"
            self.write_json(attempt / "attempt.json", {"input_path": str(self.raw)})
            (attempt / "checkpoint.vdf").write_bytes(bucket.encode())
            attempts.append((destination, attempt.name, bucket.encode()))
        history = self.source / "_resume_history" / "old-run" / "takes" / self.relative / "report.json"
        self.write_json(history, document)
        history_relative = history.relative_to(self.source / "_resume_history")
        plan = self.plan()
        original = migration._rename_no_replace

        def rename_then_interrupt(source, target):
            directory = source.is_dir()
            original(source, target)
            if directory:
                raise RuntimeError("interrupted directory move before journal save")

        # Each replay must recognize already-renamed directories by identity,
        # then advance to the next move without merging or repeating a rename.
        with mock.patch.object(migration, "_rename_no_replace", side_effect=rename_then_interrupt):
            for _ in plan["operations"]:
                with self.assertRaisesRegex(RuntimeError, "interrupted directory move"):
                    migration.migrate(self.data, [self.source], self.control, apply=True)
        resumed = migration.migrate(self.data, [self.source], self.control, apply=True)
        self.assertEqual(resumed["phase"], "complete")
        paths = self.paths()
        self.assertFalse(self.source.exists())
        for destination, name, content in attempts:
            self.assertEqual((paths[destination] / name / "checkpoint.vdf").read_bytes(), content)
        historical = paths["history_root"] / history_relative
        self.assertEqual(json.loads(historical.read_text())["input_path"], str(self.raw))
        snapshot = {str(path): path.read_bytes() for path in paths["take_root"].rglob("*") if path.is_file()}
        again = migration.migrate(self.data, [self.source], self.control, apply=True)
        self.assertEqual(again["phase"], "complete")
        self.assertEqual(snapshot, {str(path): path.read_bytes()
                                    for path in paths["take_root"].rglob("*") if path.is_file()})

    def test_nested_attempt_collision_after_interruption_never_merges_or_overwrites(self):
        report, _, _ = self.report(status="FAILED")
        attempt = self.source / "_failed" / "explicit-attempt"
        self.write_json(attempt / "attempt.json", {"input_path": str(self.raw)})
        checkpoint = attempt / "checkpoint.vdf"
        checkpoint.write_bytes(b"original failed checkpoint")
        plan = self.plan()
        original = migration._rename_no_replace

        def rename_take_then_interrupt(source, target):
            original(source, target)
            if source == report.parent:
                raise RuntimeError("interrupted canonical move")

        with mock.patch.object(migration, "_rename_no_replace", side_effect=rename_take_then_interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted canonical move"):
                migration.apply_migration(plan)
        target = self.paths()["failed_root"] / attempt.name
        target.mkdir(parents=True)
        sentinel = target / checkpoint.name
        sentinel.write_bytes(b"unrelated destination checkpoint")
        with self.assertRaisesRegex(migration.MigrationError, "target collision"):
            migration.migrate(self.data, [self.source], self.control, apply=True)
        self.assertEqual(checkpoint.read_bytes(), b"original failed checkpoint")
        self.assertEqual(sentinel.read_bytes(), b"unrelated destination checkpoint")
        self.assertFalse((target / "attempt.json").exists())

    def test_projected_attempt_collision_refuses_before_any_mutation(self):
        report, _, _ = self.report(status="FAILED")
        attempt = self.source / "_failed" / "explicit-attempt"
        self.write_json(attempt / "attempt.json", {"input_path": str(self.raw)})
        # Both trees would claim this exact directory after the canonical move.
        occupied = report.parent / "_failed" / attempt.name
        occupied.mkdir(parents=True)
        sentinel = occupied / "checkpoint.vdf"
        sentinel.write_bytes(b"canonical tree owns this checkpoint")
        with self.assertRaisesRegex(migration.MigrationError, "Projected target collision"):
            self.plan()
        self.assertEqual(sentinel.read_bytes(), b"canonical tree owns this checkpoint")
        self.assertTrue((attempt / "attempt.json").is_file())
        self.assertFalse(self.control.exists())

    def test_unknown_diagnostic_root_stays_untouched_and_optional_absence_warns(self):
        self.report()
        diagnostic = self.root / "diagnostic"
        report = self.write_json(diagnostic / "report.json", {"processing_source": {"path": "not-an-identity"}})
        before = report.read_bytes()
        missing = self.root / "optional-absent"
        plan = self.plan([self.source, diagnostic], optional_source_outputs=[missing])
        self.assertEqual(plan["untouched_diagnostics"][0]["root"], str(diagnostic))
        migration.apply_migration(plan)
        self.assertEqual(report.read_bytes(), before)
        self.assertTrue(any("Optional source root is absent" in warning for warning in plan["warnings"]))
        self.assertFalse((self.control / diagnostic.name).exists())

    def test_symlink_is_rejected_instead_of_followed(self):
        self.report()
        link = self.source / "linked"
        try:
            link.symlink_to(self.raw)
        except (NotImplementedError, OSError):
            self.skipTest("Creating symlinks is not permitted in this environment")
        with self.assertRaisesRegex(migration.MigrationError, "Symlink/reparse"):
            self.plan()
        self.assertEqual(self.raw.read_bytes(), b"original MCP\x00\xff")

    def test_streaming_lexer_preserves_numeric_diagnostics_and_chunk_boundaries(self):
        path = self.write_json(self.source / "diagnostic.json", {
            "unicode": "Unicode 雪 and escaped \\\" strings", "values": [1e-9, float("nan"), float("inf"), -float("inf")],
            "path": str(self.source / "artifacts" / "some.vdf")})
        original = path.read_bytes()
        output = io.BytesIO()
        with mock.patch.object(migration, "CHUNK", 7):
            digest, _ = migration._json_stream(path, lambda value: value, output)
        self.assertEqual(output.getvalue(), original)
        self.assertEqual(digest, hashlib.sha256(original).hexdigest())

    def test_batch_json_does_not_use_full_document_decoder(self):
        _, _, document = self.report()
        batch = self.write_json(self.source / "batch_report.json", {"jobs": [document] * 1500})
        original_small = migration._small_json

        def bounded_identity_only(path):
            self.assertNotEqual(path, batch, "Batch JSON must use the streaming token path")
            return original_small(path)

        with mock.patch.object(migration, "_small_json", side_effect=bounded_identity_only):
            plan = self.plan()
        self.assertTrue(any(row["source"] == str(batch) for row in plan["json_files"]))

    def test_supplemental_standalone_root_moves_to_diagnostics_with_separate_control(self):
        report, _, _ = self.report()
        diagnostic = self.root / "explicit-diagnostic"
        old = self.write_json(diagnostic / "report.json", {"processing_source": {"source_report": str(report)}})
        paths = self.paths(diagnostic)
        new_control = self.control / diagnostic.name
        supplemental = {"moves": [{"source": str(diagnostic), "target": str(paths["take_root"] / "diagnostics")}],
                        "controls": [{"source": str(diagnostic), "target": str(new_control), "version_key": diagnostic.name}]}
        plan = self.plan([self.source, diagnostic], supplemental=supplemental)
        self.assertFalse(plan["untouched_diagnostics"])
        migration.apply_migration(plan)
        migrated = paths["take_root"] / "diagnostics" / old.name
        self.assertEqual(json.loads(migrated.read_text())["processing_source"]["source_report"], str(self.paths()["report_path"]))
        self.assertIsNotNone(load_layout(new_control))
        self.assertFalse(paths["report_path"].exists(), "A diagnostic must not become a fake canonical report")

    def test_supplemental_mesh_files_and_generated_aggregate_are_journaled(self):
        _, source_fbx, _ = self.report()
        mesh = self.root / "mesh-version"
        output = mesh / "takes" / self.relative / "artifacts" / "native_with_mesh.fbx"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"generated mesh FBX untouched")
        audit = self.write_json(mesh / "_reports" / "native.fbx.json", {"input": str(source_fbx), "output": str(output), "status": "PARTIAL"})
        paths = self.paths(mesh)
        destination = paths["destination"] / output.name
        audit_destination = paths["take_root"] / "_reports" / audit.name
        supplemental = {
            "moves": [{"source": str(output), "target": str(destination)},
                      {"source": str(audit), "target": str(audit_destination)},
                      {"source": str(mesh), "target": str(self.control / mesh.name)}],
            "controls": [{"source": str(mesh), "target": str(self.control / mesh.name), "version_key": mesh.name}],
            "mesh_reports": [{"report_path": str(paths["report_path"]), "relative_path": self.relative.as_posix(),
                              "input_path": str(self.raw), "purpose": "mesh", "tracking_quality_changed": False,
                              "artifacts": {"mesh": str(destination)},
                              "files": [{"input": str(source_fbx), "output": str(destination), "status": "PARTIAL"}]}]}
        plan = migration.apply_migration(self.plan([self.source, mesh], supplemental=supplemental))
        self.assertEqual(destination.read_bytes(), b"generated mesh FBX untouched")
        current = json.loads(paths["report_path"].read_text())
        self.assertNotIn("report_path", current)
        self.assertEqual(current["status"], "MIGRATED_MESH")
        self.assertFalse(current["tracking_quality_changed"])
        self.assertEqual(current["files"][0]["status"], "PARTIAL")
        self.assertEqual(current["files"][0]["input"], str(self.paths()["destination"] / source_fbx.name))
        self.assertEqual(plan["new_json_files"][0]["sha256"], migration._digest(paths["report_path"]))
        migration.migrate(self.data, [self.source, mesh], self.control, supplemental=supplemental, apply=True)

    def test_atomic_rename_never_replaces_existing_file(self):
        source, target = self.root / "a.bin", self.root / "b.bin"
        source.write_bytes(b"source")
        target.write_bytes(b"target")
        with self.assertRaises(OSError):
            migration._rename_no_replace(source, target)
        self.assertEqual(source.read_bytes(), b"source")
        self.assertEqual(target.read_bytes(), b"target")


if __name__ == "__main__":
    unittest.main()

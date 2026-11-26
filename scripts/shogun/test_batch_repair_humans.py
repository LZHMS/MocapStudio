"""Consumer-visible full-scope planning boundaries; no SDK or native processes."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline_layout import initialize_layout, take_paths
from step2_repair_humans import _recheck_saved, main, plan_human_repairs


class BatchHumanPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "processed"
        self.output = self.source / "_repair_runs" / "new-run"
        (self.source / "takes").mkdir(parents=True)

    def save(self, path, report):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")

    def use_local_layout(self):
        raw = self.root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        self.source = raw / "_processing/primary"
        self.output = self.source / "_repair_runs/new-run"
        initialize_layout(self.source, raw, "primary-version")

    def report(self, name="take", *, root=None, status="NEEDS_REVIEW", mtime=100):
        root = root or self.source
        relative = f"group5/HOI/{name}/{name}.mcp"
        raw = self.root / "raw" / relative
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"original MCP")
        paths = take_paths(root, relative)
        vdf = paths["review_destination"] / f"{name}.vdf"
        vdf.parent.mkdir(parents=True, exist_ok=True)
        vdf.write_bytes(b"completed checkpoint")
        value = {"status": status, "job_id": hashlib.sha256(relative.casefold().encode()).hexdigest()[:16],
                 "relative_path": relative, "input_path": str(raw), "input_kind": "mcp",
                 "quality": {"basis": "post_processing", "usable": status == "SUCCESS"},
                 "range": {"animation_start": 1, "animation_end": 1000, "play_start": 1,
                           "play_end": 1000, "frame_count": 1000, "animation_frame_count": 1000},
                 "stages": [{"name": name, "status": "SUCCESS"} for name in
                            ("fix_occlusion", "solve_labeling", "solve_solving")],
                 "artifacts": {"vdf": str(vdf)}, "expected_subjects": ["alice", "bob"],
                 "effective_subjects": ["alice", "bob"], "expected_human_count": 2,
                 "expected_props": [], "effective_props": ["desk"],
                 "dataset_mode": True, "sequence_kind": "HOI", "split_subjects": True,
                 "props_enabled": True, "dataset": {"expected_human_count": 2,
                                                         "detected_humans": ["alice", "bob"]},
                 "subjects": [{"name": name, "rig_class": "HUMAN_RIG", "processing_target": True,
                               "quality": {"usable": True}, "review_reasons": []}
                              for name in ("alice", "bob")],
                 "props": [{"name": "desk", "unresolved_intervals": []}],
                 "policy": {"prop_jitter_max_correction": 0.0}}
        path = paths["report_path"]
        self.save(path, value)
        os.utime(path, ns=(mtime, mtime))
        return path, value

    def plan(self, **kwargs):
        return plan_human_repairs(self.source, self.output, **kwargs)

    def test_full_default_includes_success_and_pure_count_mismatch(self):
        self.report("success", status="SUCCESS")
        path, report = self.report("count-only")
        report["review_reasons"] = ["human_count_mismatch:expected=2:detected=1"]
        report["dataset"]["detected_humans"] = ["alice"]
        self.save(path, report)
        plan = self.plan()
        self.assertEqual(plan["counts"]["selected"], 2)
        self.assertEqual({job["relative_path"].split("/")[-1] for job in plan["jobs"]},
                         {"success.mcp", "count-only.mcp"})

    def test_latest_valid_repair_checkpoint_wins_and_duplicates_remain_audited(self):
        base_path, _ = self.report(mtime=100)
        repair = self.source / "_repair_runs" / "actual-repair"
        new_path, new = self.report(root=repair, mtime=200)
        new["policy"].update(camera_exemptions=["camera_1"], prop_jitter_max_correction=5.0)
        self.save(new_path, new)
        os.utime(new_path, ns=(200, 200))
        plan = self.plan()
        self.assertEqual(plan["jobs"][0]["source_report"], str(new_path))
        self.assertEqual(plan["jobs"][0]["review_policy"]["camera_exemptions"], ["camera_1"])
        self.assertEqual(plan["jobs"][0]["review_policy"]["prop_jitter_max_correction"], 5.0)
        self.assertFalse(plan["jobs"][0]["review_policy"]["auto_trim"])
        self.assertEqual(plan["counts"]["reports_scanned"], 2)
        duplicate = next(row for row in plan["inventory"] if row["source_report"] == str(base_path))
        self.assertEqual(duplicate["preferred_source_report"], str(new_path))
        self.assertNotIn("checkpoint_audit", duplicate)

    def test_invalid_newer_checkpoint_does_not_hide_valid_base(self):
        base, _ = self.report(mtime=100)
        latest, report = self.report(root=self.source / "_repair_runs" / "invalid", mtime=200)
        report["stages"] = []
        self.save(latest, report)
        plan = self.plan()
        self.assertEqual(plan["jobs"][0]["source_report"], str(base))
        self.assertEqual(plan["counts"]["invalid_checkpoint"], 1)
        self.assertIn("successful fill", next(row["reason"] for row in plan["inventory"]
                                               if row["disposition"] == "invalid_checkpoint"))

    def test_latest_failure_does_not_hide_checkpoint_or_disappear(self):
        self.report()
        path, failed = self.report(root=self.source / "_repair_runs" / "failed", status="FAILED")
        failed["error"] = "missing paired X2D"
        self.save(path, failed)
        plan = self.plan()
        self.assertEqual(plan["counts"]["selected"], 1)
        row = next(row for row in plan["inventory"] if row["disposition"] == "source_failed")
        self.assertEqual(row["source_error"], "missing paired X2D")
        self.assertEqual(plan["counts"]["source_failed"], 1)

    def test_failure_only_is_inventory_not_fake_repair(self):
        self.report(status="FAILED")
        plan = self.plan()
        self.assertEqual(plan["jobs"], [])
        self.assertEqual(plan["counts"]["source_failed"], 1)

    def test_unknown_roster_does_not_reduce_original_count_to_detected_actor(self):
        path, report = self.report()
        report["expected_subjects"] = None
        report["effective_subjects"] = ["alice"]
        report["dataset"]["detected_humans"] = ["alice"]
        report["subjects"] = report["subjects"][:1]
        self.save(path, report)
        plan = self.plan()
        job = plan["jobs"][0]
        self.assertIsNone(job["expected_subjects"])
        self.assertEqual(job["expected_human_count"], 2)
        self.assertFalse(job["confirmed_subject_roster"])
        self.assertTrue(job["roster_resolution"]["pending"])
        self.assertEqual(plan["counts"]["roster_pending"], 1)

    def test_manifest_roster_and_explicit_empty_props_take_precedence(self):
        _, report = self.report()
        manifest = {"takes": {report["relative_path"]: ["alice", "bob"]},
                    "expected_props": ["desk"], "prop_takes": {report["relative_path"]: []}}
        job = self.plan(manifest=manifest)["jobs"][0]
        self.assertEqual(job["expected_subjects"], ["alice", "bob"])
        self.assertTrue(job["confirmed_subject_roster"])
        self.assertEqual(job["expected_props"], [])

    def test_source_empty_prop_roster_is_not_replaced_by_detected_props(self):
        self.report()
        self.assertEqual(self.plan()["jobs"][0]["expected_props"], [])

    def test_malformed_reports_are_explicit_inventory_errors(self):
        self.report()
        path = self.source / "takes" / "bad" / "report.json"
        path.parent.mkdir()
        path.write_text("not json", encoding="utf-8")
        plan = self.plan()
        self.assertEqual(plan["counts"]["reports_scanned"], 2)
        self.assertEqual(plan["counts"]["invalid_checkpoint"], 1)
        row = next(row for row in plan["inventory"] if row["source_report"] == str(path))
        self.assertIn("JSONDecodeError", row["reason"])

    def test_wrong_report_shapes_remain_inventory_errors(self):
        path, report = self.report("bad-id")
        report["job_id"] = ["unhashable", "not an identity"]
        self.save(path, report)
        path, report = self.report("bad-stages")
        report["stages"] = None
        self.save(path, report)
        path = self.source / "takes" / "list-report" / "report.json"
        self.save(path, [])
        plan = self.plan()
        self.assertEqual(plan["counts"]["reports_scanned"], 3)
        self.assertEqual(plan["counts"]["invalid_checkpoint"], 3)
        self.assertEqual(plan["jobs"], [])

    def test_forged_identity_and_missing_native_checkpoint_are_not_reusable(self):
        path, report = self.report("forged")
        report["job_id"] = "wrong"
        self.save(path, report)
        _, report = self.report("missing")
        Path(report["artifacts"]["vdf"]).unlink()
        plan = self.plan()
        self.assertEqual(plan["jobs"], [])
        self.assertEqual(plan["counts"]["invalid_checkpoint"], 2)

    def test_identity_conflict_is_blocked_not_arbitrarily_deduplicated(self):
        self.report()
        path, report = self.report(root=self.source / "_repair_runs" / "other")
        alternate = self.root / "other-raw" / report["relative_path"]
        alternate.parent.mkdir(parents=True)
        alternate.write_bytes(b"another input")
        report["input_path"] = str(alternate)
        self.save(path, report)
        plan = self.plan()
        self.assertEqual(plan["jobs"], [])
        self.assertEqual(plan["counts"]["identity_conflict"], 2)

    def test_limit_and_selector_keep_full_inventory(self):
        _, first = self.report("a")
        self.report("b")
        plan = self.plan(limit=1)
        self.assertEqual(plan["counts"]["eligible_identities"], 2)
        self.assertEqual(plan["counts"]["deferred"], 1)
        self.assertEqual(len(plan["inventory"]), 2)
        selected = self.plan(job_id=first["job_id"])
        self.assertEqual(selected["jobs"][0]["relative_path"], first["relative_path"])
        selected = self.plan(relative_path=first["relative_path"].upper().replace("/", "\\"))
        self.assertEqual(len(selected["jobs"]), 1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.plan(job_id="unknown")

    def test_target_output_is_not_rescanned_as_source_on_resume(self):
        self.report()
        self.report("already-produced", root=self.output)
        self.assertEqual(self.plan()["counts"]["reports_scanned"], 1)

    def test_dry_run_has_no_output_writes(self):
        self.report()
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(["--source-output", str(self.source), "--output-dir", str(self.output), "--dry-run"])
        self.assertEqual(result, 0)
        self.assertFalse(self.output.exists())

    def test_invalid_limit_and_worker_timeout_options_fail_before_native_start(self):
        for option, value in (("--limit", "0"), ("--limit", "-1"), ("--max-workers", "0"),
                              ("--max-workers", "-2"), ("--recheck-timeout", "nan"),
                              ("--startup-timeout", "0")):
            with self.subTest(option=option, value=value):
                with self.assertRaises(ValueError):
                    main(["--source-output", str(self.source), "--output-dir", str(self.output),
                          "--dry-run", option, value])

    def test_source_overlap_is_rejected(self):
        for output in (self.source, self.source.parent, self.source / "takes" / "new"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                plan_human_repairs(self.source, output)

    def test_local_inventory_reads_primary_and_nested_controls_and_inherits_latest_layout(self):
        self.use_local_layout()
        base, _ = self.report(mtime=100)
        repair = self.source / "_repair_runs/older-repair"
        initialize_layout(repair, self.root / "raw", "older-version")
        latest, report = self.report(root=repair, mtime=200)
        before = {path: path.read_bytes() for path in (self.root / "raw").rglob("*") if path.is_file()}
        plan = self.plan()
        self.assertEqual(plan["counts"]["reports_scanned"], 2)
        self.assertEqual(len(plan["jobs"]), 1)
        job = plan["jobs"][0]
        self.assertEqual(job["source_output"], str(repair))
        self.assertEqual(job["source_report"], str(latest))
        self.assertEqual(job["source_report_sha256"], hashlib.sha256(latest.read_bytes()).hexdigest())
        relative = Path(report["relative_path"])
        local = self.root / "raw" / relative.parent / "processed/new-run" / relative.name
        self.assertEqual(Path(job["report_path"]), local / "report.json")
        self.assertEqual(Path(job["destination"]), local / "artifacts")
        self.assertEqual(Path(job["log_path"]), local / "logs/processing.log")
        duplicate = next(row for row in plan["inventory"] if row["source_report"] == str(base))
        self.assertEqual(duplicate["preferred_source_report"], str(latest))
        self.assertFalse((self.source / "takes").exists())
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {path: path.read_bytes() for path in
                                 (self.root / "raw").rglob("*") if path.is_file()})

    def test_local_newer_invalid_checkpoint_does_not_hide_primary(self):
        self.use_local_layout()
        base, _ = self.report(mtime=100)
        repair = self.source / "_repair_runs/invalid"
        initialize_layout(repair, self.root / "raw")
        path, report = self.report(root=repair, mtime=200)
        report["artifacts"]["vdf"] = str(self.root / "raw" / report["relative_path"])
        self.save(path, report)
        plan = self.plan()
        self.assertEqual(plan["jobs"][0]["source_report"], str(base))
        self.assertEqual(plan["counts"]["invalid_checkpoint"], 1)

    def test_local_target_control_is_not_included_in_source_inventory(self):
        self.use_local_layout()
        self.report("original")
        initialize_layout(self.output, self.root / "raw")
        self.report("already-produced", root=self.output)
        plan = self.plan()
        self.assertEqual(plan["counts"]["reports_scanned"], 1)
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertTrue(plan["jobs"][0]["relative_path"].endswith("original.mcp"))

    def test_local_dry_run_does_not_write_target_control_or_manifest(self):
        self.use_local_layout()
        self.report()
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch("step2_repair_humans.discover_shogun", side_effect=AssertionError("native discovery forbidden")), \
                contextlib.redirect_stdout(io.StringIO()):
            result = main(["--source-output", str(self.source), "--output-dir", str(self.output), "--dry-run"])
        self.assertEqual(result, 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_active_and_stale_source_control_locks_block_planning_in_both_layouts(self):
        for local in (False, True):
            if local:
                self.use_local_layout()
            self.report()
            repair = self.source / "_repair_runs/locked"
            if local:
                initialize_layout(repair, self.root / "raw")
            else:
                repair.mkdir(parents=True)
            for root in (self.source, repair):
                for pid in (os.getpid(), 2147483647):
                    with self.subTest(local=local, source=root, pid=pid):
                        lock = root / ".pipeline.lock"
                        payload = json.dumps({"pid": pid}).encode("utf-8")
                        lock.write_bytes(payload)
                        try:
                            with self.assertRaisesRegex(RuntimeError, "Source batch has a lock"):
                                self.plan()
                            self.assertEqual(lock.read_bytes(), payload)
                            self.assertFalse(self.output.exists())
                        finally:
                            lock.unlink()

    def test_local_nested_source_lock_blocks_planning(self):
        self.use_local_layout()
        self.report()
        repair = self.source / "_repair_runs/locked"
        initialize_layout(repair, self.root / "raw")
        lock = repair / ".pipeline.lock"
        lock.write_text('{"pid":123}', encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.plan()
        self.assertEqual(lock.read_text(encoding="utf-8"), '{"pid":123}')



class RecheckArtifactCacheTests(unittest.TestCase):
    """Controlled inspector fixture tests observable cache/quality outcomes."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve()
        self.configure_fixture("fixture.mcp")
        self.generation = 0
        self.checker = patch("step2_repair_humans.run_isolated_recheck", self.inspect_fixture)
        self.checker.start()
        self.addCleanup(self.checker.stop)

    def configure_fixture(self, relative):
        self.path = take_paths(self.output, relative)["report_path"]
        self.path.parent.mkdir(parents=True)
        self.artifacts = {}
        for role in ("vdf", "fbx", "vsk", "subject_0_fbx"):
            artifact = self.path.parent / "artifacts" / f"fixture.{role}"
            artifact.parent.mkdir(exist_ok=True)
            artifact.write_bytes(b"usable")
            self.artifacts[role] = str(artifact)
        report = {"status": "SUCCESS", "quality": {"usable": True}, "review_reasons": [],
                  "artifacts": self.artifacts,
                  "subject_exports": [{"subject": "alice", "path": self.artifacts["subject_0_fbx"],
                                       "quality": {"usable": True}}]}
        self.path.write_text(json.dumps(report), encoding="utf-8")
        self.job = {"job_id": "fixture", "relative_path": relative,
                    "report_path": str(self.path), "source_report_sha256": "source-evidence"}

    def configure_local_fixture(self):
        raw = Path(self.temp.name).resolve() / "raw"
        raw.mkdir()
        self.output = raw / "_processing/repair"
        initialize_layout(self.output, raw)
        self.configure_fixture("group1/HOI/sequence/fixture.mcp")

    def inspect_fixture(self, report_path, output, executable, recovery, timeout):
        # Deliberately not a native implementation: test fixture bytes encode
        # usable/rejected data so stale caching has a consumer-visible result.
        report = json.loads(report_path.read_text(encoding="utf-8"))
        usable = all(Path(value).read_bytes() == b"usable"
                     for value in report["artifacts"].values())
        self.generation += 1
        report.update(status="SUCCESS" if usable else "NEEDS_REVIEW",
                      quality={"usable": usable},
                      review_reasons=[] if usable else ["fixture_data_rejected"])
        for entry in report["subject_exports"]:
            entry["quality"] = {"usable": usable}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return {"status": report["status"], "usable": usable, "inspection_generation": self.generation}

    def recheck(self):
        return _recheck_saved(self.job, self.output, Path("unused-native-fixture"), 900)

    def test_unchanged_all_role_bytes_preserve_verified_result_and_report(self):
        first = self.recheck()
        snapshot = self.path.read_bytes()
        second = self.recheck()
        self.assertEqual(first, second)
        self.assertEqual(self.path.read_bytes(), snapshot)
        self.assertEqual(second["result"]["inspection_generation"], 1)
        self.assertEqual(len(second["artifact_sha256"]), len(self.artifacts) + 1)
        subject = second["artifact_sha256"]["subject_export:0"]
        self.assertEqual(subject, second["artifact_sha256"]["artifact:subject_0_fbx"])

    def test_changed_artifact_with_identical_report_cannot_reuse_old_usable_result(self):
        self.recheck()
        snapshot = self.path.read_bytes()
        native = Path(self.artifacts["vdf"])
        original_stat = native.stat()
        native.write_bytes(b"badbad")  # Same length; preserve mtime to require actual content hashing.
        os.utime(native, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        self.assertEqual(self.path.read_bytes(), snapshot)
        state = self.recheck()
        report = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(state["result"]["inspection_generation"], 2)
        self.assertEqual(state["result"]["status"], "NEEDS_REVIEW")
        self.assertFalse(report["quality"]["usable"])
        self.assertIn("fixture_data_rejected", report["review_reasons"])

    def test_changed_subject_export_is_not_hidden_by_unchanged_native_vdf(self):
        self.recheck()
        Path(self.artifacts["subject_0_fbx"]).write_bytes(b"rejected exported data")
        state = self.recheck()
        report = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(state["result"]["inspection_generation"], 2)
        self.assertFalse(report["subject_exports"][0]["quality"]["usable"])
        self.assertFalse(report["quality"]["usable"])

    def test_missing_artifact_invalidates_cache_and_retains_failure_gate(self):
        self.recheck()
        Path(self.artifacts["fbx"]).unlink()
        state = self.recheck()
        report = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "FAILED")
        self.assertNotIn("artifact_sha256", state)
        self.assertFalse(report["quality"]["usable"])
        self.assertFalse(report["subject_exports"][0]["quality"]["usable"])
        self.assertIn("independent_recheck_failed", report["review_reasons"])

    def test_legacy_pass_without_artifact_hashes_requires_new_inspection(self):
        self.recheck()
        state_path = self.path.with_name("independent_recheck.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        del state["artifact_sha256"]
        state_path.write_text(json.dumps(state), encoding="utf-8")
        state = self.recheck()
        self.assertEqual(state["result"]["inspection_generation"], 2)
        self.assertIn("artifact_sha256", state)

    def test_local_independent_cache_is_stored_beside_report_outside_control(self):
        self.configure_local_fixture()
        first = self.recheck()
        report_bytes = self.path.read_bytes()
        second = self.recheck()
        self.assertEqual(first, second)
        self.assertEqual(second["result"]["inspection_generation"], 1)
        self.assertEqual(self.path.read_bytes(), report_bytes)
        self.assertTrue(self.path.with_name("independent_recheck.json").is_file())
        self.assertFalse(self.path.is_relative_to(self.output))
        self.assertFalse((self.output / "independent_recheck.json").exists())

    def test_local_cache_does_not_reuse_changed_artifact_bytes(self):
        self.configure_local_fixture()
        self.recheck()
        native = Path(self.artifacts["vdf"])
        stat = native.stat()
        native.write_bytes(b"badbad")
        os.utime(native, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        state = self.recheck()
        self.assertEqual(state["result"]["inspection_generation"], 2)
        self.assertFalse(state["result"]["usable"])

    def test_local_recheck_cannot_use_another_take_report(self):
        self.configure_local_fixture()
        self.job["relative_path"] = "group1/HOI/sequence/other.mcp"
        with self.assertRaises(ValueError):
            self.recheck()


if __name__ == "__main__":
    unittest.main()

"""Health evidence regressions. Temporary files and injected snapshots only; no SDK/processes."""
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check_health as health
from pipeline_layout import initialize_layout, take_paths


RUN_ID = "20261024T065756Z-test0001"


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve()
        self.lock = {"pid": 100, "run_id": RUN_ID}
        self.write_json(self.output / ".pipeline.lock", self.lock)
        self.scheduler = self.process(100, 10, "python.exe", "2026-10-24T06:57:55+00:00",
                                      f'python.exe "C:/project/step1_process_takes.py" -OutputDir "{self.output}"')
        self.worker = self.process(200, 100, "pythonw.exe", "2026-10-24T08:59:55+00:00", "pythonw.exe -c multiprocessing.spawn")
        self.native = self.process(300, 200, "ShogunPostCL.exe", "2026-10-24T08:59:59+00:00",
                                   "ShogunPostCL.exe -profile profile.xml -controlStreamPort 50001")
        self.ready = self.output / ".work" / "job-live-abc" / ".control" / "session-ready.json"
        self.write_json(self.ready, {"pid": 300, "port": 50001, "version": "test", "started_at": "2026-10-24T09:00:00+00:00"})

    @staticmethod
    def process(pid, ppid, name, created, command):
        return {"pid": pid, "ppid": ppid, "name": name, "created_at": created,
                "command_line": command, "cpu_seconds": 10.0, "read_bytes": 100,
                "write_bytes": 200, "working_set_bytes": 1024}

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def snapshots(self, cpu_delta=0, io_delta=0, processes=None):
        rows = deepcopy([self.scheduler, self.worker, self.native] if processes is None else processes)
        first = {"status": "ok", "sampled_at": "2026-10-24T10:00:00+00:00", "processes": rows, "memory": None}
        second = deepcopy(first)
        second["sampled_at"] = "2026-10-24T10:00:03+00:00"
        for row in second["processes"]:
            if row["pid"] == 300:
                row["cpu_seconds"] += cpu_delta
                row["read_bytes"] += io_delta
        return first, second

    def collect(self, snapshots=None, during_sleep=None, **options):
        values = iter(snapshots or self.snapshots())

        def provider(pids):
            return deepcopy(next(values))


        def sleeper(seconds):
            if during_sleep:
                during_sleep()

        result = health.collect_health(self.output, snapshot_provider=provider, sleeper=sleeper, **options)
        return result

    def report(self, name="take", status="SUCCESS", job_id="job-one", error=None):
        path = self.output / "takes" / name / "report.json"
        products = {}
        for role in ("vdf", "fbx", "vsk"):
            artifact = path.parent / "artifacts" / f"take.{role}"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"test artifact")
            products[role] = "artifacts/" + artifact.name
        data = {"job_id": job_id, "input_path": "source/take.mcp", "input_kind": "mcp", "status": status,
                "artifacts": products, "quality": {"basis": "post_processing", "usable": status == "SUCCESS"},
                "subjects": [{"name": "person", "rig_class": "HUMAN_RIG", "processing_target": True,
                              "quality": {"basis": "post_processing", "usable": status != "FAILED"}, "review_reasons": []}],
                "props": [], "subject_exports": [], "review_reasons": []}
        if error:
            data["error"] = error
        self.write_json(path, data)
        return path, data

    def final_batch(self, reports):
        counts = {status: sum(report["status"] == status for report in reports) for status in health.STATUSES}
        self.write_json(self.output / "batch_report.json", {"run_id": RUN_ID, "counts": counts, "jobs": reports})

    def local_report(self, relative="group1/HOI/sequence/take.mcp"):
        raw = self.output / "raw"
        raw.mkdir(exist_ok=True)
        self.output = self.output / "_processing" / "run-v1"
        initialize_layout(self.output, raw, "run-v1")
        old, data = self.report()
        paths = take_paths(self.output, relative)
        paths["take_root"].parent.mkdir(parents=True, exist_ok=True)
        old.parent.rename(paths["take_root"])
        data["relative_path"] = relative
        self.write_json(paths["report_path"], data)
        return paths, data

    def test_local_canonical_reports_exclude_other_versions_and_diagnostics(self):
        paths, data = self.local_report()
        self.write_json(paths["destination"] / "report.json", data)
        self.write_json(paths["recheck_root"] / "attempt" / "report.json", data)
        neighbor = paths["take_root"].parent.parent / "run-v2" / paths["take_root"].name / "report.json"
        self.write_json(neighbor, data)
        before = {path: path.read_bytes() for path in Path(self.temporary.name).rglob("*") if path.is_file()}
        inventory, jobs = health.report_inventory(self.output)
        self.assertEqual(inventory["canonical_files"], 1)
        self.assertEqual(inventory["verified_counts"]["SUCCESS"], 1)
        self.assertEqual(jobs["job-one"]["path"], str(paths["report_path"]))
        self.assertEqual(before, {path: path.read_bytes() for path in Path(self.temporary.name).rglob("*") if path.is_file()})

    def test_local_artifacts_cannot_escape_to_raw_neighbor_or_other_version(self):
        paths, data = self.local_report()
        for target in (
            Path(self.temporary.name) / "raw" / "private.fbx",
            paths["take_root"].parent / "neighbor.mcp" / "artifacts" / "scene.fbx",
            paths["take_root"].parent.parent / "run-v2" / "take.mcp" / "artifacts" / "scene.fbx",
        ):
            with self.subTest(target=target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"not owned by this take")
                data["artifacts"]["fbx"] = str(target)
                self.write_json(paths["report_path"], data)
                inventory, _ = health.report_inventory(self.output)
                self.assertEqual(inventory["verified_counts"]["SUCCESS"], 0)
                self.assertTrue(any("artifact roots" in row["reason"] for row in inventory["artifact_issues"]))

    def test_local_sessions_use_only_exact_version_workspaces(self):
        paths, _ = self.local_report()
        ready = {"pid": 300, "port": 50001, "started_at": "2026-10-24T09:00:00+00:00"}
        owned = paths["work_root"] / "job-live-abc" / ".control" / "session-ready.json"
        self.write_json(owned, ready)
        self.write_json(paths["take_root"].parent.parent / "run-v2" / "take.mcp" /
                        ".work" / "wrong-version" / ".control" / "session-ready.json", ready)
        sessions = health.scan_sessions(self.output)
        self.assertEqual([row["path"] for row in sessions["entries"]], [str(owned)])
        self.assertEqual(sessions["errors"], [])

    def test_local_report_identity_must_match_canonical_raw_relative(self):
        paths, data = self.local_report()
        data["relative_path"] = "group1/HOI/sequence/neighbor.mcp"
        self.write_json(paths["report_path"], data)
        inventory, jobs = health.report_inventory(self.output)
        self.assertEqual(jobs, {})
        self.assertEqual(inventory["unknown_count"], 1)
        self.assertIn("identity", inventory["issues"][0])

    def test_raw_inventory_excludes_generated_processing_subtrees(self):
        raw = self.output / "raw"
        for relative in ("group1/MP/sequence/take.mcp",
                         "group1/MP/sequence/PrOcEsSeD/version/take.mcp/.work/copy.mcp",
                         "_PrOcEsSiNg/run/staged.mcp"):
            target = raw / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"inventory fixture only")
        result = self.collect(data_path=raw)
        self.assertEqual(result["source_inventory"]["mcp_files"], 1)

    def test_positive_native_cpu_delta_is_running_with_age_and_identity(self):
        result = self.collect(self.snapshots(cpu_delta=1.25))
        self.assertEqual((result["overall"], result["exit_code"]), ("RUNNING", 0))
        self.assertEqual(result["lock"]["second"]["identity"], "OWNED")
        self.assertEqual(result["sessions"]["owned_active_count"], 1)
        native = result["sessions"]["owned_active"][0]
        self.assertEqual((native["native_pid"], native["worker_pid"]), (300, 200))
        self.assertEqual(native["cpu_seconds_delta"], 1.25)
        self.assertEqual(native["age_seconds"], 3603)
        self.assertNotEqual(result["sampling"]["first_sample_at"], result["sampling"]["second_sample_at"])

    def test_review_repair_scheduler_keeps_output_and_process_ownership_checks(self):
        self.scheduler["command_line"] = f'python.exe "C:/project/step3_repair_reviews.py" -OutputDir "{self.output}"'
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["overall"], "RUNNING")
        self.assertEqual(result["lock"]["second"]["identity"], "OWNED")
        self.assertEqual(result["sessions"]["owned_active_count"], 1)

        self.scheduler["command_line"] = f'python.exe "C:/project/step3_repair_reviews.py" -OutputDir "{self.output / "other"}"'
        wrong_output = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(wrong_output["lock"]["second"]["identity"], "MISMATCH")
        self.assertEqual(wrong_output["sessions"]["owned_active_count"], 0)

    def test_human_repair_scheduler_requires_matching_long_output_option(self):
        self.scheduler["command_line"] = f'python.exe "C:/project/step2_repair_humans.py" --output-dir "{self.output}"'
        observed = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(observed["overall"], "RUNNING")
        self.assertEqual(observed["lock"]["second"]["identity"], "OWNED")
        self.assertEqual(observed["sessions"]["owned_active_count"], 1)
        self.scheduler["command_line"] = f'python.exe "C:/project/step2_repair_humans.py" --output-dir "{self.output / "other"}"'
        wrong_output = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(wrong_output["lock"]["second"]["identity"], "MISMATCH")
        self.assertEqual(wrong_output["sessions"]["owned_active_count"], 0)

    def test_zero_cpu_and_quiet_log_are_unknown_not_failed_or_deadlocked(self):
        (self.output / "production.log").write_text("old quiet log\n", encoding="utf-8")
        result = self.collect()
        self.assertEqual((result["overall"], result["exit_code"]), ("UNKNOWN", 2))
        self.assertEqual(result["sessions"]["owned_active"][0]["cpu_seconds_delta"], 0)

    def test_scheduler_polling_does_not_prove_native_progress(self):
        snapshots = self.snapshots()
        next(row for row in snapshots[1]["processes"] if row["pid"] == 100)["cpu_seconds"] += 0.1
        result = self.collect(snapshots)
        self.assertEqual((result["overall"], result["exit_code"]), ("UNKNOWN", 2))
        self.assertEqual(result["sessions"]["owned_active_count"], 1)
        self.assertFalse(result["sessions"]["owned_active"][0]["observed_activity"])

    def test_positive_io_is_observed_activity_without_cpu_growth(self):
        result = self.collect(self.snapshots(io_delta=50))
        self.assertEqual(result["overall"], "RUNNING")
        self.assertEqual(result["sessions"]["owned_active"][0]["read_bytes_delta"], 50)

    def test_historical_failures_are_stock_not_new_events_even_with_since(self):
        path, _ = self.report(status="FAILED", error="Worker exited without a result report")
        os.utime(path, (1000, 1000))
        result = self.collect(self.snapshots(cpu_delta=1), since=datetime(2026, 10, 24, tzinfo=timezone.utc))
        self.assertEqual(result["overall"], "RUNNING")
        reports = result["reports"]
        self.assertEqual(reports["status_counts"]["FAILED"], 1)
        self.assertEqual(reports["failure_classes"]["worker_exited_without_result"], 1)
        self.assertEqual(reports["newer_report_counts"]["FAILED"], 0)
        self.assertIsNone(reports["new_failure_events"])
        self.assertFalse(reports["failures"][0]["event_time_known"])
        self.assertTrue(result["warnings"])

    def test_failure_categories_preserve_missing_x2d_and_other(self):
        self.report("missing", "FAILED", "missing", "Missing X2D sidecar for reconstruction")
        self.report("other", "FAILED", "other", "Licence unavailable")
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["reports"]["failure_classes"], {"missing_x2d": 1, "worker_exited_without_result": 0, "other": 1})
        self.assertEqual(result["overall"], "RUNNING")

    def test_stale_workspaces_are_excluded_not_counted_as_active(self):
        self.write_json(self.output / ".work" / "old-job-dead" / ".control" / "session-ready.json",
                        {"pid": 999, "port": 50002, "started_at": "2026-10-20T09:00:00+00:00"})
        (self.output / ".work" / "old-unready").mkdir()
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["sessions"]["owned_active_count"], 1)
        self.assertEqual(result["sessions"]["stale_count"], 1)
        self.assertEqual(result["sessions"]["unknown_count"], 0)

    def test_reused_native_pid_is_not_owned(self):
        native = deepcopy(self.native)
        native["created_at"] = "2026-10-24T09:30:00+00:00"
        result = self.collect(self.snapshots(cpu_delta=9, processes=[self.scheduler, self.worker, native]))
        self.assertEqual(result["sessions"]["owned_active_count"], 0)
        self.assertEqual(result["sessions"]["stale_count"], 1)
        self.assertEqual(result["overall"], "UNKNOWN")

    def test_unrelated_process_reusing_native_pid_is_not_owned(self):
        native = deepcopy(self.native)
        native["name"] = "unrelated.exe"
        result = self.collect(self.snapshots(cpu_delta=9, processes=[self.scheduler, self.worker, native]))
        self.assertEqual(result["sessions"]["owned_active_count"], 0)
        self.assertEqual(result["sessions"]["observations"][0]["ownership"], "STALE")

    def test_worker_parent_chain_cannot_attribute_another_scheduler(self):
        worker = deepcopy(self.worker)
        worker["ppid"] = 999
        result = self.collect(self.snapshots(cpu_delta=9, processes=[self.scheduler, worker, self.native]))
        self.assertEqual(result["sessions"]["owned_active_count"], 0)
        self.assertEqual(result["overall"], "UNKNOWN")

    def test_reused_worker_pid_cannot_validate_old_native_parent_chain(self):
        worker = deepcopy(self.worker)
        worker["created_at"] = "2026-10-24T09:30:00+00:00"
        result = self.collect(self.snapshots(cpu_delta=9, processes=[self.scheduler, worker, self.native]))
        self.assertEqual(result["sessions"]["owned_active_count"], 0)

    def test_reused_lock_pid_is_a_confirmed_identity_error(self):
        scheduler = deepcopy(self.scheduler)
        scheduler["created_at"] = "2026-10-24T08:00:00+00:00"
        result = self.collect(self.snapshots(cpu_delta=9, processes=[scheduler, self.worker, self.native]))
        self.assertEqual((result["overall"], result["exit_code"]), ("CONFIRMED_ERROR", 1))
        self.assertEqual(result["lock"]["second"]["identity"], "MISMATCH")
        self.assertEqual(result["sessions"]["owned_active_count"], 0)

    def test_dead_unchanged_lock_is_confirmed_error(self):
        result = self.collect(self.snapshots(processes=[]))
        self.assertEqual((result["overall"], result["exit_code"]), ("CONFIRMED_ERROR", 1))
        self.assertEqual(result["lock"]["second"]["identity"], "DEAD")

    def test_lock_transition_during_sample_is_unknown_not_failure(self):
        result = self.collect(self.snapshots(processes=[]), during_sleep=lambda: (self.output / ".pipeline.lock").unlink())
        self.assertEqual((result["overall"], result["exit_code"]), ("UNKNOWN", 2))
        self.assertFalse(result["lock"]["stable_during_sample"])

    def test_relative_scheduler_output_is_unknown_not_wrong_owner(self):
        scheduler = deepcopy(self.scheduler)
        scheduler["command_line"] = 'python.exe "C:/project/step1_process_takes.py" -OutputDir ".\\out"'
        result = self.collect(self.snapshots(cpu_delta=9, processes=[scheduler, self.worker, self.native]))
        self.assertEqual(result["lock"]["second"]["identity"], "UNKNOWN")
        self.assertEqual((result["overall"], result["exit_code"]), ("UNKNOWN", 2))

    def test_missing_command_permission_does_not_assume_ownership(self):
        scheduler = deepcopy(self.scheduler)
        scheduler["command_line"] = None
        result = self.collect(self.snapshots(cpu_delta=9, processes=[scheduler, self.worker, self.native]))
        self.assertEqual(result["lock"]["second"]["identity"], "UNKNOWN")
        self.assertEqual(result["overall"], "UNKNOWN")
        self.assertEqual(result["sessions"]["owned_active_count"], 0)
        self.assertEqual(result["sessions"]["unknown_count"], 1)

    def test_process_query_failure_is_unknown_not_zero_active(self):
        unknown = {"status": "unknown", "sampled_at": "2026-10-24T10:00:00+00:00",
                   "processes": [], "error": "access denied", "memory": None}
        result = self.collect((unknown, unknown))
        self.assertEqual((result["overall"], result["exit_code"]), ("UNKNOWN", 2))
        self.assertIsNone(result["sessions"]["owned_active_count"])
        self.assertEqual(result["sessions"]["native_process_counts"], [None, None])

    def test_prop_review_does_not_become_program_failure_or_human_failure(self):
        path, data = self.report(status="NEEDS_REVIEW")
        data["review_reasons"] = ["prop:tray:unresolved_rigid_frames"]
        data["props"] = [{"name": "tray", "review_reasons": ["unresolved_rigid_frames"]}]
        data["subjects"].append({"name": "tray", "rig_class": "PROP_OR_NONHUMAN", "processing_target": False,
                                 "quality": {"basis": "post_processing", "usable": False}, "review_reasons": []})
        data["subject_exports"] = [{"subject": "person", "artifact_role": "fbx", "quality": {
            "basis": "post_processing", "usable": False, "review_reasons": data["review_reasons"]}}]
        self.write_json(path, data)
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual((result["overall"], result["exit_code"]), ("RUNNING", 0))
        self.assertEqual(result["reports"]["review_categories"], {"human": 0, "prop": 1, "roster": 0, "other": 0})
        self.assertEqual(result["reports"]["human_quality"], {"usable": 1})
        self.assertEqual(result["reports"]["subject_export_quality"], {"review": 1})
        self.assertEqual(result["reports"]["verified_counts"]["NEEDS_REVIEW"], 1)

    def test_roster_and_human_quality_categories_are_separate(self):
        path, data = self.report(status="NEEDS_REVIEW")
        data["review_reasons"] = ["human_count_mismatch:expected=2:detected=1", "person:missing_core_markers"]
        data["subjects"][0]["quality"]["usable"] = False
        self.write_json(path, data)
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["reports"]["review_categories"], {"human": 1, "prop": 0, "roster": 1, "other": 0})
        self.assertEqual(result["overall"], "RUNNING")

    def test_missing_current_reference_is_not_rescued_by_old_artifact(self):
        path, data = self.report()
        old = self.output / "_needs_review" / "take" / "old.fbx"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"old artifact must not count")
        data["artifacts"]["fbx"] = str(path.parent / "artifacts" / "missing-current.fbx")
        self.write_json(path, data)
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual((result["overall"], result["exit_code"]), ("UNKNOWN", 2))
        self.assertEqual(result["reports"]["status_counts"]["SUCCESS"], 1)
        self.assertEqual(result["reports"]["verified_counts"]["SUCCESS"], 0)
        self.assertEqual(result["reports"]["missing_artifact_count"], 1)
        self.assertEqual(result["reports"]["artifact_issues"][0]["path"], data["artifacts"]["fbx"])

    def test_empty_artifact_and_missing_required_role_are_listed(self):
        path, data = self.report()
        (path.parent / data["artifacts"]["fbx"]).write_bytes(b"")
        del data["artifacts"]["vsk"]
        self.write_json(path, data)
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["reports"]["missing_artifact_count"], 2)
        self.assertEqual(result["reports"]["verified_counts"]["SUCCESS"], 0)

    def test_malformed_report_is_unknown_not_an_empty_successful_inventory(self):
        path = self.output / "takes" / "broken" / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"job_id":', encoding="utf-8")
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["overall"], "UNKNOWN")
        self.assertEqual(result["reports"]["unknown_count"], 1)
        self.assertEqual(result["reports"]["verified_counts"]["SUCCESS"], 0)
        self.assertTrue(result["reports"]["issues"])

    def test_unreadable_report_does_not_become_success(self):
        path, _ = self.report()
        original_open = Path.open

        def guarded_open(target, *args, **kwargs):
            if target == path:
                raise PermissionError("report access denied")
            return original_open(target, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["overall"], "UNKNOWN")
        self.assertEqual(result["reports"]["unknown_count"], 1)

    def test_report_replaced_during_read_is_unknown(self):
        path, _ = self.report()
        info = path.stat()
        first = SimpleNamespace(st_size=info.st_size, st_mtime_ns=info.st_mtime_ns, st_mtime=info.st_mtime)
        changed = SimpleNamespace(st_size=info.st_size + 1, st_mtime_ns=info.st_mtime_ns + 1, st_mtime=info.st_mtime)
        with patch.object(Path, "stat", side_effect=[first, changed]):
            data, modified, error = health.read_json(path)
        self.assertIsNone(data)
        self.assertIsNone(modified)

    def test_old_quality_basis_cannot_count_as_verified_success(self):
        path, data = self.report()
        data["quality"]["basis"] = "before_processing"
        self.write_json(path, data)
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["overall"], "UNKNOWN")
        self.assertEqual(result["reports"]["verified_counts"]["SUCCESS"], 0)

    def test_duplicate_job_ids_are_deduplicated_but_ambiguous(self):
        older, _ = self.report("old", "FAILED", "duplicate", "Missing X2D")
        newer, _ = self.report("new", "SUCCESS", "duplicate")
        os.utime(older, (1000, 1000))
        os.utime(newer, (2000, 2000))
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["reports"]["canonical_files"], 2)
        self.assertEqual(result["reports"]["unique_jobs"], 1)
        self.assertEqual(result["reports"]["status_counts"], {"SUCCESS": 1, "NEEDS_REVIEW": 0, "FAILED": 0})
        self.assertEqual(result["overall"], "UNKNOWN")

    def test_no_lock_and_old_reports_do_not_prove_completion(self):
        self.report()
        (self.output / ".pipeline.lock").unlink()
        result = self.collect(self.snapshots(processes=[]))
        self.assertEqual(result["overall"], "UNKNOWN")

    def test_complete_summary_without_lock_or_native_process_can_complete(self):
        _, data = self.report(status="NEEDS_REVIEW")
        self.final_batch([data])
        (self.output / ".pipeline.lock").unlink()
        result = self.collect(self.snapshots(processes=[]))
        self.assertEqual((result["overall"], result["exit_code"]), ("COMPLETED", 0))
        self.assertTrue(result["final_summary"]["complete"])

    def test_old_final_summary_with_live_unlocked_scheduler_cannot_complete(self):
        _, data = self.report()
        self.final_batch([data])
        (self.output / ".pipeline.lock").unlink()
        result = self.collect(self.snapshots(processes=[self.scheduler]))
        self.assertEqual(result["overall"], "UNKNOWN")
        self.assertEqual(result["sessions"]["live_scheduler_candidate_pids"], [[100], [100]])

    def test_active_unowned_native_process_blocks_completion(self):
        _, data = self.report()
        self.final_batch([data])
        (self.output / ".pipeline.lock").unlink()
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["overall"], "UNKNOWN")
        self.assertEqual(result["sessions"]["owned_active_count"], 0)

    def test_summary_count_mismatch_cannot_complete(self):
        _, data = self.report()
        self.final_batch([data])
        (self.output / ".pipeline.lock").unlink()
        summary = self.output / "batch_report.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        data["counts"]["SUCCESS"] = 2
        self.write_json(summary, data)
        result = self.collect(self.snapshots(processes=[]))
        self.assertEqual(result["overall"], "UNKNOWN")
        self.assertFalse(result["final_summary"]["complete"])

    def test_newer_report_makes_old_final_summary_insufficient(self):
        path, data = self.report()
        self.final_batch([data])
        (self.output / ".pipeline.lock").unlink()
        os.utime(self.output / "batch_report.json", (1000, 1000))
        os.utime(path, (2000, 2000))
        result = self.collect(self.snapshots(processes=[]))
        self.assertEqual(result["overall"], "UNKNOWN")
        self.assertFalse(result["final_summary"]["complete"])

    def test_launch_concurrency_requires_matching_lock_identity(self):
        launch = self.output / "production_launch.json"
        self.write_json(launch, {"scheduler_pid": 100, "run_id": RUN_ID, "max_workers": 7})
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertEqual(result["concurrency"]["max_workers"], 7)
        self.write_json(launch, {"scheduler_pid": 999, "run_id": RUN_ID, "max_workers": 99})
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertIsNone(result["concurrency"])

    def test_log_read_is_bounded_and_only_default_production_log_is_used(self):
        log = self.output / "production.log"
        log.write_bytes(b"[FAILED] too-old\n" + b"x" * (health.TAIL_BYTES + 20) + b"\n[RUN] worker=200 port=50001 source/take.mcp\n[SUCCESS] recent\n")
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertTrue(result["log"]["truncated"])
        self.assertEqual(len(result["log"]["recent_events"]), 2)
        self.assertEqual(result["log"]["worker_hints"][0]["worker_pid"], 200)
        log.unlink()
        (self.output / "shogun.log").write_text("[FAILED] not the scheduler log\n", encoding="utf-8")
        result = self.collect(self.snapshots(cpu_delta=1))
        self.assertIsNone(result["log"]["path"])

    def test_explicit_data_inventory_is_not_pending_count(self):
        source = self.output / "source"
        source.mkdir()
        for name in ("one.mcp", "TWO.MCP", "ignore.mcp.bak", "ignore.vdf"):
            (source / name).write_bytes(b"")
        result = self.collect(self.snapshots(cpu_delta=1), data_path=source)
        self.assertEqual(result["source_inventory"]["mcp_files"], 2)
        self.assertTrue(result["source_inventory"]["complete"])
        self.assertNotIn("pending", result)

    def test_checker_does_not_change_output_files(self):
        self.report()
        before = {str(path.relative_to(self.output)): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.output.rglob("*") if path.is_file()}
        self.collect(self.snapshots(cpu_delta=1))
        after = {str(path.relative_to(self.output)): (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(after, before)

    def test_cli_entry_error_is_json_and_code_one_without_sampling(self):
        stream = io.StringIO()
        with patch("sys.stdout", stream):
            code = health.main(["--output-dir", str(self.output), "--sample-seconds", "nan", "--json"])
        result = json.loads(stream.getvalue())
        self.assertEqual((code, result["overall"]), (1, "ENTRY_ERROR"))

    def test_iso_since_requires_timezone_to_avoid_local_time_guess(self):
        self.assertIsNone(health.parse_time("2026-10-24T12:00:00"))
        self.assertEqual(health.parse_time("2026-10-24T14:00:00+02:00"), datetime(2026, 10, 24, 12, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()

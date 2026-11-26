"""Dashboard consumer regressions: temporary files/injected snapshots, no SDK or sockets."""
from copy import deepcopy
from email.message import Message
import io
import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Same bootstrap as dashboard.py/dashboard_data.py: the repo pipeline modules
# resolve at import time, so sys.path must be prepared before importing them.
DEFAULT_REPO_SCRIPTS = r"F:\projects\4DHOI\SceneDesign\4dhoi\scripts\shogun"


def ensure_repo_scripts_path():
    """Insert the Shogun pipeline scripts directory into sys.path at most once."""
    scripts = Path(os.environ.get("MOCAP_REPO_SCRIPTS") or DEFAULT_REPO_SCRIPTS)
    if not scripts.is_dir():
        return
    resolved = str(scripts.resolve())
    normalized = os.path.normcase(resolved)
    if not any(os.path.normcase(str(entry)) == normalized for entry in sys.path):
        sys.path.insert(0, resolved)


ensure_repo_scripts_path()

import check_health as health  # noqa: E402  (needs the sys.path bootstrap above)
import dashboard  # noqa: E402  (imported after the bootstrap for the same reason)
from dashboard_data import DashboardData, SnapshotCache, TAIL_BYTES, layout, open_scoped, shared_sessions  # noqa: E402
from dashboard_runs import MultiRunCache, RunCatalog, output_id
import step1_process_takes  # noqa: E402  (needs the sys.path bootstrap above)


RUN_ID = "20261024T065756Z-test0001"


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "output"
        self.source = self.root / "input"
        self.static = self.root / "static"
        self.output.mkdir()
        self.source.mkdir()
        self.static.mkdir()
        (self.static / "index.html").write_text("<h1>dashboard</h1>", encoding="utf-8")
        self.scheduler = self.process(100, 10, "python.exe", "2026-10-24T06:57:55+00:00",
                                      f'python.exe "C:/project/step1_process_takes.py" -OutputDir "{self.output}"')
        self.worker = self.process(200, 100, "pythonw.exe", "2026-10-24T08:59:55+00:00", "pythonw.exe -c multiprocessing.spawn")
        self.native = self.process(300, 200, "ShogunPostCL.exe", "2026-10-24T08:59:59+00:00",
                                  "ShogunPostCL.exe -controlStreamPort 50001")

    @staticmethod
    def process(pid, ppid, name, created_at, command_line):
        return {"pid": pid, "ppid": ppid, "name": name, "created_at": created_at, "command_line": command_line,
                "cpu_seconds": 10, "read_bytes": 100, "write_bytes": 200, "working_set_bytes": 4096}

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def source_job(self, name="take.mcp", kind="MP"):
        path = self.source / "group1" / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")
        path.with_suffix(".x2d").write_bytes(b"raw")
        jobs = step1_process_takes.build_jobs(self.source, self.output, self.output / "logs", "mcp",
                                             step1_process_takes.load_manifest(None), None, None, True,
                                             dataset_layout=layout.load_layout(self.output) is not None)
        job = next(job for job in jobs if job["input_path"] == str(path))
        job.setdefault("sequence_kind", step1_process_takes.dataset_sequence(Path(job["relative_path"]))[1])
        return job

    def report(self, job=None, status="SUCCESS"):
        job = job or self.source_job()
        path = Path(job["report_path"])
        products = {}
        for role in ("vdf", "fbx", "vsk"):
            product = path.parent / "artifacts" / f"take.{role}"
            product.parent.mkdir(parents=True, exist_ok=True)
            product.write_bytes(b"0123456789")
            products[role] = str(product)
        data = {"job_id": job["job_id"], "status": status, "input_path": job["input_path"], "input_kind": "mcp",
                "relative_path": job["relative_path"], "sequence_kind": job["sequence_kind"],
                "quality": {"basis": "post_processing", "usable": status == "SUCCESS"},
                "subjects": [{"name": "actor", "processing_target": True, "rig_class": "HUMAN_RIG",
                              "quality": {"basis": "post_processing", "usable": status == "SUCCESS"}}],
                "props": [], "stages": [{"name": "solve_solving", "status": "SUCCESS", "elapsed_seconds": 4}],
                "review_reasons": ["prop:chair:native_pose_incomplete"] if status == "NEEDS_REVIEW" else [],
                "artifacts": products, "range": {"frame_count": 100}}
        if status == "FAILED":
            data["error"] = "Worker exited without a result report"
        self.write_json(path, data)
        return path, data

    def active(self, job):
        self.write_json(self.output / ".pipeline.lock", {"pid": 100, "run_id": RUN_ID})
        self.write_json(self.output / ".work" / (job["job_id"] + "-workspace") / ".control" / "session-ready.json",
                        {"pid": 300, "port": 50001, "started_at": "2026-10-24T09:00:00+00:00"})

    def snapshots(self, native_cpu=0, worker_cpu=0, *, processes=None):
        rows = deepcopy(processes if processes is not None else [])
        first = {"status": "ok", "sampled_at": "2026-10-24T10:00:00+00:00", "processes": rows,
                 "memory": {"free_bytes": 1024, "total_bytes": 4096}}
        second = deepcopy(first)
        second["sampled_at"] = "2026-10-24T10:00:03+00:00"
        for row in second["processes"]:
            row["cpu_seconds"] += native_cpu if row["pid"] == 300 else worker_cpu if row["pid"] == 200 else 0
        return first, second

    def data(self, snapshots=None, source=True):
        iterator = iter(snapshots or self.snapshots())
        return DashboardData(self.output, self.source if source else None,
                             snapshot_provider=lambda pids: deepcopy(next(iterator)), sleeper=lambda seconds: None)

    def cache(self, data, snapshot):
        cache = SnapshotCache(data)
        cache._snapshot = snapshot
        return cache

    def request(self, cache, path, method="GET", headers=None):
        # Exercise the real HTTP routing/headers/body with no listening socket.
        handler = object.__new__(dashboard.DashboardHandler)
        handler.server = SimpleNamespace(cache=cache, static_root=self.static, server_address=("127.0.0.1", 8787))
        handler.path, handler.command = path, method
        handler.request_version, handler.requestline = "HTTP/1.1", f"{method} {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 10000)
        handler.wfile = io.BytesIO()
        handler.log_message = lambda *args: None
        handler.headers = Message()
        for key, value in ({"Host": "127.0.0.1:8787"} | (headers or {})).items():
            handler.headers[key] = value
        handler.close_connection = False
        getattr(handler, "do_" + method)()
        raw_headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        lines = raw_headers.decode("latin-1").split("\r\n")
        return int(lines[0].split()[1]), dict(line.split(": ", 1) for line in lines[1:]), body

    def test_old_failed_overridden_by_active_with_disjoint_counts_and_coverage(self):
        job = self.source_job()
        self.report(job, "FAILED")
        self.active(job)
        data = self.data(self.snapshots(native_cpu=2, processes=[self.scheduler, self.worker, self.native]))
        snapshot = data.collect()
        overview = snapshot["overview"]
        row = snapshot["rows"][job["job_id"]]
        self.assertEqual((row["status"], row["reported_status"]), ("RUNNING", "FAILED"))
        self.assertEqual((overview["counts"]["FAILED"], overview["counts"]["RUNNING"]), (0, 1))
        self.assertEqual((overview["coverage"]["terminal_count"], overview["coverage"]["ratio"]), (0, 0))
        self.assertEqual(overview["health"]["overall"], "RUNNING")
        detail = data.detail(snapshot, job["job_id"])
        self.assertEqual(detail["error"], "Worker exited without a result report")

    def test_python_only_progress_is_not_misreported_as_stall(self):
        job = self.source_job()
        self.active(job)
        data = self.data(self.snapshots(worker_cpu=3.5, processes=[self.scheduler, self.worker, self.native]))
        snapshot = data.collect()
        activity = snapshot["rows"][job["job_id"]]["active"]
        self.assertEqual(activity["phase"], "PYTHON_COMPUTING")
        self.assertEqual((activity["native_cpu_delta_seconds"], activity["worker_cpu_delta_seconds"]), (0, 3.5))
        runtime = snapshot["overview"]["health"]
        self.assertEqual((runtime["overall"], runtime["active_progress_count"], runtime["python_busy_count"], runtime["native_busy_count"]),
                         ("RUNNING", 1, 1, 0))
        self.assertEqual(data.detail(snapshot, job["job_id"])["stages"], [])

    def test_scheduler_polling_alone_is_not_job_progress(self):
        job = self.source_job()
        self.active(job)
        snapshots = self.snapshots(processes=[self.scheduler, self.worker, self.native])
        snapshots[1]["processes"][0]["cpu_seconds"] += 2
        overview = self.data(snapshots).collect()["overview"]
        self.assertEqual(overview["health"]["overall"], "UNKNOWN")
        self.assertEqual(overview["takes"][0]["active"]["phase"], "NO_OBSERVED_ACTIVITY")

    def test_unreported_is_inventory_evidence_not_pending(self):
        first = self.source_job("first.mcp")
        self.source_job("second.mcp", "HOI")
        self.report(first, "FAILED")
        snapshot = self.data().collect()
        overview = snapshot["overview"]
        self.assertEqual(overview["coverage"]["input_total"], 2)
        self.assertEqual(overview["coverage"]["terminal_count"], 1)
        self.assertEqual(overview["coverage"]["ratio"], 0.5)
        self.assertEqual(overview["counts"]["UNREPORTED"], 1)
        self.assertEqual(overview["health"]["overall"], "UNKNOWN")
        self.assertIn("不是实时队列进度", overview["coverage"]["label"])
        row = next(row for row in overview["takes"] if row["status"] == "UNREPORTED")
        self.assertIn("不能据此推断", row["summary_reason"])
        self.assertNotIn("pending", overview)

    def test_malformed_report_is_unknown_not_unreported_or_zero_input(self):
        job = self.source_job()
        report = Path(job["report_path"])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{in-flight", encoding="utf-8")
        snapshot = self.data().collect()
        self.assertEqual(snapshot["rows"][job["job_id"]]["status"], "UNKNOWN")
        self.assertEqual(snapshot["overview"]["counts"]["UNKNOWN"], 1)
        self.assertEqual(snapshot["overview"]["counts"]["UNREPORTED"], 0)
        self.assertEqual(snapshot["overview"]["coverage"]["input_total"], 1)
        self.assertTrue(snapshot["overview"]["warnings"])

    def test_malformed_optional_fields_do_not_crash_whole_snapshot(self):
        job = self.source_job()
        path, report = self.report(job)
        report["dataset"] = []
        self.write_json(path, report)
        snapshot = self.data().collect()
        self.assertEqual(snapshot["rows"][job["job_id"]]["status"], "UNKNOWN")
        self.assertIn("invalid dataset", snapshot["rows"][job["job_id"]]["error"])

    def test_unknown_source_total_and_process_permissions_stay_null(self):
        failed = {"status": "unknown", "sampled_at": "2026-10-24T10:00:00+00:00", "processes": [],
                  "error": "Access denied", "memory": None}
        data = self.data((failed, failed), source=False)
        overview = data.collect()["overview"]
        self.assertIsNone(overview["counts"]["UNREPORTED"])
        self.assertIsNone(overview["coverage"]["input_total"])
        self.assertIsNone(overview["coverage"]["ratio"])
        for field in ("active_count", "active_progress_count", "python_busy_count", "native_busy_count", "memory_free_bytes"):
            self.assertIsNone(overview["health"][field])
        self.assertEqual(overview["health"]["overall"], "UNKNOWN")

    def test_unreadable_canonical_directory_is_unknown_not_unreported_or_zero(self):
        job = self.source_job()
        path, _ = self.report(job)
        data = self.data()
        scandir = os.scandir

        def denied_scandir(directory):
            if Path(directory).resolve() == path.parent:
                raise PermissionError("report listing denied")
            return scandir(directory)

        with patch("os.scandir", side_effect=denied_scandir):
            snapshot = data.collect()
        overview = snapshot["overview"]
        self.assertEqual(snapshot["rows"][job["job_id"]]["status"], "UNKNOWN")
        self.assertTrue(all(value is None for value in overview["counts"].values()))
        self.assertEqual(overview["coverage"]["input_total"], 1)
        for field in ("terminal_count", "unreported_count", "ratio"):
            self.assertIsNone(overview["coverage"][field])
        self.assertTrue(any("下界" in warning for warning in overview["warnings"]))
        self.assertTrue(any("report listing denied" in warning for warning in overview["warnings"]))

    def test_source_enumeration_failure_does_not_become_empty_inventory(self):
        self.source_job()
        data = self.data()
        with patch.object(step1_process_takes, "build_jobs", side_effect=PermissionError("denied")):
            overview = data.collect()["overview"]
        self.assertIsNone(overview["coverage"]["input_total"])
        self.assertIsNone(overview["counts"]["UNREPORTED"])
        self.assertTrue(any("denied" in warning for warning in overview["warnings"]))

    def test_missing_artifact_is_not_complete_published_output(self):
        job = self.source_job()
        _, report = self.report(job)
        Path(report["artifacts"]["fbx"]).unlink()
        data = self.data()
        snapshot = data.collect()
        row = snapshot["rows"][job["job_id"]]
        self.assertEqual((row["status"], row["artifact_integrity"]), ("SUCCESS", "missing"))
        self.assertEqual(snapshot["overview"]["coverage"]["published_count"], 0)
        self.assertEqual(snapshot["overview"]["coverage"]["terminal_count"], 1)
        artifact = next(item for item in data.detail(snapshot, job["job_id"])["artifacts"] if item["role"] == "fbx")
        self.assertEqual((artifact["available"], artifact["url"]), (False, None))

    def test_stale_workspace_is_not_running(self):
        job = self.source_job()
        self.active(job)
        snapshot = self.data(self.snapshots(processes=[self.scheduler])).collect()
        self.assertEqual(snapshot["rows"][job["job_id"]]["status"], "UNREPORTED")
        self.assertEqual(snapshot["overview"]["health"]["active_count"], 0)

    def test_scene_usable_and_human_quality_are_distinct_and_nonhumans_are_separate(self):
        job = self.source_job(kind="HOI")
        path, report = self.report(job, "NEEDS_REVIEW")
        report["subjects"][0]["quality"]["usable"] = True
        report["subjects"].extend([
            {"name": "damaged-human", "processing_target": True, "rig_class": "UNKNOWN"},
            {"name": "background-human", "processing_target": False, "rig_class": "HUMAN_RIG"},
            {"name": "chair", "processing_target": False, "rig_class": "PROP_OR_NONHUMAN"},
        ])
        report["subjects"][1].update(rig_class="HUMAN_RIG", quality={"basis": "post_processing", "usable": False})
        report["props"] = [{"name": "chair", "classification": "PARTIAL", "review_reasons": ["native_pose_incomplete"]}]
        self.write_json(path, report)
        data = self.data()
        snapshot = data.collect()
        detail = data.detail(snapshot, job["job_id"])
        self.assertIs(detail["take"]["usable"], False)
        self.assertIs(detail["subjects"][0]["quality"]["usable"], True)
        self.assertEqual({subject["name"] for subject in detail["subjects"]}, {"actor", "damaged-human", "background-human"})
        self.assertEqual([prop["name"] for prop in detail["props"]], ["chair"])
        report["status"] = "FAILED"
        report["subjects"][1]["rig_class"] = "UNKNOWN"
        self.write_json(path, report)
        data = self.data()
        detail = data.detail(data.collect(), job["job_id"])
        self.assertIn("damaged-human", [subject["name"] for subject in detail["subjects"]])

    def test_completed_requires_final_summary_and_complete_published_artifacts(self):
        job = self.source_job()
        _, report = self.report(job)
        self.write_json(self.output / "batch_report.json", {
            "run_id": RUN_ID, "counts": {"SUCCESS": 1, "NEEDS_REVIEW": 0, "FAILED": 0},
            "jobs": [{"job_id": job["job_id"], "status": "SUCCESS"}],
        })
        overview = self.data().collect()["overview"]
        self.assertEqual(overview["health"]["overall"], "COMPLETED")
        Path(report["artifacts"]["fbx"]).unlink()
        overview = self.data().collect()["overview"]
        self.assertEqual(overview["health"]["overall"], "UNKNOWN")
        self.assertEqual(overview["coverage"]["published_count"], 0)

    def test_detail_uses_associated_log_layout_not_report_provided_path(self):
        job = self.source_job()
        path, report = self.report(job)
        private = self.root / "private.log"
        private.write_text("secret", encoding="utf-8")
        report["logs"] = {"processing": str(private)}
        report["relative_path"] = "group9/HOI/other.mcp"
        self.write_json(path, report)
        log = Path(job["log_path"])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"x" * (TAIL_BYTES + 100))
        data = self.data()
        snapshot = data.collect()
        detail = data.detail(snapshot, job["job_id"])
        self.assertEqual(detail["logs"]["processing"]["path"], str(log))
        self.assertEqual(len(detail["logs"]["processing"]["text"].encode()), TAIL_BYTES)
        self.assertTrue(detail["logs"]["processing"]["truncated"])
        self.assertNotIn("secret", detail["logs"]["processing"]["text"])

    def test_download_only_current_role_and_nonempty_output_file(self):
        job = self.source_job()
        path, report = self.report(job)
        data = self.data()
        snapshot = data.collect()
        cache = self.cache(data, snapshot)
        route = f"/api/artifacts/{job['job_id']}/fbx"
        status, headers, body = self.request(cache, route, "HEAD")
        self.assertEqual((status, headers["Content-Length"], body), (200, "10", b""))
        status, headers, body = self.request(cache, route, headers={"Range": "bytes=2-4"})
        self.assertEqual((status, headers["Content-Range"], body), (206, "bytes 2-4/10", b"234"))
        self.assertEqual(self.request(cache, route, headers={"Range": "bytes=100-"})[0], 416)
        del report["artifacts"]["fbx"]
        self.write_json(path, report)
        self.assertEqual(self.request(cache, route, "HEAD")[0], 404)
        self.assertEqual(self.request(cache, f"/api/artifacts/{job['job_id']}/arbitrary", "HEAD")[0], 404)

    def test_artifact_source_escape_and_query_downloads_are_blocked(self):
        job = self.source_job()
        path, report = self.report(job)
        outside = self.root / "private.txt"
        outside.write_bytes(b"secret")
        data = self.data()
        snapshot = data.collect()
        cache = self.cache(data, snapshot)
        route = f"/api/artifacts/{job['job_id']}/fbx"
        for value in (str(outside), "../../../../private.txt", job["input_path"]):
            report["artifacts"]["fbx"] = value
            self.write_json(path, report)
            status, _, body = self.request(cache, route, "HEAD")
            self.assertIn(status, (400, 403, 404))
            self.assertEqual(body, b"")
        self.assertEqual(self.request(cache, route + "?path=" + str(outside), "HEAD")[0], 400)
        self.assertEqual(self.request(cache, "/api/artifacts/unknown/fbx", "HEAD")[0], 404)

    def test_symlink_escape_guard_and_nonempty_regular_file_rule(self):
        outside = self.root / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.output / "linked.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaises(ValueError):
            open_scoped(self.output, link)
        empty = self.output / "empty.fbx"
        empty.touch()
        with self.assertRaises(ValueError):
            open_scoped(self.output, empty)
        with self.assertRaises((ValueError, IsADirectoryError, PermissionError)):
            open_scoped(self.output, self.output)

    def test_static_traversal_foreign_host_cross_origin_and_writes_are_blocked(self):
        data = self.data(source=False)
        cache = self.cache(data, data.collect())
        for path in ("/../private.txt", "/%2e%2e/private.txt", "/api/artifacts/a/%2fsecret", "/%5csecret", "/C:%5csecret"):
            status, _, body = self.request(cache, path)
            self.assertEqual(status, 400, path)
            self.assertNotIn(b"secret file content", body)
        self.assertEqual(self.request(cache, "/", headers={"Host": "evil.example:8787"})[0], 403)
        self.assertEqual(self.request(cache, "/api/overview", headers={"Origin": "https://evil.example"})[0], 403)
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "CONNECT", "CUSTOM"):
            status, headers, _ = self.request(cache, "/api/overview", method)
            self.assertEqual((status, headers["Allow"]), (405, "GET, HEAD"))
        status, _, body = self.request(cache, "/")
        self.assertEqual((status, body), (200, b"<h1>dashboard</h1>"))

    def test_removed_artifact_invalidates_a_cached_completed_report(self):
        job = self.source_job()
        _, report = self.report(job)
        data = self.data()
        data.reports({job["job_id"]: job})
        again, _ = data.reports({job["job_id"]: job})
        Path(report["artifacts"]["vdf"]).unlink()
        self.assertEqual(data.artifact_rows(job["job_id"], again[job["job_id"]])[1], "missing")

    def test_loading_is_null_and_refresh_requests_are_singleflight(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def collect():
            calls.append(1)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test sampler release timed out")
            return {"overview": {"generated_at": "snapshot-time", "health": {"overall": "UNKNOWN"}}}

        data = SimpleNamespace(output=self.output, data_path=None, refresh_seconds=3600, collect=collect)
        cache = SnapshotCache(data)
        self.addCleanup(cache.close)
        initial = cache.overview()
        self.assertEqual(initial["health"]["overall"], "LOADING")
        self.assertTrue(all(value is None for value in initial["counts"].values()))
        cache.start()
        try:
            self.assertTrue(entered.wait(2))
            for _ in range(10):
                self.assertFalse(cache.request_refresh())
            self.assertTrue(cache.overview()["refreshing"])
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
        with cache._condition:
            self.assertTrue(cache._condition.wait_for(lambda: cache._snapshot is not None and not cache._refreshing, timeout=2))
        self.assertEqual(cache.overview()["generated_at"], "snapshot-time")

    def test_failed_refresh_keeps_old_snapshot_timestamp_and_reports_error(self):
        data = SimpleNamespace(output=self.output, data_path=None, refresh_seconds=3600,
                               collect=lambda: (_ for _ in ()).throw(PermissionError("denied")))
        cache = SnapshotCache(data)
        cache._snapshot = {"overview": {"generated_at": "old-time", "health": {"overall": "UNKNOWN"}}}
        self.addCleanup(cache.close)
        cache.start()
        with cache._condition:
            self.assertTrue(cache._condition.wait_for(lambda: cache._last_error is not None and not cache._refreshing, timeout=2))
        overview = cache.overview()
        self.assertEqual(overview["generated_at"], "old-time")
        self.assertIsNotNone(overview["last_error"])

    def catalog_record(self, output, **changes):
        return {"schema_version": 1, "id": output_id(output), "run_id": RUN_ID,
                "kind": "processing", "output_dir": str(output.resolve()), "source_output": None,
                "data_path": str(self.source), "started_at": "2026-10-24T06:57:56+00:00",
                "scheduler_pid": 100, "label": output.name, "planned_total": None} | changes

    def run_manager(self):
        primary = self.data()
        catalog = RunCatalog(primary, self.root / "registry",
                             snapshot_provider=lambda pids: self.snapshots()[0])
        return MultiRunCache(primary, catalog=catalog)

    def register_run(self, manager, output, **changes):
        record = self.catalog_record(output, **changes)
        self.write_json(manager.catalog.registry / (record["id"] + ".json"), record)
        return record

    def test_catalog_discovers_new_records_after_startup_and_rejects_identity_mismatch(self):
        manager = self.run_manager()
        initial = manager.runs()
        self.assertEqual([row["id"] for row in initial["runs"]], [output_id(self.output)])
        other = self.root / "new-repair"
        other.mkdir()
        record = self.register_run(manager, other, kind="human_repair")
        catalog = manager.runs()
        self.assertEqual({row["id"] for row in catalog["runs"]}, {output_id(self.output), record["id"]})
        self.assertEqual(catalog["default_run"], output_id(self.output))
        forged = dict(record, output_dir=str(self.root / "another-root"))
        self.write_json(manager.catalog.registry / (record["id"] + ".json"), forged)
        catalog = manager.runs()
        self.assertEqual(len(catalog["runs"]), 1)
        self.assertTrue(catalog["warnings"])
        self.assertRaises(KeyError, manager.for_run, record["id"])

    def test_catalog_rejects_malformed_schema_relative_roots_and_duplicate_aliases(self):
        manager = self.run_manager()
        record = self.catalog_record(self.output)
        for changes in ({"schema_version": True}, {"output_dir": "../escape"}, {"id": "../escape"},
                        {"scheduler_pid": True}, {"started_at": "2026-10-24T06:57:56"},
                        {"kind": "human_target", "subject": None}):
            with self.subTest(changes=changes):
                self.write_json(manager.catalog.registry / (record["id"] + ".json"), record | changes)
                self.assertTrue(manager.runs()["warnings"])
        self.write_json(manager.catalog.registry / (record["id"] + ".json"), record)
        self.write_json(manager.catalog.registry / "alias.json", record)
        catalog = manager.runs()
        self.assertEqual(len(catalog["runs"]), 1)
        self.assertTrue(catalog["warnings"])

    def test_catalog_active_default_requires_actual_owned_process_not_label(self):
        manager = self.run_manager()
        other = self.root / "active-repair"
        other.mkdir()
        record = self.register_run(manager, other, label="RUNNING active latest", started_at="2026-10-25T06:57:56+00:00")
        self.assertEqual(manager.runs()["default_run"], output_id(self.output))
        process = self.process(400, 10, "python.exe", "2026-10-25T06:57:55+00:00",
                               f'python.exe C:/project/step2_repair_humans.py --output-dir "{other}"')
        self.write_json(other / ".pipeline.lock", {"pid": 400, "run_id": "20261025T065756Z-test0001"})
        manager.catalog._process_at = -float("inf")
        manager.catalog.snapshot_provider = lambda pids: self.snapshots(processes=[process])[0]
        catalog = manager.runs()
        self.assertEqual(catalog["default_run"], record["id"])
        self.assertTrue(next(row for row in catalog["runs"] if row["id"] == record["id"])["active"])

    def test_same_job_downloads_and_details_are_isolated_by_catalog_run(self):
        job = self.source_job()
        _, original = self.report(job)
        manager = self.run_manager()
        primary = manager.for_run()
        primary._snapshot = primary.data.collect()
        other = self.root / "other"
        other.mkdir()
        record = self.register_run(manager, other)
        path = other / "takes" / job["relative_path"] / "report.json"
        artifact = path.parent / "artifacts" / "different.vdf"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"second-run-only")
        self.write_json(path, dict(original, artifacts={"vdf": str(artifact)}))
        selected = manager.for_run(record["id"])
        selected.data.snapshot_provider = lambda pids: self.snapshots()[0]
        selected.data.sleeper = lambda seconds: None
        selected._snapshot = selected.data.collect()
        for identifier, expected in ((output_id(self.output), b"0123456789"), (record["id"], b"second-run-only")):
            status, _, body = self.request(manager, f"/api/artifacts/{job['job_id']}/vdf?run={identifier}")
            self.assertEqual((status, body), (200, expected))
            status, _, body = self.request(manager, f"/api/takes/{job['job_id']}?run={identifier}")
            self.assertEqual(status, 200)
            urls = [row["url"] for row in json.loads(body)["artifacts"] if row["available"]]
            self.assertTrue(all(url.endswith("?run=" + identifier) for url in urls))
        self.assertEqual(self.request(manager, f"/api/artifacts/{job['job_id']}/vdf")[2], b"0123456789")
        for selector in ("../output", "%2e%2e%2foutput", "0" * 16, record["id"] + "&run=" + record["id"]):
            self.assertIn(self.request(manager, f"/api/overview?run={selector}")[0], (400, 404))
        self.assertEqual(self.request(manager, "/api/runs?path=" + str(other))[0], 400)
        self.assertEqual(self.request(manager, f"/api/artifacts/{job['job_id']}/%2e%2e?run={record['id']}")[0], 400)
        source_inside = path.parent / "original.mcp"
        source_inside.write_bytes(b"raw-source-never-download")
        self.write_json(path, dict(original, artifacts={"vdf": str(source_inside)}))
        self.assertEqual(self.request(manager, f"/api/artifacts/{job['job_id']}/vdf?run={record['id']}")[0], 400)

    def test_repair_coverage_uses_selected_plan_not_duplicate_inventory_or_registry_total(self):
        selected = self.source_job("selected.mcp")
        unselected = self.source_job("unselected.mcp")
        self.report(selected)
        self.report(unselected)
        self.write_json(self.output / "repair_plan.json", {
            "jobs": [selected], "counts": {"source_failed": 2, "invalid_checkpoint": 1},
            "inventory": [{"job_id": selected["job_id"], "disposition": "selected"},
                          {"job_id": selected["job_id"], "disposition": "source_failed"},
                          {"job_id": "blocked", "disposition": "source_failed"}], "roster_pending": []})
        data = self.data()
        data.run = self.catalog_record(self.output, kind="human_repair", planned_total=999)
        overview = data.collect()["overview"]
        self.assertEqual(overview["coverage"]["input_total"], 1)
        self.assertEqual(overview["coverage"]["terminal_count"], 1)
        self.assertEqual(len(overview["takes"]), 1)
        self.assertEqual(overview["inventory"]["source_failed"], 2)
        self.assertEqual(overview["inventory"]["identities_without_reusable_checkpoint"], 1)
        self.assertEqual(overview["inventory_units"]["source_failed"], "source_reports")
        self.assertFalse(overview["takes"][0]["usable"])
        self.assertEqual(overview["rechecks"]["pending"], 1)

    def test_invalid_selected_repair_plan_does_not_fall_back_to_raw_input_count(self):
        job = self.source_job()
        self.write_json(self.output / "repair_plan.json", {"jobs": [dict(job, relative_path="../escape")]})
        data = self.data()
        data.run = self.catalog_record(self.output, kind="human_repair", planned_total=42)
        overview = data.collect()["overview"]
        self.assertIsNone(overview["coverage"]["input_total"])
        self.assertTrue(overview["warnings"])

    def test_latest_independent_recheck_failure_and_running_block_usable_completion(self):
        job = self.source_job()
        path, report = self.report(job)
        report["independent_recheck"] = {"status": "PASSED", "finished_at": "2026-10-24T09:00:00+00:00"}
        self.write_json(path, report)
        self.write_json(self.output / "batch_report.json", {"run_id": RUN_ID, "jobs": [report],
                        "counts": {"SUCCESS": 1, "NEEDS_REVIEW": 0, "FAILED": 0}})
        for state in ("FAILED", "RUNNING"):
            with self.subTest(state=state):
                sidecar = {"job_id": job["job_id"], "status": state, "started_at": "2026-10-24T10:00:00+00:00"}
                if state == "FAILED":
                    sidecar.update(finished_at="2026-10-24T10:00:01+00:00", error="Native failed")
                self.write_json(path.with_name("independent_recheck.json"), sidecar)
                data = self.data()
                snapshot = data.collect()
                self.assertFalse(snapshot["rows"][job["job_id"]]["usable"])
                self.assertFalse(data.detail(snapshot, job["job_id"])["quality"]["usable"])
                self.assertEqual(snapshot["overview"]["rechecks"][state.lower()], 1)
                self.assertNotEqual(snapshot["overview"]["health"]["overall"], "COMPLETED")

    def test_primary_without_recheck_evidence_does_not_invent_required_pending_rechecks(self):
        self.report()
        overview = self.data().collect()["overview"]
        self.assertTrue(all(value is None for value in overview["rechecks"].values()))

    def test_human_repair_legacy_pass_without_artifact_fingerprints_remains_unknown(self):
        job = self.source_job()
        path, report = self.report(job)
        self.write_json(path.with_name("independent_recheck.json"), {
            "job_id": job["job_id"], "status": "PASSED", "finished_at": "2026-10-24T10:00:01+00:00",
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        data = self.data()
        data.run = self.catalog_record(self.output, kind="human_repair")
        snapshot = data.collect()
        self.assertEqual(snapshot["rows"][job["job_id"]]["independent_recheck"]["status"], "UNKNOWN")
        self.assertFalse(snapshot["rows"][job["job_id"]]["usable"])

    def test_human_target_is_subject_scoped_and_download_revalidates_audit(self):
        data = self.data(source=False)
        source = self.root / "source.vdf"
        source.write_bytes(b"source")
        data.run = self.catalog_record(self.output, kind="human_target", subject="actor", input_path=str(source))
        products = {}
        for role in ("baseline", "candidate", "fbx"):
            artifact = self.output / (role + ".vdf")
            artifact.write_bytes(role.encode())
            products[role] = str(artifact)
        audit = {"status": "SUCCESS", "subject": "actor", "source": str(source), "accepted": True,
                 "source_unchanged": True, "artifacts": products, "review_reasons": [],
                 "quality_scope": {"subject": "actor", "human_geometry": "target_only",
                                   "other_humans_validated": False, "props_validated": False}}
        self.write_json(self.output / "repair_audit.json", audit)
        snapshot = data.collect()
        row = snapshot["overview"]["takes"][0]
        self.assertEqual(snapshot["overview"]["coverage"]["input_total"], 1)
        self.assertIsNone(row["usable"])
        self.assertTrue(row["target_accepted"])
        self.assertFalse(row["quality_scope"]["props_validated"])
        stream, _, _ = data.open_artifact(snapshot, row["id"], "candidate")
        with stream:
            self.assertEqual(stream.read(), b"candidate")
        self.write_json(self.output / "repair_audit.json", dict(audit, subject="other"))
        self.assertRaises(ValueError, data.open_artifact, snapshot, row["id"], "candidate")

    def test_run_snapshots_are_lazy_bounded_and_only_selection_is_scheduled(self):
        manager = self.run_manager()
        for index in range(7):
            output = self.root / ("history-" + str(index))
            output.mkdir()
            record = self.register_run(manager, output)
            # Discovery never starts a historical sampler or creates a cache.
            manager.runs()
            self.assertLessEqual(len(manager.caches), 4)
            cache = manager.for_run(record["id"])
            self.assertIsNone(cache._thread)
            self.assertIsNone(cache.snapshot())
            self.assertEqual(manager.selected, record["id"])
            self.assertLessEqual(len(manager.caches), 4)
        self.assertIn(manager.primary_id, manager.caches)

    def test_recheck_scheduler_transition_does_not_reuse_old_completed_batch_state(self):
        job = self.source_job()
        path, report = self.report(job)
        self.write_json(self.output / "batch_report.json", {"run_id": RUN_ID, "jobs": [report],
                        "counts": {"SUCCESS": 1, "NEEDS_REVIEW": 0, "FAILED": 0}})
        process = self.process(500, 10, "python.exe", "2026-10-24T09:00:00+00:00",
                               f'python.exe C:/project/step4_recheck_quality.py -OutputDir "{self.output}"')
        self.write_json(self.output / "quality_recheck_report.json", {"finished": False, "rechecked": [], "errors": []})
        data = self.data(self.snapshots(processes=[process]))
        self.assertNotEqual(data.collect()["overview"]["health"]["overall"], "COMPLETED")

    def test_passed_independent_check_does_not_override_unusable_quality(self):
        job = self.source_job()
        path, report = self.report(job, "NEEDS_REVIEW")
        self.write_json(self.output / ".quality_recheck_attempts" / "attempt" / "attempt.json",
                        {"report": str(path), "output": str(self.output), "phase": "completed",
                         "started_at": "2026-10-24T09:00:00+00:00", "finished_at": "2026-10-24T09:01:00+00:00"})
        data = self.data()
        snapshot = data.collect()
        row = snapshot["rows"][job["job_id"]]
        self.assertEqual(row["independent_recheck"]["status"], "PASSED")
        self.assertTrue(row["independent_recheck"]["operational_checked"])
        self.assertFalse(row["usable"])

    def test_inflight_recheck_without_ready_native_preserves_unknown_live_count(self):
        job = self.source_job()
        path, _ = self.report(job)
        self.write_json(self.output / ".quality_recheck_attempts" / "attempt" / "attempt.json",
                        {"report": str(path), "output": str(self.output), "phase": "pending",
                         "started_at": "2026-10-24T09:00:00+00:00", "parent_pid": 500,
                         "worker_pid": None, "native_pid": None})
        overview = self.data().collect()["overview"]
        self.assertEqual(overview["takes"][0]["independent_recheck"]["status"], "RUNNING")
        self.assertIsNone(overview["health"]["active_count"])
        self.assertNotEqual(overview["health"]["overall"], "COMPLETED")


    def test_implicit_primary_metadata_is_unknown_until_canonical_snapshot_supplies_run_id(self):
        manager = self.run_manager()
        catalog = manager.runs()
        row = next(item for item in catalog["runs"] if item["id"] == manager.primary_id)
        self.assertIsNone(row["run_id"])
        self.assertNotIn("active", row)
        loading = manager.overview()
        self.assertEqual(loading["run"]["id"], manager.primary_id)
        self.assertIsNone(loading["run"]["run_id"])
        cache = manager.for_run()
        cache._snapshot = {"overview": {"health": {"run_id": RUN_ID}, "takes": []}}
        self.assertEqual(manager.overview()["run"]["run_id"], RUN_ID)


    def test_bookmarked_run_document_reloads_without_relaxing_static_query_guards(self):
        manager = self.run_manager()
        identifier = manager.primary_id
        for document in ("/", "/index.html"):
            status, _, body = self.request(manager, document + "?run=" + identifier)
            self.assertEqual((status, body), (200, b"<h1>dashboard</h1>"))
        # Unknown syntactically valid ids reach the frontend's catalog handling.
        self.assertEqual(self.request(manager, "/?run=" + "0" * 16)[0], 200)
        for path in (f"/?run={identifier}&run={identifier}", f"/?run={identifier}&extra=1",
                     "/?run=..%2fescape", "/?run=", f"/styles/style.css?run={identifier}"):
            self.assertEqual(self.request(manager, path)[0], 400)

    def test_scoped_shared_handle_allows_report_publication_and_revalidates_new_report(self):
        job = self.source_job()
        path, original = self.report(job)
        data = self.data()
        snapshot = data.collect()
        stream, _, _ = open_scoped(self.output, path)
        with stream:
            self.assertEqual(json.load(stream), original)
        replacement = path.with_name("report-next.json")
        updated = dict(original, artifacts={"vdf": job["input_path"]})
        self.write_json(replacement, updated)
        os.replace(replacement, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), updated)
        self.assertRaises(ValueError, data.open_artifact, snapshot, job["job_id"], "vdf")

    def test_catalog_reader_uses_same_shared_handle_during_atomic_replacement(self):
        manager = self.run_manager()
        other = self.root / "registered"
        other.mkdir()
        original = self.register_run(manager, other)
        path = manager.catalog.registry / (original["id"] + ".json")
        catalog = manager.runs()
        selected = next(row for row in catalog["runs"] if row["id"] == original["id"])
        self.assertEqual(selected["label"], original["label"])
        replacement = path.with_suffix(".next")
        self.write_json(replacement, dict(original, label="new label"))
        os.replace(replacement, path)
        selected = next(row for row in manager.runs()["runs"] if row["id"] == original["id"])
        self.assertEqual(selected["label"], "new label")

    def sequence_layout(self):
        self.output = self.source / "_processing" / "current"
        self.output.mkdir(parents=True)
        layout.initialize_layout(self.output, self.source, "current")
        return self.source_job("sequence/take.mcp")

    def test_sequence_local_reports_ignore_raw_diagnostics_and_neighbor_versions(self):
        job = self.sequence_layout()
        path, report = self.report(job)
        expected = self.source / "group1/MP/sequence/processed/current/take.mcp/report.json"
        self.assertEqual(path, expected)
        for other in (path.parent / "artifacts/report.json",
                      path.parent.parent.parent / "neighbor/take.mcp/report.json",
                      path.parents[3] / "report.json",
                      self.output / "takes" / job["relative_path"] / "report.json"):
            self.write_json(other, dict(report, job_id="not-canonical"))
        data = self.data()
        snapshot = data.collect()
        self.assertEqual(set(snapshot["reports"]), {job["job_id"]})
        self.assertEqual(snapshot["rows"][job["job_id"]]["relative_path"], job["relative_path"])
        self.assertEqual(snapshot["overview"]["coverage"]["input_total"], 1)
        self.assertEqual(snapshot["rows"][job["job_id"]]["artifact_integrity"], "complete")

    def test_sequence_local_downloads_revalidate_and_reject_raw_neighbor_take_and_version(self):
        job = self.sequence_layout()
        path, report = self.report(job)
        other = self.source_job("other-sequence/other.mcp")
        _, other_report = self.report(other)
        raw = Path(job["input_path"]).with_suffix(".vdf")
        raw.write_bytes(b"raw-vdf")
        neighbor = path.parent.parent.parent / "neighbor/take.mcp/artifacts/take.fbx"
        neighbor.parent.mkdir(parents=True)
        neighbor.write_bytes(b"neighbor-version")
        private = self.source / "group1/MP/neighbor-sequence/private.fbx"
        private.parent.mkdir(parents=True)
        private.write_bytes(b"raw-neighbor")
        data = self.data()
        snapshot = data.collect()
        cache = self.cache(data, snapshot)
        route = f"/api/artifacts/{job['job_id']}/fbx"
        self.assertEqual(self.request(cache, route)[2], b"0123456789")
        for forbidden in (raw, private, neighbor, Path(other_report["artifacts"]["fbx"]),
                          Path(job["input_path"]), self.source / "_processing/other/private.fbx"):
            with self.subTest(path=forbidden):
                self.write_json(path, dict(report, artifacts={"fbx": str(forbidden)}))
                self.assertIn(self.request(cache, route, "HEAD")[0], (400, 403, 404))
                rows, _ = data.artifact_rows(job["job_id"],
                                            {"path": str(path), "report": dict(report, artifacts={"fbx": str(forbidden)})})
                self.assertFalse(rows[0]["available"])
                self.assertIsNone(rows[0]["url"])
        self.write_json(path, dict(report, artifacts={}))
        self.assertEqual(self.request(cache, route, "HEAD")[0], 404)

    def test_sequence_local_review_artifacts_are_downloadable(self):
        job = self.sequence_layout()
        path, report = self.report(job, "NEEDS_REVIEW")
        review = layout.take_paths(self.output, job["relative_path"])["review_destination"]
        review.mkdir(parents=True)
        product = review / "take.fbx"
        product.write_bytes(b"review-product")
        self.write_json(path, dict(report, artifacts={"fbx": str(product)}))
        data = self.data()
        snapshot = data.collect()
        with data.open_artifact(snapshot, job["job_id"], "fbx")[0] as stream:
            self.assertEqual(stream.read(), b"review-product")

    def test_sequence_local_logs_use_exact_identity_not_report_or_control_log_paths(self):
        job = self.sequence_layout()
        path, report = self.report(job)
        log_dir = layout.take_paths(self.output, job["relative_path"])["log_dir"]
        log_dir.mkdir()
        (log_dir / "processing.log").write_text("exact local log", encoding="utf-8")
        old = self.output / "logs" / job["relative_path"] / "processing.log"
        old.parent.mkdir(parents=True)
        old.write_text("wrong control log", encoding="utf-8")
        self.write_json(path, dict(report, relative_path="group9/HOI/other/wrong.mcp",
                                   logs={"processing": str(old)}))
        data = self.data()
        snapshot = data.collect()
        detail = data.detail(snapshot, job["job_id"])
        self.assertEqual(detail["logs"]["processing"]["path"], str(log_dir / "processing.log"))
        self.assertEqual(detail["logs"]["processing"]["text"], "exact local log")
        self.assertIsNotNone(detail["logs"]["shogun"]["error"])

    def test_sequence_local_attempts_and_work_sessions_are_exact_run_scoped(self):
        job = self.sequence_layout()
        path, _ = self.report(job)
        paths = layout.take_paths(self.output, job["relative_path"])
        attempt = {"report": str(path), "output": str(self.output), "phase": "failed",
                   "started_at": "2026-10-24T10:00:00+00:00", "error": "paused attempt"}
        self.write_json(paths["recheck_root"] / "valid/attempt.json", attempt)
        self.write_json(paths["recheck_root"] / "raw/attempt.json",
                        dict(attempt, report=str(Path(job["input_path"]).parent / "report.json")))
        self.write_json(paths["recheck_root"] / "wrong-version/attempt.json",
                        dict(attempt, report=str(path.parent.parent.parent / "neighbor/take.mcp/report.json")))
        other = self.source_job("other-sequence/other.mcp")
        other_path, _ = self.report(other)
        self.write_json(paths["recheck_root"] / "wrong-take/attempt.json",
                        dict(attempt, report=str(other_path)))
        control_attempt = dict(attempt, started_at="2026-10-24T09:00:00+00:00")
        self.write_json(self.output / ".quality_recheck_attempts/unassigned/attempt.json", control_attempt)
        session = {"pid": 300, "port": 50001, "started_at": "2026-10-24T09:00:00+00:00"}
        workspace = paths["work_root"] / (job["job_id"] + "-workspace")
        self.write_json(workspace / ".control/session-ready.json", session)
        neighbor = path.parent.parent.parent / "neighbor/take.mcp"
        self.write_json(neighbor / ".work/not-this-run/.control/session-ready.json", session)
        self.write_json(neighbor / ".quality_recheck_attempts/attempt/attempt.json", attempt)
        data = self.data()
        attempts, warnings = data.recheck_attempts()
        self.assertEqual(set(attempts), {str(path)})
        self.assertEqual(attempts[str(path)]["started_at"], attempt["started_at"])
        self.assertEqual(len(warnings), 3)
        sessions = shared_sessions(self.output)
        self.assertFalse(sessions["errors"])
        self.assertEqual([row["workspace"] for row in sessions["entries"]], [str(workspace)])
        snapshot = data.collect()
        self.assertEqual(snapshot["rows"][job["job_id"]]["independent_recheck"]["status"], "FAILED")
        self.assertFalse(snapshot["rows"][job["job_id"]]["usable"])

    def test_sequence_local_repair_plan_stays_in_control_and_ignores_saved_output_paths(self):
        job = self.sequence_layout()
        path, _ = self.report(job)
        other = self.source_job("other-sequence/other.mcp")
        self.report(other)
        self.write_json(path.parent / "repair_plan.json", {"jobs": [other]})
        self.write_json(self.output / "repair_plan.json",
                        {"jobs": [dict(job, report_path=str(self.source / "raw/report.json"))]})
        data = self.data()
        data.run = self.catalog_record(self.output, kind="human_repair", planned_total=999)
        snapshot = data.collect()
        self.assertEqual(set(snapshot["rows"]), {job["job_id"]})
        self.assertEqual(snapshot["overview"]["coverage"]["input_total"], 1)
        self.assertEqual(snapshot["reports"][job["job_id"]]["path"], str(path))
        (self.output / "repair_plan.json").unlink()
        jobs, error = data.repair_inventory()
        self.assertIsNone(jobs)
        self.assertIn("选中修复计划未知", error)

    def test_sequence_local_manifest_does_not_authorize_forged_canonical_report(self):
        job = self.sequence_layout()
        path, report = self.report(job)
        data = self.data()
        snapshot = data.collect()
        forged = path.parent / "artifacts/report.json"
        self.write_json(forged, report)
        snapshot["reports"][job["job_id"]]["path"] = str(forged)
        with self.assertRaises(ValueError):
            data.open_artifact(snapshot, job["job_id"], "fbx")
        self.write_json(self.output / ".sequence-layout.json",
                        {"schema_version": 1, "data_root": str(self.source), "version_key": "../neighbor"})
        reports, warnings = data.reports(None)
        self.assertFalse(reports)
        self.assertFalse(data._report_scan_complete)
        self.assertTrue(warnings)

    def test_sequence_local_link_cannot_authorize_neighbor_version(self):
        job = self.sequence_layout()
        path, report = self.report(job)
        outside = path.parent.parent.parent / "neighbor/take.mcp/artifacts/private.fbx"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"not-this-run")
        link = path.parent / "artifacts/linked.fbx"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        self.write_json(path, dict(report, artifacts={"fbx": str(link)}))
        data = self.data()
        snapshot = data.collect()
        with self.assertRaises(ValueError):
            data.open_artifact(snapshot, job["job_id"], "fbx")
        self.assertFalse(snapshot["details"][job["job_id"]]["artifacts"][0]["available"])

    def test_catalog_configured_source_controls_keep_raw_path_separate(self):
        self.sequence_layout()
        repair = self.output / "_repair_runs" / "repair"
        repair.mkdir(parents=True)
        layout.initialize_layout(repair, self.source, "repair")
        neighbor = self.source / "_processing/neighbor"
        neighbor.mkdir()
        manager = self.run_manager()
        catalog = manager.runs()
        self.assertEqual({row["id"] for row in catalog["runs"]},
                         {output_id(self.output), output_id(repair)})
        record = manager.catalog.records[output_id(repair)]
        self.assertEqual(record["output_dir"], str(repair))
        self.assertEqual(record["data_path"], str(self.source))
        self.assertEqual(record["source_output"], str(self.output))
        self.assertFalse(manager.catalog.registry.exists())

    def test_generic_source_inventory_excludes_generated_dataset_trees(self):
        job = self.source_job()
        self.report(job)
        self.source_job("processed/neighbor/fake.mcp")
        generated = self.source / "_processing/other/copied.mcp"
        generated.parent.mkdir(parents=True)
        generated.write_bytes(b"generated")
        generated.with_suffix(".x2d").write_bytes(b"generated")
        snapshot = self.data().collect()
        self.assertEqual(set(snapshot["rows"]), {job["job_id"]})
        self.assertEqual(snapshot["overview"]["coverage"]["input_total"], 1)
        self.assertIsNone(layout.load_layout(self.output))

    def test_sequence_local_raw_default_does_not_replace_explicit_source_configuration(self):
        self.sequence_layout()
        inferred = DashboardData(self.output)
        self.assertEqual(inferred.data_path, self.source)
        explicit = DashboardData(self.output, self.root)
        self.assertEqual(explicit.data_path, self.root)

    def test_sequence_local_unreadable_canonical_directory_preserves_unknown_counts(self):
        job = self.sequence_layout()
        path, _ = self.report(job)
        scandir = os.scandir
        for denied_directory in (path.parent.parent, Path(job["input_path"]).parent):
            with self.subTest(directory=denied_directory):
                data = self.data()
                # Keep the independently obtained input census available when
                # canonical report enumeration loses directory permission.
                data.source_inventory()

                def denied_scandir(directory):
                    if Path(directory).resolve() == denied_directory:
                        raise PermissionError("local reports denied")
                    return scandir(directory)

                with patch("os.scandir", side_effect=denied_scandir):
                    snapshot = data.collect()
                overview = snapshot["overview"]
                self.assertEqual(snapshot["rows"][job["job_id"]]["status"], "UNKNOWN")
                self.assertTrue(all(value is None for value in overview["counts"].values()))
                self.assertEqual(overview["coverage"]["input_total"], 1)
                for field in ("terminal_count", "unreported_count", "ratio"):
                    self.assertIsNone(overview["coverage"][field])
                self.assertTrue(any("local reports denied" in warning for warning in overview["warnings"]))

if __name__ == "__main__":
    unittest.main()

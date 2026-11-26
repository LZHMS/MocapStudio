"""Focused manager regressions; no native software or production datasets needed."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from web_manager import APIError, JobManager, ProcessingServer, atomic_json, allowed_authority


class ManagerFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root / "dataset"
        self.data.mkdir()
        self.source = self.data / "take.fbx"
        self.source.write_bytes(b"source fbx fixture")
        self.manager = JobManager(self.root / "state", [self.data], cancel_grace=0)
        self.addCleanup(self.manager.close)

    def enqueue(self, **fields):
        return self.manager.enqueue({"paths": [str(self.source)], **fields})["jobs"][0]

    def state(self, job_id, **fields):
        atomic_json(self.manager.jobs_dir / job_id / "state.json", fields)


class ManagerTests(ManagerFixture):

    def test_browse_never_launches_a_process(self):
        nested = self.data / "nested"
        nested.mkdir()
        (nested / "capture.mcp").write_bytes(b"mcp")
        (nested / "capture.x2d").write_bytes(b"x2d")
        for excluded in ("processed", "_processing", "node_modules", ".git"):
            directory = self.data / excluded
            directory.mkdir()
            (directory / "generated.fbx").write_bytes(b"generated")
        with patch("web_manager.subprocess.Popen") as launch:
            shallow = self.manager.browse()
            deep = self.manager.browse(recursive=True)
            launch.assert_not_called()
        self.assertEqual([item["name"] for item in shallow["sequences"]], ["take.fbx"])
        self.assertEqual({item["name"] for item in deep["sequences"]}, {"take.fbx", "capture.mcp"})
        self.assertTrue(next(item for item in deep["sequences"] if item["kind"] == "mcp")["x2d_exists"])
        self.assertIsNone(shallow["parent"])
        self.assertEqual(self.manager.browse(str(nested))["parent"], str(self.data))

    def test_path_escape_and_internal_input_are_denied(self):
        outside = self.root / "outside.fbx"
        outside.write_bytes(b"outside")
        generated = self.data / ".work"
        generated.mkdir()
        (generated / "take.fbx").write_bytes(b"generated")
        for path in (outside, generated / "take.fbx", self.data / ".."):
            with self.subTest(path=path), self.assertRaises(APIError) as error:
                self.manager.enqueue({"paths": [str(path)]})
            self.assertEqual(error.exception.status, 403)
        with self.assertRaises(APIError):
            self.manager.enqueue({"paths": ["take.fbx"]})

    def test_explicit_canonical_outputs_are_fresh_inputs_but_not_discovered(self):
        for folder in ("processed", "_processing"):
            directory = self.data / folder
            directory.mkdir()
            source = directory / "canonical.fbx"
            source.write_bytes(b"existing canonical output")
            original = source.read_bytes()
            first = self.manager.enqueue({"paths": [str(source)]})["jobs"][0]
            second = self.manager.enqueue({"paths": [str(source)]})["jobs"][0]
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(first["input_path"], str(source))
            self.assertEqual(first["state"], "QUEUED")
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(self.manager.browse(str(directory))["sequences"][0]["path"], str(source))
        discovered = self.manager.browse(recursive=True)["sequences"]
        self.assertEqual([item["path"] for item in discovered], [str(self.source)])
        with self.assertRaises(APIError) as error:
            self.manager.browse(str(self.manager.state_dir))
        self.assertEqual(error.exception.status, 403)

    def test_symlink_escape_is_denied(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "escaped.fbx").write_bytes(b"outside")
        link = self.data / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Creating symlinks requires local privileges")
        self.assertEqual([item["name"] for item in self.manager.browse(recursive=True)["sequences"]], ["take.fbx"])
        with self.assertRaises(APIError) as error:
            self.manager.enqueue({"paths": [str(link / "escaped.fbx")]})
        self.assertEqual(error.exception.status, 403)

    def test_batch_validation_is_atomic_and_deduplicates_selection(self):
        with self.assertRaises(APIError):
            self.manager.enqueue({"paths": [str(self.source), str(self.data / "missing.fbx")]})
        self.assertEqual(self.manager.jobs()["jobs"], [])
        jobs = self.manager.enqueue({"paths": [str(self.source), str(self.source)]})["jobs"]
        self.assertEqual(len(jobs), 1)
        request = json.loads((self.manager.jobs_dir / jobs[0]["id"] / "request.json").read_text())
        self.assertEqual(request["input_path"], str(self.source))
        self.assertEqual(request["job_dir"], str(self.manager.jobs_dir / jobs[0]["id"]))
        self.assertTrue(request["options"]["lightweight"])
        self.assertFalse(request["options"]["attach_meshes"])

    def test_strict_options_and_config(self):
        invalid = [{"attach_meshes": "true"}, {"prop_tolerance": 0}, {"prop_tolerance": float("nan")},
                   {"prop_jitter_max_correction": -1}, {"expected_subjects": "person"},
                   {"expected_props": [""]}, {"executable": "arbitrary.exe"}]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(APIError):
                self.enqueue(options=options)
        for count in (0, 9, True, 1.5, "2"):
            with self.subTest(count=count), self.assertRaises(APIError):
                self.manager.configure({"max_workers": count})
        self.assertEqual(self.manager.configure({"max_workers": 3})["max_workers"], 3)
        self.assertEqual(json.loads((self.manager.state_dir / "config.json").read_text())["max_workers"], 3)

    def test_clear_archive_deletes_only_archive_copies_under_data_roots(self):
        archived = self.data / "02_processed" / "body"
        archived.mkdir(parents=True)
        mirrored = archived / "take.fbx"
        mirrored.write_bytes(b"archive copy")
        keeper = archived / "other.fbx"
        keeper.write_bytes(b"keep me")
        job = self.manager.enqueue({"paths": [str(self.source)]})["jobs"][0]
        job_dir = self.manager.jobs_dir / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "state.json").write_text(json.dumps({"artifacts": [
            {"id": "a1", "name": "take.fbx", "role": "archive_processed_fbx", "path": str(mirrored), "size_bytes": 12},
            {"id": "a2", "name": "other.fbx", "role": "fbx", "path": str(keeper), "size_bytes": 7},
        ]}), encoding="utf-8")
        result = self.manager.clear_archive(job["id"])
        self.assertEqual(result["removed"], [str(mirrored)])
        self.assertFalse(mirrored.exists())
        self.assertEqual(keeper.read_bytes(), b"keep me")
        with self.assertRaises(APIError) as error:
            self.manager.clear_archive("0" * 32)
        self.assertEqual(error.exception.status, 404)

    def test_queued_cancel_and_retry_create_new_immutable_request(self):
        first = self.enqueue(mode="auto")
        request_path = self.manager.jobs_dir / first["id"] / "request.json"
        before = request_path.read_bytes()
        cancelled = self.manager.action(first["id"], {"action": "cancel"})["job"]
        self.assertEqual(cancelled["state"], "CANCELLED")
        retry = self.manager.action(first["id"], {"action": "retry"})["job"]
        self.assertNotEqual(first["id"], retry["id"])
        self.assertEqual(retry["state"], "QUEUED")
        self.assertEqual(request_path.read_bytes(), before)
        self.assertEqual(self.source.read_bytes(), b"source fbx fixture")
        with self.assertRaises(APIError) as error:
            self.manager.action(first["id"], {"action": "step"})
        self.assertEqual(error.exception.status, 409)

    def test_busy_actions_are_rejected_and_commands_are_acknowledged(self):
        job = self.enqueue()
        metadata = self.manager._jobs[job["id"]]
        metadata["state"] = "STARTING"
        self.manager._save(metadata)
        self.state(job["id"], state="WAITING", updated_at="one", command_seq=0, stages=[], artifacts=[])
        self.manager._owned[job["id"]] = {"cancel_at": None}
        try:
            result = self.manager.action(job["id"], {"action": "step"})["job"]
            self.assertEqual(result["state"], "RUNNING")
            with self.assertRaises(APIError) as error:
                self.manager.action(job["id"], {"action": "step"})
            self.assertEqual(error.exception.status, 409)
            self.state(job["id"], state="WAITING", updated_at="two", command_seq=1, stages=[], artifacts=[])
            self.assertEqual(self.manager.job(job["id"])["job"]["state"], "WAITING")
            self.manager.action(job["id"], {"action": "preview"})
            commands = [json.loads(line) for line in (self.manager.jobs_dir / job["id"] / "commands.jsonl").read_text().splitlines()]
            self.assertEqual(commands, [{"seq": 1, "action": "step"}, {"seq": 2, "action": "preview"}])
        finally:
            self.manager._owned.clear()

    def test_cancel_is_cooperative_until_worker_exits(self):
        job = self.enqueue()
        metadata = self.manager._jobs[job["id"]]
        metadata["state"] = "STARTING"
        self.state(job["id"], state="RUNNING", command_seq=0, updated_at="one")
        self.manager._owned[job["id"]] = {"cancel_at": None}
        try:
            result = self.manager.action(job["id"], {"action": "cancel"})["job"]
            self.assertEqual(result["state"], "PAUSE_REQUESTED")
            self.assertIsNotNone(self.manager._owned[job["id"]]["cancel_at"])
            self.assertNotEqual(metadata["state"], "CANCELLED")
            with self.assertRaises(APIError):
                self.manager.action(job["id"], {"action": "cancel"})
        finally:
            self.manager._owned.clear()

    def test_waiting_workers_occupy_bounded_slots(self):
        second = self.data / "second.fbx"
        second.write_bytes(b"second")
        self.manager.enqueue({"paths": [str(self.source), str(second)]})
        launched = []
        def launch(metadata):
            launched.append(metadata["id"])
            metadata["state"] = "STARTING"
            self.manager._owned[metadata["id"]] = {"cancel_at": None}
        try:
            with patch.object(self.manager, "_launch", side_effect=launch):
                self.manager.tick()
            self.assertEqual(len(launched), 1)
            self.assertEqual(sum(item["state"] == "QUEUED" for item in self.manager._jobs.values()), 1)
        finally:
            self.manager._owned.clear()

    def test_recovery_interrupts_active_and_preserves_queue(self):
        queued = self.enqueue()
        active = self.enqueue()
        self.manager._set_status(self.manager._jobs[active["id"]], "STARTING")
        self.state(active["id"], state="WAITING", stages=[{"id": "load-1", "status": "SUCCESS"}])
        self.manager.close()
        recovered = JobManager(self.manager.state_dir, [self.data], cancel_grace=0)
        try:
            self.assertEqual(recovered.job(queued["id"])["job"]["state"], "QUEUED")
            dto = recovered.job(active["id"])["job"]
            self.assertEqual(dto["state"], "INTERRUPTED")
            self.assertEqual(dto["stages"][0]["id"], "load-1")
            with self.assertRaises(APIError):
                recovered.action(active["id"], {"action": "run"})
        finally:
            recovered.close()

    def test_state_directory_has_single_owner(self):
        with self.assertRaises(RuntimeError):
            JobManager(self.manager.state_dir, [self.data])

    def test_artifact_registry_enforces_job_scope(self):
        job = self.enqueue()
        directory = self.manager.jobs_dir / job["id"]
        good = directory / "display.fbx"
        good.write_bytes(b"0123456789")
        outside = self.root / "secret.txt"
        outside.write_bytes(b"secret")
        self.state(job["id"], artifacts=[
            {"id": "display", "name": good.name, "role": "display_fbx", "path": str(good)},
            {"id": "source", "name": self.source.name, "role": "source_fbx", "path": str(self.source)},
            {"id": "secret", "path": str(outside)},
        ])
        artifacts = self.manager.job(job["id"])["job"]["artifacts"]
        self.assertEqual({item["id"] for item in artifacts}, {"source", "display"})
        self.assertEqual(self.manager.artifact(job["id"], "display"), good)
        with self.assertRaises(APIError):
            self.manager.artifact(job["id"], "secret")
        with self.assertRaises(APIError):
            self.manager.artifact(job["id"], "../request.json")

    def test_large_diagnostics_do_not_expand_queue_status_payload(self):
        job = self.enqueue()
        stage = {"id": "trim-1", "index": 0, "name": "lightweight_fbx",
                 "status": "SUCCESS", "result": {"diagnostics": "x" * (2 * 1024 * 1024)}}
        report = {"status": "NEEDS_REVIEW", "stages": [stage]}
        self.state(job["id"], state="NEEDS_REVIEW", stages=[stage])
        atomic_json(self.manager.jobs_dir / job["id"] / "report.json", report)
        public = self.manager.jobs()
        self.assertEqual(public["jobs"][0]["stages"][0]["status"], "SUCCESS")
        self.assertLess(len(json.dumps(public).encode("utf-8")), 8192)
        self.assertEqual(self.manager.report(job["id"])["report"], report)

    def test_logs_use_bounded_byte_offsets_and_report_is_real(self):
        job = self.enqueue()
        directory = self.manager.jobs_dir / job["id"]
        (directory / "processing.log").write_bytes(b"x" * 70000)
        self.assertEqual(self.manager.logs(job["id"], 0)["next_offset"], 65536)
        self.assertEqual(len(self.manager.logs(job["id"], 65536)["text"]), 4464)
        self.assertEqual(self.manager.logs(job["id"], 999999)["next_offset"], 70000)
        self.assertEqual(self.manager.report(job["id"]), {"report": None})
        atomic_json(directory / "report.json", {"status": "NEEDS_REVIEW", "quality": {"usable": False}})
        self.assertEqual(self.manager.report(job["id"])["report"]["status"], "NEEDS_REVIEW")

    def test_capture_time_dumps_are_ignored_and_rejected(self):
        dump = self.data / "longyufeng_2026-11-04_13-58-29.mcp"
        dump.write_bytes(b"raw capture dump")
        deep = self.manager.browse(recursive=True)
        self.assertNotIn(dump.name, [item["name"] for item in deep["sequences"]])
        with self.assertRaises(APIError) as error:
            self.manager.enqueue({"paths": [str(dump)]})
        self.assertEqual(error.exception.status, 403)
        self.assertIn("take.fbx", [item["name"] for item in deep["sequences"]])

    def test_superseded_terminal_jobs_are_purged(self):
        first = self.enqueue()
        self.manager._set_status(self.manager._jobs[first["id"]], "NEEDS_REVIEW")
        second = self.manager.enqueue({"paths": [first["input_path"]]})["jobs"][0]
        with self.manager._lock:
            self.manager._set_status(self.manager._jobs[second["id"]], "SUCCESS")
            self.manager._purge_superseded(self.manager._jobs[second["id"]])
        self.assertEqual([job["id"] for job in self.manager.jobs()["jobs"]], [second["id"]])
        self.assertFalse((self.manager.jobs_dir / first["id"]).exists())

    def test_approve_flips_needs_review_to_success_and_marks_report(self):
        job = self.enqueue()
        self.manager._set_status(self.manager._jobs[job["id"]], "NEEDS_REVIEW")
        report_path = self.manager.jobs_dir / job["id"] / "report.json"
        atomic_json(report_path, {"status": "NEEDS_REVIEW", "quality": {"usable": False}})
        result = self.manager.action(job["id"], {"action": "approve", "note": "RTHM6 复核无问题"})
        self.assertEqual(result["job"]["state"], "SUCCESS")
        self.assertEqual(result["job"]["review"]["note"], "RTHM6 复核无问题")
        report = self.manager.report(job["id"])["report"]
        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(report["review"]["note"], "RTHM6 复核无问题")
        self.assertTrue(report["quality"]["usable"])
        self.assertEqual(report["quality"]["scope"], "approved_override")
        with self.assertRaises(APIError) as error:
            self.manager.action(job["id"], {"action": "approve", "note": ""})
        self.assertEqual(error.exception.status, 409)

    def test_missing_worker_is_an_inspectable_failure(self):
        job = self.enqueue()
        self.manager.worker_script = self.root / "missing_worker.py"
        self.manager.tick()
        dto = self.manager.job(job["id"])["job"]
        self.assertEqual(dto["state"], "FAILED")
        self.assertIn("Worker script not found", dto["error"])


class HTTPTests(ManagerFixture):
    def setUp(self):
        super().setUp()
        self.server = ProcessingServer(("127.0.0.1", 0), self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        request_headers = headers.copy() if headers else {}
        if body is not None:
            body = json.dumps(body)
            request_headers.setdefault("Content-Type", "application/json")
        connection.request(method, "/api/processing" + path, body, request_headers)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_http_rejects_hostile_origin_host_and_missing_token(self):
        for headers in ({"Origin": "https://evil.example"}, {"Host": "evil.example:8767"},
                        {"Origin": "null"}, {"Origin": "http://localhost:3001"},
                        {"Origin": "http://localhost:5173"}):
            self.assertEqual(self.request("GET", "/config", headers=headers)[0], 403)
        self.assertEqual(self.request("POST", "/jobs", {"paths": [str(self.source)]})[0], 403)
        origin = f"http://localhost:{self.server.server_address[1]}"
        status, _, data = self.request("GET", "/config", headers={"Origin": origin})
        self.assertEqual(status, 200)
        token = json.loads(data)["csrf_token"]
        status, _, data = self.request("POST", "/jobs", {"paths": [str(self.source)]}, {"X-Mocap-Token": token})
        self.assertEqual(status, 201)
        self.assertEqual(json.loads(data)["jobs"][0]["state"], "QUEUED")

    def test_http_delete_output_route_works_and_protects_sources(self):
        results = self.data / "02_processed" / "body"
        results.mkdir(parents=True)
        mirror = results / "take.fbx"
        mirror.write_bytes(b"mirror")
        origin = f"http://localhost:{self.server.server_address[1]}"
        token = json.loads(self.request("GET", "/config", headers={"Origin": origin})[2])["csrf_token"]
        denied = self.request("POST", "/delete-output", {"path": str(self.source)},
                              {"Origin": origin, "X-Mocap-Token": token})
        self.assertEqual(denied[0], 403)
        self.assertTrue(self.source.exists())
        status, _, data = self.request("POST", "/delete-output", {"path": str(mirror)},
                                       {"Origin": origin, "X-Mocap-Token": token})
        self.assertEqual(status, 200, data)
        self.assertEqual(json.loads(data)["deleted"], str(mirror))
        self.assertFalse(mirror.exists())

    def test_http_artifact_range_head_and_invalid_ranges(self):
        job = self.enqueue()
        artifact = self.manager.jobs_dir / job["id"] / "output.fbx"
        artifact.write_bytes(b"0123456789")
        self.state(job["id"], artifacts=[{"id": "output", "name": artifact.name, "role": "fbx", "path": str(artifact)}])
        url = f"/artifacts/{job['id']}/output"
        status, headers, data = self.request("GET", url, headers={"Range": "bytes=2-5"})
        self.assertEqual((status, data), (206, b"2345"))
        self.assertEqual(headers["Content-Range"], "bytes 2-5/10")
        self.assertEqual(self.request("GET", url, headers={"Range": "bytes=-3"})[2], b"789")
        status, headers, data = self.request("HEAD", url)
        self.assertEqual((status, headers["Content-Length"], data), (200, "10", b""))
        for value in ("bytes=99-", "bytes=5-2", "bytes=0-1,3-4", "bytes=-0", "garbage"):
            with self.subTest(value=value):
                self.assertEqual(self.request("GET", url, headers={"Range": value})[0], 416)
        self.assertEqual(self.request("GET", f"/artifacts/{job['id']}/request.json")[0], 404)

    def test_http_reports_logs_and_busy_actions(self):
        job = self.enqueue()
        self.assertEqual(json.loads(self.request("GET", f"/jobs/{job['id']}/report")[2]), {"report": None})
        self.assertEqual(self.request("GET", f"/jobs/{job['id']}/logs?offset=-1")[0], 400)
        self.assertEqual(self.request("GET", f"/jobs/{job['id']}/logs?offset=bad")[0], 400)
        status, _, _ = self.request("POST", f"/jobs/{job['id']}/actions", {"action": "step"}, {"X-Mocap-Token": self.manager.csrf_token})
        self.assertEqual(status, 409)
        self.assertEqual(self.request("GET", "/browse?recursive=invalid")[0], 400)


class AuthorityTests(unittest.TestCase):
    def test_exact_loopback_hosts_and_ports(self):
        for host in ("127.0.0.1:8767", "localhost:3001", "[::1]:5173"):
            self.assertTrue(allowed_authority(host, {8767, 3001, 5173}))
        for host in ("localhost.evil:8767", "evil@localhost:8767", "127.0.0.1:9", "localhost:8767/path", "localhost:8767#bad"):
            self.assertFalse(allowed_authority(host, {8767, 3001, 5173}))


if __name__ == "__main__":
    unittest.main()

    def test_output_deletion_allowed_only_under_result_trees(self):
        results = self.data / '02_processed' / 'body'
        results.mkdir(parents=True)
        mirror = results / 'take.fbx'
        mirror.write_bytes(b'archive copy')
        tree = self.manager.tree(str(results))
        entry = next(item for item in tree['entries'] if item['name'] == 'take.fbx')
        self.assertTrue(entry['deletable'])
        source_tree = self.manager.tree(None)
        source_entry = next(item for item in source_tree['entries'] if item['name'] == 'take.fbx')
        self.assertFalse(source_entry['deletable'])
        with self.assertRaises(APIError) as error:
            self.manager.delete_output(str(self.source))
        self.assertEqual(error.exception.status, 403)
        self.assertTrue(self.source.exists())
        result = self.manager.delete_output(str(mirror))
        self.assertEqual(result['deleted'], str(mirror))
        self.assertFalse(mirror.exists())

    def test_outputs_lists_recent_result_tree_files(self):
        for folder, name in (('02_processed', 'take.fbx'), ('03_exports', 'take_display.fbx'), ('01_data_raw', 'hidden.fbx')):
            directory = self.data / folder
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_bytes(b'x')
        result = self.manager.outputs()
        names = [item['name'] for item in result['outputs']]
        self.assertEqual(names, ['take_display.fbx', 'take.fbx'])
        self.assertTrue(all(item['deletable'] for item in result['outputs']))

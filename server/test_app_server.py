"""Standalone HTTP regressions using temporary assets/state; no native workers.

Run: python -m unittest discover -s H:/MocapStudio/server -p test_app_server.py
"""
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import unittest
from urllib.parse import quote

import app_server
from web_manager import JobManager


class AppServerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.public = self.root / "public"
        self.raw = self.root / "raw"
        for directory in (self.public, self.raw):
            directory.mkdir()
        self.put(self.public / "index.html", b"<h1>Legacy dashboard</h1>")
        self.put(self.public / "workbench/dist/index.html", b"<h1>Standalone MocapStudio</h1>")
        self.put(self.public / "workbench/dist/assets/main.js", b"export const workbench = true;")
        self.put(self.public / "workbench/src/private.ts", b"not public")
        self.put(self.public / "workbench/dist/assets/leaked.map", b"not public")
        self.put(self.public / "viewer/fbx_viewer.html", b"<script type=module>/*viewer*/</script>")
        self.put(self.public / "viewer/vendor/three/build/three.module.js", b"0123456789")
        self.put(self.public / "viewer/private.json", b"not public")
        self.put(self.public / "styles/style.css", b"body { color: green }")
        self.put(self.public / "scripts/app.js", b"/*legacy*/")
        self.put(self.public / "fonts/font.woff2", b"font")
        self.put(self.public / "server/private.py", b"not public")
        self.put(self.public / "data/runs/private.json", b"not public")
        self.put(self.public / "node_modules/private.js", b"not public")
        self.input = self.raw / "take.fbx"
        self.put(self.input, b"source must not change")

    @staticmethod
    def put(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    @contextmanager
    def serving(self, *, origins=(), state=None):
        manager = JobManager(state or self.root / "state", [self.raw])
        server = app_server.AppServer(0, manager, app_root=self.public, additional_origins=origins)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        # Deliberately do not start the scheduler: HTTP enqueue is real, but no
        # SDK, Shogun, Blender or worker process may run in these regressions.
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def request(self, server, path, *, method="GET", headers=None, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, server, path, payload, *, headers=None):
        values = {"Content-Type": "application/json", "X-Mocap-Token": server.manager.csrf_token}
        values.update(headers or {})
        return self.request(server, path, method="POST", headers=values, body=json.dumps(payload))

    def test_former_dashboard_entry_points_redirect_to_workbench(self):
        with self.serving() as server:
            for path in ("/workbench", "/workbench/", "/workbench/index.html"):
                status, headers, body = self.request(server, path)
                self.assertEqual(status, 200, path)
                self.assertIn(b"Standalone MocapStudio", body)
                self.assertNotIn(b"Legacy dashboard", body)
                self.assertIn("frame-src 'self'", headers["Content-Security-Policy"])
            for path in ("/", "/index.html", "/monitor", "/monitor/", "/monitor/index.html"):
                status, headers, body = self.request(server, path)
                self.assertEqual((status, headers["Location"]), (302, "/workbench/"), path)
                self.assertNotIn(b"Legacy dashboard", body)
            status, headers, _ = self.post(server, "/", {})
            self.assertEqual(status, 405)
            self.assertEqual(headers["Allow"], "GET, HEAD")
            for path in ("/api/runs", "/api/overview", "/api/takes/unknown", "/api/artifacts/unknown/file"):
                self.assertEqual(self.request(server, path)[0], 404, path)
            # The legacy static namespaces are gone; leftover files are never served.
            for path in ("/styles/style.css", "/scripts/app.js", "/fonts/font.woff2"):
                self.assertEqual(self.request(server, path)[0], 404, path)

    def test_viewer_policy_modules_range_and_head(self):
        with self.serving() as server:
            status, headers, _ = self.request(server, "/pipeline-viewer/fbx_viewer.html?embed=1")
            self.assertEqual(status, 200)
            self.assertEqual(headers["X-Frame-Options"], "SAMEORIGIN")
            self.assertIn("frame-ancestors 'self'", headers["Content-Security-Policy"])
            self.assertIn("script-src 'self' 'unsafe-inline'", headers["Content-Security-Policy"])
            self.assertIn("img-src 'self' data: blob:", headers["Content-Security-Policy"])
            path = "/pipeline-viewer/vendor/three/build/three.module.js"
            status, headers, payload = self.request(server, path, headers={"Range": "bytes=2-5"})
            self.assertEqual((status, payload), (206, b"2345"))
            self.assertEqual(headers["Content-Range"], "bytes 2-5/10")
            status, headers, payload = self.request(server, path, method="HEAD", headers={"Range": "bytes=-3"})
            self.assertEqual((status, payload), (206, b""))
            self.assertEqual(headers["Content-Length"], "3")
            for value in ("bytes=10-", "bytes=0-1,3-4", "bytes=-0"):
                status, headers, _ = self.request(server, path, headers={"Range": value})
                self.assertEqual(status, 416)
                self.assertEqual(headers["Content-Range"], "bytes */10")

    def test_public_namespaces_cannot_expose_sources_or_state(self):
        denied = ("/server/private.py", "/data/runs/private.json", "/node_modules/private.js",
                  "/workbench/src/private.ts", "/workbench/package.json", "/workbench/dist/index.html",
                  "/workbench/assets/leaked.map", "/pipeline-viewer/private.json",
                  "/pipeline-viewer/vendor/three/../../../../server/private.py",
                  "/styles/%2e%2e/server/private.py", "/styles/%2fserver/private.py",
                  "/styles/%5cserver/private.py", "/styles/style.css:secret", "/styles/%ZZ")
        with self.serving() as server:
            self.assertEqual(self.request(server, "/workbench/assets/main.js")[0], 200)
            for path in denied:
                status, _, body = self.request(server, path)
                self.assertIn(status, (400, 403, 404), path)
                self.assertNotIn(b"not public", body, path)

    def test_link_outside_public_namespace_is_not_served(self):
        outside = self.root / "outside.js"
        self.put(outside, b"secret outside root")
        link = self.public / "workbench/dist/assets/link.js"
        try:
            link.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"Creating symlinks requires OS permission: {exc}")
        with self.serving() as server:
            status, _, body = self.request(server, "/workbench/assets/link.js")
            self.assertIn(status, (400, 403))
            self.assertNotIn(b"secret outside root", body)

    def test_pipeline_posts_use_real_queue_and_leave_input_unchanged(self):
        original = self.input.read_bytes()
        with self.serving() as server:
            status, _, payload = self.request(server, "/api/processing/config")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["viewer_base"], "/pipeline-viewer/fbx_viewer.html")
            status, _, payload = self.post(server, "/api/processing/config", {"max_workers": 2})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["max_workers"], 2)
            status, _, payload = self.post(server, "/api/processing/jobs", {"paths": [str(self.input)], "mode": "manual"})
            self.assertEqual(status, 201)
            job = json.loads(payload)["jobs"][0]
            self.assertEqual(job["state"], "QUEUED")
            self.assertTrue((server.manager.jobs_dir / job["id"] / "request.json").is_file())
            status, _, payload = self.post(server, f"/api/processing/jobs/{job['id']}/actions", {"action": "cancel"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["job"]["state"], "CANCELLED")
        self.assertEqual(self.input.read_bytes(), original)

    def test_tree_and_find_honor_roots_exclusions_and_limits(self):
        self.put(self.raw / "group" / "G01-MP-S-WALK.fbx", b"sequence one")
        self.put(self.raw / "group" / "G01-MP-S-WALK.mcp", b"mcp payload")
        self.put(self.raw / "processed" / "hidden.fbx", b"excluded")
        self.put(self.raw / "node_modules" / "junk.fbx", b"excluded")
        outside = self.root / "outside-outside.fbx"
        self.put(outside, b"outside")
        with self.serving() as server:
            status, _, payload = self.request(server, f"/api/processing/tree?path={self.raw}")
            self.assertEqual(status, 200)
            result = json.loads(payload)
            self.assertEqual([entry["name"] for entry in result["entries"] if entry["kind"] == "dir"], ["group"])
            self.assertEqual([entry["name"] for entry in result["entries"] if entry["kind"] != "dir"], ["take.fbx"])
            self.assertFalse(result["truncated"])
            status, _, payload = self.request(server, f"/api/processing/tree?path={self.raw / 'group'}")
            self.assertEqual(status, 200)
            entries = json.loads(payload)["entries"]
            self.assertEqual({entry["name"] for entry in entries}, {"G01-MP-S-WALK.fbx", "G01-MP-S-WALK.mcp"})
            mcp = next(entry for entry in entries if entry["kind"] == "mcp")
            self.assertTrue(mcp["x2d_exists"] is False and mcp["size_bytes"] == len(b"mcp payload"))
            for escaped in (str(outside), str(self.root / "state"), str(self.public)):
                status, _, _ = self.request(server, f"/api/processing/tree?path={escaped}")
                self.assertEqual(status, 403, escaped)
            status, _, payload = self.request(server, f"/api/processing/find?q=walk")
            self.assertEqual(status, 200)
            result = json.loads(payload)
            self.assertEqual({match["path"] for match in result["matches"]},
                             {str(self.raw / "group" / "G01-MP-S-WALK.fbx"), str(self.raw / "group" / "G01-MP-S-WALK.mcp")})
            self.assertFalse(result["truncated"])
            for forbidden in ("hidden", "junk", "outside-outside"):
                status, _, payload = self.request(server, f"/api/processing/find?q={forbidden}")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(payload)["matches"], [], forbidden)
            self.assertEqual(self.request(server, "/api/processing/find")[0], 400)
            self.assertEqual(self.request(server, "/api/processing/find?q=")[0], 400)

    def test_registered_processing_artifact_supports_viewer_range(self):
        with self.serving() as server:
            status, _, payload = self.post(server, "/api/processing/jobs", {"paths": [str(self.input)]})
            self.assertEqual(status, 201)
            identifier = json.loads(payload)["jobs"][0]["id"]
            artifact = server.manager.jobs_dir / identifier / "preview.fbx"
            self.put(artifact, b"0123456789")
            self.put(artifact.parent / "state.json", json.dumps({
                "artifacts": [{"id": "preview", "path": str(artifact), "name": "preview.fbx", "role": "fbx"}]
            }).encode())
            path = f"/api/processing/artifacts/{identifier}/preview"
            status, headers, payload = self.request(server, path, headers={"Range": "bytes=1-3"})
            self.assertEqual((status, payload), (206, b"123"))
            self.assertEqual(headers["Content-Range"], "bytes 1-3/10")
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(self.request(server, f"/api/processing/artifacts/{identifier}/unregistered")[0], 404)

    def test_host_origin_token_and_content_type_restrictions(self):
        with self.serving() as server:
            for path in ("/", "/monitor/", "/api/processing/config", "/pipeline-viewer/fbx_viewer.html"):
                for host in ("evil.example", "127.0.0.1:3001", "localhost:5173", "127.0.0.1:8767"):
                    self.assertEqual(self.request(server, path, headers={"Host": host})[0], 403, (path, host))
                self.assertEqual(self.request(server, path, headers={"Origin": "https://evil.example"})[0], 403)
            payload = {"max_workers": 2}
            for origin in ("http://localhost:3001", "http://localhost:5173", "null", ""):
                self.assertEqual(self.post(server, "/api/processing/config", payload, headers={"Origin": origin})[0], 403)
            for token in ("", "wrong"):
                self.assertEqual(self.post(server, "/api/processing/config", payload, headers={"X-Mocap-Token": token})[0], 403)
            self.assertEqual(self.post(server, "/api/processing/config", payload, headers={"Content-Type": "text/plain"})[0], 415)
            good_origin = f"http://127.0.0.1:{server.server_address[1]}"
            self.assertEqual(self.post(server, "/api/processing/config", payload, headers={"Origin": good_origin})[0], 200)
            self.assertEqual(self.request(server, "http://evil.example/api/processing/config")[0], 400)
            self.assertEqual(self.request(server, "/api/processing/config#fragment")[0], 400)

    def test_duplicate_host_and_origin_are_rejected(self):
        with self.serving() as server:
            port = server.server_address[1]
            for duplicates in (f"Host: 127.0.0.1:{port}\r\n", f"Origin: http://127.0.0.1:{port}\r\nOrigin: http://127.0.0.1:{port}\r\n"):
                with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
                    connection.sendall((f"GET /api/processing/config HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{duplicates}Connection: close\r\n\r\n").encode("ascii"))
                    self.assertIn(b" 403 ", connection.recv(4096).split(b"\r\n", 1)[0])

    def test_additional_origin_is_explicit_and_does_not_relax_host(self):
        origin = "http://localhost:5174"
        with self.serving(origins=[origin]) as server:
            self.assertEqual(self.post(server, "/api/processing/config", {"max_workers": 2}, headers={"Origin": origin})[0], 200)
            self.assertEqual(self.request(server, "/api/processing/config", headers={"Host": "localhost:5174", "Origin": origin})[0], 403)
        for invalid in ("*", "http://localhost:5174/", "http://user@localhost:5174", "file:///tmp"):
            with self.assertRaises(ValueError):
                app_server.explicit_origins([invalid])

    def test_missing_workbench_explains_build_without_breaking_processing(self):
        (self.public / "workbench/dist/index.html").unlink()
        with self.serving() as server:
            self.assertEqual(self.request(server, "/")[0], 302)
            status, _, payload = self.request(server, "/workbench/")
            self.assertEqual(status, 503)
            self.assertIn("build", json.loads(payload)["error"])
            self.assertEqual(self.request(server, "/api/processing/config")[0], 200)

    def test_terminal_jobs_are_preserved_and_server_close_releases_manager_lock(self):
        state = self.root / "existing-state"
        expected = {}
        for number in range(5):
            identifier = f"{number + 1:032x}"
            path = state / "jobs" / identifier / "metadata.json"
            metadata = {"id": identifier, "state": "SUCCESS", "name": self.input.name,
                        "input_path": str(self.input), "input_kind": "fbx", "mode": "manual", "options": {},
                        "created_at": "2026-11-18T00:00:00+00:00", "updated_at": "2026-11-18T00:00:00+00:00", "error": None}
            self.put(path, json.dumps(metadata).encode())
            expected[path] = path.read_bytes()
        with self.serving(state=state) as server:
            status, _, payload = self.request(server, "/api/processing/jobs")
            self.assertEqual(status, 200)
            self.assertEqual(len(json.loads(payload)["jobs"]), 5)
        for path, content in expected.items():
            self.assertEqual(path.read_bytes(), content)
        reopened = JobManager(state, [self.raw])
        reopened.close()

    def test_queue_delete_clear_finished_and_active_guard(self):
        state = self.root / "queue-state"
        states = ("SUCCESS", "FAILED", "QUEUED")
        identifiers = {}
        for index, job_state in enumerate(states):
            identifier = f"{index + 20:032x}"
            identifiers[job_state] = identifier
            self.put(state / "jobs" / identifier / "metadata.json", json.dumps({
                "id": identifier, "state": job_state, "name": f"{job_state.lower()}.mcp",
                "input_path": str(self.raw / "take.mcp"), "input_kind": "mcp", "mode": "auto", "options": {},
                "created_at": "2026-11-25T00:00:00+00:00", "updated_at": "2026-11-25T00:00:00+00:00", "error": None,
            }).encode())
        with self.serving(state=state) as server:
            self.assertEqual(self.post(server, f"/api/processing/jobs/{identifiers['QUEUED']}/delete", {})[0], 409)
            self.assertTrue((state / "jobs" / identifiers["QUEUED"]).is_dir())
            status, _, body = self.post(server, f"/api/processing/jobs/{identifiers['SUCCESS']}/delete", {})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["removed"], identifiers["SUCCESS"])
            self.assertFalse((state / "jobs" / identifiers["SUCCESS"]).exists())
            status, _, body = self.post(server, "/api/processing/jobs/clear-finished", {})
            self.assertEqual(json.loads(body)["count"], 1)
            _, _, body = self.request(server, "/api/processing/jobs")
            self.assertEqual([job["state"] for job in json.loads(body)["jobs"]], ["QUEUED"])
            self.assertEqual(self.post(server, f"/api/processing/jobs/{'f' * 32}/delete", {})[0], 404)

    def test_inspect_scans_subjects_reports_notes_and_guards(self):
        def record(name, bone="Hips"):
            return struct.pack("<I", len(name)) + name.encode() + b"\x1f\x6a\xe7\x34\x44\x00\x00\x00" + struct.pack("<I", len(bone)) + bone.encode()
        self.put(self.raw / "take.mcp", b"ViconMCPFile" + bytes(64) + record("sunjian") + bytes(120) + record("yanjie")
                 + record("desk", "root") + record("white_trash", "root"))
        with self.serving() as server:
            status, _, body = self.request(server, f"/api/processing/inspect?path={quote(str(self.raw / 'take.mcp'))}")
            self.assertEqual(status, 200)
            payload = json.loads(body)
            self.assertEqual(payload["kind"], "mcp")
            self.assertEqual([row["name"] for row in payload["subjects"]], ["sunjian", "yanjie"])
            self.assertIn("mcp_scan", payload["subjects"][0]["sources"])
            self.assertEqual([row["name"] for row in payload["objects"]], ["desk", "white_trash"])
            # The fixture FBX is not real FBX binary: parsing must degrade to a note.
            status, _, body = self.request(server, f"/api/processing/inspect?path={quote(str(self.input))}")
            self.assertEqual(status, 200)
            payload = json.loads(body)
            self.assertEqual(payload["kind"], "fbx")
            self.assertTrue(any("FBX" in note for note in payload["notes"]))
            # Guards: outside roots, missing parameter, and engine state paths are refused.
            self.assertEqual(self.request(server, "/api/processing/inspect?path=C:/Windows/system.ini")[0], 403)
            self.assertEqual(self.request(server, "/api/processing/inspect")[0], 400)
            self.assertEqual(self.request(server, f"/api/processing/inspect?path={quote(str(self.root / 'nonexistent.fbx'))}")[0], 404)

    def test_inspect_picks_up_reports_and_job_states_immediately(self):
        state = self.root / "roster-state"
        identifier = "0" * 32
        self.put(state / "jobs" / identifier / "metadata.json", json.dumps({
            "id": identifier, "state": "SUCCESS", "name": self.input.name,
            "input_path": str(self.input), "input_kind": "fbx", "mode": "manual", "options": {},
            "created_at": "2026-11-25T00:00:00+00:00", "updated_at": "2026-11-25T00:00:00+00:00", "error": None,
        }).encode())
        path = f"/api/processing/inspect?path={quote(str(self.input))}"
        with self.serving(state=state) as server:
            _, _, body = self.request(server, path)
            payload = json.loads(body)
            self.assertEqual([run["job_id"] for run in payload["runs"]], [identifier])
            self.assertNotIn("report_job", payload)
            self.assertEqual(payload["objects"], [])
            # A report published after the first inspect call must show up at once.
            self.put(state / "jobs" / identifier / "report.json", json.dumps({
                "status": "SUCCESS", "subjects": [{"name": "desk", "rig_class": "PROP_OR_NONHUMAN",
                                                   "setup": {"markers": 4}}], "props": [],
            }).encode())
            _, _, body = self.request(server, path)
            payload = json.loads(body)
            self.assertEqual(payload["report_job"], identifier)
            self.assertEqual([row["name"] for row in payload["objects"]], ["desk"])
            self.assertIn("run_report", payload["objects"][0]["sources"])


if __name__ == "__main__":
    unittest.main()

"""Worker regressions use real binary FBX files; no Shogun/native jobs are launched."""
from contextvars import copy_context
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from pipeline_processing import _skip, _stage, stage_observer
from pipeline_run_registry import open_shared_read
from test_fbx_lightweight import _write_source
from web_worker import CommandReader, Worker, atomic_json, dataset_identity, preflight_fbx, source_quality


class RecordingObserver:
    def __init__(self):
        self.events = []

    def before_stage(self, report, entry):
        self.events.append(("before", entry["name"], entry["status"]))

    def after_stage(self, report, entry):
        self.events.append(("after", entry["name"], entry["status"]))


class ObserverTests(unittest.TestCase):
    def test_scoped_nested_observers_and_unchanged_unobserved_report(self):
        report = {"stages": []}
        outer, inner = RecordingObserver(), RecordingObserver()
        with stage_observer(outer):
            with _stage(report, "outer"):
                with stage_observer(inner):
                    with _stage(report, "inner", "subject"):
                        pass
                _skip(report, "repair", "not_needed")
            isolated = copy_context()
        with _stage(report, "unobserved"):
            pass
        self.assertEqual(outer.events, [
            ("before", "outer", "RUNNING"), ("after", "repair", "SKIPPED"),
            ("after", "outer", "SUCCESS")])
        self.assertEqual(inner.events, [("before", "inner", "RUNNING"), ("after", "inner", "SUCCESS")])
        self.assertEqual(set(report["stages"][-1]), {"name", "status", "elapsed_seconds"})
        self.assertNotIn("observer", json.dumps(report))
        isolated.run(lambda: _skip(report, "captured_context", "context_local"))
        self.assertEqual(outer.events[-1], ("after", "captured_context", "SKIPPED"))

    def test_observer_sees_failures_and_context_restores_after_exception(self):
        report = {"stages": []}
        observer = RecordingObserver()
        with self.assertRaisesRegex(RuntimeError, "real stage failure"):
            with stage_observer(observer):
                with _stage(report, "failing"):
                    raise RuntimeError("real stage failure")
        with _stage(report, "unobserved"):
            pass
        self.assertEqual(observer.events[-1], ("after", "failing", "FAILED"))
        self.assertEqual(report["stages"][0]["error"], "RuntimeError: real stage failure")


class CommandAndArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.fbx"
        _write_source(self.source, labeling=True)
        self.directory = self.root / "run"

    def request(self, **values):
        return {"job_id": "worker-test", "input_path": str(self.source),
                "job_dir": str(self.directory), "mode": "manual", **values}

    def test_incremental_commands_wait_for_complete_lines_and_ignore_replays(self):
        path = self.root / "commands.jsonl"
        reader = CommandReader(path)
        self.assertEqual(reader.read(), [])
        path.write_bytes(b'{"seq":1,"action":"step"')
        self.assertEqual(reader.read(), [])
        with path.open("ab") as stream:
            stream.write(b'}\n{"seq":1,"action":"step"}\n{"seq":2,"action":"run"}\n')
        self.assertEqual(reader.read(), [{"seq": 1, "action": "step"}, {"seq": 2, "action": "run"}])
        self.assertEqual(reader.read(), [])

    def test_artifact_scope_distinct_roles_and_no_checkpoint_resume(self):
        worker = Worker(self.request())
        with self.assertRaises(ValueError):
            worker.register(self.source, "outside")
        source_id = worker.register(self.source, "source_fbx", source=True)
        final_id = worker.register(self.source, "fbx", source=True)
        self.assertNotEqual(source_id, final_id)
        self.assertEqual(source_id, worker.register(self.source, "source_fbx", source=True))
        with self.assertRaisesRegex(ValueError, "Refusing to resume"):
            Worker(self.request())

    def test_command_ack_reserves_stage_and_pause_is_boundary_only(self):
        worker = Worker(self.request())
        worker.state["state"] = "WAITING"
        worker.apply_command({"seq": 1, "action": "step"})
        state = json.loads((self.directory / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["command_seq"], 1)
        self.assertEqual(state["state"], "RUNNING")
        worker.apply_command({"seq": 2, "action": "step"})
        self.assertEqual(worker.permits, 1)
        worker.apply_command({"seq": 3, "action": "pause"})
        self.assertEqual(worker.state["state"], "PAUSE_REQUESTED")
        self.assertFalse(worker.continuous)
        self.assertEqual(worker.permits, 0)
        self.assertEqual(worker.state["command_seq"], 3)

    def test_real_preflight_and_quality_requires_declared_source(self):
        counts = preflight_fbx(self.source)
        self.assertEqual(counts["fbx_version"], 7700)
        self.assertGreater(counts["model_counts"]["Marker"], 0)
        self.assertGreater(counts["object_counts"]["AnimationCurve"], 0)
        atomic_json(self.root / "report.json", {"status": "SUCCESS", "quality": {"usable": True},
                                               "artifacts": {"fbx": "different.fbx"}})
        self.assertEqual(source_quality(self.source)[0], "NEEDS_REVIEW")
        atomic_json(self.root / "report.json", {"status": "SUCCESS", "quality": {"usable": True},
                                               "artifacts": {"fbx": self.source.name}})
        self.assertEqual(source_quality(self.source)[0], "SUCCESS")

    def test_dataset_identity_uses_actual_ancestor_hierarchy(self):
        root, mode = dataset_identity(self.root / "group11" / "HOI" / "take" / "source.mcp")
        self.assertEqual(root, self.root)
        self.assertEqual(mode, "HOI")
        self.assertEqual(dataset_identity(self.root / "group11-HOI-source.mcp"), (None, None))
        self.assertEqual(dataset_identity(self.root / "group11" / "other" / "HOI" / "source.mcp"), (None, None))
        root, mode = dataset_identity(self.root / "01_data_raw" / "body" / "actor001" / "C47" / "B001-C47.mcp")
        self.assertEqual((root, mode), (self.root / "01_data_raw", "BODY"))
        self.assertEqual(dataset_identity(self.root / "01_data_raw" / "bodylike" / "x" / "s.mcp"), (None, None))


class RealFbxWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.fbx"
        _write_source(self.source, labeling=True)
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.directory = self.root / "run"
        self.directory.mkdir()
        self.seq = 0
        self.process = None
        self.addCleanup(self.stop_worker)

    def stop_worker(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)

    def start_worker(self, mode="manual", **options):
        request = {"job_id": "real-fbx", "input_path": str(self.source),
                   "job_dir": str(self.directory), "mode": mode,
                   "data_roots": [str(self.root)],
                   "options": {"lightweight": True, **options}}
        atomic_json(self.directory / "request.json", request)
        with (self.directory / "console.log").open("wb") as console:
            self.process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("web_worker.py")),
                                             "--request", str(self.directory / "request.json")],
                                            stdin=subprocess.DEVNULL, stdout=console,
                                            stderr=subprocess.STDOUT)

    def command(self, action):
        self.seq += 1
        with (self.directory / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"seq": self.seq, "action": action}) + "\n")
        return self.seq

    def await_state(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                with open_shared_read(self.directory / "state.json") as stream:
                    last = json.load(stream)
                if predicate(last):
                    return last
            except FileNotFoundError:
                pass
            if self.process.poll() is not None:
                console = (self.directory / "console.log").read_text(encoding="utf-8", errors="replace")
                self.fail(f"Worker exited {self.process.returncode} before requested state: {last}\n{console}")
            time.sleep(0.02)
        self.fail(f"Worker did not reach requested state: {last}")

    def test_real_manual_stage_gate_preview_then_lossless_output(self):
        self.start_worker()
        first = self.await_state(lambda state: state["state"] == "WAITING" and
                                state["waiting_stage"] == "0002-fbx_preflight")
        self.assertEqual(first["stages"][0]["name"], "parse")
        self.assertEqual(first["stages"][0]["status"], "SUCCESS")
        self.assertEqual(first["stages"][0]["result"]["kind"], "fbx")
        self.assertEqual(first["stages"][1]["status"], "WAITING")
        self.assertIn("source_fbx", [artifact["role"] for artifact in first["artifacts"]])
        self.command("preview")
        failed_preview = self.await_state(lambda value: value["state"] == "WAITING" and
                                         bool(value.get("error")))
        self.assertIn("Preview export failed", failed_preview["error"])
        self.assertEqual(failed_preview["stages"][1]["status"], "WAITING")
        self.assertIn("Preview export failed",
                      (self.directory / "processing.log").read_text(encoding="utf-8"))
        self.command("step")
        state = self.await_state(lambda value: value["state"] == "WAITING" and
                                 value["waiting_stage"] == "0003-lightweight_fbx")
        self.assertEqual(state["stages"][1]["status"], "SUCCESS")
        self.assertFalse((self.directory / "display.fbx").exists())
        partial = json.loads((self.directory / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(partial["status"], "RUNNING")
        self.assertTrue(partial["partial"])
        self.command("preview")
        previewed = self.await_state(lambda value: value["state"] == "WAITING" and
                                    any(item["role"] == "intermediate_fbx" for item in value["artifacts"]))
        self.assertIsNone(previewed["error"])
        self.command("step")
        final = self.await_state(lambda value: value["state"] == "NEEDS_REVIEW")
        self.assertEqual([entry["status"] for entry in final["stages"]], ["SUCCESS"] * 3)
        self.assertEqual(final["command_seq"], self.seq)
        self.process.wait(timeout=10)
        self.assertEqual(self.process.returncode, 0)
        report = json.loads((self.directory / "report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["partial"])
        self.assertTrue(report["lightweight"]["retained_animation_payloads_exact"])
        self.assertFalse(report["lightweight"]["physical_motion_repaired"])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)
        self.assertEqual(len(list((self.directory / "reports").glob("*.json"))), 3)
        self.assertIn("export_fbx", [artifact["role"] for artifact in final["artifacts"]])
        self.assertIn("fbx", [artifact["role"] for artifact in final["artifacts"]])

    def test_cancel_waiting_retains_partial_report_without_processing(self):
        self.start_worker()
        self.await_state(lambda value: value["state"] == "WAITING")
        self.command("cancel")
        state = self.await_state(lambda value: value["state"] == "CANCELLED")
        self.assertEqual(state["stages"][0]["name"], "parse")
        self.assertEqual(state["stages"][0]["status"], "SUCCESS")
        self.assertEqual(state["stages"][1]["status"], "SKIPPED")
        self.assertFalse((self.directory / "display.fbx").exists())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_run_releases_all_remaining_real_stages(self):
        self.start_worker()
        self.await_state(lambda value: value["state"] == "WAITING")
        self.command("run")
        state = self.await_state(lambda value: value["state"] == "NEEDS_REVIEW")
        self.assertEqual(len(state["stages"]), 3)
        self.assertFalse((self.directory / "display.fbx").exists())
        report = json.loads((self.directory / "report.json").read_text(encoding="utf-8"))
        self.assertIn("03_exports", report["lightweight"]["output_fbx"].replace("\\", "/"))

    def test_mesh_requested_without_blender_is_inspectable_failure(self):
        self.start_worker(mode="auto", attach_meshes=True)
        state = self.await_state(lambda value: value["state"] == "FAILED")
        self.assertIn(state["stages"][-1]["name"], {"attach_meshes"})
        self.assertEqual(state["stages"][-1]["status"], "FAILED")
        self.assertTrue(state["error"])
        self.assertIn("source_fbx", [artifact["role"] for artifact in state["artifacts"]])
        self.assertFalse((self.directory / "display.fbx").exists())


if __name__ == "__main__":
    unittest.main()


class ArchiveOutputsTests(unittest.TestCase):
    def test_archives_final_outputs_to_processed_and_exports_with_same_structure(self):
        import threading
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / '01_data_raw' / 'body' / 'actor001' / 'C28' / 'B001-C28.mcp'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'input')
            job_dir = root / 'jobs' / 'j1'
            native = job_dir / 'native'
            native.mkdir(parents=True)
            (native / 'B001-C28.vdf').write_bytes(b'vdf')
            (native / 'B001-C28.fbx').write_bytes(b'fbx')
            (native / 'B001-C28.vsk').write_bytes(b'vsk')
            worker = Worker.__new__(Worker)
            worker.request = {'job_dir': str(job_dir), 'input_path': str(source), 'data_roots': [str(root)]}
            worker.directory = job_dir
            worker.source = source
            worker.state = {'artifacts': []}
            worker.condition = threading.Condition()
            worker.report = {'input_path': str(source),
                             'stages': [{'name': 'export_fbx', 'status': 'SUCCESS'}],
                             'artifacts': {'vdf': 'B001-C28.vdf', 'fbx': 'B001-C28.fbx', 'vsk': 'B001-C28.vsk'}}
            worker._event = lambda *args, **kwargs: None
            worker._archive_outputs(native)
            processed = root / '02_processed' / 'body' / 'actor001' / 'C28'
            self.assertEqual((processed / 'B001-C28.fbx').read_bytes(), b'fbx')
            self.assertEqual((processed / 'B001-C28.vdf').read_bytes(), b'vdf')
            self.assertEqual((processed / 'B001-C28.vsk').read_bytes(), b'vsk')
            self.assertFalse((processed / 'B001-C28_display.fbx').exists())
            # Without the lightweight stage, the full working FBX is archived
            # as the usable export copy; nothing else is duplicated.
            self.assertEqual((root / '03_exports' / 'body' / 'actor001' / 'C28' / 'B001-C28.fbx').read_bytes(), b'fbx')
            self.assertEqual({a['role'] for a in worker.state['artifacts']},
                             {'archive_processed_vdf', 'archive_processed_fbx', 'archive_processed_vsk',
                              'archive_export_fbx'})
            self.assertEqual(source.read_bytes(), b'input')

    def test_no_archive_before_successful_export(self):
        import threading
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / '01_data_raw' / 'group1' / 'MP' / 't' / 'take.mcp'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'input')
            job_dir = root / 'jobs' / 'j2'
            native = job_dir / 'native'
            native.mkdir(parents=True)
            worker = Worker.__new__(Worker)
            worker.request = {'job_dir': str(job_dir), 'input_path': str(source), 'data_roots': [str(root)]}
            worker.directory = job_dir
            worker.source = source
            worker.state = {'artifacts': []}
            worker.condition = threading.Condition()
            worker.report = {'input_path': str(source), 'stages': [{'name': 'solve_solving', 'status': 'SUCCESS'}],
                             'artifacts': {}}
            worker._event = lambda *args, **kwargs: None
            worker._archive_outputs(native)
            self.assertFalse((root / '02_processed').exists())
            self.assertFalse((root / '03_exports').exists())
            self.assertEqual(worker.state['artifacts'], [])

"""Run-discovery filesystem contracts; no SDK, native sessions or production paths."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import contextlib
from datetime import datetime
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline_run_registry import open_shared_read, read_run_record, register_run, run_key
from pipeline_layout import initialize_layout


def _publish_distinct(arguments):
    output, registry, index = arguments
    return register_run(output, registry_dir=registry, kind="processing", run_id=f"launch-{index}", planned_total=index)


class RunRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.registry = self.root / "web" / "data" / "runs"
        self.output = self.root / "processed"

    def publish(self, **overrides):
        arguments = dict(kind="processing", run_id="original", planned_total=12,
                         registry_dir=self.registry)
        arguments.update(overrides)
        return register_run(self.output, **arguments)

    def record(self):
        return read_run_record(self.registry / (run_key(self.output) + ".json"))

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_schema_paths_timestamp_and_exact_root_key(self):
        record = self.publish(source_output=self.root / "source", data_path=self.root / "raw")
        expected = hashlib.sha256(str(self.output.resolve()).casefold().encode()).hexdigest()[:16]
        self.assertEqual(record["id"], expected)
        self.assertEqual(record, self.record())
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["output_dir"], str(self.output))
        self.assertEqual(record["source_output"], str(self.root / "source"))
        self.assertEqual(record["data_path"], str(self.root / "raw"))
        self.assertEqual(record["scheduler_pid"], os.getpid())
        self.assertEqual(datetime.fromisoformat(record["started_at"]).utcoffset().total_seconds(), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.registry.iterdir()), [self.registry / (expected + ".json")])

    def test_local_layout_registers_control_root_and_separate_raw_dataset(self):
        raw = self.root / "raw"
        raw.mkdir()
        self.output = raw / "_processing" / "processing"
        initialize_layout(self.output, raw, "processing")
        record = self.publish()
        self.assertEqual(record["id"], run_key(self.output))
        self.assertEqual(record["output_dir"], str(self.output))
        self.assertEqual(record["data_path"], str(raw))
        self.assertNotEqual(record["id"], run_key(raw))
        self.assertEqual(self.record(), record)
        self.assertEqual(list(raw.iterdir()), [raw / "_processing"])

    def test_local_recheck_preserves_processing_identity_and_raw_data_path(self):
        raw = self.root / "raw"
        raw.mkdir()
        self.output = raw / "_processing" / "processing"
        initialize_layout(self.output, raw, "processing")
        original = self.publish()
        updated = self.publish(kind="quality_recheck", run_id="audit")
        self.assertEqual(updated["run_id"], original["run_id"])
        self.assertEqual(updated["data_path"], str(raw))
        self.assertEqual(updated["output_dir"], str(self.output))
        self.assertEqual(updated["quality_recheck"]["run_id"], "audit")
        self.assertEqual(self.record(), updated)

    def test_resume_same_output_replaces_one_entry(self):
        first = self.publish()
        second = register_run(self.output / ".." / self.output.name, kind="human_repair",
                              run_id="resumed", planned_total=7, registry_dir=self.registry)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.record()["run_id"], "resumed")
        self.assertEqual(self.record()["planned_total"], 7)
        self.assertEqual(len(list(self.registry.glob("*.json"))), 1)

    def test_concurrent_processes_publish_distinct_roots(self):
        arguments = [(str(self.root / f"output-{index}"), str(self.registry), index) for index in range(8)]
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            records = list(pool.map(_publish_distinct, arguments))
        self.assertTrue(all(records))
        self.assertEqual(len(list(self.registry.glob("*.json"))), len(arguments))
        self.assertEqual({record["run_id"] for record in records}, {f"launch-{index}" for index in range(8)})
        for record in records:
            self.assertEqual(json.loads((self.registry / (record["id"] + ".json")).read_text(encoding="utf-8")), record)
        self.assertFalse(list(self.registry.glob("*.tmp")))
        self.assertFalse(list(self.registry.glob("*.lock")))

    def test_readers_only_observe_complete_atomic_updates(self):
        self.publish(label="first")
        path = self.registry / (run_key(self.output) + ".json")

        def update():
            for index in range(40):
                record = self.publish(run_id=f"resume-{index}", label=(str(index) + "-trajectory-") * 500)
                if record is None:
                    raise AssertionError("catalog update failed")

        observations = 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(update)
            while not future.done():
                record = read_run_record(path)
                self.assertEqual(record["id"], run_key(self.output))
                self.assertEqual(record["planned_total"], 12)
                observations += 1
            future.result()
        self.assertGreater(observations, 0)
        self.assertEqual(self.record()["run_id"], "resume-39")

    def test_shared_reader_is_readonly_and_allows_subsequent_replacement(self):
        original = self.publish()
        path = self.registry / (run_key(self.output) + ".json")
        with open_shared_read(path) as reader:
            self.assertFalse(reader.writable())
            with self.assertRaises(io.UnsupportedOperation):
                reader.write(b"must not write")
            self.assertEqual(json.load(reader), original)
        replacement = self.publish(run_id="replacement", planned_total=9)
        self.assertIsNotNone(replacement)
        self.assertEqual(read_run_record(path), replacement)

    def test_live_report_reader_is_readonly_and_allows_subsequent_publication(self):
        from step1_process_takes import write_json

        path = self.output / "takes" / "take" / "report.json"
        original = {"job_id": "take", "status": "NEEDS_REVIEW", "quality": {"usable": False}}
        replacement = dict(original, independent_recheck={"status": "PASSED"})
        write_json(path, original)
        with open_shared_read(path) as reader:
            self.assertFalse(reader.writable())
            with self.assertRaises(io.UnsupportedOperation):
                reader.write(b"must not write")
            self.assertEqual(json.load(reader), original)
        write_json(path, replacement)
        self.assertEqual(read_run_record(path), replacement)
    def test_record_reader_rejects_oversize_nonobject_and_nonfinite_json(self):
        path = self.root / "record.json"
        for content in ('[]', '{"value": NaN}', '{"value": Infinity}', '{"value": 1e999}'):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_run_record(path)
        path.write_text('{"value": "long"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_run_record(path, max_bytes=5)

    def test_unavailable_catalog_warns_and_returns_without_aborting_caller(self):
        blocked = self.root / "not-a-directory"
        blocked.write_bytes(b"leave this unrelated file unchanged")
        warning = io.StringIO()
        with contextlib.redirect_stderr(warning):
            record = register_run(self.output, kind="human_repair", registry_dir=blocked / "runs")
        self.assertIsNone(record)
        self.assertIn("processing continues", warning.getvalue())
        self.assertEqual(blocked.read_bytes(), b"leave this unrelated file unchanged")
        self.assertFalse(self.output.exists())

    def test_environment_selects_catalog(self):
        with patch.dict(os.environ, {"MOCAP_RUN_REGISTRY": str(self.registry)}):
            record = register_run(self.output, kind="processing")
        self.assertEqual(record, self.record())

    def test_recheck_preserves_native_identity_and_adds_audit_role(self):
        original = self.publish(kind="human_repair", source_output=self.root / "source")
        inspected = self.publish(kind="quality_recheck", run_id="inspection", planned_total=None)
        for key, value in original.items():
            self.assertEqual(inspected[key], value)
        self.assertEqual(inspected["quality_recheck"]["run_id"], "inspection")
        self.assertEqual(inspected["quality_recheck"]["role"], "independent_recheck")
        self.assertEqual(self.record(), inspected)

    def test_recheck_discovers_only_explicit_matching_native_launch(self):
        launch = {"run_id": "native-run", "scheduler_pid": 12345,
                  "started_at": "2026-10-30T08:00:00+00:00", "output_dir": str(self.output),
                  "source_output": str(self.root / "source"), "planned_total": 5,
                  "command": ["python.exe", "F:\\repo\\step2_repair_humans.py"]}
        self.save(self.output / "production_launch.json", launch)
        self.save(self.output / ".pipeline.lock", {"run_id": launch["run_id"], "pid": launch["scheduler_pid"]})
        record = self.publish(kind="quality_recheck", run_id="inspection")
        self.assertEqual(record["kind"], "human_repair")
        self.assertEqual(record["run_id"], "native-run")
        self.assertEqual(record["scheduler_pid"], 12345)
        self.assertEqual(record["planned_total"], 5)

    def test_stale_launch_cannot_masquerade_as_current_native_batch(self):
        self.save(self.output / "production_launch.json", {
            "run_id": "old-run", "scheduler_pid": 12345, "kind": "human_repair",
            "started_at": "2026-10-30T08:00:00+00:00", "output_dir": str(self.output)})
        self.save(self.output / ".pipeline.lock", {"run_id": "new-run", "pid": 67890})
        record = self.publish(kind="quality_recheck", run_id="inspection", planned_total=None,
                              source_output=self.output)
        self.assertEqual(record["kind"], "quality_recheck")
        self.assertEqual(record["run_id"], "inspection")
        self.assertIsNone(record["planned_total"])
        self.assertNotIn("quality_recheck", record)

    def test_target_only_scope_retains_subject_and_source_not_scene_usability(self):
        record = self.publish(kind="human_target", subject="alice", input_path=self.root / "source.vdf",
                              planned_total=1)
        self.assertEqual(record["kind"], "human_target")
        self.assertEqual(record["subject"], "alice")
        self.assertEqual(record["input_path"], str(self.root / "source.vdf"))
        self.assertNotIn("usable", record)
        self.assertNotIn("active", record)

    def test_importing_all_entrypoints_has_no_registry_or_sdk_side_effects(self):
        script_dir = Path(__file__).resolve().parent
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "import pipeline_run_registry, step1_process_takes, step3_repair_reviews, step2_repair_humans, step4_recheck_quality, step2_repair_single_human; "
                "assert not any(name.startswith('vicon_shogun') for name in sys.modules)")
        environment = dict(os.environ, MOCAP_RUN_REGISTRY=str(self.registry))
        result = subprocess.run([sys.executable, "-B", "-c", code, str(script_dir)],
                                cwd=self.root, env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_real_dry_run_entrypoints_do_not_publish_or_write_outputs(self):
        from step1_process_takes import main as pipeline_main
        from step3_repair_reviews import main as review_main
        from step2_repair_humans import main as human_main

        raw = self.root / "MocapData" / "group1" / "MP" / "take.mcp"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"read-only inventory fixture")
        raw.with_suffix(".x2d").write_bytes(b"paired inventory fixture")
        source = self.root / "source"
        (source / "takes").mkdir(parents=True)
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        with patch.dict(os.environ, {"MOCAP_RUN_REGISTRY": str(self.registry)}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline_main(["-DatasetLayout", "-DryRun", "-DataPath", str(self.root / "MocapData"),
                                            "-OutputDir", str(self.output)]), 0)
            self.assertEqual(review_main(["-SourceOutput", str(source), "-OutputDir", str(self.output), "-DryRun"]), 0)
            self.assertEqual(human_main(["--source-output", str(source), "--output-dir", str(self.output), "--dry-run"]), 0)
        self.assertFalse(self.registry.exists())
        self.assertFalse(self.output.exists())
        after = {str(path.relative_to(self.root)): path.read_bytes()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

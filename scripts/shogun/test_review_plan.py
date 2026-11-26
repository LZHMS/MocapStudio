"""Filesystem-backed planning tests; no native session or production data writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pipeline_layout import initialize_layout, take_paths
from pipeline_review_plan import plan_review_repairs


class ReviewPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.output = self.source / "_repair_runs" / "test-run"
        self.raw = self.root / "raw"
        (self.source / "takes").mkdir(parents=True)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def use_local_layout(self):
        self.raw.mkdir(parents=True, exist_ok=True)
        self.source = self.raw / "_processing/source-run"
        self.output = self.source / "_repair_runs/test-run"
        initialize_layout(self.source, self.raw, "source-version")


    def make_report(self, name="take", *, bad=None, humans=None, expected_count=2):
        humans = ["alice", "bob"] if humans is None else humans
        relative = f"group1/HOI/{name}/{name}.mcp"
        raw = self.raw / relative
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"raw-mcp")
        paths = take_paths(self.source, relative)
        destination = paths["review_destination"]
        destination.mkdir(parents=True)
        vdf = destination / f"{name}.vdf"
        vdf.write_bytes(b"completed-native-vdf")
        report = {
            "status": "NEEDS_REVIEW", "job_id": hashlib.sha256(relative.casefold().encode()).hexdigest()[:16],
            "input_kind": "mcp", "input_path": str(raw), "relative_path": relative,
            "quality": {"basis": "post_processing", "usable": False},
            "dataset_mode": True, "sequence_kind": "HOI", "split_subjects": True,
            "props_enabled": True, "expected_human_count": expected_count,
            "dataset": {"expected_human_count": expected_count, "detected_humans": humans,
                        "sequence_kind": "HOI", "props_enabled": True,
                        "roster_source": "loaded_take_rig_definitions"},
            "expected_subjects": None, "effective_subjects": humans,
            "expected_props": None, "effective_props": ["camera_1", "desk", "chair", "cup"],
            "policy": {"min_coverage": .95, "core_markers": {"alice": ["LASI"]},
                       "prop_tolerance": 2.5, "prop_max_step": 175,
                       "prop_jitter_max_correction": 0},
            "range": {"animation_start": 1, "animation_end": 1000, "play_start": 1,
                      "play_end": 1000, "frame_count": 1000, "animation_frame_count": 1000},
            "artifacts": {"vdf": str(vdf)},
            "stages": [{"name": stage, "status": "SUCCESS"}
                       for stage in ("fix_occlusion", "solve_labeling", "solve_solving")],
            "subjects": [{"name": human, "rig_class": "HUMAN_RIG", "classification": "PRESENT",
                          "processing_target": True,
                          "quality": {"basis": "post_processing", "usable": True}}
                         for human in humans],
            "props": [{"name": "camera_1", "classification": "ZERO_DATA", "is_background": True,
                       "unresolved_intervals": [[1, 1000]]},
                      {"name": "desk", "classification": "PRESENT", "is_background": True,
                       "unresolved_intervals": []},
                      {"name": "chair", "classification": "PRESENT", "is_background": True,
                       "unresolved_intervals": []},
                      {"name": "cup", "classification": "PARTIAL", "is_background": False,
                       "unresolved_intervals": [[100, 100]] if bad is None else bad}],
        }
        path = paths["report_path"]
        self.save(path, report)
        return path, report

    def capture(self, report, subjects, *, name=None, suffix="Take01"):
        raw = Path(report["input_path"])
        path = raw.with_name(f"{raw.stem}.{suffix}.enf")
        path.write_text(f"[Node Information]\nTYPE=TRIAL\nNAME={name or raw.stem}\n"
                        f"[TRIAL_INFO]\nSUBJECTS={','.join(subjects)}\nDESCRIPTION=\nNOTES=\n",
                        encoding="utf-8")
        return path

    def completed_output(self, job, *, status="SUCCESS"):
        artifacts = {}
        destination = Path(job["destination"])
        destination.mkdir(parents=True)
        for role in ("vdf", "fbx", "vsk", "subject_0_fbx"):
            path = destination / ("subjects/alice/sequence.fbx" if role == "subject_0_fbx" else f"take.{role}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fresh-native-export")
            artifacts[role] = str(path)
        report = {"job_id": job["job_id"], "input_path": job["input_path"], "status": status,
                  "source_report_sha256": job["source_report_sha256"], "artifacts": artifacts,
                  "subject_exports": [{"path": artifacts["subject_0_fbx"], "artifact_role": "subject_0_fbx"}]}
        self.save(Path(job["report_path"]), report)
        return report

    def test_sparse_priority_union_and_limit_after_ranking(self):
        self.make_report("a-long", bad=[[100, 600]])
        sparse_path, sparse = self.make_report("z-sparse", bad=[[100, 102], [102, 103]])
        sparse["props"][1].update(classification="PARTIAL", unresolved_intervals=[[101, 104]])
        self.save(sparse_path, sparse)
        failed_path, failed = self.make_report("b-human-fails", bad=[[100, 100]])
        failed["subjects"][0]["quality"]["usable"] = False
        self.save(failed_path, failed)
        plan = plan_review_repairs(self.source, self.output, limit=1)
        self.assertEqual([job["relative_path"] for job in plan["jobs"]], [sparse["relative_path"]])
        self.assertEqual(plan["counts"]["selected_total"], 3)
        self.assertEqual(plan["counts"]["selected"], 1)
        self.assertEqual(plan["counts"]["deferred_by_limit"], 2)
        self.assertEqual(plan["counts"], {
            "reports_scanned": 3, "candidates": 3, "eligible": 3,
            "selected_total": 3, "selected": 1, "preserved": 0,
            "skipped": 0, "roster_pending": 0, "deferred_by_limit": 2,
        })
        priority = plan["jobs"][0]["priority"]
        self.assertEqual(priority["bad_intervals"], [[100, 104]])
        self.assertEqual(priority["bad_frames"], 5)
        self.assertEqual(priority["longest_bad_run"], 5)
        self.assertTrue(priority["sparse"])
        self.assertIn("desk", priority["non_camera_props"])
        self.assertNotIn("camera_1", priority["non_camera_props"])
        full = plan_review_repairs(self.source, self.output)
        self.assertEqual([job["priority"]["human_pass"] for job in full["jobs"]], [True, True, False])

    def test_positive_one_percent_sparse_precedes_zero_gap_calibration_and_dense(self):
        calibration_path, calibration = self.make_report(
            "a-zero-calibration", humans=["alice"], expected_count=2)
        calibration["props"] = []
        calibration["effective_props"] = []
        calibration["range"].update(animation_end=100000, play_end=100000,
                                    frame_count=100000, animation_frame_count=100000)
        self.save(calibration_path, calibration)
        self.capture(calibration, ["alice"])
        roster_path, roster = self.make_report("b-zero-roster-only", humans=["alice"])
        roster["props"] = []
        roster["effective_props"] = []
        self.save(roster_path, roster)
        _, sparse = self.make_report("z-positive-sparse", bad=[[100, 109]])
        _, dense = self.make_report("c-dense", bad=[[100, 110]])
        failed_path, failed = self.make_report("d-human-failing", bad=[[100, 100]])
        failed["subjects"][0]["quality"]["usable"] = False
        self.save(failed_path, failed)
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual([job["relative_path"] for job in plan["jobs"]], [
            sparse["relative_path"], calibration["relative_path"], roster["relative_path"],
            dense["relative_path"], failed["relative_path"],
        ])
        self.assertEqual([job["priority"]["tier"] for job in plan["jobs"]], [0, 1, 1, 2, 3])
        self.assertEqual(plan["jobs"][0]["priority"]["bad_fraction"], .01)
        self.assertEqual(plan["jobs"][0]["priority"]["longest_bad_run"], 10)
        self.assertEqual(plan["counts"]["selected_total"], 5)
        self.assertEqual(plan["counts"]["roster_pending"], 1)
        limited = plan_review_repairs(self.source, self.output, limit=1)
        self.assertEqual(limited["jobs"][0]["relative_path"], sparse["relative_path"])
        self.assertEqual(limited["counts"]["selected_total"], 5)

    def test_fake_one_rig_does_not_confirm_single_person(self):
        _, report = self.make_report(humans=["alice"], expected_count=2)
        plan = plan_review_repairs(self.source, self.output)
        job = plan["jobs"][0]
        self.assertEqual(job["expected_human_count"], 2)
        self.assertEqual(job["expected_subjects"], ["alice"])
        self.assertFalse(job["confirmed_subject_roster"])
        self.assertTrue(job["roster_resolution"]["pending"])
        self.assertEqual(plan["roster_pending"][0]["job_id"], report["job_id"])
        self.assertEqual(plan["counts"]["roster_pending"], 1)

    def test_unconfirmed_legacy_count_one_remains_pending(self):
        self.make_report(humans=["alice"], expected_count=1)
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        self.assertEqual(job["expected_human_count"], 1)
        self.assertTrue(job["roster_resolution"]["pending"])
        self.assertFalse(job["confirmed_subject_roster"])

    def test_manifest_take_confirms_count_and_wins_over_capture(self):
        _, report = self.make_report(humans=["alice"], expected_count=2)
        self.capture(report, ["alice", "unknown"])
        plan = plan_review_repairs(self.source, self.output,
                                   manifest={"takes": {report["relative_path"].casefold(): ["alice"]}})
        job = plan["jobs"][0]
        self.assertEqual(job["expected_human_count"], 1)
        self.assertTrue(job["confirmed_subject_roster"])
        self.assertEqual(job["roster_resolution"]["source"], "manifest_take")
        self.assertEqual(job["roster_resolution"]["source_expected_human_count"], 2)
        self.assertEqual(plan["roster_pending"], [])
        self.assertFalse(job["identity_validated"])

    def test_explicit_source_roster_is_distinct_from_effective_roster(self):
        path, report = self.make_report(humans=["alice"])
        report["expected_subjects"] = ["alice"]
        report["limits"] = {"roster_completeness": "known"}
        self.save(path, report)
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        self.assertEqual(job["expected_human_count"], 1)
        self.assertTrue(job["confirmed_subject_roster"])
        self.assertEqual(job["roster_resolution"]["source"], "source_expected_subjects")

    def test_capture_single_human_confirms_recorded_roster(self):
        _, report = self.make_report(humans=["alice"], expected_count=2)
        self.capture(report, ["camera_1", "desk", "alice"])
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        self.assertEqual(job["expected_human_count"], 1)
        self.assertEqual(job["expected_subjects"], ["alice"])
        self.assertTrue(job["confirmed_subject_roster"])
        self.assertFalse(job["identity_validated"])
        resolution = job["roster_resolution"]
        self.assertEqual(resolution["source"], "capture_enf")

    def test_mp_excluded_prop_evidence_does_not_create_unknown_human_roster(self):
        path, report = self.make_report(humans=["alice"])
        report.update(sequence_kind="MP", props_enabled=False, effective_props=[], props=[])
        report["dataset"].update(sequence_kind="MP", props_enabled=False,
                                 excluded_props=["desk", "chair", "cup"])
        self.save(path, report)
        self.capture(report, ["alice", "desk", "chair", "cup", "camera_2"])
        plan = plan_review_repairs(self.source, self.output)
        job = plan["jobs"][0]
        self.assertEqual(job["expected_human_count"], 1)
        self.assertTrue(job["confirmed_subject_roster"])
        self.assertFalse(job["props_enabled"])
        self.assertEqual(plan["roster_pending"], [])

    def test_missing_permitted_camera_is_recorded_without_inventing_a_second_person(self):
        _, report = self.make_report(humans=["alice"])
        self.capture(report, ["alice", "camera_2"])
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        self.assertEqual(job["expected_human_count"], 1)
        self.assertTrue(job["confirmed_subject_roster"])
        self.assertIn("camera_2", job["expected_props"])

    def test_capture_zero_data_second_human_still_counts(self):
        path, report = self.make_report()
        report["subjects"][1].update(classification="ZERO_DATA")
        report["subjects"][1]["quality"]["usable"] = False
        report["effective_subjects"] = ["alice"]
        self.save(path, report)
        self.capture(report, ["alice", "bob", "desk"])
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        self.assertEqual(job["expected_human_count"], 2)
        self.assertEqual(job["expected_subjects"], ["alice", "bob"])
        self.assertTrue(job["confirmed_subject_roster"])
        self.assertFalse(job["priority"]["human_pass"])

    def test_unknown_or_missing_human_capture_cannot_confirm_single(self):
        for name, capture in (("unknown", ["alice", "missing-bob"]), ("no-human", ["desk"])):
            with self.subTest(name=name):
                _, report = self.make_report(name, humans=["alice"])
                self.capture(report, capture)
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(len(plan["jobs"]), 2)
        for job in plan["jobs"]:
            self.assertEqual(job["expected_human_count"], 2)
            self.assertFalse(job["confirmed_subject_roster"])
            self.assertTrue(job["roster_resolution"]["pending"])

    def test_inconsistent_or_wrong_name_capture_stays_pending(self):
        _, first = self.make_report("inconsistent", humans=["alice"])
        self.capture(first, ["alice"], suffix="Take01")
        self.capture(first, ["alice", "desk"], suffix="Take02")
        _, second = self.make_report("wrong-name", humans=["alice"])
        self.capture(second, ["alice"], name="different-mcp")
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(len(plan["jobs"]), 2)
        self.assertEqual(len(plan["roster_pending"]), 2)
        self.assertTrue(all(job["expected_human_count"] == 2 for job in plan["jobs"]))

    def test_incomplete_native_stage_evidence_is_skipped(self):
        for stage in ("fix_occlusion", "solve_labeling", "solve_solving"):
            path, report = self.make_report(stage)
            report["stages"] = [entry for entry in report["stages"] if entry["name"] != stage]
            self.save(path, report)
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 3)

    def recovered_report(self):
        path, report = self.make_report("recovered")
        original = take_paths(self.source, report["relative_path"])["failed_root"] / "original-attempt/take.vdf"
        original.parent.mkdir(parents=True)
        original.write_bytes(Path(report["artifacts"]["vdf"]).read_bytes())
        previous = json.loads(json.dumps(report))
        previous.update(status="FAILED", error="Legacy quality-stage contract failure",
                        artifacts={"vdf": original.name})
        control = original.parent / ".control" / "result.json"
        self.save(control, previous)
        report["stages"] = [{"name": name, "status": "SKIPPED"}
                            for name in ("fix_occlusion", "solve_labeling", "solve_solving")]
        report["processing_source"] = {
            "path": str(original), "checkpoint_resume_without_fill": True,
            "native_bytes_reused": True,
            "completed_steps_reused": ["fix_occlusion", "solve_labeling", "solve_solving"]}
        self.save(path, report)
        return report, original, control

    def test_byte_identical_recovered_checkpoint_reuses_verified_original_stages(self):
        report, original, control = self.recovered_report()
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual([job["job_id"] for job in plan["jobs"]], [report["job_id"]])
        self.assertEqual(plan["skipped"], [])
        self.assertEqual(plan["jobs"][0]["checkpoint_completion"]["basis"],
                         "byte_identical_completed_recovery")

    def test_recovery_claim_does_not_excuse_changed_bytes_or_wrong_original_input(self):
        report, original, control = self.recovered_report()
        original_bytes = original.read_bytes()
        original.write_bytes(original_bytes[:-1] + b"X")
        changed = plan_review_repairs(self.source, self.output)
        self.assertEqual(changed["jobs"], [])
        self.assertEqual(len(changed["skipped"]), 1)
        original.write_bytes(original_bytes)
        previous = json.loads(control.read_bytes())
        previous["input_path"] = str(self.raw / "different-input.mcp")
        self.save(control, previous)
        wrong_input = plan_review_repairs(self.source, self.output)
        self.assertEqual(wrong_input["jobs"], [])
        self.assertEqual(len(wrong_input["skipped"]), 1)

    def test_invalid_identity_range_and_empty_vdf_are_skipped(self):
        path, report = self.make_report("bad-id")
        report["job_id"] = "another-job"
        self.save(path, report)
        path, report = self.make_report("bad-range")
        report["range"]["frame_count"] = 999
        self.save(path, report)
        _, report = self.make_report("empty-vdf")
        Path(report["artifacts"]["vdf"]).write_bytes(b"")
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 3)

    def test_escaped_artifact_and_borrowed_checkpoint_are_skipped(self):
        external = self.root / "external.vdf"
        external.write_bytes(b"outside-source")
        path, report = self.make_report("escaped")
        report["artifacts"]["vdf"] = str(external)
        self.save(path, report)
        _, other = self.make_report("owner")
        path, report = self.make_report("borrower")
        report["artifacts"]["vdf"] = other["artifacts"]["vdf"]
        self.save(path, report)
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(len(result["jobs"]), 1)
        self.assertEqual(len(result["skipped"]), 2)

    def test_escaped_symlink_checkpoint_is_skipped(self):
        path, report = self.make_report()
        external = self.root / "outside.vdf"
        external.write_bytes(b"external")
        vdf = Path(report["artifacts"]["vdf"])
        vdf.unlink()
        try:
            vdf.symlink_to(external)
        except OSError as error:
            self.skipTest(f"Host does not permit test symlinks: {error}")
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 1)

    def test_output_collisions_never_overwrite_or_retry_incomplete_report(self):
        self.make_report()
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        orphan = Path(job["review_destination"]) / "orphan.fbx"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"existing-export")
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(orphan.read_bytes(), b"existing-export")
        self.save(Path(job["report_path"]), {"status": "FAILED"})
        before = Path(job["report_path"]).read_bytes()
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(Path(job["report_path"]).read_bytes(), before)

    def test_completed_matching_output_is_preserved_and_all_exports_validate(self):
        self.make_report()
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        completed = self.completed_output(job)
        before = Path(job["report_path"]).read_bytes()
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(result["counts"]["preserved"], 1)
        self.assertEqual(result["counts"]["eligible"], 1)
        self.assertEqual(result["counts"]["selected_total"], 0)
        self.assertEqual(Path(job["report_path"]).read_bytes(), before)
        Path(completed["subject_exports"][0]["path"]).write_bytes(b"")
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["preserved"], [])
        self.assertEqual(len(result["skipped"]), 1)

    def test_resume_rejects_changed_source_hash_or_missing_artifact(self):
        path, source_report = self.make_report()
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        completed = self.completed_output(job, status="NEEDS_REVIEW")
        source_report["review_reasons"] = ["changed-source-audit"]
        self.save(path, source_report)
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["counts"]["preserved"], 0)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 1)
        completed["source_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save(Path(job["report_path"]), completed)
        Path(completed["artifacts"]["fbx"]).unlink()
        self.assertTrue(Path(job["report_path"]).is_file())
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["preserved"], [])
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 1)

    def test_planning_is_read_only_and_does_not_require_raw_x2d(self):
        path, report = self.make_report()
        timecode = Path(report["artifacts"]["vdf"]).with_suffix(".json")
        timecode.write_text('{"timecode": "01:00:00:00"}', encoding="utf-8")
        report["artifacts"]["timecode"] = str(timecode)
        self.save(path, report)
        before = {file: file.read_bytes() for file in self.source.rglob("*") if file.is_file()}
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {file: file.read_bytes() for file in self.source.rglob("*") if file.is_file()})
        self.assertFalse(Path(report["input_path"]).with_suffix(".x2d").exists())

    def test_only_old_postprocessing_review_reports_are_selected(self):
        path, report = self.make_report("already-jittered")
        report["policy"]["prop_jitter_max_correction"] = 5
        self.save(path, report)
        path, report = self.make_report("success")
        report["status"] = "SUCCESS"
        self.save(path, report)
        path, report = self.make_report("not-final")
        report["quality"]["basis"] = "before_fill"
        self.save(path, report)
        path, report = self.make_report("missing-jitter")
        del report["policy"]["prop_jitter_max_correction"]
        self.save(path, report)
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertEqual(plan["jobs"][0]["relative_path"], report["relative_path"])
        self.assertEqual(plan["counts"]["reports_scanned"], 4)
        self.assertEqual(len(plan["skipped"]), 1)

    def test_unsafe_output_roots_rejected_nested_repair_run_allowed(self):
        self.make_report()
        for output in (self.source, self.root, self.source / "takes" / "new-output"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                plan_review_repairs(self.source, output)
        self.assertEqual(len(plan_review_repairs(self.source, self.output)["jobs"]), 1)

    def test_roster_pending_tracks_unbounded_queue_and_zero_limit(self):
        self.make_report("first", humans=["alice"])
        self.make_report("second", humans=["bob"])
        plan = plan_review_repairs(self.source, self.output, limit=0)
        self.assertEqual(plan["jobs"], [])
        self.assertEqual(plan["counts"]["selected_total"], 2)
        self.assertEqual(plan["counts"]["deferred_by_limit"], 2)
        self.assertEqual(len(plan["roster_pending"]), 2)

    def test_invalid_timecode_reference_is_not_silently_dropped(self):
        path, report = self.make_report()
        report["artifacts"]["timecode"] = str(Path(report["artifacts"]["vdf"]).with_suffix(".json"))
        self.save(path, report)
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 1)

    def test_resume_rejects_subject_path_outside_declared_output(self):
        self.make_report()
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        completed = self.completed_output(job)
        outside = self.root / "outside.fbx"
        outside.write_bytes(b"foreign-subject")
        completed["subject_exports"][0]["path"] = str(outside)
        self.save(Path(job["report_path"]), completed)
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["preserved"], [])
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 1)

    def test_duplicate_casefolded_jobs_are_both_rejected(self):
        self.make_report("Take")
        lowercase = self.source / "takes/group1/HOI/take/take.mcp"
        if lowercase.exists():
            self.skipTest("Filesystem does not permit distinct casefold-colliding take paths")
        self.make_report("take")
        result = plan_review_repairs(self.source, self.output)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(len(result["skipped"]), 2)

    def test_local_source_plans_inherited_layout_without_creating_target_manifest(self):
        self.use_local_layout()
        source_path, report = self.make_report()
        before = {path: path.read_bytes() for path in self.raw.rglob("*") if path.is_file()}
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(plan["skipped"], [])
        self.assertEqual(len(plan["jobs"]), 1)
        job = plan["jobs"][0]
        relative = Path(report["relative_path"])
        take_root = self.raw / relative.parent / "processed" / self.output.name / relative.name
        self.assertEqual(Path(job["destination"]), take_root / "artifacts")
        self.assertEqual(Path(job["review_destination"]), take_root / "_needs_review/artifacts")
        self.assertEqual(Path(job["report_path"]), take_root / "report.json")
        self.assertEqual(Path(job["log_path"]), take_root / "logs/processing.log")
        self.assertEqual(job["source_report"], str(source_path))
        self.assertEqual(job["source_report_sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest())
        self.assertEqual(job["layout_data_root"], str(self.raw))
        self.assertFalse((self.source / "takes").exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(take_root.exists())
        self.assertEqual(before, {path: path.read_bytes() for path in self.raw.rglob("*") if path.is_file()})

    def test_local_scan_ignores_nested_diagnostics_and_other_versions(self):
        self.use_local_layout()
        _, report = self.make_report()
        paths = take_paths(self.source, report["relative_path"])
        for relative in ("artifacts/report.json", "_failed/attempt/report.json",
                         ".quality_recheck_attempts/attempt/report.json"):
            self.save(paths["take_root"] / relative, report)
        other = self.source / "_repair_runs/neighbor"
        initialize_layout(other, self.raw, "neighbor-version")
        self.save(take_paths(other, report["relative_path"])["report_path"], report)
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(plan["counts"]["reports_scanned"], 1)
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertEqual(plan["skipped"], [])

    def test_local_completed_repair_preserves_source_sha_and_exact_artifact_roots(self):
        self.use_local_layout()
        source_path, source_report = self.make_report()
        initialize_layout(self.output, self.raw)
        job = plan_review_repairs(self.source, self.output)["jobs"][0]
        completed = self.completed_output(job)
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(plan["counts"]["preserved"], 1)
        self.assertEqual(plan["jobs"], [])
        source_report["review_reasons"] = ["changed audit"]
        self.save(source_path, source_report)
        changed = plan_review_repairs(self.source, self.output)
        self.assertEqual(changed["preserved"], [])
        self.assertEqual(len(changed["skipped"]), 1)
        completed["source_report_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        foreign = take_paths(self.output, "group1/HOI/another/other.mcp")["destination"] / "other.fbx"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(b"foreign export")
        completed["subject_exports"][0]["path"] = str(foreign)
        self.save(Path(job["report_path"]), completed)
        borrowed = plan_review_repairs(self.source, self.output)
        self.assertEqual(borrowed["preserved"], [])
        self.assertEqual(len(borrowed["skipped"]), 1)

    def test_local_recovery_proof_uses_per_take_failed_control_and_native_hashes(self):
        self.use_local_layout()
        report, original, control = self.recovered_report()
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(plan["skipped"], [])
        proof = plan["jobs"][0]["checkpoint_completion"]
        self.assertEqual(proof["original_control_report"], str(control))
        self.assertEqual(proof["original_control_sha256"], hashlib.sha256(control.read_bytes()).hexdigest())
        self.assertEqual(proof["native_vdf_sha256"], hashlib.sha256(original.read_bytes()).hexdigest())
        original.write_bytes(original.read_bytes()[:-1] + b"X")
        changed = plan_review_repairs(self.source, self.output)
        self.assertEqual(changed["jobs"], [])
        self.assertEqual(len(changed["skipped"]), 1)

    def test_local_source_cannot_borrow_checkpoint_from_other_version(self):
        self.use_local_layout()
        path, report = self.make_report()
        neighbor = self.source / "_repair_runs/neighbor"
        initialize_layout(neighbor, self.raw, "neighbor-version")
        foreign = take_paths(neighbor, report["relative_path"])["review_destination"] / "take.vdf"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(b"completed but unrelated version")
        report["artifacts"]["vdf"] = str(foreign)
        self.save(path, report)
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(plan["jobs"], [])
        self.assertEqual(len(plan["skipped"]), 1)

    def test_local_source_raw_identity_cannot_move_to_another_dataset(self):
        self.use_local_layout()
        path, report = self.make_report()
        alternate = self.root / "another-dataset" / report["relative_path"]
        alternate.parent.mkdir(parents=True)
        alternate.write_bytes(b"same trailing relative identity is not the same raw input")
        report["input_path"] = str(alternate)
        self.save(path, report)
        plan = plan_review_repairs(self.source, self.output)
        self.assertEqual(plan["jobs"], [])
        self.assertEqual(len(plan["skipped"]), 1)



if __name__ == "__main__":
    unittest.main()

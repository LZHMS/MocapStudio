"""Physical human quality and conservative correction policy; no native SDK."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipeline_human import evaluate_geometry, read_calibration


class HumanGeometryTests(unittest.TestCase):
    def setUp(self):
        self.groups = {}
        self.points = {}
        for side in ('L', 'R'):
            offsets = {side + name: np.array(xyz, dtype=float) for name, xyz in (
                ('HEL', (0, 0, 0)), ('MT1', (120, 0, 0)),
                ('MT5', (0, 120, 0)), ('FootAux', (0, 0, 120)))}
            self.groups[side + 'Foot'] = offsets
            for name, xyz in offsets.items():
                self.points[name] = np.tile(xyz, (20, 1)) + (300 if side == 'R' else 0)

    def test_deep_flexion_toe_rise_and_walking_world_motion_are_allowed(self):
        for frame in range(20):
            angle = frame * .08
            rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            for marker in self.groups['RFoot']:
                self.points[marker][frame] = self.groups['RFoot'][marker] @ rotation + [300 + frame * 5, 0, 300]
        # Articulating toe/ankle spans deliberately differ: these are NOT rigid
        # foot-cluster evidence and must not be rejected by bilateral ratios.
        for name, xyz in {'LTOE': [200, 0, 0], 'RTOE': [900, 0, 500],
                          'LANK': [0, 0, 70], 'RANK': [300, 0, 1000]}.items():
            self.points[name] = np.tile(xyz, (20, 1))
        self.assertTrue(evaluate_geometry(self.points, self.groups)['usable'])

    def test_persistent_bad_heel_cluster_is_rejected_not_forced(self):
        self.points['RHEL'][:, 2] += 200
        verdict = evaluate_geometry(self.points, self.groups, first_frame=100)
        self.assertFalse(verdict['usable'])
        self.assertTrue(any('calibrated_segment_geometry:RFoot' in reason for reason in verdict['review_reasons']))
        self.assertTrue(any('bilateral_foot_geometry' in reason for reason in verdict['review_reasons']))



    def test_missing_saved_trajectories_are_unavailable_not_pass(self):
        for values in self.points.values():
            values[:] = np.nan
        verdict = evaluate_geometry(self.points, self.groups)
        self.assertFalse(verdict['usable'])
        self.assertIn('calibrated_local_segment_geometry', verdict['unavailable_evidence'])
        self.assertIn('comparable_bilateral_foot_geometry', verdict['unavailable_evidence'])

    def test_offsets_from_different_bones_are_never_subtracted(self):
        groups = {'RLeg': {'RANK': np.array([0., 0., 0.])},
                  'RToeBase': {'RTOE': np.array([1000., 0., 0.])}}
        verdict = evaluate_geometry(self.points, groups)
        self.assertEqual(verdict['calibrated_pairs'], [])
        self.assertFalse(verdict['usable'])

    def test_ambiguous_calibration_cannot_drive_correction(self):
        class API:
            def GetConstraintNames(self, subject, skeleton):
                return ['a', 'b']
            def GetConstraintDetails(self, subject, name, skeleton):
                return True, 'RHEL', 'RFoot', [0, 0, 0], ''
        groups, errors = read_calibration(API(), 'actor', ['RHEL'])
        self.assertEqual(groups, {})
        self.assertTrue(errors)





class OriginalLiveReferenceTests(unittest.TestCase):
    def setUp(self):
        self.ranges = dict(animation_start=1, animation_end=100, play_start=1, play_end=100, frame_count=100)
        time = np.arange(100, dtype=float)
        self.correct = {
            'RKNE': np.column_stack((time, time * .1, time * 0 + 450)),
            'RSHN': np.column_stack((time, time * .1, time * 0 + 280)),
            'RANK': np.column_stack((time, time * .1, time * 0 + 65))}
        sparse = {name: points.copy() for name, points in self.correct.items()}
        for points in sparse.values():
            points[np.arange(100) % 3 != 0] = np.nan
        self.reference = dict(ranges=self.ranges, points=sparse)

    def test_coherent_hierarchy_shift_is_rejected_even_with_complete_rigid_geometry(self):
        from pipeline_human_reference import compare_source_labels
        wrong = {'RKNE': self.correct['RKNE'] + [0, 170, 70],
                 'RSHN': self.correct['RKNE'], 'RANK': self.correct['RSHN']}
        verdict = compare_source_labels(wrong, self.reference, self.ranges)
        self.assertIn('original_live_label_disagreement:RKNE:processed_as:RSHN', verdict['review_reasons'])
        self.assertIn('original_live_label_disagreement:RSHN:processed_as:RANK', verdict['review_reasons'])

    def test_normal_crouch_and_raised_ankle_remain_accepted_when_original_labels_agree(self):
        from pipeline_human_reference import compare_source_labels
        verdict = compare_source_labels(self.correct, self.reference, self.ranges)
        self.assertEqual(verdict['review_reasons'], [])

    def test_unique_fragmented_native_tracks_can_be_identified_without_interpolation(self):
        from pipeline_human_reference import plan_reference_track_relabel
        observed = {'first': self.correct['RANK'].copy(), 'second': self.correct['RANK'].copy()}
        observed['first'][65:] = np.nan
        observed['second'][:65] = np.nan
        # Each fragment needs independent >=20 anchors.
        reference = dict(ranges=self.ranges, points={'RANK': self.correct['RANK']})
        result = plan_reference_track_relabel(reference, observed, ['RANK'])
        self.assertEqual(set(result['proposals']['RANK']['sources']), {'first', 'second'})

    def test_competing_observation_tracks_do_not_allow_a_relabel(self):
        from pipeline_human_reference import plan_reference_track_relabel
        observed = {'first': self.correct['RANK'], 'duplicate': self.correct['RANK'].copy()}
        result = plan_reference_track_relabel(self.reference, observed, ['RANK'])
        self.assertEqual(result['proposals'], {})
        self.assertTrue(result['rejections'])

    def test_a_single_far_original_anchor_rejects_an_otherwise_close_track(self):
        from pipeline_human_reference import plan_reference_track_relabel
        observed = self.correct['RANK'].copy()
        observed[30, 2] += 100.
        result = plan_reference_track_relabel(self.reference, {'native': observed}, ['RANK'])
        self.assertEqual(result['proposals'], {})

    def test_matching_prefix_does_not_authorize_long_unanchored_tail(self):
        from pipeline_human_reference import plan_reference_track_relabel
        reference = dict(ranges=self.ranges, points={'RANK': self.correct['RANK'].copy()})
        reference['points']['RANK'][20:] = np.nan
        result = plan_reference_track_relabel(reference, {'native': self.correct['RANK']}, ['RANK'])
        self.assertEqual(result['proposals'], {})


class SourceCaptureDiagnosticsTests(unittest.TestCase):
    def context(self):
        timeline = SimpleNamespace(GetAnimationStart=lambda: 1, GetAnimationEnd=lambda: 3,
                                   GetPlayStart=lambda: 1, GetPlayEnd=lambda: 3,
                                   SetTimeFrames=lambda frame: None)
        scene = SimpleNamespace(Objects=SimpleNamespace(
            FilterByType=lambda kind: [SimpleNamespace(Name="actor")]))
        return scene, timeline

    def test_failing_marker_preserves_prior_progress_and_original_exception(self):
        from pipeline_human_reference import capture_source_labels
        from pipeline_processing import _NativeDiagnostics
        failure = RuntimeError("Invalid Parameter Value")
        audit, report = {}, {"artifacts": {}}
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "native_diagnostics.jsonl"
            diagnostic = _NativeDiagnostics(report, journal, SimpleNamespace(pid=71, port=81))
            diagnostic("LoadFile", lambda path: None, "/original/take.mcp")

            def trajectory(subject, marker):
                pending = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
                self.assertEqual(pending["status"], "RUNNING")
                self.assertEqual(pending["arguments"], [subject, marker])
                if marker == "broken":
                    raise failure
                return ([1.] * 3, [2.] * 3, [3.] * 3, [True] * 3)

            session = SimpleNamespace(api=SimpleNamespace(
                GetMarkerNames=lambda name: ["good", "broken", "never_read"],
                GetTrajectory=trajectory))
            with patch("pipeline_processing._sdk_context", return_value=self.context()):
                with self.assertRaises(RuntimeError) as raised:
                    capture_source_labels(session, native_call=diagnostic, audit=audit)
            self.assertIs(raised.exception, failure)
            self.assertEqual(audit["current_subject"], "actor")
            self.assertEqual(audit["current_marker"], "broken")
            self.assertNotIn("capture_complete", audit)
            self.assertEqual(audit["captured_subjects"][0]["captured_markers"],
                             [{"name": "good", "existing_frames": 3}])
            failed = report["native_diagnostics"]["failed_calls"][0]
            self.assertEqual(failed["api"], "GetTrajectory")
            self.assertEqual(failed["arguments"], ["actor", "broken"])
            self.assertEqual(failed["scene_source"], "/original/take.mcp")
            self.assertEqual(failed["range"]["play_end"], 3)
            self.assertEqual(report["artifacts"]["native_diagnostics"], journal.name)
            events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["status"], "FAILED")
            self.assertTrue(all(row["native_pid"] == 71 for row in events))

    def test_invalid_trajectory_still_fails_after_successful_native_read(self):
        from pipeline_human_reference import capture_source_labels
        audit = {}
        session = SimpleNamespace(api=SimpleNamespace(
            GetMarkerNames=lambda name: ["broken"],
            GetTrajectory=lambda subject, marker: ([1.], [2.], [3.], [True])))
        with patch("pipeline_processing._sdk_context", return_value=self.context()):
            with self.assertRaisesRegex(RuntimeError, "Invalid trajectory length: actor/broken"):
                capture_source_labels(session, audit=audit)
        self.assertEqual(audit["captured_subjects"][0]["captured_markers"], [])
        self.assertNotIn("capture_complete", audit)

    def test_frame_evaluation_failure_records_exact_timeline_api(self):
        from pipeline_human_reference import capture_source_labels
        from pipeline_processing import _NativeDiagnostics
        scene, timeline = self.context()

        def fail(frame):
            raise ValueError("Invalid Parameter Value")

        timeline.SetTimeFrames = fail
        report = {}
        with tempfile.TemporaryDirectory() as temporary:
            call = _NativeDiagnostics(report, Path(temporary) / "native_diagnostics.jsonl",
                                      SimpleNamespace())
            with patch("pipeline_processing._sdk_context", return_value=(scene, timeline)):
                with self.assertRaises(ValueError):
                    capture_source_labels(SimpleNamespace(), native_call=call)
        self.assertEqual(report["native_diagnostics"]["failed_calls"][0]["api"],
                         "Timeline.SetTimeFrames")
        self.assertEqual(report["native_diagnostics"]["failed_calls"][0]["arguments"], [1])


if __name__ == '__main__':
    unittest.main()

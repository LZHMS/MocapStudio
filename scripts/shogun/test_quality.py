"""Final subject-quality policy regressions; no SDK or native process required."""
import unittest

from pipeline_processing import (
    classify_subject,
    evaluate_subject_quality,
    marker_coverage,
    update_subject_export_quality,
)


class SubjectQualityTests(unittest.TestCase):
    def setUp(self):
        self.setup = {'markers': 2, 'labeling_bones': 5, 'solving_bones': 5}
        self.coverage = marker_coverage({'head': 100, 'pelvis': 100}, 100)
        self.solve = {
            skeleton: {'keyed_bones': 1, 'animated_frames': 100, 'animation_coverage': 1.0}
            for skeleton in ('labeling', 'solving')
        }

    def evaluate(self, coverage=None, minimum=1.0):
        return evaluate_subject_quality(
            self.setup, self.coverage if coverage is None else coverage,
            self.solve, min_coverage=minimum, geometry={'usable': True, 'review_reasons': []},
        )

    def test_repaired_original_partial_is_usable_from_final_coverage(self):
        original = marker_coverage({'head': 100, 'pelvis': 25}, 100)
        self.assertEqual(classify_subject(2, 5, 5, original), 'PARTIAL')
        original_quality = self.evaluate(original)
        self.assertFalse(original_quality['usable'])
        final_quality = self.evaluate()
        self.assertEqual(final_quality['classification'], 'PRESENT')
        self.assertEqual(final_quality['review_reasons'], [])
        self.assertTrue(final_quality['usable'])
        # Evaluating the final state must not overwrite original audit metrics.
        self.assertEqual(original['marker_frames']['pelvis'], 25)

    def test_final_residual_marker_gap_is_not_usable(self):
        final = marker_coverage({'head': 100, 'pelvis': 99}, 100)
        quality = self.evaluate(final)
        self.assertEqual(quality['classification'], 'PARTIAL')
        self.assertFalse(quality['usable'])
        self.assertTrue(quality['review_reasons'])

    def test_final_missing_core_marker_is_not_usable_even_at_low_threshold(self):
        final = marker_coverage({'head': 100, 'pelvis': 100}, 100,
                                ['head', 'pelvis', 'ankle'])
        quality = self.evaluate(final, minimum=0.5)
        self.assertEqual(quality['classification'], 'PARTIAL')
        self.assertFalse(quality['usable'])
        self.assertTrue(quality['review_reasons'])

    def test_complete_markers_do_not_hide_missing_skeleton_animation(self):
        for skeleton in ('labeling', 'solving'):
            for metrics in (
                {'keyed_bones': 0, 'animated_frames': 0, 'animation_coverage': 0.0},
                {'keyed_bones': 1, 'animated_frames': 0, 'animation_coverage': 0.0},
                {'keyed_bones': 0, 'animated_frames': 100, 'animation_coverage': 1.0},
                {'keyed_bones': 1, 'animated_frames': 99, 'animation_coverage': 0.99},
            ):
                with self.subTest(skeleton=skeleton, metrics=metrics):
                    previous = self.solve[skeleton]
                    self.solve[skeleton] = metrics
                    try:
                        quality = self.evaluate()
                        self.assertFalse(quality['usable'])
                        self.assertTrue(quality['review_reasons'])
                    finally:
                        self.solve[skeleton] = previous

    def test_missing_template_or_either_skeleton_is_not_usable(self):
        for component in self.setup:
            with self.subTest(component=component):
                previous = self.setup[component]
                self.setup[component] = 0
                try:
                    quality = self.evaluate()
                    self.assertEqual(quality['classification'], 'MISSING_SETUP')
                    self.assertFalse(quality['usable'])
                    self.assertTrue(quality['review_reasons'])
                finally:
                    self.setup[component] = previous

    def test_marker_threshold_is_inclusive(self):
        coverage = marker_coverage({'head': 80, 'pelvis': 80}, 100)
        self.assertTrue(self.evaluate(coverage, minimum=0.8)['usable'])
        self.assertFalse(self.evaluate(coverage, minimum=0.8001)['usable'])

    def test_skeleton_animation_threshold_is_inclusive(self):
        for skeleton in ('labeling', 'solving'):
            with self.subTest(skeleton=skeleton):
                previous = self.solve[skeleton]
                self.solve[skeleton] = {
                    'keyed_bones': 1, 'animated_frames': 80, 'animation_coverage': 0.8,
                }
                try:
                    self.assertTrue(self.evaluate(minimum=0.8)['usable'])
                    self.assertFalse(self.evaluate(minimum=0.8001)['usable'])
                finally:
                    self.solve[skeleton] = previous

    def test_invalid_thresholds_are_rejected(self):
        for minimum in (-0.01, 1.01, float('nan'), float('inf')):
            with self.subTest(minimum=minimum):
                with self.assertRaises(ValueError):
                    self.evaluate(minimum=minimum)

    def test_availability_without_geometry_is_not_physical_quality_approval(self):
        quality = evaluate_subject_quality(self.setup, self.coverage, self.solve)
        self.assertFalse(quality['usable'])
        self.assertIn('human_geometry_evidence_unavailable:not_inspected', quality['review_reasons'])

    def test_abnormal_geometry_blocks_complete_marker_and_skeleton_availability(self):
        quality = evaluate_subject_quality(
            self.setup, self.coverage, self.solve,
            geometry={'usable': False, 'review_reasons': ['calibrated_segment_geometry:RightFoot:RHEL:RMT1']})
        self.assertFalse(quality['usable'])
        self.assertIn('calibrated_segment_geometry:RightFoot:RHEL:RMT1', quality['review_reasons'])


    def test_interactive_prop_issue_blocks_subject_export_quality(self):
        report = {
            'subjects': [{'name': 'actor', 'rig_class': 'HUMAN_RIG', 'processing_target': True,
                          'classification': 'PRESENT', 'review_reasons': [], 'quality': {'usable': True}}],
            'subject_exports': [{'subject': 'actor'}],
            'review_reasons': ['prop:blue_phone:unresolved_rigid_frames']
        }
        update_subject_export_quality(report, ['prop:blue_phone:unresolved_rigid_frames'])
        export_quality = report['subject_exports'][0]['quality']
        self.assertFalse(export_quality['usable'])
        self.assertIn('prop:blue_phone:unresolved_rigid_frames', export_quality['review_reasons'])

if __name__ == '__main__':
    unittest.main()

"""Consumer-visible temporal rigid-jitter regressions; no Shogun SDK required."""
import json
import unittest

import numpy as np

from pipeline_jitter import JitterOptions, repair_jitter
from pipeline_rigid import RigidOptions, match_frame


class RigidJitterTests(unittest.TestCase):
    def setUp(self):
        self.template = np.array([[0., 0., 0.], [83., 0., 0.],
                                  [17., 137., 0.], [29., 41., 101.]])
        self.rigid = RigidOptions(tolerance_mm=.01, max_step_mm=200.)

    def sequence(self, count):
        return np.repeat(self.template[None], count, axis=0)

    def repair(self, points, **kwargs):
        return repair_jitter(self.template, points, self.rigid, JitterOptions(**kwargs))

    def assert_rejected(self, result, original, reason):
        np.testing.assert_array_equal(result['points'], original)
        self.assertFalse(result['corrected_frames'].any())
        self.assertEqual(result['corrections'], [])
        self.assertEqual(result['states_after'], result['states_before'])
        self.assertTrue(any(reason in row['reason'] for row in result['rejections']),
                        result['rejections'])

    def assert_geometry(self, result):
        for frame in np.flatnonzero(result['corrected_frames']):
            fitted = match_frame(self.template, result['points'][frame],
                                 np.empty((0, 3)), self.rigid)
            self.assertEqual(fitted['state'], 'TRACKED')
            predicted = self.template @ fitted['rotation'].T + fitted['translation']
            self.assertLessEqual(np.max(np.linalg.norm(predicted - result['points'][frame],
                                                       axis=1)), self.rigid.tolerance_mm)
            self.assertAlmostEqual(np.linalg.det(fitted['rotation']), 1.)

    def rotating_sequence(self, angles, translation_step=(2., -1., .5), axis=(0., 0., 1.)):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        frames = []
        for frame, degrees in enumerate(angles):
            angle = np.deg2rad(degrees)
            rotation = (np.eye(3) * np.cos(angle)
                        + (1 - np.cos(angle)) * np.outer(axis, axis)
                        + np.sin(angle) * cross)
            translation = np.array([230., -170., 61.]) + frame * np.asarray(translation_step)
            frames.append(self.template @ rotation.T + translation)
        return np.stack(frames)

    def test_complete_wrong_existing_marker_is_corrected_with_honest_sparse_audit(self):
        points = self.sequence(3)
        points[1, 2, 1] += 4.
        original = points.copy()
        template_before = self.template.copy()
        points.flags.writeable = False
        self.template.flags.writeable = False
        result = self.repair(points)
        self.assertEqual(result['states_before'], ['TRACKED', 'INVALID_GEOMETRY', 'TRACKED'])
        self.assertEqual(result['states_after'], ['TRACKED'] * 3)
        np.testing.assert_array_equal(result['corrected_frames'], [False, True, False])
        np.testing.assert_allclose(result['points'][1], self.template, atol=1e-10)
        np.testing.assert_array_equal(result['points'][[0, 2]], original[[0, 2]])
        np.testing.assert_array_equal(result['points'][1, [0, 1, 3]], original[1, [0, 1, 3]])
        self.assertEqual(len(result['corrections']), 1)
        correction = result['corrections'][0]
        self.assertEqual((correction['frame'], correction['marker']), (1, 2))
        self.assertEqual(correction['original_xyz'], original[1, 2].tolist())
        self.assertEqual(correction['corrected_xyz'], result['points'][1, 2].tolist())
        self.assertAlmostEqual(correction['displacement_mm'], 4.)
        self.assertEqual(result['method'], 'temporal_rigid_jitter')
        self.assertEqual(result['provenance'], 'model_derived_not_measured')
        self.assertEqual(result['rejections'], [])
        json.dumps(result['corrections'], allow_nan=False)
        np.testing.assert_array_equal(points, original)
        np.testing.assert_array_equal(self.template, template_before)
        self.assertFalse(np.shares_memory(result['points'], points))
        repeated = self.repair(points)
        self.assertEqual(repeated['corrections'], result['corrections'])
        np.testing.assert_array_equal(repeated['points'], result['points'])
        self.assert_geometry(result)

    def test_original_noisy_good_frames_and_float_representation_are_untouched(self):
        points = self.sequence(5).astype(np.float32)
        points[0, 0, 0] = -0.
        points[1, 1, 2] += .002
        points[2, 2, 1] += 3.
        points[3, 0, 2] -= .002
        original = points.copy()
        result = self.repair(points)
        self.assertTrue(result['corrected_frames'][2])
        self.assertEqual(result['points'].dtype, points.dtype)
        for frame in (0, 1, 3, 4):
            self.assertEqual(result['points'][frame].tobytes(), original[frame].tobytes())
        self.assertEqual(points.tobytes(), original.tobytes())
        self.assert_geometry(result)

    def test_exact_displacement_limit_is_inclusive_and_above_limit_is_rejected(self):
        points = self.sequence(3)
        points[1, 0, 2] += 5.
        accepted = self.repair(points, max_correction_mm=5.)
        self.assertTrue(accepted['corrected_frames'][1])
        self.assertLessEqual(max(row['displacement_mm'] for row in accepted['corrections']),
                             5. + 1e-9)
        self.assert_rejected(self.repair(points, max_correction_mm=4.999999), points,
                             'correction_limit_exceeded')
        points[1, 0, 2] += .000001
        self.assert_rejected(self.repair(points), points, 'correction_limit_exceeded')

    def test_zero_disables_without_changing_original_invalid_data(self):
        points = self.sequence(3)
        points[1, 0, 2] += 3.
        self.assert_rejected(self.repair(points, max_correction_mm=0.), points, 'disabled')

    def test_exact_max_gap_is_accepted_but_longer_run_is_atomic_rejection(self):
        for length in (5, 6):
            with self.subTest(length=length):
                points = self.sequence(length + 2)
                points[1:-1, 0, 2] += 3.
                result = self.repair(points, max_gap_frames=5)
                if length == 5:
                    np.testing.assert_array_equal(result['corrected_frames'],
                                                  [False] + [True] * 5 + [False])
                    self.assert_geometry(result)
                else:
                    self.assert_rejected(result, points, 'max_gap_frames_exceeded')
                    self.assertEqual((result['rejections'][0]['start'],
                                      result['rejections'][0]['end']), (1, 6))

    def test_one_excess_displacement_rejects_every_frame_in_run(self):
        points = self.sequence(5)
        points[1:4, 0, 2] += [3., 6., 3.]
        result = self.repair(points)
        self.assert_rejected(result, points, 'correction_limit_exceeded')
        self.assertEqual((result['rejections'][0]['start'], result['rejections'][0]['end']),
                         (1, 3))

    def test_prefix_and_suffix_are_not_extrapolated(self):
        for bad in (slice(0, 2), slice(3, 5)):
            with self.subTest(bad=bad):
                points = self.sequence(5)
                points[bad, 0, 2] += 3.
                self.assert_rejected(self.repair(points), points, 'unanchored_edge')

    def test_missing_infinite_and_partial_rows_are_never_synthesized(self):
        for missing in (np.array([np.nan] * 3), np.array([np.inf] * 3),
                        np.array([np.nan, 0., 0.])):
            with self.subTest(missing=missing):
                points = self.sequence(3)
                points[1, 0, 2] += 3.
                points[1, 2] = missing
                result = self.repair(points)
                self.assert_rejected(result, points, 'missing_observations')
        points = self.sequence(3)
        points[1, :2] = np.nan
        self.assert_rejected(self.repair(points), points, 'missing_observations')

    def test_valid_partial_frame_stays_measured_and_is_not_filled(self):
        points = self.sequence(3)
        points[1, 0] = np.nan
        result = self.repair(points)
        self.assertEqual(result['states_before'], ['TRACKED'] * 3)
        np.testing.assert_array_equal(result['points'], points)
        self.assertTrue(np.isnan(result['points'][1, 0]).all())
        self.assertEqual(result['corrections'], [])

    def test_rotation_and_translation_are_interpolated_as_a_rigid_pose_not_xyz(self):
        expected = self.rotating_sequence(np.arange(7) * 20.)
        points = expected.copy()
        points[3, 2] += [0., 0., 3.]
        midpoint_xyz = (points[2] + points[4]) / 2
        self.assertGreater(np.max(np.linalg.norm(midpoint_xyz - expected[3], axis=1)), 5.)
        result = self.repair(points)
        self.assertTrue(result['corrected_frames'][3], result['rejections'])
        np.testing.assert_allclose(result['points'][3], expected[3], atol=1e-9)
        np.testing.assert_array_equal(result['points'][[0, 1, 2, 4, 5, 6]],
                                      points[[0, 1, 2, 4, 5, 6]])
        self.assert_geometry(result)

    def test_multiframe_gap_follows_constant_angular_and_translational_motion(self):
        expected = self.rotating_sequence(np.arange(9) * 12., axis=(1., 2., 3.))
        points = expected.copy()
        points[3:6, 1, 2] += 3.
        result = self.repair(points)
        np.testing.assert_array_equal(result['corrected_frames'],
                                      [False, False, False, True, True, True, False, False, False])
        np.testing.assert_allclose(result['points'][3:6], expected[3:6], atol=1e-9)
        self.assert_geometry(result)

    def test_quaternion_sign_crossing_and_near_identity_are_stable(self):
        for angles, axis in (([170., 180., 190.], (1., 0., 0.)),
                             ([170., 180., 190.], (0., 1., 0.)),
                             ([170., 180., 190.], (0., 0., 1.)),
                             ([-.000001, 0., .000001], (1., 2., 3.)),
                             ([40., 40., 40.], (1., 2., 3.))):
            with self.subTest(angles=angles, axis=axis):
                expected = self.rotating_sequence(angles, axis=axis)
                points = expected.copy()
                points[1, 2, 2] += 3.
                result = self.repair(points)
                self.assertTrue(result['corrected_frames'][1], result['rejections'])
                np.testing.assert_allclose(result['points'][1], expected[1], atol=1e-9)
                self.assert_geometry(result)

    def test_near_half_turn_uses_short_arc_but_exact_half_turn_is_ambiguous(self):
        expected = self.rotating_sequence([0., 89.9995, 179.999], translation_step=(0., 0., 0.))
        points = expected.copy()
        points[1, 0, 2] += 3.
        accepted = self.repair(points)
        self.assertTrue(accepted['corrected_frames'][1], accepted['rejections'])
        np.testing.assert_allclose(accepted['points'][1], expected[1], atol=1e-9)
        expected = self.rotating_sequence([0., 90., 180.], translation_step=(0., 0., 0.))
        points = expected.copy()
        points[1, 0, 2] += 3.
        self.assert_rejected(self.repair(points), points, 'ambiguous_rotation')

    def test_window_parameter_rejects_inconsistent_neighbor_translation(self):
        points = self.sequence(7)
        points[1] += [20., 0., 0.]
        points[3, 0, 2] += 3.
        nearest_only = self.repair(points, window_frames=1)
        self.assertTrue(nearest_only['corrected_frames'][3])
        self.assert_rejected(self.repair(points, window_frames=2), points,
                             'inconsistent_anchor_window')

    def test_window_checks_farther_samples_and_both_sides(self):
        for frame in (0, 6):
            with self.subTest(frame=frame):
                points = self.sequence(7)
                points[frame] += [0., 20., 0.]
                points[3, 0, 2] += 3.
                self.assertTrue(self.repair(points, window_frames=2)['corrected_frames'][3])
                self.assert_rejected(self.repair(points, window_frames=3), points,
                                     'inconsistent_anchor_window')

    def test_window_rejects_conflicting_angular_motion_not_just_translation(self):
        points = self.rotating_sequence([0., 20., 40., 60., 80., 100., 145.])
        points[3, 2, 2] += 3.
        self.assertTrue(self.repair(points, window_frames=2)['corrected_frames'][3])
        self.assert_rejected(self.repair(points, window_frames=3), points,
                             'inconsistent_anchor_window')

    def test_window_agreement_bound_is_inclusive(self):
        points = self.sequence(5)
        points[0] += [5., 0., 0.]
        points[2, 0, 2] += 3.
        self.assertTrue(self.repair(points, window_frames=2)['corrected_frames'][2])
        points[0] += [.000001, 0., 0.]
        self.assert_rejected(self.repair(points, window_frames=2), points,
                             'inconsistent_anchor_window')

    def test_anchor_window_never_crosses_another_bad_run_or_uses_repaired_pose(self):
        points = self.sequence(7)
        points[0] += [30., 0., 0.]
        points[0, 0, 2] += 3.  # Unanchored invalid prefix must block the window.
        points[2, 0, 2] += 3.
        points[4, 1, 2] += 3.
        result = self.repair(points, window_frames=3)
        np.testing.assert_array_equal(result['corrected_frames'],
                                      [False, False, True, False, True, False, False])
        np.testing.assert_array_equal(result['points'][0], points[0])
        self.assertEqual(len(result['rejections']), 1)
        self.assertIn('unanchored_edge', result['rejections'][0]['reason'])
        self.assert_geometry(result)

    def test_accepted_repairs_are_not_reused_as_neighbor_window_evidence(self):
        points = self.sequence(7)
        points[:3, :, 0] += np.array([-30., -20., -10.])[:, None]
        points[2, 0, 2] += 3.
        points[4, 1, 2] += 3.
        result = self.repair(points, window_frames=3)
        # Gap 2 has a 10mm/frame approach; gap 4 is stationary. Original bad
        # frame 2 terminates gap 4's window, even after it has been repaired.
        # Reusing its corrected -10mm pose would incorrectly reject gap 4.
        np.testing.assert_array_equal(result['corrected_frames'],
                                      [False, False, True, False, True, False, False])
        self.assertEqual(result['rejections'], [])
        self.assert_geometry(result)

    def test_rigid_marker_space_motion_gate_is_not_relaxed(self):
        self.rigid = RigidOptions(tolerance_mm=.01, max_step_mm=2.)
        points = self.sequence(3)
        points[1] += [3., 0., 0.]
        points[2] += [6., 0., 0.]
        points[1, 0, 2] += 3.
        self.assert_rejected(self.repair(points), points, 'corrected_geometry_or_motion_rejected')
        self.rigid = RigidOptions(tolerance_mm=.01, max_step_mm=3.)
        self.assertTrue(self.repair(points)['corrected_frames'][1])

    def test_bad_calibration_cannot_be_made_valid_by_temporal_interpolation(self):
        self.template[:, 1:] = 0.
        points = self.sequence(3)
        self.assert_rejected(self.repair(points), points, 'unanchored_edge')

    def test_parameter_and_shape_validation(self):
        for value in (-1., np.nan, np.inf, -np.inf, 'bad', None):
            with self.subTest(distance=value), self.assertRaises(ValueError):
                JitterOptions(max_correction_mm=value)
        for name in ('max_gap_frames', 'window_frames'):
            for value in (0, -1, True, np.bool_(False), 1.5, np.nan, '3', None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    JitterOptions(**{name: value})
        for template, points in ((np.zeros((4, 2)), np.zeros((3, 4, 2))),
                                 (np.zeros(3), np.zeros((3, 1, 3))),
                                 (self.template, np.zeros((4, 3))),
                                 (self.template, np.zeros((3, 5, 3))),
                                 (self.template, np.zeros((3, 4, 2)))):
            with self.subTest(template=template.shape, points=points.shape):
                with self.assertRaises(ValueError):
                    repair_jitter(template, points, self.rigid, JitterOptions())
        for count in (0, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                repair_jitter(self.template, self.sequence(count),
                              RigidOptions(tolerance_mm=-1.), JitterOptions())

    def test_empty_sequence_has_no_synthesized_frames(self):
        points = np.empty((0, len(self.template), 3))
        result = self.repair(points)
        self.assertEqual(result['points'].shape, points.shape)
        self.assertEqual(result['corrected_frames'].shape, (0,))
        self.assertEqual(result['states_before'], [])
        self.assertEqual(result['states_after'], [])
        self.assertEqual(result['corrections'], [])
        self.assertEqual(result['rejections'], [])


if __name__ == '__main__':
    unittest.main()

"""Behavioral rigid-prop regressions; pure NumPy, never starts Shogun."""
import unittest

import numpy as np

from pipeline_rigid import RigidOptions, match_frame


class RigidMatchingTests(unittest.TestCase):
    def setUp(self):
        # Distinct edge lengths avoid accidental marker-label symmetries.
        self.template = np.array([[0., 0., 0.], [83., 0., 0.],
                                  [17., 137., 0.], [29., 41., 101.]])
        self.rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.translation = np.array([230., -170., 61.])
        self.world = self.template @ self.rotation.T + self.translation
        self.absent = np.full_like(self.template, np.nan)
        self.empty = np.empty((0, 3))
        self.options = RigidOptions(tolerance_mm=0.05, max_step_mm=5.,
                                    max_hypotheses=10000)

    def match(self, observed=None, candidates=None, **kwargs):
        return match_frame(self.template, self.absent if observed is None else observed,
                           self.empty if candidates is None else candidates,
                           kwargs.pop('options', self.options), **kwargs)

    def assert_no_repair(self, result, observed):
        self.assertEqual(result['assignments'], {})
        np.testing.assert_equal(result['points'], observed)
        self.assertIsNone(result['rotation'])
        self.assertIsNone(result['translation'])
        self.assertIsNone(result['rms_mm'])

    def test_shuffled_candidates_recover_labels_not_list_order(self):
        for shuffle in ([2, 0, 3, 1], [1, 3, 0, 2]):
            candidates = np.vstack(([900., 800., 700.], self.world[shuffle]))
            result = self.match(candidates=candidates)
            self.assertEqual(result['state'], 'TRACKED')
            self.assertEqual(result['support'], 4)
            self.assertEqual(result['assignments'],
                             {marker: shuffle.index(marker) + 1 for marker in range(4)})
            np.testing.assert_allclose(result['points'], self.world)
            np.testing.assert_allclose(self.template @ result['rotation'].T
                                       + result['translation'], self.world, atol=1e-8)
            self.assertAlmostEqual(np.linalg.det(result['rotation']), 1.)

    def test_missing_prefix_uses_future_prior_without_inventing_measurements(self):
        earlier = self.world - np.array([12., 0., 0.])
        result = self.match(candidates=earlier[[3, 1, 0]], prior_points=self.world,
                            frame_distance=3)
        self.assertEqual(result['state'], 'TRACKED')
        self.assertEqual(result['assignments'], {0: 2, 1: 1, 3: 0})
        self.assertEqual(result['support'], 3)
        np.testing.assert_allclose(result['points'][[0, 1, 3]], earlier[[0, 1, 3]])
        self.assertTrue(np.isnan(result['points'][2]).all())

    def test_noise_within_tolerance_keeps_actual_measurements(self):
        noise = np.array([[.08, -.03, .02], [-.05, .02, -.04],
                          [.03, .06, -.02], [-.02, -.04, .05]])
        measured = self.world + noise
        result = self.match(candidates=measured[[1, 3, 2, 0]],
                            options=RigidOptions(tolerance_mm=.25, max_step_mm=10.))
        self.assertEqual(result['state'], 'TRACKED')
        np.testing.assert_equal(result['points'], measured)
        fitted = self.template @ result['rotation'].T + result['translation']
        self.assertLessEqual(np.max(np.linalg.norm(fitted - measured, axis=1)), .25)
        self.assertGreater(result['rms_mm'], 0.)

    def test_existing_labels_are_immutable_and_only_missing_labels_are_assigned(self):
        observed = self.absent.copy()
        observed[[0, 2, 3]] = self.world[[0, 2, 3]]
        observed[0, 0] += .01
        candidates = self.world[[1]].copy()
        before = [array.copy() for array in (self.template, observed, candidates)]
        result = self.match(observed=observed, candidates=candidates)
        self.assertEqual(result['state'], 'TRACKED')
        self.assertEqual(result['assignments'], {1: 0})
        np.testing.assert_equal(result['points'][[0, 2, 3]], observed[[0, 2, 3]])
        for original, snapshot in zip((self.template, observed, candidates), before):
            np.testing.assert_equal(original, snapshot)

    def test_inconsistent_existing_labels_cannot_be_replaced_by_good_candidates(self):
        observed = self.world.copy()
        observed[[0, 1]] = observed[[1, 0]]
        result = self.match(observed=observed, candidates=self.world)
        self.assertEqual(result['state'], 'INVALID_GEOMETRY')
        self.assert_no_repair(result, observed)

    def test_two_inconsistent_anchors_cannot_be_relabelled(self):
        observed = self.absent.copy()
        observed[0] = self.world[0]
        observed[1] = self.world[1] + [400., 0., 0.]
        result = self.match(observed=observed, candidates=self.world)
        self.assertEqual(result['state'], 'INVALID_GEOMETRY')
        self.assert_no_repair(result, observed)

    def test_existing_complete_pose_is_verified_without_candidate_search(self):
        result = self.match(observed=self.world, candidates=np.tile(self.world, (100, 1)),
                            options=RigidOptions(tolerance_mm=.01, max_hypotheses=1))
        self.assertEqual(result['state'], 'TRACKED')
        self.assertEqual(result['assignments'], {})
        self.assertEqual(result['support'], 4)
        np.testing.assert_equal(result['points'], self.world)

    def test_a_low_rms_cannot_hide_an_individual_marker_outside_tolerance(self):
        template = np.array([[-100., -100., 0.], [100., -100., 0.],
                             [100., 100., 0.], [-100., 100., 0.], [0., 0., 0.]])
        measured = template.copy()
        measured[4, 2] = 2.
        # The least-squares proper pose has RMS .8mm, but its center-marker
        # residual is 1.6mm. A 1mm measurement gate must reject it.
        result = match_frame(template, measured, self.empty,
                             RigidOptions(tolerance_mm=1.))
        self.assertEqual(result['state'], 'INVALID_GEOMETRY')
        self.assert_no_repair(result, measured)

    def test_valid_partial_observations_do_not_synthesize_a_missing_marker(self):
        observed = self.world.copy()
        observed[2] = np.nan
        result = self.match(observed=observed)
        self.assertEqual(result['state'], 'TRACKED')
        self.assertEqual(result['assignments'], {})
        self.assertEqual(result['support'], 3)
        np.testing.assert_equal(result['points'], observed)

    def test_insufficient_measurements_cannot_establish_a_pose(self):
        result = self.match(candidates=self.world[:2])
        self.assertEqual(result['state'], 'LOST')
        self.assert_no_repair(result, self.absent)

    def test_collinear_calibration_and_degenerate_observed_triangle_are_rejected(self):
        collinear = np.array([[0., 0., 0.], [100., 0., 0.], [200., 0., 0.]])
        absent = np.full_like(collinear, np.nan)
        invalid = match_frame(collinear, absent, collinear, self.options)
        self.assertEqual(invalid['state'], 'INVALID_GEOMETRY')
        almost_line = np.array([[0., 0., 0.], [100., 0., 0.], [200., .01, 0.]])
        # Large-enough tolerance makes distance checks pass, but cannot turn a
        # rank-one measured triangle into a reliable six-DOF observation.
        result = match_frame(almost_line, absent, collinear,
                             RigidOptions(tolerance_mm=.1))
        self.assertEqual(result['state'], 'LOST')
        self.assert_no_repair(result, absent)

    def test_planar_noncollinear_calibration_is_valid(self):
        planar = self.template.copy()
        planar[3, 2] = 0.
        measured = planar @ self.rotation.T + self.translation
        result = match_frame(planar, self.absent, measured[[3, 0, 2, 1]], self.options)
        self.assertEqual(result['state'], 'TRACKED')
        np.testing.assert_allclose(result['points'], measured)

    def test_identical_candidate_coordinates_are_rival_measurement_mappings(self):
        observed = self.world.copy()
        observed[3] = np.nan
        result = self.match(observed=observed, candidates=np.vstack((self.world[3], self.world[3])))
        self.assertEqual(result['state'], 'AMBIGUOUS')
        self.assertEqual(result['support'], 4)
        self.assert_no_repair(result, observed)

    def test_symmetric_marker_permutations_are_ambiguous_until_prior_excludes_them(self):
        square = np.array([[0., 0., 0.], [100., 0., 0.],
                           [100., 100., 0.], [0., 100., 0.]])
        absent = np.full_like(square, np.nan)
        candidates = square[[2, 0, 3, 1]] + [2., 0., 0.]
        ambiguous = match_frame(square, absent, candidates, self.options)
        self.assertEqual(ambiguous['state'], 'AMBIGUOUS')
        self.assert_no_repair(ambiguous, absent)
        result = match_frame(square, absent, candidates, self.options, prior_points=square)
        self.assertEqual(result['state'], 'TRACKED')
        self.assertEqual(result['assignments'], {0: 1, 1: 3, 2: 0, 3: 2})

    def test_reflected_tetrahedron_is_not_a_proper_rigid_pose(self):
        reflected = self.template * [-1., 1., 1.]
        existing = self.match(observed=reflected)
        self.assertEqual(existing['state'], 'INVALID_GEOMETRY')
        self.assert_no_repair(existing, reflected)
        unlabeled = self.match(candidates=reflected)
        self.assertNotEqual(unlabeled['state'], 'TRACKED')
        self.assert_no_repair(unlabeled, self.absent)

    def test_hard_budget_does_not_publish_an_unchecked_provisional_winner(self):
        observed = self.world.copy()
        observed[3] = np.nan
        # First branch fits. The second candidate must still be searched to
        # establish uniqueness, so a one-branch budget cannot report TRACKED.
        result = self.match(observed=observed,
                            candidates=np.vstack((self.world[3], self.world[3])),
                            options=RigidOptions(tolerance_mm=.05, max_hypotheses=1))
        self.assertEqual(result['state'], 'SEARCH_LIMIT')
        self.assert_no_repair(result, observed)

    def test_motion_gate_rejects_otherwise_valid_candidate_and_existing_poses(self):
        shifted = self.world + [9., 0., 0.]
        for observed, candidates in ((self.absent, shifted), (shifted, self.empty)):
            result = self.match(observed=observed, candidates=candidates,
                                prior_points=self.world)
            self.assertEqual(result['state'], 'LOST')
            self.assert_no_repair(result, observed)
        permitted = self.match(candidates=shifted, prior_points=self.world,
                               frame_distance=2)
        self.assertEqual(permitted['state'], 'TRACKED')
        np.testing.assert_equal(permitted['points'], shifted)

    def test_missing_prior_rows_do_not_create_false_motion_constraints(self):
        prior = self.world.copy()
        prior[3] = np.nan
        observed = self.world.copy()
        observed[3] = np.nan
        result = self.match(observed=observed, candidates=self.world[[3]],
                            prior_points=prior)
        self.assertEqual(result['state'], 'TRACKED')
        self.assertEqual(result['assignments'], {3: 0})

    def test_candidate_index_cannot_be_shared_by_two_marker_labels(self):
        template = np.vstack((self.template[:3], self.template[2]))
        observed = np.full_like(template, np.nan)
        observed[0] = template[0]
        result = match_frame(template, observed, template[[1, 2]], self.options)
        self.assertEqual(result['state'], 'AMBIGUOUS')
        self.assertEqual(result['support'], 3)
        self.assert_no_repair(result, observed)

    def test_invalid_nonfinite_rows_cannot_leak_into_geometry(self):
        for bad in ([1., np.nan, 3.], [np.inf, 2., 3.]):
            observed = self.absent.copy()
            observed[0] = bad
            result = self.match(observed=observed, candidates=self.world)
            self.assertEqual(result['state'], 'INVALID_GEOMETRY')
            self.assert_no_repair(result, observed)
        candidates = self.world.copy()
        candidates[0, 0] = np.nan
        result = self.match(candidates=candidates)
        self.assertEqual(result['state'], 'INVALID_GEOMETRY')
        self.assert_no_repair(result, self.absent)


if __name__ == '__main__':
    unittest.main()

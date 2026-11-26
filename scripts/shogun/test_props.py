"""Rigid curve behavior without native Shogun or manufactured observations."""
import unittest
import json
from types import SimpleNamespace

import numpy as np

from pipeline_props import intervals, repair_tracks, repair_prop_jitter, validate_prop_poses
from pipeline_jitter import JitterOptions
from pipeline_rigid import RigidOptions


class RigidSequenceTests(unittest.TestCase):
    def setUp(self):
        self.template = np.array([[0., 0., 0.], [70., 0., 0.],
                                  [10., 110., 0.], [20., 30., 90.]])
        self.world = np.array([self.template + [frame, 0., 0.] for frame in range(8)])
        self.options = RigidOptions(tolerance_mm=0.1, max_step_mm=20.)

    def pool(self, frames):
        return {frame: [(marker, xyz.copy()) for marker, xyz in enumerate(self.world[frame])]
                for frame in frames}

    def test_missing_prefix_is_reassociated_without_rewriting_reliable_tail(self):
        observed = self.world.copy()
        observed[:4] = np.nan
        result = repair_tracks({'phone': {'template': self.template, 'points': observed}},
                               self.pool(range(4)), self.options)['phone']
        self.assertEqual(len(result['patches']), 16)
        np.testing.assert_allclose(result['points'], self.world)
        np.testing.assert_array_equal(result['points'][4:], observed[4:])
        self.assertTrue(np.isnan(observed[:4]).all())

    def test_missing_tail_uses_forward_tracking(self):
        observed = self.world.copy()
        observed[4:] = np.nan
        result = repair_tracks({'phone': {'template': self.template, 'points': observed}},
                               self.pool(range(4, 8)), self.options)['phone']
        np.testing.assert_allclose(result['points'], self.world)
        self.assertEqual(len(result['patches']), 16)

    def test_interior_fragment_preserves_every_existing_marker(self):
        observed = self.world.copy()
        observed[2:5, 3] = np.nan
        pool = {frame: [(40, self.world[frame, 3])] for frame in range(2, 5)}
        result = repair_tracks({'phone': {'template': self.template, 'points': observed}},
                               pool, self.options)['phone']
        self.assertEqual(set(result['patches']), {(2, 3), (3, 3), (4, 3)})
        np.testing.assert_allclose(result['points'], self.world)

    def test_pose_prediction_never_creates_a_missing_observation(self):
        observed = self.world.copy()
        observed[2:5] = np.nan
        result = repair_tracks({'phone': {'template': self.template, 'points': observed}},
                               {}, self.options)['phone']
        self.assertFalse(result['patches'])
        self.assertTrue(np.isnan(result['points'][2:5]).all())
        self.assertEqual(result['states'][2:5], ['LOST'] * 3)

    def test_disagreeing_forward_and_backward_labels_do_not_publish_either(self):
        square = np.array([[0., 0., 0.], [100., 0., 0.],
                           [100., 100., 0.], [0., 100., 0.]])
        observed = np.full((3, 4, 3), np.nan)
        observed[0] = square
        observed[2] = np.roll(square, 1, axis=0)
        pool = {1: [(index, xyz) for index, xyz in enumerate(square)]}
        result = repair_tracks({'tray': {'template': square, 'points': observed}},
                               pool, self.options)['tray']
        self.assertFalse(result['patches'])
        self.assertEqual(result['states'][1], 'AMBIGUOUS')
        self.assertTrue(np.isnan(result['points'][1]).all())
        np.testing.assert_array_equal(result['points'][[0, 2]], observed[[0, 2]])

    def test_two_props_cannot_claim_the_same_observations(self):
        observed = np.full_like(self.world, np.nan)
        for names in (('phone', 'clicker'), ('clicker', 'phone')):
            with self.subTest(order=names):
                tracks = {name: {'template': self.template, 'points': observed} for name in names}
                result = repair_tracks(tracks, self.pool(range(8)), self.options)
                for prop in result.values():
                    self.assertFalse(prop['patches'])
                    self.assertTrue(np.isnan(prop['points']).all())
                    self.assertEqual(prop['states'], ['AMBIGUOUS'] * 8)

    def test_existing_geometric_mislabels_are_reported_not_overwritten(self):
        observed = self.world.copy()
        observed[3, 2] += [40., 0., 0.]
        result = repair_tracks({'phone': {'template': self.template, 'points': observed}},
                               self.pool([3]), self.options)['phone']
        self.assertEqual(result['states'][3], 'INVALID_GEOMETRY')
        self.assertFalse(result['patches'])
        np.testing.assert_array_equal(result['points'], observed)

    def test_large_boundary_jump_does_not_attach_a_remote_identical_shape(self):
        observed = self.world.copy()
        observed[3] = np.nan
        pool = {3: [(marker, xyz + [1000., 0., 0.])
                    for marker, xyz in enumerate(self.world[3])]}
        result = repair_tracks({'phone': {'template': self.template, 'points': observed}},
                               pool, self.options)['phone']
        self.assertFalse(result['patches'])
        self.assertTrue(np.isnan(result['points'][3]).all())

    def test_three_original_points_can_define_pose_without_inventing_fourth(self):
        observed = self.world.copy()
        observed[:, 3] = np.nan
        result = repair_tracks({'phone': {'template': self.template, 'points': observed}},
                               {}, self.options)['phone']
        self.assertEqual(result['states'], ['TRACKED'] * 8)
        self.assertFalse(result['patches'])
        self.assertTrue(np.isnan(result['points'][:, 3]).all())

    def test_report_intervals_use_actual_inclusive_frame_numbers(self):
        self.assertEqual(intervals([True, True, False, False, True], 15), [[15, 16], [19, 19]])


class FinalPropQualityTests(unittest.TestCase):
    """Exercise final reports against independent SDK-style trajectory/pose data."""

    class StoredPropAPI:
        # Intentionally no fill, reassociation, or native write API.
        def __init__(self, template):
            self.markers = ['a', 'b', 'c', 'd']
            self.template = template.copy()
            self.points = np.array([template + [frame, 0., 0.] for frame in range(10)])
            self.marker_exists = np.ones((10, 4), dtype=bool)
            self.translation = np.zeros((10, 3))
            self.rotation = np.zeros((10, 3))
            self.translation_exists = np.ones(10, dtype=bool)
            self.rotation_exists = np.ones(10, dtype=bool)
            self.constraints = list(self.markers)

        def GetMarkerNames(self, name):
            return list(self.markers)

        def GetConstraintNames(self, name, skeleton):
            return self.constraints

        def GetConstraintDetails(self, name, constraint, skeleton):
            return True, constraint, 'root', self.template[self.markers.index(constraint)], 1.

        def GetTrajectory(self, name, marker):
            index = self.markers.index(marker)
            return [*self.points[:, index].T, self.marker_exists[:, index]]

        def GetBoneTranslationAnimation(self, name, bone, skeleton, space):
            return [*self.translation.T, self.translation_exists]

        def GetBoneRotationAnimation(self, name, bone, skeleton, space):
            return [*self.rotation.T, self.rotation_exists]

    def setUp(self):
        self.template = np.array([[0., 0., 0.], [70., 0., 0.],
                                  [10., 110., 0.], [20., 30., 90.]])
        self.api = self.StoredPropAPI(self.template)
        self.session = SimpleNamespace(api=self.api)
        self.options = RigidOptions(tolerance_mm=0.1, max_step_mm=20.)
        self.ranges = dict(animation_frame_count=10, frame_count=6,
                           slice_start=2, slice_stop=8, play_start=101, play_end=106)
        self.row = dict(name='phone', bone='root', markers=list(self.api.markers),
                        classification='PARTIAL',
                        before=dict(marker_coverage=0.25, trackable_frames=1),
                        after=dict(marker_coverage=0.5, trackable_frames=3),
                        frame_states={'TRACKED': 3, 'LOST': 3},
                        unresolved_intervals=[[101, 103]],
                        missing_marker_intervals=[[101, 103]],
                        geometry_rms_max_mm=0.0, motion_gate_mm_per_frame=2.,
                        review_reasons=['unresolved_rigid_frames',
                                        'pose_motion_discontinuity', 'native_pose_incomplete'])

    def validate(self):
        return validate_prop_poses(self.session, [self.row], self.ranges, self.options)

    def test_final_repaired_frames_replace_provisional_partial_and_keep_audit(self):
        self.assertEqual(self.validate(), [])
        self.assertEqual(self.row['classification'], 'PRESENT')
        self.assertEqual(self.row['frame_states'], {'TRACKED': 6})
        self.assertEqual(self.row['after'], dict(marker_coverage=1.0, trackable_frames=6,
                                                native_pose_frames=6, usable_frames=6))
        self.assertEqual(self.row['unresolved_intervals'], [])
        self.assertEqual(self.row['missing_marker_intervals'], [])
        self.assertEqual(self.row['before'], dict(marker_coverage=0.25, trackable_frames=1))
        self.assertEqual(self.row['classification_before_fill'], 'PARTIAL')
        self.assertEqual(self.row['before_fill']['marker_coverage'], 0.5)
        self.assertEqual(self.row['before_fill']['frame_states'], {'TRACKED': 3, 'LOST': 3})
        self.assertEqual(self.row['before_fill']['unresolved_intervals'], [[101, 103]])
        self.assertIn('unresolved_rigid_frames', self.row['before_fill']['review_reasons'])

    def test_corrected_final_template_does_not_keep_a_provisional_template_failure(self):
        self.row['classification'] = 'MISSING_SETUP'
        self.row['review_reasons'] = ['invalid_calibrated_template']
        self.row['template_error'] = 'provisional constraint offsets were incomplete'
        self.assertEqual(self.validate(), [])
        self.assertEqual(self.row['classification'], 'PRESENT')
        self.assertNotIn('template_error', self.row)
        self.assertEqual(self.row['before_fill']['template_error'],
                         'provisional constraint offsets were incomplete')

    def test_final_repaired_track_is_read_even_if_pre_fill_had_no_trackable_frames(self):
        self.row['classification'] = 'ZERO_DATA'
        self.row['after'] = dict(marker_coverage=0.0, trackable_frames=0)
        self.row['frame_states'] = {'LOST': 6}
        self.assertEqual(self.validate(), [])
        self.assertEqual(self.row['classification'], 'PRESENT')
        self.assertEqual(self.row['after']['usable_frames'], 6)
        self.assertEqual(self.row['before_fill']['trackable_frames'], 0)
        self.assertEqual(self.row['before_fill']['classification'], 'ZERO_DATA')

    def test_full_marker_and_pose_coverage_cannot_hide_final_bad_geometry(self):
        self.api.points[4, 2] += [40., 0., 0.]
        self.assertEqual(self.validate(), ['prop:phone:unresolved_rigid_frames'])
        self.assertEqual(self.row['classification'], 'PARTIAL')
        self.assertEqual(self.row['after']['marker_coverage'], 1.0)
        self.assertEqual(self.row['after']['native_pose_frames'], 6)
        self.assertEqual(self.row['after']['trackable_frames'], 5)
        self.assertEqual(self.row['after']['usable_frames'], 5)
        self.assertEqual(self.row['frame_states'], {'TRACKED': 5, 'INVALID_GEOMETRY': 1})
        self.assertEqual(self.row['unresolved_intervals'], [[103, 103]])

    def test_pose_gaps_in_either_channel_use_processing_slice_and_remain_partial(self):
        self.api.translation_exists[0] = False  # Outside the processing slice.
        self.api.translation_exists[3] = False
        self.api.rotation_exists[6] = False
        self.assertEqual(self.validate(), ['prop:phone:native_pose_incomplete'])
        self.assertEqual(self.row['classification'], 'PARTIAL')
        self.assertEqual(self.row['after']['trackable_frames'], 6)
        self.assertEqual(self.row['native_pose']['valid_frames'], 4)
        self.assertEqual(self.row['frame_states'], {'TRACKED': 4, 'NATIVE_POSE_MISSING': 2})
        self.assertEqual(self.row['unresolved_intervals'], [[102, 102], [105, 105]])
        self.assertEqual(self.row['missing_marker_intervals'], [])

    def test_final_marker_gap_is_reported_without_rejecting_three_point_rigid_pose(self):
        self.api.marker_exists[3:5, 3] = False
        self.assertEqual(self.validate(), [])
        self.assertEqual(self.row['classification'], 'PRESENT')
        self.assertEqual(self.row['after']['trackable_frames'], 6)
        self.assertEqual(self.row['after']['usable_frames'], 6)
        self.assertEqual(self.row['missing_marker_intervals'], [[102, 103]])
        self.assertAlmostEqual(self.row['after']['marker_coverage'], 22 / 24)

    def test_insufficient_final_markers_cannot_be_hidden_by_complete_native_pose(self):
        self.api.marker_exists[4, 2:] = False
        self.assertEqual(self.validate(), ['prop:phone:unresolved_rigid_frames'])
        self.assertEqual(self.row['classification'], 'PARTIAL')
        self.assertEqual(self.row['frame_states'], {'TRACKED': 5, 'LOST': 1})
        self.assertEqual(self.row['unresolved_intervals'], [[103, 103]])
        self.assertEqual(self.row['missing_marker_intervals'], [[103, 103]])

    def test_final_motion_discontinuity_blocks_an_otherwise_complete_prop(self):
        self.api.points[5:] += [1000., 0., 0.]
        self.assertEqual(self.validate(), ['prop:phone:pose_motion_discontinuity'])
        self.assertEqual(self.row['classification'], 'PARTIAL')
        self.assertEqual(self.row['frame_states'], {'TRACKED': 5, 'MOTION_CONFLICT': 1})
        self.assertEqual(self.row['unresolved_intervals'], [[104, 104]])

    def test_missing_template_never_passes_on_complete_native_pose(self):
        self.api.constraints.pop()
        reasons = self.validate()
        self.assertIn('prop:phone:invalid_calibrated_template', reasons)
        self.assertIn('prop:phone:unresolved_rigid_frames', reasons)
        self.assertEqual(self.row['classification'], 'MISSING_SETUP')
        self.assertEqual(self.row['after']['usable_frames'], 0)
        self.assertEqual(self.row['unresolved_intervals'], [[101, 106]])

    def test_unsupported_setup_retains_structural_reason_without_becoming_present(self):
        del self.row['bone']
        self.row['review_reasons'] = ['missing_or_not_single_rigid_prop']
        self.assertEqual(self.validate(), ['prop:phone:missing_or_not_single_rigid_prop'])
        self.assertEqual(self.row['classification'], 'MISSING_SETUP')
        self.assertEqual(self.row['frame_states'], {'MISSING_SETUP': 6})
        self.assertEqual(self.row['after']['usable_frames'], 0)

    def test_nonfinite_existing_pose_in_processing_slice_is_rejected(self):
        self.api.rotation[4, 1] = np.inf
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite existing native prop pose'):
            self.validate()

    def test_malformed_pose_axis_count_is_rejected(self):
        self.api.GetBoneTranslationAnimation = lambda *args: [np.zeros(10)] * 3
        with self.assertRaisesRegex(RuntimeError, 'Invalid native prop pose array'):
            self.validate()

    def test_repeat_final_read_refreshes_quality_without_replacing_before_fill_audit(self):
        self.validate()
        self.api.rotation_exists[4] = False
        self.assertEqual(self.validate(), ['prop:phone:native_pose_incomplete'])
        self.assertEqual(self.row['classification'], 'PARTIAL')
        self.assertEqual(self.row['before_fill']['marker_coverage'], 0.5)
        self.assertEqual(self.row['before_fill']['frame_states'], {'TRACKED': 3, 'LOST': 3})
        self.assertEqual(self.row['unresolved_intervals'], [[103, 103]])

    def test_background_prop_does_not_emit_blocking_reasons_by_default(self):
        self.row['name'] = 'camera_1'
        self.api.rotation_exists[4] = False
        self.assertEqual(self.validate(), [])
        self.assertTrue(self.row['is_background'])
        self.assertEqual(self.row['review_reasons'], ['native_pose_incomplete'])
        reasons = validate_prop_poses(self.session, [self.row], self.ranges, self.options,
                                      include_background_reasons=True)
        self.assertEqual(reasons, ['prop:camera_1:native_pose_incomplete'])

    def test_interactive_prop_emits_blocking_reasons(self):
        self.row['name'] = 'blue_phone'
        self.api.rotation_exists[4] = False
        self.assertEqual(self.validate(), ['prop:blue_phone:native_pose_incomplete'])
        self.assertFalse(self.row['is_background'])
        self.assertEqual(self.row['review_reasons'], ['native_pose_incomplete'])


class NativeJitterRepairTests(unittest.TestCase):
    """SDK-shaped storage exercises real planning, sparse writes and readback."""

    class Channel:
        def __init__(self, owner, marker, field):
            self.owner, self.marker, self.field = owner, marker, field

        def __getitem__(self, key):
            start, stop = key.start - 99, key.stop - 99
            values = (self.owner.api.points[:, self.marker] if self.field == 'Translation'
                      else self.owner.metadata[self.field][:, self.marker])
            return values[start:stop].tolist()

        def __setitem__(self, key, values):
            if self.field != 'Translation':
                raise AssertionError('Inference must not invent native provenance enum values')
            self.owner.writes.append((self.marker, key.start, key.stop))
            if self.owner.failure == 'raise':
                raise RuntimeError('native setter failed')
            if self.owner.failure != 'ignore':
                self.owner.api.points[key.start - 99:key.stop - 99, self.marker] = np.asarray(values, np.float32)
            if self.owner.failure == 'drift':
                self.owner.api.points[key.start - 99, self.marker, 0] += .1
            if self.owner.failure == 'collateral':
                self.owner.api.points[2, self.marker, 0] += 1.
            if self.owner.failure == 'metadata':
                self.owner.metadata['FillType'][key.start - 99, self.marker] += 1

        def GetGaps(self):
            return [(103, 103)] if self.owner.failure == 'gap' else []

    def setUp(self):
        template = np.array([[0., 0., 0.], [70., 0., 0.],
                             [10., 110., 0.], [20., 30., 90.]])
        self.api = FinalPropQualityTests.StoredPropAPI(template)
        self.api.points[4, 2, 0] += 3.
        self.original = self.api.points.copy()
        self.metadata = {'FillType': np.zeros((10, 4), dtype=int),
                         'ReconIndex': np.full((10, 4), 23, dtype=int),
                         'Radius': np.ones((10, 4))}
        self.writes = []
        self.failure = None
        nodes = {}
        for marker, name in enumerate(self.api.markers):
            nodes['phone\\' + name] = SimpleNamespace(
                Translation=self.Channel(self, marker, 'Translation'),
                Channels={field: self.Channel(self, marker, field) for field in self.metadata})
        self.scene = SimpleNamespace(Objects=nodes)
        self.session = SimpleNamespace(api=self.api)
        self.characters = {'phone': SimpleNamespace(Path='phone')}
        self.row = dict(name='phone', bone='root', markers=list(self.api.markers),
                        classification='PARTIAL', repaired_marker_frames=7,
                        review_reasons=['unresolved_rigid_frames'])
        self.ranges = dict(animation_frame_count=10, frame_count=6,
                           slice_start=2, slice_stop=8, play_start=101, play_end=106)
        self.rigid = RigidOptions(tolerance_mm=0.1, max_step_mm=20.)

    def repair(self, options=None):
        return repair_prop_jitter(self.session, self.scene, self.characters, [self.row],
                                  self.ranges, self.rigid, options or JitterOptions())

    def test_sparse_existing_correction_keeps_measured_reassociation_separate(self):
        self.assertEqual(self.repair(), ['phone'])
        audit = self.row['jitter_repair']
        self.assertEqual(audit['corrected_frames'], 1)
        self.assertEqual(audit['corrected_intervals'], [[103, 103]])
        self.assertTrue(audit['native_readback_verified'])
        self.assertEqual(audit['provenance'], 'model_derived_not_measured')
        self.assertFalse(audit['camera_measurements'])
        self.assertFalse(audit['native_fill_type_modified'])
        self.assertEqual(audit['native_fill_type_semantics'], 'unverified_not_measured_provenance')
        self.assertEqual(self.row['repaired_marker_frames'], 7)
        self.assertTrue(self.writes)
        self.assertTrue(all((begin, end) == (103, 104) for _, begin, end in self.writes))
        np.testing.assert_array_equal(self.api.points[:4], self.original[:4])
        np.testing.assert_array_equal(self.api.points[5:], self.original[5:])
        np.testing.assert_array_equal(self.metadata['FillType'], 0)
        np.testing.assert_array_equal(self.metadata['ReconIndex'], 23)
        for correction in audit['corrections']:
            self.assertEqual(correction['frame'], 103)
            index = self.api.markers.index(correction['marker'])
            self.assertEqual(correction['original_xyz'], self.original[4, index].tolist())
            self.assertLessEqual(correction['displacement_mm'], 5.)
            self.assertEqual(correction['native_xyz'], self.api.points[4, index].tolist())
        json.dumps(audit, allow_nan=False)

    def test_six_significant_digit_native_readback_keeps_corrections_bounded(self):
        self.api.points += [1200.123, 700.456, 1300.789]
        original = self.api.points.copy()
        get_trajectory = self.api.GetTrajectory

        def text_precision(subject, marker):
            values = get_trajectory(subject, marker)
            return tuple([[float(f'{value:.6g}') for value in axis] for axis in values[:3]]
                         + [values[3]])

        self.api.GetTrajectory = text_precision
        self.assertEqual(self.repair(), ['phone'])
        audit = self.row['jitter_repair']
        self.assertTrue(audit['native_readback_verified'])
        self.assertLess(audit['native_readback_max_axis_error_mm'], .006)
        self.assertLessEqual(max(np.linalg.norm(np.asarray(item['native_xyz'])
                                               - item['original_xyz'])
                                 for item in audit['corrections']), 5.)
        np.testing.assert_array_equal(self.api.points[:4], original[:4])
        np.testing.assert_array_equal(self.api.points[5:], original[5:])

    def test_disabled_correction_never_reads_or_writes_native_data(self):
        self.session.api = None
        self.assertEqual(self.repair(JitterOptions(max_correction_mm=0.)), [])
        self.assertEqual(self.writes, [])
        self.assertEqual(self.row['jitter_repair']['reason'], 'disabled_zero_max_correction')

    def test_out_of_bound_geometry_is_audited_not_overwritten(self):
        self.api.points[4, 2, 0] += 40.
        original = self.api.points.copy()
        self.assertEqual(self.repair(), [])
        np.testing.assert_array_equal(self.api.points, original)
        self.assertEqual(self.writes, [])
        audit = self.row['jitter_repair']
        self.assertTrue(audit['rejections'])
        self.assertEqual((audit['rejections'][0]['start'], audit['rejections'][0]['end']), (103, 103))

    def test_native_failures_are_not_accepted_as_successful_corrections(self):
        for failure in ('raise', 'ignore', 'drift', 'collateral', 'metadata', 'gap'):
            with self.subTest(failure=failure):
                self.setUp()
                self.failure = failure
                with self.assertRaises(RuntimeError):
                    self.repair()
                self.assertFalse(self.row['jitter_repair']['native_readback_verified'])
                if failure == 'gap':
                    self.assertEqual(self.writes, [])

    def test_native_pose_quality_remains_authoritative_after_correction(self):
        self.assertEqual(self.repair(), ['phone'])
        self.api.rotation_exists[4] = False
        self.assertEqual(validate_prop_poses(self.session, [self.row], self.ranges, self.rigid),
                         ['prop:phone:native_pose_incomplete'])
        self.assertEqual(self.row['classification'], 'PARTIAL')
        self.assertTrue(self.row['jitter_repair']['native_readback_verified'])

if __name__ == '__main__':
    unittest.main()

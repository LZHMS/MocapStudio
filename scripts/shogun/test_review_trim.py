"""Deterministic inclusive trim policy and real binary-FBX time-boundary tests."""
from copy import deepcopy
from pathlib import Path
import struct
import tempfile
import unittest

from pipeline_review_trim import (
    common_good_windows, fbx_key_bounds, required_prop_names, trim_plan,
    unavailable_intervals, validate_fbx_clip,
)
from pipeline_split import fbx
from test_split import _array, _emit, _integer, _node, _string


def report_with_gaps(first=101, count=1000, gaps=None):
    gaps = [[first, first + 49], [first + count - 50, first + count - 1]] if gaps is None else gaps
    return {
        'range': {'play_start': first, 'play_end': first + count - 1},
        'policy': {'camera_exemptions': ['camera_1', 'camera_2', 'camera_3']},
        'subjects': [{
            'name': 'actor', 'processing_target': True,
            'setup': {'markers': 2, 'labeling_bones': 5, 'solving_bones': 5},
            'after': {'core_markers': ['head', 'pelvis'], 'unconfigured_core_markers': [],
                      'core_unavailable_intervals': deepcopy(gaps)},
            'solve': {s: {'unavailable_intervals': []} for s in ('labeling', 'solving')},
        }],
        'props': [], 'review_reasons': ['actor:partial'],
    }


def write_curve_scene(path, curves, *, frame_rate=None):
    objects = []
    for uid, times in enumerate(curves, 1):
        objects.append(_node('AnimationCurve', [_integer(uid), _string(f'curve{uid}'), _string('')], [
            _node('KeyTime', [_array('l', times)]),
            _node('KeyValueFloat', [_array('f', [float(i) for i in range(len(times))])]),
        ]))
    data = bytearray(fbx.MAGIC + struct.pack('<I', 7700))
    if frame_rate is not None:
        settings = _node('GlobalSettings', children=[_node('Properties70', children=[
            _node('P', [_string('TimeMode'), _string('enum'), _string(''), _string(''), _integer(18)]),
            _node('P', [_string('CustomFrameRate'), _string('double'), _string('Number'), _string(''),
                        b'D' + struct.pack('<d', frame_rate)]),
        ])])
        data.extend(_emit(settings, len(data)))
    for node in (_node('Objects', children=objects), _node('Connections')):
        data.extend(_emit(node, len(data)))
    data.extend(bytes(25))
    path.write_bytes(data)


class InclusiveWindowTests(unittest.TestCase):
    def test_masks_preserve_nonzero_and_negative_frame_origins(self):
        self.assertEqual(unavailable_intervals([False, False, True, False], -3), [[-3, -2], [0, 0]])
        self.assertEqual(unavailable_intervals([True, True], 51), [])
        self.assertEqual(unavailable_intervals([], 51), [])

    def test_union_overlaps_adjacency_clipping_and_single_frame_windows(self):
        bad = [[[-9, 10], [14, 16]], [[10, 11], [13, 14], [20, 40]]]
        self.assertEqual(common_good_windows(10, 20, bad), [[12, 12], [17, 19]])
        self.assertEqual(common_good_windows(7, 7, []), [[7, 7]])
        self.assertEqual(common_good_windows(7, 7, [[[7, 7]]]), [])

    def test_invalid_intervals_are_rejected(self):
        with self.assertRaises(ValueError):
            common_good_windows(3, 2, [])
        with self.assertRaises(ValueError):
            common_good_windows(1, 5, [[[4, 3]]])

    def test_exact_ten_percent_loss_is_allowed_inclusive(self):
        plan = trim_plan(report_with_gaps())
        self.assertTrue(plan['eligible'])
        self.assertEqual(plan['selected'], {'start': 151, 'end': 1050,
                                           'frame_count': 900, 'retained_fraction': .9})
        self.assertFalse(plan['concatenation_allowed'])
        worse = report_with_gaps(gaps=[[101, 151], [1051, 1100]])
        self.assertFalse(trim_plan(worse)['eligible'])

    def test_minimum_frame_boundary_is_not_rounded_up(self):
        self.assertTrue(trim_plan(report_with_gaps(1, 133, [[1, 13]]))['eligible'])
        self.assertFalse(trim_plan(report_with_gaps(1, 132, [[1, 13]]))['eligible'])

    def test_human_skeleton_gaps_intersect_marker_and_prop_windows(self):
        report = report_with_gaps()
        report['subjects'][0]['solve']['solving']['unavailable_intervals'] = [[151, 151]]
        self.assertFalse(trim_plan(report)['eligible'])
        report['subjects'][0]['solve']['solving']['unavailable_intervals'] = [[101, 149]]
        self.assertTrue(trim_plan(report)['eligible'])

    def test_long_internal_gap_retains_independent_segments_never_stitches(self):
        plan = trim_plan(report_with_gaps(1, 1000, [[451, 550]]))
        self.assertFalse(plan['eligible'])
        self.assertEqual([(s['start'], s['end']) for s in plan['segment_candidates']],
                         [(1, 450), (551, 1000)])
        self.assertIsNone(plan['selected'])
        self.assertFalse(plan['concatenation_allowed'])

    def test_roster_and_setup_blockers_override_safe_temporal_window(self):
        for reason in ('confirmed_subject_roster_required', 'expected_roster_unknown',
                       'human_count_mismatch:expected=2:detected=1', 'missing_setup:other'):
            report = report_with_gaps()
            report['review_reasons'].append(reason)
            plan = trim_plan(report)
            self.assertFalse(plan['eligible'])
            self.assertIn(reason, plan['blockers'])
            self.assertEqual(plan['segment_candidates'][0]['frame_count'], 900)
        report = report_with_gaps()
        report['subjects'][0]['setup']['solving_bones'] = 0
        self.assertFalse(trim_plan(report)['eligible'])
        report = report_with_gaps()
        report['subjects'][0]['after']['unconfigured_core_markers'] = ['ankle']
        self.assertFalse(trim_plan(report)['eligible'])

    def test_missing_evidence_never_means_complete_availability(self):
        report = report_with_gaps()
        del report['subjects'][0]['solve']['labeling']['unavailable_intervals']
        self.assertFalse(trim_plan(report)['eligible'])

    def test_only_explicit_failed_named_cameras_are_waived(self):
        props = [{'name': name, 'review_reasons': ['native_pose_incomplete']}
                 for name in ('camera_1', 'camera_2', 'camera_3', 'camera_4', 'Camera_1', 'chair_1', 'desk')]
        self.assertEqual(required_prop_names(props, ['camera_1', 'chair_1', 'desk']),
                         ['camera_2', 'camera_3', 'camera_4', 'Camera_1', 'chair_1', 'desk'])
        self.assertEqual(required_prop_names([{'name': 'camera_1', 'review_reasons': []}], ['camera_1']),
                         ['camera_1'])
        self.assertEqual(len(required_prop_names(props)), len(props))

    def test_failed_camera_does_not_shrink_window_but_chair_does(self):
        report = report_with_gaps()
        report['props'] = [{'name': 'camera_1', 'review_reasons': ['native_pose_incomplete'],
                            'unresolved_intervals': [[101, 1100]]}]
        self.assertTrue(trim_plan(report)['eligible'])
        report['props'].append({'name': 'chair_1', 'review_reasons': ['native_pose_incomplete'],
                                'unresolved_intervals': [[151, 151]]})
        report['review_reasons'].append('prop:chair_1:native_pose_incomplete')
        self.assertFalse(trim_plan(report)['eligible'])

    def test_non_temporal_prop_error_cannot_be_trimmed_away(self):
        report = report_with_gaps()
        report['props'] = [{'name': 'desk', 'review_reasons': ['invalid_calibrated_template'],
                            'unresolved_intervals': [[101, 150]]}]
        report['review_reasons'].append('prop:desk:invalid_calibrated_template')
        self.assertFalse(trim_plan(report)['eligible'])


class FbxClipTimeTests(unittest.TestCase):
    def test_real_key_bounds_validate_selected_frames_with_native_origin(self):
        ticks = 46186158000
        fps = 100.0
        step = ticks // 100
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / 'full.fbx'
            clip = Path(directory) / 'clip.fbx'
            write_curve_scene(full, [[100 * step, 200 * step, 1099 * step]])
            write_curve_scene(clip, [[150 * step, 300 * step, 1049 * step],
                                     [150 * step, 1049 * step]])
            original = fbx_key_bounds(full)
            evidence = validate_fbx_clip(clip, original,
                {'animation_start': 101, 'animation_end': 1100}, {'start': 151, 'end': 1050}, fps)
            self.assertTrue(evidence['validated'])
            self.assertEqual(evidence['native_frame_zero_tick'], -step)
            self.assertEqual(evidence['curves'], 2)
            self.assertEqual(evidence['keys'], 5)

    def test_full_range_or_one_outside_key_is_rejected(self):
        step = 461861580
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / 'full.fbx'
            clip = Path(directory) / 'clip.fbx'
            write_curve_scene(full, [[0, 999 * step]])
            original = fbx_key_bounds(full)
            for times in ([0, 999 * step], [49 * step, 949 * step], [50 * step, 950 * step]):
                write_curve_scene(clip, [list(times)])
                with self.assertRaises(ValueError):
                    validate_fbx_clip(clip, original, {'animation_start': 1, 'animation_end': 1000},
                                      {'start': 51, 'end': 950}, 100.)

    def test_missing_keys_and_unestablished_full_time_origin_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'curves.fbx'
            write_curve_scene(path, [[]])
            with self.assertRaises(ValueError):
                fbx_key_bounds(path)
            write_curve_scene(path, [[0, 100]])
            with self.assertRaises(ValueError):
                validate_fbx_clip(path, fbx_key_bounds(path),
                    {'animation_start': 1, 'animation_end': 1000}, {'start': 51, 'end': 950}, 100.)

    def test_nominal_sdk_ntsc_rounding_uses_declared_fbx_grid_not_looser_key_bounds(self):
        ticks, rate = 46186158000, 120000 / 1001
        sdk_rate = 119.87999725341797
        at = lambda frame: round(frame * ticks / rate)
        with tempfile.TemporaryDirectory() as directory:
            full, clip = Path(directory) / 'full.fbx', Path(directory) / 'clip.fbx'
            write_curve_scene(full, [[at(1), at(1000)]], frame_rate=rate)
            write_curve_scene(clip, [[at(4), at(1000)]], frame_rate=rate)
            original = fbx_key_bounds(full)
            evidence = validate_fbx_clip(clip, original,
                {'animation_start': 1, 'animation_end': 1000}, {'start': 4, 'end': 1000}, sdk_rate)
            self.assertTrue(evidence['validated'])
            self.assertEqual(evidence['frame_rate'], rate)
            write_curve_scene(clip, [[at(3), at(1000)]], frame_rate=rate)
            with self.assertRaises(ValueError):
                validate_fbx_clip(clip, original, {'animation_start': 1, 'animation_end': 1000},
                                  {'start': 4, 'end': 1000}, sdk_rate)

    def test_incompatible_declared_rate_or_truncated_full_extent_is_not_rounding(self):
        ticks, rate = 46186158000, 120000 / 1001
        at = lambda frame: round(frame * ticks / rate)
        with tempfile.TemporaryDirectory() as directory:
            full, clip = Path(directory) / 'full.fbx', Path(directory) / 'clip.fbx'
            write_curve_scene(full, [[at(1), at(999)]], frame_rate=rate)
            write_curve_scene(clip, [[at(4), at(1000)]], frame_rate=rate)
            with self.assertRaises(ValueError):
                validate_fbx_clip(clip, fbx_key_bounds(full),
                    {'animation_start': 1, 'animation_end': 1000}, {'start': 4, 'end': 1000}, rate)
            write_curve_scene(full, [[at(1), at(1000)]], frame_rate=rate)
            with self.assertRaises(ValueError):
                validate_fbx_clip(clip, fbx_key_bounds(full),
                    {'animation_start': 1, 'animation_end': 1000}, {'start': 4, 'end': 1000}, 120.)


if __name__ == '__main__':
    unittest.main()

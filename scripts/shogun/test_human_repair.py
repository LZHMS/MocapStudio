"""Native human relabel key-transfer contracts, with in-memory channel wrappers."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pipeline_human_repair import apply_reference_relabel, verify_reference_preservation, propose_short_gap


class Channel:
    def __init__(self, values):
        self.values = list(values)
    def __getitem__(self, span):
        return self.values[span.start - 1:span.stop - 1]
    def __setitem__(self, span, values):
        # Real ChannelReader serializes str(value) into native numeric syntax.
        # Parsing that wire text catches NumPy repr wrappers, unlike list copy.
        self.values[span.start - 1:span.stop - 1] = ast.literal_eval(str(values))
    def GetGaps(self):
        return []


class HumanReferenceTransferTests(unittest.TestCase):
    def test_closed_chain_preserves_displaced_coordinates_and_native_metadata(self):
        count = 100
        ranges = dict(play_start=1, play_end=count, animation_start=1, animation_end=count,
                      frame_count=count, animation_frame_count=count, slice_start=0, slice_stop=count)
        nodes = {}
        for index, (name, height) in enumerate((('RKNE', 600), ('RSHN', 450), ('RANK', 280), ('free', 65))):
            path = 'unassigned\\free' if name == 'free' else 'actor\\' + name
            points = [(float(frame), 0., float(height)) for frame in range(count)]
            nodes[path] = SimpleNamespace(Path=path, _ID=index, Translation=Channel(points),
                                         Channels={field: Channel([np.int64(index * 10 + offset)] * count)
                                                   for offset, field in enumerate(('Radius', 'ReconIndex', 'FillType'))})
        api = SimpleNamespace(GetMarkerNames=lambda subject: ['RKNE', 'RSHN', 'RANK'])
        session = SimpleNamespace(api=api)
        scene = SimpleNamespace(Objects=nodes)
        characters = {'actor': SimpleNamespace(Path='actor')}
        reference = dict(ranges=ranges, points={name: np.column_stack((np.arange(count), np.zeros(count), np.full(count, height)))
                                               for name, height in (('RKNE', 450), ('RSHN', 280), ('RANK', 65))})
        pool = {frame: [(3, np.array([frame, 0., 65.]))] for frame in range(count)}
        def trajectory(api, subject, marker, ranges):
            return np.asarray(nodes['actor\\' + marker].Translation.values, dtype=float).copy()
        with patch('pipeline_human_repair._trajectory', side_effect=trajectory), \
             patch('pipeline_human_repair._read_pool', return_value=(pool, {3: nodes['unassigned\\free']})):
            with patch.object(Channel, '__setitem__', lambda channel, span, values: None):
                with self.assertRaisesRegex(RuntimeError, 'coordinate write/readback mismatch'):
                    apply_reference_relabel(session, scene, characters, 'actor', reference, ranges)
            audit, expected = apply_reference_relabel(session, scene, characters, 'actor', reference, ranges)
        self.assertEqual(len(audit['corrections']), 400)
        self.assertEqual(nodes['actor\\RKNE'].Translation.values[0][2], 450.)
        self.assertEqual(nodes['actor\\RSHN'].Translation.values[0][2], 280.)
        self.assertEqual(nodes['actor\\RANK'].Translation.values[0][2], 65.)
        self.assertEqual(nodes['unassigned\\free'].Translation.values[0][2], 600.)
        self.assertEqual(nodes['actor\\RANK'].Channels['ReconIndex'].values, [31] * count)
        self.assertEqual(nodes['unassigned\\free'].Channels['ReconIndex'].values, [1] * count)
        self.assertEqual(verify_reference_preservation(scene, expected, ranges), [])
        nodes['unassigned\\free'].Channels['FillType'].values[50] = 999
        self.assertTrue(verify_reference_preservation(scene, expected, ranges))

    def test_virtual_metadata_in_missing_spans_is_not_an_observed_key(self):
        ranges = dict(play_start=1, play_end=5, frame_count=5)
        points = np.zeros((5, 3))
        points[3:] = np.nan
        node = SimpleNamespace(Path='free', Translation=Channel(np.zeros((5, 3)).tolist()),
                               Channels={'Radius': Channel([7., 7., 7., 500., 600.])})
        node.Translation.GetGaps = lambda: [(4, 5)]
        scene = SimpleNamespace(Objects={'free': node})
        expected = {'free': dict(points=points, metadata={'Radius': [7., 7., 7., 10., 20.]})}
        self.assertEqual(verify_reference_preservation(scene, expected, ranges), [])
        node.Channels['Radius'].values[1] = 9.
        self.assertTrue(verify_reference_preservation(scene, expected, ranges))

    def test_derived_gap_materializes_model_metadata_instead_of_virtual_neighbor_values(self):
        class InterpolatingChannel(Channel):
            def __init__(self, values):
                super().__init__(values)
                self.keyed = np.zeros(len(values), dtype=bool)
            def __setitem__(self, span, values):
                super().__setitem__(span, values)
                self.keyed[span.start - 1:span.stop - 1] = True
            def __getitem__(self, span):
                keys = np.flatnonzero(self.keyed)
                if not len(keys):
                    return super().__getitem__(span)
                return np.interp(np.arange(span.start - 1, span.stop - 1), keys,
                                 np.asarray(self.values)[keys]).tolist()
        count = 100
        ranges = dict(play_start=1, play_end=count, frame_count=count)
        nodes = {}
        specifications = [('actor\\RKNI', 0., 600., (7., -1, 1)),
                          ('actor\\RKNE', 100., 65., (7., -1, 1)),
                          ('free', 0., 65., (6., 210, 0))]
        for index, (path, offset, height, fields) in enumerate(specifications):
            nodes[path] = SimpleNamespace(
                Path=path, _ID=index, Translation=Channel([(frame + offset, 0., height) for frame in range(count)]),
                Channels={field: InterpolatingChannel([value] * count)
                          for field, value in zip(('Radius', 'ReconIndex', 'FillType'), fields)})
        nodes['free'].Translation.GetGaps = lambda: [(51, 51)]
        scene = SimpleNamespace(Objects=nodes)
        session = SimpleNamespace(api=SimpleNamespace(GetMarkerNames=lambda subject: ['RKNI', 'RKNE']))
        reference = dict(ranges=ranges, points={'RKNI': np.column_stack((np.arange(count), np.zeros(count), np.full(count, 65.)))})
        pool = {frame: [(2, np.array([frame, 0., 65.]))] for frame in range(count) if frame != 50}
        def trajectory(api, subject, marker, ranges):
            return np.asarray(nodes['actor\\' + marker].Translation.values, dtype=float).copy()
        groups = {'leg': {'RKNI': np.zeros(3), 'RKNE': np.array([100., 0., 0.])}}
        with patch('pipeline_human_repair._trajectory', side_effect=trajectory), \
             patch('pipeline_human_repair._read_pool', return_value=(pool, {2: nodes['free']})), \
             patch('pipeline_human.read_calibration', return_value=(groups, [])):
            audit, expected = apply_reference_relabel(
                session, scene, {'actor': SimpleNamespace(Path='actor')}, 'actor', reference, ranges)
        gap = next(row for row in audit['corrections'] if row.get('provenance') == 'bounded_short_linear_model_derived_not_measured')
        self.assertEqual(gap['frame'], 51)
        self.assertEqual(gap['metadata_written'], {'Radius': 7., 'ReconIndex': -1, 'FillType': 1})
        self.assertEqual(nodes['actor\\RKNI'].Channels['ReconIndex'][51:52], [-1.])
        self.assertTrue(nodes['actor\\RKNI'].Channels['ReconIndex'].keyed[50])
        self.assertEqual(verify_reference_preservation(scene, expected, ranges), [])


    def test_identified_single_gap_is_model_derived_and_local_shape_checked(self):
        track = np.column_stack((np.arange(30) * .1, np.zeros(30), np.zeros(30)))
        recovered = np.ones(30, dtype=bool)
        recovered[10] = False
        original = track.copy()
        track[10] = [500, 500, 500]
        neighbors = {'neighbor': original + [0., 100., 0.]}
        offsets = [{'target': np.zeros(3), 'neighbor': np.array([0., 100., 0.])}]
        proposed, reason = propose_short_gap('target', track, recovered, 10, 11, offsets, neighbors)
        self.assertIsNone(reason)
        np.testing.assert_allclose(proposed[0], original[10])
        self.assertEqual(track[10, 0], 500, 'Planning must preserve original keys')
        neighbors['neighbor'][10, 1] = 500
        proposed, reason = propose_short_gap('target', track, recovered, 10, 11, offsets, neighbors)
        self.assertIsNone(proposed)
        self.assertEqual(reason, 'short_gap_calibrated_local_shape_conflict')

    def test_long_edge_and_conflicting_temporal_gaps_are_not_filled(self):
        track = np.zeros((30, 3))
        recovered = np.ones(30, dtype=bool)
        offsets = [{'target': np.zeros(3), 'neighbor': np.array([0., 100., 0.])}]
        neighbors = {'neighbor': track + [0., 100., 0.]}
        for lo, hi in ((10, 16), (0, 1), (29, 30)):
            proposed, _ = propose_short_gap('target', track, recovered, lo, hi, offsets, neighbors)
            self.assertIsNone(proposed)
        track[8, 0] = 100
        proposed, reason = propose_short_gap('target', track, recovered, 10, 11, offsets, neighbors)
        self.assertIsNone(proposed)
        self.assertEqual(reason, 'short_gap_temporal_anchor_conflict')

class HumanOnlyCheckpointScopeTests(unittest.TestCase):
    def test_retained_props_still_block_quality_while_actual_human_keys_are_repaired(self):
        from pipeline_processing import process_take
        count = 100
        nodes = {}
        for index, (name, height) in enumerate((('RKNE', 600), ('RSHN', 450), ('RANK', 280), ('free', 65))):
            path = 'unassigned\\free' if name == 'free' else 'actor\\' + name
            nodes[path] = SimpleNamespace(Path=path, _ID=index,
                Translation=Channel([(float(frame), 0., float(height)) for frame in range(count)]),
                Channels={field: Channel([index * 10 + offset] * count)
                          for offset, field in enumerate(('Radius', 'ReconIndex', 'FillType'))})
        actor = SimpleNamespace(Name='actor', Path='actor', Type='Character', _ID=10)
        bone = SimpleNamespace(Name='root', Path='actor\\root')

        class Objects(dict):
            def FilterByType(self, kind):
                return [actor] if kind == 'Character' else []

        selected = []
        scene = SimpleNamespace(Objects=Objects(nodes), New=lambda: None,
            DeselectAllObjects=selected.clear, SelectObject=selected.append,
            GetSelectedObjects=lambda: list(selected))
        timeline = SimpleNamespace(GetAnimationStart=lambda: 1, GetAnimationEnd=lambda: count,
            GetPlayStart=lambda: 1, GetPlayEnd=lambda: count, SetTimeFrames=lambda frame: None)

        def trajectory(subject, marker):
            points = np.asarray(nodes[subject + '\\' + marker].Translation.values, dtype=float)
            return (*points.T.tolist(), [True] * count)

        def hsl(command):
            if command.startswith('getNumKeys'):
                return str(count)
            if command.startswith('saveFile '):
                Path(ast.literal_eval(command[len('saveFile '):].rstrip(';'))).write_bytes(b'export')
            return ''

        session = SimpleNamespace(hsl=hsl, api=SimpleNamespace(
            GetMarkerNames=lambda subject: ['RKNE', 'RSHN', 'RANK'],
            GetTrajectory=trajectory, LoadFile=lambda path: None,
            SaveFile=lambda path: Path(path).write_bytes(b'fresh native'),
            GetBoneTranslationAnimation=lambda *args: ([0.] * count,) * 3 + ([True] * count,),
            GetBoneRotationAnimation=lambda *args: ([0.] * count,) * 3 + ([True] * count,)))
        ranges = dict(animation_start=1, animation_end=count, play_start=1, play_end=count)
        reference = dict(ranges=dict(ranges, frame_count=count), points={
            name: np.column_stack((np.arange(count), np.zeros(count), np.full(count, height)))
            for name, height in (('RKNE', 450), ('RSHN', 280), ('RANK', 65))})
        pool = {frame: [(3, np.array([frame, 0., 65.]))] for frame in range(count)}
        original_props = [dict(name='missing_prop', classification='MISSING_SETUP',
                               review_reasons=['missing_or_not_single_rigid_prop'])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, checkpoint, previous = root / 'take.mcp', root / 'checkpoint.vdf', root / 'report.json'
            raw.write_bytes(b'original MCP')
            checkpoint.write_bytes(b'proven checkpoint')
            previous.write_bytes(b'proven report payload')
            stage = root / 'stage'
            stage.mkdir()
            stat = checkpoint.stat()
            job = dict(job_id='take', input_path=str(raw), input_kind='mcp',
                expected_subjects=['actor'], expected_props=['missing_prop'],
                checkpoint_vdf=str(checkpoint), checkpoint_audit=dict(
                    job_id='take', input_path=str(raw), artifacts={'vdf': str(checkpoint)},
                    range=ranges, props=deepcopy(original_props),
                    stages=[dict(name=name, status='SUCCESS')
                            for name in ('fix_occlusion', 'solve_labeling', 'solve_solving')]),
                review_repair=True, repair_checkpoint_props=False,
                review_policy=dict(auto_trim=False, camera_exemptions=[]),
                source_report=str(previous), source_report_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
                checkpoint_fingerprint=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns))
            with patch('pipeline_processing._sdk_context', return_value=(scene, timeline)), \
                 patch('pipeline_processing._descendants',
                       side_effect=lambda session, scene, character, kind, **kwargs:
                       [bone] if kind in ('LabelingBone', 'SolvingBone') else []), \
                 patch('pipeline_human_reference.capture_source_labels', return_value={'actor': reference}), \
                 patch('pipeline_human_repair._read_pool', return_value=(pool, {3: nodes['unassigned\\free']})), \
                 patch('pipeline_human.inspect_human_geometry', return_value=dict(usable=True, review_reasons=[])), \
                 patch('pipeline_props.process_props', side_effect=AssertionError('must retain prop bindings')), \
                 patch('pipeline_props.repair_prop_jitter', side_effect=AssertionError('must not jitter props')), \
                 patch('pipeline_review_trim.apply_review_trim', return_value=[]), \
                 patch('pipeline_review_trim.fbx_key_bounds', return_value={}):
                result = process_take(session, job, stage, {})
            if getattr(session, '_native_load_directory', None) is not None:
                session._native_load_directory.cleanup()
            self.assertEqual(result['status'], 'NEEDS_REVIEW', result.get('error'))
            self.assertEqual(raw.read_bytes(), b'original MCP')
            self.assertEqual(checkpoint.read_bytes(), b'proven checkpoint')
        self.assertEqual(nodes['actor\\RKNE'].Translation.values[0][2], 450.)
        self.assertEqual(nodes['actor\\RSHN'].Translation.values[0][2], 280.)
        self.assertEqual(nodes['actor\\RANK'].Translation.values[0][2], 65.)
        self.assertTrue(result['processing_source']['prop_bindings_retained'])
        self.assertEqual(result['processing_source']['reference_repaired_humans'], ['actor'])
        self.assertFalse(result['quality']['usable'])
        self.assertTrue(any(reason.startswith('prop:missing_prop:') for reason in result['review_reasons']))
        self.assertEqual(result['props'][0]['classification'], 'MISSING_SETUP')
        self.assertEqual([entry['status'] for entry in result['stages']
                          if entry['name'] == 'validate_prop_poses'], ['SUCCESS', 'SUCCESS'])
        self.assertEqual(next(entry['status'] for entry in result['stages']
                              if entry['name'] == 'repair_original_live_label_disagreement'), 'SUCCESS')



if __name__ == '__main__':
    unittest.main()

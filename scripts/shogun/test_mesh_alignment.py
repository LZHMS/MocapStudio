"""Behavioral calibration tests using small, independently encoded binary FBXs."""
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mesh_alignment import analyze_fbx, load_catalog, load_mapping


_ASYMMETRIC = np.array([[0., 0., 0.], [71., 0., 0.],
                        [11., 93., 0.], [17., 22., 51.]])
_SQUARE = np.array([[0., 0., 0.], [80., 0., 0.],
                    [80., 80., 0.], [0., 80., 0.]])
_TICKS = 46186158000


def _string(value):
    raw = value.encode('utf-8')
    return b'S' + struct.pack('<I', len(raw)) + raw


def _long(value):
    return b'L' + struct.pack('<q', value)


def _double(value):
    return b'D' + struct.pack('<d', value)


def _array(kind, values):
    fmt = 'q' if kind == 'l' else kind
    blob = struct.pack('<' + fmt * len(values), *values)
    return kind.encode() + struct.pack('<III', len(values), 0, len(blob)) + blob


def _node(name, properties=(), children=()):
    return (name.encode(), list(properties), list(children))


def _encode(node, offset):
    name, properties, children = node
    payload = b''.join(properties)
    position = offset + 25 + len(name) + len(payload)
    bodies = []
    for child in children:
        encoded = _encode(child, position)
        bodies.append(encoded)
        position += len(encoded)
    if children:
        bodies.append(bytes(25))
        position += 25
    return (struct.pack('<QQQB', position, len(properties), len(payload), len(name))
            + name + payload + b''.join(bodies))


def _property(name, values):
    return _node('P', [_string(name), _string('Vector3D' if len(values) == 3 else 'double'),
                       _string(''), _string('A')] + [_double(value) for value in values])


def _scene(path, *, prop='white_cup', points=_ASYMMETRIC, unit_scale=1., frames=130,
           missing_marker=False, malformed=None, drift=False, meshed=False,
           pre_rotation=False, human_only=False, unscoped_markers=False):
    """Encode real FBX nodes/curves, not mocks of the alignment implementation."""
    objects, connections = [], []
    next_uid = 100

    def model(uid, name, kind, parent, properties=()):
        objects.append(_node('Model', [_long(uid), _string(name + '\x00\x01Model'),
                                       _string(kind)], [_node('Properties70', children=properties)]))
        connections.append(_node('C', [_string('OO'), _long(uid), _long(parent)]))

    def curves(uid, attr, values):
        nonlocal next_uid
        acn = next_uid
        next_uid += 1
        objects.append(_node('AnimationCurveNode', [_long(acn), _string(attr), _string('')]))
        connections.append(_node('C', [_string('OP'), _long(acn), _long(uid), _string(attr)]))
        connections.append(_node('C', [_string('OO'), _long(acn), _long(91)]))
        for axis, component in zip('XYZ', values.T):
            if malformed == 'missing' and uid == 2 and attr == 'Lcl Rotation' and axis == 'Z':
                continue
            curve_uid = next_uid
            next_uid += 1
            times = [index * (_TICKS // 60) for index in range(frames)]
            component = component.tolist()
            if malformed == 'unequal' and uid == 3 and axis == 'Y':
                times[-1] += 1
            if malformed == 'nan' and uid == 3 and axis == 'Z':
                component[-1] = float('nan')
            objects.append(_node('AnimationCurve', [_long(curve_uid), _string('curve'), _string('')],
                                 [_node('KeyTime', [_array('l', times)]),
                                  _node('KeyValueFloat', [_array('f', component)])]))
            connections.append(_node('C', [_string('OP'), _long(curve_uid),
                                           _long(acn), _string('d|' + axis)]))

    objects.extend([_node('AnimationStack', [_long(90), _string('Take'), _string('')]),
                    _node('AnimationLayer', [_long(91), _string('BaseLayer'), _string('')])])
    connections.append(_node('C', [_string('OO'), _long(91), _long(90)]))
    if human_only:
        model(1, 'person', 'Null', 0)
        model(2, 'person:Hips', 'LimbNode', 1)
        if unscoped_markers:
            for index in range(4):
                model(10 + index, 'M_' + str(index + 1), 'Marker', 0)
    else:
        model(1, prop, 'Null', 0)
        properties = [_property('PreRotation', [0., 0., 10.])] if pre_rotation else []
        model(2, prop + ':Labeling:root', 'LimbNode', 1, properties)
        asset_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        offset = np.array([17., -9., 23.])
        local = points @ asset_rotation.T + offset
        translation = np.column_stack([np.linspace(2., 8., frames),
                                       np.full(frames, 3.), np.full(frames, 5.)])
        euler = np.zeros((frames, 3))
        euler[:, 2] = np.linspace(0., 65., frames)
        world = np.empty((frames, 4, 3))
        for index, angle in enumerate(euler[:, 2]):
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            world[index] = (local @ rotation.T) / (unit_scale * 10.) + translation[index]
        if drift:
            world[-1, 0, 0] += 20. / (unit_scale * 10.)
        curves(2, 'Lcl Translation', translation)
        curves(2, 'Lcl Rotation', euler)
        for marker in range(3 if missing_marker else 4):
            uid = marker + 3
            model(uid, prop + ':M_' + str(marker + 1), 'Marker', 1)
            curves(uid, 'Lcl Translation', world[:, marker])
        if meshed:
            model(10, prop + ':body', 'Mesh', 2)
    tops = [_node('GlobalSettings', children=[_node('Properties70', children=[
                _property('UnitScaleFactor', [unit_scale]), _property('UpAxis', [2])])]),
            _node('Objects', children=objects), _node('Connections', children=connections)]
    result = b'Kaydara FBX Binary  \x00\x1a\x00' + struct.pack('<I', 7700)
    for top in tops:
        result += _encode(top, len(result))
    path.write_bytes(result + bytes(25))
    return asset_rotation if not human_only else None


def _catalog(identifier='cup/cup', points=_ASYMMETRIC):
    return {identifier: {'id': identifier, 'glb': '/fixture/cup.glb',
                         'calibration_json': '/fixture/cup.json',
                         'holes': [{'id': 'Arrow' + str(index + 1), 'point_mm': point.tolist()}
                                   for index, point in enumerate(points)]}}


def _mapping(*identifiers, marker_map=None):
    result = {'white_cup': {'candidates': list(identifiers or ['cup/cup'])}}
    if marker_map is not None:
        result['white_cup']['marker_map'] = marker_map
    return result


def _asset_files(directory, stem='cup'):
    directory.mkdir(parents=True, exist_ok=True)
    glb = directory / 'cup.glb'
    glb.write_bytes(b'glTF')
    metadata = {'holes': [{'hole_id': f'Arrow{index + 1}', 'center_mm': point.tolist()}
                          for index, point in enumerate(_ASYMMETRIC)]}
    (directory / (stem + '_track_print.json')).write_text(json.dumps(metadata), encoding='utf-8')


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'take.fbx'

    def test_unscoped_markers_do_not_invent_props_in_a_human_sequence(self):
        _scene(self.source, human_only=True, unscoped_markers=True)
        plan = analyze_fbx(self.source, _catalog(), _mapping())
        self.assertEqual(plan['detected_prop_count'], 0)
        self.assertEqual(plan['matched_props'], [])
        self.assertEqual(plan['skipped_props'], [])

    def test_name_gate_never_selects_near_name_or_global_geometry_neighbor(self):
        _scene(self.source, prop='white_cups')
        plan = analyze_fbx(self.source, _catalog(), _mapping())
        self.assertEqual(plan['skipped_props'][0]['reason'], 'NO_MODEL')
        _scene(self.source)
        catalog = _catalog(points=_ASYMMETRIC * 3.)
        catalog.update(_catalog('cup/cups', _ASYMMETRIC))
        plan = analyze_fbx(self.source, catalog, _mapping())
        self.assertEqual(plan['skipped_props'][0]['reason'], 'FIT_REJECTED')
        self.assertFalse(plan['matched_props'])

    def test_units_animation_transform_evidence_and_source_immutability(self):
        for scale in (1., .1, 100.):
            with self.subTest(unit_scale=scale):
                expected_rotation = _scene(self.source, unit_scale=scale)
                before = hashlib.sha256(self.source.read_bytes()).hexdigest()
                plan = analyze_fbx(self.source, _catalog(), _mapping())
                self.assertFalse(plan['skipped_props'])
                self.assertEqual(plan['source_sha256'], before)
                self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)
                self.assertEqual(plan['native_unit_mm'], scale * 10.)
                result = plan['matched_props'][0]
                np.testing.assert_allclose(result['asset_to_bone_rotation'], expected_rotation, atol=1e-5)
                np.testing.assert_allclose(result['asset_to_bone_translation_mm'], [17., -9., 23.], atol=.002)
                # Native float32 key quantization grows with millimetres/native-unit.
                self.assertLess(result['full_sequence']['max_mm'], .005)
                self.assertEqual(result['frame_count'], 130)
                self.assertEqual(result['held_out_frame_count'], 10)
                self.assertEqual(len(result['runtime_samples']), 13)
                self.assertEqual(result['runtime_samples'][0]['source_frame_index'], 0)
                self.assertEqual(result['runtime_samples'][-1]['source_frame_index'], 129)
                self.assertIn('bone_translation_native', result['runtime_samples'][0])
                json.dumps(plan, allow_nan=False)

    def test_symmetric_orientation_requires_measured_marker_map(self):
        _scene(self.source, points=_SQUARE)
        catalog = _catalog(points=_SQUARE)
        plan = analyze_fbx(self.source, catalog, _mapping())
        self.assertEqual(plan['skipped_props'][0]['reason'], 'AMBIGUOUS_ORIENTATION')
        manual = {f'M_{index}': f'Arrow{index}' for index in range(1, 5)}
        plan = analyze_fbx(self.source, catalog, _mapping(marker_map=manual))
        self.assertEqual(len(plan['matched_props']), 1)
        self.assertLess(plan['matched_props'][0]['full_sequence']['max_mm'], .001)

    def test_near_equal_named_assets_are_not_lexicographic_winners(self):
        _scene(self.source)
        catalog = _catalog()
        catalog.update(_catalog('cup/second', _ASYMMETRIC + .001))
        plan = analyze_fbx(self.source, catalog, _mapping('cup/cup', 'cup/second'))
        self.assertEqual(plan['skipped_props'][0]['reason'], 'AMBIGUOUS_ASSET')

    def test_missing_nonfinite_or_unequally_sampled_data_is_not_interpolated(self):
        for malformed in ('missing', 'nan', 'unequal'):
            with self.subTest(malformed=malformed):
                _scene(self.source, malformed=malformed)
                plan = analyze_fbx(self.source, _catalog(), _mapping())
                self.assertEqual(plan['skipped_props'][0]['reason'], 'INVALID_CURVES')
                json.dumps(plan, allow_nan=False)
        _scene(self.source, missing_marker=True)
        plan = analyze_fbx(self.source, _catalog(), _mapping())
        self.assertEqual(plan['skipped_props'][0]['reason'], 'MISSING_MARKERS')

    def test_held_out_outlier_rejects_otherwise_good_calibration(self):
        _scene(self.source, drift=True)
        plan = analyze_fbx(self.source, _catalog(), _mapping())
        skipped = plan['skipped_props'][0]
        self.assertEqual(skipped['reason'], 'FULL_SEQUENCE_RESIDUAL')
        self.assertGreater(skipped['details']['full_sequence']['max_mm'], 19.)

    def test_existing_mesh_and_nontrivial_transform_are_explicit_skips(self):
        for options, reason in (({'meshed': True}, 'ALREADY_HAS_MESH'),
                                ({'pre_rotation': True}, 'UNSUPPORTED_TRANSFORM')):
            _scene(self.source, **options)
            plan = analyze_fbx(self.source, _catalog(), _mapping())
            self.assertEqual(plan['skipped_props'][0]['reason'], reason)

    def test_human_only_is_valid_empty_plan_with_units(self):
        _scene(self.source, human_only=True)
        plan = analyze_fbx(self.source, {}, {})
        self.assertEqual(plan['detected_prop_count'], 0)
        self.assertEqual(plan['native_unit_mm'], 10.)
        self.assertEqual(plan['matched_props'], [])
        self.assertEqual(plan['skipped_props'], [])

    def test_catalog_understands_standard_and_gooseneck_holes_not_tracking_glbs(self):
        for stem, point_key in (('cup_track_print', 'center_mm'), ('gooseneck', 'mouth_m')):
            base = stem.removesuffix('_track_print')
            metadata = {'holes': [{'hole_id' if point_key == 'center_mm' else 'marker': f'Arrow{index + 1}',
                                   point_key: (dict(zip('xyz', point)) if point_key == 'center_mm'
                                               else (point / 1000.).tolist())}
                                  for index, point in enumerate(_ASYMMETRIC)]}
            (self.root / (stem + '.json')).write_text(json.dumps(metadata), encoding='utf-8')
            (self.root / (base + '.glb')).write_bytes(b'glTF')
            (self.root / (base + '_track.glb')).write_bytes(b'glTF')
        catalog = load_catalog(self.root)
        self.assertEqual(set(catalog), {'cup', 'gooseneck'})
        for entry in catalog.values():
            np.testing.assert_allclose([hole['point_mm'] for hole in entry['holes']], _ASYMMETRIC)
            self.assertEqual(set(entry), {'id', 'glb', 'calibration_json', 'holes'})
            self.assertNotIn('_track', Path(entry['glb']).stem)

    def test_parent_library_mapping_uses_same_canonical_id_as_preparation_request(self):
        library = self.root / 'library'
        _asset_files(library / 'meeting_room' / 'cup')
        catalog = load_catalog(library)
        alias, canonical = 'cup/cup', 'meeting_room/cup/cup'
        self.assertIs(catalog[alias], catalog[canonical])
        _scene(self.source)
        plan = analyze_fbx(self.source, catalog, _mapping(alias, canonical))
        self.assertFalse(plan['skipped_props'])
        self.assertEqual([entry['asset_id'] for entry in plan['matched_props']], [canonical])
        requested = [catalog[entry['asset_id']] for entry in plan['matched_props']]
        self.assertEqual([entry['id'] for entry in requested], [canonical])

    def test_implicit_stem_aliases_do_not_create_an_ambiguous_asset(self):
        library = self.root / 'library'
        _asset_files(library / 'meeting_room' / 'cup')
        catalog = load_catalog(library)
        _scene(self.source, prop='cup')
        plan = analyze_fbx(self.source, catalog, {})
        self.assertFalse(plan['skipped_props'])
        self.assertEqual(plan['matched_props'][0]['asset_id'], 'meeting_room/cup/cup')

    def test_overlapping_library_roots_share_one_source_identity(self):
        library = self.root / 'library'
        _asset_files(library / 'meeting_room' / 'cup')
        catalog = load_catalog([library, library / 'meeting_room'])
        self.assertEqual({entry['id'] for entry in catalog.values()}, {'meeting_room/cup/cup'})
        _scene(self.source)
        plan = analyze_fbx(self.source, catalog, _mapping('cup/cup', 'meeting_room/cup/cup'))
        self.assertFalse(plan['skipped_props'])
        self.assertEqual(len(plan['matched_props']), 1)

    def test_distinct_calibrations_sharing_one_glb_are_not_deduplicated(self):
        directory = self.root / 'library' / 'cup'
        _asset_files(directory, 'cup_1')
        _asset_files(directory, 'cup_2')
        catalog = load_catalog(directory.parent)
        self.assertEqual(catalog['cup/cup_1']['glb'], catalog['cup/cup_2']['glb'])
        _scene(self.source)
        plan = analyze_fbx(self.source, catalog, _mapping('cup/cup_1', 'cup/cup_2'))
        self.assertEqual(plan['skipped_props'][0]['reason'], 'AMBIGUOUS_ASSET')

    def test_mapping_casefold_duplicate_rejection_and_no_implicit_namespace_alias(self):
        path = self.root / 'mapping.json'
        path.write_text(json.dumps({'schema_version': 1, 'props': {
            'WHITE_CUP': {'candidates': ['cup/cup']}}}), encoding='utf-8')
        mapping = load_mapping(path)
        _scene(self.source, prop='WHITE_CUP')
        self.assertEqual(len(analyze_fbx(self.source, _catalog(), mapping)['matched_props']), 1)
        _scene(self.source, prop='other:WHITE_CUP')
        self.assertEqual(analyze_fbx(self.source, _catalog(), mapping)['skipped_props'][0]['reason'], 'NO_MODEL')
        path.write_text(json.dumps({'schema_version': 1, 'props': {
            'Cup': {'candidates': ['cup/cup']},
            'cup': {'candidates': ['cup/cup']}}}), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_mapping(path)

    def test_missing_candidate_and_degenerate_calibration_are_honest_failures(self):
        _scene(self.source)
        plan = analyze_fbx(self.source, {}, _mapping())
        self.assertEqual(plan['skipped_props'][0]['reason'], 'NO_MODEL')
        self.assertEqual(plan['warnings'][0]['code'], 'MISSING_CATALOG_CANDIDATES')
        line = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
        plan = analyze_fbx(self.source, _catalog(points=line), _mapping())
        self.assertEqual(plan['skipped_props'][0]['reason'], 'INVALID_CALIBRATION')


    def test_invalid_units_and_unsupported_version_are_not_guessed(self):
        _scene(self.source, unit_scale=-1.)
        with self.assertRaisesRegex(ValueError, 'UnitScaleFactor'):
            analyze_fbx(self.source, _catalog(), _mapping())
        _scene(self.source)
        data = bytearray(self.source.read_bytes())
        struct.pack_into('<I', data, 23, 7400)
        self.source.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Unsupported FBX version'):
            analyze_fbx(self.source, _catalog(), _mapping())

    def test_duplicate_exact_stem_requires_explicit_mapping_before_geometry(self):
        _scene(self.source, prop='cup')
        catalog = _catalog('left/cup')
        catalog.update(_catalog('right/cup', _ASYMMETRIC * 3.))
        plan = analyze_fbx(self.source, catalog, {})
        self.assertEqual(plan['skipped_props'][0]['reason'], 'AMBIGUOUS_ASSET')


if __name__ == '__main__':
    unittest.main()

"""Pure NumPy regression tests; no Blender, assets, FBX mutation or animation."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np


_spec = importlib.util.spec_from_file_location(
    '_mounting_assets_unit', Path(__file__).with_name('mounting_assets.py'))
assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assets)


_VERTICES = np.array([[0., 0., 0.], [80., 0., 0.], [0., 45., 0.],
                      [0., 0., 30.], [13., 17., 29.], [-21., 11., 7.]])
_TRIANGLES = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], dtype=np.int32)
_ANGLE = .43
_ROTATION = np.array([[np.cos(_ANGLE), -np.sin(_ANGLE), 0.],
                      [np.sin(_ANGLE), np.cos(_ANGLE), 0.], [0., 0., 1.]])
_TRANSLATION = np.array([123.4, -56.7, 8.9])


def _source(name='Body', vertices=None):
    return {'name': name, 'vertices_mm': _VERTICES.copy() if vertices is None else vertices,
            'triangles': _TRIANGLES.copy()}


def _existing(source, *, unit=10., sampled=False, prop='camera_1', part_index=0):
    entry = {'name': f"Mesh_{prop}_{source['name']}_{part_index}",
             'source_name': source['name'], 'vertex_count': len(source['vertices_mm']),
             'triangle_count': len(source['triangles']),
             'triangles_sha256': hashlib.sha256(
                 np.asarray(source['triangles'], dtype='<i4').tobytes()).hexdigest()}
    target = (source['vertices_mm'] @ _ROTATION.T + _TRANSLATION) / unit
    if sampled:
        indices = np.array([5, 2, 0, 3])
        entry.update(sample_indices=indices.tolist(), sample_vertices_native=target[indices].tolist())
    else:
        entry['vertices_native'] = target.tolist()
    return entry


class RigidRegistrationTests(unittest.TestCase):
    def test_known_proper_rigid_transform_and_native_unit_conversion(self):
        source = _source()
        result = assets.register_parts([source], [_existing(source)], 10.)
        np.testing.assert_allclose(result['rotation'], _ROTATION, atol=1e-12)
        np.testing.assert_allclose(result['translation_mm'], _TRANSLATION, atol=1e-12)
        self.assertLess(result['residual_mm']['max'], 1e-11)
        self.assertEqual(result['residual_mm']['count'], len(_VERTICES))
        self.assertEqual(result['limitations'], [])
        self.assertTrue(result['parts'][0]['topology_verified'])

    def test_samples_use_explicit_original_vertex_indices_and_record_coverage(self):
        source = _source()
        result = assets.register_parts([source], [_existing(source, sampled=True)], 10.)
        np.testing.assert_allclose(result['rotation'], _ROTATION, atol=1e-12)
        np.testing.assert_allclose(result['translation_mm'], _TRANSLATION, atol=1e-12)
        self.assertEqual(result['residual_mm']['count'], 4)
        self.assertEqual(result['limitations'], ['Unsampled vertex positions are not verified'])
        self.assertFalse(result['parts'][0]['all_vertices_verified'])

    def test_all_parts_share_one_fit_even_when_existing_order_differs(self):
        sources = [_source('Camera_REFERENCE'), _source('Lens_REFERENCE', _VERTICES * .4 + 70.)]
        existing = [_existing(sources[1], part_index=1), _existing(sources[0])]
        result = assets.register_parts(sources, existing, 10.)
        np.testing.assert_allclose(result['rotation'], _ROTATION, atol=1e-12)
        self.assertEqual(result['residual_mm']['count'], 12)
        self.assertEqual([row['source_name'] for row in result['parts']],
                         ['Lens_REFERENCE', 'Camera_REFERENCE'])

    def test_changed_second_part_cannot_pass_via_first_part(self):
        sources = [_source('Camera'), _source('Lens', _VERTICES * .4 + 70.)]
        existing = [_existing(source, part_index=i) for i, source in enumerate(sources)]
        sources[1]['vertices_mm'][4, 1] += 1.
        with self.assertRaisesRegex(ValueError, 'residual'):
            assets.register_parts(sources, existing, 10.)

    def test_changed_full_vertex_geometry_and_uniform_scale_are_rejected(self):
        source = _source()
        existing = _existing(source)
        for changed in (_VERTICES * 1.01, _VERTICES + np.eye(6, 3) * .2):
            with self.subTest(changed=changed.tolist()), self.assertRaisesRegex(ValueError, 'residual'):
                assets.register_parts([_source(vertices=changed)], [existing], 10.)

    def test_reflection_is_not_accepted_as_proper_rotation(self):
        source = _source()
        existing = _existing(source)
        source['vertices_mm'][:, 0] *= -1.
        with self.assertRaisesRegex(ValueError, 'residual'):
            assets.register_parts([source], [existing], 10.)

    def test_planar_correspondences_are_valid_but_collinear_are_rejected(self):
        source = _VERTICES[:4].copy()
        source[:, 2] = 0.
        target = source @ _ROTATION.T + _TRANSLATION
        rotation, translation = assets.fit_rigid([(source, target)])
        np.testing.assert_allclose(rotation, _ROTATION, atol=1e-12)
        np.testing.assert_allclose(translation, _TRANSLATION, atol=1e-12)
        line = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
        with self.assertRaisesRegex(ValueError, 'Collinear'):
            assets.fit_rigid([(line, line)])

    def test_maximum_not_only_rms_controls_fixed_point_zero_one_mm_limit(self):
        rng = np.random.default_rng(123)
        source = _source(vertices=rng.normal(size=(1000, 3)) * 100.)
        existing = _existing(source, unit=1.)
        existing['vertices_native'][0][0] += .02
        with self.assertRaisesRegex(ValueError, 'exceeds 0.01 mm'):
            assets.register_parts([source], [existing], 1.)


class SourceIdentityTests(unittest.TestCase):
    def test_same_count_changed_topology_hash_and_explicit_triangles_rejected(self):
        for use_hash in (False, True):
            source = _source()
            existing = _existing(source)
            if not use_hash:
                existing.pop('triangles_sha256')
                existing['triangles'] = source['triangles'].tolist()
            source['triangles'][0, [1, 2]] = source['triangles'][0, [2, 1]]
            with self.subTest(use_hash=use_hash), self.assertRaisesRegex(ValueError, 'topology'):
                assets.register_parts([source], [existing], 10.)

    def test_exact_topology_array_and_legacy_generated_mesh_name(self):
        source = _source('Camera_REFERENCE_73x27x46')
        existing = _existing(source)
        existing.pop('source_name')
        existing.pop('triangles_sha256')
        existing['triangles'] = source['triangles'].tolist()
        result = assets.register_parts([source], [existing], 10.)
        self.assertTrue(result['parts'][0]['topology_verified'])
        self.assertEqual(result['limitations'], [])

    def test_missing_extra_duplicate_and_ambiguous_parts_rejected(self):
        sources = [_source('Body'), _source('Lens')]
        existing = [_existing(source, part_index=i) for i, source in enumerate(sources)]
        for changed_sources, changed_existing in (
                (sources, existing[:1]), (sources[:1], existing),
                (sources, [existing[0], copy.deepcopy(existing[0])]),
                ([_source('Body'), _source('Body')], existing)):
            with self.subTest(count=len(changed_sources)), self.assertRaises(ValueError):
                assets.register_parts(changed_sources, changed_existing, 10.)

    def test_wrong_counts_and_names_rejected_even_if_vertex_coordinates_fit(self):
        source = _source()
        for field, replacement in (('vertex_count', 7), ('triangle_count', 5),
                                   ('source_name', 'Some_other_body')):
            existing = _existing(source)
            existing[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'source geometry'):
                assets.register_parts([source], [existing], 10.)

    def test_invalid_sample_indices_and_nonfinite_coordinates_rejected(self):
        source = _source()
        for indices in ([0, 0, 2, 3], [-1, 1, 2, 3], [0, 1, 2, 6], [0., 1., 2., 3.]):
            existing = _existing(source, sampled=True)
            existing['sample_indices'] = indices
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, 'sample_indices'):
                assets.register_parts([source], [existing], 10.)
        existing = _existing(source)
        existing['vertices_native'][0][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            assets.register_parts([source], [existing], 10.)

    def test_counts_only_topology_has_explicit_limitation(self):
        source = _source()
        existing = _existing(source)
        existing.pop('triangles_sha256')
        result = assets.register_parts([source], [existing], 10.)
        self.assertFalse(result['parts'][0]['topology_verified'])
        self.assertIn('Only topology counts, not triangle indices, were supplied', result['limitations'])

    def test_invalid_unit_and_nonproper_furniture_rotation_rejected(self):
        source = _source()
        for unit in (0., -1., float('inf'), True):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                assets.register_parts([source], [_existing(source)], unit)
        with self.assertRaisesRegex(ValueError, 'proper rigid'):
            assets._rigid_transform(np.diag([-1., 1., 1.]), [0., 0., 0.])


if __name__ == '__main__':
    unittest.main()

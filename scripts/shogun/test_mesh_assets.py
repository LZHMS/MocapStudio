"""GLB scene selection, material fidelity and source protection without Blender."""
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


_spec = importlib.util.spec_from_file_location(
    '_mesh_assets_unit', Path(__file__).with_name('mesh_assets.py'))
assets = importlib.util.module_from_spec(_spec)
with patch.dict(sys.modules, {'bpy': SimpleNamespace()}):
    _spec.loader.exec_module(assets)


def _glb(path, document):
    payload = json.dumps(document).encode('utf-8')
    payload += b' ' * (-len(payload) % 4)
    path.write_bytes(struct.pack('<4sIIII', b'glTF', 2, 20 + len(payload),
                                 len(payload), 0x4E4F534A) + payload)


def _material(name, color):
    return SimpleNamespace(name=name, diffuse_color=color, use_nodes=False)


def _object(name):
    return SimpleNamespace(name=name, type='MESH', users_collection=[])


class GlbSceneTests(unittest.TestCase):
    def test_default_scene_excludes_print_layout_and_single_scene_needs_no_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'asset.glb'
            _glb(path, {'scene': 1, 'scenes': [{'name': 'Print'}, {'name': 'Assembly'}]})
            result = assets._glb_scene_selection(path)
            self.assertEqual(result['scene_index'], 1)
            self.assertEqual(result['scene_name'], 'Assembly')
            self.assertEqual(result['excluded_scenes'], [{'scene_index': 0, 'scene_name': 'Print'}])
            _glb(path, {'scenes': [{'name': 'Assembly'}]})
            self.assertEqual(assets._glb_scene_selection(path)['scene_index'], 0)

    def test_ambiguous_missing_and_invalid_default_scenes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'asset.glb'
            for document in ({'scenes': [{}, {}]}, {'scenes': []},
                             {'scene': 2, 'scenes': [{}]}, {'scene': True, 'scenes': [{}]}):
                _glb(path, document)
                with self.subTest(document=document), self.assertRaises(ValueError):
                    assets._glb_scene_selection(path)


class GlbMaterialTests(unittest.TestCase):
    def test_compaction_preserves_source_face_regions_and_actual_color_opacity(self):
        data = {'name': 'panel', 'indices': np.array([2, 0, 2], dtype=np.int32),
                'slots': [_material('dark', [0., 0., 0., .5]),
                          _material('unused', [1., 0., 0., 1.]),
                          _material('light', [1., 1., 1., 1.])]}
        assets._compact_materials(data)
        np.testing.assert_array_equal(data['indices'], [1, 0, 1])
        self.assertEqual([item['name'] for item in data['materials']], ['panel_dark', 'panel_light'])
        np.testing.assert_allclose(data['materials'][0]['diffuse_rgb'], [0., 0., 0.])
        self.assertEqual(data['materials'][0]['opacity'], .5)
        np.testing.assert_allclose(data['materials'][1]['diffuse_rgb'], [1., 1., 1.])
        self.assertEqual([item['source_slot'] for item in data['material_provenance']], [0, 2])

    def test_absent_material_uses_only_specification_default_and_bad_indices_fail(self):
        data = {'name': 'unassigned', 'indices': np.array([0], dtype=np.int32), 'slots': []}
        assets._compact_materials(data)
        self.assertEqual(data['materials'][0]['diffuse_rgb'], [1., 1., 1.])
        self.assertIn('glTF specification default', data['material_provenance'][0]['source'])
        with self.assertRaisesRegex(ValueError, 'Invalid source material index'):
            assets._compact_materials({'name': 'bad', 'indices': np.array([1]), 'slots': []})

    def test_principled_socket_color_and_alpha_override_display_color(self):
        color = SimpleNamespace(is_linked=False, default_value=[.25, .5, 1., 1.])
        alpha = SimpleNamespace(is_linked=False, default_value=.4)
        shader = SimpleNamespace(type='BSDF_PRINCIPLED',
                                 inputs={'Base Color': color, 'Alpha': alpha})
        output = SimpleNamespace(type='OUTPUT_MATERIAL', is_active_output=True,
                                 inputs={'Surface': SimpleNamespace(
                                     is_linked=True, links=[SimpleNamespace(from_node=shader)])})
        tree = SimpleNamespace(as_pointer=lambda: 1, nodes=[output, shader])
        material = SimpleNamespace(name='constant', diffuse_color=[0., 0., 0., 1.],
                                   use_nodes=True, node_tree=tree)
        result, provenance = assets._material_record(material, 'panel')
        np.testing.assert_allclose(result['diffuse_rgb'], [.53709873, .73535698, 1.])
        self.assertEqual(result['opacity'], .4)
        self.assertIn('Principled BSDF', provenance)

    def test_image_texture_in_node_group_is_rejected_not_replaced(self):
        nested = SimpleNamespace(as_pointer=lambda: 2, nodes=[SimpleNamespace(type='TEX_IMAGE')])
        tree = SimpleNamespace(as_pointer=lambda: 1,
                               nodes=[SimpleNamespace(type='GROUP', node_tree=nested)])
        material = SimpleNamespace(name='image', diffuse_color=[.5, .5, .5, 1.],
                                   use_nodes=True, node_tree=tree)
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_IMAGE_TEXTURE'):
            assets._material_record(material, 'panel')

    def test_linked_principled_color_is_rejected_not_replaced(self):
        color = SimpleNamespace(is_linked=True, default_value=[.5, .5, .5, 1.])
        shader = SimpleNamespace(type='BSDF_PRINCIPLED', inputs={'Base Color': color})
        output = SimpleNamespace(type='OUTPUT_MATERIAL', is_active_output=True,
                                 inputs={'Surface': SimpleNamespace(
                                     is_linked=True, links=[SimpleNamespace(from_node=shader)])})
        tree = SimpleNamespace(as_pointer=lambda: 1, nodes=[output, shader])
        material = SimpleNamespace(name='procedural', diffuse_color=[.5, .5, .5, 1.],
                                   use_nodes=True, node_tree=tree)
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_LINKED_COLOR'):
            assets._material_record(material, 'panel')


class GlbPreparationTests(unittest.TestCase):
    def test_preparation_keeps_selected_glb_geometry_normals_and_material_regions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / 'asset.glb'
            _glb(path, {'scene': 0, 'scenes': [{'name': 'Assembly'}, {'name': 'Print'}]})
            selected, excluded = _object('panel'), _object('print_copy')
            context = SimpleNamespace(
                view_layer=SimpleNamespace(objects=[selected]),
                scene=SimpleNamespace(objects=[selected, excluded],
                                      unit_settings=SimpleNamespace(scale_length=1.)),
                evaluated_depsgraph_get=Mock(return_value=object()))
            fake_bpy = SimpleNamespace(context=context, ops=SimpleNamespace(
                wm=SimpleNamespace(read_factory_settings=Mock()),
                import_scene=SimpleNamespace(gltf=Mock())))
            vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
            triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
            normals = np.array([[0., 0., 1.]] * 4)
            source = {'vertices': vertices, 'triangles': triangles, 'normals': normals,
                      'name': selected.name, 'indices': np.array([1, 0], dtype=np.int32),
                      'slots': [_material('black', [0., 0., 0., 1.]),
                                _material('white', [1., 1., 1., .75])],
                      'dropped': 0, 'reflected_world_transform': False}
            with patch.object(assets, 'bpy', fake_bpy), patch.object(
                    assets, '_mesh_data', return_value=source) as extract:
                result = assets._prepare_asset({'id': 'asset', 'glb': str(path)}, root, {path})
            self.assertEqual(result['status'], 'READY')
            extract.assert_called_once_with(selected, context.evaluated_depsgraph_get.return_value, 1.)
            fake_bpy.ops.import_scene.gltf.assert_called_once_with(
                filepath=str(path), import_scene_as_collection=True)
            self.assertEqual(result['audit']['excluded_glb_mesh_objects'], ['print_copy'])
            part = result['parts'][0]
            self.assertEqual(part['source_glb_object'], 'panel')
            self.assertEqual([item['name'] for item in part['materials']], ['panel_black', 'panel_white'])
            self.assertEqual(part['materials'][1]['opacity'], .75)
            with np.load(part['arrays'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['vertices_mm'], vertices)
                np.testing.assert_array_equal(data['triangles'], triangles)
                np.testing.assert_array_equal(data['normals'], normals)
                np.testing.assert_array_equal(data['triangle_material_indices'], [1, 0])
            self.assertEqual(result['audit']['parts'][0]['triangle_material_counts'], {'0': 1, '1': 1})

    def test_support_names_are_rejected_including_collection_membership(self):
        for obj in (_object('Arrow1'), SimpleNamespace(
                name='panel', users_collection=[SimpleNamespace(name='tracking_support')])):
            with self.subTest(name=obj.name):
                self.assertTrue(assets._is_support(obj))
        self.assertFalse(assets._is_support(_object('laptop_base')))

    def test_cache_cannot_overwrite_source_or_escape_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'asset.glb'
            source.write_bytes(b'original source')
            with self.assertRaisesRegex(ValueError, 'read-only source'):
                assets._write_npz(source, {'vertices_mm': np.zeros((3, 3))}, root, {source})
            with self.assertRaisesRegex(ValueError, 'escapes output_dir'):
                assets._protected_output(root.parent / 'outside.npz', root, {source})
            self.assertEqual(source.read_bytes(), b'original source')


if __name__ == '__main__':
    unittest.main()

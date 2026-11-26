"""Consumer-visible folder routing, source protection and native-unit transforms."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from step6_attach_prop_meshes import discover_inputs, _check_roots, _destinations, _build_meshes
from pipeline_layout import initialize_layout, load_layout, report_paths, take_paths
from step6_attach_prop_meshes import _discover_inputs, _mesh_layout, _write_take_reports, _arguments, run


class MeshBatchRoutingTests(unittest.TestCase):
    def test_recursive_same_basename_keeps_distinct_sequence_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input'
            paths = [source / 'take_a' / 'sequence.fbx', source / 'take_b' / 'sequence.fbx']
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_bytes(b'filename discovery does not parse the FBX')
            generated = source / 'take_a' / 'sequence_with_mesh.fbx'
            generated.write_bytes(b'prior generated output')
            found_root, found = discover_inputs(source)
            self.assertEqual(found, paths)
            destinations = [_destinations(found_root, path, root / 'output')[0] for path in found]
            self.assertEqual(destinations, [root / 'output' / 'take_a' / 'sequence_with_mesh.fbx',
                                            root / 'output' / 'take_b' / 'sequence_with_mesh.fbx'])
            with self.assertRaises(ValueError):
                discover_inputs(source, recursive=False)

    def test_explicit_input_list_never_ingests_adjacent_review_or_backup_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in ('take.fbx', 'take.full.fbx', 'review.fbx'):
                (root / name).write_bytes(b'FBX discovery fixture')
            selected = root / 'selection.json'
            selected.write_text(json.dumps(['take.fbx']), encoding='utf-8')
            _, found = discover_inputs(root, input_list=selected)
            self.assertEqual(found, [root / 'take.fbx'])

    def test_input_list_rejects_escape_duplicates_and_generated_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'input'
            source.mkdir()
            for path in (source / 'take.fbx', source / 'take_with_mesh.fbx', root / 'outside.fbx'):
                path.write_bytes(b'FBX discovery fixture')
            selected = root / 'selection.json'
            for paths in ([str(root / 'outside.fbx')], ['take.fbx', str(source / 'take.fbx')],
                          ['take_with_mesh.fbx'], []):
                selected.write_text(json.dumps(paths), encoding='utf-8')
                with self.subTest(paths=paths), self.assertRaises(ValueError):
                    discover_inputs(source, input_list=selected)

    def test_output_cannot_share_input_or_asset_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, assets = root / 'input', root / 'assets'
            for output in (source, source / 'generated', root, assets, assets / 'generated'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    _check_roots(source, output, assets)
            _check_roots(source, root / 'output', assets)

    def test_environment_option_accepts_glb_or_directory_and_rejects_missing(self):
        from step6_attach_prop_meshes import _arguments
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            glb = root / 'room.glb'
            glb.write_bytes(b'glTF')
            args = _arguments(['--input', str(root), '--output-dir', str(root / 'out'),
                              '--assets-root', str(root), '--environment', str(glb)])
            self.assertEqual(args.environment, glb)
            self.assertEqual(args.environment_parent, 'System')


class SequenceLocalMeshTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.control = self.raw / '_processing' / 'tracking'
        initialize_layout(self.control, self.raw)
        self.relative = Path('group/HOI/sequence/take.mcp')
        raw_input = self.raw / self.relative
        raw_input.parent.mkdir(parents=True)
        raw_input.write_bytes(b'untouched raw')
        self.paths = take_paths(self.control, self.relative)
        self.full = self.paths['destination'] / 'sequence.fbx'
        self.subject = self.paths['destination'] / 'subjects' / 'alice--123' / 'sequence.fbx'
        for path in (self.full, self.subject):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'FBX routing fixture')
        self.report = {
            'relative_path': self.relative.as_posix(), 'status': 'SUCCESS',
            'artifacts': {'fbx': str(self.full), 'subject_0_fbx': str(self.subject)},
            'subject_exports': [{'subject': 'alice', 'artifact_role': 'subject_0_fbx',
                                 'path': str(self.subject)}]}
        self.write_report()
        self.output = self.raw / '_processing' / 'mesh'

    def write_report(self):
        self.paths['report_path'].write_text(json.dumps(self.report), encoding='utf-8')

    def selection(self, values):
        path = self.root / 'selection.json'
        path.write_text(json.dumps([str(value) for value in values]), encoding='utf-8')
        return path

    def arguments(self, *extra):
        assets = self.root / 'assets'
        assets.mkdir(exist_ok=True)
        mapping = self.root / 'mapping.json'
        mapping.write_text('{}', encoding='utf-8')
        return _arguments(['--input', str(self.control), '--output-dir', str(self.output),
                           '--assets-root', str(assets), '--mapping', str(mapping), *extra])

    def test_canonical_discovery_includes_full_and_subject_without_old_takes_tree(self):
        # Nearby undeclared/diagnostic and neighboring-version files are not inputs.
        for path in (self.paths['destination'] / 'backup.fbx',
                     self.paths['failed_root'] / 'failed.fbx',
                     self.raw / self.relative.parent / 'processed/other/take.mcp/artifacts/other.fbx'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'not a declared FBX')
        root, found = discover_inputs(self.control, recursive=False)
        self.assertEqual(root, self.control)
        self.assertEqual(set(found), {self.full, self.subject})
        self.assertFalse((self.control / 'takes').exists())
        selected = self.selection([self.subject])
        self.assertEqual(discover_inputs(self.control, input_list=selected)[1], [self.subject])
        selected = self.selection([self.paths['destination'] / 'backup.fbx'])
        with self.assertRaises(ValueError):
            discover_inputs(self.control, input_list=selected)

    def test_declared_paths_reject_other_take_version_raw_and_control_files(self):
        other_take = take_paths(self.control, self.relative.with_name('other.mcp'))['destination'] / 'take.fbx'
        foreign = (
            other_take,
            self.raw / self.relative.parent / 'processed/other/take.mcp/artifacts/take.fbx',
            self.raw / self.relative.parent / 'raw.fbx',
            self.control / 'unassigned.fbx',
            self.paths['failed_root'] / 'failed.fbx',
        )
        for path in foreign:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'out of exact declared take artifact scope')
            self.report['artifacts']['fbx'] = str(path)
            self.write_report()
            with self.subTest(path=path), self.assertRaises(ValueError):
                discover_inputs(self.control)

    def test_report_identity_and_subject_alias_must_match(self):
        self.report['relative_path'] = 'group/HOI/sequence/other.mcp'
        self.write_report()
        with self.assertRaises(ValueError):
            discover_inputs(self.control)
        self.report['relative_path'] = self.relative.as_posix()
        self.report['subject_exports'][0]['path'] = str(self.full)
        self.write_report()
        with self.assertRaises(ValueError):
            discover_inputs(self.control)

    def test_local_selection_rejects_undeclared_duplicates_and_raw_paths(self):
        raw_fbx = self.raw / self.relative.parent / 'raw.fbx'
        raw_fbx.write_bytes(b'raw FBX not authorized by a processed declaration')
        for values in ([raw_fbx], [self.full, self.full], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                discover_inputs(self.control, input_list=self.selection(values))

    def test_declared_and_selected_symlink_paths_are_rejected(self):
        alias = self.paths['destination'] / 'alias.fbx'
        try:
            alias.symlink_to(self.full)
        except (OSError, NotImplementedError):
            self.skipTest('File symlinks unavailable')
        with self.assertRaises(ValueError):
            discover_inputs(self.control, input_list=self.selection([alias]))
        self.report['artifacts']['fbx'] = str(alias)
        self.write_report()
        with self.assertRaises(ValueError):
            discover_inputs(self.control)

    def test_local_destinations_preserve_subject_folders_and_do_not_initialize(self):
        root, inputs, scope = _discover_inputs(self.control)
        layout = _mesh_layout(root, self.output, scope)
        self.assertEqual(layout['data_root'], str(self.raw))
        self.assertEqual(layout['version_key'], self.output.name)
        paths = take_paths(self.output, self.relative, data_root=self.raw)
        for source in inputs:
            destination, report, verification = _destinations(
                root, source, self.output, scope=scope, data_root=self.raw)
            relative = source.relative_to(self.paths['destination'])
            self.assertEqual(destination, paths['destination'] / relative.parent / 'sequence_with_mesh.fbx')
            self.assertEqual(report, paths['take_root'] / '_reports' / relative.parent / 'sequence.fbx.json')
            self.assertEqual(verification, report.with_name('sequence.fbx.verification.json'))
        self.assertFalse(self.output.exists())
        self.assertFalse(paths['take_root'].exists())

    def test_output_manifest_must_match_raw_root_and_directory_version(self):
        root, _, scope = _discover_inputs(self.control)
        other_raw = self.root / 'other_raw'
        other_raw.mkdir()
        initialize_layout(self.output, other_raw)
        with self.assertRaises(ValueError):
            _mesh_layout(root, self.output, scope)
        mismatch_output = self.raw / '_processing' / 'mesh_alias'
        initialize_layout(mismatch_output, self.raw, version_key='different')
        with self.assertRaises(ValueError):
            _mesh_layout(root, mismatch_output, scope)
        same_version = self.root / 'tracking'
        with self.assertRaises(ValueError):
            _mesh_layout(root, same_version, scope)

    def test_canonical_mesh_report_aggregates_declared_full_and_subject_outputs(self):
        root, inputs, scope = _discover_inputs(self.control)
        initialize_layout(self.output, self.raw)
        files = [{'input': str(source),
                  'output': str(_destinations(root, source, self.output, scope=scope)[0]),
                  'status': 'SUCCESS'} for source in inputs]
        _write_take_reports(self.output, scope, {'files': files, 'finished_utc': 'fixture'})
        canonical = take_paths(self.output, self.relative)['report_path']
        self.assertEqual(list(report_paths(self.output)), [canonical])
        report = json.loads(canonical.read_text(encoding='utf-8'))
        self.assertEqual(report['relative_path'], self.relative.as_posix())
        self.assertEqual(report['source_report'], str(self.paths['report_path']))
        self.assertEqual(set(report['artifacts']), {'fbx', 'subject_0_fbx'})
        self.assertEqual(report['subject_exports'][0]['path'], report['artifacts']['subject_0_fbx'])
        self.assertEqual(len(report['files']), 2)
        self.assertEqual((self.raw / self.relative).read_bytes(), b'untouched raw')

    def test_dry_run_does_not_initialize_layout_or_start_blender(self):
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        args = self.arguments('--dry-run')
        before = (self.control / '.sequence-layout.json').read_bytes()
        with patch('mesh_alignment.load_catalog', return_value={'fixture': {}}), \
                patch('mesh_alignment.load_mapping', return_value={}), \
                patch('mesh_alignment.analyze_fbx', return_value={}), \
                patch('step6_attach_prop_meshes.initialize_layout') as initialize, \
                patch('step6_attach_prop_meshes._find_blender') as blender, \
                patch('mesh_fbx.load_parser') as parser, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(run(args), 0)
        initialize.assert_not_called()
        blender.assert_not_called()
        parser.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual((self.control / '.sequence-layout.json').read_bytes(), before)
        planned = take_paths(self.output, self.relative, data_root=self.raw)
        self.assertFalse(planned['take_root'].exists())

    def test_invalid_preflight_cannot_initialize_a_mesh_manifest(self):
        args = self.arguments()
        # Keep an FBX declaration, but point outside the exact processed take.
        outside = self.raw / self.relative.parent / 'outside.fbx'
        outside.write_bytes(b'unauthorized')
        self.report['artifacts']['fbx'] = str(outside)
        self.write_report()
        before = (self.control / '.sequence-layout.json').read_bytes()
        with patch('step6_attach_prop_meshes.initialize_layout') as initialize, \
                patch('step6_attach_prop_meshes._find_blender') as blender:
            with self.assertRaises(ValueError):
                run(args)
        initialize.assert_not_called()
        blender.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual((self.control / '.sequence-layout.json').read_bytes(), before)


    def test_review_artifact_subject_subfolder_keeps_raw_take_identity(self):
        review = self.paths['review_destination'] / 'subjects/alice--123/sequence.fbx'
        review.parent.mkdir(parents=True)
        self.subject.replace(review)
        self.report['artifacts']['subject_0_fbx'] = str(review)
        self.report['subject_exports'][0]['path'] = str(review)
        self.write_report()
        root, inputs, scope = _discover_inputs(self.control)
        self.assertEqual(set(inputs), {self.full, review})
        destination, _, _ = _destinations(root, review, self.output, scope=scope, data_root=self.raw)
        paths = take_paths(self.output, self.relative, data_root=self.raw)
        self.assertEqual(destination, paths['destination'] / 'subjects/alice--123/sequence_with_mesh.fbx')

    def test_declared_traversal_and_symlinked_output_folder_are_rejected(self):
        self.report['artifacts']['fbx'] = str(self.paths['destination'] / '..' / 'artifacts/sequence.fbx')
        self.write_report()
        with self.assertRaises(ValueError):
            discover_inputs(self.control)
        self.report['artifacts']['fbx'] = str(self.full)
        self.write_report()
        root, _, scope = _discover_inputs(self.control)
        paths = take_paths(self.output, self.relative, data_root=self.raw)
        paths['take_root'].mkdir(parents=True)
        foreign = self.root / 'foreign'
        foreign.mkdir()
        try:
            paths['destination'].symlink_to(foreign, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest('Directory symlinks unavailable')
        with self.assertRaises(ValueError):
            _mesh_layout(root, self.output, scope)
        self.assertIsNone(load_layout(self.output))
        self.assertEqual(list(foreign.iterdir()), [])

    def test_local_run_initializes_manifest_and_publishes_only_sequence_local_files(self):
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        import hashlib

        args = self.arguments()
        source_bytes = self.full.read_bytes()
        plan = {'matched_props': [], 'skipped_props': [], 'native_unit_mm': 1.,
                'source_sha256': hashlib.sha256(source_bytes).hexdigest()}

        verification_audit = {
            'passed': True, 'checks': {'original_animation_preserved': True},
            'source_sha256': plan['source_sha256'],
            'output_sha256': hashlib.sha256(b'generated mesh fixture').hexdigest(),
            'source_bytes': len(source_bytes), 'output_bytes': len(b'generated mesh fixture'),
            'new_object_count': 1, 'inherited_dangling_reference_count': 0}

        def publish(source, destination, meshes, *, verification_path, overwrite):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b'generated mesh fixture')
            verification_path.parent.mkdir(parents=True, exist_ok=True)
            verification_path.write_text(json.dumps(verification_audit), encoding='utf-8')
            return verification_audit

        with patch('mesh_alignment.load_catalog', return_value={'fixture': {}}), \
                patch('mesh_alignment.load_mapping', return_value={}), \
                patch('mesh_alignment.analyze_fbx', return_value=plan), \
                patch('step6_attach_prop_meshes._find_blender', return_value='unused-blender'), \
                patch('step6_attach_prop_meshes._prepare_assets'), \
                patch('step6_attach_prop_meshes._build_meshes', return_value=([{}], [], [])), \
                patch('mesh_fbx.graft_file', side_effect=publish), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(run(args), 2)
        self.assertEqual(load_layout(self.output),
                         {'schema_version': 1, 'data_root': str(self.raw), 'version_key': 'mesh'})
        paths = take_paths(self.output, self.relative)
        canonical = json.loads(paths['report_path'].read_text(encoding='utf-8'))
        expected = {
            'fbx': paths['destination'] / 'sequence_with_mesh.fbx',
            'subject_0_fbx': paths['destination'] / 'subjects/alice--123/sequence_with_mesh.fbx'}
        self.assertEqual(canonical['artifacts'], {role: str(path) for role, path in expected.items()})
        for path in expected.values():
            self.assertEqual(path.read_bytes(), b'generated mesh fixture')
        for entry in canonical['files']:
            self.assertEqual(entry['status'], 'PARTIAL')
            self.assertEqual(entry['roster_audit']['status'], 'UNVERIFIED')
            self.assertFalse(entry['roster_audit']['all_complete'])
            self.assertTrue(Path(entry['report']).is_relative_to(paths['take_root']))
            self.assertTrue(Path(entry['verification_path']).is_relative_to(paths['take_root']))
            self.assertTrue(Path(entry['report']).is_file())
            audit = json.loads(Path(entry['verification_path']).read_text(encoding='utf-8'))
            self.assertEqual(entry['verification'], verification_audit)
            self.assertEqual(audit, entry['verification'])
            file_report = json.loads(Path(entry['report']).read_text(encoding='utf-8'))
            self.assertEqual(file_report['verification_path'], entry['verification_path'])
            self.assertEqual(file_report['verification'], entry['verification'])
        self.assertTrue((self.output / 'mesh_batch_report.json').is_file())
        self.assertFalse((self.output / 'takes').exists())
        self.assertFalse((self.output / '_reports').exists())
        self.assertEqual(self.full.read_bytes(), source_bytes)
        self.assertEqual(self.subject.read_bytes(), source_bytes)
        self.assertEqual((self.raw / self.relative).read_bytes(), b'untouched raw')


class MeshBatchTransformTests(unittest.TestCase):
    def test_native_units_rotation_and_failed_asset_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / 'asset.npz'
            vertices = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]])
            normals = np.array([[0., 0., 1.]] * 3)
            np.savez_compressed(archive, vertices_mm=vertices, normals=normals,
                                triangles=np.array([[0, 1, 2]], dtype=np.int32),
                                triangle_material_indices=np.array([0], dtype=np.int32))
            prepared = {'good': {'status': 'READY', 'parts': [{
                'name': 'cup', 'arrays': str(archive),
                'materials': [{'name': 'blue', 'diffuse_rgb': [.1, .3, .8], 'opacity': 1., 'color_space': 'sRGB'}]}]},
                        'bad': {'status': 'FAILED', 'error': 'UNSUPPORTED_IMAGE_TEXTURE'}}
            rotation = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
            records = [{'prop': name, 'asset_id': asset, 'bone_uid': 42,
                        'asset_to_bone_rotation': rotation, 'asset_to_bone_translation_mm': [20., 30., 40.]}
                       for name, asset in [('cup', 'good'), ('other', 'bad')]]
            cache = {}
            meshes, attached, rejected = _build_meshes(records, prepared, cache, 10., np)
            self.assertEqual([item['prop'] for item in attached], ['cup'])
            self.assertEqual([(item['prop'], item['reason']) for item in rejected],
                             [('other', 'ASSET_PREPARATION_FAILED')])
            np.testing.assert_allclose(meshes[0]['vertices'], [[2., 3., 4.], [2., 4., 4.], [1., 3., 4.]])
            np.testing.assert_allclose(meshes[0]['normals'], normals)
            np.testing.assert_array_equal(cache['good'][0]['vertices_mm'], vertices)
            self.assertFalse(cache['good'][0]['vertices_mm'].flags.writeable)


if __name__ == '__main__':
    unittest.main()

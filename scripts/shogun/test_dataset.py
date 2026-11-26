"""Dataset layout regressions: real filesystem planning, no SDK or Shogun runs."""
from contextlib import redirect_stdout
import hashlib
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from step1_process_takes import build_jobs, dataset_root, load_manifest, main, parser


class DatasetPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / 'MocapData'
        self.data.mkdir()
        self.output = self.root / 'processed'
        self.logs = self.root / 'logs'
        self.manifest = load_manifest(None)

    def source(self, relative, paired=True):
        source = self.data / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b'planning fixture, not native data')
        if source.suffix.lower() == '.mcp' and paired:
            source.with_suffix('.x2d').write_bytes(b'camera fixture')
        return source

    def plan(self, path=None, **overrides):
        options = dict(data_path=path or self.data, output=self.output, logs=self.logs,
                       input_type='mcp', manifest=self.manifest, expected=None,
                       repair_subject=None, overwrite=False, dataset_layout=True)
        options.update(overrides)
        return build_jobs(**options)

    def dry_run(self, *arguments):
        stream = io.StringIO()
        with patch('step1_process_takes.discover_shogun', side_effect=AssertionError('native discovery forbidden')), \
                redirect_stdout(stream):
            result = main(['-DatasetLayout', '-DryRun', '-DataPath', str(self.data), *arguments])
        return result, stream.getvalue()

    def input_list(self, values):
        path = self.root / 'selected.json'
        path.write_text(json.dumps(values), encoding='utf-8')
        return path

    def test_input_list_selects_exact_seven_with_rosters_and_unchanged_dataset_identity(self):
        relatives = [f'group15/HOI/INFO_X_TEA_IN0{index}/take.mcp' for index in range(1, 6)]
        relatives += [f'group15/HOI/INFO_X_TEA_SO0{index}/take.mcp' for index in range(1, 3)]
        for relative in [*relatives, 'group15/HOI/unselected/take.mcp', 'group14/HOI/old/take.mcp']:
            self.source(relative)
        manifest = self.root / 'subjects.json'
        manifest.write_text(json.dumps({'takes': {path: ['tianyongxin'] for path in relatives}}))
        selected = self.input_list([str(self.data / relatives[0]), *relatives[1:]])
        baseline = {job['relative_path']: job for job in self.plan(manifest=load_manifest(str(manifest)))}
        with patch.object(Path, 'rglob', side_effect=AssertionError('selection must not scan')):
            jobs = self.plan(input_list=selected, manifest=load_manifest(str(manifest)), overwrite=True)
        self.assertEqual(jobs, [baseline[path] for path in relatives])
        self.assertTrue(all(job['expected_human_count'] == 1 for job in jobs))
        before = {path: path.read_bytes() for path in self.data.rglob('*') if path.is_file()}
        result, text = self.dry_run('-InputList', str(selected), '-FilterMarkers',
                                    '-SubjectManifest', str(manifest))
        self.assertEqual(result, 0)
        self.assertIn('[PLAN] inputs=7 ', text)
        self.assertIn('filter_markers=True', text)
        self.assertNotIn('unselected', text)
        self.assertNotIn('group14', text)
        self.assertEqual(before, {path: path.read_bytes() for path in self.data.rglob('*') if path.is_file()})

    def test_input_list_subgroup_scope_does_not_expand_to_dataset_root(self):
        source = self.source('group15/HOI/selected.mcp')
        outside = self.source('group14/HOI/other.mcp')
        selected = self.input_list(['HOI/selected.mcp'])
        self.assertEqual(self.plan(self.data / 'group15', input_list=selected), self.plan(source))
        for value in (str(outside), '../group14/HOI/other.mcp'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'outside DataPath'):
                self.plan(self.data / 'group15', input_list=self.input_list([value]))
        with self.assertRaisesRegex(ValueError, 'outside DataPath'):
            self.plan(source, input_list=self.input_list([str(outside)]))

    def test_input_list_rejects_invalid_duplicate_missing_and_generated_takes(self):
        source = self.source('group15/HOI/raw.mcp')
        generated = self.source('group15/HOI/processed/old/raw.vdf')
        control = self.source('_processing/control/old.vdf')
        invalid = ({}, [], 'raw.mcp', [None], [''], [str(source), str(source.parent / './raw.mcp')],
                   ['group15/HOI/missing.mcp'], ['group15/HOI/raw.x2d'],
                   [str(source.parent)], [str(generated)], [str(control)])
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.plan(input_list=self.input_list(values))
        with self.assertRaisesRegex(ValueError, 'generated output'):
            self.plan(generated, dataset_layout=False, input_list=self.input_list([str(generated)]))

    def test_input_list_exact_mcp_and_vdf_paths_override_extension_inventory(self):
        mcp = self.source('group15/HOI/raw.mcp')
        vdf = self.source('group15/HOI/existing.vdf')
        self.source('group15/HOI/adjacent.mcp')
        selected = self.input_list([str(mcp), 'group15\\HOI\\existing.vdf'])
        jobs = self.plan(input_list=selected)
        self.assertEqual({job['input_path'] for job in jobs}, {str(mcp), str(vdf)})
        self.assertEqual({job['input_kind'] for job in jobs}, {'mcp', 'vdf'})

    def test_mp_and_hoi_preserve_layout_and_set_processing_contract(self):
        self.source('group1/MP/session/take.mcp')
        self.source('group1/HOI/session/take.mcp')
        jobs = self.plan()
        self.assertEqual([job['sequence_kind'] for job in jobs], ['MP', 'HOI'])
        for job in jobs:
            relative = job['relative_path']
            self.assertEqual(Path(job['input_path']), self.data / relative)
            take_root = self.data / Path(relative).parent / 'processed' / self.output.name / Path(relative).name
            self.assertEqual(Path(job['destination']), take_root / 'artifacts')
            self.assertEqual(Path(job['review_destination']), take_root / '_needs_review/artifacts')
            self.assertEqual(Path(job['report_path']), take_root / 'report.json')
            self.assertEqual(Path(job['log_path']), take_root / 'logs/processing.log')
            self.assertEqual(Path(job['native_log_path']), take_root / 'logs/shogun.log')
            self.assertEqual(job['layout_data_root'], str(self.data))
            self.assertEqual(job['job_id'], hashlib.sha256(relative.casefold().encode()).hexdigest()[:16])
            self.assertTrue(job['dataset_mode'])
            self.assertTrue(job['split_subjects'])
            self.assertEqual(job['expected_human_count'], 2)
            self.assertEqual(job['props_enabled'], job['sequence_kind'] == 'HOI')
            self.assertIsNone(job['expected_subjects'])
            self.assertFalse(job['preflight_errors'])

    def test_subgroup_and_single_take_keep_full_dataset_relative_identity(self):
        source = self.source('group2/HOI/subgroup/odd name.mcp')
        whole = self.plan()[0]
        for selected in (self.data / 'group2', self.data / 'group2/HOI', source.parent, source):
            with self.subTest(selected=selected):
                self.assertEqual(dataset_root(selected), self.data)
                self.assertEqual(self.plan(selected), [whole])

    def test_numeric_group_then_mp_hoi_then_path_order(self):
        relatives = ['group10/MP/a.mcp', 'group2/HOI/a.mcp', 'group2/MP/z.mcp',
                     'group1/HOI/a.mcp', 'group2/MP/a.mcp']
        for relative in relatives:
            self.source(relative)
        self.assertEqual([job['relative_path'] for job in self.plan()],
                         ['group1/HOI/a.mcp', 'group2/MP/a.mcp', 'group2/MP/z.mcp',
                          'group2/HOI/a.mcp', 'group10/MP/a.mcp'])

    def test_mixed_and_unknown_categories_are_failed_not_promoted_to_hoi(self):
        relatives = ['group1/MP/good.mcp', 'group1/HOI/good.mcp',
                     'group1/Other/bad.mcp', 'group1/unclassified.mcp',
                     'unclassified.mcp', 'misc/group2/HOI/nested.mcp']
        for relative in relatives:
            self.source(relative)
        jobs = {job['relative_path']: job for job in self.plan()}
        self.assertEqual(set(jobs), set(relatives))
        for relative in relatives[2:]:
            job = jobs[relative]
            self.assertIsNone(job['sequence_kind'])
            self.assertFalse(job['props_enabled'])
            self.assertTrue(any('Unclassified DatasetLayout' in error for error in job['preflight_errors']))
        self.assertFalse(jobs[relatives[0]]['preflight_errors'])
        self.assertFalse(jobs[relatives[1]]['preflight_errors'])
        unknown_file = self.data / 'group1/Other/bad.mcp'
        self.assertTrue(self.plan(unknown_file)[0]['preflight_errors'])

    def test_nonstandard_mcp_names_are_included_but_capture_and_backups_are_not(self):
        expected = ['group1/MP/2026-04-09 10-12-35.mcp', 'group1/MP/odd.MCP']
        for relative in expected + ['group1/MP/take.mcp.capture', 'group1/MP/take.mcp.bak',
                                    'group1/MP/take.capture']:
            self.source(relative)
        (self.data / 'group1/MP/directory.mcp').mkdir()
        jobs = self.plan()
        self.assertEqual([job['relative_path'] for job in jobs], expected)
        self.assertTrue(all(not job['preflight_errors'] for job in jobs))

    def test_missing_or_empty_camera_file_is_per_job_failure(self):
        self.source('group1/MP/good.mcp')
        self.source('group1/MP/missing.mcp', paired=False)
        empty = self.source('group1/HOI/empty.mcp')
        empty.with_suffix('.x2d').write_bytes(b'')
        jobs = {Path(job['input_path']).stem: job for job in self.plan()}
        self.assertFalse(jobs['good']['preflight_errors'])
        for name in ('missing', 'empty'):
            self.assertTrue(any('X2D' in error for error in jobs[name]['preflight_errors']))
        result, _ = self.dry_run()
        self.assertEqual(result, 1)
        self.assertFalse(self.output.exists())

    def test_manifest_and_explicit_rosters_keep_full_relative_lookup(self):
        source = self.source('group2/MP/sub/take.mcp')
        self.manifest['takes']['group2/mp/sub/take.mcp'] = ['actual actor']
        self.manifest['expected_props'] = ['scene prop']
        self.assertEqual(self.plan(source)[0]['expected_subjects'], ['actual actor'])
        explicit = self.plan(source, expected=['override actor'], expected_props=['override prop'])[0]
        self.assertEqual(explicit['expected_subjects'], ['override actor'])
        self.assertEqual(explicit['expected_props'], ['override prop'])
        self.assertFalse(explicit['props_enabled'])

    def test_single_person_requires_declared_roster_not_filename(self):
        source = self.source('group1/HOI/OUT04/actor_2026-10-19_14-06-20.mcp')
        self.assertEqual(self.plan(source)[0]['expected_human_count'], 2)
        self.manifest['takes']['group1/hoi/out04/actor_2026-10-19_14-06-20.mcp'] = ['actor']
        confirmed = self.plan(source)[0]
        self.assertEqual(confirmed['expected_human_count'], 1)
        self.assertEqual(confirmed['expected_subjects'], ['actor'])

    def test_output_and_logs_cannot_write_outside_control_area_in_raw_dataset_tree(self):
        source = self.source('group1/MP/take.mcp')
        for selection in (self.data, source.parent, source):
            for output in (self.data, self.data / 'generated', self.data / 'group2/results'):
                with self.subTest(selection=selection, output=output), self.assertRaises(ValueError):
                    self.plan(selection, output=output, overwrite=True)
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.plan(selection, logs=self.data / 'logs')
        self.assertFalse((self.data / 'generated').exists())

    def test_output_under_processing_or_outside_dataset_is_allowed_without_manifest_writes(self):
        self.source('group1/MP/take.mcp')
        for output in (self.data / '_processing/custom', self.root / 'external-control'):
            with self.subTest(output=output):
                jobs = self.plan(output=output, logs=output / 'logs')
                expected = self.data / 'group1/MP/processed' / output.name / 'take.mcp'
                self.assertEqual(Path(jobs[0]['report_path']), expected / 'report.json')
                self.assertFalse(output.exists())
                self.assertFalse(expected.exists())
        with self.assertRaises(ValueError):
            self.plan(output=self.data / '_processing')

    def test_generated_trees_are_excluded_case_insensitively_from_raw_inventory(self):
        self.source('group1/MP/raw.mcp')
        generated = [
            'group1/MP/processed/old/raw.mcp/copied.mcp',
            'group1/MP/PrOcEsSeD/version/artifacts/other.mcp',
            '_processing/control/takes/old.mcp',
            'group2/HOI/_PROCESSING/control/old.mcp',
        ]
        for relative in generated:
            self.source(relative)
        self.assertEqual([job['relative_path'] for job in self.plan()], ['group1/MP/raw.mcp'])
        for relative in generated:
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.plan(self.data / relative)

    def test_same_sequence_multiple_mcp_files_keep_full_filename_identity(self):
        for relative in ('group1/HOI/sequence/a.mcp', 'group1/HOI/sequence/b.mcp'):
            self.source(relative)
        jobs = self.plan()
        self.assertEqual({Path(job['report_path']).parent.name for job in jobs}, {'a.mcp', 'b.mcp'})
        self.assertEqual(len({job['job_id'] for job in jobs}), 2)

    def test_existing_nested_subject_artifacts_and_reports_block_without_overwrite(self):
        source = self.source('group1/HOI/take.mcp')
        job = self.plan(source)[0]
        targets = [Path(job['destination']) / 'subjects/actor/take.actor.fbx',
                   Path(job['review_destination']) / 'subjects/actor/take.actor.fbx',
                   Path(job['report_path'])]
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'previous result')
        errors = '\n'.join(self.plan(source)[0]['preflight_errors'])
        for target in targets:
            self.assertIn(str(target), errors)
        self.assertFalse(self.plan(source, overwrite=True)[0]['preflight_errors'])

    def test_dataset_dry_run_defaults_to_processing_control_without_mutation(self):
        self.source('group1/MP/take.mcp')
        before = {path.relative_to(self.root): path.read_bytes()
                  for path in self.root.rglob('*') if path.is_file()}
        result, text = self.dry_run()
        self.assertEqual(result, 0)
        self.assertIn('max_workers=1', text)
        control = self.data / '_processing/processing'
        self.assertIn(str(control), text)
        self.assertIn(str(self.data / 'group1/MP/processed/processing/take.mcp/artifacts'), text)
        self.assertFalse(control.exists())
        self.assertFalse((control / '.sequence-layout.json').exists())
        after = {path.relative_to(self.root): path.read_bytes()
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(set(self.root.iterdir()), {self.data})

    def test_dry_resume_reads_local_reports_before_any_manifest_is_created(self):
        self.source('group1/MP/take.mcp')
        control = self.data / '_processing/processing'
        job = self.plan(output=control, logs=control / 'logs')[0]
        destination = Path(job['destination'])
        destination.mkdir(parents=True)
        artifacts = {}
        for role in ('vdf', 'fbx', 'vsk'):
            product = destination / ('take.' + role)
            product.write_bytes(b'existing processed result')
            artifacts[role] = str(product)
        Path(job['report_path']).write_text(json.dumps({
            'status': 'SUCCESS', 'job_id': job['job_id'], 'input_path': job['input_path'],
            'artifacts': artifacts,
        }), encoding='utf-8')
        before = {path: path.read_bytes() for path in self.data.rglob('*') if path.is_file()}
        result, text = self.dry_run('-Resume')
        self.assertEqual(result, 0)
        self.assertIn('preserved=1 pending=0', text)
        self.assertEqual(before, {path: path.read_bytes() for path in self.data.rglob('*') if path.is_file()})
        self.assertFalse(control.exists())

    def test_explicit_workers_override_dataset_default(self):
        self.source('group1/MP/take.mcp')
        result, text = self.dry_run('-MaxWorkers', '3')
        self.assertEqual(result, 0)
        self.assertIn('max_workers=3', text)
        with self.assertRaises(ValueError):
            self.dry_run('-MaxWorkers', '0')

    def test_parser_leaves_worker_default_for_mode_and_standard_jobs_unchanged(self):
        self.assertIsNone(parser().parse_args([]).MaxWorkers)
        self.assertTrue(parser().parse_args(['-DatasetLayout']).DatasetLayout)
        self.source('take.mcp')
        job = self.plan(dataset_layout=False)[0]
        self.assertEqual(job['relative_path'], 'take.mcp')
        self.assertNotIn('dataset_mode', job)
        self.assertNotIn('sequence_kind', job)

    def test_dataset_target_only_repair_is_rejected_before_native_startup(self):
        source = self.source('group1/MP/take.vdf')
        with self.assertRaisesRegex(ValueError, 'DatasetLayout cannot be combined with RepairSubject'):
            self.plan(source, input_type='vdf', repair_subject='actor')
        with patch('step1_process_takes.discover_shogun', side_effect=AssertionError('native discovery forbidden')):
            with self.assertRaisesRegex(ValueError, 'DatasetLayout cannot be combined with RepairSubject'):
                main(['-DatasetLayout', '-RepairSubject', 'actor', '-DataPath', str(source)])


if __name__ == '__main__':
    unittest.main()

"""Deterministic policy/filesystem regressions; never starts Shogun."""
from contextlib import redirect_stderr, redirect_stdout
import io
import configparser
import os
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline_layout import initialize_layout, take_paths
from step1_process_takes import build_jobs, load_manifest, publish, write_json, main
from pipeline_processing import marker_coverage, classify_subject, roster_review_reasons, process_take
from pipeline_session import _private_profile


class PipelineFilesystemTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data ! & [x] 数据'
        self.output = self.root / 'output'
        self.logs = self.root / 'logs'
        self.data.mkdir()
        self.manifest = load_manifest(None)

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, relative, x2d=True):
        source = self.data / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b'preflight fixture, not native data')
        if source.suffix == '.mcp' and x2d:
            source.with_suffix('.x2d').write_bytes(b'preflight fixture')
        return source

    def plan(self, path=None, **kwargs):
        options = dict(input_type='mcp', manifest=self.manifest, expected=None,
                       repair_subject=None, overwrite=False)
        options.update(kwargs)
        return build_jobs(path or self.data, self.output, self.logs, **options)

    def test_same_basename_takes_have_distinct_products_and_logs(self):
        self.source('a/same.mcp')
        self.source('b/same.mcp')
        jobs = self.plan()
        self.assertNotEqual(jobs[0]['destination'], jobs[1]['destination'])
        self.assertNotEqual(jobs[0]['log_path'], jobs[1]['log_path'])
        self.assertNotEqual(jobs[0]['job_id'], jobs[1]['job_id'])

    def test_missing_camera_file_does_not_hide_other_valid_take(self):
        self.source('good/take.mcp')
        self.source('bad/take.mcp', x2d=False)
        jobs = {job['relative_path']: job for job in self.plan()}
        self.assertFalse(jobs['good/take.mcp']['preflight_errors'])
        self.assertTrue(jobs['bad/take.mcp']['preflight_errors'])

    def test_current_output_tree_is_not_reingested_as_vdf_input(self):
        self.source('original.vdf')
        output = self.data / 'generated'
        output.mkdir()
        (output / 'prior.vdf').write_bytes(b'old generated data')
        jobs = build_jobs(self.data, output, self.logs, 'vdf', self.manifest, None, None, False)
        self.assertEqual([job['relative_path'] for job in jobs], ['original.vdf'])

    def test_directory_with_vdf_suffix_is_not_a_take(self):
        self.source('folder.vdf/inside.vdf')
        jobs = self.plan(input_type='vdf')
        self.assertEqual([job['relative_path'] for job in jobs], ['folder.vdf/inside.vdf'])

    def test_valid_source_hierarchy_cannot_block_another_take_output(self):
        self.source('same.mcp')
        self.source('same.vdf/other.mcp')
        reports = []
        for job in self.plan():
            stage = self.output / '.work' / job['job_id']
            stage.mkdir(parents=True)
            stem = Path(job['input_path']).stem
            artifacts = {}
            for role in ('vdf', 'fbx', 'vsk'):
                filename = stem + '.' + role
                (stage / filename).write_bytes(b'fresh native artifact fixture')
                artifacts[role] = filename
            write_json(stage / '.control' / 'result.json', {'status': 'SUCCESS', 'artifacts': artifacts})
            reports.append(publish(job, stage, self.output, False))
        self.assertEqual([report['status'] for report in reports], ['SUCCESS', 'SUCCESS'])

    def test_logs_cannot_alias_artifact_or_report_directories(self):
        self.source('take.mcp')
        with self.assertRaises(ValueError):
            build_jobs(self.data, self.output, self.output / 'takes', 'mcp',
                       self.manifest, None, None, False)

    def test_take_roster_override_wins_over_manifest_default(self):
        self.source('a/take.mcp')
        self.source('b/take.mcp')
        config = self.root / 'subjects.json'
        config.write_text(json.dumps({'expected_subjects': ['actor1'],
                                      'takes': {'a\\take.mcp': ['actor2']}}))
        manifest = load_manifest(str(config))
        rosters = {job['relative_path']: job['expected_subjects'] for job in self.plan(manifest=manifest)}
        self.assertEqual(rosters, {'a/take.mcp': ['actor2'], 'b/take.mcp': ['actor1']})

    def test_duplicate_case_insensitive_manifest_paths_are_rejected(self):
        config = self.root / 'subjects.json'
        config.write_text(json.dumps({'takes': {'A/take.mcp': ['actor1'], 'a/TAKE.mcp': ['actor2']}}))
        with self.assertRaises(ValueError):
            load_manifest(str(config))

    def test_prop_roster_override_and_explicit_no_props_remain_distinct(self):
        self.source('a/take.mcp')
        self.source('b/take.mcp')
        config = self.root / 'props.json'
        config.write_text(json.dumps({'expected_props': ['phone'],
                                      'prop_takes': {'a\\take.mcp': []}}))
        manifest = load_manifest(str(config))
        jobs = self.plan(manifest=manifest)
        self.assertEqual({job['relative_path']: job['expected_props'] for job in jobs},
                         {'a/take.mcp': [], 'b/take.mcp': ['phone']})
        override = self.plan(manifest=manifest, expected_props=['cup'])
        self.assertEqual([job['expected_props'] for job in override], [['cup'], ['cup']])

    def test_prop_roster_cannot_alias_or_escape_a_take_path(self):
        config = self.root / 'props.json'
        for mapping in ({'a/./take.mcp': ['phone'], 'A/take.mcp': ['cup']},
                        {'../outside.mcp': ['phone']}, {'take.mcp': None}):
            with self.subTest(mapping=mapping):
                config.write_text(json.dumps({'prop_takes': mapping}))
                with self.assertRaises(ValueError):
                    load_manifest(str(config))

    def test_repair_cannot_reconstruct_mcp_or_replace_source_vdf(self):
        mcp = self.source('take.mcp')
        self.assertTrue(self.plan(mcp, repair_subject='actor')[0]['preflight_errors'])
        vdf = self.source('take.vdf')
        job = build_jobs(vdf, vdf.parent, self.logs, 'vdf', self.manifest, None, 'actor', True)[0]
        self.assertTrue(job['preflight_errors'])

    def fixture_result(self, status='SUCCESS', kind='vdf'):
        source = self.source('take.' + kind)
        job = self.plan(source, input_type=kind)[0]
        stage = self.output / '.work' / 'current'
        stage.mkdir(parents=True)
        (stage / 'take.vdf').write_bytes(b'new native fixture')
        (stage / 'take.fbx').write_bytes(b'new export fixture')
        write_json(stage / '.control' / 'result.json', {'status': status, 'artifacts': {'vdf': 'take.vdf', 'fbx': 'take.fbx'}})
        return job, stage

    def test_review_result_is_not_published_as_successful_take(self):
        job, stage = self.fixture_result('NEEDS_REVIEW')
        report = publish(job, stage, self.output, False)
        self.assertEqual(report['status'], 'NEEDS_REVIEW')
        self.assertEqual(Path(report['artifacts']['vdf']).parent, Path(job['review_destination']))
        self.assertFalse((Path(job['destination']) / 'take.vdf').exists())

    def test_mcp_cannot_be_published_successfully_without_vsk(self):
        job, stage = self.fixture_result(kind='mcp')
        report = publish(job, stage, self.output, False)
        self.assertEqual(report['status'], 'FAILED')
        self.assertFalse((Path(job['destination']) / 'take.vdf').exists())

    def test_stale_external_artifact_cannot_satisfy_fresh_output_check(self):
        job, stage = self.fixture_result()
        stale = self.root / 'stale.vdf'
        stale.write_bytes(b'previous run')
        write_json(stage / '.control' / 'result.json', {'status': 'SUCCESS', 'artifacts': {'vdf': str(stale), 'fbx': 'take.fbx'}})
        report = publish(job, stage, self.output, False)
        self.assertEqual(report['status'], 'FAILED')
        self.assertTrue(stale.is_file())
        self.assertFalse((Path(job['destination']) / 'take.vdf').exists())

    def test_output_created_during_processing_is_not_overwritten(self):
        job, stage = self.fixture_result()
        destination = Path(job['destination'])
        destination.mkdir(parents=True, exist_ok=True)
        existing = destination / 'take.vdf'
        existing.write_bytes(b'valid pre-existing result')
        report = publish(job, stage, self.output, False)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(existing.read_bytes(), b'valid pre-existing result')


    def test_failed_checkpoint_paths_follow_retained_diagnostic_directory(self):
        job, stage = self.fixture_result('FAILED')
        report = publish(job, stage, self.output, False)
        checkpoint = Path(report['artifacts']['vdf'])
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(checkpoint.parent, Path(report['diagnostic_directory']))
        self.assertTrue(checkpoint.is_file())
        self.assertFalse(stage.exists())

    def local_result(self, name, status):
        source = self.source(f'group1/HOI/sequence/{name}.mcp')
        initialize_layout(self.output, self.data, 'sequence-version')
        job = self.plan(source, dataset_layout=True)[0]
        paths = take_paths(self.output, job['relative_path'])
        stage = paths['work_root'] / (job['job_id'] + '-attempt')
        stage.mkdir(parents=True)
        artifacts = {}
        for role in ('vdf', 'fbx', 'vsk'):
            filename = name + '.' + role
            (stage / filename).write_bytes(b'fresh native artifact fixture')
            artifacts[role] = filename
        write_json(stage / '.control/result.json', {
            'status': status, 'job_id': job['job_id'], 'input_path': job['input_path'],
            'relative_path': job['relative_path'], 'artifacts': artifacts,
            'logs': {'processing': job['log_path'], 'shogun': job['native_log_path']},
        })
        return job, paths, stage

    def test_local_publish_success_and_review_stay_in_exact_take_roots(self):
        for name, status, field in (('a', 'SUCCESS', 'destination'),
                                    ('b', 'NEEDS_REVIEW', 'review_destination')):
            with self.subTest(status=status):
                job, paths, stage = self.local_result(name, status)
                raw = Path(job['input_path'])
                before = raw.read_bytes(), raw.with_suffix('.x2d').read_bytes()
                report = publish(job, stage, self.output, False)
                self.assertEqual(report['status'], status)
                self.assertEqual(Path(report['artifacts']['vdf']).parent, paths[field])
                self.assertEqual(json.loads(paths['report_path'].read_bytes()), report)
                self.assertEqual(Path(report['logs']['processing']), paths['log_dir'] / 'processing.log')
                self.assertEqual(Path(report['logs']['shogun']), paths['log_dir'] / 'shogun.log')
                self.assertFalse(stage.exists())
                self.assertEqual((raw.read_bytes(), raw.with_suffix('.x2d').read_bytes()), before)
        for centralized in ('takes', '_needs_review', '.work', '_failed', 'logs'):
            self.assertFalse((self.output / centralized).exists())

    def test_local_failed_checkpoint_is_retained_in_own_diagnostic_bucket(self):
        job, paths, stage = self.local_result('failed', 'FAILED')
        report = publish(job, stage, self.output, False)
        diagnostic = Path(report['diagnostic_directory'])
        self.assertEqual(diagnostic.parent, paths['failed_root'])
        self.assertEqual(Path(report['artifacts']['vdf']).parent, diagnostic)
        self.assertTrue((diagnostic / '.control/result.json').is_file())
        self.assertEqual(json.loads(paths['report_path'].read_bytes()), report)
        self.assertFalse(stage.exists())
        self.assertFalse((self.output / '_failed').exists())

    def test_generic_publish_retains_centralized_paths_without_layout_manifest(self):
        job, stage = self.fixture_result()
        report = publish(job, stage, self.output, False)
        self.assertEqual(Path(job['report_path']), self.output / 'takes/take.vdf/report.json')
        self.assertEqual(Path(report['artifacts']['vdf']).parent, self.output / 'takes/take.vdf/artifacts')
        self.assertFalse((self.output / '.sequence-layout.json').exists())

    def test_local_resume_archives_failed_report_and_recheck_beside_take(self):
        job, paths, stage = self.local_result('retry', 'FAILED')
        publish(job, stage, self.output, False)
        failed = json.loads(paths['report_path'].read_bytes())
        sidecar = paths['report_path'].with_name('independent_recheck.json')
        sidecar.write_bytes(b'{"failed_attempt_evidence":true}')
        executable = self.root / 'installation/ShogunPostCL.exe'
        sdk = executable.parent / 'SDK/Win64/vicon_shogun_post.py'
        sdk.parent.mkdir(parents=True)
        executable.write_bytes(b'not an executable; startup is forbidden')
        sdk.write_bytes(b'not imported; SDK checking is patched')
        with patch('pipeline_session.check_sdk') as check_sdk, \
                patch('step1_process_takes.discover_shogun', side_effect=AssertionError('discovery forbidden')), \
                patch('step1_process_takes.register_run'), \
                patch('step1_process_takes.run_jobs', return_value=[]) as dispatch, \
                redirect_stdout(io.StringIO()):
            result = main(['-DatasetLayout', '-Resume', '-DataPath', str(self.data),
                           '-OutputDir', str(self.output), '-ShogunCL', str(executable)])
        self.assertEqual(result, 0)
        check_sdk.assert_called_once_with(str(executable))
        self.assertEqual([item['job_id'] for item in dispatch.call_args.args[0]], [job['job_id']])
        archives = list(paths['history_root'].iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual(json.loads((archives[0] / 'report.json').read_bytes()), failed)
        self.assertEqual((archives[0] / 'independent_recheck.json').read_bytes(), sidecar.read_bytes())
        self.assertFalse((self.output / '_resume_history').exists())


class CliExitCodeTests(unittest.TestCase):
    def test_usage_error_does_not_masquerade_as_review_exit_code(self):
        with redirect_stderr(io.StringIO()):
            result = main(['-MaxWorkers', 'not-an-integer'])
        self.assertEqual(result, 1)

    def test_nonphysical_prop_thresholds_are_rejected_before_tool_startup(self):
        for option, value in (('-PropTolerance', 'nan'), ('-PropTolerance', 'inf'),
                              ('-PropMaxStep', '0'), ('-PropMaxStep', '-1'),
                              ('-PropJitterMaxCorrection', 'nan'), ('-PropJitterMaxCorrection', 'inf'),
                              ('-PropJitterMaxCorrection', '-1'), ('-PropJitterMaxGap', '0'),
                              ('-PropJitterMaxGap', '-1'), ('-PropJitterWindow', '0')):
            with self.subTest(option=option, value=value):
                with self.assertRaises(ValueError):
                    main([option, value, '-CheckOnly'])

    def test_noninteger_jitter_window_is_a_usage_error(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(['-PropJitterWindow', '1.5', '-CheckOnly']), 1)

    def test_invalid_programmatic_jitter_policy_fails_before_native_context(self):
        for policy in ({'prop_jitter_max_correction': float('nan')},
                       {'prop_jitter_max_correction': -1.},
                       {'prop_jitter_max_gap': True}, {'prop_jitter_max_gap': 0},
                       {'prop_jitter_window': 1.5}, {'prop_jitter_window': False}):
            with self.subTest(policy=policy), patch('pipeline_processing._sdk_context') as native:
                report = process_take(SimpleNamespace(),
                                      {'input_path': 'not-loaded.vdf', 'input_kind': 'vdf'}, '.', policy)
                self.assertEqual(report['status'], 'FAILED')
                self.assertIn('ValueError', report['error'])
                native.assert_not_called()


class PropJitterProcessingTests(unittest.TestCase):
    def run_take(self, policy, *, kind='vdf', props_enabled=True, checkpoint=False, filter_failure=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / ('original.' + kind)
            source.write_bytes(b'input')
            stage = root / 'stage'
            stage.mkdir()
            character = SimpleNamespace(Name='phone', Path='phone', Type='Character', _ID=7)
            selected = []
            events = []
            scene = SimpleNamespace(
                New=lambda: None,
                Objects=SimpleNamespace(FilterByType=lambda kind: [character]),
                DeselectAllObjects=selected.clear,
                RemoveObject=lambda node: events.append(('remove', node.Name)),
                SelectObject=selected.append,
                GetSelectedObjects=lambda: list(selected))
            timeline = SimpleNamespace(GetAnimationStart=lambda: 99, GetAnimationEnd=lambda: 108,
                                       GetPlayStart=lambda: 101, GetPlayEnd=lambda: 106,
                                       SetTimeFrames=lambda frame: None)
            api = SimpleNamespace(LoadFile=lambda path: None,
                                  GetMarkerNames=lambda name: ['a', 'b', 'c', 'd'],
                                  SaveFile=lambda path: Path(path).write_bytes(b'native'))
            def hsl(command):
                events.append(('hsl', command, [node.Name for node in selected]))
                if filter_failure and 'filter 0.100 35.000;' in command:
                    raise RuntimeError('official filter failed')

            session = SimpleNamespace(api=api, hsl=hsl)
            row = dict(name='phone', bone='root', markers=['a', 'b', 'c', 'd'],
                       review_reasons=[], classification='PARTIAL')
            job = dict(job_id='take', input_path=str(source), input_kind=kind,
                       expected_subjects=None, expected_props=['phone'], props_enabled=props_enabled)
            if checkpoint:
                native = root / 'checkpoint.vdf'
                native.write_bytes(b'completed native bytes')
                job.update(checkpoint_vdf=str(native), checkpoint_audit=dict(
                    job_id='take', input_path=str(source), artifacts={'vdf': str(native)},
                    props=[row], range=dict(animation_start=99, animation_end=108,
                                           play_start=101, play_end=106),
                    stages=[dict(name=name, status='SUCCESS')
                            for name in ('fix_occlusion', 'solve_labeling', 'solve_solving')]
                    + [dict(name='filter_markers', status='SUCCESS', cutoff=0.1, threshold=35.0)]))
            coverage = marker_coverage({name: 6 for name in row['markers']}, 6)
            coverage['animation_keyed_marker_frames'] = 40

            def jitter(*args):
                events.append(('jitter',))
                return ['phone']

            with patch('pipeline_processing._sdk_context', return_value=(scene, timeline)), \
                    patch('pipeline_processing._descendants',
                          side_effect=lambda session, scene, character, kind, **kwargs:
                          [SimpleNamespace(Name='root')] if kind == 'LabelingBone' else []), \
                    patch('pipeline_processing._inspect', return_value=coverage), \
                    patch('pipeline_processing.inspect_final_quality', return_value=[]), \
                    patch('pipeline_human_reference.capture_source_labels', return_value={}), \
                    patch('pipeline_props.process_props', return_value=([row], [], [])), \
                    patch('pipeline_props.repair_prop_jitter', side_effect=jitter) as repair:
                report = process_take(session, job, stage, policy)
            if getattr(session, '_native_load_directory', None) is not None:
                session._native_load_directory.cleanup()
            if filter_failure:
                self.assertEqual(report['status'], 'FAILED')
            else:
                self.assertNotEqual(report['status'], 'FAILED', report.get('error'))
            if checkpoint:
                self.assertEqual((stage / 'checkpoint.vdf').read_bytes(), native.read_bytes())
            return report, events, repair.call_count

    def test_vdf_corrected_prop_is_solved_even_without_measured_reassociation(self):
        report, events, calls = self.run_take({'prop_jitter_max_correction': 5.})
        self.assertEqual(calls, 1)
        solves = [event for event in events if event[0] == 'hsl' and 'solvelabelingonly' in event[1]]
        self.assertEqual(len(solves), 1)
        self.assertEqual(solves[0][2], ['phone'])
        self.assertIn('-selected', solves[0][1])
        self.assertLess(events.index(('jitter',)), events.index(solves[0]))
        self.assertEqual(report['policy']['prop_jitter_max_correction'], 5.)

    def test_mcp_corrects_after_occlusion_and_before_native_solving(self):
        _, events, calls = self.run_take({'prop_jitter_max_correction': 5.}, kind='mcp')
        self.assertEqual(calls, 1)
        occlusion = next(index for index, event in enumerate(events)
                         if event[0] == 'hsl' and 'fixocclusiononly' in event[1])
        solve = next(index for index, event in enumerate(events)
                     if event[0] == 'hsl' and 'solvelabelingonly' in event[1])
        self.assertLess(occlusion, events.index(('jitter',)))
        self.assertLess(events.index(('jitter',)), solve)

    def test_explicit_official_filter_runs_after_fill_before_jitter_and_solve(self):
        report, events, calls = self.run_take(
            {'filter_markers': True, 'prop_jitter_max_correction': 5.}, kind='mcp')
        self.assertEqual(calls, 1)
        occlusion = next(index for index, event in enumerate(events)
                         if event[0] == 'hsl' and 'fixocclusiononly' in event[1])
        filtered = next(index for index, event in enumerate(events)
                        if event[0] == 'hsl' and 'filter 0.100 35.000;' in event[1])
        solve = next(index for index, event in enumerate(events)
                     if event[0] == 'hsl' and 'solvelabelingonly' in event[1])
        self.assertLess(occlusion, filtered)
        self.assertLess(filtered, events.index(('jitter',)))
        self.assertLess(events.index(('jitter',)), solve)
        self.assertEqual(events[filtered][1],
                         'selectByType Marker; selectRange -all; selectProperty Translation; filter 0.100 35.000;')
        stage = next(item for item in report['stages'] if item['name'] == 'filter_markers')
        self.assertEqual(stage['status'], 'SUCCESS')
        self.assertEqual((stage['cutoff'], stage['threshold']), (0.1, 35.0))
        self.assertEqual(stage['parameters_source'], 'Scripts/DefaultOps/FilterData.hsl')
        self.assertTrue(report['policy']['filter_markers'])
        self.assertTrue(report['policy']['filter_markers_executed'])
        self.assertEqual(report['policy']['marker_filter_parameters']['channels'], 'AllTranslations')

    def test_filter_is_opt_in_and_never_refilters_vdf_or_completed_checkpoint(self):
        for policy, extra in (({}, {'kind': 'mcp'}),
                              ({'filter_markers': False}, {'kind': 'mcp'}),
                              ({'filter_markers': True}, {}),
                              ({'filter_markers': True}, {'kind': 'mcp', 'checkpoint': True})):
            with self.subTest(policy=policy, extra=extra):
                report, events, _ = self.run_take(policy, **extra)
                self.assertFalse(any(event[0] == 'hsl' and 'filter 0.100 35.000;' in event[1]
                                     for event in events))
                stage = next(item for item in report['stages'] if item['name'] == 'filter_markers')
                self.assertEqual(stage['status'], 'SKIPPED')
                self.assertFalse(report['policy']['filter_markers_executed'])
                if extra.get('checkpoint'):
                    source = report['processing_source']
                    self.assertIn('filter_markers', source['completed_steps_reused'])
                    self.assertEqual(source['filter_markers_evidence_reused'][0]['cutoff'], 0.1)
                    self.assertTrue(source['native_bytes_reused'])

    def test_filter_failure_records_failed_stage_and_never_continues_to_jitter_or_solve(self):
        report, events, calls = self.run_take(
            {'filter_markers': True, 'prop_jitter_max_correction': 5.}, kind='mcp', filter_failure=True)
        self.assertEqual(calls, 0)
        stage = next(item for item in report['stages'] if item['name'] == 'filter_markers')
        self.assertEqual(stage['status'], 'FAILED')
        self.assertIn('official filter failed', report['error'])
        self.assertFalse(report['policy']['filter_markers_executed'])
        self.assertFalse(any(event[0] == 'hsl' and 'solvelabelingonly' in event[1] for event in events))
        self.assertEqual(report['native_diagnostics']['failed_calls'][0]['api'], 'HSL.filter_markers')

    def test_disabled_old_policy_mp_legacy_and_completed_checkpoint_do_not_mutate_props(self):
        for policy, extra in (({}, {}), ({'prop_jitter_max_correction': 0.}, {}),
                              ({'prop_jitter_max_correction': 5.}, {'props_enabled': False}),
                              ({'prop_jitter_max_correction': 5., 'repair_subject': 'actor'}, {}),
                              ({'prop_jitter_max_correction': 5.}, {'checkpoint': True})):
            with self.subTest(policy=policy, extra=extra):
                report, events, calls = self.run_take(policy, **extra)
                self.assertEqual(calls, 0)
                stage = next(item for item in report['stages'] if item['name'] == 'repair_prop_jitter')
                self.assertEqual(stage['status'], 'SKIPPED')
                if extra.get('props_enabled') is False:
                    # Human-only datasets: prop subjects are removed at configure
                    # so they never reach the exported FBX.
                    self.assertIn(('remove', 'phone'), events)
                if extra.get('checkpoint'):
                    self.assertTrue(report['processing_source']['native_bytes_reused'])


class BindingPolicyTests(unittest.TestCase):
    def test_zero_data_requires_real_setup_before_repair_classification(self):
        coverage = marker_coverage({'head': 0, 'pelvis': 0}, 100)
        self.assertEqual(classify_subject(2, 5, 5, coverage), 'ZERO_DATA')
        self.assertEqual(classify_subject(2, 0, 5, coverage), 'MISSING_SETUP')

    def test_partial_trajectory_is_not_classified_as_whole_subject_loss(self):
        coverage = marker_coverage({'head': 100, 'pelvis': 1}, 100)
        self.assertEqual(classify_subject(2, 5, 5, coverage), 'PARTIAL')

    def test_coverage_threshold_is_inclusive(self):
        coverage = marker_coverage({'head': 80, 'pelvis': 80}, 100)
        self.assertEqual(classify_subject(2, 5, 5, coverage, 0.8), 'PRESENT')
        self.assertEqual(classify_subject(2, 5, 5, coverage, 0.8001), 'PARTIAL')

    def test_missing_core_marker_cannot_be_hidden_by_other_dense_tracks(self):
        coverage = marker_coverage({'head': 100, 'pelvis': 100}, 100, ['head', 'pelvis', 'ankle'])
        self.assertEqual(coverage['core_coverage'], 2 / 3)
        self.assertEqual(coverage['unconfigured_core_markers'], ['ankle'])
        self.assertEqual(classify_subject(2, 5, 5, coverage, 0.5), 'PARTIAL')

    def test_configured_core_set_does_not_let_absent_finger_tracks_fail_body_gate(self):
        coverage = marker_coverage({'head': 100, 'pelvis': 100, 'finger': 0}, 100, ['head', 'pelvis'])
        self.assertEqual(classify_subject(3, 5, 5, coverage), 'PRESENT')

    def test_frame_count_outside_processing_range_is_rejected(self):
        with self.assertRaises(ValueError):
            marker_coverage({'head': 101}, 100)
        with self.assertRaises(ValueError):
            marker_coverage({'head': -1}, 100)


class RosterPolicyTests(unittest.TestCase):
    def test_unknown_roster_cannot_claim_every_actor_was_checked(self):
        self.assertEqual(roster_review_reasons(['actor1'], None, None), ['expected_roster_unknown'])

    def test_unexpected_human_is_not_silently_ignored_by_a_full_roster(self):
        self.assertEqual(roster_review_reasons(['actor1', 'actor2'], ['actor1'], None), ['unexpected_human:actor2'])

    def test_targeted_repair_does_not_assert_other_actors_are_absent(self):
        self.assertEqual(roster_review_reasons(['actor1', 'actor2'], None, 'actor1'), [])


@unittest.skipUnless(os.name == 'nt', 'Windows profile protection')
class SessionProfileTests(unittest.TestCase):
    def test_autosave_is_disabled_without_changing_source_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.ini'
            original = '[MainFrame]\nAutoSave=1\nSavePath=E:/original_capture\n'
            source.write_text(original, encoding='utf-8')
            work = root / 'private'
            work.mkdir()
            private = _private_profile(root / 'ShogunPost1.21' / 'ShogunPostCL.exe', '1.21', work, str(source))
            settings = configparser.ConfigParser()
            settings.read(private, encoding='utf-8')
            self.assertEqual(settings['MainFrame']['AutoSave'], '0')
            self.assertEqual(source.read_text(encoding='utf-8'), original)



class ResumeCliTests(unittest.TestCase):
    def test_resume_and_overwrite_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "Use Resume or Overwrite, not both"):
            main(['-Resume', '-Overwrite', '-CheckOnly'])

    def test_resume_dryrun_reports_preserved_without_starting_native(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "t1.vdf").write_bytes(b"content")
            (data / "t2.vdf").write_bytes(b"content")
            output = root / "output"
            # Setup t1 as completed
            art_dir = output / "takes" / "t1.vdf" / "artifacts"
            art_dir.mkdir(parents=True)
            (art_dir / "t1.vdf").write_bytes(b"vdf_data")
            (art_dir / "t1.fbx").write_bytes(b"fbx_data")
            rep_file = output / "takes" / "t1.vdf" / "report.json"
            from step1_process_takes import build_jobs, load_manifest
            jobs = build_jobs(data, output, output / "logs", "vdf", load_manifest(None), None, None, True)
            j1 = next(j for j in jobs if j["relative_path"] == "t1.vdf")
            rep_file.write_text(json.dumps({
                "status": "SUCCESS", "job_id": j1["job_id"], "input_path": j1["input_path"],
                "artifacts": {"vdf": str(art_dir / "t1.vdf"), "fbx": str(art_dir / "t1.fbx")}
            }), encoding="utf-8")
            out_buf = io.StringIO()
            with redirect_stderr(out_buf), patch('sys.stdout', out_buf):
                code = main(['-DataPath', str(data), '-OutputDir', str(output), '-InputType', 'vdf', '-Resume', '-DryRun'])
            self.assertEqual(code, 0)
            output_text = out_buf.getvalue()
            self.assertIn("[RESUME] total=2 preserved=1 pending=1", output_text)

if __name__ == '__main__':
    unittest.main()

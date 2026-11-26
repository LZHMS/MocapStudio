"""Real-filesystem recovery regressions; never connects to or starts Shogun."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from step1_process_takes import publish, write_json
from pipeline_layout import initialize_layout, report_paths, take_paths
from step4_recheck_quality import (
    recover_quality_transition,
    recheck_report,
    refresh_batch_reports,
    recovery_source,
    transition_failure,
)


TRANSITION_ERRORS = (
    "TypeError: validate_prop_poses() missing 1 required positional argument: 'options'",
    'TypeError: validate_prop_poses() takes 3 positional arguments but 4 were given',
)
COMPLETED_STAGES = ('fix_occlusion', 'solve_labeling', 'solve_solving')
CONNECTION_ERROR = 'ConnectionError: simulated lost Shogun connection'


class RecheckRecoveryTests(unittest.TestCase):
    local_layout = False

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / 'output 数据'
        self.relative = ('group1/HOI/sequence/take.mcp' if self.local_layout
                         else 'group/take.mcp')
        source = self.root / 'input' / self.relative
        source.parent.mkdir(parents=True)
        source.write_bytes(b'original input fixture; not native data')
        if self.local_layout:
            initialize_layout(self.output, self.root / 'input', 'current')
        self.paths = take_paths(self.output, self.relative)
        self.report_path = self.paths['report_path']
        job = {
            'job_id': 'original-take-job',
            'input_path': str(source),
            'input_kind': 'mcp',
            'relative_path': self.relative,
            'expected_subjects': ['actor1'],
            'expected_props': [],
            'report_path': str(self.report_path),
            'destination': str(self.paths['destination']),
            'review_destination': str(self.paths['review_destination']),
        }
        stage = self.paths['work_root'] / 'original'
        stage.mkdir(parents=True)
        (stage / 'take.vdf').write_bytes(b'completed fill and solve checkpoint\x00\xff')
        original = {
            **{key: job[key] for key in ('job_id', 'input_path', 'input_kind',
                                         'relative_path', 'expected_subjects', 'expected_props')},
            'schema_version': 1,
            'status': 'FAILED',
            'error': TRANSITION_ERRORS[0],
            'artifacts': {'vdf': 'take.vdf'},
            'subjects': [],
            'props': [],
            'stages': [{'name': name, 'status': 'SUCCESS'} for name in COMPLETED_STAGES],
            'policy': {'min_coverage': 1.0},
            'logs': {},
        }
        write_json(stage / '.control' / 'result.json', original)
        # Exercise the actual publisher: the archived control report keeps a
        # relative VDF, while the consumer-facing report has an absolute VDF.
        self.previous = publish(job, stage, self.output, False)
        self.checkpoint = Path(self.previous['artifacts']['vdf'])
        self.control_path = self.checkpoint.parent / '.control' / 'result.json'

    def overwrite_with_connection_failure(self):
        overwritten = copy.deepcopy(self.previous)
        overwritten.update(error=CONNECTION_ERROR, artifacts={}, stages=[])
        overwritten['quality_before_recheck'] = {
            'status': self.previous['status'],
            'error': self.previous['error'],
            'resumed_completed_checkpoint': str(self.checkpoint),
            'subjects': [],
        }
        write_json(self.report_path, overwritten)
        return overwritten

    def assert_checkpoint_source(self, previous):
        before = copy.deepcopy(previous)
        recovered, checkpoint = recovery_source(previous, self.output)
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(recovered['job_id'], self.previous['job_id'])
        self.assertEqual(recovered['input_path'], self.previous['input_path'])
        self.assertEqual(recovered['error'], previous.get('quality_before_recheck', {}).get(
            'error', previous['error']))
        completed = {row['name'] for row in recovered['stages'] if row['status'] == 'SUCCESS'}
        self.assertTrue(set(COMPLETED_STAGES) <= completed)
        self.assertEqual(previous, before, 'Resolving a checkpoint must not mutate the current report')
        return recovered

    def test_old_transition_failure_uses_its_completed_checkpoint(self):
        for error in TRANSITION_ERRORS:
            with self.subTest(error=error):
                previous = copy.deepcopy(self.previous)
                previous['error'] = error
                self.assertTrue(transition_failure(previous))
                self.assert_checkpoint_source(previous)

    def test_connection_overwrite_recovers_archived_report_and_absolute_checkpoint(self):
        previous = self.overwrite_with_connection_failure()
        before = {path: path.read_bytes() for path in
                  (self.report_path, self.checkpoint, self.control_path)}
        archived = json.loads(self.control_path.read_text(encoding='utf-8'))
        self.assertEqual(archived['artifacts']['vdf'], 'take.vdf')
        self.assertEqual(previous['artifacts'], {})
        self.assertTrue(transition_failure(previous))
        recovered = self.assert_checkpoint_source(previous)
        self.assertEqual(recovered['stages'], archived['stages'])
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_connection_failure_without_transition_audit_is_not_recoverable(self):
        previous = copy.deepcopy(self.previous)
        previous.update(error=CONNECTION_ERROR, artifacts={}, stages=[])
        self.assertFalse(transition_failure(previous))
        with self.assertRaises(ValueError):
            recovery_source(previous, self.output)

    def test_archived_report_for_another_job_or_sequence_is_rejected(self):
        previous = self.overwrite_with_connection_failure()
        original = json.loads(self.control_path.read_text(encoding='utf-8'))
        for field, value in (('job_id', 'different-sequence-job'),
                             ('input_path', str(self.root / 'input' / 'another.mcp')),
                             ('relative_path', 'group1/HOI/sequence/another.mcp')):
            with self.subTest(field=field):
                archived = copy.deepcopy(original)
                archived[field] = value
                write_json(self.control_path, archived)
                with self.assertRaises(ValueError):
                    recovery_source(previous, self.output)
        write_json(self.control_path, original)

    def test_every_required_stage_needs_success_evidence(self):
        archived_original = json.loads(self.control_path.read_text(encoding='utf-8'))
        for overwritten in (False, True):
            for name in COMPLETED_STAGES:
                for status in (None, 'SKIPPED', 'FAILED'):
                    with self.subTest(overwritten=overwritten, stage=name, status=status):
                        source = copy.deepcopy(archived_original if overwritten else self.previous)
                        source['stages'] = [row for row in source['stages'] if row['name'] != name]
                        if status is not None:
                            source['stages'].append({'name': name, 'status': status})
                        if overwritten:
                            previous = self.overwrite_with_connection_failure()
                            write_json(self.control_path, source)
                        else:
                            previous = source
                        with self.assertRaises(ValueError):
                            recovery_source(previous, self.output)
        write_json(self.control_path, archived_original)

    def test_checkpoint_outside_this_outputs_failed_tree_is_rejected(self):
        outside = self.root / 'unrelated-output' / '_failed' / 'take.vdf'
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b'checkpoint belonging to a different output')
        for overwritten in (False, True):
            with self.subTest(overwritten=overwritten):
                if overwritten:
                    previous = self.overwrite_with_connection_failure()
                    previous['quality_before_recheck']['resumed_completed_checkpoint'] = str(outside)
                else:
                    previous = copy.deepcopy(self.previous)
                    previous['artifacts']['vdf'] = str(outside)
                with self.assertRaises(ValueError):
                    recovery_source(previous, self.output)

    def test_missing_archived_report_does_not_fall_back_to_empty_failure(self):
        previous = self.overwrite_with_connection_failure()
        self.control_path.unlink()
        with self.assertRaises(ValueError):
            recovery_source(previous, self.output)
        self.assertEqual(json.loads(self.report_path.read_text(encoding='utf-8')), previous)
        self.assertTrue(self.checkpoint.is_file())

    def test_incomplete_checkpoint_evidence_cannot_bypass_processing(self):
        from pipeline_processing import process_take
        stage = self.paths['work_root'] / 'unprocessed-checkpoint'
        stage.mkdir(parents=True)
        job = {key: self.previous[key] for key in
               ('job_id', 'input_path', 'input_kind', 'expected_subjects')}
        audit = copy.deepcopy(self.previous)
        audit['stages'] = [row for row in audit['stages'] if row['name'] != 'solve_solving']
        job.update(checkpoint_vdf=str(self.checkpoint), checkpoint_audit=audit)
        original = self.checkpoint.read_bytes()
        result = process_take(object(), job, stage, {})
        self.assertEqual(result['status'], 'FAILED')
        self.assertFalse(result['quality']['usable'])
        self.assertEqual(self.checkpoint.read_bytes(), original)

    def test_checkpoint_from_a_different_job_cannot_bypass_processing(self):
        from pipeline_processing import process_take
        stage = self.paths['work_root'] / 'wrong-checkpoint'
        stage.mkdir(parents=True)
        job = {key: self.previous[key] for key in
               ('job_id', 'input_path', 'input_kind', 'expected_subjects')}
        audit = copy.deepcopy(self.previous)
        audit['job_id'] = 'some-other-job'
        job.update(checkpoint_vdf=str(self.checkpoint), checkpoint_audit=audit)
        original = self.checkpoint.read_bytes()
        result = process_take(object(), job, stage, {})
        self.assertEqual(result['status'], 'FAILED')
        self.assertFalse(result['quality']['usable'])
        self.assertEqual(self.checkpoint.read_bytes(), original)

    def assert_failed_attempt_is_atomic(self, overwritten, raises):
        previous = (self.overwrite_with_connection_failure() if overwritten
                    else copy.deepcopy(self.previous))
        paths = (self.report_path, self.checkpoint, self.control_path)
        before = {path: path.read_bytes() for path in paths}
        existing_json = set(self.paths['failed_root'].rglob('*.json'))
        session = object()
        calls = []

        def disconnect(actual_session, job, stage, policy):
            # Only replace the SDK boundary. Recovery, file reads, diagnosis and
            # any publication all run as real production code against temp files.
            self.assertIs(actual_session, session)
            self.assertEqual(Path(job['checkpoint_vdf']), self.checkpoint.resolve())
            self.assertEqual(job['job_id'], previous['job_id'])
            self.assertEqual(job['input_path'], previous['input_path'])
            self.assertEqual(policy, self.previous['policy'])
            self.assertTrue(Path(stage).is_dir())
            self.assertEqual(Path(stage).parent, self.paths['work_root'])
            self.assertEqual(Path(job['destination']), self.paths['destination'])
            self.assertEqual(Path(job['review_destination']), self.paths['review_destination'])
            calls.append(Path(stage))
            (Path(stage) / 'processing.log').write_text(CONNECTION_ERROR, encoding='utf-8')
            if raises:
                raise ConnectionError('simulated lost Shogun connection')
            return {
                'status': 'FAILED', 'error': CONNECTION_ERROR, 'artifacts': {},
                'subjects': [], 'props': [], 'stages': [],
                'quality': {'schema_version': 2, 'basis': 'post_processing', 'usable': False},
            }

        error_type = ConnectionError if raises else RuntimeError
        with patch('pipeline_processing.process_take', side_effect=disconnect):
            with self.assertRaisesRegex(error_type, 'simulated lost Shogun connection'):
                recover_quality_transition(session, self.report_path, self.output)
        self.assertEqual(len(calls), 1)
        for path, snapshot in before.items():
            self.assertEqual(path.read_bytes(), snapshot, f'Original evidence overwritten: {path}')
        # Check the actual downstream view, not merely a mocked publish call.
        current = json.loads(self.report_path.read_text(encoding='utf-8'))
        recovered, checkpoint = recovery_source(current, self.output)
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(recovered['job_id'], previous['job_id'])
        self.assertEqual(recovered['error'], TRANSITION_ERRORS[0])
        attempt_results = [path for path in self.paths['failed_root'].rglob('result.json')
                           if path not in existing_json and path.parent.name == '.control']
        self.assertEqual(len(attempt_results), 1, 'Failure must retain a separate diagnostic result')
        self.assertTrue(attempt_results[0].is_relative_to(self.paths['failed_root']))
        diagnostic = json.loads(attempt_results[0].read_text(encoding='utf-8'))
        self.assertEqual(diagnostic['status'], 'FAILED')
        self.assertIn('simulated lost Shogun connection', diagnostic['error'])
        self.assertTrue((attempt_results[0].parent.parent / 'processing.log').is_file())

    def test_sdk_connection_exception_preserves_original_report_and_checkpoint(self):
        self.assert_failed_attempt_is_atomic(overwritten=False, raises=True)

    def test_failed_processing_result_preserves_original_report_and_checkpoint(self):
        self.assert_failed_attempt_is_atomic(overwritten=False, raises=False)

    def test_connection_exception_preserves_checkpoint_audit_after_prior_overwrite(self):
        self.assert_failed_attempt_is_atomic(overwritten=True, raises=True)

    def test_failed_result_preserves_checkpoint_audit_after_prior_overwrite(self):
        self.assert_failed_attempt_is_atomic(overwritten=True, raises=False)


class LocalRecheckRecoveryTests(RecheckRecoveryTests):
    local_layout = True

    def test_artifact_traversal_cannot_cross_into_sibling_take(self):
        sibling = take_paths(self.output, 'group1/HOI/sequence/second.mcp')
        artifact = sibling['destination'] / 'take.vdf'
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b'foreign native file')
        traversal = self.paths['destination'] / '..' / '..' / 'second.mcp' / 'artifacts' / 'take.vdf'
        write_json(self.report_path, self.completed_report(traversal))
        with patch('step4_recheck_quality._sdk_context') as sdk:
            with self.assertRaisesRegex(ValueError, 'Invalid completed artifact'):
                recheck_report(object(), self.report_path, self.output)
        sdk.assert_not_called()

    def other_locations(self):
        sibling = take_paths(self.output, 'group1/HOI/sequence/second.mcp')
        alternate = self.root / 'alternate-control'
        initialize_layout(alternate, self.root / 'input', 'older')
        older = take_paths(alternate, self.relative)
        return sibling, older

    def test_checkpoint_from_sibling_take_or_version_is_rejected_even_with_matching_job(self):
        for paths in self.other_locations():
            checkpoint = paths['failed_root'] / 'same-job' / 'take.vdf'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(self.checkpoint.read_bytes())
            original = json.loads(self.control_path.read_text(encoding='utf-8'))
            write_json(checkpoint.parent / '.control' / 'result.json', original)
            for overwritten in (False, True):
                with self.subTest(root=paths['take_root'], overwritten=overwritten):
                    previous = (self.overwrite_with_connection_failure() if overwritten
                                else copy.deepcopy(self.previous))
                    if overwritten:
                        previous['quality_before_recheck']['resumed_completed_checkpoint'] = str(checkpoint)
                    else:
                        previous['artifacts']['vdf'] = str(checkpoint)
                    with self.assertRaises(ValueError):
                        recovery_source(previous, self.output)

    def test_archived_checkpoint_must_reference_exact_native_file(self):
        previous = self.overwrite_with_connection_failure()
        original = json.loads(self.control_path.read_text(encoding='utf-8'))
        original['artifacts']['vdf'] = 'different.vdf'
        (self.checkpoint.parent / 'different.vdf').write_bytes(b'different checkpoint')
        write_json(self.control_path, original)
        with self.assertRaisesRegex(ValueError, 'different native artifact'):
            recovery_source(previous, self.output)

    def test_canonical_report_identity_is_checked_before_recovery_processing(self):
        wrong = copy.deepcopy(self.previous)
        wrong['relative_path'] = 'group1/HOI/sequence/second.mcp'
        write_json(self.report_path, wrong)
        with patch('pipeline_processing.process_take') as process:
            with self.assertRaisesRegex(ValueError, 'raw relative identity'):
                recover_quality_transition(object(), self.report_path, self.output)
        process.assert_not_called()

    def completed_report(self, artifact):
        result = copy.deepcopy(self.previous)
        result.update(status='NEEDS_REVIEW', artifacts={'vdf': str(artifact)},
                      review_reasons=[], subjects=[], props=[], limits={},
                      range={key: 1 for key in ('play_start', 'play_end', 'animation_start', 'animation_end')})
        result.pop('error', None)
        return result

    def test_completed_artifact_cannot_belong_to_another_take_or_version(self):
        for paths in self.other_locations():
            artifact = paths['destination'] / 'take.vdf'
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b'foreign native file')
            write_json(self.report_path, self.completed_report(artifact))
            with self.subTest(root=paths['take_root']):
                with patch('step4_recheck_quality._sdk_context') as sdk:
                    with self.assertRaisesRegex(ValueError, 'Invalid completed artifact'):
                        recheck_report(object(), self.report_path, self.output)
                sdk.assert_not_called()

    def test_completed_recheck_moves_only_own_artifact_and_preserves_source_sha(self):
        artifact = self.paths['review_destination'] / 'take.vdf'
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b'unchanged completed native bytes')
        sibling, older = self.other_locations()
        foreign = []
        for paths in (sibling, older):
            path = paths['review_destination'] / 'take.vdf'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'untouched foreign artifact')
            foreign.append(path)
        report = self.completed_report(artifact)
        write_json(self.report_path, report)
        snapshot = self.report_path.read_bytes()
        raw = Path(report['input_path']).read_bytes()
        scene = SimpleNamespace(New=lambda: None,
                                Objects=SimpleNamespace(FilterByType=lambda kind: []))
        with patch('step4_recheck_quality._sdk_context', return_value=(scene, object())), \
                patch('step4_recheck_quality._ranges', return_value=report['range']), \
                patch('step4_recheck_quality.load_native_file'), \
                patch('pipeline_human_reference.capture_source_labels', return_value={}), \
                patch('step4_recheck_quality.inspect_final_quality', return_value=[]), \
                patch('step4_recheck_quality.update_subject_export_quality'):
            result = recheck_report(object(), self.report_path, self.output)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(result['quality_before_recheck']['report_sha256'],
                         hashlib.sha256(snapshot).hexdigest())
        destination = self.paths['destination'] / 'take.vdf'
        self.assertEqual(Path(result['artifacts']['vdf']), destination)
        self.assertEqual(destination.read_bytes(), b'unchanged completed native bytes')
        self.assertFalse(artifact.exists())
        self.assertEqual(Path(report['input_path']).read_bytes(), raw)
        for path in foreign:
            self.assertEqual(path.read_bytes(), b'untouched foreign artifact')

    def test_batch_refresh_reads_local_reports_and_rejects_wrong_source_identity(self):
        sibling, older = self.other_locations()
        sibling_row = copy.deepcopy(self.previous)
        sibling_row.update(relative_path='group1/HOI/sequence/second.mcp',
                           input_path=str(self.root / 'input' / 'group1/HOI/sequence/second.mcp'))
        write_json(sibling['report_path'], sibling_row)
        foreign_row = copy.deepcopy(self.previous)
        foreign_row.update(status='NEEDS_REVIEW')
        write_json(older['report_path'], foreign_row)
        changed = copy.deepcopy(self.previous)
        changed.update(status='SUCCESS')
        write_json(self.report_path, changed)
        jobs = [copy.deepcopy(self.previous), dict(sibling_row, input_path='wrong-source.mcp')]
        batch = {'run_id': 'local-run', 'jobs': jobs}
        for name in ('batch-local-run.json', 'batch_report.json'):
            write_json(self.output / name, batch)
        refresh_batch_reports(self.output, 'local-run')
        for name in ('batch-local-run.json', 'batch_report.json'):
            actual = json.loads((self.output / name).read_text(encoding='utf-8'))
            self.assertEqual(actual['jobs'][0], changed)
            self.assertEqual(actual['jobs'][1], jobs[1])
            self.assertEqual(actual['counts'], {'SUCCESS': 1, 'NEEDS_REVIEW': 0, 'FAILED': 1})

    def test_canonical_scan_excludes_other_versions_and_diagnostic_reports(self):
        from step4_recheck_quality import main
        sibling, older = self.other_locations()
        sibling_row = dict(self.previous, relative_path='group1/HOI/sequence/second.mcp')
        write_json(sibling['report_path'], sibling_row)
        write_json(older['report_path'], self.previous)
        write_json(self.paths['destination'] / 'report.json', self.previous)
        write_json(self.paths['recheck_root'] / 'diagnostic' / 'report.json', self.previous)
        expected = {self.report_path, sibling['report_path']}
        self.assertEqual(set(report_paths(self.output)), expected)

        def completed(path, output, executable, recovery):
            return {'relative_path': json.loads(path.read_bytes())['relative_path'],
                    'status': 'SUCCESS', 'usable': True, 'worker_pid': 1,
                    'native_pid': 2, 'diagnostic_directory': 'mocked-process-boundary'}

        with patch('sys.argv', ['recheck', '-OutputDir', str(self.output), '-ShogunCL', 'unused']), \
                patch('step4_recheck_quality.register_run'), \
                patch('step4_recheck_quality.time.sleep'), \
                patch('recheck_worker.run_isolated_recheck', side_effect=completed) as worker:
            self.assertEqual(main(), 0)
        self.assertEqual({call.args[0] for call in worker.call_args_list}, expected)
        self.assertEqual(worker.call_count, 2)

    def test_worker_start_failures_keep_attempts_in_exact_take_and_version(self):
        from recheck_worker import run_isolated_recheck
        sibling, older = self.other_locations()
        alternate = self.root / 'alternate-control'
        locations = [(self.output, self.paths, self.relative),
                     (self.output, sibling, 'group1/HOI/sequence/second.mcp'),
                     (alternate, older, self.relative)]
        runtime = self.root / 'runtime'
        runtime.mkdir()
        (runtime / 'pythonw.exe').write_bytes(b'unused interpreter fixture')
        for output, paths, relative in locations:
            row = dict(self.previous, relative_path=relative)
            write_json(paths['report_path'], row)
            with self.subTest(output=output, relative=relative):
                with patch('recheck_worker.sys.executable', str(runtime / 'python.exe')), \
                        patch('recheck_worker.mp.set_executable'), \
                        patch('recheck_worker.mp.get_context', side_effect=RuntimeError('simulated unavailable worker')):
                    with self.assertRaisesRegex(RuntimeError, 'simulated unavailable worker'):
                        run_isolated_recheck(paths['report_path'], output, 'unused', False)
                attempts = list(paths['recheck_root'].iterdir())
                self.assertEqual(len(attempts), 1)
                state = json.loads((attempts[0] / 'attempt.json').read_text(encoding='utf-8'))
                self.assertEqual(state['phase'], 'failed')
                self.assertEqual(state['report'], str(paths['report_path']))
                self.assertTrue(state['scratch_removed'])
                self.assertTrue((attempts[0] / 'result.json').is_file())
        self.assertFalse((self.output / '.quality_recheck_attempts').exists())


if __name__ == '__main__':
    unittest.main()

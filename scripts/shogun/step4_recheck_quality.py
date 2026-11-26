"""Re-evaluate saved VDFs in canonical take-local or generic output storage."""
from __future__ import annotations

import argparse
import copy
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from step1_process_takes import write_json
from pipeline_processing import (
    _descendants, _ranges, _sdk_context, inspect_final_quality,
    update_subject_export_quality,
)
from pipeline_rigid import RigidOptions
from pipeline_session import load_native_file
from pipeline_run_registry import register_run
from pipeline_layout import artifact_roots, in_run, report_paths, report_relative, take_paths


def recheck_report(session, report_path: Path, output: Path) -> dict:
    relative = report_relative(output, report_path)
    snapshot = report_path.read_bytes()
    report = json.loads(snapshot)
    if report.get('status') not in ('SUCCESS', 'NEEDS_REVIEW'):
        raise ValueError('Only completed native results can be rechecked')
    if Path(report.get('relative_path', '')) != relative:
        raise ValueError('Canonical report location differs from raw relative identity')
    roots = artifact_roots(output, relative)
    artifacts = {}
    for role, value in report['artifacts'].items():
        path = Path(value).absolute()
        if (not in_run(output, path) or not path.is_file()
                or not any(path.is_relative_to(root) and path.resolve().is_relative_to(root)
                           for root in roots)):
            raise ValueError(f'Invalid completed artifact: {path}')
        artifacts[role] = path.resolve()
    if 'vdf' not in artifacts:
        raise ValueError('Completed native VDF is required')
    from pipeline_human_reference import capture_source_labels
    references = {}
    load_audit = report.setdefault('native_loads', [])
    scene, _ = _sdk_context()
    original = Path(report.get('input_path', ''))
    if original.suffix.lower() == '.mcp' and original.is_file():
        scene.New()
        load_native_file(session, original, audit=load_audit)
        references = capture_source_labels(session)
    scene.New()
    load_native_file(session, artifacts['vdf'], audit=load_audit)
    scene, timeline = _sdk_context()
    ranges = _ranges(timeline)
    saved_range = report['range']
    if any(ranges[key] != saved_range[key] for key in ('play_start', 'play_end', 'animation_start', 'animation_end')):
        raise ValueError('Saved VDF range differs from its report')
    report['quality_before_recheck'] = {
        'status': report['status'], 'review_reasons': list(report.get('review_reasons', [])),
        'report_sha256': hashlib.sha256(snapshot).hexdigest(),
    }
    previous_human_reasons = {f"{row['name']}:{reason}" for row in report['subjects']
                             for reason in row.get('review_reasons', [])}
    report['review_reasons'] = [reason for reason in report.get('review_reasons', [])
                                if reason not in previous_human_reasons and not reason.startswith('prop:')]
    characters = {node.Name: node for node in scene.Objects.FilterByType('Character')}
    detected = {}
    for row in report['subjects']:
        row['before_fill'] = copy.deepcopy(row.get('after') or row.get('before'))
        row['classification_before_fill'] = row.get('classification')
        row['repair_warnings'] = [reason for reason in row.get('review_reasons', []) if reason == 'repair_no_data_gain']
        row['review_reasons'] = []
        row['quality'] = {'basis': 'post_processing', 'usable': False}
        if not row.get('processing_target'):
            continue
        name = row['name']
        if name not in characters:
            row['classification'] = 'MISSING_SETUP'
            row['review_reasons'] = ['subject_absent']
            report['review_reasons'].append(f'missing_setup:{name}')
            continue
        character = characters[name]
        markers = list(session.api.GetMarkerNames(name))
        labeling = _descendants(session, scene, character, 'LabelingBone')
        solving = _descendants(session, scene, character, 'SolvingBone')
        if (len(markers), len(labeling), len(solving)) != tuple(row['setup'][key] for key in
                                                              ('markers', 'labeling_bones', 'solving_bones')):
            raise ValueError(f'Saved VDF setup differs from report: {name}')
        detected[name] = (character, markers, labeling, solving, row)
    policy = report.get('policy', {})
    options = RigidOptions(tolerance_mm=float(policy.get('prop_tolerance', 2.0)),
                           max_step_mm=float(policy.get('prop_max_step', 200.0)))
    bg_set = policy.get('background_props')
    context_reasons = inspect_final_quality(
        session, report, detected, ranges, float(policy.get('min_coverage', 1.0)),
        policy.get('core_markers') or {}, options,
        not policy.get('repair_subject') and report.get('props_enabled', True),
        bg_set, references)
    update_subject_export_quality(report, context_reasons)
    report['review_reasons'] = list(dict.fromkeys(report['review_reasons']))
    report['status'] = 'NEEDS_REVIEW' if report['review_reasons'] else 'SUCCESS'
    report['quality'] = {'schema_version': 2, 'human_geometry_version': 1,
                         'basis': 'post_processing', 'usable': report['status'] == 'SUCCESS'}
    report['limits']['coverage_basis'] = 'post-processing SDK marker existence in the requested play range'
    report['quality_rechecked_at'] = datetime.now(timezone.utc).isoformat()
    if report_path.read_bytes() != snapshot:
        raise RuntimeError('Report changed during recheck; no update published')
    target_root = roots[0] if report['status'] == 'SUCCESS' else roots[1]
    moved = []
    try:
        for role, source in artifacts.items():
            source_root = next(root for root in roots if source.is_relative_to(root))
            target = target_root / source.relative_to(source_root)
            if not in_run(output, target):
                raise ValueError(f'Recheck destination escapes this run: {target}')
            if source != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, target)
                moved.append((source, target))
            report['artifacts'][role] = str(target)
        for entry in report.get('subject_exports', []):
            entry['path'] = report['artifacts'][entry['artifact_role']]
        write_json(report_path, report)
    except BaseException:
        for source, target in reversed(moved):
            os.rename(target, source)
        raise
    # Remove only empty source containers after an accepted relocation.
    for root in roots:
        if root == target_root or not root.exists():
            continue
        for path in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if not any(path.iterdir()):
                path.rmdir()
        if not any(root.iterdir()):
            root.rmdir()
    return report


_TRANSITION_ERRORS = {
    "TypeError: validate_prop_poses() missing 1 required positional argument: 'options'",
    'TypeError: validate_prop_poses() takes 3 positional arguments but 4 were given',
}


def recovery_source(previous: dict, output: Path) -> tuple[dict, Path]:
    """Recover the original solved checkpoint, including failed recheck attempts."""
    output = output.resolve()
    failed_root = take_paths(output, previous['relative_path'])['failed_root']

    def checkpoint_path(value):
        path = Path(value).absolute()
        if (not in_run(output, path) or not path.is_relative_to(failed_root)
                or not path.resolve().is_relative_to(failed_root)):
            raise ValueError('Checkpoint escapes this take failed tree')
        return path.resolve()

    if previous.get('error') in _TRANSITION_ERRORS:
        original = copy.deepcopy(previous)
        value = previous.get('artifacts', {}).get('vdf')
    else:
        value = previous.get('quality_before_recheck', {}).get('resumed_completed_checkpoint')
        if not value:
            raise ValueError('No original completed checkpoint evidence')
        checkpoint = checkpoint_path(value)
        control = checkpoint.parent / '.control' / 'result.json'
        if not in_run(output, control):
            raise ValueError('Checkpoint control report escapes this run')
        try:
            original = json.loads(control.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError(f'Cannot read original checkpoint control report: {control}') from error
        if original.get('error') not in _TRANSITION_ERRORS:
            raise ValueError('Control report is not the original transition failure')
    if not value:
        raise ValueError('Native checkpoint is missing')
    checkpoint = checkpoint_path(value)
    if not checkpoint.is_file():
        raise ValueError('Recovery requires a native checkpoint inside this take failed tree')
    if any(original.get(key) != previous.get(key)
           for key in ('job_id', 'input_path', 'relative_path')):
        raise ValueError('Checkpoint belongs to a different job or take')
    archived_value = original.get('artifacts', {}).get('vdf')
    if not archived_value:
        raise ValueError('Original checkpoint report has no native artifact')
    archived_path = Path(archived_value)
    if not archived_path.is_absolute():
        archived_path = checkpoint.parent / archived_path
    if checkpoint_path(archived_path) != checkpoint:
        raise ValueError('Original checkpoint report references a different native artifact')
    completed = {entry['name'] for entry in original['stages'] if entry['status'] == 'SUCCESS'}
    if not {'fix_occlusion', 'solve_labeling', 'solve_solving'} <= completed:
        raise ValueError('Checkpoint is not known to have completed fill and solve')
    original['artifacts']['vdf'] = str(checkpoint)
    return original, checkpoint


def recover_quality_transition(session, report_path: Path, output: Path) -> dict:
    """Resume a solved checkpoint; failed attempts never overwrite its evidence."""
    import uuid
    from step1_process_takes import publish
    from pipeline_processing import process_take
    relative = report_relative(output, report_path)
    snapshot = report_path.read_bytes()
    current = json.loads(snapshot)
    if Path(current.get('relative_path', '')) != relative:
        raise ValueError('Canonical report location differs from raw relative identity')
    previous, checkpoint = recovery_source(current, output)
    paths = take_paths(output, relative)
    job = {key: previous[key] for key in
           ('job_id', 'input_path', 'input_kind', 'relative_path', 'expected_subjects')}
    job['expected_props'] = previous.get('expected_props')
    for key in ('dataset_mode', 'sequence_kind', 'split_subjects', 'props_enabled', 'expected_human_count'):
        if key in previous:
            job[key] = previous[key]
    job.update(checkpoint_vdf=str(checkpoint), checkpoint_audit=previous, report_path=str(report_path),
               destination=str(paths['destination']),
               review_destination=str(paths['review_destination']))
    stage = paths['work_root'] / ('quality-recovery-' + uuid.uuid4().hex)
    for key in ('work_root', 'failed_root', 'destination', 'review_destination'):
        if not in_run(output, paths[key]):
            raise ValueError(f'Recovery destination escapes this run: {paths[key]}')
    stage.mkdir(parents=True)
    result = None
    try:
        result = process_take(session, job, stage, previous['policy'])
        if result['status'] not in ('SUCCESS', 'NEEDS_REVIEW'):
            raise RuntimeError(result.get('error', 'Checkpoint recovery failed'))
        if report_path.read_bytes() != snapshot:
            raise RuntimeError('Original report changed during checkpoint recovery')
    except BaseException as error:
        attempt = result if result is not None else {
            'status': 'FAILED', 'error': f'{type(error).__name__}: {error}', 'artifacts': {}}
        write_json(stage / '.control' / 'result.json', attempt)
        diagnostic = paths['failed_root'] / stage.name
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, diagnostic)
        raise
    for key in ('schema_version', 'job_id', 'input_path', 'input_kind', 'relative_path',
                'expected_subjects', 'expected_props', 'dataset_mode', 'sequence_kind',
                'split_subjects', 'props_enabled', 'expected_human_count', 'policy', 'logs'):
        if key in previous:
            result[key] = previous[key]
    result['quality_before_recheck'] = {
        'status': previous['status'], 'error': previous['error'],
        'previous_attempt_error': current.get('error'),
        'resumed_completed_checkpoint': str(checkpoint),
        'subjects': [{'name': row['name'], 'before': row.get('before'),
                      'classification': row.get('classification')}
                     for row in previous.get('subjects', [])],
    }
    write_json(stage / '.control' / 'result.json', result)
    return publish(job, stage, output, False)


def transition_failure(report: dict) -> bool:
    if report.get('status') != 'FAILED':
        return False
    audit = report.get('quality_before_recheck', {})
    return (report.get('error') in _TRANSITION_ERRORS
            or (str(report.get('error', '')).startswith('ConnectionError:')
                and audit.get('error') in _TRANSITION_ERRORS
                and bool(audit.get('resumed_completed_checkpoint'))))


class ProcessWait:
    """Hold the original Windows process handle; never follow a reused PID."""
    def __init__(self, pid):
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.OpenProcess(0x100000, False, pid)
        if not self.handle and ctypes.get_last_error() != 87:
            raise ctypes.WinError(ctypes.get_last_error())
    def alive(self):
        if not self.handle:
            return False
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 258
    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def refresh_batch_reports(output, run_id):
    for path in (output / f'batch-{run_id}.json', output / 'batch_report.json'):
        if not path.is_file():
            continue
        batch = json.loads(path.read_text(encoding='utf-8'))
        if batch.get('run_id') != run_id:
            continue
        for index, row in enumerate(batch['jobs']):
            relative = row.get('relative_path')
            if relative:
                current = take_paths(output, relative)['report_path']
                if current.is_file():
                    replacement = json.loads(current.read_text(encoding='utf-8'))
                    if (replacement.get('job_id') == row.get('job_id')
                            and Path(replacement.get('relative_path', '')) == report_relative(output, current)
                            and replacement.get('input_path') == row.get('input_path')):
                        batch['jobs'][index] = replacement
        batch['counts'] = {status: sum(row['status'] == status for row in batch['jobs'])
                           for status in ('SUCCESS', 'NEEDS_REVIEW', 'FAILED')}
        batch['quality_reconciled_at'] = datetime.now(timezone.utc).isoformat()
        write_json(path, batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-OutputDir', required=True)
    parser.add_argument('-ShogunCL', required=True)
    parser.add_argument('-LegacyWorkers', nargs='*', type=int, default=[])
    parser.add_argument('-FollowRun', type=int)
    args = parser.parse_args()
    output = Path(args.OutputDir).resolve()
    waits = [ProcessWait(pid) for pid in args.LegacyWorkers]
    parent = ProcessWait(args.FollowRun) if args.FollowRun else None
    lock = json.loads((output / '.pipeline.lock').read_text()) if parent else None
    if lock and lock['pid'] != args.FollowRun:
        raise RuntimeError('FollowRun does not own the output lock')
    attempted = set()
    summary = {'basis': 'post_processing', 'isolation': 'per_sequence_process',
               'rechecked': [], 'errors': [], 'finished': False}
    register_run(output, kind="quality_recheck", source_output=output,
                 label=f"Independent quality recheck: {output.name}")
    try:
        stable = 0
        while True:
            pending = []
            for path in sorted(report_paths(output)):
                data = path.read_bytes()
                row = json.loads(data)
                if Path(row.get('relative_path', '')) != report_relative(output, path):
                    raise ValueError(f'Canonical report location differs from raw relative identity: {path}')
                completed = row.get('status') in ('SUCCESS', 'NEEDS_REVIEW')
                if not transition_failure(row) and (not completed or row.get('quality', {}).get('human_geometry_version') == 1):
                    continue
                identity = (str(path), hashlib.sha256(data).hexdigest())
                if identity not in attempted:
                    pending.append((path, identity, transition_failure(row)))
            pending.sort(key=lambda item: not item[2])
            from recheck_worker import run_isolated_recheck
            for path, identity, recovery in pending:
                attempted.add(identity)
                try:
                    result = run_isolated_recheck(path, output, args.ShogunCL, recovery)
                    summary['rechecked'].append({
                        'relative_path': result['relative_path'], 'status': result['status'],
                        'usable': result['usable'], 'checkpoint_recovery': recovery,
                        'worker_pid': result['worker_pid'], 'native_pid': result['native_pid'],
                        'diagnostic_directory': result['diagnostic_directory']})
                    print(f"[RECHECKED] {result['status']} {result['relative_path']}", flush=True)
                except Exception as error:
                    summary['errors'].append({'report': str(path), 'error': f'{type(error).__name__}: {error}'})
                    print(f'[ERROR] {path}: {error}', flush=True)
                write_json(output / 'quality_recheck_report.json', summary)
            if not any(wait.alive() for wait in waits):
                stable += 1
                if stable >= 2:
                    break
            time.sleep(5)
        summary['finished'] = True
        write_json(output / 'quality_recheck_report.json', summary)
        # The already-running scheduler has old in-memory result dictionaries.
        # Reconcile its final batch indexes only after it releases its run.
        if parent:
            while parent.alive():
                time.sleep(5)
            refresh_batch_reports(output, lock['run_id'])
        return 1 if summary['errors'] else 0
    finally:
        for wait in waits:
            wait.close()
        if parent:
            parent.close()


if __name__ == '__main__':
    raise SystemExit(main())

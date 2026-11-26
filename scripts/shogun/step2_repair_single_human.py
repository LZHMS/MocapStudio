"""Isolated, source-preserving human postprocessing with audited candidate rejection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from pipeline_human import inspect_human_geometry, read_calibration
from pipeline_processing import _SOLVE_OPTIONS, _descendants, _quote, _ranges, _sdk_context, _select, _bone_metrics
from pipeline_props import _trajectory
from pipeline_session import load_native_file
from pipeline_run_registry import register_run


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def repair_existing(session, source, output, subject, reference_mcp=None):
    """Native candidate -> solve -> save -> reload -> geometry/preservation checks.

    Rejection never overwrites either source or the baseline. Candidate VDF is
    retained as diagnostic evidence; only a fully verified result gets an FBX.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.suffix.lower() != '.vdf' or not source.is_file():
        raise ValueError('An existing completed VDF is required; MCP/X2D are never edited')
    if output.exists():
        raise ValueError('Output directory must not exist (no published artifacts overwritten)')
    if source.is_relative_to(output):
        raise ValueError('Output cannot contain the source')
    output.mkdir(parents=True)
    fingerprint = _sha256(source)
    baseline = output / 'baseline.vdf'
    shutil.copy2(source, baseline)
    audit = dict(status='FAILED', source=str(source), source_sha256=fingerprint,
                 subject=subject, source_unchanged=False, accepted=False,
                 artifacts={'baseline': str(baseline)}, review_reasons=[])
    audit['quality_scope'] = dict(subject=subject, human_geometry='target_only',
                                 other_humans_validated=False, props_validated=False)
    reference = None
    reference_path = Path(reference_mcp).resolve() if reference_mcp else None
    if reference_path and (not reference_path.is_file() or reference_path.suffix.lower() != '.mcp'):
        raise ValueError('--reference-mcp must name an existing original MCP')
    reference_hash = _sha256(reference_path) if reference_path else None
    try:
        scene, timeline = _sdk_context()
        if reference_path:
            from pipeline_human_reference import capture_source_labels
            scene.New()
            load_native_file(session, reference_path, audit=audit.setdefault('native_loads', []))
            reference = capture_source_labels(session).get(subject)
            audit['reference'] = dict(path=str(reference_path), sha256=reference_hash,
                                      available=reference is not None)
        scene.New()
        load_native_file(session, baseline, audit=audit.setdefault('native_loads', []))
        scene, timeline = _sdk_context()
        ranges = _ranges(timeline)
        audit['range'] = ranges
        full_ranges = dict(ranges, play_start=ranges['animation_start'], play_end=ranges['animation_end'],
                           frame_count=ranges['animation_frame_count'], slice_start=0,
                           slice_stop=ranges['animation_frame_count'])
        characters = {node.Name: node for node in scene.Objects.FilterByType('Character')}
        if subject not in characters:
            raise ValueError(f'Human Character not found: {subject}')
        character = characters[subject]
        labeling = _descendants(session, scene, character, 'LabelingBone')
        solving = _descendants(session, scene, character, 'SolvingBone')
        if not labeling or not solving:
            raise ValueError('Both labeling and solving human skeletons required')
        all_points, metadata = {}, {}
        for name, node in characters.items():
            for marker in session.api.GetMarkerNames(name):
                all_points[name, marker] = _trajectory(session.api, name, marker, full_ranges)
                target = scene.Objects[node.Path + '\\' + marker]
                if target is None:
                    raise RuntimeError(f'Marker channel node missing: {name}/{marker}')
                for field in ('Radius', 'ReconIndex', 'FillType'):
                    metadata[name, marker, field] = list(target.Channels[field][ranges['animation_start']:ranges['animation_end'] + 1])
        markers = list(session.api.GetMarkerNames(subject))
        groups, _ = read_calibration(session.api, subject, markers)
        before = inspect_human_geometry(session.api, subject, markers, ranges, labeling, solving, reference)
        audit['before'] = before
        changed = {}
        unassigned_expected = {}
        if reference:
            from pipeline_human_repair import apply_reference_relabel
            reference_audit, unassigned_expected = apply_reference_relabel(
                session, scene, characters, subject, reference, ranges)
            audit['reference_relabel'] = reference_audit
            for correction in reference_audit['corrections']:
                if correction.get('marker'):
                    changed[subject, correction['marker'], correction['frame']] = correction
        timeline.SetTimeFrames(ranges['play_start'])
        from pipeline_human_reference import compare_source_labels
        after_reference_points = {marker: _trajectory(session.api, subject, marker, ranges) for marker in markers}
        audit['after_reference_before_solve'] = {
            'source_label_consistency': compare_source_labels(after_reference_points, reference, ranges),
            'first_frame_positions': {marker: values[0].tolist() for marker, values in after_reference_points.items()
                                      if any(key[1] == marker for key in changed)}}
        # Even without safe coordinate edits a genuine targeted re-solve is
        # attempted. It can fix stale solves, but cannot excuse marker geometry.
        audit['solve_attempted'] = True
        _select(scene, character)
        session.hsl(_SOLVE_OPTIONS + 'quickPost "solvelabelingonly" -selected;')
        _select(scene, character)
        session.hsl(_SOLVE_OPTIONS + 'solve -selectedCharacters;')
        retargeting = _descendants(session, scene, character, 'RetargetingBone')
        if retargeting:
            _select(scene, character)
            session.hsl('retarget -selectedCharacters;')
        scene.DeselectAllObjects()
        candidate = output / 'candidate.vdf'
        session.api.SaveFile(candidate.as_posix())
        audit['artifacts']['candidate'] = str(candidate)
        scene.New()
        load_native_file(session, candidate, audit=audit.setdefault('native_loads', []))
        scene, timeline = _sdk_context()
        if _ranges(timeline) != ranges:
            raise RuntimeError('Saved candidate range changed after native reload')
        reloaded_characters = {node.Name: node for node in scene.Objects.FilterByType('Character')}
        if set(reloaded_characters) != set(characters):
            raise RuntimeError('Saved candidate changed the Character roster')
        from pipeline_human_repair import verify_reference_preservation
        preservation = []
        for (name, marker), original in all_points.items():
            actual = _trajectory(session.api, name, marker, full_ranges)
            allowed = np.zeros(len(original), dtype=bool)
            for frame in range(ranges['play_start'], ranges['play_end'] + 1):
                correction = changed.get((name, marker, frame))
                if correction:
                    index = frame - ranges['animation_start']
                    allowed[index] = True
                    proposed = np.asarray(correction['corrected_xyz'])
                    # SDK stream uses six significant digits plus float32 storage.
                    magnitude = np.maximum(np.abs(proposed), np.finfo(float).tiny)
                    precision = .5 * 10. ** (np.floor(np.log10(magnitude)) - 5) + np.abs(np.spacing(proposed.astype(np.float32))) + 1e-9
                    if np.any(np.abs(actual[index] - proposed) > precision):
                        preservation.append(f'corrected_coordinate_readback_mismatch:{name}:{marker}:{frame}')
            if not np.array_equal(actual[~allowed], original[~allowed], equal_nan=True):
                preservation.append(f'untouched_marker_changed:{name}:{marker}')
            target = scene.Objects[reloaded_characters[name].Path + '\\' + marker]
            for field in ('Radius', 'ReconIndex', 'FillType'):
                values = list(target.Channels[field][ranges['animation_start']:ranges['animation_end'] + 1])
                expected = list(metadata[name, marker, field])
                for (target_subject, target_marker, frame), correction in changed.items():
                    if target_subject == name and target_marker == marker and correction.get('metadata_written'):
                        expected[frame - ranges['animation_start']] = correction['metadata_written'][field]
                    if target_subject == name and target_marker == marker and correction.get('source_metadata'):
                        expected[frame - ranges['animation_start']] = correction['source_metadata'][field]
                present = np.isfinite(original).all(axis=1) | allowed
                if not np.array_equal(np.asarray(values)[present], np.asarray(expected)[present], equal_nan=True):
                    preservation.append(f'marker_metadata_changed:{name}:{marker}:{field}')
        preservation.extend(verify_reference_preservation(scene, unassigned_expected, ranges))
        character = reloaded_characters[subject]
        labeling = _descendants(session, scene, character, 'LabelingBone')
        solving = _descendants(session, scene, character, 'SolvingBone')
        actual_groups, actual_errors = read_calibration(session.api, subject, markers)
        same_calibration = (set(actual_groups) == set(groups) and not actual_errors and
                            all(set(actual_groups[bone]) == set(offsets) and
                                all(np.array_equal(actual_groups[bone][name], xyz)
                                    for name, xyz in offsets.items())
                                for bone, offsets in groups.items()))
        if not same_calibration:
            preservation.append('calibrated_marker_constraints_changed_or_unavailable')
        after = inspect_human_geometry(session.api, subject, markers, ranges, labeling, solving, reference)
        audit['after'] = after
        audit['preservation'] = dict(passed=not preservation, violations=preservation,
                                     checked_marker_tracks=len(all_points), metadata=['Radius', 'ReconIndex', 'FillType'])
        audit['solve'] = {kind.lower(): _bone_metrics(session, session.api, subject, kind, bones, ranges)
                          for kind, bones in (('Labeling', labeling), ('Solving', solving))}
        reasons = list(after['review_reasons']) + preservation
        if any(not np.isfinite(_trajectory(session.api, subject, marker, ranges)).all() for marker in markers):
            reasons.append('postrepair_marker_gaps')
        for kind, metrics in audit['solve'].items():
            if metrics['animation_coverage'] < 1. or not metrics['keyed_bones']:
                reasons.append(f'{kind}_skeleton_animation_partial')
        audit['review_reasons'] = list(dict.fromkeys(reasons))
        audit['accepted'] = not reasons
        audit['status'] = 'SUCCESS' if audit['accepted'] else 'NEEDS_REVIEW'
        if audit['accepted']:
            fbx = output / 'verified.fbx'
            scene.DeselectAllObjects()
            session.hsl('fbxExportOptions -exportActiveClipOnly true -collapseSubjects false -exportPlayRangeOnly false -exportLabelingSetups true;')
            session.hsl(f'saveFile {_quote(fbx.as_posix())};')
            audit['artifacts']['fbx'] = str(fbx)
        else:
            audit['candidate_rejected'] = True
            audit['review_reasons'].append('persistent_or_ambiguous_evidence_no_forced_correction')
            scene.New()
            load_native_file(session, baseline, audit=audit.setdefault('native_loads', []))
    except Exception as error:
        audit.update(status='FAILED', accepted=False, error=f'{type(error).__name__}: {error}')
    finally:
        audit['source_unchanged'] = _sha256(source) == fingerprint
        if not audit['source_unchanged']:
            audit.update(status='FAILED', accepted=False, error='Source hash changed during repair')
        if reference_path and 'reference' in audit:
            audit['reference']['source_unchanged'] = _sha256(reference_path) == reference_hash
            if not audit['reference']['source_unchanged']:
                audit.update(status='FAILED', accepted=False, error='Original reference MCP hash changed')
        (output / 'repair_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-vdf', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--shogun-cl', required=True)
    parser.add_argument('--reference-mcp', help='Original sparse Live labels; never reconstructed or saved over')
    args = parser.parse_args()
    from step1_process_takes import free_port
    from pipeline_session import ShogunSession
    # This CLI is its own Python process; never connects to or modifies a GUI.
    source = Path(args.input_vdf).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if source.suffix.lower() != '.vdf' or not source.is_file():
        raise ValueError('An existing completed VDF is required; MCP/X2D are never edited')
    if output.exists():
        raise ValueError('Output directory must not exist (no published artifacts overwritten)')
    if source.is_relative_to(output):
        raise ValueError('Output cannot contain the source')
    if args.reference_mcp and (not Path(args.reference_mcp).is_file() or Path(args.reference_mcp).suffix.lower() != '.mcp'):
        raise ValueError('--reference-mcp must name an existing original MCP')
    register_run(output, kind="human_target", input_path=source,
                 data_path=args.reference_mcp or source, subject=args.subject, planned_total=1,
                 label=f"Target-only human repair ({args.subject}): {output.name}")
    with tempfile.TemporaryDirectory(prefix='shogun-human-repair-') as temporary:
        with ShogunSession(args.shogun_cl, Path(temporary), Path(temporary) / 'native.log', free_port(set()), None, 90) as session:
            result = repair_existing(session, args.input_vdf, args.output_dir, args.subject, args.reference_mcp)
            native_log = Path(temporary) / 'native.log'
            if native_log.is_file():
                shutil.copy2(native_log, Path(args.output_dir) / 'native.log')
    print(json.dumps({
        'status': result['status'], 'accepted': result['accepted'],
        'source_unchanged': result.get('source_unchanged'),
        'quality_scope': result['quality_scope'],
        'review_reason_count': len(result.get('review_reasons', [])),
        'review_reasons': result.get('review_reasons', [])[:20],
        'corrected_marker_frames': sum(bool(row.get('marker')) for row in result.get('reference_relabel', {}).get('corrections', [])),
        'audit_path': str(Path(args.output_dir).resolve() / 'repair_audit.json'),
        'artifacts': result.get('artifacts', {}), 'error': result.get('error'),
    }, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'SUCCESS' else 2 if result['status'] == 'NEEDS_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Attach original prop GLB meshes to processed Shogun FBXs without re-solving.

Geometry follows named rigid bodies; original animation/markers remain unchanged.
Requires NumPy and Blender for one-time asset extraction. No Shogun SDK.
Configured sequence controls discover only canonical reports' declared FBXs and
publish beside the raw sequence under a separate mesh version. Standalone FBX
files/directories retain their ordinary, separate output-tree layout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from pipeline_layout import (artifact_roots, in_run, initialize_layout, load_layout,
                             report_paths, report_relative, take_paths)
from theme_roster import audit_take_mesh_completeness

DEFAULT_MAPPING = Path(__file__).with_name('prop_meshes.json')


def _json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical_inputs(source):
    """Authorize declared full/subject FBXs against each report's exact take."""
    declared = {}
    for report_path in report_paths(source):
        relative = report_relative(source, report_path)
        report = json.loads(report_path.read_text(encoding='utf-8-sig'))
        if not isinstance(report, dict) or Path(report.get('relative_path', '')) != relative:
            raise ValueError(f'Canonical report disagrees with raw take identity: {report_path}')
        products = report.get('artifacts', {})
        exports = report.get('subject_exports', [])
        if not isinstance(products, dict) or not isinstance(exports, list):
            raise ValueError(f'Invalid artifact declarations: {report_path}')
        entries = [(role, value, None) for role, value in products.items()
                   if isinstance(value, str) and Path(value).suffix.casefold() == '.fbx']
        for index, entry in enumerate(exports):
            if not isinstance(entry, dict):
                raise ValueError(f'Invalid subject export: {report_path}')
            role = entry.get('artifact_role', f'subject_{index}_fbx')
            value = entry.get('path')
            if role in products and products[role] != value:
                raise ValueError(f'Subject export disagrees with declared artifact: {report_path}')
            if not isinstance(value, str) or Path(value).suffix.casefold() != '.fbx':
                raise ValueError(f'Invalid subject FBX declaration: {report_path}')
            entries.append((role, value, entry))
        roots = artifact_roots(source, relative)
        for role, value, subject in entries:
            candidate = Path(value)
            if (not isinstance(role, str) or not role or not candidate.is_absolute()
                    or '..' in candidate.parts
                    or not in_run(source, candidate) or not candidate.is_file()
                    or candidate.stem.casefold().endswith('_with_mesh')):
                raise ValueError(f'Invalid or out-of-run declared FBX: {candidate}')
            artifact_root = next((root for root in roots if candidate.is_relative_to(root)), None)
            if artifact_root is None:
                raise ValueError(f'Declared FBX escapes this take artifact roots: {candidate}')
            candidate = candidate.resolve(strict=True)
            metadata = {'relative_path': relative,
                        'artifact_relative': candidate.relative_to(artifact_root),
                        'artifact_role': role, 'source_report': report_path}
            if subject is not None:
                metadata['subject_export'] = subject
            previous = declared.get(candidate)
            if previous is not None:
                if (previous['relative_path'], previous['artifact_role']) != (relative, role):
                    raise ValueError(f'FBX declared under conflicting take/role identities: {candidate}')
                previous.update(metadata)
            else:
                declared[candidate] = metadata
    return declared


def _discover_inputs(source, recursive=True, input_list=None):
    """Return FBXs and optional canonical scope; explicit lists never scan."""
    source = source.expanduser().absolute()
    layout = load_layout(source) if source.is_dir() else None
    scope = _canonical_inputs(source) if layout else None
    source = source.resolve(strict=True)
    if input_list is not None:
        if not source.is_dir():
            raise ValueError('--input must be a directory boundary when using --input-list')
        selected = json.loads(Path(input_list).read_text(encoding='utf-8-sig'))
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(value, str) or not value for value in selected)):
            raise ValueError('--input-list must contain a nonempty JSON array of FBX paths')
        paths, seen = [], set()
        for value in selected:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = source / candidate
            if candidate.is_symlink():
                raise ValueError(f'Input list must name ordinary files, not symlinks: {candidate}')
            candidate = candidate.resolve(strict=True)
            if (not (candidate in scope if scope is not None else candidate.is_relative_to(source))
                    or (scope is not None and not in_run(source, Path(value).expanduser()
                        if Path(value).expanduser().is_absolute() else source / value))
                    or not candidate.is_file()
                    or candidate.suffix.casefold() != '.fbx'
                    or candidate.stem.casefold().endswith('_with_mesh')):
                raise ValueError(f'Input list contains an invalid or out-of-root FBX: {candidate}')
            identity = str(candidate).casefold()
            if identity in seen:
                raise ValueError(f'Duplicate FBX in input list: {candidate}')
            seen.add(identity)
            paths.append(candidate)
        return source, paths, scope
    if scope is not None:
        paths = sorted(scope, key=lambda path: (
            scope[path]['relative_path'].as_posix().casefold(),
            scope[path]['artifact_relative'].as_posix().casefold()))
        if not paths:
            raise ValueError('No declared FBX files found in canonical sequence reports')
        return source, paths, scope
    if source.is_file():
        if source.suffix.casefold() != '.fbx':
            raise ValueError('--input must be an FBX file or a directory')
        return source.parent, [source], None
    if not source.is_dir():
        raise ValueError('--input must be an FBX file or a directory')
    candidates = source.rglob('*') if recursive else source.iterdir()
    paths = sorted((p for p in candidates if p.is_file() and not p.is_symlink()
                    and p.suffix.casefold() == '.fbx'
                    and not p.stem.casefold().endswith('_with_mesh')),
                   key=lambda p: p.relative_to(source).as_posix().casefold())
    if not paths:
        raise ValueError('No FBX files found. A Windows .lnk success view is not an FBX directory; use the actual artifacts directory.')
    return source, paths, None


def discover_inputs(source, recursive=True, input_list=None):
    """Return actual FBXs, including authorized sequence-local declarations."""
    root, paths, _ = _discover_inputs(source, recursive, input_list)
    return root, paths


def _check_roots(source_root, output, assets):
    # Disjoint trees prevent recursive re-ingestion and source/asset collisions.
    if (source_root == output or output.is_relative_to(source_root)
            or source_root.is_relative_to(output)):
        raise ValueError('--output-dir and the input directory must be disjoint; inputs are never overwritten')
    asset_list = assets if isinstance(assets, (list, tuple)) else [assets]
    for asset_dir in asset_list:
        if output == asset_dir or output.is_relative_to(asset_dir) or asset_dir.is_relative_to(output):
            raise ValueError('--output-dir must be separate from the asset library')

def _destinations(root, source, output, *, scope=None, data_root=None, stage='legacy'):
    if scope is not None:
        metadata = scope[source]
        paths = take_paths(output, metadata['relative_path'], data_root=data_root)
        relative = metadata['artifact_relative']
        base = paths['destination'] / relative.parent
        rep_base = paths['take_root'] / '_reports' / relative.parent
    else:
        relative = source.relative_to(root)
        base = output / relative.parent
        rep_base = output / '_reports' / relative.parent

    dest_mesh = base / (source.stem + '_with_mesh.fbx')
    dest_props = base / (source.stem + '_with_props.fbx')
    dest_scene = base / (source.stem + '_with_scene.fbx')
    report = rep_base / (source.name + '.json')
    ver_mesh = report.with_name(source.name + '.verification.json')
    ver_props = report.with_name(source.name + '.with_props.verification.json')
    ver_scene = report.with_name(source.name + '.with_scene.verification.json')

    for path in (dest_mesh, dest_props, dest_scene, report, ver_mesh, ver_props, ver_scene):
        if os.path.normcase(str(path.absolute())) != os.path.normcase(str(path.resolve())):
            raise ValueError(f'Mesh output traverses a junction or symbolic link: {path}')

    if stage == 'props':
        return dest_props, report, ver_props
    elif stage == 'scene':
        return dest_scene, report, ver_scene
    elif stage == 'both':
        return {
            'dest_props': dest_props,
            'dest_scene': dest_scene,
            'dest_mesh': dest_mesh,
            'destination': dest_mesh,
            'report': report,
            'ver_props': ver_props,
            'ver_scene': ver_scene,
            'verification': ver_mesh,
        }
    return dest_mesh, report, ver_mesh

def _mesh_layout(root, output, scope):
    """Validate inherited layout without creating a manifest or directory."""
    existing = load_layout(output)
    if scope is None:
        if existing is not None:
            raise ValueError('A configured mesh output requires a configured sequence input')
        return None
    source_layout = load_layout(root)
    expected = {'schema_version': 1, 'data_root': source_layout['data_root'],
                'version_key': output.name}
    if expected['version_key'].casefold() == source_layout['version_key'].casefold():
        raise ValueError('Mesh output version must be separate from the source version')
    if existing is not None and existing != expected:
        raise ValueError('Mesh output layout belongs to a different raw dataset/version')
    for source in scope:
        _destinations(root, source, output, scope=scope, data_root=expected['data_root'])
    return expected


def _write_take_reports(output, scope, summary):
    """Publish canonical per-take summaries; detailed FBX reports stay local."""
    grouped = {}
    for entry in summary['files']:
        metadata = scope[Path(entry['input'])]
        grouped.setdefault(metadata['relative_path'], []).append((metadata, entry))
    for relative, rows in grouped.items():
        files = [entry for _, entry in rows]
        counts = Counter(entry['status'] for entry in files)
        status = next((value for value in ('FAILED', 'PARTIAL', 'NEEDS_REVIEW', 'NO_MATCH',
                                          'NO_PROPS', 'SKIPPED_EXISTING')
                       if counts.get(value)), 'SUCCESS')
        artifacts, exports = {}, []
        for metadata, entry in rows:
            value = entry.get('output')
            if not value:
                continue
            artifacts[metadata['artifact_role']] = value
            if 'subject_export' in metadata:
                subject = dict(metadata['subject_export'])
                subject.update(artifact_role=metadata['artifact_role'], path=value)
                exports.append(subject)
        _json_write(take_paths(output, relative)['report_path'],
                    {'schema_version': 1, 'relative_path': relative.as_posix(),
                     'source_report': str(rows[0][0]['source_report']),
                     'status': status, 'tracking_quality_changed': False,
                     'artifacts': artifacts, 'subject_exports': exports,
                     'files': files, 'counts': dict(counts),
                     'finished_utc': summary['finished_utc']})


def _find_blender(value):
    executable = shutil.which(value)
    if executable:
        return executable
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    raise ValueError('Blender was not found. Put blender on PATH or pass --blender <path/to/blender.exe>.')


def _prepare_assets(executable, records, cache, output):
    request = cache / 'request.json'
    _json_write(request, {'output_dir': str(cache / 'assets'), 'assets': records})
    log_path = output / 'mesh_asset_preparation.log'
    command = [executable, '--background', '--factory-startup', '--threads', '2',
               '--python-exit-code', '1', '--python', str(Path(__file__).with_name('mesh_assets.py')),
               '--', '--request', str(request)]
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
    print(f'[ASSETS] Preparing {len(records)} unique assets once; log: {log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=flags, check=False)
    index_path = cache / 'assets' / 'index.json'
    if completed.returncode != 0 or not index_path.is_file():
        raise RuntimeError(f'Blender asset extraction failed (exit {completed.returncode}); see {log_path}')
    index = json.loads(index_path.read_text(encoding='utf-8-sig'))
    return index['assets']


def _load_asset(record, np):
    parts = []
    for part in record['parts']:
        with np.load(part['arrays'], allow_pickle=False) as data:
            arrays = {key: data[key] for key in
                      ('vertices_mm', 'triangles', 'triangle_material_indices')}
            if 'normals' in data:
                arrays['normals'] = data['normals']
        for array in arrays.values():
            array.setflags(write=False)
        parts.append({**part, **arrays})
    if not parts:
        raise ValueError('Prepared asset has no actual mesh parts')
    return parts


def _build_meshes(matched, prepared, loaded, native_unit_mm, np):
    meshes, attached, rejected = [], [], []
    for record in matched:
        asset_id = record['asset_id']
        asset = prepared.get(asset_id)
        if not asset or asset.get('status') != 'READY':
            rejected.append({'prop': record['prop'], 'asset_id': asset_id,
                             'reason': 'ASSET_PREPARATION_FAILED',
                             'details': (asset or {}).get('error', 'No prepared asset entry')})
            continue
        try:
            if asset_id not in loaded:
                loaded[asset_id] = _load_asset(asset, np)
            rotation = np.asarray(record['asset_to_bone_rotation'], dtype=float)
            translation = np.asarray(record['asset_to_bone_translation_mm'], dtype=float)
            prop_parts, part_reports = [], []
            for index, part in enumerate(loaded[asset_id]):
                name = f"Mesh_{record['prop']}_{part['name']}_{index}"
                mesh = {'name': name, 'prop': record['prop'], 'parent_uid': record['bone_uid'],
                        'vertices': (part['vertices_mm'] @ rotation.T + translation) / native_unit_mm,
                        'triangles': part['triangles'], 'materials': part['materials'],
                        'triangle_material_indices': part['triangle_material_indices']}
                if 'normals' in part:
                    mesh['normals'] = part['normals'] @ rotation.T
                prop_parts.append(mesh)
                part_reports.append({'name': name, 'source_glb': part.get('source_glb'),
                                     'source_glb_object': part.get('source_glb_object'),
                                     'materials': part['materials'],
                                     'vertices': len(part['vertices_mm']), 'triangles': len(part['triangles'])})
            meshes.extend(prop_parts)
            attached.append({'prop': record['prop'], 'asset_id': asset_id,
                             'mesh_parts': len(prop_parts), 'parts': part_reports,
                             'asset_audit': asset.get('audit', {})})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append({'prop': record['prop'], 'asset_id': asset_id,
                             'reason': 'ASSET_PREPARATION_FAILED', 'details': str(exc)})
    return meshes, attached, rejected



def _audit_roster(identity, plan, mesh_props):
    """Audit current-stage evidence; detected names never shrink the roster."""
    scene_props = {record['prop'] for key in ('matched_props', 'skipped_props')
                   for record in plan.get(key, [])}
    return audit_take_mesh_completeness(str(identity), scene_props, set(mesh_props))


def _roster_identity(source, scope):
    return str(scope[source]['relative_path']) if scope is not None else str(source)


def _empty_status(plan):
    if not plan.get('detected_prop_count'):
        return 'NO_PROPS'
    benign = {'NO_MODEL', 'ALREADY_HAS_MESH'}
    reasons = {entry['reason'] for entry in plan.get('skipped_props', [])}
    return 'NO_MATCH' if reasons <= benign else 'NEEDS_REVIEW'


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Configured sequence control, processed FBX or standalone directory')
    parser.add_argument('--input-list', type=Path, help='JSON array of exact FBXs: declared local artifacts for configured controls, otherwise paths under --input')
    parser.add_argument('--output-dir', required=True, type=Path, help='Separate mesh control/output tree; local controls inherit the raw dataset and use this directory name as mesh version')
    parser.add_argument('--assets-root', required=True, help='Library or semicolon-separated libraries containing original GLBs and hole JSONs')
    parser.add_argument('--mapping', type=Path, default=DEFAULT_MAPPING, help='Prop-name/candidate asset JSON mapping')
    parser.add_argument('--blender', default='blender', help='Blender executable name or absolute path')
    parser.add_argument('--max-rms-mm', type=float, default=2., help='Maximum nominal calibration RMS in millimeters')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false', help='Only scan FBXs directly in input directory')
    parser.add_argument('--overwrite', action='store_true', help='Replace prior generated results, never source FBXs')
    parser.add_argument('--dry-run', action='store_true', help='Only analyze names/geometry; no Blender, output files or directories')
    parser.add_argument('--environment', type=Path, help='Optional static scene environment GLB file or directory containing meeting_room_environment.glb')
    parser.add_argument('--environment-parent', default='System', help='Root model name in FBX to attach static environment to (default: System)')
    parser.add_argument('--export-stage', choices=['props', 'scene', 'both', 'legacy'], default='legacy',
                        help="Export stage: 'props' (only props mesh), 'scene' (props + room scene mesh), 'both' (exports both FBX files), or 'legacy' (_with_mesh.fbx)")
    parser.add_argument('--no-strip-unlabeled', dest='strip_unlabeled', action='store_false',
                        help='Retain Unlabeled_Markers in exported FBX (default: strip them to reduce file size)')
    parser.add_argument('--max-outlier-multiplier', type=float, default=35.0,
                        help='Tolerance factor for instantaneous single-frame occlusion spikes (default: 35.0)')
    return parser.parse_args(argv)


def run(args):
    if not math.isfinite(args.max_rms_mm) or args.max_rms_mm <= 0:
        raise ValueError('--max-rms-mm must be finite and positive')
    try:
        import numpy as np
        from mesh_alignment import load_catalog, load_mapping, analyze_fbx
        from mesh_fbx import graft_file, load_parser, parse_checked
    except ImportError as exc:
        raise RuntimeError('Requires NumPy in this Python environment: python -m pip install numpy') from exc
    root, inputs, scope = _discover_inputs(args.input, args.recursive, args.input_list)
    output = args.output_dir.expanduser().absolute()
    mesh_layout = _mesh_layout(root, output, scope)
    output = output.resolve()
    raw_assets = [Path(p).expanduser().resolve(strict=True) for p in str(args.assets_root).split(';') if p.strip()]
    if not raw_assets:
        raise ValueError('At least one directory is required in --assets-root')
    env_glb = None
    if getattr(args, 'environment', None) is not None:
        cand = args.environment.expanduser().resolve(strict=True)
        if cand.is_dir():
            candidates = sorted(cand.rglob('*.glb'))
            if not candidates:
                raise ValueError(f'No GLB file found under --environment directory: {cand}')
            target_glb = next((c for c in candidates if 'environment' in c.stem.lower()), candidates[0])
            env_glb = target_glb
        elif cand.is_file() and cand.suffix.lower() == '.glb':
            env_glb = cand
        else:
            raise ValueError(f'--environment must be a .glb file or directory containing .glb: {args.environment}')
    _check_roots(root, output, raw_assets + ([env_glb.parent] if env_glb else []))
    if scope is not None:
        for source in inputs:
            local_paths = take_paths(output, scope[source]['relative_path'],
                                     data_root=mesh_layout['data_root'])
            _check_roots(source.parent, local_paths['take_root'],
                         raw_assets + ([env_glb.parent] if env_glb else []))
    catalog = load_catalog(raw_assets)
    mapping = load_mapping(args.mapping.expanduser().resolve(strict=True))
    if not catalog:
        raise ValueError('No original GLB + valid hole-reference pairs found under --assets-root')
    blender = None if args.dry_run else _find_blender(args.blender)
    summary = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
               'input': str(args.input.resolve()), 'output_dir': str(output),
               'assets_root': [str(p) for p in raw_assets], 'mapping': str(args.mapping.resolve()),
               'dry_run': args.dry_run, 'files': [],
               'limits': ['Mesh attachment does not change the source tracking-quality classification.',
                          'Hole-mouth positions are nominal references; physical sphere-center seating offset is not validated.',
                          'Missing models and ambiguous correspondences are reported, never fabricated.']}
    if env_glb is not None:
        summary['environment'] = {
            'glb': str(env_glb),
            'parent_model': getattr(args, 'environment_parent', 'System')
        }
    if args.input_list is not None:
        summary['input_list'] = str(args.input_list.resolve())
        summary['input_list_sha256'] = hashlib.sha256(args.input_list.read_bytes()).hexdigest()
    if args.dry_run:
        for index, source in enumerate(inputs, 1):
            destination, _, _ = _destinations(root, source, output, scope=scope,
                                               data_root=mesh_layout['data_root'] if mesh_layout else None)
            try:
                plan = analyze_fbx(source, catalog, mapping, max_rms_mm=args.max_rms_mm,
                                   max_outlier_multiplier=args.max_outlier_multiplier)
                entry = {'input': str(source), 'output': str(destination), 'status': 'PLANNED',
                         'plan': plan, 'existing_output': destination.exists()}
                plan['roster_identity'] = _roster_identity(source, scope)
                plan['roster_audit'] = _audit_roster(
                    plan['roster_identity'], plan,
                    (record['prop'] for record in plan.get('matched_props', [])))
            except Exception as exc:
                entry = {'input': str(source), 'status': 'FAILED', 'error': str(exc)}
            summary['files'].append(entry)
            print(f"[PLAN {index}/{len(inputs)}] {source.name}: {entry['status']}", file=sys.stderr, flush=True)
    else:
        if mesh_layout is not None:
            initialize_layout(output, mesh_layout['data_root'], version_key=output.name)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.mesh-work-', dir=output) as temporary:
            cache = Path(temporary)
            jobs, required_ids = [], set()
            for index, source in enumerate(inputs, 1):
                destination, report_path, verification = _destinations(root, source, output, scope=scope)
                if destination.exists() and not args.overwrite:
                    summary['files'].append({'input': str(source), 'output': str(destination),
                                             'status': 'SKIPPED_EXISTING', 'existing_output_validated': False})
                    print(f'[SKIP {index}/{len(inputs)}] Existing output: {destination}', flush=True)
                    continue
                try:
                    plan = analyze_fbx(source, catalog, mapping, max_rms_mm=args.max_rms_mm,
                                       max_outlier_multiplier=args.max_outlier_multiplier)
                    plan['roster_identity'] = _roster_identity(source, scope)
                    plan['roster_audit'] = _audit_roster(
                        plan['roster_identity'], plan,
                        (record['prop'] for record in plan['matched_props']))
                    plan_path = cache / 'plans' / f'{index}.json'
                    _json_write(plan_path, plan)
                    job = {'input': str(source), 'output': str(destination), 'report': str(report_path),
                           'verification': str(verification), 'plan_path': str(plan_path)}
                    jobs.append(job)
                    required_ids.update(record['asset_id'] for record in plan['matched_props'])
                    print(f"[PLAN {index}/{len(inputs)}] {source.name}: {len(plan['matched_props'])} props eligible, {len(plan.get('skipped_props', []))} skipped/review", flush=True)
                except Exception as exc:
                    entry = {'input': str(source), 'status': 'FAILED', 'error': str(exc), 'report': str(report_path)}
                    _json_write(report_path, entry)
                    summary['files'].append(entry)
                    print(f'[FAILED] {source}: {exc}', file=sys.stderr, flush=True)
            assets_to_prep = [catalog[key] for key in sorted(required_ids)]
            if env_glb is not None:
                assets_to_prep.append({'id': 'environment', 'glb': str(env_glb)})
            prepared = {}
            if assets_to_prep:
                try:
                    prepared = _prepare_assets(blender, assets_to_prep, cache, output)
                except Exception as exc:
                    prepared = {entry['id']: {'status': 'FAILED', 'error': str(exc)} for entry in assets_to_prep}
            loaded = {}
            for index, job in enumerate(jobs, 1):
                plan = json.loads(Path(job['plan_path']).read_text(encoding='utf-8'))
                entry = {key: value for key, value in job.items() if key != 'plan_path'}
                entry['plan'] = plan
                entry['tracking_quality_changed'] = False
                entry['roster_audits'] = {'matched': plan['roster_audit']}
                entry['roster_audit'] = _audit_roster(plan['roster_identity'], plan, ())
                entry['roster_audits']['attached'] = entry['roster_audit']
                published = False
                try:
                    meshes, attached, rejected = _build_meshes(plan['matched_props'], prepared, loaded,
                                                              plan['native_unit_mm'], np)
                    prop_meshes = list(meshes)
                    entry['prepared_props'] = attached
                    entry['attached_props'] = []
                    entry['unresolved_props'] = plan.get('skipped_props', []) + rejected
                    entry['roster_audits']['prepared'] = _audit_roster(
                        plan['roster_identity'], plan, (record['prop'] for record in attached))
                    env_mesh_parts = []
                    if env_glb is not None:
                        env_asset = prepared.get('environment')
                        if not env_asset or env_asset.get('status') != 'READY':
                            rejected.append({'prop': 'environment', 'asset_id': 'environment',
                                             'reason': 'ASSET_PREPARATION_FAILED',
                                             'details': (env_asset or {}).get('error', 'No prepared environment asset')})
                        else:
                            if 'environment' not in loaded:
                                loaded['environment'] = _load_asset(env_asset, np)
                            parser = load_parser()
                            source_bytes = Path(job['input']).read_bytes()
                            graph = parser.build_graph(source_bytes, parse_checked(parser, source_bytes, allow_unterminated_children=True))
                            parent_name = getattr(args, 'environment_parent', 'System')
                            parent_candidates = [uid for uid, (n, _) in graph.model_names.items() if n == parent_name]
                            if not parent_candidates:
                                rejected.append({'prop': 'environment', 'asset_id': 'environment',
                                                 'reason': 'PARENT_NOT_FOUND',
                                                 'details': f"Model '{parent_name}' not found in FBX"})
                            else:
                                parent_uid = parent_candidates[0]
                                env_mesh_parts = []
                                for part in loaded['environment']:
                                    m = {
                                        'name': f"Environment_{part['name']}",
                                        'prop': 'environment',
                                        'parent_uid': parent_uid,
                                        'vertices': part['vertices_mm'] / plan['native_unit_mm'],
                                        'triangles': part['triangles'],
                                        'materials': part['materials'],
                                        'triangle_material_indices': part['triangle_material_indices'],
                                    }
                                    if 'normals' in part:
                                        m['normals'] = part['normals']
                                    env_mesh_parts.append(m)
                                meshes.extend(env_mesh_parts)
                                entry['environment_attached'] = {
                                    'parts': len(env_mesh_parts),
                                    'parent_model': parent_name,
                                    'parent_uid': parent_uid,
                                    'triangles': sum(len(m['triangles']) for m in env_mesh_parts)
                                }
                    stage = getattr(args, 'export_stage', 'legacy')
                    if meshes:
                        source_target = Path(job['input'])
                        if hashlib.sha256(source_target.read_bytes()).hexdigest() != plan['source_sha256']:
                            raise ValueError('Source FBX changed since alignment; refusing stale calibration')
                        source_p = source_target
                        if getattr(args, 'strip_unlabeled', True):
                            try:
                                from extract_groups import parse_file, build_graph, branch_uids, splice
                                buf, ver, tops = parse_file(str(source_p))
                                g = build_graph(buf, tops)
                                if 'Unlabeled_Markers' in g.roots:
                                    keep_roots = [r for r in g.roots if r != 'Unlabeled_Markers']
                                    keep = set()
                                    for r in keep_roots:
                                        keep |= branch_uids(g, g.roots[r])
                                    changed = True
                                    while changed:
                                        changed = False
                                        for conn in g.conns:
                                            if conn[0] == 'OP' and conn[2] in keep and conn[1] not in keep:
                                                keep.add(conn[1])
                                                changed = True
                                    keep &= set(g.uid2node)
                                    for c in g.obj_children:
                                        if c.name in (b'AnimationStack', b'AnimationLayer'):
                                            keep.add(c.uid)
                                    stripped_bytes, _, _ = splice(buf, tops, keep)
                                    clean_tmp = cache / f"clean_{index}.fbx"
                                    clean_tmp.write_bytes(stripped_bytes)
                                    source_p = clean_tmp
                                    entry['unlabeled_markers_stripped'] = True
                            except Exception as exc:
                                entry['unlabeled_strip_warning'] = str(exc)
                        source_orig = Path(job['input'])
                        if stage == 'both':
                            dest_p, _, ver_p = _destinations(root, source_orig, output, scope=scope, stage='props')
                            dest_s, _, ver_s = _destinations(root, source_orig, output, scope=scope, stage='scene')
                            if prop_meshes:
                                graft_file(source_p, dest_p, prop_meshes, verification_path=ver_p, overwrite=args.overwrite)
                                entry['output_props'] = str(dest_p)
                            if env_mesh_parts:
                                verification = graft_file(source_p, dest_s, meshes, verification_path=ver_s, overwrite=args.overwrite)
                                entry['output_scene'] = str(dest_s)
                                entry['output'] = str(dest_s)
                                entry['verification_path'] = str(ver_s)
                            else:
                                verification = graft_file(source_p, dest_p, prop_meshes, verification_path=ver_p, overwrite=args.overwrite)
                                entry['output'] = str(dest_p)
                                entry['verification_path'] = str(ver_p)
                        elif stage == 'props':
                            dest_p, _, ver_p = _destinations(root, source_orig, output, scope=scope, stage='props')
                            verification = graft_file(source_p, dest_p, prop_meshes, verification_path=ver_p, overwrite=args.overwrite)
                            entry['output'] = str(dest_p)
                            entry['output_props'] = str(dest_p)
                            entry['verification_path'] = str(ver_p)
                        elif stage == 'scene':
                            dest_s, _, ver_s = _destinations(root, source_orig, output, scope=scope, stage='scene')
                            verification = graft_file(source_p, dest_s, meshes, verification_path=ver_s, overwrite=args.overwrite)
                            entry['output'] = str(dest_s)
                            entry['output_scene'] = str(dest_s)
                            entry['verification_path'] = str(ver_s)
                        else:
                            verification = graft_file(source_p, Path(job['output']), meshes,
                                                      verification_path=Path(job['verification']), overwrite=args.overwrite)
                            entry['verification_path'] = job['verification']
                        published = True
                        if not verification.get('passed'):
                            raise ValueError('Attached FBX verification did not pass')
                        entry['attached_props'] = entry.pop('prepared_props')
                        entry['verification'] = {key: verification[key] for key in
                                                 ('passed', 'checks', 'source_sha256', 'output_sha256',
                                                  'source_bytes', 'output_bytes', 'new_object_count',
                                                  'inherited_dangling_reference_count')
                                                 if key in verification}
                        entry['roster_audit'] = _audit_roster(
                            plan['roster_identity'], plan,
                            (record['prop'] for record in entry['attached_props']))
                        entry['roster_audits']['attached'] = entry['roster_audit']
                        entry['status'] = ('PARTIAL' if entry['unresolved_props']
                                           or not entry['roster_audit']['all_complete'] else 'SUCCESS')
                    else:
                        entry['status'] = 'FAILED' if rejected else _empty_status(plan)
                        if entry['status'] != 'FAILED' and not entry['roster_audit']['all_complete']:
                            entry['status'] = 'NEEDS_REVIEW'
                        entry['output'] = None
                    _json_write(Path(job['report']), entry)
                except Exception as exc:
                    entry['status'] = 'FAILED'
                    entry['error'] = str(exc)
                    entry['output_published_this_run'] = published
                    entry['requested_output'] = job['output']
                    if not published:
                        entry['output'] = None
                    _json_write(Path(job['report']), entry)
                summary['files'].append({key: value for key, value in entry.items() if key != 'plan'})
                print(f"[{entry['status']} {index}/{len(jobs)}] {job['input']} -> {entry.get('output')}", flush=True)
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    summary['counts'] = dict(Counter(entry['status'] for entry in summary['files']))
    if not args.dry_run:
        if scope is not None:
            _write_take_reports(output, scope, summary)
        _json_write(output / 'mesh_batch_report.json', summary)
    print(json.dumps(summary if args.dry_run else {'counts': summary['counts'],
                                                 'report': str(output / 'mesh_batch_report.json')},
                     ensure_ascii=False, indent=2, allow_nan=False))
    if summary['counts'].get('FAILED'):
        return 1
    if any(summary['counts'].get(status) for status in ('PARTIAL', 'NO_MATCH', 'NEEDS_REVIEW')):
        return 2
    return 0


def main(argv=None):
    args = _arguments(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print('Interrupted; no source FBX was modified.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'step6_attach_prop_meshes: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

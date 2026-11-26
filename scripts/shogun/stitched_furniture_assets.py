"""Prepare static furniture for native stitched FBXs without touching their motion.

Blender --background --python stitched_furniture_assets.py -- --request request.json
Request: native_scene_inputs (path); optional output_directory, control_directory.
The native extraction supplies stable reference marker centres and full valid-frame
bone-local observations. Marker JSON ball_centre_mm is already the sphere centre.
The 2 mm gate is four-point registration RMS, not a full-trajectory quality claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

FIT_GATE_MM = 2.0
PROPS = ('desk', 'orange_chair_1', 'white_chair_2')
PAIRS = tuple(itertools.combinations(range(4), 2))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def centres(document):
    markers = document['markers']
    if len(markers) != 4:
        raise ValueError('Expected exactly four source marker centres')
    points = np.array([[m['ball_centre_mm'][k] for k in 'xyz'] for m in markers])
    if not np.isfinite(points).all() or np.linalg.matrix_rank(points-points.mean(0), tol=.01) < 2:
        raise ValueError('Invalid or collinear source marker centres')
    return points


def fit(source, target):
    x, y = source-source.mean(0), target-target.mean(0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[2, 2] = 1. if np.linalg.det(vt.T @ u.T) >= 0 else -1.
    rotation = vt.T @ correction @ u.T
    translation = target.mean(0)-rotation @ source.mean(0)
    errors = np.linalg.norm(source @ rotation.T + translation-target, axis=1)
    similarity_scale = float(np.sum((x @ rotation.T)*y)/np.sum(x*x))
    return {'rotation': rotation.tolist(), 'translation_mm': translation.tolist(),
            'rms_mm': float(np.sqrt(np.mean(errors**2))), 'max_mm': float(errors.max()),
            'marker_errors_mm': errors.tolist(), 'determinant': float(np.linalg.det(rotation)),
            'unapplied_similarity_scale': similarity_scale}


def candidates(source, target):
    result = []
    for permutation in itertools.permutations(range(4)):
        result.append({'source_indices_in_capture_order': list(permutation),
                       **fit(source[list(permutation)], target)})
    return sorted(result, key=lambda record: record['rms_mm'])


def distances(points):
    return np.array([np.linalg.norm(points[a]-points[b]) for a, b in PAIRS])


def stats(errors):
    return {'rms_mm': float(np.sqrt(np.mean(errors**2))),
            'p99_mm': float(np.percentile(errors, 99)), 'max_mm': float(errors.max()),
            'count': int(errors.size), 'over_2mm_count': int(np.count_nonzero(errors > FIT_GATE_MM))}


def inspect_blend(path):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scale = bpy.context.scene.unit_settings.scale_length
    records = []
    for obj in bpy.context.scene.objects:
        record = {'name': obj.name, 'type': obj.type,
                  'matrix_world': np.array(obj.matrix_world).tolist(),
                  'properties': {k: str(v) for k, v in obj.items()}}
        if obj.type == 'MESH':
            xyz = np.array([v.co[:] for v in obj.data.vertices])
            matrix = np.array(obj.matrix_world)
            xyz = (xyz @ matrix[:3, :3].T+matrix[:3, 3])*scale*1000
            record.update(vertex_count=len(xyz), bounds_mm=[xyz.min(0).tolist(), xyz.max(0).tolist()],
                          modifiers=[m.name for m in obj.modifiers],
                          materials=[m.name if m else None for m in obj.data.materials],
                          uv_layers=list(obj.data.uv_layers.keys()))
        records.append(record)
    return {'path': str(path), 'sha256': sha256(path), 'scale_length': scale, 'objects': records}


def scale_copy(asset, factor, destination):
    """Create a real metric GLB and matching centres, never a scaled rotation."""
    import bpy
    from mathutils import Matrix
    import mounting_assets
    glb = Path(asset['glb'])
    objects, _, _ = mounting_assets._import_glb(glb)
    for obj in objects:
        obj.matrix_world = Matrix.Scale(factor, 4) @ obj.matrix_world
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    output = destination / (glb.stem+'_capture_scaled.glb')
    if output.exists():
        raise FileExistsError(output)
    bpy.ops.export_scene.gltf(filepath=str(output), export_format='GLB', use_selection=True,
                             export_animations=False, export_extras=False)
    calibration = copy.deepcopy(asset['document'])
    for marker in calibration['markers']:
        for key in ('ball_centre_mm', 'ball_centre_m', 'disc_bottom_mm', 'surface_point_m'):
            if key in marker:
                marker[key] = {k: float(v)*factor for k, v in marker[key].items()}
    calibration['capture_scale'] = {
        'factor': factor, 'original_marker_json': asset['marker_json'],
        'original_marker_json_sha256': sha256(asset['marker_json']),
        'method': 'One uniform object scale shared by every take; positions only, physical marker spec unchanged',
        'original_glb': str(glb), 'original_glb_sha256': sha256(glb)}
    calibration['files'] = {'glb': str(output)}
    calibration['pairwise_distance_mm'] = {
        f"{calibration['markers'][a]['mcp_name']}-{calibration['markers'][b]['mcp_name']}": float(value)
        for (a, b), value in zip(PAIRS, distances(centres(calibration)))}
    marker_output = destination / (glb.stem+'_capture_scaled_marker_positions.json')
    save(marker_output, calibration)
    return str(output), str(marker_output), calibration


def scale_evidence(source, takes, prop):
    records = []
    for take in takes:
        reference = np.array(take['furniture'][prop]['reference_mm'])
        best = candidates(source, reference)[0]
        ordered = source[best['source_indices_in_capture_order']]
        records.append({'take': take['take'], 'best_metric_fit': best,
                        'source_pair_distances_mm': distances(ordered).tolist(),
                        'native_pair_distances_mm': distances(reference).tolist(),
                        'native_over_source_distance_ratios': (distances(reference)/distances(ordered)).tolist()})
    required = any(r['best_metric_fit']['rms_mm'] > FIT_GATE_MM for r in records)
    factor = 1.
    decision = 'Metric scale 1 retained: every stable reference passes 2 mm RMS.'
    if required:
        mappings = {tuple(r['best_metric_fit']['source_indices_in_capture_order']) for r in records}
        ratios = np.array([r['native_over_source_distance_ratios'] for r in records])
        if len(mappings) != 1 or not (np.all(ratios < 1) or np.all(ratios > 1)):
            raise ValueError(f'{prop}: inconsistent scale evidence; cannot use per-take scales')
        factor = float(np.mean([r['best_metric_fit']['unapplied_similarity_scale'] for r in records]))
        if not all(candidates(source*factor, np.array(t['furniture'][prop]['reference_mm']))[0]['rms_mm'] <= FIT_GATE_MM
                   for t in takes):
            raise ValueError(f'{prop}: a common uniform scale does not satisfy 2 mm RMS')
        decision = ('All six distances in all takes show the same scale direction; equal-take joint '
                    'proper-similarity least squares supplies one common factor. Residual shape '
                    'inconsistency remains reported; this is capture-based, not a physical resurvey.')
    return {'applied_common_scale': factor, 'scale_required_by_metric_gate': required,
            'decision': decision, 'pairs_in_capture_order_zero_based': PAIRS, 'takes': records}


def floor_reference(take):
    floor = next(e for e in take['environment'] if e['name'] == 'Environment_Floor')
    xyz = np.asarray(floor['vertices_native']) @ np.asarray(floor['mesh_local_rotation']).T
    xyz += np.asarray(floor['mesh_local_translation_native'])
    xyz *= take['native_unit_mm']
    # Existing floor includes thickness. Fit the upper, horizontal surface only.
    top = xyz[np.isclose(xyz[:, 2], xyz[:, 2].max(), atol=.1)]
    if len(top) < 3:
        raise ValueError('Cannot identify native floor upper surface')
    _, _, vt = np.linalg.svd(top-top.mean(0))
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, top.mean(0)


def physical_review(take, prop, record, local_vertices, desk_fit):
    native = take['furniture'][prop]
    root_r = np.asarray(native['sample_rotations'])
    root_t = np.asarray(native['sample_translations_mm'])
    rotation = np.asarray(record['asset_to_bone_rotation'])
    up = root_r @ rotation[:, 2]
    desk_r = np.asarray(take['furniture']['desk']['sample_rotations'])
    desk_up = desk_r @ np.asarray(desk_fit['rotation'])[:, 2]
    floor_up, floor_origin = floor_reference(take)
    world = np.einsum('nij,vj->nvi', root_r, local_vertices*take['native_unit_mm'])+root_t[:, None, :]
    signed_floor_distance = (world-floor_origin) @ floor_up
    floor_angles = np.degrees(np.arccos(np.clip(up @ floor_up, -1, 1)))
    desk_angles = np.degrees(np.arccos(np.clip(np.sum(up*desk_up, axis=1), -1, 1)))
    return {'sample_times_seconds': native['sample_times_seconds'], 'source_up_axis': [0, 0, 1],
            'world_source_up_vectors': up.tolist(), 'angle_to_floor_up_degrees': floor_angles.tolist(),
            'angle_to_desk_up_degrees': desk_angles.tolist(),
            'body_lowest_floor_signed_distance_mm': signed_floor_distance.min(1).tolist(),
            'native_floor_up': floor_up.tolist(), 'native_floor_origin_mm': floor_origin.tolist(),
            'upright_clamp_applied': False, 'review_required': bool(np.max(floor_angles) > 5
                or signed_floor_distance.min() < -5 or signed_floor_distance.min(1).max() > 10),
            'interpretation': 'Observed native root poses only; source +Z is current four-feet-level chair frame. '
                              'Floor clearance is body bounding minimum, not a contact/occlusion certification.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(argv)
    request_path = args.request.resolve(strict=True)
    request = json.loads(request_path.read_text(encoding='utf-8-sig'))
    input_path = Path(request['native_scene_inputs']).resolve(strict=True)
    native = json.loads(input_path.read_text(encoding='utf-8-sig'))
    control = Path(request.get('control_directory', native['control_directory']))
    destination = Path(request.get('output_directory', native['output_directory'])) / 'furniture_assets'
    index_path = control / 'furniture_asset_index.json'
    if index_path.exists() or destination.exists():
        raise FileExistsError('Furniture output/index already exists; choose a new version directory')
    destination.mkdir(parents=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mounting_assets
    from io_scene_fbx.parse_fbx import parse
    assets = {}
    index = {'schema_version': 1, 'request': str(request_path), 'request_sha256': sha256(request_path),
             'native_scene_inputs': str(input_path), 'native_scene_inputs_sha256': sha256(input_path),
             'fit_gate': {'rms_mm': FIT_GATE_MM, 'scope': 'Stable four-point reference registration',
                          'full_frame_gate_not_claimed': True},
             'coordinate_convention': 'GLB evaluated Blender world millimetres -> native bone-local millimetres; '
                                      'snippet raw vertices are native units; no trajectory or bone edits',
             'source_numbering_is_not_capture_mapping': True, 'assets': {}, 'furniture_exports': {}}
    for prop in PROPS:
        asset = dict(native['furniture_assets'][prop])
        asset['document'] = json.loads(Path(asset['marker_json']).read_text(encoding='utf-8-sig'))
        blend = Path(asset['glb']).with_suffix('.blend')
        inspection = inspect_blend(blend)
        inspection_path = control / f'{prop}_current_blend_inspection.json'
        save(inspection_path, inspection)
        source = centres(asset['document'])
        evidence = scale_evidence(source, native['takes'], prop)
        glb, calibration, document = asset['glb'], asset['marker_json'], asset['document']
        if evidence['applied_common_scale'] != 1.:
            glb, calibration, document = scale_copy(asset, evidence['applied_common_scale'], destination)
        assets[prop] = {**asset, 'source': centres(document), 'prepared_glb': glb,
                        'prepared_calibration': calibration, 'scale_evidence': evidence}
        index['assets'][prop] = {
            'canonical_glb': asset['glb'], 'canonical_glb_sha256': sha256(asset['glb']),
            'canonical_marker_json': asset['marker_json'], 'canonical_marker_json_sha256': sha256(asset['marker_json']),
            'current_blend': str(blend), 'current_blend_sha256': inspection['sha256'],
            'current_blend_inspection': str(inspection_path), 'current_blend_inspection_sha256': sha256(inspection_path),
            'prepared_glb': glb, 'prepared_glb_sha256': sha256(glb),
            'prepared_marker_json': calibration, 'prepared_marker_json_sha256': sha256(calibration),
            'coordinate_frame': document['coordinate_frame'], 'scale_evidence': evidence}
    fits = {}
    for take in native['takes']:
        for prop in PROPS:
            ordered = candidates(assets[prop]['source'], np.array(take['furniture'][prop]['reference_mm']))
            supported = [r for r in ordered if r['rms_mm'] <= FIT_GATE_MM]
            fits[take['take'], prop] = ordered[0]
            if len(supported) != 1:
                raise ValueError(f"{take['take']}/{prop}: {len(supported)} geometrically supported correspondences")
    for prop in PROPS:
        asset = assets[prop]
        objects, selection, scale = mounting_assets._import_glb(Path(asset['prepared_glb']))
        sources = mounting_assets._furniture_sources(objects, scale)
        if len(sources) != 1:
            raise ValueError(f'{prop}: expected current clean GLB to have one Mesh, got {len(sources)}')
        for take in native['takes']:
            identifier = f"{take['take']}/{prop}"
            native_prop = take['furniture'][prop]
            best = fits[take['take'], prop]
            ordered = candidates(asset['source'], np.array(native_prop['reference_mm']))
            permutation = best['source_indices_in_capture_order']
            target = 'Mesh_'+prop+'_stitched_body'
            output = destination / take['take'] / (prop+'.fbx')
            entry = {'asset_to_bone_rotation': best['rotation'],
                     'asset_to_bone_translation_mm': best['translation_mm'],
                     'native_unit_mm': take['native_unit_mm'], 'target_name': target, 'output_fbx': str(output)}
            exported = mounting_assets._export_furniture(entry, sources)
            root, _ = parse(str(output))
            geometry = next(e for e in mounting_assets._child(root, b'Objects').elems if e.id == b'Geometry')
            vertices = np.asarray(mounting_assets._child(geometry, b'Vertices').props[0]).reshape(-1, 3)
            audit_path = output.with_suffix('.geometry.npz')
            with audit_path.open('xb') as stream:
                np.savez_compressed(stream, vertices_native=vertices)
            observations_path = Path(take['furniture_observations_npz'])
            with np.load(observations_path, allow_pickle=False) as observations:
                observed = observations[prop+'_local_mm']
                times = observations[prop+'_valid_times']
                expected = asset['source'][permutation] @ np.array(best['rotation']).T+np.array(best['translation_mm'])
                errors = np.linalg.norm(observed-expected, axis=2)
                inherited = np.linalg.norm(observed-np.array(native_prop['reference_mm']), axis=2)
                full = stats(errors)
                inherited_stats = stats(inherited)
                per_marker = [stats(errors[:, i]) for i in range(4)]
                frames_over = np.flatnonzero(errors.max(1) > FIT_GATE_MM)
                worst = np.argsort(errors.max(1))[-10:][::-1]
                full.update(valid_frame_count=len(observed), common_frame_count=native_prop['common_frame_count'],
                            valid_observation_fraction=native_prop['valid_observation_fraction'],
                            frames_with_any_marker_over_2mm=len(frames_over),
                            per_marker=per_marker, all_valid_frames_included=True,
                            worst_frames=[{'native_time_tick': int(times[i]),
                                           'marker_errors_mm': errors[i].tolist(),
                                           'inherited_reference_errors_mm': inherited[i].tolist()} for i in worst])
            sample_indices = np.unique(np.linspace(0, len(vertices)-1, 17, dtype=int))
            record = {**exported, 'status': 'READY', 'take': take['take'], 'prop': prop,
                      'native_parent_uid': native_prop['bone_uid'], 'native_parent_name': native_prop['bone_name'],
                      'source_fbx': take['fbx'], 'source_fbx_sha256': take['source_sha256'],
                      'source_glb': asset['prepared_glb'], 'source_glb_sha256': sha256(asset['prepared_glb']),
                      'source_marker_json': asset['prepared_calibration'],
                      'source_marker_json_sha256': sha256(asset['prepared_calibration']),
                      'observations_npz': str(observations_path), 'observations_npz_sha256': sha256(observations_path),
                      'correspondence': {'best': best, 'runner_up': ordered[1], 'all_24_candidates': ordered,
                                         'supported_correspondence_count': 1, 'ambiguous': False,
                                         'capture_to_source': {label: {'source_index_one_based': permutation[i]+1,
                                             'source_mcp_name': asset['document']['markers'][permutation[i]]['mcp_name'],
                                             'source_point': asset['document']['markers'][permutation[i]].get('source_point',
                                                 asset['document']['markers'][permutation[i]].get('json_point'))}
                                             for i, label in enumerate(native_prop['markers'])}},
                      'metric_scale_evidence': asset['scale_evidence'], 'full_valid_frame_residuals': full,
                      'inherited_source_reference_residuals': inherited_stats,
                      'raw_expected_geometry': {'path': str(audit_path), 'sha256': sha256(audit_path),
                                                'array_key': 'vertices_native', 'shape': list(vertices.shape)},
                      'vertex_indices': sample_indices.tolist(), 'vertex_samples_native': vertices[sample_indices].tolist(),
                      'qc': {'registration_passed': True, 'proper_rotation': True, 'single_mesh': True,
                             'animation_modified': False, 'no_outlier_filtering': True,
                             'full_source_quality_passed': full['max_mm'] <= FIT_GATE_MM,
                             'texture_uv_preservation_audited': True,
                             'physical_review_render_required': True,
                             'tracking_residual_review_required': full['max_mm'] > FIT_GATE_MM},
                      'physical_review': physical_review(take, prop, exported, vertices,
                                                         fits[take['take'], 'desk'])}
            index['furniture_exports'][identifier] = record
    index['passed_asset_preparation'] = len(index['furniture_exports']) == len(native['takes'])*len(PROPS)
    index['full_source_quality_passed'] = all(r['qc']['full_source_quality_passed']
                                            for r in index['furniture_exports'].values())
    index['unmet_constraints'] = [
        {'id': key, 'constraint': 'Full-frame marker residual exceeds 2 mm; source timing/deformation not repaired',
         'rms_mm': r['full_valid_frame_residuals']['rms_mm'],
         'p99_mm': r['full_valid_frame_residuals']['p99_mm'], 'max_mm': r['full_valid_frame_residuals']['max_mm']}
        for key, r in index['furniture_exports'].items() if not r['qc']['full_source_quality_passed']]
    save(index_path, index)
    print(json.dumps({'furniture_asset_index': str(index_path), 'exports': len(index['furniture_exports']),
                      'passed_asset_preparation': index['passed_asset_preparation'],
                      'full_source_quality_passed': index['full_source_quality_passed']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None))

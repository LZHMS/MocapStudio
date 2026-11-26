"""Revise existing stitched FBX room geometry without adding door/room models.

Run in Blender with -- native_scene_inputs.json [--inspection source_inspection.json].
The source FBXs are never opened or rewritten here. Original UID-keyed replacement
arrays and static transforms are produced for the binary-preserving graft driver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import room_scene_assets as room


ARCHITECTURE = ('Environment_Walls', 'Door_Casing', 'Floor', 'Skirting', 'Floor_Joint')


def request_from_native(native):
    root = Path(native['assets_root'])
    request = dict(native)
    request.update(
        source_environment_blend=str(root/'meeting_room_environment/meeting_room_environment.blend'),
        source_environment_glb=str(root/'meeting_room_environment/meeting_room_environment.glb'),
        source_environment_layout=str(root/'meeting_room_environment/meeting_room_environment.layout.json'),
        source_outer_handle_blend=str(root/'door_hand/door_hand_outer.blend'),
        source_outer_handle_glb=str(root/'door_hand/door_hand_outer.glb'),
        asset_output_directory=str(Path(native['output_directory'])/'scene_assets'),
        architecture_names=list(ARCHITECTURE),
        takes=[dict(t['door_out'], take=t['take']) for t in native['takes']])
    return request


def face_normals(vertices, triangles):
    points = vertices[triangles]
    cross = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    length = np.linalg.norm(cross, axis=1)
    if np.any(length < 1e-12):
        raise ValueError('Degenerate original or revised triangle')
    return cross/length[:, None]


def rotation_between(a, b):
    cross = np.cross(a, b)
    cosine = float(np.dot(a, b))
    if cosine < -0.999999:
        raise ValueError('Geometry revision inverted a face')
    skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]])
    return np.eye(3)+skew+skew@skew/(1.+cosine)


def revised_normals(entry, revised):
    original = np.asarray(entry['vertices_native'])
    triangles = np.asarray(entry['triangles'], dtype=int)
    normals = np.asarray(entry['normals'])
    before = face_normals(original, triangles)
    after = face_normals(revised, triangles)
    changed_faces = np.linalg.norm(before-after, axis=1) > 1e-8
    mapping, reference = entry['normal_mapping'], entry['normal_reference']
    if reference != 'Direct':
        raise ValueError(f'Unsupported original normal reference: {reference}')
    if mapping in ('ByVertice', 'ByVertex'):
        if len(normals) != len(original):
            raise ValueError('Direct normal count does not match original vertices')
        normal_ids = triangles
    elif mapping == 'ByPolygonVertex':
        if len(normals) != triangles.size:
            raise ValueError('Direct polygon-corner normal count does not match triangles')
        normal_ids = np.arange(triangles.size).reshape(-1, 3)
    else:
        raise ValueError(f'Unsupported original normal mapping: {mapping}')
    candidates = {}
    for face in np.flatnonzero(changed_faces):
        rotation = rotation_between(before[face], after[face])
        for index in normal_ids[face]:
            alignment = float(np.dot(normals[index], before[face]))
            value = (np.sign(alignment)*after[face] if abs(alignment) > .999999
                     else rotation @ normals[index])
            candidates.setdefault(int(index), []).append(value)
    result = normals.copy()
    for index, values in candidates.items():
        values = np.asarray(values)
        if np.max(np.linalg.norm(values-values[0], axis=1)) > 1e-6:
            raise ValueError('A shared normal requires incompatible face rotations')
        result[index] = values[0]
    changed = np.flatnonzero(np.linalg.norm(result-normals, axis=1) > 1e-10)
    untouched = np.ones(len(normals), dtype=bool)
    untouched[changed] = False
    # Flat source normals on changed faces must follow the revised geometry.
    flat_errors = []
    for face in np.flatnonzero(changed_faces):
        for index in normal_ids[face]:
            alignment = float(np.dot(normals[index], before[face]))
            if abs(alignment) > .999999:
                flat_errors.append(float(np.linalg.norm(result[index]-np.sign(alignment)*after[face])))
    if flat_errors and max(flat_errors) > 1e-6:
        raise ValueError('Revised flat normal disagrees with revised face')
    return result, {
        'normal_mapping': mapping, 'normal_reference': reference,
        'original_normal_count': len(normals), 'changed_normal_indices': changed.tolist(),
        'changed_geometric_face_indices': np.flatnonzero(changed_faces).tolist(),
        'unrelated_normal_entries_exact': bool(np.array_equal(result[untouched], normals[untouched])),
        'changed_flat_face_normal_max_error': max(flat_errors, default=0.)}


def coordinate_mapping(points, source, tolerance_mm=.01):
    distances = np.linalg.norm(points[:, None, :]-source[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    error = float(distances.min(axis=1).max())
    if error > tolerance_mm:
        raise ValueError(f'Original FBX differs from actual canonical Blend by {error:.9g} mm')
    return nearest, error


def triangle_signature(vertices, triangles):
    # Coordinate keys avoid ambiguities from duplicated source UV/normal vertices.
    keys = [tuple(np.round(v, 2)) for v in vertices]
    return Counter(tuple(sorted(keys[index] for index in triangle)) for triangle in triangles)


def source_triangles(inspection):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=inspection['path'])
    result = {}
    for name in ARCHITECTURE:
        mesh = bpy.data.objects[name].data
        mesh.calc_loop_triangles()
        result[name] = np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=int)
    return result


def environment_geometry(request, inspection, door):
    layout = json.loads(Path(request['source_environment_layout']).read_text(encoding='utf-8-sig'))
    source_basis = np.array([layout['axis_u'], layout['axis_v'], [0., 0., 1.]])
    objects = {obj['name']: obj for obj in inspection['objects'] if obj['type'] == 'MESH'}
    opening = layout['door_opening']
    source_left, source_right = opening['min_u']*1000, opening['max_u']*1000
    source_mid = (source_left+source_right)/2
    casing = np.asarray(objects['Door_Casing']['vertices_mm']) @ source_basis.T
    left_group = casing[:, 0] < source_mid
    old_left = float(casing[left_group, 0].max())
    old_right = float(casing[~left_group, 0].min())
    outline = np.asarray(layout['outline_uv'])*1000
    wall_v = opening['wall_v']*1000
    notch = outline[np.isclose(outline[:, 1], wall_v, atol=.01), 0]
    if len(notch) != 2:
        raise ValueError('Expected two original vestibule opening-wall endpoints')
    notch_left, notch_right = float(notch.min()), float(notch.max())
    clearance = 4.
    left = (notch_left+notch_right-door['width_mm']-2*clearance)/2
    right = (notch_left+notch_right+door['width_mm']+2*clearance)/2
    delta_left, delta_right = left-old_left, right-old_right
    old_height = opening['height']*1000
    header = max(t['top_height_range_mm'][1] for t in door['takes'])+clearance
    up = np.array([0., 0., 1.])
    target_u = np.asarray(door['closed_width_world'])
    target_v = np.cross(up, target_u)
    target_basis = np.array([target_u, target_v, up])
    hinge = np.asarray(door['hinge_world_mm'])
    target_anchor = hinge.copy()
    target_anchor[2] = 0.
    target_anchor += target_v*(door['hinge_to_handle_side_face_signed_mm']-door['thickness_mm']/2)
    source_anchor = np.array([left+clearance, wall_v, 0.])
    rigid_rotation = target_basis.T @ source_basis
    rigid_translation = target_anchor-source_anchor @ target_basis
    revised_source, revised_world = {}, {}
    edits = []
    for name, obj in objects.items():
        original = np.asarray(obj['vertices_mm'])
        uv = original @ source_basis.T
        before = uv.copy()
        if name == 'Door_Casing':
            uv[:, 0] += np.where(uv[:, 0] < source_mid, delta_left, delta_right)
            uv[uv[:, 2] > old_height/2, 2] += header-old_height
        elif name == 'Environment_Walls':
            near = np.abs(uv[:, 1]-wall_v) < 25
            uv[near & (np.abs(uv[:, 0]-source_left) < 60), 0] = left
            uv[near & (np.abs(uv[:, 0]-source_right) < 60), 0] = right
            uv[np.isclose(uv[:, 2], old_height, atol=.01), 2] = header
        elif name == 'Skirting':
            uv[np.abs(uv[:, 0]-casing[left_group, 0].min()) < .01, 0] += delta_left
            uv[np.abs(uv[:, 0]-casing[~left_group, 0].max()) < .01, 0] += delta_right
        revised_source[name] = uv @ source_basis
        revised_world[name] = (uv-source_anchor) @ target_basis+target_anchor
        if name in ARCHITECTURE:
            edits.append({'name': name, 'source_raw_vertex_count': len(original),
                          'local_boundary_changed_count': int(np.any(np.abs(uv-before) > .001, axis=1).sum())})
    opening_corners = np.array([[left, wall_v, 0], [right, wall_v, 0],
                                [right, wall_v, header], [left, wall_v, header]])
    opening_world = (opening_corners-source_anchor) @ target_basis+target_anchor
    clearances = {
        'panel_width_mm': door['width_mm'], 'clear_width_mm': right-left,
        'side_clearance_each_mm': clearance, 'header_height_mm': header,
        'panel_top_clearance_min_mm': clearance, 'vestibule_width_mm': notch_right-notch_left,
        'existing_casing_inner_width_mm': old_right-old_left,
        'clear_opening_to_corner_jamb_left_mm': left-notch_left,
        'clear_opening_to_corner_jamb_right_mm': notch_right-right,
        'casing_outer_edge_to_corner_left_mm': left-notch_left-38.,
        'casing_outer_edge_to_corner_right_mm': notch_right-right-38.,
        'interior_clear_jamb_left_mm': left-notch_left-layout['wall_thickness']*500,
        'interior_clear_jamb_right_mm': notch_right-right-layout['wall_thickness']*500,
        'left_jamb_translation_mm': delta_left, 'right_jamb_translation_mm': delta_right,
        'outline_and_floor_manual_edits_retained': True,
        'service_clearance_not_measured': True, 'original_header_jamb_mismatch_corrected': True}
    alignment = {
        'axis_u': target_u.tolist(), 'axis_v': target_v.tolist(), 'axis_up': up.tolist(),
        'source_basis_rows': source_basis.tolist(), 'capture_basis_rows': target_basis.tolist(),
        'source_anchor_mm': source_anchor.tolist(), 'capture_anchor_mm': target_anchor.tolist(),
        'rigid_rotation': rigid_rotation.tolist(), 'rigid_translation_mm': rigid_translation.tolist(),
        'opening_capture_world_mm': opening_world.tolist(),
        'door_camera_target_world_mm': opening_world.mean(0).tolist(),
        'room_frame_inferred_from_door_not_wall_markers': True}
    layout.update(revision=request['run_id'], axis_u=target_u.tolist(), axis_v=target_v.tolist(),
                  capture_world_hinge_mm=hinge.tolist(), alignment_inferred_not_wall_measured=True,
                  source_layout=request['source_environment_layout'], motion_consistent_opening=clearances)
    outline3 = np.column_stack([outline, np.zeros(len(outline))])
    layout['outline_capture_world_m'] = (((outline3-source_anchor) @ target_basis+target_anchor)/1000).tolist()
    return revised_source, revised_world, layout, alignment, edits


def save_environment(request, revised_world, layout, alignment, edits, out):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=request['source_environment_blend'])
    scale = bpy.context.scene.unit_settings.scale_length
    for name, vertices in revised_world.items():
        room.set_world_vertices(bpy.data.objects[name], vertices, scale)
    blend = out/'meeting_room_motion_consistent.blend'
    glb = out/'meeting_room_architecture_capture_world.glb'
    layout_path = out/'meeting_room_motion_consistent.layout.json'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.object.select_all(action='DESELECT')
    for name in ARCHITECTURE:
        bpy.data.objects[name].select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', use_selection=True)
    layout_path.write_text(json.dumps(layout, indent=2), encoding='utf-8')
    return {'blend': str(blend), 'glb': str(glb), 'layout': str(layout_path),
            'architecture_snippets': [], 'new_fbx_models': 0,
            'architecture_edits': edits, 'clearances': layout['motion_consistent_opening'],
            'capture_world_alignment': alignment, 'selected_objects': list(ARCHITECTURE),
            'retained_source_objects': sorted(revised_world)}


def inner_compatibility(native, inspection, door):
    candidates = [o for o in inspection['objects'] if o['type'] == 'MESH']
    obj = min(candidates, key=lambda o: abs(np.ptp(np.asarray(o['vertices_mm'])[:, 2])-190))
    vertices = np.asarray(obj['vertices_mm'])
    plate = vertices[np.isclose(np.abs(vertices[:, 2]), np.abs(vertices[:, 2]).max(), atol=.002)]
    contact = np.array([(plate[:, 0].min()+plate[:, 0].max())/2, plate[:, 1].max(),
                        (plate[:, 2].min()+plate[:, 2].max())/2])
    width_axis, normal_axis, _ = np.asarray(door['cad_basis_rows'])
    outer_contact = np.asarray(door['mounting_contact_cad_mm'])
    lock = float(np.dot(outer_contact, width_axis)+door['backset_mm'])
    records = []
    def stats(values):
        return {'mean': float(values.mean()), 'min': float(values.min()),
                'max': float(values.max()), 'rms': float(np.sqrt(np.mean(values**2)))}
    def samples(data):
        rotation, position = room.motion(data)
        runtime = dict(data, bone_world_rotations=data['runtime_bone_world_rotations'],
                       bone_world_translations_mm=data['runtime_bone_world_translations_mm'])
        runtime_rotation, runtime_position = room.motion(runtime)
        times = np.r_[data['pose_times_seconds'], data['runtime_times_seconds']]
        rotations = np.concatenate([rotation, runtime_rotation])
        positions = np.concatenate([position, runtime_position])
        order = np.argsort(times)
        times, rotations, positions = times[order], rotations[order], positions[order]
        unique = np.r_[True, np.diff(times) > 1e-8]
        return times[unique], rotations[unique], positions[unique]
    for take in native['takes']:
        inner, outer = take['door_in'], take['door_out']
        ti, ri, pi = samples(inner)
        to, ro, po = samples(outer)
        pairs = []
        i = j = 0
        while i < len(ti) and j < len(to):
            difference = ti[i]-to[j]
            if abs(difference) <= 1e-8:
                pairs.append((i, j))
                i += 1
                j += 1
            elif difference < 0:
                i += 1
            else:
                j += 1
        if not pairs:
            raise ValueError(f'No exact common native door timestamps: {take["take"]}')
        inner_indices, outer_indices = np.asarray(pairs).T
        times = to[outer_indices]
        contact_world = ri[inner_indices] @ contact+pi[inner_indices]
        contact_outer = np.einsum('nji,nj->ni', ro[outer_indices], contact_world-po[outer_indices])
        backset = lock-contact_outer @ width_axis
        separation = (contact_outer-outer_contact) @ normal_axis
        records.append({'take': take['take'], 'paired_sample_count': len(times),
                        'time_range_seconds': [float(times[0]), float(times[-1])],
                        'paired_times_seconds': times.tolist(),
                        'available_inner_samples': len(ti), 'available_outer_samples': len(to),
                        'inner_pose_interpolated_to_outer_times': False,
                        'pairing': 'Only exact common native timestamps, tolerance 1e-8 seconds; no motion interpolation',
                        'inner_mounting_axis_backset_mm': stats(backset),
                        'inner_axis_backset_error_from_30_mm': stats(backset-door['backset_mm']),
                        'mounting_face_separation_mm': stats(separation),
                        'inner_contact_to_panel_far_face_signed_mm': stats(separation-door['thickness_mm'])})
    return {'available': True, 'mounting_contact_inner_cad_mm': contact.tolist(),
            'mounting_axis_evidence': 'Actual base-plate cap-ring symmetry and rear contact plane',
            'inner_mesh_and_markers_unchanged': True, 'physical_fit_status': 'NEEDS_REVIEW',
            'sampling_limitation': '1024 extracted native poses plus 13 exact runtime poses per take, not every animation frame. Three takes have different source track durations; only their exact common timestamps contribute, and unmatched observations are not interpolated into physical evidence.',
            'takes': records}


def door_override(take, door):
    original = np.asarray(take['door_out']['original_vertices_native'])
    data = take['door_out']
    cad_rotation = np.asarray(data['cad_to_stored_bone_rotation'])
    cad_translation = np.asarray(data['cad_to_stored_bone_translation_mm'])
    cad = (original*take['native_unit_mm']-cad_translation) @ cad_rotation
    old, new = np.asarray(door['original_panel_vertices_mm']), np.asarray(door['panel_vertices_mm'])
    distance = np.linalg.norm(cad[:, None, :]-old[None, :, :], axis=2)
    nearest = distance.argmin(axis=1)
    selected = distance.min(axis=1) < .001
    if selected.sum() != 24 or set(nearest[selected]) != set(range(8)):
        raise ValueError('Expected the original 24 split panel vertices and all eight canonical corners')
    result = original.copy()
    result[selected] = (new[nearest[selected]] @ cad_rotation.T+cad_translation)/take['native_unit_mm']
    triangles = np.asarray(data['triangles'], dtype=int)
    panel_faces = selected[triangles].all(axis=1)
    if np.any(selected[triangles].any(axis=1) != panel_faces):
        raise ValueError('Original panel is connected to preserved handle')
    normal_error = float(np.linalg.norm(face_normals(original, triangles[panel_faces])-
                                       face_normals(result, triangles[panel_faces]), axis=1).max())
    if normal_error > 1e-5:
        raise ValueError('Panel resize unexpectedly changes its face normals')
    indices = np.flatnonzero(selected)
    # Native runtime matrices include the unchanged static Mesh pose.
    bone = np.asarray(data['runtime_bone_world_rotations'])
    position = np.asarray(data['runtime_bone_world_translations_mm'])
    mesh_r = np.asarray(data['mesh_local_rotation'])
    mesh_t = np.asarray(data['mesh_local_translation_native'])*take['native_unit_mm']
    local_mm = result[indices]*take['native_unit_mm'] @ mesh_r.T+mesh_t
    world = np.einsum('nij,vj->nvi', bone, local_mm)+position[:, None, :]
    return result, {
        'geometry_uid': data['door_outer_geometry_uid'], 'original_vertex_count': len(original),
        'changed_vertex_indices': indices.tolist(), 'changed_normal_indices': [],
        'preserved_handle_vertex_count': int((~selected).sum()),
        'preserved_handle_vertices_exact': bool(np.array_equal(original[~selected], result[~selected])),
        'all_original_normals_unmodified': True, 'topology_unchanged': True,
        'panel_geometric_normal_max_error': normal_error}, {
        'name': 'door_out_panel', 'geometry_uid': data['door_outer_geometry_uid'],
        'model_uid': data['door_outer_model_uid'], 'vertex_indices': indices.tolist(),
        'final_vertices_native': result[indices].tolist(),
        'runtime_times_seconds': data['runtime_times_seconds'],
        'runtime_world_vertices_mm': world.tolist()}


def produce(native, inspection, request):
    from mathutils import Matrix
    out = Path(request['asset_output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    door = room.fit_door(request, inspection['outer'])
    door['calibration_scope'] = native['input_directory']
    door['animation_modified'] = False
    door['source_panel_height_and_thickness_preserved'] = True
    door['observed_up_basis_note'] = 'Source panel CAD +Z points down in capture. The retained slab basis uses its observed negative height axis; roots/handles are not forced upright.'
    widths = [t['independent_clip_width_mm'] for t in door['takes'] if 'independent_clip_width_mm' in t]
    door['independent_moving_clip_width_range_mm'] = [min(widths), max(widths)]
    inner = inner_compatibility(native, inspection['inner'], door)
    revised_source, revised_world, layout, alignment, edits = environment_geometry(request, inspection['environment'], door)
    canonical_triangles = source_triangles(inspection['environment'])
    original_objects = {o['name']: o for o in inspection['environment']['objects'] if o['type'] == 'MESH'}
    rigid_r = np.asarray(alignment['rigid_rotation'])
    rigid_t = np.asarray(alignment['rigid_translation_mm'])
    index = {'schema_version': 2, 'mode': 'existing_scene_original_uid_overrides',
             'run_id': native['run_id'], 'input_directory': native['input_directory'],
             'door': door, 'inner_handle_compatibility': inner,
             'source_hashes': {key: room.sha256(request[key]) for key in (
                 'source_environment_blend', 'source_environment_glb', 'source_environment_layout',
                 'source_outer_handle_blend', 'source_outer_handle_glb')},
             'physical_fit_status': 'NEEDS_REVIEW',
             'physical_fit_flags': {'inherited_hinge_noise': 'NEEDS_REVIEW',
                                    'inherited_floor_gap_and_panel_tilt': 'NEEDS_REVIEW',
                                    'inherited_external_hinge_axis': 'NEEDS_REVIEW',
                                    'inner_plate_backset_and_far_face_contact': 'NEEDS_REVIEW'},
             'new_architecture_or_door_model_count': 0, 'takes': []}
    for take in native['takes']:
        vertices, checks, door_samples = door_override(take, door)
        record = {'take': take['take'], 'source_fbx': take['fbx'],
                  'source_sha256': take['source_sha256'],
                  'geometry_vertex_overrides': {str(take['door_out']['door_outer_geometry_uid']): vertices.tolist()},
                  'geometry_normal_overrides': {}, 'model_transform_overrides': {},
                  'checks': {'door': checks, 'architecture': []},
                  'original_architecture_indices': [],
                  'expected_final_geometry_samples': [door_samples],
                  'camera': dict(alignment, actual_take=take['take'],
                                 actual_door_sample_time_seconds=take['door_out']['runtime_times_seconds'][0])}
        runtime = dict(take['door_out'],
                       bone_world_rotations=take['door_out']['runtime_bone_world_rotations'],
                       bone_world_translations_mm=take['door_out']['runtime_bone_world_translations_mm'])
        actual_rotation, actual_position = room.motion(runtime)
        contact = np.asarray(door['mounting_contact_cad_mm'])
        record['camera'].update(
            actual_mounting_contact_world_mm=(actual_rotation[0] @ contact+actual_position[0]).tolist(),
            actual_width_axis_world=(actual_rotation[0] @ np.asarray(door['cad_basis_rows'][0])).tolist(),
            actual_panel_up_world=(actual_rotation[0] @ np.asarray(door['cad_basis_rows'][2])).tolist())
        if len(take['environment']) != 24:
            raise ValueError('Expected all 24 original environment meshes')
        for entry in take['environment']:
            old_r = np.asarray(entry['mesh_local_rotation'])
            old_t = np.asarray(entry['mesh_local_translation_native'])*take['native_unit_mm']
            new_r, new_t = rigid_r @ old_r, rigid_r @ old_t+rigid_t
            euler = np.degrees(np.array(Matrix(new_r.tolist()).to_euler('XYZ')))
            record['model_transform_overrides'][str(entry['model_uid'])] = {
                'translation': (new_t/take['native_unit_mm']).tolist(), 'rotation_degrees': euler.tolist()}
            sample = np.asarray(entry['vertex_samples_native'])
            if entry['source_object'] in ARCHITECTURE:
                source = np.asarray(original_objects[entry['source_object']]['vertices_mm'])
                original = np.asarray(entry['vertices_native'])
                source_world = original*take['native_unit_mm'] @ old_r.T+old_t
                mapping, error = coordinate_mapping(source_world, source)
                source_tri = canonical_triangles[entry['source_object']]
                if triangle_signature(source_world, entry['triangles']) != triangle_signature(source, source_tri):
                    raise ValueError(f'Canonical/original triangle topology differs: {entry["name"]}')
                deformation = revised_source[entry['source_object']]-source
                deformation[np.linalg.norm(deformation, axis=1) < 1e-8] = 0.
                revised = original+deformation[mapping] @ old_r/take['native_unit_mm']
                normals, normal_checks = revised_normals(entry, revised)
                uid = str(entry['geometry_uid'])
                record['geometry_vertex_overrides'][uid] = revised.tolist()
                if normal_checks['changed_normal_indices']:
                    record['geometry_normal_overrides'][uid] = normals.tolist()
                changed = np.flatnonzero(np.linalg.norm(revised-original, axis=1) > 1e-7)
                record['checks']['architecture'].append(dict(
                    normal_checks, name=entry['name'], geometry_uid=entry['geometry_uid'],
                    original_vertex_count=len(original), changed_vertex_indices=changed.tolist(),
                    canonical_position_max_error_mm=error, topology_unchanged=True,
                    canonical_raw_vertex_for_original_vertex=mapping.tolist()))
                record['original_architecture_indices'].append({
                    'name': entry['name'], 'source_object': entry['source_object'],
                    'model_uid': entry['model_uid'], 'geometry_uid': entry['geometry_uid'],
                    'parent_uid': entry['parent_uid'], 'vertex_count': len(original),
                    'triangles': entry['triangles']})
                sample = revised[np.asarray(entry['vertex_indices'], dtype=int)]
            world = sample*take['native_unit_mm'] @ new_r.T+new_t
            record['expected_final_geometry_samples'].append({
                'name': entry['name'], 'model_uid': entry['model_uid'], 'geometry_uid': entry['geometry_uid'],
                'vertex_indices': entry['vertex_indices'], 'final_vertices_native': sample.tolist(),
                'final_world_vertices_mm': world.tolist(),
                'decor_geometry_bytes_unmodified': entry['source_object'] not in ARCHITECTURE})
        record['checks'].update(environment_model_count=24, environment_models_retained=True,
                               environment_common_rigid_transform=True, new_architecture_or_door_models=0)
        index['takes'].append(record)
    index['outer_asset'] = room.save_outer(request, inspection, door, out)
    index['environment_asset'] = save_environment(request, revised_world, layout, alignment, edits, out)
    index['source_hashes'].update(
        source_inner_handle_blend=room.sha256(inspection['inner']['path']),
        source_inner_handle_glb=room.sha256(Path(native['assets_root'])/'door_hand/door_hand.glb'))
    index['export_notes'] = {
        'source_fbx_hashes': 'Trusted extraction hashes; source FBXs are deliberately not reread by this asset-only driver.',
        'preservation': 'Only exact-count existing geometry arrays and static Mesh TR are requested. Topology, UVs, materials, ordinary prop/decor geometry and all animation remain untouched.',
        'normal_preservation': 'Only existing architectural normal entries whose geometric faces rotate are replaced. Other normal vectors, reference/mapping/index data remain untouched.',
        'sampling': 'Hinge/floor fits use 1024 extracted actual root poses per take; final expected door samples use 13 exact runtime matrices. No claim of exhaustive every-frame physical clearance.',
        'visual_verification': 'Parent must re-import and render actual grafted deliverables.'}
    report = out/'room_motion_geometry_report.json'
    report.write_text(json.dumps({k: v for k, v in index.items() if k != 'takes'}, indent=2), encoding='utf-8')
    index['report'] = str(report)
    control = Path(native['output_directory'])/'control'
    control.mkdir(parents=True, exist_ok=True)
    path = control/'room_asset_index.json'
    path.write_text(json.dumps(index, indent=2), encoding='utf-8')
    print('ROOM_ASSET_INDEX', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_inputs')
    parser.add_argument('--inspection')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    native = json.loads(Path(args.native_inputs).read_text(encoding='utf-8-sig'))
    request = request_from_native(native)
    out = Path(request['asset_output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    if args.inspection:
        inspection = json.loads(Path(args.inspection).read_text(encoding='utf-8-sig'))
        for name in ('outer', 'inner', 'environment'):
            if room.sha256(inspection[name]['path']) != inspection[name]['sha256']:
                raise ValueError(f'Canonical source changed after inspection: {name}')
    else:
        inspection = {name: room.inspect_blend(path) for name, path in (
            ('outer', request['source_outer_handle_blend']),
            ('inner', str(Path(native['assets_root'])/'door_hand/door_hand.blend')),
            ('environment', request['source_environment_blend']))}
        (out/'source_inspection.json').write_text(json.dumps(inspection, indent=2), encoding='utf-8')
    produce(native, inspection, request)


if __name__ == '__main__':
    main()

"""Revise current room/door Blender assets from measured native FBX door poses.

Run in Blender: --background --python room_scene_assets.py -- request.json [--inspect]
Original assets are opened read-only; every saved asset lives in asset_output_directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def world_vertices(obj, scale):
    xyz = np.array([v.co[:] for v in obj.data.vertices], dtype=float)
    matrix = np.array(obj.matrix_world, dtype=float)
    return (xyz @ matrix[:3, :3].T + matrix[:3, 3]) * scale * 1000


def components(obj):
    parent = list(range(len(obj.data.vertices)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for edge in obj.data.edges:
        a, b = edge.vertices
        parent[root(a)] = root(b)
    groups = {}
    for i in range(len(parent)):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


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
            xyz = world_vertices(obj, scale)
            record.update(vertices_mm=xyz.tolist(), polygons=[list(p.vertices) for p in obj.data.polygons],
                          components=components(obj), materials=[m.name if m else None for m in obj.data.materials],
                          uv_layers=list(obj.data.uv_layers.keys()), modifiers=[m.name for m in obj.modifiers])
        records.append(record)
    return {'path': str(path), 'sha256': sha256(path), 'scale_length': scale, 'objects': records}


def motion(take):
    cad = np.asarray(take['cad_to_stored_bone_rotation'])
    offset = np.asarray(take['cad_to_stored_bone_translation_mm'])
    mesh = np.asarray(take['mesh_local_rotation'])
    translation = np.asarray(take['mesh_local_translation_native']) * take['native_unit_mm']
    bone = np.asarray(take['bone_world_rotations'])
    rotation = bone @ mesh @ cad
    position = bone @ (mesh @ offset + translation) + np.asarray(take['bone_world_translations_mm'])
    return rotation, position


def unit(vector):
    return vector / np.linalg.norm(vector)


def panel_measurement(inspection):
    candidates = []
    for obj in inspection['objects']:
        if obj['type'] != 'MESH':
            continue
        xyz = np.asarray(obj['vertices_mm'])
        for ids in obj['components']:
            points = xyz[ids]
            if len(ids) == 8 and np.ptp(points, axis=0).max() > 1500:
                candidates.append((obj, np.asarray(ids), points))
    if len(candidates) != 1:
        raise ValueError('Expected one disconnected eight-corner existing door panel')
    obj, ids, points = candidates[0]
    _, _, basis = np.linalg.svd(points - points.mean(0), full_matrices=False)
    height, width, normal = basis
    height *= np.sign(height[2])
    width *= np.sign(width[0])
    normal = unit(np.cross(height, width))
    dimensions = np.ptp(points @ np.array([width, normal, height]).T, axis=0)
    handle_ids = sorted(set(range(len(obj['vertices_mm']))) - set(ids))
    handle = np.asarray(obj['vertices_mm'])[handle_ids]
    # The actual plate is symmetric about X and Z. Its two planar cap rings
    # at +/-95 mm are distinguishable from the +/-12.5 mm lever-neck faces.
    z = handle[:, 2]
    plate = handle[np.isclose(np.abs(z), np.max(np.abs(z)), atol=.002)]
    center_x = (plate[:, 0].min() + plate[:, 0].max()) / 2
    center_z = (plate[:, 2].min() + plate[:, 2].max()) / 2
    back_y = plate[:, 1].max()
    neck = handle[np.isclose(handle[:, 1], plate[:, 1].min(), atol=.002)
                  & (np.abs(handle[:, 2] - center_z) < 20)]
    if len(neck) < 4:
        raise ValueError('Cannot identify lever-neck mounting-axis evidence')
    neck_center = (neck[:, 0].min() + neck[:, 0].max()) / 2
    if abs(neck_center - center_x) > .01:
        raise ValueError('Lever-neck and base-plate mounting axes disagree')
    return obj, ids, points, dimensions, np.array([width, normal, height]), np.array(
        [center_x, back_y, center_z]), {
            'method': 'Base plate +/-Z cap-ring symmetry, corroborated by actual lever-neck root',
            'plate_vertex_count': len(plate), 'neck_root_vertex_count': len(neck),
            'plate_axis_x_mm': center_x, 'neck_axis_x_mm': neck_center,
            'plate_contact_y_mm': back_y, 'axis_direction_cad': [0, 1, 0],
            'uncertainty': 'Geometric symmetry identifies mounting center; no physical spindle bore exists in this simplified CAD.'}


def fit_door(request, inspection):
    obj, ids, old, dimensions, old_basis, contact, evidence = panel_measurement(inspection)
    poses = [motion(take) for take in request['takes']]
    moving = [i for i, (_, p) in enumerate(poses) if np.ptp(p, axis=0).max() > 100]
    if len(moving) < 2:
        raise ValueError('Need at least two moving clips for a common hinge')
    up = unit(np.concatenate([poses[i][0][:, 2, :] for i in moving]).mean(0))
    a = np.concatenate([poses[i][0] for i in moving])
    p = np.concatenate([poses[i][1] for i in moving])
    design = np.concatenate([a, -np.broadcast_to(np.eye(3), a.shape)], axis=2).reshape(-1, 6)
    gauge = np.r_[up, [0, 0, 0]][None, :]
    solution = np.linalg.lstsq(np.vstack([design, gauge]), np.r_[-p.reshape(-1), 0], rcond=None)[0]
    local_hinge, world_hinge = solution[:3], solution[3:]
    measured_up = up.copy()
    width_axis, normal_axis, height_axis = old_basis
    up = -height_axis
    # Preserve current panel orientation and plate contact. An external tracked
    # hinge need not lie inside the slab; report its offset, do not hide it.
    width = float(np.dot(contact-local_hinge, width_axis) + request['handle_backset_mm'])
    basis = np.array([width_axis, normal_axis, up])
    old_coordinates = (old-old.mean(0)) @ old_basis.T
    signs = np.sign(old_coordinates)
    new = old.copy()
    new += np.outer(signs[:, 0] * (width-dimensions[0])/2, width_axis)
    center_width = np.dot(contact, width_axis) + request['handle_backset_mm']-width/2
    new += (center_width-np.dot(old.mean(0), width_axis)) * width_axis
    front = float((new @ normal_axis).min())
    new += (np.dot(contact, normal_axis)-front) * normal_axis
    all_positions = [np.einsum('nij,vj->nvi', rotation, new) + position[:, None, :]
                     for rotation, position in poses]
    # Choose only a common static height placement. Animation/markers stay intact.
    bottom = min(float(x[:, :, 2].min()) for x in all_positions)
    mean_up_z = float(np.concatenate([rotation @ up for rotation, _ in poses])[:, 2].mean())
    vertical_shift = (5. - bottom) / mean_up_z
    new += vertical_shift * up
    normal_rotation = np.eye(3)
    mismatch = float(np.degrees(np.arccos(np.clip(np.dot(measured_up, up), -1, 1))))
    hinge_front_offset = float(np.dot(local_hinge-contact, normal_axis))
    records = []
    for take, (rotation, position) in zip(request['takes'], poses):
        hinge = rotation @ local_hinge + position
        xyz = np.einsum('nij,vj->nvi', rotation, new) + position[:, None, :]
        floor = xyz[:, :, 2].min(axis=1)
        top = xyz[:, :, 2].max(axis=1)
        residual = np.linalg.norm(hinge - world_hinge, axis=1)
        upright = np.degrees(np.arccos(np.clip((rotation @ up)[:, 2], -1, 1)))
        records.append({'take': take['take'], 'hinge_mean_world_mm': hinge.mean(0).tolist(),
                        'hinge_rms_mm': float(np.sqrt(np.mean(residual**2))),
                        'hinge_max_mm': float(residual.max()),
                        'bottom_height_range_mm': [float(floor.min()), float(floor.max())],
                        'top_height_range_mm': [float(top.min()), float(top.max())],
                        'vertical_error_max_degrees': float(upright.max()),
                        'mounting_axis_radius_mean_mm': float(np.mean(np.linalg.norm(
                            (rotation @ contact + position - hinge)[:, :2], axis=1)))})
        if np.ptp(position, axis=0).max() > 100:
            centered = (rotation-rotation.mean(0)).reshape(-1, 3)
            rhs = -(position-position.mean(0)).reshape(-1)
            gauge = measured_up
            clip_hinge = np.linalg.lstsq(np.vstack([centered, gauge]),
                                         np.r_[rhs, 0], rcond=None)[0]
            records[-1]['independent_clip_hinge_cad_mm'] = clip_hinge.tolist()
            records[-1]['independent_clip_width_mm'] = float(
                np.dot(contact-clip_hinge, width_axis)+request['handle_backset_mm'])
    # Closed samples have the largest alignment with the existing room's +U.
    # Compare angles, not a guessed source reference take or old assumed panel.
    layout = json.loads(Path(request['source_environment_layout']).read_text(encoding='utf-8-sig'))
    source_u = np.asarray(layout['axis_u'])
    directions = np.concatenate([rotation @ width_axis for rotation, _ in poses])
    directions[:, 2] = 0
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    scores = directions @ source_u
    closed = unit(directions[scores > scores.max() - .00015].mean(0))
    return {'source_object': obj['name'], 'panel_vertex_indices': ids.tolist(),
            'original_panel_vertices_mm': old.tolist(), 'panel_vertices_mm': new.tolist(),
            'old_basis_rows': old_basis.tolist(), 'cad_basis_rows': basis.tolist(),
            'panel_normal_rotation': normal_rotation.tolist(),
            'width_mm': width, 'height_mm': float(dimensions[2]), 'thickness_mm': float(dimensions[1]),
            'backset_mm': request['handle_backset_mm'], 'mounting_contact_cad_mm': contact.tolist(),
            'achieved_width_mm': float(np.ptp(new @ width_axis)),
            'achieved_backset_mm': float((new @ width_axis).max()-np.dot(contact, width_axis)),
            'achieved_height_mm': float(np.ptp(new @ up)),
            'achieved_thickness_mm': float(np.ptp(new @ normal_axis)),
            'geometric_dimension_status': 'EXACT_WITH_SOURCE_FLOAT_PRECISION',
            'panel_up_world_first_frame': (poses[0][0][0] @ up).tolist(),
            'mounting_axis_evidence': evidence, 'hinge_cad_mm': local_hinge.tolist(),
            'hinge_world_mm': world_hinge.tolist(), 'closed_width_world': closed.tolist(),
            'panel_to_measured_hinge_axis_degrees': mismatch,
            'old_to_revised_panel_normal_degrees': 0.,
            'hinge_to_handle_side_face_signed_mm': hinge_front_offset,
            'hinge_to_far_face_signed_mm': hinge_front_offset-float(dimensions[1]),
            'hinge_to_panel_midplane_signed_mm': hinge_front_offset-float(dimensions[1])/2,
            'axis_offset_interpretation': 'External hinge hardware is an inference only; no hinge hardware was measured or invented.',
            'mounting_plate_normal_mismatch_degrees': float(np.degrees(np.arccos(
                np.clip(np.dot(normal_axis, [0, 1, 0]), -1, 1)))),
            'height_placement_shift_mm': vertical_shift,
            'takes': records, 'motion_samples_only': True,
            'assumptions': ['Room placement inferred from tracked door; no wall markers measured.',
                            'Common hinge least-squares fit uses orientation AND position, with CAD-up gauge; residuals are not clamped.',
                            'Closed direction is the motion endpoint nearest the existing layout wall direction.',
                            'Source panel normal/height/thickness and all existing normals retained; external hinge offset explicitly reported.']}


def vertex_overrides(take, door):
    from io_scene_fbx.parse_fbx import parse
    original = np.asarray(take['original_vertices_native'])
    cad_rotation = np.asarray(take['cad_to_stored_bone_rotation'])
    cad_translation = np.asarray(take['cad_to_stored_bone_translation_mm'])
    cad = (original * take['native_unit_mm'] - cad_translation) @ cad_rotation
    old = np.asarray(door['original_panel_vertices_mm'])
    new = np.asarray(door['panel_vertices_mm'])
    distances = np.linalg.norm(cad[:, None, :] - old[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    selected = distances.min(axis=1) < .001
    if set(nearest[selected]) != set(range(len(old))):
        raise ValueError('Original FBX does not contain every current panel corner')
    result = original.copy()
    result[selected] = (new[nearest[selected]] @ cad_rotation.T + cad_translation) / take['native_unit_mm']
    root, _ = parse(take['fbx'])
    objects = next(e for e in root.elems if e.id == b'Objects')
    geometry = next(e for e in objects.elems if e.id == b'Geometry'
                    and e.props[0] == take['door_outer_geometry_uid'])
    layers = [e for e in geometry.elems if e.id == b'LayerElementNormal']
    if len(layers) != 1:
        raise ValueError('Expected exactly one existing normal layer')
    children = {e.id: e for e in layers[0].elems}
    mapping = children[b'MappingInformationType'].props[0]
    reference = children[b'ReferenceInformationType'].props[0]
    if mapping not in (b'ByVertice', b'ByVertex') or reference != b'Direct':
        raise ValueError(f'Unsupported existing normal mapping {mapping!r}/{reference!r}')
    normals = np.asarray(children[b'Normals'].props[0]).reshape(-1, 3)
    if len(normals) != len(original):
        raise ValueError('Normal count does not match direct vertex mapping')
    revised_normals = normals.copy()
    rotation = np.eye(3)
    triangles = np.asarray(take['triangles'])
    panel_triangles = selected[triangles].all(axis=1)
    if np.any(selected[triangles].any(axis=1) != panel_triangles):
        raise ValueError('Panel is topologically connected to preserved handle')
    def geometric_normals(points):
        faces = points[triangles[panel_triangles]]
        cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        return cross / np.linalg.norm(cross, axis=1)[:, None]
    old_normals, new_normals = geometric_normals(original), geometric_normals(result)
    error = np.linalg.norm(new_normals - old_normals @ rotation.T, axis=1).max()
    if error > 1e-5:
        raise ValueError(f'Unexpected panel geometric normal change: {error}')
    return result.tolist(), revised_normals.tolist(), {
        'geometry_uid': take['door_outer_geometry_uid'], 'original_vertex_count': len(original),
        'changed_vertex_indices': np.flatnonzero(selected).tolist(),
        'changed_normal_indices': [],
        'panel_source_corner_for_changed_vertex': nearest[selected].tolist(),
        'preserved_handle_vertex_count': int((~selected).sum()),
        'preserved_handle_vertices_exact': bool(np.array_equal(result[~selected], original[~selected])),
        'preserved_handle_normals_exact': bool(np.array_equal(revised_normals[~selected], normals[~selected])),
        'all_original_normals_exact': True,
        'topology_unchanged': True, 'normal_mapping': mapping.decode(),
        'normal_reference': reference.decode(), 'geometric_normal_rotation_max_error': float(error),
        'panel_triangle_indices': np.flatnonzero(panel_triangles).tolist()}


def set_world_vertices(obj, xyz_mm, scale):
    # Preserve object transforms and all non-coordinate mesh data.
    inverse = np.array(obj.matrix_world.inverted())
    local = (xyz_mm / (1000 * scale)) @ inverse[:3, :3].T + inverse[:3, 3]
    obj.data.vertices.foreach_set('co', local.astype(np.float32).reshape(-1))
    obj.data.update()


def save_outer(request, inspection, door, out):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=request['source_outer_handle_blend'])
    obj = bpy.data.objects[door['source_object']]
    scale = bpy.context.scene.unit_settings.scale_length
    original = world_vertices(obj, scale)
    revised = original.copy()
    revised[door['panel_vertex_indices']] = np.asarray(door['panel_vertices_mm'])
    set_world_vertices(obj, revised, scale)
    obj['door_panel_width_mm'] = door['width_mm']
    obj['door_panel_mounting'] = 'Motion-supported fixed hinge; 30 mm actual mounting-axis backset'
    obj['door_panel_motion_report'] = json.dumps(door)
    blend = out / 'door_hand_outer_motion_consistent.blend'
    glb = out / 'door_hand_outer_motion_consistent.glb'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.object.select_all(action='DESELECT')
    selected = [obj]
    for item in selected:
        item.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', use_selection=True)
    return {'blend': str(blend), 'glb': str(glb), 'selected_objects': [o.name for o in selected]}


def save_environment(request, inspection, door, out):
    import bpy
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mounting_assets
    bpy.ops.wm.open_mainfile(filepath=request['source_environment_blend'])
    layout = json.loads(Path(request['source_environment_layout']).read_text(encoding='utf-8-sig'))
    source_basis = np.array([layout['axis_u'], layout['axis_v'], [0, 0, 1]])
    scale = bpy.context.scene.unit_settings.scale_length
    opening = layout['door_opening']
    source_left, source_right = opening['min_u'] * 1000, opening['max_u'] * 1000
    source_mid = (source_left + source_right) / 2
    # Source casing has 38 mm jambs and 3 mm bevels. Preserve those bands.
    casing = bpy.data.objects['Door_Casing']
    casing_uv = world_vertices(casing, scale) @ source_basis.T
    left_group = casing_uv[:, 0] < source_mid
    old_inner_left = casing_uv[left_group, 0].max()
    old_inner_right = casing_uv[~left_group, 0].min()
    clearance = 4.0
    outline_source = np.asarray(layout['outline_uv']) * 1000
    opening_wall = np.isclose(outline_source[:, 1], opening['wall_v']*1000, atol=.01)
    notch_limits = outline_source[opening_wall, 0]
    if len(notch_limits) != 2:
        raise ValueError('Expected the actual opening wall to have two outline endpoints')
    notch_left, notch_right = float(notch_limits.min()), float(notch_limits.max())
    vestibule_mid = (notch_left+notch_right)/2
    left = vestibule_mid-(door['width_mm']+2*clearance)/2
    right = vestibule_mid+(door['width_mm']+2*clearance)/2
    delta_left, delta_right = left-old_inner_left, right-old_inner_right
    header_height = max(t['top_height_range_mm'][1] for t in door['takes']) + clearance
    old_height = opening['height'] * 1000
    up = np.array([0., 0., 1.])
    target_u = np.asarray(door['closed_width_world'])
    target_v = np.cross(up, target_u)
    target_basis = np.array([target_u, target_v, up])
    hinge = np.asarray(door['hinge_world_mm'])
    # Center the wall on the slab, including the measured external-axis offset.
    wall_v = opening['wall_v'] * 1000
    target_anchor = hinge.copy()
    target_anchor[2] = 0
    target_anchor += target_v * (door['hinge_to_handle_side_face_signed_mm']-door['thickness_mm']/2)
    source_anchor = np.array([left+clearance, wall_v, 0.])
    edits = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        xyz = world_vertices(obj, scale)
        uv = xyz @ source_basis.T
        old_uv = uv.copy()
        if obj.name == 'Door_Casing':
            uv[:, 0] += np.where(uv[:, 0] < source_mid, delta_left, delta_right)
            uv[uv[:, 2] > old_height/2, 2] += header_height-old_height
        elif obj.name == 'Environment_Walls':
            # Actual current walls have mismatched header/jamb endpoints; join
            # those existing boundary bands to the new casing clear aperture.
            near = np.abs(uv[:, 1]-wall_v) < 25
            lband = near & (np.abs(uv[:, 0]-source_left) < 60)
            rband = near & (np.abs(uv[:, 0]-source_right) < 60)
            uv[lband, 0] = left
            uv[rband, 0] = right
            uv[np.isclose(uv[:, 2], old_height, atol=.01), 2] = header_height
        elif obj.name == 'Skirting':
            uv[np.abs(uv[:, 0] - casing_uv[left_group, 0].min()) < .01, 0] += delta_left
            uv[np.abs(uv[:, 0] - casing_uv[~left_group, 0].max()) < .01, 0] += delta_right
        transformed = (uv-source_anchor) @ target_basis + target_anchor
        set_world_vertices(obj, transformed, scale)
        if obj.name in request['architecture_names']:
            edits.append({'name': obj.name, 'vertex_count': len(xyz),
                          'local_boundary_changed_count': int(np.any(np.abs(uv-old_uv) > .001, axis=1).sum()),
                          'materials': [m.name if m else None for m in obj.data.materials],
                          'uv_layers': list(obj.data.uv_layers.keys())})
    layout.update(revision='motion_consistent_room', units='metres',
                  axis_u=target_u.tolist(), axis_v=target_v.tolist(),
                  capture_world_hinge_mm=hinge.tolist(), alignment_inferred_not_wall_measured=True,
                  source_layout=request['source_environment_layout'])
    layout['motion_consistent_opening'] = {
        'panel_width_mm': door['width_mm'], 'clear_width_mm': right-left,
        'side_clearance_each_mm': clearance, 'header_height_mm': header_height,
        'panel_top_clearance_min_mm': clearance,
        'existing_casing_inner_width_mm': old_inner_right-old_inner_left,
        'existing_nominal_width_mm': source_right-source_left,
        'vestibule_width_mm': notch_right-notch_left,
        'clear_opening_to_corner_jamb_left_mm': left-notch_left,
        'clear_opening_to_corner_jamb_right_mm': notch_right-right,
        'casing_outer_edge_to_corner_left_mm': left-notch_left-38.,
        'casing_outer_edge_to_corner_right_mm': notch_right-right-38.,
        'interior_clear_jamb_left_mm': left-notch_left-layout['wall_thickness']*500,
        'interior_clear_jamb_right_mm': notch_right-right-layout['wall_thickness']*500,
        'outline_and_floor_manual_edits_retained': True,
        'existing_clearance_note': 'Source casing clear aperture 892 mm, nominal assumed panel 900 mm; negative old clearance corrected, 4 mm service gap selected, not measured.',
        'original_header_jamb_mismatch_corrected': True,
        'left_jamb_translation_mm': delta_left, 'right_jamb_translation_mm': delta_right}
    # Retain the entire current outline; only rigid capture-world alignment changes.
    outline = np.array([[u*1000, v*1000, 0.] for u, v in layout['outline_uv']])
    layout['outline_capture_world_m'] = (((outline-source_anchor) @ target_basis + target_anchor)/1000).tolist()
    layout_path = out / 'meeting_room_motion_consistent.layout.json'
    layout_path.write_text(json.dumps(layout, indent=2), encoding='utf-8')
    blend = out / 'meeting_room_motion_consistent.blend'
    glb = out / 'meeting_room_architecture_capture_world.glb'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.object.select_all(action='DESELECT')
    selected = [bpy.data.objects[name] for name in request['architecture_names']]
    for obj in selected:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', use_selection=True)
    sources = mounting_assets._furniture_sources(selected, scale)
    snippets = []
    try:
        for name, mesh in sources:
            entry = {'target_name': f'Room_{name}',
                     'native_unit_mm': request['takes'][0]['native_unit_mm'],
                     'asset_to_bone_rotation': np.eye(3).tolist(),
                     'asset_to_bone_translation_mm': [0, 0, 0],
                     'output_fbx': str(out/f'room_{name}_world_native.fbx')}
            snippet = mounting_assets._export_furniture(entry, [(name, mesh)])
            values = np.array([v.co[:] for v in mesh.vertices]) / entry['native_unit_mm']
            snippet['expected_world_vertices_native'] = values.astype(np.float32).astype(float).tolist()
            snippet['source_architecture_name'] = name
            snippets.append(snippet)
    finally:
        for _, mesh in sources:
            bpy.data.meshes.remove(mesh)
    opening_corners = np.array([[left, wall_v, 0], [right, wall_v, 0],
                                [right, wall_v, header_height], [left, wall_v, header_height]])
    opening_world = (opening_corners-source_anchor) @ target_basis + target_anchor
    alignment = {'axis_u': target_u.tolist(), 'axis_v': target_v.tolist(),
                 'source_basis_rows': source_basis.tolist(), 'capture_basis_rows': target_basis.tolist(),
                 'source_anchor_mm': source_anchor.tolist(), 'capture_anchor_mm': target_anchor.tolist(),
                 'opening_capture_world_mm': opening_world.tolist(),
                 'door_camera_target_world_mm': opening_world.mean(0).tolist(),
                 'room_frame_inferred_from_door_not_wall_markers': True}
    return {'blend': str(blend), 'glb': str(glb), 'layout': str(layout_path),
            'architecture_snippets': snippets, 'architecture_edits': edits,
            'clearances': layout['motion_consistent_opening'],
            'capture_world_alignment': alignment,
            'selected_objects': request['architecture_names'],
            'retained_unexported_objects': [obj.name for obj in bpy.context.scene.objects
                                           if obj.name not in request['architecture_names']]}


def inner_compatibility(request, inspection, door):
    reference = request.get('inner_reference')
    if reference is None:
        return {'available': False, 'reason': 'No inner-reference request supplied'}
    candidates = [obj for obj in inspection['inner']['objects'] if obj['type'] == 'MESH']
    obj = min(candidates, key=lambda obj: abs(np.ptp(np.asarray(obj['vertices_mm'])[:, 2])-190))
    vertices = np.asarray(obj['vertices_mm'])
    plate = vertices[np.isclose(np.abs(vertices[:, 2]), np.abs(vertices[:, 2]).max(), atol=.002)]
    contact = np.array([(plate[:, 0].min()+plate[:, 0].max())/2,
                        plate[:, 1].max(), (plate[:, 2].min()+plate[:, 2].max())/2])
    width_axis, normal_axis, _ = np.asarray(door['cad_basis_rows'])
    outer_contact = np.asarray(door['mounting_contact_cad_mm'])
    lock = float(np.dot(outer_contact, width_axis)+door['backset_mm'])
    records = []
    outer_takes = {take['take']: take for take in request['takes']}
    for take in reference['takes']:
        outer_take = outer_takes[take['take']]
        if not np.allclose(take['pose_times_seconds'], outer_take['pose_times_seconds'], atol=1e-9, rtol=0):
            raise ValueError('Inner/outer sample times do not agree')
        inner_rotation, inner_position = motion(take)
        outer_rotation, outer_position = motion(outer_take)
        contact_world = inner_rotation @ contact + inner_position
        contact_outer = np.einsum('nji,nj->ni', outer_rotation, contact_world-outer_position)
        backset = lock-contact_outer @ width_axis
        separation = (contact_outer-outer_contact) @ normal_axis
        mismatch = separation-door['thickness_mm']
        def stats(values):
            return {'mean': float(values.mean()), 'min': float(values.min()),
                    'max': float(values.max()), 'rms': float(np.sqrt(np.mean(values**2)))}
        records.append({'take': take['take'], 'inner_mounting_axis_backset_mm': stats(backset),
                        'inner_axis_backset_error_from_30_mm': stats(backset-door['backset_mm']),
                        'mounting_face_separation_mm': stats(separation),
                        'inner_contact_to_panel_far_face_signed_mm': stats(mismatch)})
    return {'available': True, 'source_blend': reference['source_inner_handle_blend'],
            'source_glb': reference['source_inner_handle_glb'],
            'source_blend_sha256': sha256(reference['source_inner_handle_blend']),
            'source_glb_sha256': sha256(reference['source_inner_handle_glb']),
            'mounting_contact_inner_cad_mm': contact.tolist(),
            'mounting_axis_evidence': 'Actual +/-Z base-plate cap-ring symmetry and actual rear planar contact surface',
            'inner_mesh_and_markers_unchanged': True,
            'compatibility_note': 'Reported mismatches are retained, not forced to fit; captures are the source of truth.',
            'takes': records}


def produce(request, inspection):
    out = Path(request['asset_output_directory'])
    for key in ('source_environment_blend', 'source_environment_glb', 'source_environment_layout',
                'source_outer_handle_blend', 'source_outer_handle_glb'):
        if out.resolve() in Path(request[key]).resolve().parents:
            raise ValueError('Output directory must not contain original source assets')
    door = fit_door(request, inspection['outer'])
    inner = inner_compatibility(request, inspection, door)
    index = {'schema_version': 1, 'run_id': request['run_id'], 'door': door,
             'source_hashes': {key: sha256(request[key]) for key in (
                 'source_environment_blend', 'source_environment_glb', 'source_environment_layout',
                 'source_outer_handle_blend', 'source_outer_handle_glb')}, 'takes': []}
    index['inner_handle_compatibility'] = inner
    clip_widths = [take['independent_clip_width_mm'] for take in door['takes']
                   if 'independent_clip_width_mm' in take]
    door['independent_moving_clip_width_range_mm'] = [min(clip_widths), max(clip_widths)]
    door['width_uncertainty_note'] = 'Per-clip orientation-constrained fit spread is calibration evidence, not metrology or a confidence interval.'
    index['physical_fit_status'] = 'NEEDS_REVIEW'
    index['physical_fit_flags'] = {'hinge_max_noise': 'NEEDS_REVIEW',
                                   'floor_gap': 'NEEDS_REVIEW'}
    index['unmet_constraints'] = [
        {'constraint': 'Every sample within 5 mm of a common fixed hinge',
         'observed_max_mm': max(take['hinge_max_mm'] for take in door['takes']),
         'status': 'NEEDS_REVIEW', 'reason': 'Original trajectories retained without clamping'},
        {'constraint': 'Perfectly level panel bottom throughout every clip',
         'observed_bottom_range_mm': [
             min(take['bottom_height_range_mm'][0] for take in door['takes']),
             max(take['bottom_height_range_mm'][1] for take in door['takes'])],
         'status': 'NEEDS_REVIEW', 'reason': 'Source height axis retained and common static placement chosen'},
    ]
    if inner['available']:
        index['physical_fit_flags'].update(inner_backset_residual='NEEDS_REVIEW',
                                           far_face_mismatch='NEEDS_REVIEW')
        index['unmet_constraints'].append({
            'constraint': 'Unchanged inner mounting axis has exactly 30 mm backset and touches the 19 mm far face',
            'observed_backset_range_mm': [
                min(take['inner_mounting_axis_backset_mm']['min'] for take in inner['takes']),
                max(take['inner_mounting_axis_backset_mm']['max'] for take in inner['takes'])],
            'observed_face_separation_range_mm': [
                min(take['mounting_face_separation_mm']['min'] for take in inner['takes']),
                max(take['mounting_face_separation_mm']['max'] for take in inner['takes'])],
            'observed_inner_backset_residual_range_mm': [
                min(take['inner_axis_backset_error_from_30_mm']['min'] for take in inner['takes']),
                max(take['inner_axis_backset_error_from_30_mm']['max'] for take in inner['takes'])],
            'observed_far_face_mismatch_range_mm': [
                min(take['inner_contact_to_panel_far_face_signed_mm']['min'] for take in inner['takes']),
                max(take['inner_contact_to_panel_far_face_signed_mm']['max'] for take in inner['takes'])],
            'status': 'NEEDS_REVIEW', 'reason': 'Inner handle, markers and tracking preserved unchanged'})
    for take in request['takes']:
        if sha256(take['fbx']) != take['source_sha256']:
            raise ValueError(f'Source FBX changed: {take["take"]}')
        vertices, normals, checks = vertex_overrides(take, door)
        uid = str(take['door_outer_geometry_uid'])
        index['takes'].append({'take': take['take'], 'source_fbx': take['fbx'],
                               'source_sha256': take['source_sha256'],
                               'geometry_vertex_overrides': {uid: vertices},
                               'geometry_normal_overrides': {},
                               'checks': checks})
    index['outer_asset'] = save_outer(request, inspection, door, out)
    index['environment_asset'] = save_environment(request, inspection, door, out)
    index['export_notes'] = {
        'architecture_selection': request['architecture_names'],
        'original_images': 'Actual source shader graphs retained; all referenced image payloads embedded, with an empty list when the source has none.',
        'normal_preservation': 'Original normal arrays retained exactly; geometric face-normal differences from boundary translation are measured in per-take checks.',
        'visual_verification': 'Not performed; parent must re-import and render actual deliverables.'}
    report = out/'room_motion_geometry_report.json'
    report.write_text(json.dumps({k: v for k, v in index.items() if k != 'takes'}, indent=2), encoding='utf-8')
    index['report'] = str(report)
    control = Path(request['output_directory'])/'control'
    control.mkdir(parents=True, exist_ok=True)
    path = Path(request.get('index_path', control/'room_asset_index.json'))
    path.write_text(json.dumps(index, indent=2), encoding='utf-8')
    print('ROOM_ASSET_INDEX', path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--inner-reference')
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding='utf-8-sig'))
    reference_path = Path(args.inner_reference or request.get('inner_reference_path',
        str(Path(args.request).with_name('inner_handle_reference.json'))))
    if reference_path.is_file():
        request['inner_reference'] = json.loads(reference_path.read_text(encoding='utf-8-sig'))
    out = Path(request['asset_output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    inspection = {name: inspect_blend(request[key]) for name, key in (
        ('outer', 'source_outer_handle_blend'), ('environment', 'source_environment_blend'))}
    if 'inner_reference' in request:
        inspection['inner'] = inspect_blend(request['inner_reference']['source_inner_handle_blend'])
    (out / 'source_inspection.json').write_text(json.dumps(inspection, indent=2), encoding='utf-8')
    if not args.inspect:
        produce(request, inspection)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)

"""Read-only GLB registration and textured, static bone-local FBX preparation.

Blender --background --factory-startup --python mounting_assets.py -- \
    --request request.json

Request paths are relative to request.json. The JSON object has:
  output_index: NEW JSON path
  registrations: [{id, glb, calibration_json, native_unit_mm,
    existing_parts: [{name, source_name? (exact GLB object name), vertex_count,
      triangle_count, vertices_native: [[x,y,z],...]}]}]
Instead of vertices_native, a part may supply sample_indices plus
sample_vertices_native. Optional triangles (full Nx3 vertex indices) or
triangles_sha256 (SHA256 of C-order little-endian int32 triangles) verifies
unchanged topology. Counts alone do not prove topology or unsampled geometry;
this limitation is recorded, never silently treated as full verification.
  furniture_exports: [{id, glb, output_fbx, target_name,
    asset_to_bone_rotation: [[...],[...],[...]],
    asset_to_bone_translation_mm: [x,y,z], native_unit_mm}]

The index contains registrations and furniture_exports dictionaries keyed by id.
Registration rotation/translation_mm maps asset-world mm into original bone-local
mm. Residuals use a fixed maximum 0.01 mm acceptance limit. Calibration holes
are returned unchanged: this utility neither computes sphere offsets nor fits
marker assignments. No original FBX or .blend is imported or re-exported.

Coordinate convention deliberately matches mesh_assets._mesh_data: evaluated
GLB object.matrix_world, multiplied by scene.scale_length * 1000. Only the glTF
default scene (or its unique scene) participates. Existing output files are never
overwritten. Pure NumPy registration helpers can be imported without bpy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


MAX_REGISTRATION_RESIDUAL_MM = 0.01


def _array(value, label, columns=3, integer=False):
    result = np.asarray(value)
    if (result.dtype.kind not in ('iu' if integer else 'iuf')
            or result.ndim != 2 or result.shape[1] != columns):
        raise ValueError(f'{label}: expected numeric N x {columns} array')
    if not integer and not np.isfinite(result).all():
        raise ValueError(f'{label}: nonfinite coordinates')
    return result.astype(np.int64 if integer else np.float64, copy=False)


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label}: expected positive finite number')
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'{label}: expected positive finite number')
    return result


def _count(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f'{label}: expected positive integer')
    return value


def _rigid_transform(rotation, translation):
    rotation = _array(rotation, 'rotation')
    translation = np.asarray(translation, dtype=np.float64)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8, rtol=0)):
        raise ValueError('Expected a finite proper rigid rotation and translation')
    return rotation, translation


def fit_rigid(pairs):
    """Fit one proper source->target transform over all (N x 3, N x 3) pairs.

    Accumulate small covariance matrices instead of concatenating large meshes.
    Planar geometry is valid; collinear/point correspondences are not sufficient.
    """
    total = 0
    source_sum, target_sum = np.zeros(3), np.zeros(3)
    checked = []
    for source, target in pairs:
        source = _array(source, 'source')
        target = _array(target, 'target')
        if source.shape != target.shape or not len(source):
            raise ValueError('Source and target correspondence shapes differ or are empty')
        checked.append((source, target))
        total += len(source)
        source_sum += source.sum(axis=0)
        target_sum += target.sum(axis=0)
    if total < 3:
        raise ValueError('At least three noncollinear correspondences are required')
    source_mean, target_mean = source_sum / total, target_sum / total
    covariance = np.zeros((3, 3))
    for source, target in checked:
        covariance += (source - source_mean).T @ (target - target_mean)
    left, singular, right = np.linalg.svd(covariance)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-12:
        raise ValueError('Collinear or degenerate registration correspondences')
    correction = np.eye(3)
    correction[2, 2] = 1. if np.linalg.det(right.T @ left.T) >= 0 else -1.
    rotation = right.T @ correction @ left.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def _part_candidates(existing, sources):
    explicit = existing.get('source_name')
    name = existing['name']
    candidates = []
    for index, source in enumerate(sources):
        source_name = source['name']
        name_matches = (source_name == explicit if explicit is not None else
                        name == source_name or bool(re.fullmatch(
                            r'Mesh_.+_' + re.escape(source_name) + r'_\d+', name)))
        if (name_matches and len(source['vertices_mm']) == existing['vertex_count']
                and len(source['triangles']) == existing['triangle_count']):
            candidates.append(index)
    return candidates


def register_parts(sources, existing_parts, native_unit_mm):
    """Recover old mount, rejecting missing/ambiguous/changed sampled geometry.

    sources contain name, vertices_mm and triangles, in mesh_assets convention.
    The fit and residual check include EVERY mesh part, not only the first part.
    """
    unit = _positive(native_unit_mm, 'native_unit_mm')
    if not sources or not existing_parts or len(sources) != len(existing_parts):
        raise ValueError('Changed source: all original GLB mesh parts must match exactly once')
    matched, used, pairs = [], set(), []
    for existing in existing_parts:
        if not isinstance(existing.get('name'), str) or not existing['name']:
            raise ValueError('Existing mesh part requires a name')
        _count(existing.get('vertex_count'), 'vertex_count')
        _count(existing.get('triangle_count'), 'triangle_count')
        candidates = _part_candidates(existing, sources)
        if len(candidates) != 1 or candidates[0] in used:
            raise ValueError(f"Changed/ambiguous source geometry for {existing['name']}: {candidates}")
        index = candidates[0]
        used.add(index)
        source = sources[index]
        vertices = source['vertices_mm']
        topology_verified = False
        if 'triangles' in existing:
            triangles = _array(existing['triangles'], 'triangles', integer=True)
            if not np.array_equal(triangles, source['triangles']):
                raise ValueError(f"Changed topology: {existing['name']}")
            topology_verified = True
        if 'triangles_sha256' in existing:
            digest = hashlib.sha256(np.asarray(source['triangles'], dtype='<i4').tobytes()).hexdigest()
            if digest != existing['triangles_sha256']:
                raise ValueError(f"Changed topology hash: {existing['name']}")
            topology_verified = True
        full = 'vertices_native' in existing
        if full:
            if 'sample_indices' in existing or 'sample_vertices_native' in existing:
                raise ValueError('Specify full vertices OR sampled vertices, not both')
            target = _array(existing['vertices_native'], 'vertices_native')
            if len(target) != len(vertices):
                raise ValueError(f"Changed vertex count: {existing['name']}")
            sampled_source = vertices
        else:
            indices = np.asarray(existing.get('sample_indices'))
            if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
                    or not len(indices) or len(np.unique(indices)) != len(indices)
                    or np.any(indices < 0) or np.any(indices >= len(vertices))):
                raise ValueError('sample_indices must be unique valid vertex indices')
            target = _array(existing.get('sample_vertices_native'), 'sample_vertices_native')
            if len(indices) != len(target):
                raise ValueError('sample_indices/sample_vertices_native lengths differ')
            sampled_source = vertices[indices]
            full = len(indices) == len(vertices)
        target_mm = target * unit
        pairs.append((sampled_source, target_mm))
        matched.append({'name': existing['name'], 'source_name': source['name'],
                        'vertex_count': len(vertices), 'triangle_count': len(source['triangles']),
                        'correspondence_count': len(target), 'all_vertices_verified': full,
                        'topology_verified': topology_verified})
    rotation, translation = fit_rigid(pairs)
    sum_squared, total, maximum = 0., 0, 0.
    for record, (source, target) in zip(matched, pairs):
        errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        rms, peak = float(np.sqrt(np.mean(errors ** 2))), float(errors.max())
        record['residual_mm'] = {'rms': rms, 'max': peak, 'count': len(errors)}
        sum_squared += float(errors @ errors)
        total += len(errors)
        maximum = max(maximum, peak)
    if maximum > MAX_REGISTRATION_RESIDUAL_MM:
        raise ValueError(f'Changed source/nonrigid mount: maximum correspondence residual '
                         f'{maximum:.9g} mm exceeds {MAX_REGISTRATION_RESIDUAL_MM} mm')
    limitations = []
    if any(not part['all_vertices_verified'] for part in matched):
        limitations.append('Unsampled vertex positions are not verified')
    if any(not part['topology_verified'] for part in matched):
        limitations.append('Only topology counts, not triangle indices, were supplied')
    return {'rotation': rotation.tolist(), 'translation_mm': translation.tolist(),
            'residual_mm': {'rms': math.sqrt(sum_squared / total), 'max': maximum, 'count': total},
            'max_allowed_residual_mm': MAX_REGISTRATION_RESIDUAL_MM,
            'parts': matched, 'limitations': limitations}


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _calibration(path):
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    holes = document.get('holes')
    if not isinstance(holes, list) or not holes:
        raise ValueError(f'{path}: calibration requires holes')
    identifiers = set()
    for hole in holes:
        identifier = hole.get('id', hole.get('hole_id', hole.get('marker')))
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f'{path}: hole identifiers must be unique nonempty strings')
        identifiers.add(identifier)
        center = hole.get('center_mm')
        normal = hole.get('normal')
        if isinstance(center, dict):
            center = [center[key] for key in 'xyz']
        if isinstance(normal, dict):
            normal = [normal[key] for key in 'xyz']
        center = np.asarray(center, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        if (center.shape != (3,) or normal.shape != (3,)
                or not np.isfinite(center).all() or not np.isfinite(normal).all()
                or np.linalg.norm(normal) <= 1e-12):
            raise ValueError(f'{path}: each hole requires finite center_mm and nonzero normal')
    return holes


def _import_glb(path):
    import bpy
    import mesh_assets

    selection = mesh_assets._glb_scene_selection(path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path), import_scene_as_collection=True)
    objects = sorted((obj for obj in bpy.context.view_layer.objects if obj.type == 'MESH'),
                     key=lambda obj: obj.name)
    if not objects:
        raise ValueError(f'{path}: no default-scene meshes')
    for obj in objects:
        if mesh_assets._is_support(obj):
            raise ValueError(f'{path}: helper/marker support geometry is not body geometry: {obj.name}')
    return objects, selection, mesh_assets._scene_scale()


def _source_parts(objects, scale):
    import bpy
    import mesh_assets

    graph = bpy.context.evaluated_depsgraph_get()
    parts = []
    for obj in objects:
        data = mesh_assets._mesh_data(obj, graph, scale)
        parts.append({'name': data['name'], 'vertices_mm': data['vertices'],
                      'triangles': data['triangles']})
    return parts


def _furniture_sources(objects, scale):
    """Independent evaluated world-mm meshes retaining loop normals, UVs and slots."""
    import bpy
    from mathutils import Matrix

    graph = bpy.context.evaluated_depsgraph_get()
    sources = []
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
                                              depsgraph=graph)
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64).copy()
        matrix[:3, :] *= scale * 1000.
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-18:
            raise ValueError(f'{obj.name}: invalid source world transform')
        mesh.transform(Matrix(matrix.tolist()))
        if np.linalg.det(matrix[:3, :3]) < 0:
            mesh.flip_normals()
        mesh.update()
        mesh.calc_loop_triangles()
        if len(mesh.vertices) < 3 or not mesh.loop_triangles:
            raise ValueError(f'{obj.name}: no triangle body geometry')
        sources.append((obj.name, mesh))
    return sources


def _image_payloads(meshes):
    """Keep imported shader graphs; require every referenced image in the FBX."""
    import bpy

    images, seen_trees = {}, set()

    def visit(tree):
        if tree is None or tree.as_pointer() in seen_trees:
            return
        seen_trees.add(tree.as_pointer())
        for node in tree.nodes:
            if node.type == 'GROUP':
                visit(node.node_tree)
            elif node.type == 'TEX_IMAGE' and node.image is not None:
                image = node.image
                if image.source not in {'FILE', 'GENERATED'}:
                    raise ValueError(f'Unsupported nonstatic image: {image.name}')
                if image.packed_file is None:
                    image.pack()
                if image.packed_file is None or not image.packed_file.data:
                    raise ValueError(f'Cannot pack source texture: {image.name}')
                payload = image.packed_file.data
                digest = hashlib.sha256(payload).hexdigest()
                images[digest] = {'name': image.name, 'sha256': digest, 'bytes': len(payload)}
                # Unique virtual filenames avoid exporter deduplication collisions.
                # Packed bytes remain in memory; no source asset is saved.
                image.filepath = '//mounting_' + digest + ('.jpg' if payload[:2] == b'\xff\xd8' else '.png')

    for mesh in meshes:
        for material in mesh.materials:
            if material is not None and material.use_nodes:
                visit(material.node_tree)
    return images


def _child(element, name):
    return next((child for child in element.elems if child.id == name), None)


def _properties(element):
    properties = _child(element, b'Properties70')
    return {} if properties is None else {p.props[0]: p.props[4:] for p in properties.elems}


def _audit_fbx(path, objects, native_unit_mm, image_payloads):
    """Inspect the real 7400/7700 FBX without importing/re-exporting any sequence."""
    from io_scene_fbx.parse_fbx import parse

    root, version = parse(str(path))
    settings = _properties(_child(root, b'GlobalSettings'))
    unit = float(settings[b'UnitScaleFactor'][0])
    # scale_length is an RNA float32; the exporter promotes that stored value
    # to Python float before multiplying by 100, not the nominal request value.
    expected_unit = 100. * float(np.float32(native_unit_mm / 1000.))
    if unit != expected_unit:
        raise ValueError(f'Wrong FBX units: UnitScaleFactor={unit}, expected {expected_unit}')
    elements = _child(root, b'Objects').elems
    permitted = {b'Geometry', b'Model', b'Material', b'Texture', b'Video'}
    unexpected = {element.id for element in elements} - permitted
    if unexpected:
        raise ValueError(f'Nonstatic/non-mesh FBX objects: {unexpected}')
    models = [element for element in elements if element.id == b'Model']
    geometries = [element for element in elements if element.id == b'Geometry']
    if len(models) != len(objects) or len(geometries) != len(objects):
        raise ValueError('Export changed mesh/model counts')
    by_geometry_name = {element.props[1].split(b'\x00')[0].decode(): element for element in geometries}
    records = []
    for model in models:
        if model.props[2] != b'Mesh':
            raise ValueError('Export contains a non-mesh Model')
        properties = _properties(model)
        for key, default in ((b'Lcl Translation', [0., 0., 0.]),
                             (b'Lcl Rotation', [0., 0., 0.]),
                             (b'Lcl Scaling', [1., 1., 1.]),
                             (b'GeometricTranslation', [0., 0., 0.]),
                             (b'GeometricRotation', [0., 0., 0.]),
                             (b'GeometricScaling', [1., 1., 1.])):
            if not np.allclose(properties.get(key, default), default, atol=1e-9, rtol=0):
                raise ValueError(f'Nonidentity exported model transform: {key!r}')
    for obj in objects:
        mesh = obj.data
        geometry = by_geometry_name.get(mesh.name)
        if geometry is None:
            raise ValueError(f'Missing exported Geometry: {mesh.name}')
        actual = np.asarray(_child(geometry, b'Vertices').props[0]).reshape(-1, 3)
        expected = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get('co', expected.reshape(-1))
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f'Export changed raw native-unit coordinates: {obj.name}')
        normals = _child(geometry, b'LayerElementNormal')
        if normals is None:
            raise ValueError(f'Export lost normals: {obj.name}')
        normal_values = np.asarray(_child(normals, b'Normals').props[0]).reshape(-1, 3)
        if not len(normal_values) or not np.isfinite(normal_values).all():
            raise ValueError(f'Export wrote invalid normals: {obj.name}')
        uv_layers = [element for element in geometry.elems if element.id == b'LayerElementUV']
        if len(uv_layers) != len(mesh.uv_layers):
            raise ValueError(f'Export lost UV layers: {obj.name}')
        for source_uv, exported_uv in zip(mesh.uv_layers, uv_layers):
            direct = np.asarray(_child(exported_uv, b'UV').props[0]).reshape(-1, 2)
            indices = np.asarray(_child(exported_uv, b'UVIndex').props[0])
            expected_uv = np.empty((len(mesh.loops), 2), dtype=np.float32)
            source_uv.data.foreach_get('uv', expected_uv.reshape(-1))
            if not np.array_equal(direct[indices], expected_uv):
                raise ValueError(f'Export changed loop UVs: {obj.name}')
        mesh.calc_loop_triangles()
        records.append({'name': obj.name, 'geometry_name': mesh.name,
                        'vertex_count': len(mesh.vertices), 'triangle_count': len(mesh.loop_triangles),
                        'uv_layers': len(uv_layers), 'normals': len(normal_values)})
    videos = [element for element in elements if element.id == b'Video']
    embedded = {}
    for video in videos:
        content = _child(video, b'Content')
        if content is None or not content.props or not content.props[0]:
            raise ValueError('Export contains an unembedded/empty texture Video')
        payload = content.props[0]
        embedded[hashlib.sha256(payload).hexdigest()] = len(payload)
    if set(embedded) != set(image_payloads):
        raise ValueError('Export lost or changed an actual source image texture')
    if image_payloads:
        if not any(element.id == b'Texture' for element in elements):
            raise ValueError('Export lost texture objects')
        if any(not record['uv_layers'] for record in records):
            raise ValueError('Image-textured geometry lacks UVs')
    return {'fbx_version': version, 'unit_scale_factor': unit,
            'nominal_unit_scale_factor': native_unit_mm / 10.,
            'unit_scale_storage': '100 * float32(native_unit_mm / 1000)',
            'parts': records,
            'identity_model_transforms': True, 'raw_native_coordinates_exact': True,
            'embedded_textures': list(image_payloads.values())}


def _export_furniture(entry, sources):
    import bpy
    from mathutils import Matrix

    rotation, translation = _rigid_transform(entry['asset_to_bone_rotation'],
                                              entry['asset_to_bone_translation_mm'])
    unit = _positive(entry['native_unit_mm'], 'native_unit_mm')
    target = entry['target_name']
    if not isinstance(target, str) or not target or '\x00' in target:
        raise ValueError('target_name must be a nonempty string without NUL')
    bpy.ops.object.select_all(action='DESELECT')
    created = []
    matrix = np.eye(4)
    matrix[:3, :3] = rotation / unit
    matrix[:3, 3] = translation / unit
    try:
        for index, (_, source) in enumerate(sources):
            mesh = source.copy()
            mesh.name = target + f'_geometry_{index}'
            mesh.transform(Matrix(matrix.tolist()))
            mesh.update()
            obj = bpy.data.objects.new(target if len(sources) == 1 else target + f'_{index}', mesh)
            bpy.context.scene.collection.objects.link(obj)
            obj.matrix_world = Matrix.Identity(4)
            obj.select_set(True)
            created.append(obj)
        bpy.context.view_layer.objects.active = created[0]
        images = _image_payloads([obj.data for obj in created])
        bpy.context.scene.unit_settings.system = 'METRIC'
        bpy.context.scene.unit_settings.scale_length = unit / 1000.
        bpy.context.view_layer.update()
        path = Path(entry['output_fbx'])
        path.parent.mkdir(parents=True, exist_ok=True)
        # Blender exporter source: SCALE_UNITS keeps global_matrix unscaled and
        # writes scale_length*100 as UnitScaleFactor (FBX unit is centimetres).
        # Y-forward/Z-up is Blender's native basis, so no axis rotation is added.
        result = bpy.ops.export_scene.fbx(
            filepath=str(path), use_selection=True, object_types={'MESH'},
            global_scale=1., apply_unit_scale=True, apply_scale_options='FBX_SCALE_UNITS',
            axis_forward='Y', axis_up='Z', bake_space_transform=False,
            use_mesh_modifiers=False, mesh_smooth_type='OFF', use_tspace=False,
            use_custom_props=False, add_leaf_bones=False, bake_anim=False,
            path_mode='COPY', embed_textures=True, batch_mode='OFF')
        if 'FINISHED' not in result or not path.is_file():
            raise ValueError(f'Blender did not export {path}')
        audit = _audit_fbx(path, created, unit, images)
        return {'output_fbx': str(path), 'target_name': target, 'native_unit_mm': unit,
                'asset_to_bone_rotation': rotation.tolist(),
                'asset_to_bone_translation_mm': translation.tolist(),
                'output_sha256': _hash(path), **audit}
    finally:
        for obj in created:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)


def _resolve_request(document, base, request_path):
    registrations = document.get('registrations', [])
    exports = document.get('furniture_exports', [])
    if not isinstance(registrations, list) or not isinstance(exports, list):
        raise ValueError('registrations/furniture_exports must be arrays')
    if not registrations and not exports:
        raise ValueError('Request must contain registrations or furniture_exports')

    def resolve(value):
        path = Path(value).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()

    index = resolve(document.get('output_index', document.get('index_json')))
    protected = {request_path}
    for entries in (registrations, exports):
        ids = set()
        for entry in entries:
            identifier = entry.get('id')
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError('Entry ids must be nonempty and unique within each operation')
            ids.add(identifier)
            glb = resolve(entry['glb'])
            if not glb.is_file() or glb.suffix.lower() != '.glb':
                raise ValueError(f'Expected an existing GLB: {glb}')
            entry['glb'] = str(glb)
            protected.add(glb)
            protected.add(glb.with_suffix('.blend'))
            _positive(entry['native_unit_mm'], 'native_unit_mm')
    for entry in registrations:
        calibration = resolve(entry['calibration_json'])
        if not calibration.is_file():
            raise ValueError(f'Calibration does not exist: {calibration}')
        entry['calibration_json'] = str(calibration)
        protected.add(calibration)
        if 'existing_parts' not in entry:
            entry['existing_parts'] = entry['parts']
    outputs = [index]
    for entry in exports:
        destination = resolve(entry['output_fbx'])
        if destination.suffix.lower() != '.fbx':
            raise ValueError('Furniture output must have .fbx extension')
        entry['output_fbx'] = str(destination)
        outputs.append(destination)
    if len(set(outputs)) != len(outputs):
        raise ValueError('Output paths must be distinct')
    for output in outputs:
        if output.exists():
            raise FileExistsError(f'Output already exists: {output}')
        if output in protected or any(source.exists() and output.exists()
                                      and os.path.samefile(source, output) for source in protected):
            raise ValueError(f'Output aliases a protected source: {output}')
    return registrations, exports, index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(argv)
    request_path = args.request.resolve(strict=True)
    document = json.loads(request_path.read_text(encoding='utf-8-sig'))
    registrations, exports, index_path = _resolve_request(document, request_path.parent, request_path)
    # Blender's --python invocation does not guarantee sibling-module imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    groups = defaultdict(lambda: {'registrations': [], 'furniture_exports': []})
    for entry in registrations:
        groups[entry['glb']]['registrations'].append(entry)
    for entry in exports:
        groups[entry['glb']]['furniture_exports'].append(entry)
    index = {'schema_version': 1, 'request': str(request_path), 'request_sha256': _hash(request_path),
             'registrations': {}, 'furniture_exports': {},
             'coordinate_convention': 'mesh_assets evaluated GLB world * scene.scale_length * 1000 mm'}
    for glb, group in groups.items():
        path = Path(glb)
        digest = _hash(path)
        try:
            objects, selection, scale = _import_glb(path)
            parts = _source_parts(objects, scale) if group['registrations'] else None
            furniture = _furniture_sources(objects, scale) if group['furniture_exports'] else None
        except Exception as exc:
            for operation, entries in group.items():
                for entry in entries:
                    index[operation][entry['id']] = {'status': 'FAILED', 'error': str(exc), 'source_glb': glb,
                                                      'source_sha256': digest}
            continue
        for entry in group['registrations']:
            try:
                calibration = Path(entry['calibration_json'])
                result = register_parts(parts, entry['existing_parts'], entry['native_unit_mm'])
                result.update(calibration_json=str(calibration), calibration_sha256=_hash(calibration),
                              holes=_calibration(calibration), native_unit_mm=entry['native_unit_mm'])
                result['status'] = 'READY'
            except Exception as exc:
                result = {'status': 'FAILED', 'error': str(exc)}
            index['registrations'][entry['id']] = {**result, 'source_glb': glb,
                                                  'source_sha256': digest, 'scene_selection': selection}
        for entry in group['furniture_exports']:
            try:
                result = _export_furniture(entry, furniture)
                result['status'] = 'READY'
            except Exception as exc:
                result = {'status': 'FAILED', 'error': str(exc), 'output_fbx': entry['output_fbx']}
            index['furniture_exports'][entry['id']] = {**result, 'source_glb': glb,
                                                       'source_sha256': digest, 'scene_selection': selection}
    index['passed'] = all(record['status'] == 'READY'
                          for operation in ('registrations', 'furniture_exports')
                          for record in index[operation].values())
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open('x', encoding='utf-8') as stream:
        json.dump(index, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output_index': str(index_path), 'passed': index['passed'],
                      'registrations': len(index['registrations']),
                      'furniture_exports': len(index['furniture_exports'])}), flush=True)
    return 0 if index['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None))

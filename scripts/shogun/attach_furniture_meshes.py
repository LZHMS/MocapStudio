"""Attach calibrated, metre-scaled PLY furniture to existing animated FBX bones.

Inputs are explicit FBX paths. Existing meshes, animation and markers are retained
by mesh_fbx's binary graft; no Shogun or Blender animation re-export is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mesh_alignment import _holes, _json, analyze_fbx
from mesh_fbx import graft_file


_VERTEX_DTYPE = np.dtype([(axis, '<f4') for axis in 'xyz']
                         + [(channel, 'u1') for channel in ('r', 'g', 'b', 'a')])
_FACE_DTYPE = np.dtype([('count', 'u1'), ('indices', '<i4', (3,))])


def _read_ply(path):
    """Read the explicit binary XYZ/RGBA triangle layout used by these scans."""
    with path.open('rb') as stream:
        header = []
        for _ in range(100):
            line = stream.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        else:
            raise ValueError(f'{path}: missing bounded PLY header')
        if header[:2] != ['ply', 'format binary_little_endian 1.0']:
            raise ValueError(f'{path}: requires binary little-endian PLY')
        layout = [line for line in header[2:] if not line.startswith('comment ')]
        if (len(layout) != 11 or not layout[0].startswith('element vertex ')
                or layout[1:8] != ['property float x', 'property float y', 'property float z',
                                  'property uchar red', 'property uchar green',
                                  'property uchar blue', 'property uchar alpha']
                or not layout[8].startswith('element face ')
                or layout[9:] != ['property list uchar int vertex_indices', 'end_header']):
            raise ValueError(f'{path}: unsupported vertex/face property layout')
        vertex_count, face_count = int(layout[0].split()[-1]), int(layout[8].split()[-1])
        if vertex_count < 3 or face_count < 1:
            raise ValueError(f'{path}: empty geometry')
        vertices = np.fromfile(stream, dtype=_VERTEX_DTYPE, count=vertex_count)
        faces = np.fromfile(stream, dtype=_FACE_DTYPE, count=face_count)
        if len(vertices) != vertex_count or len(faces) != face_count or stream.read(1):
            raise ValueError(f'{path}: truncated geometry or unexpected trailing payload')
    xyz = np.column_stack([vertices[axis] for axis in 'xyz']).astype(np.float64)
    triangles = faces['indices']
    if (not np.isfinite(xyz).all() or np.any(faces['count'] != 3)
            or np.any(triangles < 0) or np.any(triangles >= vertex_count)):
        raise ValueError(f'{path}: invalid vertices or triangle indices')
    rgba = np.column_stack([vertices[channel] for channel in ('r', 'g', 'b', 'a')])
    return xyz, triangles, rgba


def _load_furniture(path):
    config = _json(path)
    if config.get('schema_version') != 1 or config.get('coordinate_unit') != 'mm':
        raise ValueError(f'{path}: requires version 1 millimetre marker configuration')
    prop = config['object_name']
    holes = _holes(config)
    marker_map = config['marker_map']
    if (len(holes) != 4 or set(marker_map) != {'M_1', 'M_2', 'M_3', 'M_4'}
            or set(marker_map.values()) != {hole['id'] for hole in holes}):
        raise ValueError(f'{path}: requires a bijective four-marker mapping')
    mesh = config['mesh']
    if mesh.get('scaled_coordinate_unit') != 'm':
        raise ValueError(f'{path}: scaled PLY must explicitly use metres')
    scaled = (path.parent / mesh['scaled_file']).resolve(strict=True)
    source = (path.parent / mesh['file']).resolve(strict=True)
    xyz_m, triangles, rgba = _read_ply(scaled)
    original, original_triangles, original_rgba = _read_ply(source)
    scale = mesh['scale_to_mm'] / 1000.
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'{path}: invalid source scale')
    np.testing.assert_allclose(xyz_m, original * scale, rtol=2e-7, atol=2e-7)
    if not np.array_equal(triangles, original_triangles) or not np.array_equal(rgba, original_rgba):
        raise ValueError(f'{path}: scaled PLY changed topology or vertex colours')
    audit = {'config': str(path), 'scaled_ply': str(scaled),
             'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'scaled_ply_sha256': hashlib.sha256(scaled.read_bytes()).hexdigest(),
             'source_ply_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'source_scale_to_m': scale, 'scaled_coordinates_verified': True,
             'vertices': len(xyz_m), 'triangles': len(triangles),
             'appearance': 'Original per-vertex sRGB RGBA; white material, no invented texture'}
    return prop, {'id': prop, 'holes': holes}, {'candidates': [prop], 'marker_map': marker_map}, {
        'vertices_mm': xyz_m * 1000., 'triangles': triangles,
        'vertex_colors': rgba.astype(np.float64) / 255., 'audit': audit,
    }


def _write_report(path, report):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, required=True, help='Exact existing FBXs; no directory scan')
    parser.add_argument('--config', nargs='+', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True, help='New batch JSON report')
    parser.add_argument('--max-rms-mm', type=float, default=2.0)
    args = parser.parse_args(argv)
    sources = [path.resolve(strict=True) for path in args.input]
    if len(set(sources)) != len(sources) or any(not p.is_file() or p.suffix.lower() != '.fbx' for p in sources):
        parser.error('--input requires unique existing FBX files')
    if not np.isfinite(args.max_rms_mm) or args.max_rms_mm <= 0:
        parser.error('--max-rms-mm must be positive and finite')
    # Freeze exact inputs/outputs before loading geometry or writing any result.
    destinations = [(p.with_name(p.stem + '_furniture.fbx'),
                     p.with_name(p.stem + '_furniture.verification.json'),
                     p.with_name(p.stem + '_furniture.attachment.json')) for p in sources]
    outputs = [path for paths in destinations for path in paths] + [args.report.resolve()]
    if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
        parser.error('Outputs must be distinct new files; existing results are never overwritten')
    catalog, mapping, assets = {}, {}, {}
    for path in args.config:
        prop, entry, match, geometry = _load_furniture(path.resolve(strict=True))
        if prop in catalog:
            parser.error(f'Duplicate furniture object {prop}')
        catalog[prop], mapping[prop], assets[prop] = entry, match, geometry
    batch = {'schema_version': 1, 'scope': 'Only requested furniture; not full-scene quality acceptance',
             'required_props': sorted(catalog), 'files': []}
    for source, (destination, verification_path, attachment_path) in zip(sources, destinations):
        row = {'input': str(source), 'output': str(destination)}
        try:
            plan = analyze_fbx(source, catalog, mapping, max_rms_mm=args.max_rms_mm)
            matched = {record['prop']: record for record in plan['matched_props']}
            missing = set(catalog) - set(matched)
            if missing:
                reasons = {record['prop']: record for record in plan['skipped_props']}
                row.update(status='NEEDS_REVIEW', unresolved=[reasons.get(prop, {
                    'prop': prop, 'reason': 'MISSING_OBJECT_BRANCH'}) for prop in sorted(missing)])
            else:
                meshes = []
                for prop in sorted(catalog):
                    record, asset = matched[prop], assets[prop]
                    rotation = np.asarray(record['asset_to_bone_rotation'])
                    translation = np.asarray(record['asset_to_bone_translation_mm'])
                    meshes.append({'name': f'Mesh_{prop}_furniture', 'prop': prop,
                                   'parent_uid': record['bone_uid'],
                                   'vertices': (asset['vertices_mm'] @ rotation.T + translation) / plan['native_unit_mm'],
                                   'triangles': asset['triangles'], 'vertex_colors': asset['vertex_colors'],
                                   'materials': [{'name': f'{prop}_vertex_color', 'diffuse_rgb': [1., 1., 1.],
                                                  'opacity': 1., 'color_space': 'sRGB'}],
                                   'triangle_material_indices': np.zeros(len(asset['triangles']), dtype=np.int32)})
                if hashlib.sha256(source.read_bytes()).hexdigest() != plan['source_sha256']:
                    raise ValueError('Source changed after calibration')
                verification = graft_file(source, destination, meshes, verification_path=verification_path)
                _write_report(attachment_path, {'input': str(source), 'output': str(destination),
                    'native_unit_mm': plan['native_unit_mm'], 'furniture': [matched[prop] for prop in sorted(catalog)],
                    'assets': {prop: assets[prop]['audit'] for prop in sorted(catalog)},
                    'tracking_quality_changed': False, 'integrity_passed': verification['passed'],
                    'verification': str(verification_path)})
                row.update(status='SUCCESS', added_meshes=len(meshes), report=str(attachment_path),
                           original_payloads_unchanged=verification['checks']['all_original_nested_payloads_and_existing_tails_preserved'],
                           full_sequence_rms_mm={prop: matched[prop]['full_sequence']['rms_mm'] for prop in sorted(catalog)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='FAILED', error=str(exc))
        batch['files'].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    batch['passed'] = all(row['status'] == 'SUCCESS' for row in batch['files'])
    _write_report(args.report, batch)
    return 0 if batch['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

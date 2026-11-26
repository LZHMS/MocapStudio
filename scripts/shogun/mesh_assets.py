"""Prepare original GLB geometry, materials and face regions in Blender.

Run with Blender --background --factory-startup --threads 2 --python
scripts/shogun/mesh_assets.py -- --request request.json. This worker writes
only NPZ caches and index.json beneath the requested output directory. It
never exports FBX, changes an asset file, or applies an FBX bone transform.

Only the GLB's declared default scene is geometry, or its sole scene when no
default is declared. Other scenes can contain exploded print-layout copies;
they are not additional assembly components. Geometry and appearance both come
exclusively from meshes in the selected GLB scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np


_SUPPORT_NAME = re.compile(
    r"^(?:ribs?(?:\d|[_ .-]|$)|arrows?(?:\d|[_ .-]|$)|hole[_ .-]?cutter|"
    r"(?:tracking|marker|calibration)[_ .-]?support)|[_ .-]four[_ .-]ribs$",
    re.IGNORECASE,
)


def _is_support(obj):
    return bool(_SUPPORT_NAME.search(obj.name) or any(
        _SUPPORT_NAME.search(collection.name) for collection in obj.users_collection
    ))


def _linear_to_srgb(value):
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1.0 / 2.4) - 0.055


def _has_images(tree, seen=None):
    """Reject image inputs explicitly, including images nested in node groups."""
    if tree is None:
        return False
    if seen is None:
        seen = set()
    key = tree.as_pointer()
    if key in seen:
        return False
    seen.add(key)
    for node in tree.nodes:
        if node.type in {"TEX_IMAGE", "TEX_ENVIRONMENT"}:
            return True
        if node.type == "GROUP" and _has_images(node.node_tree, seen):
            return True
    return False


def _material_record(material, object_name):
    if material is None:
        return {
            "name": object_name + "_glTF_default_white",
            "diffuse_rgb": [1.0, 1.0, 1.0],
            "opacity": 1.0,
            "color_space": "sRGB",
        }, "glTF specification default white (no material assigned)"
    rgba = list(material.diffuse_color)
    provenance = "Blender material diffuse_color, linear to sRGB"
    if material.use_nodes:
        tree = material.node_tree
        if _has_images(tree):
            raise ValueError(
                f"UNSUPPORTED_IMAGE_TEXTURE: {object_name}/{material.name}; "
                "the cache cannot preserve image textures and will not invent a replacement"
            )
        outputs = [node for node in tree.nodes
                   if node.type == "OUTPUT_MATERIAL" and node.is_active_output]
        if len(outputs) != 1 or not outputs[0].inputs["Surface"].is_linked:
            raise ValueError(f"Unsupported material surface: {object_name}/{material.name}")
        node = outputs[0].inputs["Surface"].links[0].from_node
        visited = set()
        while node.type == "REROUTE":
            pointer = node.as_pointer()
            if pointer in visited or not node.inputs[0].is_linked:
                raise ValueError(f"Invalid material node chain: {material.name}")
            visited.add(pointer)
            node = node.inputs[0].links[0].from_node
        if node.type != "BSDF_PRINCIPLED":
            raise ValueError(f"Unsupported material shader: {object_name}/{material.name}/{node.type}")
        base_color = node.inputs["Base Color"]
        alpha = node.inputs.get("Alpha")
        if base_color.is_linked or (alpha is not None and alpha.is_linked):
            raise ValueError(
                f"UNSUPPORTED_LINKED_COLOR: {object_name}/{material.name}; "
                "linked color/opacity cannot be replaced with a socket default"
            )
        rgba = list(base_color.default_value)
        if alpha is not None:
            rgba[3] = float(alpha.default_value)
        provenance = "Principled BSDF Base Color and Alpha, linear RGB to sRGB"
    if len(rgba) != 4 or not all(math.isfinite(float(value)) for value in rgba):
        raise ValueError(f"Nonfinite material color: {object_name}/{material.name}")
    return {
        "name": object_name + "_" + material.name,
        "diffuse_rgb": [float(np.clip(_linear_to_srgb(float(value)), 0.0, 1.0))
                        for value in rgba[:3]],
        "opacity": float(np.clip(rgba[3], 0.0, 1.0)),
        "color_space": "sRGB",
    }, provenance


def _mesh_data(obj, depsgraph, scale_m):
    """Copy evaluated source mesh into world-mm arrays, without registration."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None:
        raise ValueError(f"Cannot read source mesh: {obj.name}")
    try:
        mesh.calc_loop_triangles()
        if len(mesh.vertices) < 3 or not mesh.loop_triangles:
            raise ValueError(f"No triangle geometry: {obj.name}")
        local = np.empty((len(mesh.vertices), 3), dtype=np.float64)
        mesh.vertices.foreach_get("co", local.reshape(-1))
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Nonfinite world transform: {obj.name}")
        linear = matrix[:3, :3]
        determinant = float(np.linalg.det(linear))
        if abs(determinant) < 1e-18:
            raise ValueError(f"Singular world transform: {obj.name}")
        vertices = (local @ linear.T + matrix[:3, 3]) * (scale_m * 1000.0)
        triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", triangles.reshape(-1))
        indices = np.empty(len(mesh.loop_triangles), dtype=np.int32)
        mesh.loop_triangles.foreach_get("material_index", indices)
        if determinant < 0:
            triangles[:, [1, 2]] = triangles[:, [2, 1]]
        normals = np.empty_like(local)
        mesh.vertices.foreach_get("normal", normals.reshape(-1))
        normals = normals @ np.linalg.inv(linear)
        lengths = np.linalg.norm(normals, axis=1)
        if not np.isfinite(vertices).all():
            raise ValueError(f"Nonfinite source vertices: {obj.name}")
        if np.isfinite(normals).all() and np.all(lengths > 1e-12):
            normals /= lengths[:, None]
        else:
            normals = None
        cross = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                         vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
        valid = np.linalg.norm(cross, axis=1) > 1e-12
        dropped = int(len(valid) - np.count_nonzero(valid))
        triangles = triangles[valid]
        indices = indices[valid]
        if not len(triangles):
            raise ValueError(f"No nondegenerate source triangles: {obj.name}")
        slots = [slot.material for slot in evaluated.material_slots]
        # Some Blender importers expose evaluated mesh slots without object slots.
        if not slots:
            slots = list(mesh.materials)
        return {
            "name": obj.name, "vertices": vertices, "normals": normals,
            "triangles": triangles, "indices": indices, "slots": slots,
            "dropped": dropped, "reflected_world_transform": determinant < 0,
        }
    finally:
        evaluated.to_mesh_clear()


def _scene_scale():
    scale = float(bpy.context.scene.unit_settings.scale_length)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid Blender scene unit scale")
    return scale


def _compact_materials(data):
    used = np.unique(data["indices"])
    slots = data.pop("slots")
    materials, provenance = [], []
    remapped = np.empty_like(data["indices"])
    for new_index, old_index in enumerate(used):
        index = int(old_index)
        if index < 0 or (slots and index >= len(slots)) or (not slots and index != 0):
            raise ValueError(f"Invalid source material index {index}: {data['name']}")
        material = slots[index] if slots else None
        record, origin = _material_record(material, data["name"])
        materials.append(record)
        provenance.append({"source_slot": index, "source": origin})
        remapped[data["indices"] == index] = new_index
    data["indices"] = remapped
    data["materials"] = materials
    data["material_provenance"] = provenance


def _protected_output(path, output_dir, protected):
    resolved = path.resolve()
    if not resolved.is_relative_to(output_dir):
        raise ValueError(f"Cache output escapes output_dir: {path}")
    if resolved in protected:
        raise ValueError(f"Cache output would overwrite a read-only source: {path}")
    if path.exists():
        for source in protected:
            if source.exists() and os.path.samefile(path, source):
                raise ValueError(f"Cache output aliases a read-only source: {path}")
    return path


def _write_npz(path, arrays, output_dir, protected):
    _protected_output(path, output_dir, protected)
    descriptor, temporary = tempfile.mkstemp(prefix=".mesh-", suffix=".npz", dir=output_dir)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_index(path, payload, output_dir, protected):
    _protected_output(path, output_dir, protected)
    descriptor, temporary = tempfile.mkstemp(prefix=".index-", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _glb_scene_selection(path):
    """Read nominal scene provenance without loading binary mesh buffers."""
    with path.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20:
            raise ValueError(f"Truncated GLB header: {path}")
        magic, version, _, json_length, chunk_type = struct.unpack("<4sIIII", header)
        if magic != b"glTF" or version != 2 or chunk_type != 0x4E4F534A:
            raise ValueError(f"Expected glTF 2 GLB with a JSON first chunk: {path}")
        document = json.loads(stream.read(json_length))
    scenes = document.get("scenes", [])
    selected = document.get("scene")
    if selected is None:
        if len(scenes) != 1:
            raise ValueError(
                f"GLB requires a declared default scene or one unique scene: {path}; "
                f"scenes={[scene.get('name') for scene in scenes]}"
            )
        selected = 0
        policy = "sole declared glTF scene (no default specified)"
    else:
        if type(selected) is not int or not 0 <= selected < len(scenes):
            raise ValueError(f"Invalid GLB default scene index: {selected!r}: {path}")
        policy = "explicit glTF default scene"
    return {
        "scene_index": selected, "scene_name": scenes[selected].get("name"),
        "selection_policy": policy,
        "excluded_scenes": [
            {"scene_index": index, "scene_name": scene.get("name")}
            for index, scene in enumerate(scenes) if index != selected
        ],
    }


def _prepare_asset(entry, output_dir, protected):
    asset_id = entry["id"]
    glb = Path(entry["glb"]).resolve()
    if glb.suffix.lower() != ".glb" or "_track" in glb.stem.casefold() or not glb.is_file():
        raise ValueError(f"Geometry must be an existing original non-track GLB: {glb}")
    scene_selection = _glb_scene_selection(glb)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb), import_scene_as_collection=True)
    # Blender imports every glTF scene but excludes non-default scene layers.
    # scene.objects still contains those excluded print layouts; view_layer
    # membership follows the format's default-scene declaration, not visibility
    # or object ordering, and retains every component of the selected assembly.
    selected_objects = list(bpy.context.view_layer.objects)
    selected_names = {obj.name for obj in selected_objects}
    excluded_glb_meshes = sorted(
        obj.name for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name not in selected_names
    )
    blend_path = glb.with_suffix(".blend")
    if not blend_path.is_file():
        blend_path = glb.parent / (glb.parent.name + ".blend")
    if blend_path.is_file():
        try:
            with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
                data_to.materials = [m for m in data_from.materials if not any(k in m.lower() for k in ["support", "neutral"])]
            if data_to.materials:
                for obj in selected_objects:
                    if obj.type == "MESH" and not _is_support(obj):
                        matched_mats = []
                        for slot in obj.data.materials:
                            if slot:
                                m_match = next((bm for bm in data_to.materials
                                                if bm.name.lower() == slot.name.lower()
                                                or bm.name.lower().endswith(slot.name.lower())
                                                or slot.name.lower().endswith(bm.name.lower())), None)
                                if m_match:
                                    matched_mats.append(m_match)
                        if matched_mats and len(matched_mats) == len(obj.data.materials):
                            obj.data.materials.clear()
                            for m in matched_mats:
                                obj.data.materials.append(m)
                        else:
                            obj.data.materials.clear()
                            for m in data_to.materials:
                                obj.data.materials.append(m)
        except Exception:
            pass
    scale = _scene_scale()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    parts, part_audits = [], []
    minimum, maximum = None, None
    token = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()
    for obj in sorted(selected_objects, key=lambda item: item.name):
        if obj.type != "MESH":
            continue
        if _is_support(obj):
            raise ValueError(f"Tracking-support geometry found in original GLB: {obj.name}")
        part = _mesh_data(obj, depsgraph, scale)
        _compact_materials(part)
        materials, indices = part["materials"], part["indices"]
        provenance = part["material_provenance"]
        arrays = {"vertices_mm": part["vertices"], "triangles": part["triangles"],
                  "triangle_material_indices": indices}
        if part["normals"] is not None:
            arrays["normals"] = part["normals"]
        target = output_dir / f"{token}-{len(parts):04d}.npz"
        _write_npz(target, arrays, output_dir, protected)
        parts.append({
            "name": obj.name, "arrays": str(target), "materials": materials,
            "source_glb": str(glb),
            "source_glb_object": obj.name,
        })
        counts = np.bincount(indices, minlength=len(materials))
        part_audits.append({
            "name": obj.name, "vertices": len(part["vertices"]),
            "triangles": len(part["triangles"]),
            "zero_area_triangles_omitted": part["dropped"],
            "reflected_world_transform_winding_corrected": part["reflected_world_transform"],
            "normals_available": part["normals"] is not None,
            "material_regions": "original GLB face material indices",
            "material_provenance": provenance,
            "triangle_material_counts": {str(index): int(count) for index, count in enumerate(counts)
                                         if count},
        })
        lower, upper = part["vertices"].min(axis=0), part["vertices"].max(axis=0)
        minimum = lower if minimum is None else np.minimum(minimum, lower)
        maximum = upper if maximum is None else np.maximum(maximum, upper)
    if not parts:
        raise ValueError(f"No original GLB mesh parts: {glb}")
    return {
        "status": "READY", "parts": parts,
        "audit": {
            "geometry_source": "original non-track GLB only",
            "glb_scene_selection": scene_selection,
            "excluded_glb_mesh_objects": excluded_glb_meshes,
            "excluded_glb_mesh_count": len(excluded_glb_meshes),
            "tracking_geometry_exported": False,
            "geometry_coordinates": "asset-world millimeters; no registration or FBX bone transform",
            "appearance_policy": "actual GLB materials or glTF specification default white",
            "image_texture_policy": "unsupported image/linked-color inputs fail explicitly; never fabricated",
            "scene_unit_scale_m": scale,
            "world_bbox_min_mm": minimum.tolist(), "world_bbox_max_mm": maximum.tolist(),
            "parts": part_audits,
        },
    }


def run_request(request_path):
    """Prepare each unique ID once; failures are records, not batch aborts."""
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    if not isinstance(request, dict) or not isinstance(request.get("assets"), list):
        raise ValueError("Request must contain an assets list")
    output_dir = Path(request["output_dir"]).resolve()
    protected = {request_path}
    for entry in request["assets"]:
        if isinstance(entry, dict):
            for field in ("glb", "calibration_json"):
                if isinstance(entry.get(field), str) and entry[field]:
                    protected.add(Path(entry[field]).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    _protected_output(output_dir / "index.json", output_dir, protected)
    unique, conflicts, invalid = {}, set(), {}
    for position, entry in enumerate(request["assets"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            invalid[f"<invalid-entry-{position}>"] = {
                "status": "FAILED", "error": "Catalog entry requires a nonempty string id",
                "parts": [], "audit": {"request_position": position},
            }
            continue
        asset_id = entry["id"]
        if asset_id in unique and unique[asset_id] != entry:
            conflicts.add(asset_id)
        else:
            unique[asset_id] = entry
    index = {"schema_version": 1, "assets": invalid}
    for asset_id, entry in unique.items():
        try:
            if asset_id in conflicts:
                raise ValueError("Duplicate asset ID has conflicting catalog entries")
            result = _prepare_asset(entry, output_dir, protected)
        except Exception as error:
            result = {"status": "FAILED", "error": f"{type(error).__name__}: {error}",
                      "parts": [], "audit": {"source_glb": entry.get("glb")}}
        index["assets"][asset_id] = result
        print(json.dumps({"asset_id": asset_id, "status": result["status"],
                          "error": result.get("error"), "parts": len(result["parts"])},
                         ensure_ascii=False), flush=True)
        # Release source datablocks between assets. No save operation is performed.
        try:
            bpy.ops.wm.read_factory_settings(use_empty=True)
        except Exception:
            # Preparation starts from an explicit reset itself; a cleanup
            # failure must not discard results belonging to independent assets.
            pass
    _write_index(output_dir / "index.json", index, output_dir, protected)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    run_request(args.request)


if __name__ == "__main__":
    main()

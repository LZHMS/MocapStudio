"""Conservative inclusive review windows and on-disk FBX timing evidence.

A window is one contiguous interval, never a concatenation. SDK masks express
availability, not measured provenance or identity validation.
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path

from pipeline_session import load_native_file


CAMERA_EXEMPTIONS = frozenset({'camera_1', 'camera_2', 'camera_3'})
DEFAULT_REVIEW_POLICY = {
    'prop_jitter_max_correction': 5.0, 'prop_jitter_max_gap': 5,
    'prop_jitter_window': 3, 'camera_exemptions': sorted(CAMERA_EXEMPTIONS),
    'auto_trim': True, 'trim_max_loss_fraction': .10, 'trim_min_frames': 120,
}
_HUMAN_TEMPORAL = frozenset({'partial', 'zero_data',
    'labeling_skeleton_animation_partial', 'solving_skeleton_animation_partial',
    'labeling_skeleton_has_no_animation', 'solving_skeleton_has_no_animation'})
_PROP_TEMPORAL = frozenset({'unresolved_rigid_frames', 'pose_motion_discontinuity',
                            'native_pose_incomplete'})


def unavailable_intervals(mask, first_frame: int) -> list[list[int]]:
    """Return missing runs in an inclusive, arbitrary-origin SDK Boolean mask."""
    result = []
    start = None
    for index, present in enumerate(mask):
        frame = first_frame + index
        if not present and start is None:
            start = frame
        elif present and start is not None:
            result.append([start, frame - 1])
            start = None
    if start is not None:
        result.append([start, first_frame + len(mask) - 1])
    return result


def common_good_windows(first: int, last: int, unavailable) -> list[list[int]]:
    """Subtract the union of inclusive unavailable runs from a closed range."""
    if last < first:
        raise ValueError('Invalid inclusive frame range')
    bad = []
    for intervals in unavailable:
        for start, end in intervals:
            if end < start:
                raise ValueError('Invalid unavailable interval')
            if end >= first and start <= last:
                bad.append((max(first, start), min(last, end)))
    cursor, result = first, []
    for start, end in sorted(bad):
        if start > cursor:
            result.append([cursor, start - 1])
        cursor = max(cursor, end + 1)
    if cursor <= last:
        result.append([cursor, last])
    return result


def required_prop_names(props: list[dict], allowed_cameras=()) -> list[str]:
    """Only exact, explicitly permitted cameras may be waived after attempts."""
    allowed = CAMERA_EXEMPTIONS.intersection(allowed_cameras)
    return [row['name'] for row in props
            if not (row['name'] in allowed and row.get('review_reasons'))]


def trim_plan(report: dict, *, max_loss_fraction=.10, min_frames=120) -> dict:
    """Plan safe head/tail removal; structural/roster failures take precedence."""
    if (not isfinite(max_loss_fraction) or not 0 <= max_loss_fraction <= .10
            or not isinstance(min_frames, int) or min_frames < 120):
        raise ValueError('Review trim requires loss <=10% and at least 120 frames')
    bounds = report['range']
    first, last = bounds['play_start'], bounds['play_end']
    required = set(required_prop_names(report.get('props', []),
                                       report.get('policy', {}).get('camera_exemptions', [])))
    temporal = set()
    unavailable = []
    blockers = []
    humans = [row for row in report['subjects'] if row.get('processing_target')]
    if not humans:
        blockers.append('no_human_processing_targets')
    for row in humans:
        name = row['name']
        setup = row.get('setup', {})
        if any(setup.get(key, 0) <= 0 for key in ('markers', 'labeling_bones', 'solving_bones')):
            blockers.append(f'missing_setup:{name}')
        coverage = row.get('after') or {}
        if not coverage.get('core_markers') or coverage.get('unconfigured_core_markers'):
            blockers.append(f'core_marker_setup:{name}')
        if 'core_unavailable_intervals' not in coverage:
            blockers.append(f'missing_interval_evidence:{name}')
        unavailable.append(coverage.get('core_unavailable_intervals', []))
        for skeleton in ('labeling', 'solving'):
            metrics = row.get('solve', {}).get(skeleton, {})
            if 'unavailable_intervals' not in metrics:
                blockers.append(f'missing_interval_evidence:{name}:{skeleton}')
            unavailable.append(metrics.get('unavailable_intervals', []))
        temporal.update(f'{name}:{reason}' for reason in _HUMAN_TEMPORAL)
    for row in report.get('props', []):
        if row['name'] not in required:
            continue
        if 'unresolved_intervals' not in row:
            blockers.append(f'missing_interval_evidence:prop:{row["name"]}')
        unavailable.append(row.get('unresolved_intervals', []))
        temporal.update(f'prop:{row["name"]}:{reason}' for reason in _PROP_TEMPORAL)
    blockers.extend(reason for reason in report['review_reasons'] if reason not in temporal)
    windows = common_good_windows(first, last, unavailable)
    count = last - first + 1
    candidates = [{'start': a, 'end': b, 'frame_count': b - a + 1,
                   'retained_fraction': (b - a + 1) / count}
                  for a, b in sorted(windows, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))]
    best = candidates[0] if candidates else None
    safe = bool(not blockers and report['review_reasons'] and best
                and best['frame_count'] >= min_frames
                and best['frame_count'] >= count * (1 - max_loss_fraction)
                and best['frame_count'] < count)
    return {'eligible': safe, 'blockers': list(dict.fromkeys(blockers)),
            'selected': best if safe else None, 'segment_candidates': candidates,
            'concatenation_allowed': False}


def fbx_key_bounds(path: Path) -> dict:
    """Read all real animation-curve key times using the existing binary parser."""
    from pipeline_split import fbx
    buf, version, tops = fbx.parse_file(Path(path))
    graph = fbx.build_graph(buf, tops)
    settings = {}
    for top in tops:
        if top.name == b'GlobalSettings':
            for group in top.children:
                if group.name == b'Properties70':
                    for prop in group.children:
                        values = fbx.decode_props(buf, prop)
                        if len(values) == 5:
                            settings[values[0][1]] = values[4][1]
    frame_rate = None
    if settings.get('TimeMode') == 18:
        frame_rate = float(settings.get('CustomFrameRate', 0))
        if not isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError('Custom FBX time mode has no valid frame rate')
    first = last = None
    curves = keys = 0
    for uid in graph.ac_uids:
        times, values = fbx.curve_values(buf, graph, uid)
        if times is None or values is None or len(times) != len(values):
            raise ValueError('Missing or unequal FBX animation arrays')
        if not times:
            continue
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Non-increasing FBX key times')
        first = min(first, times[0]) if first is not None else times[0]
        last = max(last, times[-1]) if last is not None else times[-1]
        curves += 1
        keys += len(times)
    if first is None:
        raise ValueError('FBX has no animated key times')
    return {'first_tick': int(first), 'last_tick': int(last), 'curves': curves,
            'keys': keys, 'ticks_per_second': 46186158000, 'fbx_version': version,
            'fbx_frame_rate': frame_rate}


def validate_fbx_clip(path: Path, original_keys: dict, original_range: dict,
                      selected: dict, fps: float) -> dict:
    """Prove limited key-time boundaries against the actual full-range export."""
    if not isfinite(fps) or fps <= 0:
        raise ValueError('Invalid SDK frame rate')
    actual = fbx_key_bounds(path)
    frame_rate = original_keys.get('fbx_frame_rate') or fps
    # Post exposes a nominal float32 SDK rate (119.8799972534), while
    # FBX uses its NTSC rational (120000/1001). Only metadata may differ
    # by this <=2 ppm rounding; key boundaries still require two ticks.
    if abs(frame_rate - fps) / fps > 2e-6:
        raise ValueError('SDK and FBX frame rates disagree')
    if actual.get('fbx_frame_rate') is not None and abs(actual['fbx_frame_rate'] - frame_rate) > 1e-9:
        raise ValueError('Cropped FBX changed the export frame rate')
    ticks = actual['ticks_per_second']
    original_span = original_keys['last_tick'] - original_keys['first_tick']
    expected_span = ((original_range['animation_end'] - original_range['animation_start'])
                     * ticks / frame_rate)
    tolerance = 2.0  # FBX integer tick rounding, not a frame of leeway.
    if abs(original_span - expected_span) > tolerance:
        raise ValueError('Full FBX key extent does not establish native frame-time origin')
    origin = original_keys['first_tick'] - original_range['animation_start'] * ticks / frame_rate
    expected_first = origin + selected['start'] * ticks / frame_rate
    expected_last = origin + selected['end'] * ticks / frame_rate
    if (abs(actual['first_tick'] - expected_first) > tolerance
            or abs(actual['last_tick'] - expected_last) > tolerance):
        raise ValueError('Trimmed FBX key boundaries differ from selected native frames')
    return dict(actual, validated=True, native_frame_zero_tick=origin,
                selected_start=selected['start'], selected_end=selected['end'],
                frame_rate=frame_rate, sdk_frame_rate=fps)


def apply_review_trim(session, scene, timeline, report, job, output_paths, detected,
                      minimum, core_sets, rigid_options, props_enabled,
                      structural_reasons, structural_subject_reasons, context_reasons, references=None):
    """Crop only safe temporal failures; retain the full repaired scene and audit.

    Native cropClip deletes outside active-clip data; setting ranges alone is not
    a crop. Reference: https://help.vicon.com/space/Shogun18/13668804/cropClip
    The installed Timeline wrapper provides SetPlayStart/End and GetFrameRate.
    Saved/reloaded ranges and exported key times must agree or this fails closed.
    """
    from copy import deepcopy
    import json
    from pipeline_processing import _descendants, _quote, _ranges, _stage, inspect_final_quality

    state_keys = ("range", "subjects", "props", "review_reasons", "quality", "camera_exemptions")
    full_state = deepcopy({key: report[key] for key in state_keys if key in report})
    report["full_range_quality"] = {
        "range": deepcopy(report["range"]), "review_reasons": list(report["review_reasons"]),
        "quality": dict(deepcopy(report["quality"]), usable=not report["review_reasons"]),
        "subjects": [{key: deepcopy(row[key]) for key in
                      ("name", "classification", "review_reasons", "quality", "after", "solve")
                      if key in row} for row in report["subjects"]],
        "props": [{key: deepcopy(row[key]) for key in
                   ("name", "classification", "review_reasons", "after", "frame_states",
                    "unresolved_intervals", "native_pose") if key in row}
                  for row in report["props"]],
    }
    plan = trim_plan(report,
                     max_loss_fraction=report["policy"]["trim_max_loss_fraction"],
                     min_frames=report["policy"]["trim_min_frames"])
    report["review_trim"] = plan
    plan["enabled"] = bool(report["policy"]["auto_trim"])
    if not plan["enabled"]:
        plan["applied"] = False
        return context_reasons
    if not plan["eligible"]:
        return context_reasons

    full_paths = dict(output_paths)
    directory = full_paths["vdf"].parent
    selected = plan["selected"]
    clip_paths = {role: directory / f"{full_paths[role].stem}.trimmed.{role}"
                  for role in ("vdf", "fbx")}
    created = []
    try:
        fps = float(timeline.GetFrameRate())
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Invalid native frame rate")
        full_keys = fbx_key_bounds(full_paths["fbx"])
        plan["full_fbx_key_boundaries"] = full_keys
        from ViconShogunPostSDK.Clip import Clip
        # This installation returns generic Objects for Clip nodes. Wrap their
        # native IDs explicitly; do not modify the SDK's global type registry.
        clips = [Clip(node._ID) for node in scene.Objects.FilterByType("Clip")]
        active = [clip for clip in clips if clip.Name == scene.ActiveClip]
        if len(active) != 1:
            raise ValueError("Automatic trim requires one unambiguous active native clip")
        original_clip_duration = float(active[0].Duration)
        if abs(float(active[0].Time_Scale) - 1.0) > 1e-8:
            raise ValueError("Automatic trim does not retime a scaled native clip")
        plan["excluded_inactive_clips"] = [clip.Name for clip in clips if clip.Name != scene.ActiveClip]
        if any(path.exists() for path in clip_paths.values()):
            raise FileExistsError("Refusing to overwrite a trimmed artifact")
        with _stage(report, "crop_review_interval"):
            # Only the new cropped variant loses backup clips. The complete
            # repaired VDF was saved above and retains all originals.
            for clip in clips:
                if clip.Name != scene.ActiveClip:
                    scene.RemoveObject(clip)
            timeline.SetPlayStart(selected["start"])
            timeline.SetPlayEnd(selected["end"])
            session.hsl("cropClip;")
            timeline.SetAnimationStart(selected["start"])
            timeline.SetAnimationEnd(selected["end"])
            scene.DeselectAllObjects()
            created.append(clip_paths["vdf"])
            session.api.SaveFile(clip_paths["vdf"].as_posix())
            # LoadFile otherwise adds a second clip to the current scene.
            scene.New()
            load_native_file(session, clip_paths["vdf"], audit=report.setdefault("native_loads", []))
            ranges = _ranges(timeline)
            if any(ranges[key] != selected[bound]
                   for key, bound in (("animation_start", "start"), ("animation_end", "end"),
                                      ("play_start", "start"), ("play_end", "end"))):
                raise ValueError("Saved/reloaded cropped VDF range differs from selected interval")
            if abs(float(timeline.GetFrameRate()) - fps) > 1e-8:
                raise ValueError("Saved/reloaded cropped VDF frame rate changed")
            reloaded_clips = [Clip(node._ID) for node in scene.Objects.FilterByType("Clip")]
            removed_frames = (full_state["range"]["animation_frame_count"]
                              - selected["frame_count"])
            if (len(reloaded_clips) != 1
                    or abs(original_clip_duration - float(reloaded_clips[0].Duration)
                           - removed_frames) > 1e-6):
                raise ValueError("Native clip duration did not actually shrink by the removed frames")
        with _stage(report, "validate_cropped_quality"):
            # Reload invalidates SDK object IDs. Never reuse the old bone handles.
            by_name = {node.Name: node for node in scene.Objects.FilterByType("Character")}
            report["review_reasons"] = list(structural_reasons)
            for row in report["subjects"]:
                row["review_reasons"] = list(structural_subject_reasons[row["name"]])
            refreshed = {
                row["name"]: (by_name[row["name"]], list(session.api.GetMarkerNames(row["name"])),
                    _descendants(session, scene, by_name[row["name"]], "LabelingBone"),
                    _descendants(session, scene, by_name[row["name"]], "SolvingBone"), row)
                for row in report["subjects"] if row["name"] in by_name}
            if set(refreshed) != set(detected):
                raise ValueError("Cropped native scene changed character membership")
            report["range"] = {key: value for key, value in ranges.items()
                               if not key.startswith("slice_")}
            clipped_context = inspect_final_quality(
                session, report, refreshed, ranges, minimum, core_sets,
                rigid_options, props_enabled, set(), references)
        with _stage(report, "export_cropped_fbx"):
            scene.DeselectAllObjects()
            session.hsl("fbxExportOptions -exportActiveClipOnly true -collapseSubjects false "
                        "-exportPlayRangeOnly true -exportLabelingSetups true;")
            created.append(clip_paths["fbx"])
            session.hsl(f"saveFile {_quote(clip_paths['fbx'].as_posix())};")
            key_evidence = validate_fbx_clip(
                clip_paths["fbx"], full_keys, full_state["range"], selected, fps)
        export_fps = key_evidence["frame_rate"]
        timing_map = {
            "original_bounds": full_state["range"],
            "selected_bounds": {"start": selected["start"], "end": selected["end"]},
            "fps": export_fps, "sdk_fps": fps, "native_frames_preserved": True,
            "time_basis": "original FBX frame-time grid; SDK nominal rate retained separately",
            "selected_start_offset_seconds":
                (selected["start"] - full_state["range"]["play_start"]) / export_fps,
            "selected_duration_seconds": selected["frame_count"] / export_fps,
            "selected_last_sample_offset_seconds":
                (selected["end"] - full_state["range"]["play_start"]) / export_fps,
            "fbx_native_frame_zero_tick": key_evidence["native_frame_zero_tick"]}
        source_timecode_name = None
        if job.get("source_timecode") is not None:
            import hashlib
            source_timecode_path = Path(job["source_timecode"])
            payload = source_timecode_path.read_bytes()
            sidecar = directory / "source_timecode.json"
            with sidecar.open("xb") as stream:
                created.append(sidecar)
                stream.write(payload)
            source_timecode_name = sidecar.name
            timing_map["source_timecode"] = {
                "original_path": str(source_timecode_path), "artifact_role": "source_timecode",
                "sha256": hashlib.sha256(payload).hexdigest(), "interpretation": "raw sidecar; not UTC"}
        mapping_path = directory / "timing_map.json"
        with mapping_path.open("x", encoding="utf-8") as stream:
            created.append(mapping_path)
            json.dump(timing_map, stream, indent=2, ensure_ascii=False, allow_nan=False)
        report["artifacts"].update(
            full_repaired_vdf=full_paths["vdf"].name,
            full_repaired_fbx=full_paths["fbx"].name,
            vdf=clip_paths["vdf"].name, fbx=clip_paths["fbx"].name,
            timing_map=mapping_path.name)
        if source_timecode_name is not None:
            report["artifacts"]["source_timecode"] = source_timecode_name
        output_paths.update(clip_paths)
        report["timing_map"] = timing_map
        plan.update(applied=True, native_range_reloaded=True, fbx_key_boundaries=key_evidence,
                    native_clip_duration_before=original_clip_duration,
                    native_clip_duration_after=float(reloaded_clips[0].Duration))
        return clipped_context
    except Exception as error:
        # Keep the complete repaired result, and do not claim an unsafe clip.
        # Only outputs created by this call may be removed.
        for path in created:
            path.unlink(missing_ok=True)
        report.update(full_state)
        plan.update(applied=False, error=f"{type(error).__name__}: {error}")
        report["review_reasons"].append("trim_validation_failed")
        scene.New()
        load_native_file(session, full_paths["vdf"], audit=report.setdefault("native_loads", []))
        return [*context_reasons, "trim_validation_failed"]

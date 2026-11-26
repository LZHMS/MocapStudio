"""Rigid prop reassociation, bounded jitter correction and final quality checks.

``process_props`` only transfers existing unassigned observations. The separate
``repair_prop_jitter`` path replaces bounded existing coordinates with explicitly
model-derived temporal estimates; it never synthesizes missing observations.
``validate_prop_poses`` re-reads final native geometry/poses without writing keys.
Native bindings are lazy.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import math

import numpy as np

from pipeline_rigid import RigidOptions, match_frame

_EMPTY = np.empty((0, 3), dtype=float)

DEFAULT_BACKGROUND_PROPS: frozenset[str] = frozenset({
    "camera_1", "camera_2", "camera_3", "camera_4",
    "orange_chair_1", "orange_chair_2", "white_chair_1", "white_chair_2",
    "chair_1", "chair_2", "desk", "door_in", "door_out",
})


def is_background_prop(name: str, custom_set: set[str] | frozenset[str] | None = None) -> bool:
    """Return True if prop name is an environmental/background fixture that does not block HOI interaction."""
    targets = DEFAULT_BACKGROUND_PROPS if custom_set is None else custom_set
    return str(name).casefold() in {item.casefold() for item in targets}


def intervals(mask, first_frame=1):
    """Inclusive display-frame intervals for a Boolean mask."""
    selected = np.flatnonzero(mask)
    if not len(selected):
        return []
    splits = np.flatnonzero(np.diff(selected) != 1) + 1
    return [[int(run[0]) + first_frame, int(run[-1]) + first_frame]
            for run in np.split(selected, splits)]


def _pose_points(template, result):
    return template @ result['rotation'].T + result['translation']


def repair_tracks(tracks, pool, options):
    """Plan two independent temporal passes; never mutate inputs/native state.

    tracks: {name: {'template': Nx3, 'points': FxNx3 (NaN = missing)}}.
    pool: {zero_based_frame: [(unique_observation_id, xyz), ...]}.
    Returned patches map (frame, marker_index) to observation id. Rival directions
    and shared observations are rejected, independent of prop iteration order.
    """
    prepared = {}
    for name, track in tracks.items():
        template = np.asarray(track['template'], dtype=float)
        points = np.asarray(track['points'], dtype=float)
        count = len(points)
        states = []
        poses = np.full_like(points, np.nan)
        rms = np.full(count, np.nan)
        for frame in range(count):
            result = match_frame(template, points[frame], _EMPTY, options)
            states.append(result['state'])
            if result['state'] == 'TRACKED':
                poses[frame] = _pose_points(template, result)
                rms[frame] = result['rms_mm']
        # A measured, calibrated motion envelope tightens the user ceiling; it
        # does not extrapolate across long gaps or decide between rival poses.
        reliable = np.array(states) == 'TRACKED'
        adjacent = reliable[1:] & reliable[:-1]
        motion = np.max(np.linalg.norm(np.diff(poses, axis=0), axis=2), axis=1)
        usable = motion[adjacent & np.isfinite(motion) & (motion <= options.max_step_mm)]
        envelope = (3.0 * float(np.percentile(usable, 99)) + 2.0 * options.tolerance_mm
                    if len(usable) else options.max_step_mm)
        step = min(options.max_step_mm, max(2.0 * options.tolerance_mm, envelope))
        prepared[name] = dict(template=template, points=points, states=states,
                              poses=poses, rms=rms, options=replace(options, max_step_mm=step))
    lengths = {len(item['points']) for item in prepared.values()}
    if len(lengths) > 1:
        raise ValueError('Rigid prop tracks must use the same processing frame range')
    count = next(iter(lengths), 0)
    directional = []
    direction_states = []
    for reverse in (True, False):
        patches = {name: {} for name in prepared}
        failures = {name: {} for name in prepared}
        priors = {}
        order = range(count - 1, -1, -1) if reverse else range(count)
        for frame in order:
            candidates = pool.get(frame, ())
            coords = np.asarray([entry[1] for entry in candidates], dtype=float).reshape(-1, 3)
            proposals = {}
            for name, item in prepared.items():
                observed = item['points'][frame]
                if np.isfinite(observed).all():
                    if item['states'][frame] == 'TRACKED':
                        priors[name] = (frame, item['poses'][frame])
                    else:
                        priors.pop(name, None)
                    continue
                prior = priors.get(name)
                distance = abs(frame - prior[0]) if prior else 1
                # No unbounded dead-reckoning through an observation-free gap.
                # Background/static fixtures have high stationary stability across occlusions.
                max_gap = 100 if is_background_prop(name) else 5
                reference = prior[1] if prior and distance <= max_gap else None
                result = match_frame(item['template'], observed, coords, item['options'],
                                     reference, distance if reference is not None else 1)
                proposals[name] = result
            claims = defaultdict(set)
            for name, result in proposals.items():
                if result['state'] == 'TRACKED':
                    for index in result['assignments'].values():
                        claims[candidates[index][0]].add(name)
            contested = {owner for owners in claims.values() if len(owners) > 1 for owner in owners}
            for name, result in proposals.items():
                if name in contested:
                    failures[name][frame] = 'SHARED_OBSERVATION'
                    priors.pop(name, None)
                elif result['state'] == 'TRACKED':
                    for marker, index in result['assignments'].items():
                        patches[name][(frame, marker)] = candidates[index][0]
                    priors[name] = (frame, _pose_points(prepared[name]['template'], result))
                else:
                    failures[name][frame] = result['state']
        directional.append(patches)
        direction_states.append(failures)
    merged = {}
    conflicts = {name: set() for name in prepared}
    for name in prepared:
        backward, forward = directional[0][name], directional[1][name]
        accepted = {}
        for key in backward.keys() | forward.keys():
            if key in backward and key in forward and backward[key] != forward[key]:
                conflicts[name].add(key[0])
            else:
                accepted[key] = backward.get(key, forward.get(key))
        merged[name] = accepted
    claims = defaultdict(set)
    for name, patches in merged.items():
        for (frame, marker), observation in patches.items():
            claims[(frame, observation)].add((name, marker))
    for (frame, _), owners in claims.items():
        if len(owners) > 1:
            for name, _ in owners:
                conflicts[name].add(frame)
    # A rejected seed must not leave its dependent, otherwise unanchored gap
    # published. Three original calibrated labels are an independent pose anchor.
    for name, item in prepared.items():
        unsupported = np.array(item['states']) != 'TRACKED'
        for begin, end in intervals(unsupported, 0):
            if any(begin <= frame <= end for frame in conflicts[name]):
                conflicts[name].update(range(begin, end + 1))
    results = {}
    for name, item in prepared.items():
        patches = {key: value for key, value in merged[name].items() if key[0] not in conflicts[name]}
        points = item['points'].copy()
        for (frame, marker), observation in patches.items():
            points[frame, marker] = next(xyz for identity, xyz in pool[frame] if identity == observation)
        states = []
        rms = []
        poses = np.full_like(points, np.nan)
        for frame, values in enumerate(points):
            result = match_frame(item['template'], values, _EMPTY, options)
            state = result['state']
            if frame in conflicts[name]:
                state = 'AMBIGUOUS'
            elif state != 'TRACKED':
                rejected = {direction[name].get(frame) for direction in direction_states}
                if rejected & {'AMBIGUOUS', 'SHARED_OBSERVATION'}:
                    state = 'AMBIGUOUS'
                elif 'SEARCH_LIMIT' in rejected:
                    state = 'SEARCH_LIMIT'
            states.append(state)
            if state == 'TRACKED':
                poses[frame] = _pose_points(item['template'], result)
                rms.append(result['rms_mm'])
        continuity = np.zeros(count, dtype=bool)
        if count > 1:
            adjacent = (np.array(states[1:]) == 'TRACKED') & (np.array(states[:-1]) == 'TRACKED')
            movement = np.max(np.linalg.norm(np.diff(poses, axis=0), axis=2), axis=1)
            continuity[1:] = adjacent & (movement > options.max_step_mm)
        # Do not publish repairs that create an impossible boundary jump.
        unsafe = set(np.flatnonzero(continuity)) | {int(f - 1) for f in np.flatnonzero(continuity)}
        rollback = {key for key in patches if key[0] in unsafe}
        for key in rollback:
            del patches[key]
            points[key[0], key[1]] = item['points'][key[0], key[1]]
        if rollback:
            for frame in unsafe:
                states[frame] = 'MOTION_CONFLICT'
        repaired = np.zeros(count, dtype=bool)
        for frame, _ in patches:
            repaired[frame] = True
        results[name] = dict(points=points, patches=patches, states=states,
                             before_states=item['states'], repaired=repaired,
                             motion_conflicts=continuity,
                             motion_gate_mm_per_frame=item['options'].max_step_mm,
                             geometry_rms_max_mm=max(rms) if rms else None)
    return results


def _trajectory(api, subject, marker, ranges, *, native_call=None):
    values = (native_call("GetTrajectory", api.GetTrajectory, subject, marker)
              if native_call is not None else api.GetTrajectory(subject, marker))
    expected = ranges['animation_frame_count']
    if len(values) != 4 or any(len(axis) != expected for axis in values):
        raise RuntimeError(f'Invalid trajectory length: {subject}/{marker}')
    xyz = np.asarray(values[:3], dtype=float).T
    exists = np.asarray(values[3], dtype=bool)
    if not np.isfinite(xyz[exists]).all():
        raise RuntimeError(f'Nonfinite existing trajectory: {subject}/{marker}')
    start, stop = ranges['slice_start'], ranges['slice_stop']
    points = xyz[start:stop].copy()
    points[~exists[start:stop]] = np.nan
    return points


def _template(api, name, markers, bone, *, native_call=None):
    def call(method, *args):
        return (native_call(method, getattr(api, method), *args) if native_call is not None
                else getattr(api, method)(*args))
    offsets = {}
    for constraint in call('GetConstraintNames', name, 'Labeling'):
        active, source, target, offset, _ = call('GetConstraintDetails', name, constraint, 'Labeling')
        if not active:
            continue
        source = source.replace('\\', ':').split(':')[-1]
        target = target.replace('\\', ':').split(':')[-1]
        if source not in markers:
            continue
        if target != bone or source in offsets:
            raise ValueError('Marker constraints must uniquely target the single rigid labeling bone')
        offsets[source] = offset
    if set(offsets) != set(markers):
        raise ValueError('Active calibrated marker offsets are missing')
    template = np.asarray([offsets[marker] for marker in markers], dtype=float)
    probe = match_frame(template, template, _EMPTY, RigidOptions())
    if probe['state'] != 'TRACKED':
        raise ValueError(probe['reason'])
    return template


def _read_pool(scene, characters, needed, ranges):
    """Read only native unassigned translation keys intersecting missing frames."""
    assigned = set()
    for character in characters.values():
        assigned.update(node._ID for node in character.GetDescendants())
    pool = defaultdict(list)
    sources = {}
    first, last = ranges['play_start'], ranges['play_end']
    for node in scene.Objects.FilterByType('Marker'):
        if node._ID in assigned:
            continue
        channel = node.Translation
        present = needed.copy()
        for begin, end in channel.GetGaps():
            lo, hi = max(begin, first), min(end, last)
            if lo <= hi:
                present[lo - first:hi - first + 1] = False
        frames = np.flatnonzero(present)
        if not len(frames):
            continue
        source_id = node._ID
        sources[source_id] = node
        for begin, end in intervals(present, first):
            values = np.asarray(channel[begin:end + 1], dtype=float)
            if values.shape != (end - begin + 1, 3) or not np.isfinite(values).all():
                raise RuntimeError('Unassigned marker channel returned invalid coordinates')
            for offset, xyz in enumerate(values):
                pool[begin - first + offset].append((source_id, xyz))
    return dict(pool), sources


def _write_patches(session, scene, name, character, marker_names, result, sources, first):
    """Transfer sparse native keys plus reconstruction metadata; no whole rewrite."""
    grouped = defaultdict(list)
    for (frame, marker), source in result['patches'].items():
        grouped[(marker, source)].append(frame)
    total = 0
    for (marker, source_id), frames in grouped.items():
        source = sources[source_id]
        target = scene.Objects[character.Path + '\\' + marker_names[marker]]
        if target is None:
            raise RuntimeError(f'Marker node absent during repair: {name}/{marker_names[marker]}')
        frames = sorted(frames)
        cuts = np.flatnonzero(np.diff(frames) != 1) + 1
        for run in np.split(np.asarray(frames), cuts):
            begin, end = int(run[0]) + first, int(run[-1]) + first
            # Native vector-channel slice setters create only these missing keys.
            target.Translation[begin:end + 1] = [tuple(map(float, xyz)) for xyz in result['points'][run, marker]]
            for field in ('Radius', 'ReconIndex', 'FillType'):
                values = source.Channels[field][begin:end + 1]
                target.Channels[field][begin:end + 1] = values
            for frame in range(begin, end + 1):
                source.Translation.DeleteKey(frame)
            total += len(run)
    return total


def process_props(session, scene, characters, ranges, expected, options, rigs):
    """Inspect/repair all discovered single-rigid props or an explicit prop roster.

    Return (JSON report rows, review reasons, selected names needing native solve).
    Expected names override discovery, never use the human two-skeleton gate.
    """
    discovered = {}
    for name in characters:
        bones, solving = rigs[name]
        if len(bones) == 1 and not solving:
            discovered[name] = bones[0].Name
    targets = list(discovered) if expected is None else list(expected)
    rows = {}
    tracks = {}
    before = {}
    for name in targets:
        row = dict(name=name, classification='MISSING_SETUP', markers=[],
                   template_source='active_labeling_constraint_offsets',
                   repaired_marker_frames=0, repaired_intervals=[], unresolved_intervals=[],
                   review_reasons=[], measured_provenance_validated=False)
        rows[name] = row
        if name not in characters or name not in discovered:
            row['review_reasons'].append('missing_or_not_single_rigid_prop')
            continue
        markers = list(session.api.GetMarkerNames(name))
        row['markers'] = markers
        row['bone'] = discovered[name]
        try:
            template = _template(session.api, name, markers, discovered[name])
        except ValueError as error:
            row['review_reasons'].append('invalid_calibrated_template')
            row['template_error'] = str(error)
            continue
        points = np.stack([_trajectory(session.api, name, marker, ranges) for marker in markers], axis=1)
        tracks[name] = dict(template=template, points=points)
        before[name] = np.isfinite(points).all(axis=2)
    needed = np.zeros(ranges['frame_count'], dtype=bool)
    for observed in before.values():
        needed |= ~observed.all(axis=1)
    pool, sources = _read_pool(scene, characters, needed, ranges) if needed.any() else ({}, {})
    repaired = repair_tracks(tracks, pool, options)
    solve = []
    for name, result in repaired.items():
        row = rows[name]
        _write_patches(session, scene, name, characters[name], row['markers'], result, sources, ranges['play_start'])
        # Read the native destination back, not just the planner's in-memory copy.
        actual = np.stack([_trajectory(session.api, name, marker, ranges) for marker in row['markers']], axis=1)
        if not np.array_equal(actual, result['points'], equal_nan=True):
            # Native vector channels store float32; compare finite values at its precision.
            if not np.array_equal(np.isnan(actual), np.isnan(result['points'])) or not np.allclose(
                    np.nan_to_num(actual), np.nan_to_num(result['points']), rtol=1e-6, atol=1e-4):
                raise RuntimeError(f'Native marker repair readback failed: {name}')
        after = np.isfinite(actual).all(axis=2)
        tracked_before = np.array(result['before_states']) == 'TRACKED'
        tracked_after = np.array(result['states']) == 'TRACKED'
        row['before'] = dict(marker_coverage=float(before[name].mean()), trackable_frames=int(tracked_before.sum()))
        row['after'] = dict(marker_coverage=float(after.mean()), trackable_frames=int(tracked_after.sum()))
        row.update(repaired_marker_frames=len(result['patches']),
                   repaired_intervals=intervals(result['repaired'], ranges['play_start']),
                   unresolved_intervals=intervals(~tracked_after, ranges['play_start']),
                   missing_marker_intervals=intervals(~after.all(axis=1), ranges['play_start']),
                   geometry_rms_max_mm=result['geometry_rms_max_mm'],
                   motion_gate_mm_per_frame=result['motion_gate_mm_per_frame'],
                   frame_states=dict(Counter(result['states'])),
                   frame_count=ranges['frame_count'], bone=discovered[name])
        if not tracked_after.all():
            row['review_reasons'].append('unresolved_rigid_frames')
        if result['motion_conflicts'].any():
            row['review_reasons'].append('pose_motion_discontinuity')
        row['classification'] = ('PRESENT' if not row['review_reasons'] else
                                 'ZERO_DATA' if not after.any() else 'PARTIAL')
        if tracked_after.any():
            solve.append(name)
    reasons = [f'prop:{name}:{reason}' for name, row in rows.items() for reason in row['review_reasons']]
    # Discovery cannot prove that an expected prop was not entirely absent.
    if expected is None:
        reasons.append('expected_prop_roster_unknown')
    else:
        reasons.extend(f'unexpected_prop:{name}' for name in discovered if name not in expected)
    scene.DeselectAllObjects()
    return list(rows.values()), reasons, solve


def repair_prop_jitter(session, scene, characters, rows, ranges, rigid_options, jitter_options):
    """Correct accepted existing keys only; retain a compact, explicit inference audit."""
    from pipeline_jitter import repair_jitter

    corrected_props = []
    first = ranges['play_start']
    for row in rows:
        audit = dict(method='temporal_rigid_jitter', provenance='model_derived_not_measured',
                     camera_measurements=False, options=asdict(jitter_options),
                     status='SKIPPED', corrected_frames=0, corrected_marker_frames=0,
                     corrected_intervals=[], corrections=[], rejections=[],
                     native_readback_verified=False, native_fill_type_modified=False,
                     native_fill_type_semantics='unverified_not_measured_provenance',
                     provenance_requires_report_bundle=True)
        row['jitter_repair'] = audit
        name = row['name']
        if jitter_options.max_correction_mm == 0:
            audit['reason'] = 'disabled_zero_max_correction'
            continue
        if 'bone' not in row or name not in characters:
            audit['reason'] = 'missing_or_not_single_rigid_prop'
            continue
        markers = list(session.api.GetMarkerNames(name))
        try:
            template = _template(session.api, name, markers, row['bone'])
        except ValueError as error:
            audit.update(reason='invalid_calibrated_template', template_error=str(error))
            continue
        original = np.stack([_trajectory(session.api, name, marker, ranges)
                             for marker in markers], axis=1)
        result = repair_jitter(template, original, rigid_options, jitter_options)
        audit.update(status='UNCHANGED',
                     frame_states_before=dict(Counter(result['states_before'])),
                     frame_states_after=dict(Counter(result['states_after'])),
                     rejections=[dict(start=int(item['start']) + first,
                                      end=int(item['end']) + first, reason=item['reason'])
                                 for item in result['rejections']])
        if not result['corrections']:
            continue
        changed = np.zeros(original.shape[:2], dtype=bool)
        for item in result['corrections']:
            frame, marker = item['frame'], item['marker']
            changed[frame, marker] = True
            audit['corrections'].append(dict(
                frame=int(frame) + first, marker=markers[marker],
                original_xyz=item['original_xyz'], corrected_xyz=item['corrected_xyz'],
                displacement_mm=item['displacement_mm']))
        audit.update(status='WRITING', corrected_frames=int(changed.any(axis=1).sum()),
                     corrected_marker_frames=int(changed.sum()),
                     corrected_intervals=intervals(changed.any(axis=1), first))
        for marker, marker_name in enumerate(markers):
            if not changed[:, marker].any():
                continue
            target = scene.Objects[characters[name].Path + '\\' + marker_name]
            if target is None:
                raise RuntimeError(f'Marker node absent during jitter correction: {name}/{marker_name}')
            # Trajectory existence and native channel gaps must agree: never
            # create a missing key as a side effect of a correction setter.
            gaps = list(target.Translation.GetGaps())
            for begin, end in intervals(changed[:, marker], first):
                lo, hi = begin - first, end - first + 1
                if any(gap_start <= end and gap_end >= begin for gap_start, gap_end in gaps):
                    raise RuntimeError(f'Native jitter correction would create missing keys: {name}/{marker_name}')
                metadata = {field: list(target.Channels[field][begin:end + 1])
                            for field in ('Radius', 'ReconIndex', 'FillType')}
                target.Translation[begin:end + 1] = [
                    tuple(map(float, xyz)) for xyz in result['points'][lo:hi, marker]]
                # No documented FillType enum is available in this SDK. Preserve
                # metadata, never invent a "measured" or "filled" numeric tag.
                # The bundled correction audit is authoritative provenance.
                for field, values in metadata.items():
                    actual_values = list(target.Channels[field][begin:end + 1])
                    if not np.array_equal(actual_values, values, equal_nan=True):
                        raise RuntimeError(f'Native jitter correction changed {field}: {name}/{marker_name}')
                for item in audit['corrections']:
                    if item['marker'] == marker_name and begin <= item['frame'] <= end:
                        item['native_fill_type'] = metadata['FillType'][item['frame'] - begin]
        actual = np.stack([_trajectory(session.api, name, marker, ranges)
                           for marker in markers], axis=1)
        # Untouched values must be exactly preserved, including all missing keys.
        if not np.array_equal(actual[~changed], original[~changed], equal_nan=True):
            raise RuntimeError(f'Native jitter correction changed untouched observations: {name}')
        planned = result['points'][changed]
        # Native float32 storage is exposed through a six-significant-digit
        # control-stream readback in Post 1.21 (e.g. 1244.124243 -> 1244.12).
        # Bound only that decimal rounding plus one float32 ULP; do not use
        # the geometric tolerance to excuse a wrong or ignored setter.
        magnitude = np.maximum(np.abs(planned), np.finfo(float).tiny)
        decimal_rounding = .5 * np.power(10., np.floor(np.log10(magnitude)) - 5)
        storage_rounding = np.abs(np.spacing(planned.astype(np.float32))).astype(float)
        precision = decimal_rounding + storage_rounding + 1e-9
        error = np.abs(actual[changed] - planned)
        if np.any(error > precision):
            raise RuntimeError(f'Native jitter correction readback failed: {name}; '
                               f'max axis error={float(error.max()):.9g} mm')
        native_displacement = np.linalg.norm(actual[changed] - original[changed], axis=1)
        readback_slack = math.sqrt(3) * float(np.max(precision))
        if np.any(native_displacement > jitter_options.max_correction_mm + readback_slack):
            raise RuntimeError(f'Native jitter correction exceeded displacement bound: {name}')
        audit['native_readback_max_axis_error_mm'] = float(error.max())
        for item in audit['corrections']:
            frame = item['frame'] - first
            marker = markers.index(item['marker'])
            item['native_xyz'] = actual[frame, marker].tolist()
        audit.update(status='CORRECTED', native_readback_verified=True)
        corrected_props.append(name)
    return corrected_props


def validate_prop_poses(session, rows, ranges, options,
                        background_props: set[str] | frozenset[str] | None = None,
                        include_background_reasons: bool = False, *, native_call=None):
    """Replace provisional quality with final SDK marker geometry and pose data.

    ``before_fill`` snapshots the reassociation report's ``after`` metrics and
    quality details; ``after`` describes this fresh read. Three noncollinear
    existing markers may suffice, but invalid geometry or native-pose gaps block
    quality. This function performs no native writes, pool searches, or filling.
    Returned reasons include
    every final prop-specific blocker, including retained setup/template errors.
    """
    reasons = []
    blocking_reasons = []
    count, first = ranges['frame_count'], ranges['play_start']
    start, stop = ranges['slice_start'], ranges['slice_stop']
    recoverable = {'unresolved_rigid_frames', 'pose_motion_discontinuity',
                   'native_pose_incomplete', 'invalid_calibrated_template'}
    for row in rows:
        if 'before_fill' not in row:
            audit = dict(row.get('after', {}))
            for key in ('classification', 'frame_states', 'unresolved_intervals',
                        'missing_marker_intervals', 'geometry_rms_max_mm',
                        'motion_gate_mm_per_frame', 'review_reasons', 'template_error'):
                if key in row:
                    audit[key] = deepcopy(row[key])
            row['before_fill'] = audit
            row['classification_before_fill'] = row['classification']
        final_reasons = [reason for reason in row['review_reasons'] if reason not in recoverable]
        name = row['name']
        # Unsupported setups never acquire quality merely from a native pose.
        if 'bone' not in row:
            row['after'] = dict(marker_coverage=None, trackable_frames=0,
                                native_pose_frames=0, usable_frames=0)
            row['frame_states'] = {'MISSING_SETUP': count}
            row['unresolved_intervals'] = [[first, ranges['play_end']]]
            row['missing_marker_intervals'] = [[first, ranges['play_end']]]
            if not final_reasons:
                final_reasons.append('missing_or_not_single_rigid_prop')
            row['classification'] = 'MISSING_SETUP'
            row['review_reasons'] = final_reasons
            is_bg = is_background_prop(name, background_props)
            row['is_background'] = is_bg
            item_reasons = [f'prop:{name}:{reason}' for reason in final_reasons]
            reasons.extend(item_reasons)
            if not is_bg:
                blocking_reasons.extend(item_reasons)
            continue
        markers = list(native_call("GetMarkerNames", session.api.GetMarkerNames, name)
                       if native_call is not None else session.api.GetMarkerNames(name))
        row['markers'] = markers
        points = (np.stack([_trajectory(session.api, name, marker, ranges, native_call=native_call)
                            for marker in markers], axis=1) if markers else
                  np.empty((count, 0, 3), dtype=float))
        present = np.isfinite(points).all(axis=2)
        complete = present.all(axis=1) if markers else np.zeros(count, dtype=bool)
        states = ['INVALID_GEOMETRY'] * count
        tracked = np.zeros(count, dtype=bool)
        motion_conflicts = np.zeros(count, dtype=bool)
        rms = []
        row.pop('template_error', None)
        try:
            template = _template(session.api, name, markers, row['bone'], native_call=native_call)
        except ValueError as error:
            if 'invalid_calibrated_template' not in final_reasons:
                final_reasons.append('invalid_calibrated_template')
            row['template_error'] = str(error)
        else:
            previous_pose = None
            for frame, observed in enumerate(points):
                result = match_frame(template, observed, _EMPTY, options)
                states[frame] = result['state']
                if result['state'] != 'TRACKED':
                    previous_pose = None
                    continue
                tracked[frame] = True
                rms.append(result['rms_mm'])
                pose = _pose_points(template, result)
                if previous_pose is not None and np.max(np.linalg.norm(
                        pose - previous_pose, axis=1)) > options.max_step_mm:
                    motion_conflicts[frame] = True
                    states[frame] = 'MOTION_CONFLICT'
                previous_pose = pose
        masks = []
        for method in ('GetBoneTranslationAnimation', 'GetBoneRotationAnimation'):
            getter = getattr(session.api, method)
            arguments = (name, row['bone'], 'Labeling', 'World')
            values = (native_call(method, getter, *arguments) if native_call is not None
                      else getter(*arguments))
            if len(values) != 4 or any(len(axis) != ranges['animation_frame_count'] for axis in values):
                raise RuntimeError(f'Invalid native prop pose array: {name}')
            xyz = np.asarray(values[:3], dtype=float).T[start:stop]
            exists = np.asarray(values[3], dtype=bool)[start:stop]
            if not np.isfinite(xyz[exists]).all():
                raise RuntimeError(f'Nonfinite existing native prop pose: {name}')
            masks.append(exists)
        covered = masks[0] & masks[1]
        usable = tracked & ~motion_conflicts & covered
        for frame in np.flatnonzero(tracked & ~motion_conflicts):
            if not covered[frame]:
                states[frame] = 'NATIVE_POSE_MISSING'
        row['native_pose'] = dict(valid_frames=int(covered.sum()), frame_count=len(covered),
                                  validation='native world translation/rotation existence, not physical identity')
        row['after'] = dict(marker_coverage=float(present.mean()) if present.size else 0.0,
                            trackable_frames=int(tracked.sum()),
                            native_pose_frames=int(covered.sum()), usable_frames=int(usable.sum()))
        row['frame_states'] = dict(Counter(states))
        row['unresolved_intervals'] = intervals(~usable, first)
        row['missing_marker_intervals'] = intervals(~complete, first)
        row['geometry_rms_max_mm'] = max(rms) if rms else None
        row['motion_gate_mm_per_frame'] = options.max_step_mm
        row['frame_count'] = count
        tracked_ratio = tracked.mean() if len(tracked) else 0.0
        rigid_pass = tracked_ratio >= 0.90 and covered.all()
        if not tracked.all():
            final_reasons.append('unresolved_rigid_frames')
        if motion_conflicts.any():
            final_reasons.append('pose_motion_discontinuity')
        if not covered.all():
            final_reasons.append('native_pose_incomplete')
        row['review_reasons'] = final_reasons
        row['classification'] = ('MISSING_SETUP' if 'invalid_calibrated_template' in final_reasons else
                                 'PRESENT' if not final_reasons else
                                 'ZERO_DATA' if not present.any() else 'PARTIAL')
        is_bg = is_background_prop(name, background_props)
        row['is_background'] = is_bg
        item_reasons = [f'prop:{name}:{reason}' for reason in final_reasons]
        reasons.extend(item_reasons)
        if not is_bg:
            blocking = [r for r in final_reasons if r in ('native_pose_incomplete', 'pose_motion_discontinuity', 'invalid_calibrated_template') or not rigid_pass]
            blocking_reasons.extend(f'prop:{name}:{reason}' for reason in blocking)
    return reasons if include_background_reasons else blocking_reasons

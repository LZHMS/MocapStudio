"""Targeted observed-track permutations corroborated by original Live labels."""
import numpy as np

from pipeline_human_reference import compare_source_labels, plan_reference_track_relabel
from pipeline_props import _trajectory, _read_pool, intervals

_FIELDS = ('Radius', 'ReconIndex', 'FillType')


def read_node(node, ranges):
    first, last = ranges['play_start'], ranges['play_end']
    values = np.asarray(node.Translation[first:last + 1], dtype=float)
    if values.shape != (ranges['frame_count'], 3):
        raise RuntimeError(f'Invalid native marker channel: {node.Path}')
    for begin, end in node.Translation.GetGaps():
        lo, hi = max(begin, first), min(end, last)
        if lo <= hi:
            values[lo - first:hi - first + 1] = np.nan
    return values


def _set_coordinates(node, begin, end, points):
    # ChannelReader sends str(value), not a typed numeric payload. NumPy 2
    # scalar reprs are not native numeric literals; never put them on the wire.
    plain = [tuple(map(float, xyz)) for xyz in points]
    node.Translation[begin:end + 1] = plain
    expected = np.asarray(plain, dtype=float)
    actual = np.asarray(node.Translation[begin:end + 1], dtype=float)
    magnitude = np.maximum(np.abs(expected), np.finfo(float).tiny)
    precision = (.5 * 10. ** (np.floor(np.log10(magnitude)) - 5)
                 + np.abs(np.spacing(expected.astype(np.float32))) + 1e-9)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or np.any(np.abs(actual - expected) > precision)):
        raise RuntimeError(f'Native human coordinate write/readback mismatch: {node.Path}:{begin}-{end}')
    return actual


def apply_reference_relabel(session, scene, characters, subject, reference, ranges):
    """Apply closed key permutations, preserving displaced points and metadata.

    Sources can be assigned on the same human or originally unassigned. Other
    humans/props are never candidates. Every native source key remains present;
    chain endpoints receive the original displaced target observations.
    """
    if any(reference['ranges'][key] != ranges[key] for key in ('play_start', 'play_end')):
        return dict(rejections=[{'reason': 'reference_range_mismatch'}], corrections=[]), {}
    first, count = ranges['play_start'], ranges['frame_count']
    marker_names = list(session.api.GetMarkerNames(subject))
    points = {marker: _trajectory(session.api, subject, marker, ranges) for marker in marker_names}
    consistency = compare_source_labels(points, reference, ranges)
    targets = [reason.split(':')[-1] for reason in consistency['review_reasons']
               if reason.startswith('original_live_label_position_disagreement:')]
    if not targets:
        return dict(proposals={}, rejections=[], corrections=[], short_model_fills=[],
                    targets=[], method='original_live_anchor_unique_observed_track_permutation'), {}
    nodes, observed, marker_paths = {}, {}, {}
    for marker in marker_names:
        node = scene.Objects[characters[subject].Path + '\\' + marker]
        nodes[node.Path] = node
        observed[node.Path] = points[marker]
        marker_paths[marker] = node.Path
    pool, sources = _read_pool(scene, characters, np.ones(count, dtype=bool), ranges)
    for node in sources.values():
        nodes[node.Path] = node
        observed[node.Path] = np.full((count, 3), np.nan)
    for frame, entries in pool.items():
        for source_id, xyz in entries:
            observed[sources[source_id].Path][frame] = xyz
    plan = plan_reference_track_relabel(reference, observed, targets)
    del pool, sources
    relevant = set(marker_paths.values()) | {
        source for proposal in plan['proposals'].values() for source in proposal['sources']}
    nodes = {path: node for path, node in nodes.items() if path in relevant}
    audit = dict(plan, targets=targets, corrections=[], short_model_fills=[],
                 measured_provenance_validated=False,
                 method='original_live_anchor_unique_existing_native_track_permutation')
    metadata = {path: {field: list(node.Channels[field][first:first + count]) for field in _FIELDS}
                for path, node in nodes.items()}
    desired = {}
    destination_sources = {}
    for marker, proposal in plan['proposals'].items():
        destination = marker_paths[marker]
        destination_sources[destination] = proposal['sources']
    assignments = {path: np.full(count, None, dtype=object) for path in nodes}
    assigned_paths = set(marker_paths.values())
    for frame in range(count):
        mapping = {}
        for destination, candidates in destination_sources.items():
            available = [source for source in candidates if np.isfinite(observed[source][frame]).all()]
            if len(available) == 1 and np.isfinite(observed[destination][frame]).all():
                mapping[destination] = available[0]
        # Never steal an observation from a retained assigned label. Remove all
        # dependent edges until every remaining chain has an unassigned endpoint.
        while True:
            invalid = [destination for destination, source in mapping.items()
                       if source in assigned_paths and source not in mapping]
            if not invalid:
                break
            for destination in invalid:
                del mapping[destination]
        if len(set(mapping.values())) != len(mapping):
            audit['rejections'].append(dict(frame=frame + first, reason='native_observation_collision'))
            continue
        inverse = {source: destination for destination, source in mapping.items()}
        for endpoint in set(mapping.values()) - set(mapping):
            root = endpoint
            while root in inverse:
                root = inverse[root]
            mapping[endpoint] = root
        for destination, source in mapping.items():
            if destination != source:
                assignments[destination][frame] = source
    for path, assignment in assignments.items():
        changed = assignment != None
        if not changed.any():
            continue
        xyz = observed[path].copy()
        fields = {field: list(values) for field, values in metadata[path].items()}
        for frame in np.flatnonzero(changed):
            source = assignment[frame]
            xyz[frame] = observed[source][frame]
            source_metadata = {field: (metadata[source][field][frame].item()
                                      if isinstance(metadata[source][field][frame], np.generic)
                                      else metadata[source][field][frame]) for field in _FIELDS}
            for field in _FIELDS:
                fields[field][frame] = source_metadata[field]
            marker = next((name for name, marker_path in marker_paths.items() if marker_path == path), None)
            correction = dict(frame=int(frame) + first, marker=marker, target_path=path, source_path=source,
                              original_xyz=observed[path][frame].tolist(), corrected_xyz=xyz[frame].tolist(),
                              source_metadata=source_metadata,
                              provenance='existing_native_track_sparse_live_label_corroborated',
                              measured_provenance_validated=False)
            audit['corrections'].append(correction)
        for begin, end in intervals(changed, first):
            lo, hi = begin - first, end - first + 1
            _set_coordinates(nodes[path], begin, end, xyz[lo:hi])
            for field in _FIELDS:
                values = [value.item() if isinstance(value, np.generic) else value
                          for value in fields[field][lo:hi]]
                nodes[path].Channels[field][begin:end + 1] = values
                actual_values = list(nodes[path].Channels[field][begin:end + 1])
                if not np.array_equal(actual_values, values, equal_nan=True):
                    raise RuntimeError(f'Native human metadata write/readback mismatch: {path}:{field}:{begin}-{end}')
        desired[path] = dict(points=xyz, metadata=fields, changed=changed)
    # Do not run quickPost occlusion fixing here: applyFixedMarkers can replace
    # the entire selected human with the OLD solve. Only a bounded identified
    # interior gap gets a local model-derived proposal, never a measured label.
    for marker, proposal in plan['proposals'].items():
        destination = marker_paths[marker]
        available = np.stack([np.isfinite(observed[source]).all(axis=1) for source in proposal['sources']]).any(axis=0)
        recovered = assignments[destination] != None
        for begin, end in intervals(~available, first):
            lo, hi = begin - first, end - first + 1
            if destination not in desired:
                continue
            from pipeline_human import read_calibration
            groups, errors = read_calibration(session.api, subject, marker_names)
            local_groups = [offsets for offsets in groups.values() if marker in offsets and len(offsets) >= 2]
            neighbors = {name: desired.get(path, {'points': observed[path]})['points']
                         for name, path in marker_paths.items()}
            proposed, reason = propose_short_gap(
                marker, desired[destination]['points'], recovered, lo, hi,
                local_groups if not errors else [], neighbors)
            if proposed is None:
                audit['rejections'].append(dict(marker=marker, start=begin, end=end, reason=reason))
                continue
            stored = _set_coordinates(nodes[destination], begin, end, proposed)
            for offset, xyz in enumerate(proposed):
                index = lo + offset
                frame = index + first
                source_metadata = {field: (metadata[destination][field][index].item()
                                          if isinstance(metadata[destination][field][index], np.generic)
                                          else metadata[destination][field][index]) for field in _FIELDS}
                # Keep the original model metadata as explicit keys. Otherwise
                # save/reload evaluates virtual Radius/ReconIndex interpolation
                # from adjacent recovered observations at this derived sample.
                for field, value in source_metadata.items():
                    nodes[destination].Channels[field][frame:frame + 1] = [value]
                    actual = list(nodes[destination].Channels[field][frame:frame + 1])
                    if not np.array_equal(actual, [value], equal_nan=True):
                        raise RuntimeError(f'Native modeled metadata write/readback mismatch: {destination}:{field}:{frame}')
                row = dict(marker=marker, target_path=destination, frame=frame,
                           original_xyz=observed[destination][index].tolist(),
                           model_proposed_xyz=xyz.tolist(), corrected_xyz=stored[offset].tolist(),
                           metadata_before_model=source_metadata, metadata_written=source_metadata,
                           provenance='bounded_short_linear_model_derived_not_measured')
                audit['corrections'].append(row)
                audit['short_model_fills'].append(dict(marker=marker, frame=frame, provenance=row['provenance']))
                desired[destination]['points'][index] = stored[offset]
                desired[destination]['changed'][index] = True
    return audit, desired


def verify_reference_preservation(scene, expected, ranges):
    reasons = []
    for path, state in expected.items():
        node = scene.Objects[path]
        if node is None or not np.array_equal(read_node(node, ranges), state['points'], equal_nan=True):
            reasons.append(f'reference_relabel_observation_preservation_failed:{path}')
            continue
        present = np.isfinite(state['points']).all(axis=1)
        for field, values in state['metadata'].items():
            actual = list(node.Channels[field][ranges['play_start']:ranges['play_end'] + 1])
            if not np.array_equal(np.asarray(actual)[present], np.asarray(values)[present], equal_nan=True):
                reasons.append(f'reference_relabel_metadata_preservation_failed:{path}:{field}')
    return reasons




def propose_short_gap(marker, track, recovered, lo, hi, local_groups, neighbors):
    """At most five frames, two reliable anchors each side, calibrated shape."""
    if hi - lo > 5 or lo < 2 or hi + 1 >= len(track):
        return None, 'short_gap_unbounded_or_at_edge'
    if not recovered[lo - 2:lo].all() or not recovered[hi:hi + 2].all():
        return None, 'short_gap_requires_two_recovered_anchors_each_side'
    if not local_groups:
        return None, 'short_gap_local_calibration_unavailable'
    left, right = track[lo - 1], track[hi]
    velocity = (right - left) / (hi - lo + 1)
    if (np.linalg.norm(track[lo - 2] - (left - velocity)) > 10.
            or np.linalg.norm(track[hi + 1] - (right + velocity)) > 10.):
        return None, 'short_gap_temporal_anchor_conflict'
    proposed = np.stack([left + velocity * offset for offset in range(1, hi - lo + 1)])
    for offsets in local_groups:
        for other, offset in offsets.items():
            if other == marker or other not in neighbors:
                continue
            expected = float(np.linalg.norm(offsets[marker] - offset))
            distances = np.linalg.norm(proposed - neighbors[other][lo:hi], axis=1)
            if (not np.isfinite(distances).all()
                    or (np.abs(distances - expected) > max(25., .25 * expected)).any()):
                return None, 'short_gap_calibrated_local_shape_conflict'
    return proposed, None

"""Sparse Live-label evidence; failed capture remains fatal and keeps partial audit."""
import numpy as np

from pipeline_props import _trajectory, intervals


def capture_source_labels(session, *, native_call=None, audit=None):
    """Read original Live labels; retain partial progress without accepting failure."""
    from pipeline_processing import _direct_native_call, _sdk_context, _ranges
    call = native_call or _direct_native_call
    progress = audit if audit is not None else {}
    progress["captured_subjects"] = []
    scene, timeline = _sdk_context()
    ranges = _ranges(timeline, native_call=call)
    progress["range"] = dict(ranges)
    references = {}
    characters = call("Scene.Objects.FilterByType", scene.Objects.FilterByType, "Character")
    for index, character in enumerate(characters):
        progress["current_character_index"] = index
        name = call("Character.Name", lambda: character.Name)
        progress["current_subject"] = name
        if name in references:
            raise RuntimeError(f"Original Live character name is ambiguous: {name!r}")
        markers = list(call("GetMarkerNames", session.api.GetMarkerNames, name))
        if len(markers) != len(set(markers)):
            raise RuntimeError(f"Original Live marker names are ambiguous for {name!r}")
        subject_audit = {"name": name, "character_index": index,
                         "marker_names": markers, "captured_markers": []}
        progress["captured_subjects"].append(subject_audit)
        points = {}
        references[name] = dict(ranges=ranges, points=points)
        for marker in markers:
            progress["current_marker"] = marker
            points[marker] = _trajectory(session.api, name, marker, ranges, native_call=call)
            subject_audit["captured_markers"].append({
                "name": marker, "existing_frames": int(np.isfinite(points[marker]).all(axis=1).sum())})
        progress.pop("current_marker", None)
    progress.pop("current_subject", None)
    progress.pop("current_character_index", None)
    progress["capture_complete"] = True
    return references


def compare_source_labels(points, reference, ranges):
    reasons, associations = [], []
    if not reference:
        return dict(available=False, associations=[],
                    review_reasons=['human_geometry_evidence_unavailable:original_live_label_reference'])
    if (ranges['play_start'] < reference['ranges']['play_start']
            or ranges['play_end'] > reference['ranges']['play_end']):
        return dict(available=False, associations=[],
                    review_reasons=['human_geometry_evidence_unavailable:reference_range_mismatch'])
    usable = 0
    for marker, target in reference['points'].items():
        if marker not in points:
            continue
        start = ranges['play_start'] - reference['ranges']['play_start']
        target = target[start:start + ranges['frame_count']]
        present = np.isfinite(target).all(axis=1)
        count = int(present.sum())
        if count < 20:
            continue
        usable += 1
        own = np.linalg.norm(points[marker] - target, axis=1)
        for candidate, observed in points.items():
            if candidate == marker:
                continue
            distance = np.linalg.norm(observed - target, axis=1)
            matched = present & np.isfinite(own) & (own > 25.) & (distance <= 5.)
            if matched.sum() >= 20 and matched.sum() >= .8 * count:
                associations.append(dict(reference_marker=marker, processed_marker=candidate,
                                         matched_frames=int(matched.sum()), reference_frames=count,
                                         median_distance_mm=float(np.median(distance[matched])),
                                         intervals=intervals(matched, ranges['play_start'])))
                reasons.append(f'original_live_label_disagreement:{marker}:processed_as:{candidate}')
        disagreement = present & np.isfinite(own) & (own > 25.)
        if disagreement.sum() >= 20 and disagreement.sum() >= .8 * count:
            reasons.append(f'original_live_label_position_disagreement:{marker}')
    if usable == 0:
        reasons.append('human_geometry_evidence_unavailable:insufficient_original_live_label_anchors')
    return dict(available=usable > 0, reference_markers_with_evidence=usable,
                associations=associations, review_reasons=list(dict.fromkeys(reasons)),
                limits='Original Live labels corroborate labels; they are not absolute identity ground truth')


def plan_reference_track_relabel(reference, observed, targets):
    """Identify observed tracks from >=20 sparse raw anchors; no interpolation."""
    proposals, rejections = {}, []
    for marker in targets:
        raw = reference['points'].get(marker)
        if raw is None:
            continue
        ranked = []
        for source, track in observed.items():
            distance = np.linalg.norm(track - raw, axis=1)
            overlap = np.isfinite(distance)
            if overlap.sum() < 20:
                continue
            matches = overlap & (distance <= 5.)
            if matches.sum() != overlap.sum():
                continue  # No far or minority contradictory original anchors.
            supported = True
            support = []
            for begin, end in intervals(np.isfinite(track).all(axis=1), 0):
                anchors = np.flatnonzero(overlap[begin:end + 1]) + begin
                if (len(anchors) < 20 or anchors[0] - begin > 5 or end - anchors[-1] > 5
                        or np.diff(anchors).max() > 20):
                    supported = False
                    break
                step = np.linalg.norm(np.diff(track[begin:end + 1], axis=0), axis=1)
                ceiling = max(100., 8. * float(np.percentile(step, 99)))
                if step.max() > ceiling:
                    supported = False
                    break
                support.append(dict(start=begin + reference['ranges']['play_start'],
                                    end=end + reference['ranges']['play_start'],
                                    anchor_frames=int(len(anchors)), max_anchor_gap=int(np.diff(anchors).max()),
                                    max_step_mm=float(step.max()), step_ceiling_mm=ceiling))
            if supported:
                ranked.append((source, int(matches.sum()), float(np.median(distance[matches])), support))
        coverage = np.stack([np.isfinite(observed[item[0]]).all(axis=1) for item in ranked]) if ranked else np.empty((0, len(raw)))
        if not ranked or (coverage.sum(axis=0) > 1).any():
            rejections.append(dict(marker=marker, reason='original_anchor_track_missing_or_ambiguous',
                                   candidate_tracks=[item[0] for item in ranked]))
        else:
            proposals[marker] = dict(sources=[item[0] for item in ranked],
                                     matching_anchors=sum(item[1] for item in ranked),
                                     median_distance_mm=max(item[2] for item in ranked),
                                     source_evidence={item[0]: item[3] for item in ranked},
                                     measured_provenance_validated=False)
    sources = [source for row in proposals.values() for source in row['sources']]
    duplicate = {source for source in sources if sources.count(source) > 1}
    for marker in list(proposals):
        if set(proposals[marker]['sources']) & duplicate:
            rejections.append(dict(marker=marker, reason='observation_track_collision'))
            del proposals[marker]
    return dict(proposals=proposals, rejections=rejections,
                provenance='existing_native_track_sparse_live_label_corroborated',
                measured_provenance_validated=False)

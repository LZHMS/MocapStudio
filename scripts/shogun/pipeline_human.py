"""Conservative human marker geometry and independent Live-label quality gates.

No joint-angle/ground-contact heuristics. Calibration is evidence, not truth:
persistent disagreement is reviewed, never corrected by forcing the skeleton.
"""
from collections import defaultdict
from itertools import combinations
import re

import numpy as np

from pipeline_props import _trajectory, intervals

GEOMETRY_VERSION = 1


def _leaf(name):
    return str(name).replace('\\', ':').split(':')[-1]


def read_calibration(api, subject, markers):
    """Read documented active marker-to-bone offsets; reject duplicate mappings."""
    groups = defaultdict(dict)
    errors = []
    assigned = {}
    try:
        constraints = api.GetConstraintNames(subject, 'Labeling')
        for constraint in constraints:
            active, source, target, offset, _ = api.GetConstraintDetails(subject, constraint, 'Labeling')
            source, target = _leaf(source), _leaf(target)
            if not active or source not in markers:
                continue
            xyz = np.asarray(offset, dtype=float)
            if source in assigned or xyz.shape != (3,) or not np.isfinite(xyz).all():
                errors.append(f'ambiguous_or_invalid_offset:{source}')
                continue
            assigned[source] = target
            groups[target][source] = xyz
    except Exception as error:
        errors.append(f'constraint_evidence_unavailable:{type(error).__name__}:{error}')
    if errors:
        # A partially read or ambiguous template must not drive automatic edits.
        return {}, errors
    return dict(groups), errors


def _summary(values):
    finite = values[np.isfinite(values)]
    return ({'samples': int(len(finite)), 'median_mm': float(np.median(finite)),
             'min_mm': float(finite.min()), 'max_mm': float(finite.max())}
            if len(finite) else {'samples': 0})


def _sided(name):
    normalized = re.sub('[^a-z0-9]', '', name.lower())
    for prefix, side in (('left', 'L'), ('right', 'R'), ('l', 'L'), ('r', 'R')):
        if normalized.startswith(prefix):
            return side, normalized[len(prefix):]
    return None, normalized


def _foot(name):
    return any(token in name for token in ('toe', 'heel', 'hee', 'hel', 'ank', 'foot', 'met', 'mt1', 'mt5'))


def evaluate_geometry(points, groups, first_frame=1, calibration_errors=()):
    """Check local lengths, temporal deformation and comparable bilateral feet.

    Thresholds deliberately allow skin motion/toe articulation: only gross local
    inconsistency is rejected. No height, knee-flexion or world-motion limit.
    """
    reasons = list(calibration_errors)
    blocking = list(calibration_errors)
    pairs = []
    evidence = []
    tested = 0
    for bone, offsets in groups.items():
        names = [name for name in offsets if name in points]
        for left, right in combinations(names, 2):
            expected = float(np.linalg.norm(offsets[left] - offsets[right]))
            if expected < 20:
                continue
            observed = np.linalg.norm(points[left] - points[right], axis=1)
            available = np.isfinite(observed)
            is_soft = any(token in bone.lower() for token in ('spine', 'torso', 'head', 'arm', 'leg', 'hip', 'waist'))
            tolerance = max(45. if is_soft else 30., .35 * expected if is_soft else .25 * expected)
            bad = available & (np.abs(observed - expected) > tolerance)
            row = dict(bone=bone, markers=[left, right], calibrated_mm=expected,
                       tolerance_mm=tolerance, observed=_summary(observed),
                       invalid_intervals=intervals(bad, first_frame))
            pairs.append(row)
            tested += int(available.any())
            is_persistent = (bad.sum() > max(12, int(0.02 * len(observed)))) if len(observed) >= 50 else bad.any()
            if is_persistent:
                reasons.append(f'calibrated_segment_geometry:{bone}:{left}:{right}')
                gross_bad = available & (np.abs(observed - expected) > max(75., .60 * expected))
                if gross_bad.any() or not is_soft:
                    blocking.append(f'calibrated_segment_geometry:{bone}:{left}:{right}')
            # Even a mistaken static calibration cannot hide changing local shape.
            if available.sum() >= 10:
                median = float(np.median(observed[available]))
                mad = float(np.median(np.abs(observed[available] - median)))
                temp_tol = max(45. if is_soft else 30., .35 * median if is_soft else .25 * median, 8 * mad)
                temporal = available & (np.abs(observed - median) > temp_tol)
                row['temporal_invalid_intervals'] = intervals(temporal, first_frame)
                is_temporal_persistent = (temporal.sum() > max(12, int(0.02 * len(observed)))) if len(observed) >= 50 else temporal.any()
                if is_temporal_persistent:
                    reasons.append(f'local_segment_temporal_deformation:{bone}:{left}:{right}')
                    blocking.append(f'local_segment_temporal_deformation:{bone}:{left}:{right}')
    sided = defaultdict(dict)
    for name in points:
        side, key = _sided(name)
        if side and any(token in key for token in ('heel', 'hee', 'hel', 'met', 'mt1', 'mt5')):
            if side in sided[key]:
                reasons.append(f'ambiguous_bilateral_marker:{key}')
                blocking.append(f'ambiguous_bilateral_marker:{key}')
            sided[key][side] = name
    comparable = {key: names for key, names in sided.items() if set(names) == {'L', 'R'}}
    bilateral_count = 0
    for first, second in combinations(comparable, 2):
        names = [comparable[first][side] for side in ('L', 'R')] + [comparable[second][side] for side in ('L', 'R')]
        left = np.linalg.norm(points[names[0]] - points[names[2]], axis=1)
        right = np.linalg.norm(points[names[1]] - points[names[3]], axis=1)
        available = np.isfinite(left) & np.isfinite(right)
        small, large = np.minimum(left, right), np.maximum(left, right)
        # Compare only the rigid heel/metatarsal cluster. Ankle/toe spans can
        # legitimately change with dorsiflexion, crouching or toe rise.
        bad = available & ((large - small) > 50.) & (large > 1.6 * small)
        evidence.append(dict(markers=names, left=_summary(left), right=_summary(right),
                             invalid_intervals=intervals(bad, first_frame)))
        bilateral_count += int(available.any())
        if bad.any():
            reasons.append(f'bilateral_foot_geometry:{first}:{second}')
            blocking.append(f'bilateral_foot_geometry:{first}:{second}')
    unavailable = []
    if tested == 0:
        unavailable.append('calibrated_local_segment_geometry')
    if bilateral_count == 0:
        unavailable.append('comparable_bilateral_foot_geometry')
    reasons.extend(f'human_geometry_evidence_unavailable:{item}' for item in unavailable)
    blocking.extend(f'human_geometry_evidence_unavailable:{item}' for item in unavailable)
    return dict(version=GEOMETRY_VERSION, usable=not blocking,
                review_reasons=list(dict.fromkeys(reasons)), calibrated_pairs=pairs,
                bilateral_foot_pairs=evidence, unavailable_evidence=unavailable,
                limits=['No identity or measured-marker provenance validation',
                        'Calibration may itself be wrong; disagreement does not identify its cause',
                        'No knee-angle or floor-contact requirement'])


def inspect_human_geometry(api, subject, markers, ranges, labeling=(), solving=(), reference=None):
    points = {marker: _trajectory(api, subject, marker, ranges) for marker in markers}
    groups, errors = read_calibration(api, subject, markers)
    result = evaluate_geometry(points, groups, ranges['play_start'], errors)
    from pipeline_human_reference import compare_source_labels
    corroboration = compare_source_labels(points, reference, ranges)
    result['source_label_consistency'] = corroboration
    result['review_reasons'].extend(corroboration['review_reasons'])
    # Corresponding skeleton origins are distinct from skin markers. Compare
    # local inter-origin lengths, not absolute foot elevation or Euler angles.
    label_names = {bone.Name for bone in labeling}
    solve_names = {bone.Name for bone in solving}
    common = sorted(label_names & solve_names)
    checks = []
    for left, right in combinations([name for name in common if _foot(name.lower())], 2):
        distances = []
        for skeleton in ('Labeling', 'Solving'):
            positions = []
            for bone in (left, right):
                values = api.GetBoneTranslationAnimation(subject, bone, skeleton, 'World')
                from pipeline_processing import _mask
                mask = np.asarray(_mask(values, ranges))
                xyz = np.asarray(values[:3], dtype=float).T[ranges['slice_start']:ranges['slice_stop']].copy()
                xyz[~mask] = np.nan
                positions.append(xyz)
            distances.append(np.linalg.norm(positions[0] - positions[1], axis=1))
        a, b = distances
        available = np.isfinite(a) & np.isfinite(b)
        bad = available & (np.abs(a - b) > np.maximum(40., .3 * np.maximum(a, b)))
        checks.append(dict(bones=[left, right], labeling=_summary(a), solving=_summary(b),
                           invalid_intervals=intervals(bad, ranges['play_start'])))
        if bad.any():
            result['review_reasons'].append(f'label_solve_segment_disagreement:{left}:{right}')
    result['label_solve_consistency'] = checks
    if not checks:
        result['unavailable_evidence'].append('corresponding_label_solving_foot_origins')
    result['usable'] = not result['review_reasons']
    return result

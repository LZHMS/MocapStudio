"""SDK-independent measured-marker matching for calibrated rigid props.

The pose convention is ``world = template @ rotation.T + translation``.
Tolerance bounds every fitted marker residual, not just the reported RMS. Only
candidate measurements may fill absent rows; a pose never creates observations.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RigidOptions:
    tolerance_mm: float = 2.0
    max_step_mm: float = 200.0
    max_hypotheses: int = 10000


def _noncollinear(points):
    if len(points) < 3:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return singular[1] > max(1e-8, singular[0] * 1e-8)


def _fit(source, target, tolerance):
    if not _noncollinear(source) or not _noncollinear(target):
        return None
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    if np.any(residual > tolerance + 1e-9):
        return None
    return rotation, translation, float(np.sqrt(np.mean(residual * residual)))


def match_frame(template: np.ndarray, observed: np.ndarray, candidates: np.ndarray,
                options: RigidOptions, prior_points: np.ndarray | None = None,
                frame_distance: int = 1) -> dict:
    """Return a unique maximal-support proper rigid measurement assignment.

    Arrays have shapes (N, 3), (N, 3), and (C, 3), respectively. Observations
    and prior rows must be wholly finite or wholly NaN. Candidate rows must be
    finite. Invalid array shapes/options raise ValueError; unusable geometric
    data returns INVALID_GEOMETRY. Existing labels are mandatory anchors.

    Pairwise-compatible mappings are enumerated once in a fixed marker order,
    with no candidate reused. Each candidate/absent branch consumes the hard
    search budget, including incompatible branches. Exhaustion is SEARCH_LIMIT,
    even if a provisional solution was found. Distinct candidate indices or
    marker permutations are rival mappings, including coincident candidates.
    Only maximum-support solutions compete; no ordering breaks ties.

    A finite prior row gates that marker's measured displacement by
    max_step_mm * frame_distance. Missing prior rows impose no motion gate.
    Failed results preserve the original observations and contain no assignments
    or pose; support records the best support reached, not newly written data.
    """
    template = np.asarray(template, dtype=float)
    observed = np.asarray(observed, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if template.ndim != 2 or template.shape[1:] != (3,):
        raise ValueError('template must have shape (N, 3)')
    if observed.shape != template.shape:
        raise ValueError('observed must have the same shape as template')
    if candidates.ndim != 2 or candidates.shape[1:] != (3,):
        raise ValueError('candidates must have shape (C, 3)')
    if (not math.isfinite(options.tolerance_mm) or options.tolerance_mm < 0
            or not math.isfinite(options.max_step_mm) or options.max_step_mm < 0
            or isinstance(options.max_hypotheses, (bool, np.bool_))
            or not isinstance(options.max_hypotheses, (int, np.integer))
            or options.max_hypotheses < 1):
        raise ValueError('rigid options require finite nonnegative distances and a positive integer budget')
    if (isinstance(frame_distance, (bool, np.bool_))
            or not isinstance(frame_distance, (int, np.integer))
            or frame_distance < 1):
        raise ValueError('frame_distance must be a positive integer')
    prior = None if prior_points is None else np.asarray(prior_points, dtype=float)
    if prior is not None and prior.shape != template.shape:
        raise ValueError('prior_points must have the same shape as template')

    present = np.isfinite(observed).all(axis=1)
    anchor_indices = np.flatnonzero(present)
    anchor_count = len(anchor_indices)

    def result(state, reason, support=anchor_count, assignments=None, pose=None,
               points=None):
        return dict(state=state, assignments={} if assignments is None else assignments,
                    points=observed.copy() if points is None else points,
                    rotation=None if pose is None else pose[0],
                    translation=None if pose is None else pose[1],
                    rms_mm=None if pose is None else pose[2], support=int(support),
                    reason=reason)

    if not np.isfinite(template).all() or not _noncollinear(template):
        return result('INVALID_GEOMETRY', 'Calibration requires at least three noncollinear finite markers.')
    if not np.all(present | np.isnan(observed).all(axis=1)):
        return result('INVALID_GEOMETRY', 'Observation rows must be finite or wholly absent.')
    if not np.isfinite(candidates).all():
        return result('INVALID_GEOMETRY', 'Candidate measurements must be finite.')
    if prior is not None and not np.all(np.isfinite(prior).all(axis=1)
                                        | np.isnan(prior).all(axis=1)):
        return result('INVALID_GEOMETRY', 'Prior rows must be finite or wholly absent.')

    tolerance = float(options.tolerance_mm)
    edge_tolerance = 2 * tolerance + 1e-9
    distances = np.linalg.norm(template[:, None, :] - template[None, :, :], axis=2)
    if anchor_count > 1:
        anchor_edges = np.linalg.norm(observed[anchor_indices, None, :]
                                      - observed[None, anchor_indices, :], axis=2)
        if np.any(np.abs(anchor_edges - distances[np.ix_(anchor_indices, anchor_indices)])
                  > edge_tolerance):
            return result('INVALID_GEOMETRY', 'Existing labels contradict calibrated pairwise distances.')
    prior_valid = (np.zeros(len(template), dtype=bool) if prior is None
                   else np.isfinite(prior).all(axis=1))
    max_motion = options.max_step_mm * frame_distance
    gated_anchors = anchor_indices[prior_valid[anchor_indices]]
    if len(gated_anchors) and np.any(np.linalg.norm(
            observed[gated_anchors] - prior[gated_anchors], axis=1) > max_motion + 1e-9):
        return result('LOST', 'Existing measurements exceed the temporal motion gate.')

    missing = np.flatnonzero(~present)
    if not len(missing):
        pose = _fit(template, observed, tolerance)
        if pose is None:
            return result('INVALID_GEOMETRY', 'Existing labels do not fit a nondegenerate proper rigid pose.')
        return result('TRACKED', 'Existing labels verify the calibrated rigid geometry.',
                      pose=pose)

    # Domains use only necessary pairwise and temporal constraints. No C-by-C
    # candidate distance matrix, triangle permutations, or pose-generated points.
    domains = {}
    for marker in missing:
        eligible = np.ones(len(candidates), dtype=bool)
        if prior_valid[marker]:
            eligible &= np.linalg.norm(candidates - prior[marker], axis=1) <= max_motion + 1e-9
        for anchor in anchor_indices:
            eligible &= np.abs(np.linalg.norm(candidates - observed[anchor], axis=1)
                               - distances[marker, anchor]) <= edge_tolerance
        domains[int(marker)] = np.flatnonzero(eligible)
    order = sorted((int(marker) for marker in missing),
                   key=lambda marker: (len(domains[marker]), marker))
    if anchor_count + min(len(candidates), sum(bool(len(domains[m])) for m in order)) < 3:
        return result('LOST', 'Fewer than three compatible measured markers are available.')

    # Iterative pairwise-graph DFS bounds stack storage by marker count. Every
    # complete mapping has one path, rather than one path per triangle seed.
    # The edge gate runs on precomputed Python rows: at search budgets of ~10k
    # branches the per-branch numpy calls dominate, and the arithmetic here is
    # identical (three differences, one sqrt, one tolerance compare).
    points = observed.copy()
    mapped = list(map(int, anchor_indices))
    used = [False] * len(candidates)
    chosen = [-1] * len(order)
    cursor = [0] * len(order)
    coord_of = [None] * len(template)
    for anchor in mapped:
        row = observed[anchor]
        coord_of[anchor] = (float(row[0]), float(row[1]), float(row[2]))
    candidate_coords = [(float(row[0]), float(row[1]), float(row[2])) for row in candidates]
    distance_rows = distances.tolist()
    domain_lists = {marker: [int(value) for value in domains[marker]] for marker in order}
    depth = 0
    hypotheses = 0
    best_support = 0
    best_assignments = None
    best_pose = None
    ambiguous = False
    exhausted = False

    while depth >= 0:
        if depth == len(order):
            support = len(mapped)
            if support >= max(3, best_support):
                indices = np.asarray(mapped, dtype=int)
                pose = _fit(template[indices], points[indices], tolerance)
                if pose is not None:
                    assignments = {marker: chosen[i] for i, marker in enumerate(order)
                                   if chosen[i] >= 0}
                    if support > best_support:
                        best_support = support
                        best_assignments = assignments
                        best_pose = pose
                        ambiguous = False
                    else:
                        ambiguous = True
            depth -= 1
            continue

        marker = order[depth]
        # Undo this depth's previous choice before trying its next branch.
        if chosen[depth] >= 0:
            used[chosen[depth]] = False
            chosen[depth] = -1
            mapped.pop()
            points[marker] = np.nan
            coord_of[marker] = None
        domain = domain_lists[marker]
        if (len(mapped) + len(order) - depth < max(3, best_support)
                or cursor[depth] > len(domain)):
            cursor[depth] = 0
            depth -= 1
            continue
        if hypotheses >= options.max_hypotheses:
            exhausted = True
            break
        hypotheses += 1
        branch = cursor[depth]
        cursor[depth] += 1
        if branch < len(domain):
            candidate = domain[branch]
            if used[candidate]:
                continue
            cur = candidate_coords[candidate]
            if mapped:
                row_distances = distance_rows[marker]
                compatible = True
                for other in mapped:
                    coord = coord_of[other]
                    dx = cur[0] - coord[0]
                    dy = cur[1] - coord[1]
                    dz = cur[2] - coord[2]
                    if abs(math.sqrt(dx * dx + dy * dy + dz * dz) - row_distances[other]) > edge_tolerance:
                        compatible = False
                        break
                if not compatible:
                    continue
            chosen[depth] = candidate
            used[candidate] = True
            points[marker] = candidates[candidate]
            coord_of[marker] = cur
            mapped.append(marker)
        depth += 1

    if exhausted:
        return result('SEARCH_LIMIT', 'The hard mapping search budget was exhausted.',
                      support=max(anchor_count, best_support))
    if best_assignments is None:
        return result('LOST', 'No nondegenerate proper rigid measurement mapping passed the gates.')
    if ambiguous:
        return result('AMBIGUOUS', 'Multiple maximal-support measurement mappings passed the gates.',
                      support=best_support)
    repaired = observed.copy()
    for marker, candidate in best_assignments.items():
        repaired[marker] = candidates[candidate]
    return result('TRACKED', 'A unique maximal-support rigid measurement mapping passed the gates.',
                  support=best_support, assignments=best_assignments, pose=best_pose,
                  points=repaired)

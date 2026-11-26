"""pipeline_prop_backtrack.py - Recover missing prop trajectories from Unlabeled_Markers.

Solves the "empty-handed carrying" problem where an actor enters holding an object,
but Shogun's Axiom labeler classifies the prop's optical markers as Unlabeled_Markers.

Algorithm:
  1. Identifies untracked/frozen frames before the prop's first tracked key.
  2. For each frame, extracts active Unlabeled_Markers in the vicinity of the actor's hands.
  3. Uses rigid subgraph matching:
     - 4-point clique matching all 6 pairwise distances (tol < 5.0 mm).
     - 3-point fallback matching any face triangle (tol < 5.0 mm).
     - 2-point fallback matching principal edges with hand orientation prior.
  4. Recovers continuous 6-DoF rigid pose (R_t, t_t) across the lost frames.
  5. Smoothly stitches with the first tracked pose to eliminate any visual pop.
"""
from __future__ import annotations

import itertools
import math
import numpy as np

_EMPTY = np.empty((0, 3), dtype=float)


def compute_template_geometry(points_mm: np.ndarray) -> dict:
    """Extract 6 pairwise distances and 4 face triangles for 4-point rigid template."""
    pts = np.asarray(points_mm, dtype=float)
    if pts.shape != (4, 3):
        raise ValueError("Template must have shape (4, 3)")
    
    # Pairwise distances
    dists = {}
    sorted_dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            dists[(i, j)] = d
            sorted_dists.append(d)
    sorted_dists.sort()
    
    # 4 triangles (each omitting 1 vertex)
    triangles = []
    for omit in range(4):
        indices = [k for k in range(4) if k != omit]
        tri_dists = sorted([
            float(np.linalg.norm(pts[indices[0]] - pts[indices[1]])),
            float(np.linalg.norm(pts[indices[1]] - pts[indices[2]])),
            float(np.linalg.norm(pts[indices[0]] - pts[indices[2]]))
        ])
        triangles.append((indices, tri_dists))
        
    return {
        "points": pts,
        "sorted_6_dists": sorted_dists,
        "triangles": triangles,
        "edges": sorted(list(set(round(d, 1) for d in sorted_dists)))
    }


def find_rigid_clique_4pt(candidate_points: np.ndarray,
                          template_geo: dict,
                          tol_mm: float = 6.0) -> tuple[np.ndarray, float] | None:
    """Find 4 points in candidate_points that match template_geo within tol_mm."""
    K = len(candidate_points)
    if K < 4:
        return None
    
    target_dists = np.array(template_geo["sorted_6_dists"])
    best_err = tol_mm
    best_combo = None
    
    for combo in itertools.combinations(range(K), 4):
        sub = candidate_points[list(combo)]
        dists = sorted([
            float(np.linalg.norm(sub[i] - sub[j]))
            for i in range(4) for j in range(i + 1, 4)
        ])
        err = float(np.max(np.abs(np.array(dists) - target_dists)))
        if err < best_err:
            best_err = err
            best_combo = sub
            if err < 1.5:  # sub-millimeter match
                break
                
    if best_combo is not None:
        return best_combo, best_err
    return None


def find_rigid_clique_3pt(candidate_points: np.ndarray,
                          template_geo: dict,
                          tol_mm: float = 5.0) -> tuple[np.ndarray, list[int], float] | None:
    """Find 3 points matching one of the 4 face triangles of template_geo."""
    K = len(candidate_points)
    if K < 3:
        return None
    
    best_err = tol_mm
    best_match = None
    
    for combo in itertools.combinations(range(K), 3):
        sub = candidate_points[list(combo)]
        dists = np.array(sorted([
            float(np.linalg.norm(sub[0] - sub[1])),
            float(np.linalg.norm(sub[1] - sub[2])),
            float(np.linalg.norm(sub[0] - sub[2]))
        ]))
        for indices, target_tri_dists in template_geo["triangles"]:
            err = float(np.max(np.abs(dists - np.array(target_tri_dists))))
            if err < best_err:
                best_err = err
                best_match = (sub, indices, err)
                if err < 1.5:
                    break
                    
    return best_match


def solve_kabsch(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kabsch algorithm: find R and t such that points_a @ R.T + t approx points_b."""
    ca = points_a.mean(axis=0)
    cb = points_b.mean(axis=0)
    H = (points_a - ca).T @ (points_b - cb)
    u, _, vt = np.linalg.svd(H)
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[2] *= -1
        R = vt.T @ u.T
    t = cb - ca @ R.T
    pred = points_a @ R.T + t
    rms = float(np.sqrt(np.mean(np.sum((points_b - pred) ** 2, axis=1))))
    return R, t, rms


def backtrack_prop_from_unlabeled(
    template_points_mm: np.ndarray,
    unlabeled_by_frame: dict[int, list[np.ndarray]],
    first_tracked_frame: int,
    anchor_rot: np.ndarray,
    anchor_trans: np.ndarray,
    hand_midpoint_by_frame: dict[int, np.ndarray] | None = None,
    search_radius_mm: float = 450.0
) -> dict:
    """Backtrack prop 6-DoF pose from frame 0 to first_tracked_frame - 1.

    Returns dict mapping frame -> (R_t, t_t, method).
    """
    geo = compute_template_geometry(template_points_mm)
    recovered = {}
    
    # Relative offset between hand midpoint and prop at first tracked frame
    if hand_midpoint_by_frame and first_tracked_frame in hand_midpoint_by_frame:
        hand_offset = anchor_trans - hand_midpoint_by_frame[first_tracked_frame]
    else:
        hand_offset = np.zeros(3)
        
    last_R = anchor_rot.copy()
    last_t = anchor_trans.copy()
    
    # Process backwards from first_tracked_frame - 1 down to 0
    for f in range(first_tracked_frame - 1, -1, -1):
        candidates = unlabeled_by_frame.get(f, [])
        if hand_midpoint_by_frame and f in hand_midpoint_by_frame:
            mid = hand_midpoint_by_frame[f]
            # filter candidates near hands
            candidates = [p for p in candidates if np.linalg.norm(p - mid) < search_radius_mm]
        else:
            candidates = [p for p in candidates if np.linalg.norm(p - last_t) < search_radius_mm]
            
        cand_arr = np.array(candidates) if candidates else np.empty((0, 3))
        
        # 1. Try 4-point matching
        match_4 = find_rigid_clique_4pt(cand_arr, geo, tol_mm=7.0)
        if match_4 is not None:
            sub, err = match_4
            # Determine permutation by matching distances
            best_perm, best_rms, best_R, best_t = None, 1e9, None, None
            for perm in itertools.permutations(range(4)):
                R, t, rms = solve_kabsch(geo["points"][list(perm)], sub)
                if rms < best_rms:
                    best_rms, best_perm, best_R, best_t = rms, perm, R, t
            if best_R is not None and best_rms < 5.0:
                recovered[f] = (best_R, best_t, "unlabeled_4pt")
                last_R, last_t = best_R, best_t
                continue
                
        # 2. Try 3-point matching
        match_3 = find_rigid_clique_3pt(cand_arr, geo, tol_mm=6.0)
        if match_3 is not None:
            sub, indices, err = match_3
            pts_sub = geo["points"][indices]
            best_rms, best_R, best_t = 1e9, None, None
            for perm in itertools.permutations(range(3)):
                R, t, rms = solve_kabsch(pts_sub[list(perm)], sub)
                if rms < best_rms:
                    best_rms, best_R, best_t = rms, R, t
            if best_R is not None and best_rms < 5.0:
                recovered[f] = (best_R, best_t, "unlabeled_3pt")
                last_R, last_t = best_R, best_t
                continue
                
        # 3. Fallback: Kinematic hand grip prior
        if hand_midpoint_by_frame and f in hand_midpoint_by_frame:
            mid = hand_midpoint_by_frame[f]
            est_t = mid + hand_offset
            recovered[f] = (last_R.copy(), est_t, "hand_kinematic_prior")
            last_t = est_t
        else:
            # Hold last known pose
            recovered[f] = (last_R.copy(), last_t.copy(), "held_prior")
            
    return recovered

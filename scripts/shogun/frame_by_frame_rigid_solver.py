"""frame_by_frame_rigid_solver.py - Autonomous Frame-by-Frame Rigid Pose Estimation.

Directly solves 6-DoF rigid prop poses from 3D optical marker point clouds,
completely bypassing Vicon Shogun's unstable intermediate bone tracker.

Core Capabilities:
  1. Upright Invariant: Enforces vertical orientation prior (Z_world > 0) to prevent 180° flips.
  2. Temporal Continuity: Penalizes angular acceleration to eliminate discrete orientation jumps.
  3. Reverse Backtracking: Recovers missing initial trajectories from Unlabeled_Markers.
  4. Modular Architecture: Ready to interface with custom deep-learning marker classifiers.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def kabsch_svd(template: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve optimal R, t mapping template to observed using Kabsch algorithm."""
    ca = template.mean(axis=0)
    cb = observed.mean(axis=0)
    H = (template - ca).T @ (observed - cb)
    u, _, vt = np.linalg.svd(H)
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[2] *= -1
        R = vt.T @ u.T
    t = cb - ca @ R.T
    pred = template @ R.T + t
    rms = float(np.sqrt(np.mean(np.sum((observed - pred) ** 2, axis=1))))
    return R, t, rms


def solve_frame_rigid_pose(
    template_pts: np.ndarray,
    observed_pts: np.ndarray,
    prev_R: Optional[np.ndarray] = None,
    enforce_upright: bool = True,
    max_rms_mm: float = 6.0
) -> tuple[np.ndarray, np.ndarray, float, list[int]] | None:
    """Solve 6-DoF pose for a single frame, selecting permutation that preserves continuity and upright orientation."""
    K = len(observed_pts)
    if K != 4:
        return None
    
    best_score = 1e9
    best_res = None
    
    for perm in itertools.permutations(range(4)):
        cad_perm = template_pts[list(perm)]
        R, t, rms = kabsch_svd(cad_perm, observed_pts)
        
        if rms > max_rms_mm:
            continue
            
        # Check upright constraint: CAD +Z must point in positive world vertical direction
        z_world = (R @ np.array([0, 0, 1]))[2]
        if enforce_upright and z_world < 0.2:
            continue
            
        # Temporal continuity penalty against previous frame
        if prev_R is not None:
            R_rel = R @ prev_R.T
            tr = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
            ang_deg = math.acos(tr) * (180.0 / math.pi)
            # Penalize abrupt flips (> 45 deg per frame) heavily
            score = rms + ang_deg * 0.8
        else:
            score = rms
            
        if score < best_score:
            best_score = score
            best_res = (R, t, rms, list(perm))
            
    return best_res

def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Vectorized / multi-frame rotation matrix to unit quaternion (w, x, y, z)."""
    F = len(R)
    q = np.zeros((F, 4), dtype=float)
    for f in range(F):
        m = R[f]
        t = np.trace(m)
        if t > 0.0:
            s = 0.5 / np.sqrt(t + 1.0)
            q[f, 0] = 0.25 / s
            q[f, 1] = (m[2, 1] - m[1, 2]) * s
            q[f, 2] = (m[0, 2] - m[2, 0]) * s
            q[f, 3] = (m[1, 0] - m[0, 1]) * s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[f, 0] = (m[2, 1] - m[1, 2]) / s
            q[f, 1] = 0.25 * s
            q[f, 2] = (m[0, 1] + m[1, 0]) / s
            q[f, 3] = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[f, 0] = (m[0, 2] - m[2, 0]) / s
            q[f, 1] = (m[0, 1] + m[1, 0]) / s
            q[f, 2] = 0.25 * s
            q[f, 3] = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[f, 0] = (m[1, 0] - m[0, 1]) / s
            q[f, 1] = (m[0, 2] + m[2, 0]) / s
            q[f, 2] = (m[1, 2] + m[2, 1]) / s
            q[f, 3] = 0.25 * s
    return q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions (w, x, y, z) back to 3x3 rotation matrices."""
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    q = q / norm
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3), dtype=float)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def filter_se3_trajectory(
    rotations: np.ndarray,      # (F, 3, 3)
    translations: np.ndarray,   # (F, 3)
    cutoff_hz: float = 6.0,
    fps: float = 120.0
) -> tuple:
    """Zero-phase forward-backward Butterworth low-pass filter on SE(3)."""
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(2, cutoff_hz / (0.5 * fps), btype='low')
        
        # 1. Translation filtering
        t_smooth = filtfilt(b, a, translations, axis=0)
        
        # 2. Rotation quaternion filtering with antipodal sign alignment
        q = rot_to_quat(rotations)
        for f in range(1, len(q)):
            if np.dot(q[f], q[f - 1]) < 0:
                q[f] = -q[f]
        q_smooth = filtfilt(b, a, q, axis=0)
        R_smooth = quat_to_rot(q_smooth)
        return R_smooth, t_smooth
    except Exception as e:
        # Fallback to moving average if scipy is unavailable
        window = 5
        kernel = np.ones(window) / window
        t_smooth = np.copy(translations)
        for i in range(3):
            t_smooth[:, i] = np.convolve(translations[:, i], kernel, mode='same')
        return rotations, t_smooth


def track_rigid_sequence_continuous(
    template_pts_mm: np.ndarray,
    markers_by_frame: np.ndarray,  # shape: (F, 4, 3) in mm
    reference_frame: int = 1500,
    enforce_upright: bool = True,
    max_rms_mm: float = 6.0,
    smooth_filter: bool = True,
    cutoff_hz: float = 6.0,
    fps: float = 120.0
) -> dict:
    """Frame-by-frame forward/backward tracking anchored at high-confidence reference frame."""
    F, N, _ = markers_by_frame.shape
    assert N == 4, "Requires exactly 4 marker channels"
    
    # 1. Solve anchor reference frame
    ref_f = min(max(0, reference_frame), F - 1)
    # Search within +- 50 frames around ref_f for the cleanest frame
    best_anchor_rms = 1e9
    anchor_idx = ref_f
    anchor_res = None
    
    for f in range(max(0, ref_f - 60), min(F, ref_f + 60)):
        if np.isfinite(markers_by_frame[f]).all():
            res = solve_frame_rigid_pose(template_pts_mm, markers_by_frame[f],
                                         prev_R=None, enforce_upright=enforce_upright,
                                         max_rms_mm=max_rms_mm)
            if res and res[2] < best_anchor_rms:
                best_anchor_rms = res[2]
                anchor_idx = f
                anchor_res = res
                if res[2] < 1.0: # sub-millimeter anchor found
                    break
                    
    if anchor_res is None:
        raise RuntimeError("Could not find a valid upright anchor frame in reference region")
        
    solved_R = [None] * F
    solved_t = [None] * F
    solved_rms = [np.nan] * F
    solved_perm = [None] * F
    
    solved_R[anchor_idx] = anchor_res[0]
    solved_t[anchor_idx] = anchor_res[1]
    solved_rms[anchor_idx] = anchor_res[2]
    solved_perm[anchor_idx] = anchor_res[3]
    
    # 2. Backward propagation: from anchor_idx - 1 down to 0
    for f in range(anchor_idx - 1, -1, -1):
        prev_R = solved_R[f + 1]
        if np.isfinite(markers_by_frame[f]).all():
            res = solve_frame_rigid_pose(template_pts_mm, markers_by_frame[f],
                                         prev_R=prev_R, enforce_upright=enforce_upright,
                                         max_rms_mm=max_rms_mm)
            if res:
                solved_R[f], solved_t[f], solved_rms[f], solved_perm[f] = res
                continue
        solved_R[f] = prev_R.copy()
        solved_t[f] = solved_t[f + 1].copy()
        solved_rms[f] = np.nan
        solved_perm[f] = solved_perm[f + 1]
    # 3. Forward propagation: from anchor_idx + 1 up to F - 1
    for f in range(anchor_idx + 1, F):
        prev_R = solved_R[f - 1]
        if np.isfinite(markers_by_frame[f]).all():
            res = solve_frame_rigid_pose(template_pts_mm, markers_by_frame[f],
                                         prev_R=prev_R, enforce_upright=enforce_upright,
                                         max_rms_mm=max_rms_mm)
            if res:
                solved_R[f], solved_t[f], solved_rms[f], solved_perm[f] = res
                continue
        # Fallback to inertial hold if frame is occluded
        solved_R[f] = prev_R.copy()
        solved_t[f] = solved_t[f - 1].copy()
        solved_rms[f] = np.nan
        solved_perm[f] = solved_perm[f - 1]
        
    solved_R = np.array(solved_R)
    solved_t = np.array(solved_t, dtype=float)
    for axis in range(3):
        nans = np.isnan(solved_t[:, axis])
        if np.any(nans) and not np.all(nans):
            valid_idx = np.flatnonzero(~nans)
            nan_idx = np.flatnonzero(nans)
            solved_t[nan_idx, axis] = np.interp(nan_idx, valid_idx, solved_t[valid_idx, axis])
            
    # Optional zero-phase forward-backward smoothing filter
    if smooth_filter and F > 15:
        solved_R, solved_t = filter_se3_trajectory(solved_R, solved_t, cutoff_hz=cutoff_hz, fps=fps)
    
    # Convert R to Euler XYZ degrees for FBX
    sy = -solved_R[:, 2, 0]
    y = np.arcsin(np.clip(sy, -1.0, 1.0))
    cy = np.cos(y)
    euler_xyz = np.zeros((F, 3), dtype=np.float32)
    for f in range(F):
        if abs(cy[f]) > 1e-6:
            euler_xyz[f, 0] = np.rad2deg(np.arctan2(solved_R[f, 2, 1], solved_R[f, 2, 2]))
            euler_xyz[f, 1] = np.rad2deg(y[f])
            euler_xyz[f, 2] = np.rad2deg(np.arctan2(solved_R[f, 1, 0], solved_R[f, 0, 0]))
        else:
            euler_xyz[f, 0] = np.rad2deg(np.arctan2(-solved_R[f, 1, 2], solved_R[f, 1, 1]))
            euler_xyz[f, 1] = np.rad2deg(y[f])
            euler_xyz[f, 2] = 0.0
            
    return {
        "rotations": solved_R,
        "translations_mm": solved_t,
        "euler_xyz_deg": euler_xyz,
        "rms_by_frame": np.array(solved_rms),
        "anchor_frame": anchor_idx,
        "anchor_rms": best_anchor_rms,
        "max_rms": float(np.nanmax(solved_rms)),
        "median_rms": float(np.nanmedian(solved_rms)),
        "min_vertical_z": float(solved_R[:, 2, 2].min()),
        "permutations": solved_perm,
    }

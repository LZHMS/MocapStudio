"""pipeline_denoise.py - suppress non-physical high-frequency jitter on solved
human bones AND their labeled markers during intervals where the limb stays
in place.

Discriminator per window (the physically meaningful one):
  * NET drift small          -> the limb returns to where it was (no transit)
  * 2nd-difference (jerk) RMS large -> violent oscillation exists
Real fast motion fails "net drift"; smooth deliberate motion fails "jerk";
genuinely still segments fail "jerk" and are left untouched. Inside qualifying
windows a zero-phase Butterworth low-pass removes the oscillation.

SDK notes (Post 1.21): bone animation getters/setters use Euler angles in
DEGREES, translation in cm; marker trajectories use Get/SetTrajectory; boost
accepts only plain Python lists.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.signal import butter, filtfilt

WINDOW = 31                  # frames evaluated together
MIN_STATIC_FRAMES = 21       # qualifying frames required before smoothing
PAD = 12                     # context frames around a run for filter edges
BUTTER_ORDER = 4
CUTOFF = 0.05                # normalized cutoff (fraction of Nyquist)
SMOOTH_PASSES = 2            # repeated zero-phase passes for steeper rolloff

NET_TRANS_CM = 2.0           # |end-start| translation over the window (cm)
NET_ROT_DEG = 25.0           # |end-start| per-channel Euler delta (deg)
JERK_ROT_DEG = 0.6           # 2nd-difference RMS rotation (deg/frame^2)
JERK_TRANS_CM = 0.02         # 2nd-difference RMS translation (cm/frame^2)

MARKER_NET_TRANS_CM = 3.0    # marker windows: net translation ceiling (cm)
MARKER_JERK_TRANS_CM = 0.03  # marker windows: 2nd-difference RMS floor (cm)


def _jerk_rms(values):
    second = np.diff(values, n=2, axis=0)
    return float(np.sqrt((second ** 2).mean(axis=0).max())) if second.size else 0.0


def _runs_from_mask(mask):
    runs = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= MIN_STATIC_FRAMES:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(mask) - start >= MIN_STATIC_FRAMES:
        runs.append((start, len(mask) - 1))
    return runs


def _qualifying_mask_bone(trans, rot, exists, stats=None):
    """A bone window qualifies when the bone as a whole does not travel
    (small net drift in BOTH translation and rotation) while at least one
    channel shows violent oscillation (high jerk)."""
    n = len(exists)
    score = np.zeros(n, dtype=bool)
    if n < WINDOW:
        return score
    for start in range(n - WINDOW + 1):
        window = slice(start, start + WINDOW)
        if not exists[window].all():
            continue
        first, last = start, start + WINDOW - 1
        net_t = float(np.abs(trans[last] - trans[first]).max())
        net_r = float(np.abs(rot[last] - rot[first]).max())
        jerk = max(_jerk_rms(trans[window]), _jerk_rms(rot[window]))
        if stats is not None:
            stats.append((net_t, net_r, jerk))
        if net_t > NET_TRANS_CM or net_r > NET_ROT_DEG:
            continue
        if jerk < min(JERK_TRANS_CM, JERK_ROT_DEG):
            continue
        score[window] = True
    return score


def _qualifying_mask(matrix, exists, net_limit, jerk_floor):
    n = len(exists)
    score = np.zeros(n, dtype=bool)
    if n < WINDOW:
        return score
    for start in range(n - WINDOW + 1):
        window = slice(start, start + WINDOW)
        if not exists[window].all():
            continue
        first, last = start, start + WINDOW - 1
        net = np.abs(matrix[last] - matrix[first]).max()
        if net > net_limit:
            continue
        if _jerk_rms(matrix[window]) < jerk_floor:
            continue
        score[window] = True
    return score


def _smooth_channel(values, low, high):
    span = values[low:high + 1]
    if len(span) <= 3 * BUTTER_ORDER:
        return None
    try:
        b, a = butter(BUTTER_ORDER, CUTOFF, btype="low")
        for _ in range(SMOOTH_PASSES):
            span = filtfilt(b, a, span, axis=0)
        return span
    except ValueError:
        return None


def _smooth_matrix(matrix, exists, mask):
    runs = _runs_from_mask(mask)
    if not runs:
        return matrix, 0
    new = matrix.copy()
    smoothed_frames = 0
    for low, high in runs:
        edge_low = max(0, low - PAD)
        edge_high = min(len(exists) - 1, high + PAD)
        if not exists[edge_low:edge_high + 1].all():
            continue
        smoothed_frames += high - low + 1
        for channel in range(matrix.shape[1]):
            result = _smooth_channel(matrix[:, channel], edge_low, edge_high)
            if result is not None:
                new[low:high + 1, channel] = result[low - edge_low:high - edge_low + 1]
    return new, smoothed_frames


def _bone_series(native_call, api, getter, subject, bone, skeleton, ranges):
    raw = native_call(getter, getattr(api, getter), subject, bone, skeleton, "Local")
    if len(raw) != 4:
        raise RuntimeError(f"{getter} returned {len(raw)} arrays, expected 4")
    arrays = [np.asarray(channel, dtype=float)[ranges["slice_start"]:ranges["slice_stop"]] for channel in raw]
    return np.column_stack(arrays[:-1]), arrays[-1].astype(bool)


def denoise_bone(native_call, api, subject, bone, skeleton, ranges, with_diag=False):
    trans, exists_t = _bone_series(native_call, api, "GetBoneTranslationAnimation", subject, bone, skeleton, ranges)
    rot, exists_r = _bone_series(native_call, api, "GetBoneRotationAnimation", subject, bone, skeleton, ranges)
    # Solve writes rotation channels on limb/finger bones but leaves local
    # translation static (exists all False). Judge windows on the animated
    # channels; touch translation only where it is actually animated.
    animated = exists_t | exists_r
    stats = [] if with_diag else None
    mask = _qualifying_mask_bone(trans, rot, animated, stats)
    new_rot, frames_r = _smooth_matrix(rot, exists_r, mask)
    diag = None
    if stats:
        arr = np.asarray(stats)
        diag = {"net_t_med": round(float(np.median(arr[:, 0])), 3) if len(arr) else None,
                "net_r_med": round(float(np.median(arr[:, 1])), 2) if len(arr) else None,
                "jerk_med": round(float(np.median(arr[:, 2])), 3) if len(arr) else None,
                "windows": len(stats), "rot_frames": int(exists_r.sum()), "trans_frames": int(exists_t.sum())}
    smoothed = frames_r
    new_trans = trans
    if exists_t.any():
        new_trans, frames_t = _smooth_matrix(trans, exists_t, mask)
        smoothed = max(smoothed, frames_t)
    if not smoothed:
        return {"smoothed_frames": 0, "diag": diag} if with_diag else {"smoothed_frames": 0}
    if exists_r.any():
        setter = "SetBoneRotationAnimation"
        if not hasattr(api, setter):
            return {"applied": False, "reason": "setter_unavailable", "setter": setter}
        # boost bindings require plain Python lists, never numpy arrays.
        native_call(setter, getattr(api, setter), subject, bone, skeleton, "Local",
                    *[new_rot[:, index].tolist() for index in range(new_rot.shape[1])],
                    [bool(value) for value in exists_r.tolist()])
    if exists_t.any():
        setter = "SetBoneTranslationAnimation"
        if not hasattr(api, setter):
            return {"applied": False, "reason": "setter_unavailable", "setter": setter}
        native_call(setter, getattr(api, setter), subject, bone, skeleton, "Local",
                    *[new_trans[:, index].tolist() for index in range(new_trans.shape[1])],
                    [bool(value) for value in exists_t.tolist()])
    return {"smoothed_frames": smoothed, "diag": diag} if with_diag else {"smoothed_frames": smoothed}


def denoise_marker(native_call, api, subject, marker, ranges):
    raw = native_call("GetTrajectory", api.GetTrajectory, subject, marker)
    if len(raw) != 4:
        raise RuntimeError(f"GetTrajectory returned {len(raw)} arrays, expected 4")
    arrays = [np.asarray(channel, dtype=float)[ranges["slice_start"]:ranges["slice_stop"]] for channel in raw]
    matrix = np.column_stack(arrays[:-1])
    exists = arrays[-1].astype(bool)
    mask = _qualifying_mask(matrix, exists, MARKER_NET_TRANS_CM, MARKER_JERK_TRANS_CM)
    new_matrix, smoothed = _smooth_matrix(matrix, exists, mask)
    if not smoothed:
        return {"smoothed_frames": 0}
    if not hasattr(api, "SetTrajectory"):
        return {"applied": False, "reason": "setter_unavailable", "setter": "SetTrajectory"}
    native_call("SetTrajectory", api.SetTrajectory, subject, marker,
                *[new_matrix[:, index].tolist() for index in range(new_matrix.shape[1])],
                [bool(value) for value in exists.tolist()])
    return {"smoothed_frames": smoothed}


def denoise_static_humans(native_call, api, detected, ranges, skeleton="Solving"):
    summary = {}
    for name, (character, markers, labeling, solving, row) in detected.items():
        if not row.get("processing_target") or row.get("tolerated_no_data") or not solving:
            continue
        per_bone = {}
        for bone in solving:
            # SDK scene nodes expose .Name/.Path (capitalized).
            bone_name = getattr(bone, "Name", None) or getattr(bone, "name", None) or str(bone)
            try:
                per_bone[bone_name] = denoise_bone(native_call, api, name, bone_name, skeleton, ranges, with_diag=True)
            except Exception as exc:  # one bone must not sink the run; review sees raw
                per_bone[bone_name] = {"applied": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}
        per_marker = {}
        for marker in markers:
            marker_name = getattr(marker, "Name", None) or getattr(marker, "name", None) or str(marker)
            try:
                per_marker[marker_name] = denoise_marker(native_call, api, name, marker_name, ranges)
            except Exception as exc:
                per_marker[marker_name] = {"applied": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}
        top = sorted(per_bone.items(), key=lambda item: item[1].get("smoothed_frames", 0), reverse=True)
        summary[name] = {"bones": len(per_bone),
                         "bone_smoothed_frames": sum(item.get("smoothed_frames", 0) for item in per_bone.values()),
                         "markers": len(per_marker),
                         "marker_smoothed_frames": sum(item.get("smoothed_frames", 0) for item in per_marker.values()),
                         "top_bones": {bone: item.get("smoothed_frames", 0) for bone, item in top[:10]},
                         "diag": {bone: item["diag"] for bone, item in per_bone.items() if item.get("diag")},
                         "failed": {**{bone: item.get("reason", "?") for bone, item in per_bone.items()
                                       if item.get("applied") is False},
                                    **{marker: item.get("reason", "?") for marker, item in per_marker.items()
                                       if item.get("applied") is False}}}
    return summary

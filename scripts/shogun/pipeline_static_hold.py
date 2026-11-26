"""Static table hold and stationary jitter clamping for rigid props.

Detects time intervals where a tracked prop is resting stationary (e.g. on a table
or in a fixed fixture) and clamps marker trajectories to the robust median 6-DoF pose.
Boundary transitions are blended with Hermite smoothstep to guarantee C1 continuity.
"""
from dataclasses import dataclass
import math
import numpy as np

from pipeline_rigid import RigidOptions, match_frame
from pipeline_jitter import _quaternion, _rotation

_EMPTY = np.empty((0, 3), dtype=float)


@dataclass(frozen=True)
class StaticHoldOptions:
    enabled: bool = True
    window_frames: int = 15
    max_static_disp_mm: float = 2.5
    max_static_rot_deg: float = 2.5
    min_hold_frames: int = 12
    blend_frames: int = 5
    fps: float = 120.0


def _quaternion_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic rotation angle in degrees between two unit quaternions."""
    dot = abs(float(np.clip(np.dot(q1, q2), -1.0, 1.0)))
    return 2.0 * math.acos(dot) * (180.0 / math.pi)


def clamp_static_prop_jitter(template: np.ndarray, points: np.ndarray,
                             rigid_options: RigidOptions,
                             options: StaticHoldOptions = StaticHoldOptions()) -> dict:
    """Stabilize stationary prop markers by clamping to median pose during static intervals.

    Returns dict containing:
      - 'points': array of shape (F, N, 3) with static segments clamped
      - 'static_intervals': list of [start, end] inclusive frame index ranges
      - 'clamped_frames': total number of frames clamped
      - 'max_correction_mm': maximum displacement applied to any marker
      - 'provenance': 'static_table_clamped'
    """
    template = np.asarray(template, dtype=float)
    observed = np.asarray(points, dtype=float)
    F, N, _ = observed.shape

    if not options.enabled or F < options.min_hold_frames:
        return {
            'points': observed.copy(),
            'static_intervals': [],
            'clamped_frames': 0,
            'max_correction_mm': 0.0,
            'provenance': 'unmodified'
        }

    # 1. Evaluate rigid pose at every frame
    poses = [match_frame(template, frame, _EMPTY, rigid_options) for frame in observed]
    tracked = np.array([p['state'] == 'TRACKED' for p in poses])

    translations = np.full((F, 3), np.nan)
    quaternions = np.full((F, 4), np.nan)
    for f in range(F):
        if tracked[f]:
            translations[f] = poses[f]['translation']
            quaternions[f] = _quaternion(poses[f]['rotation'])

    # 2. Identify candidate static frames using windowed dispersion from median
    W = max(3, int(options.window_frames))
    is_static = np.zeros(F, dtype=bool)

    for f in range(F):
        if not tracked[f]:
            continue
        f0 = max(0, f - W // 2)
        f1 = min(F, f + W // 2 + 1)
        sub_t = translations[f0:f1]
        valid_t = sub_t[np.isfinite(sub_t).all(axis=1)]
        if len(valid_t) < max(3, W // 3):
            continue

        med_t = np.median(valid_t, axis=0)
        disp_mm = np.max(np.linalg.norm(valid_t - med_t, axis=1))

        # Check rotational dispersion
        sub_q = quaternions[f0:f1]
        valid_q = sub_q[np.isfinite(sub_q).all(axis=1)].copy()
        for i in range(1, len(valid_q)):
            if np.dot(valid_q[0], valid_q[i]) < 0:
                valid_q[i] = -valid_q[i]
        med_q = np.median(valid_q, axis=0)
        norm_q = np.linalg.norm(med_q)
        if norm_q > 0:
            med_q /= norm_q
            rot_deg = max(_quaternion_angle_deg(q, med_q) for q in valid_q)
        else:
            rot_deg = 0.0

        if disp_mm <= options.max_static_disp_mm and rot_deg <= options.max_static_rot_deg:
            is_static[f] = True

    # 3. Form contiguous intervals satisfying min_hold_frames
    runs = []
    in_run = False
    run_start = 0
    for f in range(F):
        if is_static[f] and not in_run:
            in_run = True
            run_start = f
        elif not is_static[f] and in_run:
            in_run = False
            if f - run_start >= options.min_hold_frames:
                runs.append((run_start, f - 1))
    if in_run and F - run_start >= options.min_hold_frames:
        runs.append((run_start, F - 1))

    # 4. Clamp marker positions during static runs with smooth boundary transitions
    output = observed.copy()
    clamped_frames_mask = np.zeros(F, dtype=bool)
    max_disp_applied = 0.0
    static_intervals = []

    for s, e in runs:
        # Require that most frames in the interval were tracked
        run_tracked = tracked[s:e+1]
        if run_tracked.sum() < (e - s + 1) * 0.7:
            continue

        static_intervals.append([int(s), int(e)])

        # Robust median of real physical markers over the stationary segment
        seg_obs = observed[s:e+1][run_tracked]
        clamped_marker_pos = np.median(seg_obs, axis=0)

        # Boundary blend width
        B = min(options.blend_frames, max(1, (e - s) // 4))

        for f in range(s, e + 1):
            if not tracked[f]:
                continue
            weight = 1.0
            if f < s + B:
                xi = (f - s + 1) / (B + 1)
                weight = 3.0 * (xi ** 2) - 2.0 * (xi ** 3)
            elif f > e - B:
                xi = (e - f + 1) / (B + 1)
                weight = 3.0 * (xi ** 2) - 2.0 * (xi ** 3)

            target_pos = (1.0 - weight) * output[f] + weight * clamped_marker_pos
            diff = np.linalg.norm(target_pos - output[f], axis=1)
            max_disp_applied = max(max_disp_applied, float(np.max(diff)))
            output[f] = target_pos
            clamped_frames_mask[f] = True

    return {
        'points': output,
        'static_intervals': static_intervals,
        'clamped_frames': int(clamped_frames_mask.sum()),
        'max_correction_mm': float(max_disp_applied),
        'provenance': 'static_table_clamped' if static_intervals else 'unmodified'
    }

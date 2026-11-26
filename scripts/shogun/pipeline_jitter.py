"""Bounded, model-derived repair of interior rigid-prop geometric jitter.

Missing observations are never filled; corrected coordinates are model-derived.
Original TRACKED poses supply every anchor, so accepted repairs cannot cascade.
The pose convention matches pipeline_rigid: world = template @ rotation.T + t.
"""
from dataclasses import dataclass
import math

import numpy as np

from pipeline_rigid import RigidOptions, match_frame


@dataclass(frozen=True)
class JitterOptions:
    max_correction_mm: float = 5.0
    max_gap_frames: int = 5
    window_frames: int = 3

    def __post_init__(self):
        try:
            valid_distance = (math.isfinite(self.max_correction_mm)
                              and self.max_correction_mm >= 0)
        except (TypeError, ValueError, OverflowError):
            valid_distance = False
        if not valid_distance:
            raise ValueError('max_correction_mm must be finite and nonnegative')
        for name in ('max_gap_frames', 'window_frames'):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f'{name} must be a positive integer')


def _quaternion(rotation):
    """Unit (w, x, y, z), including rotations with trace close to -1."""
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2 * math.sqrt(max(0., 1 + trace))
        quaternion = np.array([scale / 4,
                               (rotation[2, 1] - rotation[1, 2]) / scale,
                               (rotation[0, 2] - rotation[2, 0]) / scale,
                               (rotation[1, 0] - rotation[0, 1]) / scale])
    else:
        axis = int(np.argmax(np.diag(rotation)))
        other = (axis + 1) % 3
        last = (axis + 2) % 3
        scale = 2 * math.sqrt(max(0., 1 + rotation[axis, axis]
                                  - rotation[other, other] - rotation[last, last]))
        quaternion = np.empty(4)
        quaternion[0] = (rotation[last, other] - rotation[other, last]) / scale
        quaternion[axis + 1] = scale / 4
        quaternion[other + 1] = (rotation[other, axis] + rotation[axis, other]) / scale
        quaternion[last + 1] = (rotation[last, axis] + rotation[axis, last]) / scale
    return quaternion / np.linalg.norm(quaternion)


def _rotation(quaternion):
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                      2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                      2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w),
                      1 - 2 * (x * x + y * y)]])


def _motion_model(left, right):
    first = _quaternion(left['rotation'])
    second = _quaternion(right['rotation'])
    dot = float(np.clip(first @ second, -1., 1.))
    # At a half-turn, the two opposite angular velocities are equally valid.
    # A deterministic arbitrary choice is not evidence of the actual motion.
    if abs(dot) <= 1e-10:
        return None
    if dot < 0:
        second = -second
        dot = -dot
    tangent = second - dot * first
    length = float(np.linalg.norm(tangent))
    angle = math.atan2(length, dot)
    if length > np.finfo(float).eps:
        tangent /= length
    else:
        tangent[:] = 0
        angle = 0.
    return (first, tangent, angle, left['translation'],
            right['translation'] - left['translation'])


def _model_points(template, model, fraction):
    first, tangent, angle, translation, delta = model
    quaternion = first * math.cos(fraction * angle) + tangent * math.sin(fraction * angle)
    return template @ _rotation(quaternion).T + translation + fraction * delta


def repair_jitter(template: np.ndarray, points: np.ndarray,
                  rigid_options: RigidOptions, options: JitterOptions) -> dict:
    """Return a copied sequence, sparse corrections, states, and rejected runs.

    The two closest original TRACKED poses define constant translation/angular
    velocity. Up to window_frames contiguous original TRACKED samples on each
    side must agree with that model in marker space within max_correction_mm.
    Evaluating the model outside the gap is only a consistency test: no point
    outside the strictly interior gap is ever corrected or extrapolated.

    Missing/nonfinite observations, mixed non-TRACKED states, long/edge gaps,
    ambiguous half-turns, conflicting windows, excess correction, and excess
    rigid max_step_mm motion reject the entire run. Geometry is checked again
    with the original RigidOptions; tolerance is never enlarged.
    """
    if not isinstance(options, JitterOptions):
        raise ValueError('options must be JitterOptions')
    try:
        template = np.asarray(template, dtype=float)
        source = np.asarray(points)
        observed = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('template and points must be numeric arrays') from exc
    if template.ndim != 2 or template.shape[1:] != (3,):
        raise ValueError('template must have shape (N, 3)')
    if observed.ndim != 3 or observed.shape[1:] != template.shape:
        raise ValueError('points must have shape (F, N, 3) matching template')

    empty = np.empty((0, 3))
    poses = [match_frame(template, frame, empty, rigid_options) for frame in observed]
    if not poses:
        # Keep option validation identical for empty and nonempty sequences.
        match_frame(template, np.full_like(template, np.nan), empty, rigid_options)
    states = [pose['state'] for pose in poses]
    output = (source.copy() if np.issubdtype(source.dtype, np.floating)
              else observed.copy())
    result = dict(points=output, corrected_frames=np.zeros(len(observed), dtype=bool),
                  corrections=[], states_before=states, states_after=states.copy(),
                  rejections=[], method='temporal_rigid_jitter',
                  provenance='model_derived_not_measured')
    bound = float(options.max_correction_mm)

    def reject(start, end, reason):
        result['rejections'].append(dict(start=int(start), end=int(end), reason=reason))

    index = 0
    while index < len(observed):
        if states[index] == 'TRACKED':
            index += 1
            continue
        start = index
        while index < len(observed) and states[index] != 'TRACKED':
            index += 1
        end = index - 1
        if bound == 0:
            reject(start, end, 'disabled: max_correction_mm is zero')
            continue
        if not np.isfinite(observed[start:index]).all():
            reject(start, end, 'missing_observations: every corrected marker must already be finite')
            continue
        if any(state != 'INVALID_GEOMETRY' for state in states[start:index]):
            reject(start, end, 'unsupported_state: only INVALID_GEOMETRY runs are eligible')
            continue
        if index - start > options.max_gap_frames:
            reject(start, end, f'max_gap_frames_exceeded: {index - start} > {options.max_gap_frames}')
            continue
        if start == 0 or index == len(observed):
            reject(start, end, 'unanchored_edge: original trusted poses are required on both sides')
            continue

        left, right = start - 1, index
        model = _motion_model(poses[left], poses[right])
        if model is None:
            reject(start, end, 'ambiguous_rotation: anchor poses differ by a half-turn')
            continue
        distance = right - left
        window = [left, right]
        for direction, anchor in ((-1, left), (1, right)):
            for offset in range(1, options.window_frames):
                neighbor = anchor + direction * offset
                if not 0 <= neighbor < len(observed) or states[neighbor] != 'TRACKED':
                    break
                window.append(neighbor)
        conflict = None
        for neighbor in window:
            predicted = _model_points(template, model, (neighbor - left) / distance)
            fitted = template @ poses[neighbor]['rotation'].T + poses[neighbor]['translation']
            deviation = float(np.max(np.linalg.norm(predicted - fitted, axis=1)))
            if deviation > bound + 1e-9:
                conflict = (neighbor, deviation)
                break
        if conflict is not None:
            reject(start, end, 'inconsistent_anchor_window: frame '
                   f'{conflict[0]} deviates {conflict[1]:.9g} mm > {bound:.9g} mm')
            continue

        proposed = np.stack([_model_points(template, model, (frame - left) / distance)
                             for frame in range(start, right)])
        # Validate the coordinates that will actually be stored, including any
        # rounding required to preserve the input floating-point representation.
        proposed = proposed.astype(output.dtype).astype(float)
        displacements = np.linalg.norm(proposed - observed[start:right], axis=2)
        maximum = float(np.max(displacements))
        if maximum > bound + 1e-9:
            reject(start, end, f'correction_limit_exceeded: {maximum:.9g} mm > {bound:.9g} mm')
            continue
        # Do not turn floating-point pose-fit roundoff into spurious point edits.
        unchanged = displacements <= 1e-9
        proposed[unchanged] = observed[start:right][unchanged]
        previous = observed[left]
        failed = None
        for offset, frame_points in enumerate(proposed):
            verification = match_frame(template, frame_points, empty, rigid_options,
                                       prior_points=previous)
            if verification['state'] != 'TRACKED':
                failed = (start + offset, verification)
                break
            previous = frame_points
        if failed is None:
            verification = match_frame(template, observed[right], empty, rigid_options,
                                       prior_points=previous)
            if verification['state'] != 'TRACKED':
                failed = (right, verification)
        if failed is not None:
            reject(start, end, f'corrected_geometry_or_motion_rejected: frame {failed[0]} '
                   f'{failed[1]["state"]}: {failed[1]["reason"]}')
            continue

        for offset, replacement in enumerate(proposed):
            frame = start + offset
            for marker in np.flatnonzero(~unchanged[offset]):
                original = observed[frame, marker]
                corrected = replacement[marker]
                result['corrections'].append(dict(
                    frame=int(frame), marker=int(marker), original_xyz=original.tolist(),
                    corrected_xyz=corrected.tolist(),
                    displacement_mm=float(np.linalg.norm(corrected - original))))
            result['points'][frame] = replacement
            result['corrected_frames'][frame] = True
            result['states_after'][frame] = 'TRACKED'
    return result

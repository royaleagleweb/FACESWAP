"""Yaw estimate and landmark repair for side and profile faces.

InSwapper aligns five points onto a frontal template. On a profile the far eye
and the far mouth corner collapse, the similarity blows up, and the paste
misses the cheek. ``repair_landmarks`` puts those hidden points back on the
ear side of the visible features so the warp stays stable. The original yaw
is still used to widen the cheek and jaw mask toward the nose.
"""

from __future__ import annotations

import numpy as np

from .coverage import face_axes

# Above this, the first-lock threshold and the track hold both loosen.
PROFILE_YAW = 0.45
_REPAIR_YAW = 0.34


def face_yaw(kps: np.ndarray) -> float:
    """Signed yaw in about [-1, 1]. Positive means the nose leads ``side_u``.

    ``side_u`` is the same axis ``coverage`` uses (toward the right mouth
    corner). Zero is a frontal five-point set. The value grows when the eyes
    collapse or the nose leaves the eye midpoint, which is what a three-quarter
    or profile detection looks like.
    """
    pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    axes = face_axes(pts)
    if axes is None:
        return 0.0
    eye_c, _mouth_c, _down_u, side_u, em, eye_dist = axes
    if em < 2.0:
        return 0.0
    nose_lat = float(np.dot(pts[2] - eye_c, side_u)) / em
    span = eye_dist / em
    compression = float(np.clip((0.87 - span) / 0.55, 0.0, 1.0))
    mouth = float(np.linalg.norm(pts[4] - pts[3])) / em
    mouth_comp = float(np.clip((0.72 - mouth) / 0.45, 0.0, 1.0))
    nose_mag = float(np.clip(abs(nose_lat) / 0.55, 0.0, 1.0))
    magnitude = max(compression, 0.65 * mouth_comp, nose_mag * 0.85)
    if magnitude < 0.08:
        return 0.0
    sign = 1.0 if nose_lat >= 0.0 else -1.0
    return float(np.clip(sign * magnitude, -1.0, 1.0))


def pose_name(yaw: float) -> str:
    """Short label for the face list. Front faces stay unlabeled in the hint."""
    magnitude = abs(float(yaw))
    if magnitude >= 0.62:
        return "Profile"
    if magnitude >= 0.28:
        return "Three-quarter"
    return "Front"


def repair_landmarks(kps: np.ndarray) -> np.ndarray:
    """Spread a collapsed profile back toward the frontal five-point spacing.

    The visible eye, the nose, and the visible mouth corner stay put. The far
    eye and the far mouth corner move toward the ear so Umeyama can match the
    InSwapper template without a huge scale. Frontal points are returned as-is.
    """
    pts = np.asarray(kps, dtype=np.float64).reshape(5, 2).copy()
    axes = face_axes(pts)
    if axes is None:
        return pts.astype(np.float32)
    _eye_c, _mouth_c, _down_u, side_u, em, _eye_dist = axes
    yaw = face_yaw(pts)
    if abs(yaw) < _REPAIR_YAW:
        return pts.astype(np.float32)
    strength = float(np.clip((abs(yaw) - 0.20) / 0.55, 0.0, 1.0))
    direction = 1.0 if yaw >= 0.0 else -1.0
    target_eye = 0.86 * em
    target_mouth = 0.70 * em
    pts = _restore_pair(pts, (0, 1), side_u, direction, target_eye, strength)
    pts = _restore_pair(pts, (3, 4), side_u, direction, target_mouth, strength)
    return pts.astype(np.float32)


def _restore_pair(
    pts: np.ndarray,
    indexes: tuple[int, int],
    side_u: np.ndarray,
    direction: float,
    target_span: float,
    strength: float,
) -> np.ndarray:
    """Keep the point nearer the nose and push the other one out to ``target_span``."""
    i, j = indexes
    lat_i = float(np.dot(pts[i], side_u))
    lat_j = float(np.dot(pts[j], side_u))
    if lat_i * direction >= lat_j * direction:
        visible, far = i, j
    else:
        visible, far = j, i
    desired = pts[visible] - side_u * (direction * target_span)
    pts[far] = (1.0 - strength) * pts[far] + strength * desired
    return pts

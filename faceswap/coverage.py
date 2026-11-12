"""Paste an InSwapper crop so the jaw and beard are actually replaced.

InsightFace warps the 128×128 swap back into the frame and then erodes that
square. The mouth sits near the bottom of the crop, so the erosion removes
the chin and leaves the target's beard showing around a tight oval.

Full coverage (the default) keeps the aligned crop for the eyes, nose, and
mouth, stretches the swapped chin across a landmark beard footprint, and
feathers only the outer band. The interior of that footprint is fully
replaced.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

COVERAGE_FULL = "full"
COVERAGE_NORMAL = "normal"
COVERAGE_CHOICES = (COVERAGE_FULL, COVERAGE_NORMAL)
DEFAULT_COVERAGE = COVERAGE_FULL

# InsightFace arcface 5-point template, shifted for the 128px InSwapper crop
# (see insightface.utils.face_align.estimate_norm with image_size=128).
_ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

# How far below the mouth the full mask reaches, in eye-to-mouth lengths.
# ~1 length reaches the chin. ~2 lengths covers a goatee and a typical beard.
FULL_BEARD_DEPTH_EM = 2.0
NORMAL_BEARD_DEPTH_EM = 0.42

# Half-width of the jaw as a fraction of the eye-to-eye distance.
FULL_JAW_WIDTH = 1.05
NORMAL_JAW_WIDTH = 0.46


def template_128() -> np.ndarray:
    dst = _ARCFACE_112.copy()
    dst[:, 0] += 8.0
    return dst


def _mouth_metrics() -> tuple[float, float]:
    dst = template_128()
    eye_y = float(dst[0:2, 1].mean())
    mouth_y = float(dst[3:5, 1].mean())
    return mouth_y, mouth_y - eye_y


def normalize_coverage(coverage: Optional[str]) -> str:
    mode = (coverage or DEFAULT_COVERAGE).strip().lower()
    if mode not in COVERAGE_CHOICES:
        raise ValueError(
            f"Unknown face coverage '{coverage}'. Choose one of: {', '.join(COVERAGE_CHOICES)}."
        )
    return mode


def face_axes(kps: np.ndarray) -> Optional[tuple]:
    """Eye center, mouth center, down vector, side vector, and two lengths.

    ``down`` points from the eyes toward the mouth. ``side`` points toward the
    mouth corner on the positive template-x side. Lengths are in pixels.
    """
    pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    if pts.shape != (5, 2):
        return None
    eye_center = pts[0:2].mean(axis=0)
    mouth_center = pts[3:5].mean(axis=0)
    down = mouth_center - eye_center
    eye_to_mouth = float(np.linalg.norm(down))
    eye_dist = float(np.linalg.norm(pts[1] - pts[0]))
    if eye_to_mouth < 2.0 or eye_dist < 2.0:
        return None
    down_u = down / eye_to_mouth
    side_u = np.array([down_u[1], -down_u[0]], dtype=np.float64)
    if float(np.dot(pts[4] - mouth_center, side_u)) < 0.0:
        side_u = -side_u
    return eye_center, mouth_center, down_u, side_u, eye_to_mouth, eye_dist


def _pt(origin: np.ndarray, down_u: np.ndarray, side_u: np.ndarray, along: float, lateral: float) -> np.ndarray:
    return origin + down_u * along + side_u * lateral


def coverage_polygon(
    kps: np.ndarray,
    coverage: str = DEFAULT_COVERAGE,
    yaw: float = 0.0,
) -> Optional[np.ndarray]:
    """Landmark polygon covering forehead, cheeks, jaw, and beard or a tight oval.

    ``yaw`` is the signed profile amount from ``face_yaw``. A side view extends
    the cheek and jaw, and shifts the polygon toward the nose so the leading
    cheek is inside the swap.
    """
    axes = face_axes(kps)
    if axes is None:
        return None
    eye_c, mouth_c, down_u, side_u, em, eye_dist = axes
    mode = normalize_coverage(coverage)
    profile = float(np.clip(abs(float(yaw)) / 0.75, 0.0, 1.0))
    if mode == COVERAGE_FULL:
        depth = FULL_BEARD_DEPTH_EM * em * (1.0 + 0.35 * profile)
        jaw = FULL_JAW_WIDTH * eye_dist * (1.0 + 0.65 * profile)
        cheek = 0.90 * eye_dist * (1.0 + 0.45 * profile)
        temple = 0.72 * eye_dist * (1.0 + 0.25 * profile)
        forehead = 1.05 * em
        beard_w = 0.72 * eye_dist * (1.0 + 0.50 * profile)
    else:
        depth = NORMAL_BEARD_DEPTH_EM * em
        jaw = NORMAL_JAW_WIDTH * eye_dist
        cheek = 0.42 * eye_dist
        temple = 0.40 * eye_dist
        forehead = 0.55 * em
        beard_w = 0.30 * eye_dist

    pts = np.stack(
        [
            _pt(eye_c, down_u, side_u, -forehead, 0.0),
            _pt(eye_c, down_u, side_u, -forehead * 0.75, -temple),
            _pt(eye_c, down_u, side_u, 0.10 * em, -temple),
            _pt(eye_c, down_u, side_u, 0.55 * em, -cheek),
            _pt(mouth_c, down_u, side_u, 0.20 * em, -jaw),
            _pt(mouth_c, down_u, side_u, 0.55 * depth, -beard_w),
            _pt(mouth_c, down_u, side_u, 0.96 * depth, -0.42 * beard_w),
            _pt(mouth_c, down_u, side_u, 0.96 * depth, 0.42 * beard_w),
            _pt(mouth_c, down_u, side_u, 0.55 * depth, beard_w),
            _pt(mouth_c, down_u, side_u, 0.20 * em, jaw),
            _pt(eye_c, down_u, side_u, 0.55 * em, cheek),
            _pt(eye_c, down_u, side_u, 0.10 * em, temple),
            _pt(eye_c, down_u, side_u, -forehead * 0.75, temple),
        ]
    )
    if profile > 0.0:
        shift = float(np.clip(yaw, -1.0, 1.0)) * (0.35 * profile) * eye_dist
        pts = pts + side_u * shift
    return pts


def _hull_points(kps: np.ndarray, yaw: float = 0.0) -> Optional[np.ndarray]:
    """Face oval around the landmarks. The chin stops just under the mouth.

    This is not the old beard polygon. That shape ran two eye-to-mouth
    lengths down the chest and pasted a hard triangle there.
    """
    axes = face_axes(kps)
    if axes is None:
        return None
    eye_c, mouth_c, down_u, side_u, em, eye_dist = axes
    profile = float(np.clip(abs(float(yaw)) / 0.75, 0.0, 1.0))
    shift = side_u * (float(np.clip(yaw, -1.0, 1.0)) * 0.2 * profile * eye_dist)
    chin = mouth_c + down_u * (0.36 * em)
    cheeks = [
        mouth_c + side_u * (sign * (0.95 + 0.35 * profile) * eye_dist) + down_u * (0.05 * em)
        for sign in (-1.0, 1.0)
    ]
    temples = [
        eye_c + side_u * (sign * (0.85 + 0.25 * profile) * eye_dist) - down_u * (0.15 * em)
        for sign in (-1.0, 1.0)
    ]
    forehead = eye_c - down_u * (0.85 * em)
    pts = np.vstack([
        np.asarray(kps, dtype=np.float64).reshape(-1, 2),
        chin,
        forehead,
        *cheeks,
        *temples,
    ])
    return pts + shift


def coverage_alpha(
    shape: tuple[int, int],
    kps: np.ndarray,
    coverage: str = DEFAULT_COVERAGE,
    yaw: float = 0.0,
) -> np.ndarray:
    """Feathered face mask. The rim is soft. Nothing hard is drawn below the chin.

    ``full`` and ``normal`` both use the landmark hull. ``normal`` is a tighter
    margin. A long beard polygon used to paste a hard triangle onto the chest.
    """
    height, width = shape
    alpha = np.zeros((height, width), dtype=np.float32)
    mode = normalize_coverage(coverage)
    points = _hull_points(kps, yaw=yaw) if mode == COVERAGE_FULL else None
    if points is None:
        polygon = coverage_polygon(kps, COVERAGE_NORMAL, yaw=yaw)
        axes = face_axes(kps)
        if polygon is None or axes is None:
            return alpha
        points = polygon
    axes = face_axes(kps)
    if axes is None:
        return alpha
    _eye_c, _mouth_c, _down_u, _side_u, _em, eye_dist = axes
    # Soft margin. Full is the face plus a little chin. Normal stays tighter.
    feather = max(10.0, (0.28 if mode == COVERAGE_FULL else 0.16) * eye_dist)
    hull = cv2.convexHull(np.round(points).astype(np.int32))
    xs = hull[:, 0, 0]
    ys = hull[:, 0, 1]
    margin = int(np.ceil(feather)) + 3
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(width, int(xs.max()) + margin + 1)
    y1 = min(height, int(ys.max()) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return alpha
    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = hull.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    cv2.fillConvexPoly(local, shifted, 255)
    grow = max(1, int(round(feather * 0.35)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
    local = cv2.dilate(local, kernel)
    if not np.any(local):
        return alpha
    dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    t = np.clip(dist / feather, 0.0, 1.0)
    soft = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    alpha[y0:y1, x0:x1] = soft
    return alpha


def _sample_maps(cx: np.ndarray, cy: np.ndarray, size: int, coverage: str) -> tuple[np.ndarray, np.ndarray]:
    """Map frame→crop coordinates onto the swapped image.

    Above the crop bottom this is 1:1, so eyes and mouth stay on the
    landmarks. Past the crop, full coverage replays the chin band downward
    across the beard instead of stopping.
    """
    # Stay inside the swapped crop. Replaying the chin band below the crop
    # painted a hard skin wedge onto the chest.
    sx = np.clip(cx, 1.0, size - 2.0)
    sy = np.clip(cy, 1.0, size - 2.0)
    return sx.astype(np.float32), sy.astype(np.float32)


def face_roi(kps: np.ndarray, shape: tuple[int, int], pad_scale: float = 0.7) -> tuple[int, int, int, int]:
    """Tight box around the landmarks. Paste and masks stay inside it."""
    height, width = shape
    pts = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 8.0)
    margin = max(4, int(round(span * pad_scale)))
    x0 = max(0, int(np.floor(float(pts[:, 0].min()))) - margin)
    y0 = max(0, int(np.floor(float(pts[:, 1].min()))) - margin)
    x1 = min(width, int(np.ceil(float(pts[:, 0].max()))) + margin + 1)
    y1 = min(height, int(np.ceil(float(pts[:, 1].max()))) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, width, height
    return x0, y0, x1, y1


def paste_swapped_face(
    frame_bgr: np.ndarray,
    swapped_bgr: np.ndarray,
    matrix: np.ndarray,
    kps: np.ndarray,
    coverage: str = DEFAULT_COVERAGE,
    alpha: Optional[np.ndarray] = None,
    previous_delta: Optional[np.ndarray] = None,
    color_state: Optional[dict] = None,
    yaw: float = 0.0,
    color_match: Optional[float] = None,
) -> np.ndarray:
    """Blend ``swapped_bgr`` (the 128px InSwapper output) onto ``frame_bgr``.

    ``matrix`` is the 2×3 frame→crop transform InSwapper used to build the crop.
    """
    mode = normalize_coverage(coverage)
    if swapped_bgr is None or swapped_bgr.size == 0 or matrix is None:
        return frame_bgr
    size = int(swapped_bgr.shape[0])
    matrix = orient_paste_matrix(matrix, kps, size=size)
    if not _paste_matrix_ok(matrix, kps, size):
        return frame_bgr
    if alpha is None:
        alpha = coverage_alpha(frame_bgr.shape[:2], kps, mode, yaw=yaw)
    x0, y0, x1, y1 = face_roi(kps, frame_bgr.shape[:2])
    weight = np.asarray(alpha[y0:y1, x0:x1], dtype=np.float32)
    if weight.size == 0 or float(weight.max()) <= 0.0:
        return frame_bgr
    # OpenCV warpAffine maps source → dest. ``matrix`` is frame → crop, so invert it.
    affine = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32).reshape(2, 3))
    affine[0, 2] -= x0
    affine[1, 2] -= y0
    warped = cv2.warpAffine(
        swapped_bgr,
        affine,
        (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    out = frame_bgr.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    weight = weight[:, :, None]
    warped_f = warped.astype(np.float32)
    if mode == COVERAGE_FULL:
        warped_f = match_edge_color(
            warped_f,
            roi,
            weight,
            previous=previous_delta,
            state=color_state,
            color_match=color_match,
        )
    blended = warped_f * weight + roi * (1.0 - weight)
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def match_edge_color(
    warped: np.ndarray,
    roi: np.ndarray,
    weight: np.ndarray,
    previous: Optional[np.ndarray] = None,
    state: Optional[dict] = None,
    color_match: Optional[float] = None,
) -> np.ndarray:
    """Move the soft rim of the swap toward the original frame color.

    The interior (alpha near 1) keeps the swapped color. Only the feather
    band shifts, so the blend is not a muddy average of two palettes.
    """
    alpha = weight[..., 0] if weight.ndim == 3 else weight
    rim = (alpha > 0.12) & (alpha < 0.88)
    if int(np.count_nonzero(rim)) < 16:
        return warped
    delta = roi[rim].mean(axis=0) - warped[rim].mean(axis=0)
    if previous is not None:
        from .quality import smooth_delta

        delta = smooth_delta(previous, delta)
    if state is not None:
        state["delta"] = np.asarray(delta, dtype=np.float32).copy()
    # A strong shift paints the original skin back onto the rim and the swap
    # reads as the source clip. Keep only a light edge correction.
    # color_match 0.60 is the historical 0.35 rim. 0 turns the correction off.
    gain = 0.35 if color_match is None else 0.35 * (float(color_match) / 0.60)
    strength = np.clip((1.0 - alpha) * gain, 0.0, 1.0).astype(np.float32)[..., None]
    shifted = warped.astype(np.float32) + delta.astype(np.float32) * strength
    return np.clip(shifted, 0.0, 255.0)


def _paste_matrix_ok(matrix: np.ndarray, kps: np.ndarray, size: int) -> bool:
    """Refuse a collapsed or sheared warp. That paints a triangle off the face."""
    del kps, size
    linear = np.asarray(matrix, dtype=np.float64).reshape(2, 3)[:, :2]
    if not np.isfinite(linear).all():
        return False
    singular = np.linalg.svd(linear, compute_uv=False)
    if not np.isfinite(singular).all() or float(singular.min()) < 1e-3:
        return False
    if float(singular.max()) > float(singular.min()) * 6.0:
        return False
    return True


def orient_paste_matrix(matrix: np.ndarray, kps: np.ndarray, size: int = 128) -> np.ndarray:
    """Return the 2×3 frame→crop matrix that lands landmarks on the template.

    InsightFace ``estimate_norm`` maps the frame landmarks onto the 128px
    template. If a session instead returns the crop→frame matrix, sampling
    it as frame→crop pastes the identity in the wrong place.
    """
    forward = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    template = template_128()
    if size != 128:
        template = template * (float(size) / 128.0)
    if _maps_onto_template(forward, kps, template):
        return forward
    try:
        inverse = cv2.invertAffineTransform(forward)
    except cv2.error:
        inverse = None
    if inverse is not None and _maps_onto_template(inverse, kps, template):
        return np.asarray(inverse, dtype=np.float64)
    if face_axes(kps) is None:
        return forward
    rebuilt = _estimate_similarity(np.asarray(kps, dtype=np.float64).reshape(-1, 2), template)
    if rebuilt is not None and _maps_onto_template(rebuilt, kps, template):
        return rebuilt
    return forward


def _maps_onto_template(matrix: np.ndarray, kps: np.ndarray, template: np.ndarray, tol: float = 8.0) -> bool:
    pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    if pts.shape != template.shape:
        return False
    mapped = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ np.asarray(matrix, dtype=np.float64).reshape(2, 3).T
    return float(np.linalg.norm(mapped - template, axis=1).max()) <= tol


def _estimate_similarity(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Umeyama similarity: ``src`` (frame landmarks) → ``dst`` (template)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 2:
        return None
    count = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    variance = float((src_c ** 2).sum() / count)
    if variance < 1e-8:
        return None
    cov = (dst_c.T @ src_c) / count
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = mu_dst - scale * (rotation @ mu_src)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def mask_reach_below_mouth(alpha: np.ndarray, kps: np.ndarray, threshold: float = 0.5) -> float:
    """How far below the mouth the mask still fully covers, in pixels."""
    axes = face_axes(kps)
    if axes is None:
        return 0.0
    _eye_c, mouth_c, down_u, _side_u, _em, _eye_dist = axes
    ys, xs = np.nonzero(alpha >= threshold)
    if ys.size == 0:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    along = (pts - mouth_c) @ down_u
    return float(along.max())

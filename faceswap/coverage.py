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


def coverage_polygon(kps: np.ndarray, coverage: str = DEFAULT_COVERAGE) -> Optional[np.ndarray]:
    """Landmark polygon covering forehead, cheeks, jaw, and beard or a tight oval."""
    axes = face_axes(kps)
    if axes is None:
        return None
    eye_c, mouth_c, down_u, side_u, em, eye_dist = axes
    mode = normalize_coverage(coverage)
    if mode == COVERAGE_FULL:
        depth = FULL_BEARD_DEPTH_EM * em
        jaw = FULL_JAW_WIDTH * eye_dist
        cheek = 0.90 * eye_dist
        temple = 0.72 * eye_dist
        forehead = 1.05 * em
        beard_w = 0.72 * eye_dist
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
            _pt(mouth_c, down_u, side_u, depth, 0.0),
            _pt(mouth_c, down_u, side_u, 0.55 * depth, beard_w),
            _pt(mouth_c, down_u, side_u, 0.20 * em, jaw),
            _pt(eye_c, down_u, side_u, 0.55 * em, cheek),
            _pt(eye_c, down_u, side_u, 0.10 * em, temple),
            _pt(eye_c, down_u, side_u, -forehead * 0.75, temple),
        ]
    )
    return pts


def coverage_alpha(shape: tuple[int, int], kps: np.ndarray, coverage: str = DEFAULT_COVERAGE) -> np.ndarray:
    """Float mask in ``[0, 1]``. The beard interior is 1; only the rim is soft."""
    height, width = shape
    alpha = np.zeros((height, width), dtype=np.float32)
    polygon = coverage_polygon(kps, coverage)
    axes = face_axes(kps)
    if polygon is None or axes is None:
        return alpha
    _eye_c, _mouth_c, _down_u, _side_u, _em, eye_dist = axes
    mode = normalize_coverage(coverage)
    # Full coverage keeps a short rim. A wide feather mixes two skin colors
    # and looks muddy; the beard interior stays fully replaced.
    feather = max(4.0, (0.11 if mode == COVERAGE_FULL else 0.10) * eye_dist)

    binary = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(binary, [np.round(polygon).astype(np.int32)], 255)
    if not np.any(binary):
        return alpha
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Reach full replacement a short distance inside the outline. A wide
    # ramp leaves the original face showing through the cheeks and jaw.
    t = np.clip(dist / (feather * 0.55), 0.0, 1.0)
    # Smoothstep: flat 0 outside, flat 1 once `feather` px inside the outline.
    alpha = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    return alpha


def _sample_maps(cx: np.ndarray, cy: np.ndarray, size: int, coverage: str) -> tuple[np.ndarray, np.ndarray]:
    """Map frame→crop coordinates onto the swapped image.

    Above the crop bottom this is 1:1, so eyes and mouth stay on the
    landmarks. Past the crop, full coverage replays the chin band downward
    across the beard instead of stopping.
    """
    sx = np.clip(cx, 1.0, size - 2.0)
    if normalize_coverage(coverage) != COVERAGE_FULL:
        sy = np.clip(cy, 1.0, size - 2.0)
        return sx.astype(np.float32), sy.astype(np.float32)

    mouth_y, eye_mouth = _mouth_metrics()
    cy_end = mouth_y + FULL_BEARD_DEPTH_EM * eye_mouth
    chin0 = 0.78 * (size - 1)
    chin1 = 0.96 * (size - 1)
    sy = np.clip(cy, 1.0, size - 2.0)
    below = cy > (size - 1)
    if np.any(below):
        span = max(cy_end - (size - 1), 1.0)
        u = np.clip((cy[below] - (size - 1)) / span, 0.0, 1.0)
        sy[below] = chin0 + u * (chin1 - chin0)
    return sx.astype(np.float32), sy.astype(np.float32)


def paste_swapped_face(
    frame_bgr: np.ndarray,
    swapped_bgr: np.ndarray,
    matrix: np.ndarray,
    kps: np.ndarray,
    coverage: str = DEFAULT_COVERAGE,
) -> np.ndarray:
    """Blend ``swapped_bgr`` (the 128px InSwapper output) onto ``frame_bgr``.

    ``matrix`` is the 2×3 frame→crop transform InSwapper used to build the crop.
    """
    mode = normalize_coverage(coverage)
    if swapped_bgr is None or swapped_bgr.size == 0 or matrix is None:
        return frame_bgr
    matrix = orient_paste_matrix(matrix, kps, size=int(swapped_bgr.shape[0]))
    alpha = coverage_alpha(frame_bgr.shape[:2], kps, mode)
    if float(alpha.max()) <= 0.0:
        return frame_bgr

    ys, xs = np.nonzero(alpha > 0.01)
    if ys.size == 0:
        return frame_bgr
    pad = 2
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(frame_bgr.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(frame_bgr.shape[1], int(xs.max()) + 1 + pad)

    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    m = np.asarray(matrix, dtype=np.float64)
    cx = m[0, 0] * grid_x + m[0, 1] * grid_y + m[0, 2]
    cy = m[1, 0] * grid_x + m[1, 1] * grid_y + m[1, 2]
    map_x, map_y = _sample_maps(cx, cy, swapped_bgr.shape[0], mode)
    warped = cv2.remap(
        swapped_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    out = frame_bgr.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    weight = alpha[y0:y1, x0:x1, None]
    warped_f = warped.astype(np.float32)
    if mode == COVERAGE_FULL:
        warped_f = match_edge_color(warped_f, roi, weight)
    blended = warped_f * weight + roi * (1.0 - weight)
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def match_edge_color(warped: np.ndarray, roi: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Move the soft rim of the swap toward the original frame color.

    The interior (alpha near 1) keeps the swapped color. Only the feather
    band shifts, so the blend is not a muddy average of two palettes.
    """
    alpha = weight[..., 0] if weight.ndim == 3 else weight
    rim = (alpha > 0.12) & (alpha < 0.88)
    if int(np.count_nonzero(rim)) < 16:
        return warped
    delta = roi[rim].mean(axis=0) - warped[rim].mean(axis=0)
    # A strong shift paints the original skin back onto the rim and the swap
    # reads as the source clip. Keep only a light edge correction.
    strength = np.clip((1.0 - alpha) * 0.35, 0.0, 1.0).astype(np.float32)[..., None]
    shifted = warped.astype(np.float32) + delta.astype(np.float32) * strength
    return np.clip(shifted, 0.0, 255.0)


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

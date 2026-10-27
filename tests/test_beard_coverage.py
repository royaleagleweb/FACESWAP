"""Full coverage replaces the beard. The tight oval does not."""

from __future__ import annotations

import numpy as np

import cv2

from faceswap.coverage import (
    FULL_BEARD_DEPTH_EM,
    coverage_alpha,
    face_axes,
    mask_reach_below_mouth,
    match_edge_color,
    orient_paste_matrix,
    paste_swapped_face,
    template_128,
)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    count = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    variance = (src_c ** 2).sum() / count
    cov = (dst_c.T @ src_c) / count
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(singular) @ sign) / variance
    translation = mu_dst - scale * (rotation @ mu_src)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def _landmarks() -> np.ndarray:
    """A real InSwapper template, placed in the frame by a known similarity."""
    dst = template_128()
    scale = 2.2
    center = dst.mean(axis=0)
    return (dst - center) * scale + np.array([200.0, 190.0])


def _beard_pixel(kps: np.ndarray, depth_em: float, lateral_eyes: float = 0.0) -> tuple[int, int]:
    eye_c, mouth_c, down_u, side_u, em, eye_dist = face_axes(kps)
    point = mouth_c + down_u * (depth_em * em) + side_u * (lateral_eyes * eye_dist)
    return int(round(point[0])), int(round(point[1]))


def test_full_mask_covers_beard_and_jaw_normal_does_not() -> None:
    kps = _landmarks()
    shape = (480, 460)
    full = coverage_alpha(shape, kps, "full")
    normal = coverage_alpha(shape, kps, "normal")

    beard_x, beard_y = _beard_pixel(kps, 1.55)
    jaw_x, jaw_y = _beard_pixel(kps, 0.35, lateral_eyes=0.80)
    cheek_x, cheek_y = _beard_pixel(kps, -0.25, lateral_eyes=0.70)

    assert full[beard_y, beard_x] > 0.95
    assert normal[beard_y, beard_x] < 0.05
    assert full[jaw_y, jaw_x] > 0.90
    assert normal[jaw_y, jaw_x] < 0.15
    assert full[cheek_y, cheek_x] > 0.90

    _eye_c, mouth_c, down_u, _side, em, _eye = face_axes(kps)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    along = (xx - mouth_c[0]) * down_u[0] + (yy - mouth_c[1]) * down_u[1]
    below = along > 0
    full_below = int(np.count_nonzero((full > 0.5) & below))
    normal_below = int(np.count_nonzero((normal > 0.5) & below))
    assert normal_below > 0
    assert full_below >= 2.5 * normal_below

    reach = mask_reach_below_mouth(full, kps)
    assert reach >= 1.7 * em
    assert mask_reach_below_mouth(normal, kps) < 0.7 * em


def test_full_paste_replaces_beard_pixels_normal_leaves_them() -> None:
    kps = _landmarks()
    matrix = _umeyama(kps, template_128())
    mapped = np.hstack([kps, np.ones((5, 1))]) @ matrix.T
    assert np.linalg.norm(mapped - template_128(), axis=1).max() < 0.5

    frame = np.zeros((480, 460, 3), dtype=np.uint8)
    # The whole target starts green, including the beard under the mouth.
    frame[:, :] = (0, 190, 0)

    swapped = np.full((128, 128, 3), 70, dtype=np.uint8)
    swapped[100:128, :] = (0, 0, 255)  # chin / beard color from the swap
    eye = template_128()[0]
    ex, ey = int(round(eye[0])), int(round(eye[1]))
    swapped[ey - 4:ey + 5, ex - 4:ex + 5] = (255, 0, 0)

    full = paste_swapped_face(frame, swapped, matrix, kps, "full")
    normal = paste_swapped_face(frame, swapped, matrix, kps, "normal")

    bx, by = _beard_pixel(kps, 1.5)
    # Full replacement: the green beard is gone at the beard center.
    assert full[by, bx, 1] < 40
    assert full[by, bx, 2] > 200
    # Normal stops short, so that same pixel is still the original green.
    assert normal[by, bx, 1] > 150
    assert normal[by, bx, 2] < 40

    # Eyes stay on the landmarks (the stretched chin is only used below the crop).
    left_eye = kps[0].astype(int)
    assert full[left_eye[1], left_eye[0], 0] > 200

    # The rim is soft, the interior is solid.
    alpha = coverage_alpha(frame.shape[:2], kps, "full")
    assert alpha.max() >= 0.99
    assert np.any((alpha > 0.2) & (alpha < 0.8))
    assert FULL_BEARD_DEPTH_EM >= 1.8


def test_inverted_matrix_is_reoriented_before_paste() -> None:
    kps = _landmarks()
    matrix = _umeyama(kps, template_128())
    inverse = cv2.invertAffineTransform(matrix.astype(np.float64))
    recovered = orient_paste_matrix(inverse, kps)
    assert np.allclose(recovered, matrix, atol=1e-4)

    frame = np.zeros((480, 460, 3), dtype=np.uint8)
    frame[:, :] = (0, 190, 0)
    swapped = np.full((128, 128, 3), 70, dtype=np.uint8)
    swapped[100:128, :] = (0, 0, 255)
    eye = template_128()[0]
    ex, ey = int(round(eye[0])), int(round(eye[1]))
    swapped[ey - 4:ey + 5, ex - 4:ex + 5] = (255, 0, 0)

    corrected = paste_swapped_face(frame, swapped, inverse, kps, "full")
    bx, by = _beard_pixel(kps, 1.5)
    assert corrected[by, bx, 2] > 200
    assert corrected[by, bx, 1] < 40
    left_eye = kps[0].astype(int)
    assert corrected[left_eye[1], left_eye[0], 0] > 200


def test_degenerate_landmarks_keep_the_given_matrix() -> None:
    matrix = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, 4.0]], dtype=np.float64)
    kept = orient_paste_matrix(matrix, np.zeros((5, 2), dtype=np.float64))
    assert np.allclose(kept, matrix)


def test_edge_color_match_keeps_the_interior() -> None:
    warped = np.zeros((20, 20, 3), dtype=np.float32)
    warped[:] = (0, 0, 255)
    roi = np.zeros_like(warped)
    roi[:] = (0, 180, 0)
    weight = np.full((20, 20, 1), 0.35, dtype=np.float32)
    weight[4:16, 4:16] = 1.0
    shifted = match_edge_color(warped, roi, weight)
    assert shifted[8, 8, 2] == 255
    assert shifted[8, 8, 1] == 0
    assert shifted[0, 0, 2] < 255
    assert shifted[0, 0, 1] > 20

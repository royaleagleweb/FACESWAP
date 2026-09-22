"""BGR frame helpers for the desktop previews."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from faceswap.face_analyzer import Face


def crop_face(frame_bgr: np.ndarray, face: Face, pad: float = 0.35) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    pad_x = int((x2 - x1) * pad)
    pad_y = int((y2 - y1) * pad)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr.copy()
    return frame_bgr[y1:y2, x1:x2].copy()


def side_by_side(original_bgr: np.ndarray, swapped_bgr: np.ndarray) -> np.ndarray:
    """Original on the left, swapped frame on the right, same height."""
    left = np.array(original_bgr, copy=True)
    right = np.array(swapped_bgr, copy=True)
    if right.shape[0] != left.shape[0] or right.shape[1] != left.shape[1]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.putText(left, "Original", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "Swapped", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    gap = np.full((left.shape[0], 8, 3), 32, dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def bgr_to_qpixmap(image: np.ndarray, max_edge: Optional[int] = None) -> QPixmap:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    if max_edge is not None:
        height, width = rgb.shape[:2]
        longest = max(height, width)
        if longest > max_edge and longest > 0:
            scale = max_edge / longest
            rgb = cv2.resize(
                rgb,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            rgb = np.ascontiguousarray(rgb)
    height, width, _ = rgb.shape
    qimg = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

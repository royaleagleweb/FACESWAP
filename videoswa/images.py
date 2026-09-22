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

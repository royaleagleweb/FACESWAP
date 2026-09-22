"""Occlusion mask for normal swaps.

``_occ_mask`` always runs. It keeps the swap inside the face polygon and
punches out objects that do not match the face color (a lollipop, food, a
hand that is not the same color as the cheek). When ``models/xseg.onnx`` is
present it is an extra XSeg pass, warped back into that face region only.
The file is downloaded into ``models/xseg.onnx`` on the first swap.
``models/bisenet.onnx`` downloads only when precise edges are on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .coverage import coverage_alpha, face_axes
from .utils import logger

XSEG_FILENAME = "xseg.onnx"
BISENET_FILENAME = "bisenet.onnx"
_COLOR_DISTANCE = 70.0
_MAX_OBJECT_FRACTION = 0.55


def _occ_mask(
    frame_bgr: np.ndarray,
    kps: np.ndarray,
    coverage: str = "full",
    *,
    neural: Optional[np.ndarray] = None,
    precise: Optional[np.ndarray] = None,
    yaw: float = 0.0,
) -> np.ndarray:
    """Swap weight for one face. 1 replaces, 0 keeps the original.

    The result is zero outside the face polygon. It is not a full-frame mask.
    ``neural`` and ``precise`` are optional face-sized maps (1 = occluder for
    neural, 1 = keep for precise) already warped into the frame.
    """
    alpha = coverage_alpha(frame_bgr.shape[:2], kps, coverage, yaw=yaw)
    if float(alpha.max()) <= 0.0:
        return alpha
    keepout = _color_objects(frame_bgr, alpha, kps)
    if neural is not None and neural.shape[:2] == alpha.shape:
        neural_f = np.clip(neural.astype(np.float32), 0.0, 1.0)
        keepout = np.maximum(keepout, neural_f * (alpha > 0.05))
    weight = alpha * (1.0 - np.clip(keepout, 0.0, 1.0))
    if precise is not None and precise.shape[:2] == alpha.shape:
        # Tighten only the soft rim. The beard interior stays put.
        edge = (alpha > 0.05) & (alpha < 0.95)
        precise_f = np.clip(precise.astype(np.float32), 0.0, 1.0)
        weight = np.where(edge, weight * precise_f, weight).astype(np.float32)
    return weight.astype(np.float32)


def _color_objects(frame_bgr: np.ndarray, alpha: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """1 on compact regions whose color is far from the face median.

    A wide dark band under the mouth is a beard, not a lollipop, so it stays
    in the swap. A small blob (food, a stick, a hand of a different color)
    is punched out.
    """
    keepout = np.zeros(alpha.shape, dtype=np.float32)
    core = alpha > 0.85
    count = int(np.count_nonzero(core))
    if count < 32:
        return keepout
    median = np.median(frame_bgr[core].astype(np.float32), axis=0)
    distance = np.linalg.norm(frame_bgr.astype(np.float32) - median, axis=2)
    odd = (distance > _COLOR_DISTANCE) & (alpha > 0.4)
    odd = _spare_beard(odd, kps)
    if int(np.count_nonzero(odd)) > _MAX_OBJECT_FRACTION * count:
        return keepout
    speckle = cv2.morphologyEx(
        odd.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    keepout[speckle > 0] = 1.0
    return keepout


def _spare_beard(odd: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Drop wide components that sit below the mouth. Those are beard, not objects."""
    axes = face_axes(kps)
    if axes is None or not np.any(odd):
        return odd
    _eye_c, mouth_c, down_u, side_u, em, eye_dist = axes
    count, labels = cv2.connectedComponents(odd.astype(np.uint8))
    if count <= 1:
        return odd
    spared = odd.copy()
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if xs.size < 8:
            continue
        px = xs.astype(np.float64)
        py = ys.astype(np.float64)
        along = (px - mouth_c[0]) * down_u[0] + (py - mouth_c[1]) * down_u[1]
        lateral = (px - mouth_c[0]) * side_u[0] + (py - mouth_c[1]) * side_u[1]
        below_mouth = float(np.mean(along > 0.15 * em)) > 0.7
        spans_jaw = float(lateral.max() - lateral.min()) > 0.7 * eye_dist
        if below_mouth and spans_jaw:
            spared[labels == label] = False
    return spared


def warp_into_face(
    crop_mask: np.ndarray,
    frame_to_crop: np.ndarray,
    frame_shape: tuple[int, int],
    face_alpha: np.ndarray,
) -> np.ndarray:
    """Warp a crop mask into the frame and zero it outside the face."""
    height, width = frame_shape
    inverse = cv2.invertAffineTransform(np.asarray(frame_to_crop, dtype=np.float64))
    warped = cv2.warpAffine(
        crop_mask.astype(np.float32),
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    if warped.shape != face_alpha.shape:
        warped = cv2.resize(warped, (face_alpha.shape[1], face_alpha.shape[0]))
    return warped * (face_alpha > 0.05).astype(np.float32)


class OcclusionModels:
    """Optional XSeg and BiSeNet sessions on the same provider chain as InSwapper."""

    def __init__(self, execution: str = "auto", *, precise: bool = False) -> None:
        self.execution = execution
        self.precise = bool(precise)
        self.xseg = None
        self.bisenet = None
        self.xseg_note = ""
        self.bisenet_note = ""
        self._load()

    def _load(self) -> None:
        from .utils import BISENET_SHA256, BISENET_URLS, XSEG_SHA256, XSEG_URLS, ensure_model

        xseg_path = ensure_model(XSEG_URLS, XSEG_FILENAME, XSEG_SHA256, "XSeg")
        self.xseg, self.xseg_note = _optional_session(xseg_path, self.execution, "XSeg")
        if not self.precise:
            self.bisenet = None
            self.bisenet_note = ""
            return
        bisenet_path = ensure_model(BISENET_URLS, BISENET_FILENAME, BISENET_SHA256, "BiSeNet")
        self.bisenet, self.bisenet_note = _optional_session(bisenet_path, self.execution, "BiSeNet")

    @property
    def status(self) -> str:
        parts = [self.xseg_note]
        if self.bisenet_note:
            parts.append(self.bisenet_note)
        return " ".join(part for part in parts if part)

    def xseg_keepout(
        self,
        frame_bgr: np.ndarray,
        matrix: np.ndarray,
        face_alpha: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.xseg is None:
            return None
        crop = _warp_crop(frame_bgr, matrix, _input_hw(self.xseg))
        mask = _session_mask(self.xseg, crop, occluder=True)
        if mask is None:
            return None
        return warp_into_face(mask, matrix, frame_bgr.shape[:2], face_alpha)

    def bisenet_keep(
        self,
        frame_bgr: np.ndarray,
        matrix: np.ndarray,
        face_alpha: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.bisenet is None:
            return None
        crop = _warp_crop(frame_bgr, matrix, _input_hw(self.bisenet))
        mask = _session_mask(self.bisenet, crop, occluder=False)
        if mask is None:
            return None
        return warp_into_face(mask, matrix, frame_bgr.shape[:2], face_alpha)


def _optional_session(path: Optional[Path], execution: str, label: str):
    if path is None or not path.is_file():
        note = (
            f"{label} is not on disk yet. "
            + (
                "Built-in object mask is on."
                if label == "XSeg"
                else "Precise edges stay off until the download succeeds."
            )
        )
        logger.info(note)
        return None, note
    try:
        from .providers import open_onnx_session

        session, _providers, active = open_onnx_session(path, execution, what=label)
    except Exception as exc:
        note = f"{label} did not start ({exc}). The swap continues without it."
        logger.warning(note)
        return None, note
    logger.info("%s active providers: %s", label, active)
    return session, f"{label} is on ({', '.join(active) or 'unreported'})."


def _input_hw(session) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    if len(shape) == 4 and shape[-1] == 3:
        height = shape[1] if isinstance(shape[1], int) else 256
        width = shape[2] if isinstance(shape[2], int) else 256
        return int(height), int(width)
    height = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 256
    width = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else 256
    return int(height), int(width)


def _warp_crop(frame_bgr: np.ndarray, matrix: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    height, width = hw
    crop = cv2.warpAffine(
        frame_bgr,
        np.asarray(matrix, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop


# CelebAMask-HQ labels that belong on the swapped face, including ears and hair
# so a side view is not cut off at the cheek.
_BISE_KEEP = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _session_mask(session, crop_bgr: np.ndarray, *, occluder: bool) -> Optional[np.ndarray]:
    try:
        blob = _model_input(session, crop_bgr, imagenet=not occluder)
        name = session.get_inputs()[0].name
        output = session.run(None, {name: blob})[0]
    except Exception as exc:
        logger.warning("Occlusion model failed: %s", exc)
        return None
    array = np.asarray(output)
    if not occluder and array.ndim == 4 and array.shape[1] >= 19:
        classes = np.argmax(array[0], axis=0).astype(np.int32)
        keep = np.isin(classes, list(_BISE_KEEP)).astype(np.float32)
        return keep
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] <= 32:
        mask = array[0] if occluder else (array[1:].max(axis=0) if array.shape[0] > 1 else array[0])
    elif array.ndim == 3:
        mask = array[:, :, 0]
    else:
        mask = array
    mask = mask.astype(np.float32)
    if float(np.nanmax(mask)) > 1.5:
        mask = mask / 255.0
    mask = np.clip(mask, 0.0, 1.0)
    # The shipped XSeg file is a face mask (1 = face). Punch out the rest.
    if occluder:
        mask = 1.0 - mask
    return mask


def _model_input(session, crop_bgr: np.ndarray, *, imagenet: bool) -> np.ndarray:
    """NHWC BGR for XSeg, NCHW RGB for BiSeNet."""
    shape = session.get_inputs()[0].shape
    height, width = _input_hw(session)
    if len(shape) == 4 and shape[-1] == 3:
        image = cv2.resize(crop_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        return (image.astype(np.float32) / 255.0)[None, ...]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if imagenet:
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(image, (2, 0, 1))
    return chw[None, ...].astype(np.float32)

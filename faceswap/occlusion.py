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

from .coverage import coverage_alpha, face_axes, face_roi
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
    base_alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Swap weight for one face. 1 replaces, 0 keeps the original.

    The result is zero outside the face polygon. It is not a full-frame mask.
    ``neural`` and ``precise`` are optional face-sized maps (1 = occluder for
    neural, 1 = keep for precise) already warped into the frame.
    """
    alpha = base_alpha if base_alpha is not None else coverage_alpha(frame_bgr.shape[:2], kps, coverage, yaw=yaw)
    if alpha.size == 0 or float(np.max(alpha)) <= 0.0:
        return alpha
    # The landmark box, not a scan of the whole 1080p frame.
    x0, y0, x1, y1 = face_roi(kps, alpha.shape[:2], pad_scale=1.6)
    if float(np.max(alpha[y0:y1, x0:x1])) <= 0.0:
        return alpha
    keep = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    color = _color_objects(frame_bgr, alpha, kps)
    if color is not None:
        cy0, cy1, cx0, cx1, speckle = color
        iy0, iy1 = max(y0, cy0), min(y1, cy1)
        ix0, ix1 = max(x0, cx0), min(x1, cx1)
        if iy1 > iy0 and ix1 > ix0:
            keep[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = np.maximum(
                keep[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0],
                speckle[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0],
            )
    _paint_keep(keep, neural, y0, x0)
    rim = np.array(alpha[y0:y1, x0:x1], dtype=np.float32, copy=True)
    precise_view = _layer_view(precise, y0, x0, keep.shape)
    alpha[y0:y1, x0:x1] = rim * (1.0 - np.clip(keep, 0.0, 1.0))
    if precise_view is not None:
        # Tighten only the soft rim. The beard interior stays put.
        updated = alpha[y0:y1, x0:x1]
        edge = (rim > 0.05) & (rim < 0.95)
        alpha[y0:y1, x0:x1] = np.where(edge, updated * precise_view, updated)
    return alpha.astype(np.float32)


def _layer_view(layer, y0: int, x0: int, shape: tuple[int, int]) -> Optional[np.ndarray]:
    """The face-box slice of a full-frame map or a (y, x, patch)."""
    if layer is None:
        return None
    height, width = shape
    if isinstance(layer, tuple):
        ly, lx, patch = int(layer[0]), int(layer[1]), np.asarray(layer[2], dtype=np.float32)
        view = np.ones((height, width), dtype=np.float32)
        ph, pw = patch.shape[:2]
        iy0, iy1 = max(0, ly - y0), min(height, ly - y0 + ph)
        ix0, ix1 = max(0, lx - x0), min(width, lx - x0 + pw)
        if iy1 <= iy0 or ix1 <= ix0:
            return None
        view[iy0:iy1, ix0:ix1] = patch[iy0 - (ly - y0):iy1 - (ly - y0), ix0 - (lx - x0):ix1 - (lx - x0)]
        return np.clip(view, 0.0, 1.0)
    if getattr(layer, "ndim", 0) < 2:
        return None
    return np.clip(np.asarray(layer[y0:y0 + height, x0:x0 + width], dtype=np.float32), 0.0, 1.0)


def _paint_keep(keep, layer, y0: int, x0: int, rim_only: Optional[np.ndarray] = None) -> None:
    """Max a full-frame map or a (y, x, patch) into the face-box keepout."""
    if layer is None:
        return
    if isinstance(layer, tuple):
        ly, lx, patch = layer
        patch = np.clip(np.asarray(patch, dtype=np.float32), 0.0, 1.0)
        py0, px0 = int(ly), int(lx)
        ph, pw = patch.shape[:2]
        iy0, iy1 = max(y0, py0), min(y0 + keep.shape[0], py0 + ph)
        ix0, ix1 = max(x0, px0), min(x0 + keep.shape[1], px0 + pw)
        if iy1 <= iy0 or ix1 <= ix0:
            return
        block = patch[iy0 - py0:iy1 - py0, ix0 - px0:ix1 - px0]
        dest = keep[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        if rim_only is not None:
            rim = rim_only[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
            edge = (rim > 0.05) & (rim < 0.95)
            dest[...] = np.where(edge, np.maximum(dest, block), dest)
            return
        np.maximum(dest, block, out=dest)
        return
    if getattr(layer, "shape", None) is None:
        return
    view = np.clip(np.asarray(layer[y0:y0 + keep.shape[0], x0:x0 + keep.shape[1]], dtype=np.float32), 0.0, 1.0)
    if rim_only is not None:
        edge = (rim_only > 0.05) & (rim_only < 0.95)
        keep[...] = np.where(edge, np.maximum(keep, view), keep)
        return
    np.maximum(keep, view, out=keep)


def _color_objects(frame_bgr: np.ndarray, alpha: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """1 on compact regions whose color is far from the face median.

    The work happens on the face crop, scaled so the long side is at most
    128. A full-frame connected-component pass was about 50 ms at 480p.
    """
    x0, y0, x1, y1 = face_roi(kps, alpha.shape[:2], pad_scale=1.6)
    roi_alpha = alpha[y0:y1, x0:x1]
    if int(np.count_nonzero(roi_alpha > 0.2)) < 32:
        return None
    roi_bgr = frame_bgr[y0:y1, x0:x1]
    long_side = max(roi_alpha.shape[:2])
    scale = 1.0 if long_side <= 128 else 128.0 / float(long_side)
    if scale < 1.0:
        size = (max(1, int(round(roi_alpha.shape[1] * scale))), max(1, int(round(roi_alpha.shape[0] * scale))))
        small_alpha = cv2.resize(roi_alpha, size, interpolation=cv2.INTER_AREA)
        small_bgr = cv2.resize(roi_bgr, size, interpolation=cv2.INTER_AREA)
    else:
        small_alpha = roi_alpha
        small_bgr = roi_bgr
    core = small_alpha > 0.85
    count = int(np.count_nonzero(core))
    if count < 16:
        return None
    median = np.median(small_bgr[core].astype(np.float32), axis=0)
    distance = np.linalg.norm(small_bgr.astype(np.float32) - median, axis=2)
    odd = (distance > _COLOR_DISTANCE) & (small_alpha > 0.4)
    local_kps = (np.asarray(kps, dtype=np.float32).reshape(-1, 2) - np.array([x0, y0], dtype=np.float32)) * scale
    odd = _spare_beard(odd, local_kps)
    if int(np.count_nonzero(odd)) > _MAX_OBJECT_FRACTION * count:
        return None
    speckle = cv2.morphologyEx(
        odd.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    if scale < 1.0:
        speckle = cv2.resize(speckle, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return y0, y1, x0, x1, (speckle > 0).astype(np.float32)


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
        px = xs.astype(np.float32)
        py = ys.astype(np.float32)
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
    compact: bool = False,
    kps: Optional[np.ndarray] = None,
):
    """Warp a crop mask into the frame and zero it outside the face."""
    height, width = frame_shape
    if kps is not None:
        x0, y0, x1, y1 = face_roi(kps, frame_shape, pad_scale=1.6)
        if x1 <= x0 or y1 <= y0 or float(np.max(face_alpha[y0:y1, x0:x1])) <= 0.05:
            ys = np.empty(0)
        else:
            ys = np.array([y0])
    else:
        ys, xs = np.nonzero(face_alpha > 0.05)
    if ys.size == 0:
        if compact:
            return 0, 0, np.zeros((1, 1), dtype=np.float32)
        return np.zeros((height, width), dtype=np.float32)
    if kps is None:
        y0 = max(0, int(ys.min()) - 2)
        x0 = max(0, int(xs.min()) - 2)
        y1 = min(height, int(ys.max()) + 3)
        x1 = min(width, int(xs.max()) + 3)
    inverse = cv2.invertAffineTransform(np.asarray(frame_to_crop, dtype=np.float64))
    inverse = inverse.copy()
    inverse[0, 2] -= x0
    inverse[1, 2] -= y0
    warped = cv2.warpAffine(
        crop_mask.astype(np.float32),
        inverse,
        (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    patch = warped * (face_alpha[y0:y1, x0:x1] > 0.05)
    if compact:
        return y0, x0, np.ascontiguousarray(patch.astype(np.float32))
    out = np.zeros((height, width), dtype=np.float32)
    out[y0:y1, x0:x1] = patch
    return out


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
        canvas: int = 128,
    ) -> Optional[np.ndarray]:
        if self.xseg is None:
            return None
        hw = _input_hw(self.xseg)
        aligned = _scale_matrix(matrix, int(canvas), hw)
        crop = _warp_crop(frame_bgr, aligned, hw)
        mask = _session_mask(self.xseg, crop, occluder=True)
        if mask is None:
            return None
        return warp_into_face(mask, aligned, frame_bgr.shape[:2], face_alpha)

    def bisenet_keep(
        self,
        frame_bgr: np.ndarray,
        matrix: np.ndarray,
        face_alpha: np.ndarray,
        canvas: int = 512,
    ) -> Optional[np.ndarray]:
        if self.bisenet is None:
            return None
        hw = _input_hw(self.bisenet)
        aligned = _scale_matrix(matrix, int(canvas), hw)
        crop = _warp_crop(frame_bgr, aligned, hw)
        mask = _session_mask(self.bisenet, crop, occluder=False)
        if mask is None:
            return None
        return warp_into_face(mask, aligned, frame_bgr.shape[:2], face_alpha)


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


def _scale_matrix(matrix: np.ndarray, canvas: int, hw: tuple[int, int]) -> np.ndarray:
    """Scale a frame→canvas matrix so the face fills this model's input.

    A 256px pixel-boost matrix used with a smaller mask input only covered
    the top-left of the face, and the mask came back off the face. The paste
    then looked unchanged and was thrown away.
    """
    mapped = np.asarray(matrix, dtype=np.float64).reshape(2, 3).copy()
    if canvas <= 0:
        return mapped
    mapped[0] *= float(hw[1]) / float(canvas)
    mapped[1] *= float(hw[0]) / float(canvas)
    return mapped


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


def session_masks(session, crops: list[np.ndarray], *, occluder: bool) -> list[Optional[np.ndarray]]:
    """Run one mask model on several face crops. One batch when the session allows it."""
    if not crops:
        return []
    blobs = [_model_input(session, crop, imagenet=not occluder) for crop in crops]
    stacked = np.concatenate(blobs, axis=0)
    name = session.get_inputs()[0].name
    outputs = None
    if stacked.shape[0] > 1 and getattr(session, "_videoswa_mask_batch", True):
        try:
            outputs = session.run(None, {name: stacked})[0]
            session._videoswa_mask_batch = True
        except Exception:
            session._videoswa_mask_batch = False
            outputs = None
    masks: list[Optional[np.ndarray]] = []
    if outputs is None:
        for crop in crops:
            masks.append(_session_mask(session, crop, occluder=occluder))
        return masks
    for index in range(outputs.shape[0]):
        masks.append(_mask_from_output(outputs[index:index + 1], occluder=occluder))
    return masks


def _session_mask(session, crop_bgr: np.ndarray, *, occluder: bool) -> Optional[np.ndarray]:
    try:
        blob = _model_input(session, crop_bgr, imagenet=not occluder)
        name = session.get_inputs()[0].name
        output = session.run(None, {name: blob})[0]
    except Exception as exc:
        logger.warning("Occlusion model failed: %s", exc)
        return None
    return _mask_from_output(output, occluder=occluder)


def _mask_from_output(output, *, occluder: bool) -> Optional[np.ndarray]:
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

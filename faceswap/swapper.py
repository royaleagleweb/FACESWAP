"""Face swap engine wrapping the InsightFace inswapper_128 ONNX model.

Optionally enhances results with GFPGAN if installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .face_analyzer import Face
from .utils import ensure_inswapper, logger, select_providers


class FaceSwapper:
    """Swap a target face in a frame with a source face's identity."""

    def __init__(self, use_gpu: bool = True, enhance: bool = False) -> None:
        import insightface  # local import

        model_path = ensure_inswapper()
        providers = select_providers(use_gpu=use_gpu)
        logger.info("FaceSwapper providers: %s", providers)
        self.swapper = insightface.model_zoo.get_model(
            str(model_path), providers=providers
        )
        self._enhancer = None
        if enhance:
            self._enhancer = _try_load_gfpgan(use_gpu=use_gpu)

    def swap(
        self,
        frame: np.ndarray,
        target_face: Face,
        source_face: Face,
        paste_back: bool = True,
    ) -> np.ndarray:
        """Replace target_face in frame with source_face identity."""
        # InsightFace expects its own Face objects, but the swapper actually only
        # uses .kps and .normed_embedding. We pass a small shim.
        src_shim = _FaceShim(source_face)
        tgt_shim = _FaceShim(target_face)
        out = self.swapper.get(frame, tgt_shim, src_shim, paste_back=paste_back)
        if self._enhancer is not None:
            out = _enhance_face_region(out, target_face.bbox, self._enhancer)
        return out


class _FaceShim:
    """Duck-typed object compatible with insightface's INSwapper.get."""

    def __init__(self, face: Face) -> None:
        self.kps = face.kps
        self.bbox = face.bbox
        self.embedding = face.embedding
        self.normed_embedding = face.normed_embedding
        self.det_score = face.det_score


def _try_load_gfpgan(use_gpu: bool):
    try:
        from gfpgan import GFPGANer
    except Exception as exc:
        logger.warning("GFPGAN not available (%s); skipping enhancement", exc)
        return None
    try:
        weight_path = _ensure_gfpgan_weights()
        device = "cuda" if use_gpu else "cpu"
        return GFPGANer(
            model_path=str(weight_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
    except Exception as exc:
        logger.warning("Failed to initialize GFPGAN: %s", exc)
        return None


def _ensure_gfpgan_weights() -> Path:
    from .utils import MODELS_DIR, download_file
    target = MODELS_DIR / "GFPGANv1.4.pth"
    urls = [
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    ]
    return download_file(urls, target, expected_sha256=None)


def _enhance_face_region(frame: np.ndarray, bbox: np.ndarray, enhancer) -> np.ndarray:
    """Run GFPGAN on the cropped face region with a generous margin."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    pad_x = int((x2 - x1) * 0.4)
    pad_y = int((y2 - y1) * 0.4)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return frame
    crop = frame[y1:y2, x1:x2].copy()
    try:
        _, _, restored = enhancer.enhance(
            crop, has_aligned=False, only_center_face=True, paste_back=True
        )
    except Exception as exc:
        logger.debug("Enhancer failed on region: %s", exc)
        return frame
    if restored is None:
        return frame
    if restored.shape[:2] != crop.shape[:2]:
        import cv2
        restored = cv2.resize(restored, (crop.shape[1], crop.shape[0]))
    out = frame.copy()
    out[y1:y2, x1:x2] = restored
    return out

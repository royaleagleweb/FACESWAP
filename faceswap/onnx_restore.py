"""Optional ONNX face restorer.

GPEN-BFR-512 and a CodeFormer ONNX file are used when they are already in
``models/``. There is no official checksummed ONNX download for either, so
Videoswa does not fetch one. GFPGAN remains the downloadable fallback.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .utils import MODELS_DIR, logger

GPEN_FILENAME = "GPEN-BFR-512.onnx"
CODEFORMER_ONNX = "codeformer.onnx"


def _find_model():
    for name in (GPEN_FILENAME, CODEFORMER_ONNX):
        path = MODELS_DIR / name
        if path.is_file():
            return path
    return None


def gpen_session(execution: str = "auto"):
    path = _find_model()
    if path is None:
        return None
    try:
        from .providers import open_onnx_session

        session, _providers, active = open_onnx_session(path, execution, what="GPEN")
    except Exception as exc:
        logger.info("GPEN ONNX did not start (%s). GFPGAN remains the fallback.", exc)
        return None
    logger.info("GPEN ONNX providers=%s file=%s", active, path.name)
    return session


def gpen_status(session) -> str:
    if session is None:
        return (
            "GPEN ONNX is not in models/. "
            "Place GPEN-BFR-512.onnx or codeformer.onnx there to use it. GFPGAN is the fallback."
        )
    return "GPEN ONNX is loaded."


def restore_gpen(session, bgr: np.ndarray, strength: float) -> np.ndarray:
    """Run the ONNX restorer on one face crop and blend it back."""
    shape = session.get_inputs()[0].shape
    side = 512
    if len(shape) >= 4 and isinstance(shape[-1], int) and shape[-1] != 3:
        side = int(shape[-1]) if isinstance(shape[-1], int) else 512
    elif len(shape) >= 3 and isinstance(shape[2], int):
        side = int(shape[2])
    resized = cv2.resize(bgr, (side, side), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    name = session.get_inputs()[0].name
    pred = session.run(None, {name: blob})[0]
    plane = pred[0] if pred.ndim == 4 else pred
    if plane.shape[0] == 3:
        rgb_out = np.transpose(plane, (1, 2, 0))
    else:
        rgb_out = plane
    if float(np.nanmax(rgb_out)) <= 1.5:
        rgb_out = rgb_out * 255.0
    out = cv2.cvtColor(np.clip(rgb_out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if out.shape[0] != bgr.shape[0] or out.shape[1] != bgr.shape[1]:
        out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    gain = float(np.clip(strength, 0.0, 1.0))
    if gain >= 0.999:
        return out
    mixed = bgr.astype(np.float32) * (1.0 - gain) + out.astype(np.float32) * gain
    return np.clip(mixed, 0, 255).astype(np.uint8)

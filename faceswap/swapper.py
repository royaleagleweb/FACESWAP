"""Face swap engine wrapping the InsightFace inswapper_128 ONNX model.

Optionally enhances results with GFPGAN if installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .coverage import DEFAULT_COVERAGE, coverage_alpha, normalize_coverage, paste_swapped_face
from .face_analyzer import Face
from .occlusion import _occ_mask
from .pose import face_yaw, repair_landmarks
from .quality import RESTORE_MIN_PX, face_span_px
from .providers import (
    active_providers_from_sessions,
    provider_attempts,
    run_with_cuda_fallback,
    run_with_provider_fallback,
    uses_gpu,
)
from .utils import ensure_inswapper, logger


class FaceSwapper:
    """Swap a target face in a frame with a source face's identity.

    ``execution="auto"`` registers TensorRT, then CUDA, then CPU. A failed
    TensorRT session is retried without it. ``use_gpu=False`` forces CPU.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        enhance: bool = False,
        execution: str = "auto",
    ) -> None:
        self._enhance_requested = bool(enhance)
        self.object_mask = True
        self.precise_edges = False
        self.allow_restore = bool(enhance)
        self.color_memory: dict[int, np.ndarray] = {}
        self._color_key: Optional[int] = None
        self._occlusion = None
        self._load(use_gpu=use_gpu, execution=execution, enhance=enhance)

    def _load(self, use_gpu: bool, execution: str, enhance: bool) -> None:
        import insightface  # local import

        model_path = ensure_inswapper()
        mode = "cpu" if not use_gpu else execution
        attempts = provider_attempts(mode)

        def _load(providers):
            model = insightface.model_zoo.get_model(str(model_path), providers=providers)
            session = getattr(model, "session", None)
            active = active_providers_from_sessions([session] if session is not None else [])
            if uses_gpu(providers) and active == ["CPUExecutionProvider"]:
                raise RuntimeError(
                    "InSwapper fell back to CPU. TensorRT or CUDA libraries "
                    "are probably missing from PATH."
                )
            return model, active

        loaded, providers = run_with_provider_fallback(
            attempts, _load, what="InSwapper (inswapper_128)"
        )
        self.swapper, self.active_providers = loaded
        self.providers = providers
        self.execution = mode
        self._on_cpu = not uses_gpu(providers)
        logger.info("InSwapper active providers: %s", self.active_providers or "(unreported)")
        self._enhancer = None
        if enhance:
            self._enhancer = _try_load_gfpgan(use_gpu=uses_gpu(providers))

    def adopt_cpu(self) -> None:
        """Drop a GPU session that failed while the graph was running."""
        if self._on_cpu:
            return
        self._load(use_gpu=False, execution="cpu", enhance=self._enhance_requested)

    def adopt_execution(self, execution: str) -> None:
        """Reload on DirectML or CPU. InSwapper weights stay on disk."""
        if execution == "cpu":
            self.adopt_cpu()
            return
        self._load(use_gpu=True, execution=execution, enhance=self._enhance_requested)

    def set_enhance(self, enhance: bool) -> Optional[str]:
        """Toggle GFPGAN without reloading the InSwapper session.

        Returns an install tip when sharpening was requested and could not start.
        """
        enhance = bool(enhance)
        if enhance and self._enhancer is not None:
            self._enhance_requested = True
            return None
        if not enhance:
            self._enhance_requested = False
            self._enhancer = None
            return None
        tip = gfpgan_install_tip()
        if tip is not None:
            self._enhance_requested = False
            self._enhancer = None
            return tip
        self._enhance_requested = True
        self._enhancer = _try_load_gfpgan(use_gpu=not self._on_cpu)
        if self._enhancer is None:
            self._enhance_requested = False
            return (
                "GFPGAN is installed but did not start. "
                "Videoswa downloads GFPGANv1.4.pth into the models folder on first use."
            )
        return None

    def begin_face(self, key: int) -> None:
        """Remember which tracked person the next ``swap`` call belongs to."""
        self._color_key = key

    def reset_color_memory(self) -> None:
        self.color_memory.clear()
        self._color_key = None

    def occlusion_models(self):
        precise = bool(self.precise_edges)
        current = self._occlusion
        if (
            current is None
            or getattr(current, "execution", None) != self.execution
            or getattr(current, "precise", False) != precise
        ):
            from .occlusion import OcclusionModels

            self._occlusion = OcclusionModels(self.execution, precise=precise)
        return self._occlusion

    def swap(
        self,
        frame: np.ndarray,
        target_face: Face,
        source_face: Face,
        paste_back: bool = True,
        coverage: str = DEFAULT_COVERAGE,
    ) -> np.ndarray:
        """Replace target_face in frame with source_face identity.

        ``coverage="full"`` (default) extends the paste over the jaw and beard.
        ``coverage="normal"`` keeps the tight face oval.
        """
        # InsightFace expects its own Face objects, but the swapper actually only
        # uses .kps and .normed_embedding. We pass a small shim.
        src_shim = _FaceShim(source_face)
        tgt_shim = _FaceShim(target_face)
        # Profile landmarks collapse the far eye. Repair them before InSwapper
        # builds the frontal crop, and keep that same set for the paste.
        raw_kps = np.asarray(target_face.kps, dtype=np.float32)
        yaw = float(face_yaw(raw_kps))
        align_kps = repair_landmarks(raw_kps)
        tgt_shim.kps = align_kps
        # paste_back=False returns the 128px swap plus the frame→crop matrix.
        # Videoswa composites that itself so the beard is not cropped off.
        swapped, matrix = run_with_cuda_fallback(
            getattr(self, "cuda_guard", None),
            lambda: self.swapper.get(frame, tgt_shim, src_shim, paste_back=False),
        )
        if not paste_back:
            return swapped
        mode = normalize_coverage(coverage)
        alpha = None
        if self.object_mask:
            models = self.occlusion_models()
            face_alpha = coverage_alpha(frame.shape[:2], align_kps, mode, yaw=yaw)
            neural = models.xseg_keepout(frame, matrix, face_alpha) if models.xseg is not None else None
            precise = None
            if self.precise_edges and models.bisenet is not None:
                precise = models.bisenet_keep(frame, matrix, face_alpha)
            alpha = _occ_mask(
                frame,
                align_kps,
                mode,
                neural=neural,
                precise=precise,
                yaw=yaw,
            )
        previous = self.color_memory.get(self._color_key) if self._color_key is not None else None
        color_state: dict = {}
        out = paste_swapped_face(
            frame,
            swapped,
            matrix,
            align_kps,
            coverage=mode,
            yaw=yaw,
            alpha=alpha,
            previous_delta=previous,
            color_state=color_state,
        )
        if self._color_key is not None and "delta" in color_state:
            self.color_memory[self._color_key] = color_state["delta"]
        if (
            self._enhancer is not None
            and self.allow_restore
            and face_span_px(target_face.bbox) >= RESTORE_MIN_PX
        ):
            out = _enhance_face_region(
                out,
                _enhance_bbox(target_face.bbox, coverage),
                self._enhancer,
            )
        return out


class _FaceShim:
    """Duck-typed object compatible with insightface's INSwapper.get."""

    def __init__(self, face: Face) -> None:
        self.kps = face.kps
        self.bbox = face.bbox
        self.embedding = face.embedding
        self.normed_embedding = face.normed_embedding
        self.det_score = face.det_score


def gfpgan_install_tip() -> Optional[str]:
    """How to enable sharpening, or None when GFPGAN can be imported."""
    try:
        import gfpgan  # noqa: F401
    except Exception:
        return (
            "GFPGAN is optional and is not installed in this Python.\n\n"
            "From the Videoswa virtualenv run:\n"
            "pip install -r requirements-enhance.txt\n\n"
            "The first enhanced swap downloads GFPGANv1.4.pth into models/."
        )
    return None


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
    from .utils import ensure_gfpgan

    return ensure_gfpgan()


def _enhance_bbox(bbox: np.ndarray, coverage: str) -> np.ndarray:
    """Grow the enhancer window so a full-coverage beard is inside it."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if normalize_coverage(coverage) == "full":
        x1 -= 0.30 * width
        x2 += 0.30 * width
        y1 -= 0.20 * height
        y2 += 0.95 * height
    else:
        x1 -= 0.15 * width
        x2 += 0.15 * width
        y1 -= 0.15 * height
        y2 += 0.20 * height
    return np.array([x1, y1, x2, y2], dtype=np.float32)


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

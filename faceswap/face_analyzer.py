"""Face detection + embedding via InsightFace's buffalo_l pack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .providers import (
    CPU,
    active_providers_from_sessions,
    provider_attempts,
    run_with_cuda_fallback,
    run_with_provider_fallback,
    uses_gpu,
)
from .utils import logger


@dataclass
class Face:
    """A detected face with its identity embedding and geometry."""
    bbox: np.ndarray            # (4,) xyxy
    kps: np.ndarray             # (5, 2) landmarks
    embedding: np.ndarray       # (512,)
    det_score: float
    gender: Optional[int] = None
    age: Optional[int] = None

    @property
    def normed_embedding(self) -> np.ndarray:
        norm = np.linalg.norm(self.embedding) + 1e-8
        return self.embedding / norm

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    @property
    def gender_label(self) -> str:
        return format_gender(self.gender)


class FaceAnalyzer:
    """Wraps insightface.FaceAnalysis for detection + embedding extraction.

    ``execution`` selects the ONNX Runtime provider order. ``auto`` (the
    default) is TensorRT, then CUDA, then CPU. ``use_gpu=False`` forces CPU
    and overrides ``execution``.
    """

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        use_gpu: bool = True,
        det_thresh: float = 0.30,
        execution: str = "auto",
    ) -> None:
        self.det_size = det_size
        self.det_thresh = det_thresh
        self._load(use_gpu=use_gpu, execution=execution)

    def _load(self, use_gpu: bool, execution: str) -> None:
        from insightface.app import FaceAnalysis  # local import; heavy

        mode = "cpu" if not use_gpu else execution
        attempts = provider_attempts(mode)

        def _load(providers):
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(
                ctx_id=0 if uses_gpu(providers) else -1,
                det_size=self.det_size,
                det_thresh=self.det_thresh,
            )
            sessions = [
                getattr(model, "session", None)
                for model in getattr(app, "models", {}).values()
            ]
            active = active_providers_from_sessions(session for session in sessions if session)
            if uses_gpu(providers) and active == [CPU]:
                raise RuntimeError(
                    "ONNX Runtime fell back to CPU. TensorRT or CUDA libraries "
                    "are probably missing from PATH."
                )
            return app, active

        loaded, providers = run_with_provider_fallback(attempts, _load, what="Face detection (buffalo_l)")
        self.app, active = loaded
        self.providers = providers
        self.active_providers = active
        self._on_cpu = not uses_gpu(providers)
        logger.info("FaceAnalyzer active providers: %s", active or "(unreported)")

    def adopt_cpu(self) -> None:
        """Drop a GPU session that failed while the graph was running."""
        if self._on_cpu:
            return
        self._load(use_gpu=False, execution="cpu")

    def adopt_execution(self, execution: str) -> None:
        """Reload on DirectML or CPU without rebuilding from a fresh object."""
        if execution == "cpu":
            self.adopt_cpu()
            return
        self._load(use_gpu=True, execution=execution)

    def analyze(self, image_bgr: np.ndarray) -> List[Face]:
        """Detect every face, including side and profile views, left to right.

        Nothing is dropped for yaw. Profile detections often score under 0.5,
        so the detector threshold defaults to 0.30 and every returned face is
        kept for the face list and for swapping.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []
        return run_with_cuda_fallback(getattr(self, "cuda_guard", None), lambda: self._analyze_impl(image_bgr))

    def _analyze_impl(self, image_bgr: np.ndarray) -> List[Face]:
        raw = self.app.get(image_bgr)
        faces: List[Face] = []
        for f in raw:
            faces.append(
                Face(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32),
                    embedding=np.asarray(f.normed_embedding * np.linalg.norm(f.embedding), dtype=np.float32),
                    det_score=float(getattr(f, "det_score", 0.0)),
                    gender=coerce_gender(getattr(f, "gender", None)),
                    age=getattr(f, "age", None),
                )
            )
        faces.sort(key=lambda x: x.bbox[0])  # left to right
        return faces

    def best_face(self, image_bgr: np.ndarray) -> Optional[Face]:
        """Return the largest face — useful for source reference images."""
        faces = self.analyze(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)


def coerce_gender(value) -> Optional[int]:
    """InsightFace genderage: 0 is female, 1 is male. Anything else is unknown."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    if code in (0, 1):
        return code
    return None


def format_gender(gender) -> str:
    code = coerce_gender(gender)
    if code == 1:
        return "Male"
    if code == 0:
        return "Female"
    return "Unknown"


def face_label(index: int, gender) -> str:
    """UI caption such as ``Face 1 — Male``."""
    return f"Face {index + 1} — {format_gender(gender)}"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

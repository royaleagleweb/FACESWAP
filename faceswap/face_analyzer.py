"""Face detection + embedding via InsightFace's buffalo_l pack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
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
        self._cpu_app = None
        self._gpu_detector_blind = False
        self.last_detect_note = opencv_detection_warning() or ""
        if self.last_detect_note:
            logger.warning(self.last_detect_note)
        logger.info("FaceAnalyzer active providers: %s", active or "(unreported)")

    def set_detection(self, det_size: tuple[int, int] | int | None = None, det_thresh: float | None = None) -> None:
        """Re-prepare the detector. Size is snapped to a multiple of 32."""
        if det_size is not None:
            if isinstance(det_size, int):
                side = det_size
            else:
                side = int(det_size[0])
            side = max(32, int(round(side / 32.0) * 32))
            self.det_size = (side, side)
        if det_thresh is not None:
            self.det_thresh = float(np.clip(det_thresh, 0.05, 0.9))
        ctx = -1 if self._on_cpu else 0
        prepare = getattr(self.app, "prepare", None)
        if callable(prepare):
            prepare(ctx_id=ctx, det_size=self.det_size, det_thresh=self.det_thresh)
        cpu = getattr(self, "_cpu_app", None)
        if cpu is not None and callable(getattr(cpu, "prepare", None)):
            cpu.prepare(ctx_id=-1, det_size=self.det_size, det_thresh=self.det_thresh)

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

        DirectML and some TensorRT sessions return an empty list without
        raising. When that happens and a CPU pass finds faces, later frames
        stay on the CPU detector.
        """
        if image_bgr is None:
            return []
        return run_with_cuda_fallback(getattr(self, "cuda_guard", None), lambda: self._analyze_impl(image_bgr))

    def _analyze_impl(self, image_bgr: np.ndarray) -> List[Face]:
        image = prepare_detection_image(image_bgr)
        if image.size == 0:
            return []
        faces = self._faces_from_raw(self._raw_get(self._active_detector(), image))
        if faces or self._on_cpu or getattr(self, "_gpu_detector_blind", False):
            return faces
        cpu_faces = self._faces_from_raw(self._raw_get(self._cpu_detector(), image))
        if cpu_faces:
            self._gpu_detector_blind = True
            self.last_detect_note = (
                "GPU detection returned no faces. Videoswa switched this detector to CPU "
                "so DirectML and TensorRT sessions that drop scores still find people."
            )
            logger.warning(self.last_detect_note)
            return cpu_faces
        larger = self._retry_larger(image)
        if larger:
            return larger
        warning = opencv_detection_warning()
        if warning:
            self.last_detect_note = warning
        return []

    def _active_detector(self):
        if getattr(self, "_gpu_detector_blind", False) and getattr(self, "_cpu_app", None) is not None:
            return self._cpu_app
        return self.app

    def _cpu_detector(self):
        cpu = getattr(self, "_cpu_app", None)
        if cpu is not None:
            return cpu
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=self.det_size, det_thresh=self.det_thresh)
        self._cpu_app = app
        return app

    def _retry_larger(self, image: np.ndarray) -> List[Face]:
        """One pass at 960px when a large frame still has no face."""
        height, width = image.shape[:2]
        if max(height, width) < 720 or self.det_size[0] >= 960:
            return []
        app = self._cpu_detector()
        prepare = getattr(app, "prepare", None)
        if not callable(prepare):
            return []
        saved = self.det_size
        thresh = min(float(self.det_thresh), 0.25)
        try:
            prepare(ctx_id=-1, det_size=(960, 960), det_thresh=thresh)
            faces = self._faces_from_raw(self._raw_get(app, image))
        finally:
            prepare(ctx_id=-1, det_size=saved, det_thresh=self.det_thresh)
        if faces:
            self._gpu_detector_blind = True
            note = "A larger detector size was needed to find a face on this frame."
            self.last_detect_note = f"{self.last_detect_note} {note}".strip()
        return faces

    def _raw_get(self, app, image: np.ndarray):
        getter = getattr(app, "get", None)
        if not callable(getter):
            return []
        return getter(image) or []

    def _faces_from_raw(self, raw) -> List[Face]:
        faces: List[Face] = []
        for f in raw or []:
            embedding = getattr(f, "embedding", None)
            normed = getattr(f, "normed_embedding", None)
            if embedding is None and normed is not None:
                embedding = np.asarray(normed, dtype=np.float32)
            if embedding is None:
                continue
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            faces.append(
                Face(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32),
                    embedding=embedding,
                    det_score=float(getattr(f, "det_score", 0.0)),
                    gender=coerce_gender(getattr(f, "gender", None)),
                    age=getattr(f, "age", None),
                )
            )
        faces.sort(key=lambda x: x.bbox[0])
        return faces

    def best_face(self, image_bgr: np.ndarray) -> Optional[Face]:
        """Largest face in a source photo, after retries for size and padding.

        Phone photos are often huge, tiny, or not square. SCRFD stretches the
        whole image into ``det_size``, so a very wide frame hides the face.
        This pads to a square, scales into a useful range, then tries detector
        sizes 640, 320, and 1024 with a lower threshold.
        """
        image = prepare_source_image(image_bgr)
        if image.size == 0:
            self.last_detect_note = "The source image was empty."
            return None
        saved_size = self.det_size
        saved_thresh = float(self.det_thresh)
        sizes: list[int] = []
        for side in (int(saved_size[0]), 640, 320, 1024):
            if side > 0 and side not in sizes:
                sizes.append(side)
        try:
            for index, side in enumerate(sizes):
                thresh = saved_thresh if index == 0 else min(saved_thresh, 0.20)
                self.set_detection((side, side), thresh)
                faces = self.analyze(image)
                if faces:
                    self.last_detect_note = ""
                    return max(faces, key=lambda item: item.area)
            padded = pad_detection_image(image, border=64)
            self.set_detection((640, 640), min(saved_thresh, 0.15))
            faces = self.analyze(padded)
            if faces:
                self.last_detect_note = ""
                return max(faces, key=lambda item: item.area)
        finally:
            self.set_detection(saved_size, saved_thresh)
        tried = ", ".join(str(side) for side in sizes)
        self.last_detect_note = (
            "No face found in this photo. Videoswa applied rotation from the file, "
            f"tried detector sizes {tried}, then a padded 640px pass at a lower threshold. "
            "Use a clearer JPG or PNG of one person, with the face large enough to see."
        )
        return None

    def best_source_face(self, image_bgr: np.ndarray) -> Optional[Face]:
        return self.best_face(image_bgr)


def prepare_source_image(image: np.ndarray, max_side: int = 1600, min_side: int = 640) -> np.ndarray:
    """Square, uint8 BGR image in a range SCRFD can actually see."""
    image = prepare_detection_image(image)
    if image.size == 0:
        return image
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side > max_side:
        scale = max_side / float(long_side)
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    elif 0 < long_side < min_side:
        scale = min_side / float(long_side)
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    return pad_to_square(image)


def pad_to_square(image: np.ndarray, fill: int = 114) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    if side == height and side == width:
        return np.ascontiguousarray(image)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    canvas[y0:y0 + height, x0:x0 + width] = image
    return canvas


def pad_detection_image(image: np.ndarray, border: int = 64) -> np.ndarray:
    """Extra gray margin. A face against the edge survives the detector resize."""
    image = prepare_detection_image(image)
    if image.size == 0:
        return image
    return cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


def prepare_detection_image(image: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 BGR image.

    InsightFace's SCRFD returns an empty list on float 0–1, grayscale, or
    RGBA without raising. That looked like "no faces" on otherwise good frames.
    """
    if image is None:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("A detection frame must be a gray, BGR, or BGRA image.")
    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        peak = float(np.max(arr)) if arr.size else 0.0
        if peak <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def opencv_detection_warning() -> Optional[str]:
    """OpenCV 4.11 and 5.x break InsightFace detection. 4.10.0.84 is the pin."""
    parts = str(cv2.__version__).split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None
    if major > 4 or (major == 4 and minor >= 11):
        return (
            f"OpenCV {cv2.__version__} is installed. Face detection needs "
            "opencv-python==4.10.0.84. Newer OpenCV, including 5.x pulled in by "
            "GFPGAN, makes InsightFace return zero faces. "
            "Run: pip install opencv-python==4.10.0.84"
        )
    return None


def sensitivity_to_thresh(value: int) -> float:
    """Map the Detect sensitivity slider (0–100) to an SCRFD threshold.

    63 is about 0.30, which keeps side and profile faces. Higher sensitivity
    lowers the threshold.
    """
    return float(np.clip(0.55 - 0.40 * (float(value) / 100.0), 0.15, 0.60))


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

"""Swap requests shared by the desktop window and the worker thread."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from faceswap.core import DEFAULT_SIMILARITY, FaceMapping, FaceSwapEngine
from faceswap.coverage import DEFAULT_COVERAGE
from faceswap.face_analyzer import Face, coerce_gender
from faceswap.video import MAX_VIDEO_SECONDS, VIDEO_SUFFIXES, assert_duration_allowed

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif"}
MIN_ROI_CHANGE = 8.0
FACE_MODE_SINGLE = "single"
FACE_MODE_MULTIPLE = "multiple"


@dataclass
class FaceSource:
    """One detected person and the source image that should replace them."""

    face: Face
    source_path: Path
    label: str = ""
    gender: Optional[int] = None
    identity_paths: list[Path] = field(default_factory=list)


@dataclass
class SwapRequest:
    video_path: Path
    output_path: Path
    execution: str = "auto"
    enhance: bool = False
    similarity: float = DEFAULT_SIMILARITY
    coverage: str = DEFAULT_COVERAGE
    keep_audio: bool = True
    crf: int = 18
    preset: str = "medium"
    scale: float = 1.0
    face_mode: str = FACE_MODE_SINGLE
    single_source: Optional[Path] = None
    selected_face: Optional[Face] = None
    apply_to_all: bool = False
    face_sources: list[FaceSource] = field(default_factory=list)
    precise_edges: bool = False
    object_mask: bool = True
    fast_draft_preview: bool = True
    detect_every_other: bool = True
    min_face_px: int = 0
    rotation_mode: str = "per_person"
    rotation_seconds: float = 10.0
    rotation_paths: list[Path] = field(default_factory=list)
    identity_paths: list[Path] = field(default_factory=list)
    source_gender: Optional[int] = None
    match_gender: bool = True
    det_size: int = 640
    det_thresh: float = 0.30
    encoder: str = "auto"
    color_match: float = 0.60
    sharpen: float = 0.12
    restore_strength: float = 0.65


def read_image(path: Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        heic = _read_heic(path)
        if heic is not None:
            return heic
        raise ValueError(
            f"Could not read {path.name}. HEIC needs pillow-heif "
            "(pip install pillow-heif), or export the photo as JPG or PNG."
        )
    raise ValueError(f"Could not read image: {path}")


def _read_heic(path: Path) -> Optional[np.ndarray]:
    try:
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None
    return None


def roi_mean_change(original: np.ndarray, swapped: np.ndarray, boxes) -> float:
    """Largest mean absolute difference inside a swapped face box.

    A tiny change on the face means the preview still looks like the original,
    even when a swap was attempted.
    """
    if original is None or swapped is None or original.shape != swapped.shape:
        return 0.0
    height, width = original.shape[:2]
    best = 0.0
    found = False
    for box in boxes or []:
        x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(box).reshape(-1)[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        delta = np.abs(original[y1:y2, x1:x2].astype(np.int16) - swapped[y1:y2, x1:x2].astype(np.int16))
        best = max(best, float(np.mean(delta)))
        found = True
    if found:
        return best
    return float(np.mean(np.abs(original.astype(np.int16) - swapped.astype(np.int16))))


def validate_request(request: SwapRequest) -> None:
    """Reject a swap before models load when the inputs are not usable."""
    video = Path(request.video_path)
    if not video.exists():
        raise ValueError(f"Video not found: {video}")
    if video.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Choose a video file (mp4, mov, avi, mkv, webm, or m4v).")

    output = Path(request.output_path)
    if output.suffix.lower() != ".mp4":
        raise ValueError("The output file must be an .mp4.")

    assert_duration_allowed(video, limit_s=MAX_VIDEO_SECONDS)

    if request.face_mode == FACE_MODE_SINGLE:
        if request.single_source is None:
            raise ValueError("Choose one source image.")
        _require_image(Path(request.single_source))
        if not request.apply_to_all and request.selected_face is None:
            raise ValueError("Detect faces, then select the face to replace.")
        return

    if not request.face_sources:
        raise ValueError(
            "Detect faces, then choose a source image for at least one person. "
            "Leave a person without a source to keep their face unchanged."
        )
    for item in request.face_sources:
        _require_image(Path(item.source_path))


def _require_image(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Source image not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Source must be an image file: {path.name}")


def build_mappings(engine: FaceSwapEngine, request: SwapRequest) -> list[FaceMapping]:
    if request.face_mode == FACE_MODE_SINGLE:
        if request.single_source is None:
            raise ValueError("Choose one source image.")
        if request.apply_to_all:
            if request.identity_paths or coerce_gender(request.source_gender) is not None:
                source_face = _source_face(
                    engine,
                    Path(request.single_source),
                    request.identity_paths,
                    request.source_gender,
                )
                return [FaceMapping(source_face=source_face, reference_face=None, label="all")]
            image = read_image(Path(request.single_source))
            return [engine.build_mapping(image, reference_image_bgr=None, label="all")]
        if request.selected_face is None:
            raise ValueError("Detect faces, then select the face to replace.")
        source_face = _source_face(
            engine,
            Path(request.single_source),
            request.identity_paths,
            request.source_gender,
        )
        return [
            FaceMapping(
                source_face=source_face,
                reference_face=request.selected_face,
                label="selected",
            )
        ]

    mappings: list[FaceMapping] = []
    for item in request.face_sources:
        source_face = _source_face(
            engine,
            Path(item.source_path),
            item.identity_paths or request.identity_paths,
            item.gender if item.gender is not None else request.source_gender,
        )
        mappings.append(
            FaceMapping(
                source_face=source_face,
                reference_face=item.face,
                label=item.label or Path(item.source_path).name,
            )
        )
    if not mappings:
        raise ValueError("No source faces to apply.")
    return mappings


def _source_face(engine: FaceSwapEngine, path: Path, identity_paths: list[Path], gender: Optional[int]) -> Face:
    image = read_image(path)
    face = engine.analyzer.best_face(image)
    if face is None:
        raise ValueError(
            f"No face found in '{path.name}'. Use a clear photo of one person "
            "(JPG, PNG, or HEIC), or lower Min face size if the face is small."
        )
    embeddings = [np.asarray(face.normed_embedding, dtype=np.float32)]
    for extra in identity_paths or []:
        extra_image = read_image(Path(extra))
        extra_face = engine.analyzer.best_face(extra_image)
        if extra_face is None:
            raise ValueError(
                f"No face found in '{Path(extra).name}'. Add a clearer photo to this identity, "
                "or remove it."
            )
        embeddings.append(np.asarray(extra_face.normed_embedding, dtype=np.float32))
    if len(embeddings) > 1:
        mean = np.mean(np.stack(embeddings, axis=0), axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        face = replace(face, embedding=mean.astype(np.float32))
    code = coerce_gender(gender)
    if code is not None:
        face = replace(face, gender=code)
    return face

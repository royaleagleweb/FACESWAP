"""Swap requests shared by the desktop window and the worker thread."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from faceswap.core import DEFAULT_SIMILARITY, FaceMapping, FaceSwapEngine
from faceswap.coverage import DEFAULT_COVERAGE
from faceswap.face_analyzer import Face
from faceswap.video import MAX_VIDEO_SECONDS, VIDEO_SUFFIXES, assert_duration_allowed

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FACE_MODE_SINGLE = "single"
FACE_MODE_MULTIPLE = "multiple"


@dataclass
class FaceSource:
    """One detected person and the source image that should replace them."""

    face: Face
    source_path: Path
    label: str = ""


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
    rotation_seconds: float = 5.0
    rotation_paths: list[Path] = field(default_factory=list)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


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
            source = read_image(Path(request.single_source))
            return [engine.build_mapping(source, reference_image_bgr=None, label="all")]
        if request.selected_face is None:
            raise ValueError("Detect faces, then select the face to replace.")
        source_image = read_image(Path(request.single_source))
        source_face = engine.analyzer.best_face(source_image)
        if source_face is None:
            raise ValueError(
                f"No face found in '{Path(request.single_source).name}'. "
                "Use a photo with one clear face."
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
        source_image = read_image(Path(item.source_path))
        source_face = engine.analyzer.best_face(source_image)
        if source_face is None:
            raise ValueError(
                f"No face found in '{Path(item.source_path).name}'. "
                "Use a photo with one clear face."
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

"""High-level FaceSwapEngine: mapping multiple source faces onto target faces.

A FaceMapping pairs one *reference* face (cropped from a target image/video frame)
with a *source* face (cropped from a source image whose identity will be used
for the swap). At inference time, every face detected in the input video frame
is matched to the closest reference embedding; if it's close enough, the swap
fires using the corresponding source identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .face_analyzer import Face, FaceAnalyzer, cosine_similarity
from .swapper import FaceSwapper
from .utils import logger


@dataclass
class FaceMapping:
    """One source identity + the target face(s) it should replace.

    `reference_face` is detected in a snapshot from the *target* media — it
    tells the engine "this is the person to replace." `source_face` provides
    the identity to paste in.
    """
    source_face: Face
    reference_face: Optional[Face] = None  # None => apply to ALL detected faces
    label: str = ""

    def matches(self, candidate: Face, threshold: float) -> float:
        """Return similarity if reference present and above threshold, else -inf."""
        if self.reference_face is None:
            return 1.0  # wildcard match
        return cosine_similarity(self.reference_face.normed_embedding, candidate.normed_embedding)


@dataclass
class SwapStats:
    frames: int = 0
    faces_detected: int = 0
    faces_swapped: int = 0
    faces_unmatched: int = 0


@dataclass
class FaceSwapEngine:
    """Glue between detection, mapping, and the swap model."""

    analyzer: FaceAnalyzer
    swapper: FaceSwapper
    similarity_threshold: float = 0.45
    apply_to_all_when_no_reference: bool = True
    stats: SwapStats = field(default_factory=SwapStats)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        mappings: Sequence[FaceMapping],
    ) -> np.ndarray:
        """Detect faces in `frame_bgr` and apply each mapping that matches."""
        if not mappings:
            return frame_bgr

        faces = self.analyzer.analyze(frame_bgr)
        self.stats.frames += 1
        self.stats.faces_detected += len(faces)
        if not faces:
            return frame_bgr

        # Decide best mapping for each detected face. Wildcard mappings (no
        # reference) only apply when no specific reference matched.
        wildcard = [m for m in mappings if m.reference_face is None]
        specific = [m for m in mappings if m.reference_face is not None]

        out = frame_bgr
        for face in faces:
            best: Optional[FaceMapping] = None
            best_sim = -1.0
            for m in specific:
                sim = m.matches(face, self.similarity_threshold)
                if sim > best_sim:
                    best_sim = sim
                    best = m
            if best is not None and best_sim >= self.similarity_threshold:
                out = self.swapper.swap(out, target_face=face, source_face=best.source_face)
                self.stats.faces_swapped += 1
                continue

            if wildcard and (self.apply_to_all_when_no_reference or not specific):
                # If multiple wildcard mappings exist, use the first one — caller
                # is expected to provide just one wildcard.
                m = wildcard[0]
                out = self.swapper.swap(out, target_face=face, source_face=m.source_face)
                self.stats.faces_swapped += 1
            else:
                self.stats.faces_unmatched += 1
        return out

    # ----- helpers for building mappings -----

    def build_mapping(
        self,
        source_image_bgr: np.ndarray,
        reference_image_bgr: Optional[np.ndarray] = None,
        label: str = "",
    ) -> FaceMapping:
        src = self.analyzer.best_face(source_image_bgr)
        if src is None:
            raise ValueError(f"No face found in source image '{label or '<unnamed>'}'")
        ref = None
        if reference_image_bgr is not None:
            ref = self.analyzer.best_face(reference_image_bgr)
            if ref is None:
                raise ValueError(
                    f"No face found in reference image for mapping '{label or '<unnamed>'}'"
                )
        return FaceMapping(source_face=src, reference_face=ref, label=label)

    def build_mappings(
        self,
        pairs: Iterable[tuple[np.ndarray, Optional[np.ndarray]]],
    ) -> List[FaceMapping]:
        mappings: List[FaceMapping] = []
        for i, (src_img, ref_img) in enumerate(pairs):
            mappings.append(self.build_mapping(src_img, ref_img, label=f"map_{i}"))
        logger.info(
            "Built %d mappings (%d with reference, %d wildcard)",
            len(mappings),
            sum(1 for m in mappings if m.reference_face is not None),
            sum(1 for m in mappings if m.reference_face is None),
        )
        return mappings

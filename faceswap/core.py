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

from .coverage import DEFAULT_COVERAGE
from .face_analyzer import Face, FaceAnalyzer, coerce_gender, cosine_similarity
from .providers import CudaRuntimeGuard
from .pose import PROFILE_YAW, face_yaw
from .quality import SourceRotation, face_span_px, lock_gender
from .swapper import FaceSwapper
from .utils import logger

# First-lock cosine similarity. Pose changes often fall under 0.45, which
# used to drop the swap and show the original face again. 0.32 still rejects
# a clearly different person; raise it when the wrong face is selected.
DEFAULT_SIMILARITY = 0.32
# Once a face is locked, keep it down to this floor when the box still overlaps.
_HOLD_FLOOR = 0.22
_HOLD_MARGIN = 0.15
# Overlap required to treat a detection as the same person as last frame.
_TRACK_IOU = 0.30
_TRACK_MISS_LIMIT = 6
# Similarity to the running embedding. Independent of the original reference,
# so a turned head stays locked without accepting an orthogonal stranger.
_TRACK_EMBED_HOLD = 0.55
# A turned head drops both scores. Keep the lock when the landmarks are a profile.
_PROFILE_LOCK = 0.22
_PROFILE_REF_HOLD = 0.16
_PROFILE_EMBED_HOLD = 0.34
_SWITCH_MARGIN = 0.08
_EMA_KEEP = 0.65


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
    faces_held: int = 0
    faces_skipped: int = 0


@dataclass
class _FaceTrack:
    """A person locked to one source across frames."""

    mapping: FaceMapping
    bbox: np.ndarray
    embedding: np.ndarray
    missed: int = 0
    gender_votes: list = field(default_factory=list)
    locked_gender: Optional[int] = None


@dataclass
class FaceSwapEngine:
    """Glue between detection, mapping, and the swap model."""

    analyzer: FaceAnalyzer
    swapper: FaceSwapper
    similarity_threshold: float = DEFAULT_SIMILARITY
    coverage: str = DEFAULT_COVERAGE
    apply_to_all_when_no_reference: bool = True
    stats: SwapStats = field(default_factory=SwapStats)
    detect_stride: int = 1
    min_face_px: int = 0

    def __post_init__(self) -> None:
        self.cuda_guard = CudaRuntimeGuard()
        self._tracks: List[_FaceTrack] = []
        self._cached_faces: List[Face] = []
        self._detect_tick = 0
        self._from_detector = True
        self.rotation = SourceRotation()
        self.match_gender = False
        self.last_boxes: List[np.ndarray] = []
        for member in (self.analyzer, self.swapper):
            if hasattr(member, "adopt_cpu"):
                self.cuda_guard.attach(member)

    def _gender_blocks(self, source: Face, target: Face) -> bool:
        """Block only when Match gender is on and both genders are known and different."""
        if not self.match_gender:
            return False
        source_gender = coerce_gender(source.gender)
        target_gender = coerce_gender(target.gender)
        if source_gender is None or target_gender is None:
            return False
        return source_gender != target_gender

    def _lock_threshold(self, face: Face) -> float:
        """First-lock cosine. Profile faces clear a lower bar than frontal ones."""
        if abs(face_yaw(face.kps)) >= PROFILE_YAW:
            return min(float(self.similarity_threshold), _PROFILE_LOCK)
        return float(self.similarity_threshold)

    def _hold_limits(self, face: Face) -> tuple[float, float]:
        """Reference floor and running-embedding floor for a face already locked."""
        if abs(face_yaw(face.kps)) >= PROFILE_YAW:
            return min(self.hold_similarity(), _PROFILE_REF_HOLD), _PROFILE_EMBED_HOLD
        return self.hold_similarity(), _TRACK_EMBED_HOLD

    def hold_similarity(self) -> float:
        """Similarity that keeps an already locked person from reverting."""
        return max(_HOLD_FLOOR, float(self.similarity_threshold) - _HOLD_MARGIN)

    def reset_tracks(self) -> None:
        """Drop cross-frame locks. Call this at the start of each video."""
        self._tracks = []
        self._cached_faces = []
        self._detect_tick = 0
        self.rotation.reset()
        reset_color = getattr(self.swapper, "reset_color_memory", None)
        if callable(reset_color):
            reset_color()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        mappings: Sequence[FaceMapping],
        time_s: float = 0.0,
    ) -> np.ndarray:
        """Detect faces in `frame_bgr` and apply each mapping that matches.

        A face that cleared the threshold stays on that source while its box
        still overlaps and the embedding is still the same person. A stranger
        in that box is left as the original.
        """
        self.last_boxes = []
        if not mappings:
            return frame_bgr

        faces = self._faces_for_frame(frame_bgr)
        self.stats.frames += 1
        self.stats.faces_detected += len(faces)
        if self.min_face_px > 0:
            kept = []
            for face in faces:
                if face_span_px(face.bbox) < self.min_face_px:
                    self.stats.faces_skipped += 1
                    continue
                kept.append(face)
            faces = kept
        if not faces:
            self._age_tracks(set())
            return frame_bgr

        wildcard = [m for m in mappings if m.reference_face is None]
        specific = [m for m in mappings if m.reference_face is not None]
        decisions = self._assign(faces, specific, wildcard)
        touched: set[int] = set()

        out = frame_bgr
        for face in faces:
            decision = decisions.get(id(face))
            if decision is None:
                self.stats.faces_unmatched += 1
                continue
            mapping, held = decision
            track = self._touch_track(face, mapping)
            touched.add(id(track))
            source = self.rotation.choose(mapping.source_face, face, time_s)
            if self._gender_blocks(source, face):
                self.stats.faces_unmatched += 1
                continue
            begin = getattr(self.swapper, "begin_face", None)
            if callable(begin):
                begin(id(track))
            out = self.swapper.swap(
                out, target_face=face, source_face=source, coverage=self.coverage
            )
            self.last_boxes.append(np.asarray(face.bbox, dtype=np.float32).copy())
            self.stats.faces_swapped += 1
            if held:
                self.stats.faces_held += 1
        self._age_tracks(touched)
        return out

    def _faces_for_frame(self, frame_bgr: np.ndarray) -> List[Face]:
        """Detect on every Nth frame and reuse the last landmarks between them."""
        self._detect_tick += 1
        stride = max(1, int(self.detect_stride))
        use_cache = stride > 1 and self._cached_faces and (self._detect_tick % stride != 1)
        if use_cache:
            self._from_detector = False
            return list(self._cached_faces)
        self._from_detector = True
        faces = self.analyzer.analyze(frame_bgr)
        self._cached_faces = list(faces)
        return faces

    def _assign(
        self,
        faces: Sequence[Face],
        specific: Sequence[FaceMapping],
        wildcard: Sequence[FaceMapping],
    ) -> dict[int, tuple[FaceMapping, bool]]:
        """Map each face to ``(mapping, held_below_threshold)``."""
        decisions: dict[int, tuple[FaceMapping, bool]] = {}
        claimed: set[int] = set()
        ranked = {id(face): self._rank_specific(face, specific) for face in faces}

        fresh: list[tuple[float, float, Face, FaceMapping]] = []
        for face in faces:
            scores = ranked[id(face)]
            if not scores or scores[0][0] < self._lock_threshold(face):
                continue
            fresh.append((scores[0][0], face.area, face, scores[0][1]))
        fresh.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _sim, _area, face, mapping in fresh:
            chosen = self._stable_choice(face, mapping, ranked[id(face)], claimed)
            if chosen is None:
                continue
            decisions[id(face)] = (chosen, False)
            if chosen.reference_face is not None:
                claimed.add(id(chosen))

        for face in sorted(faces, key=lambda item: item.area, reverse=True):
            if id(face) in decisions:
                continue
            held = self._hold_choice(face, claimed)
            if held is None:
                continue
            decisions[id(face)] = (held, True)
            if held.reference_face is not None:
                claimed.add(id(held))

        if wildcard and (self.apply_to_all_when_no_reference or not specific):
            mapping = wildcard[0]
            for face in faces:
                if id(face) not in decisions:
                    decisions[id(face)] = (mapping, False)
        return decisions

    def _rank_specific(
        self, face: Face, specific: Sequence[FaceMapping]
    ) -> list[tuple[float, FaceMapping]]:
        scored = [
            (cosine_similarity(mapping.reference_face.normed_embedding, face.normed_embedding), mapping)
            for mapping in specific
            if mapping.reference_face is not None
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _stable_choice(
        self,
        face: Face,
        best: FaceMapping,
        scores: Sequence[tuple[float, FaceMapping]],
        claimed: set[int],
    ) -> Optional[FaceMapping]:
        """Keep the locked source unless another person is clearly a better match."""
        best_sim = scores[0][0] if scores else -1.0
        track = self._best_track(face, claimed)
        if (
            track is not None
            and track.mapping.reference_face is not None
            and id(track.mapping) not in claimed
        ):
            track_sim = cosine_similarity(
                track.mapping.reference_face.normed_embedding, face.normed_embedding
            )
            if track_sim >= self.hold_similarity() and best_sim - track_sim < _SWITCH_MARGIN:
                return track.mapping
        if id(best) in claimed:
            for sim, mapping in scores:
                if sim < self._lock_threshold(face):
                    break
                if id(mapping) not in claimed:
                    return mapping
            return None
        return best

    def _hold_choice(self, face: Face, claimed: set[int]) -> Optional[FaceMapping]:
        track = self._best_track(face, claimed)
        if track is None:
            return None
        mapping = track.mapping
        if mapping.reference_face is not None and id(mapping) in claimed:
            return None
        emb_sim = cosine_similarity(track.embedding, face.normed_embedding)
        ref_need, emb_need = self._hold_limits(face)
        if mapping.reference_face is None:
            return mapping if emb_sim >= emb_need else None
        ref_sim = cosine_similarity(mapping.reference_face.normed_embedding, face.normed_embedding)
        # Pose change: the reference score dips, but the running embedding still
        # matches. A profile lowers both floors. An orthogonal stranger still fails.
        if ref_sim >= ref_need or emb_sim >= emb_need:
            return mapping
        return None

    def _best_track(self, face: Face, claimed: set[int]) -> Optional[_FaceTrack]:
        best: Optional[_FaceTrack] = None
        best_iou = _TRACK_IOU
        for track in self._tracks:
            if track.mapping.reference_face is not None and id(track.mapping) in claimed:
                continue
            overlap = _bbox_iou(track.bbox, face.bbox)
            if overlap >= best_iou:
                best_iou = overlap
                best = track
        return best

    def _touch_track(self, face: Face, mapping: FaceMapping) -> _FaceTrack:
        best: Optional[_FaceTrack] = None
        best_iou = _TRACK_IOU
        for track in self._tracks:
            if track.mapping is not mapping:
                continue
            overlap = _bbox_iou(track.bbox, face.bbox)
            if overlap >= best_iou:
                best_iou = overlap
                best = track
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        if best is None:
            best = _FaceTrack(
                mapping=mapping,
                bbox=np.asarray(face.bbox, dtype=np.float32).copy(),
                embedding=embedding.copy(),
            )
            self._tracks.append(best)
        else:
            best.bbox = np.asarray(face.bbox, dtype=np.float32).copy()
            mixed = _EMA_KEEP * best.embedding + (1.0 - _EMA_KEEP) * embedding
            best.embedding = mixed / (np.linalg.norm(mixed) + 1e-8)
            best.missed = 0
        self._lock_track_gender(best, face)
        return best

    def _lock_track_gender(self, track: _FaceTrack, face: Face) -> None:
        """Majority gender from the first detections, then keep it."""
        if track.locked_gender is not None:
            face.gender = track.locked_gender
            return
        if not self._from_detector or face.gender not in (0, 1):
            return
        track.gender_votes.append(int(face.gender))
        locked = lock_gender(track.gender_votes)
        if locked is None:
            return
        track.locked_gender = locked
        face.gender = locked

    def _age_tracks(self, touched: set[int]) -> None:
        for track in self._tracks:
            if id(track) not in touched:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= _TRACK_MISS_LIMIT]

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


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in np.asarray(a).reshape(-1)[:4])
    bx1, by1, bx2, by2 = (float(v) for v in np.asarray(b).reshape(-1)[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return inter / union

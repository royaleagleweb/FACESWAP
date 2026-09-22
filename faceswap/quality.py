"""Export versus preview quality, and which source is active this frame."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .face_analyzer import Face, cosine_similarity

RESTORE_MIN_PX = 96
GENDER_LOCK_VOTES = 5
ROTATION_PER_PERSON = "per_person"
ROTATION_SCENE = "scene"
ROTATION_INTERVAL = "interval"
ROTATION_MODES = (ROTATION_PER_PERSON, ROTATION_SCENE, ROTATION_INTERVAL)
_SCENE_BREAK = 0.35


def resolve_quality(
    *,
    enhance: bool,
    precise_edges: bool,
    object_mask: bool,
    fast_draft: bool,
    preview: bool,
) -> tuple[bool, bool, bool]:
    """Return restore, precise edges, and object mask for this pass.

    Fast draft (default during preview) skips GFPGAN and BiSeNet. The object
    mask stays on so hands and food are still preserved while tuning.
    """
    if preview and fast_draft:
        return False, False, True
    return bool(enhance), bool(precise_edges), bool(object_mask)


def face_span_px(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox).reshape(-1)[:4])
    return max(0.0, x2 - x1, y2 - y1)


def should_restore(bbox: np.ndarray, enabled: bool, minimum: float = RESTORE_MIN_PX) -> bool:
    """GFPGAN only runs on faces that are large enough to restore."""
    return bool(enabled) and face_span_px(bbox) >= minimum


def smooth_delta(previous: Optional[np.ndarray], current: np.ndarray, keep: float = 0.65) -> np.ndarray:
    """Ease a per-person color correction so the rim does not shimmer."""
    current = np.asarray(current, dtype=np.float32)
    if previous is None:
        return current
    return (keep * np.asarray(previous, dtype=np.float32) + (1.0 - keep) * current).astype(np.float32)


class SourceRotation:
    """Pick the source identity for scene cuts or a fixed interval."""

    def __init__(self) -> None:
        self.mode = ROTATION_PER_PERSON
        self.seconds = 5.0
        self.sources: list[Face] = []
        self.index = 0
        self.anchor: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.index = 0
        self.anchor = None

    def choose(self, fallback: Face, candidate: Face, time_s: float) -> Face:
        if self.mode == ROTATION_PER_PERSON or len(self.sources) <= 1:
            return fallback
        if self.mode == ROTATION_INTERVAL:
            step = max(0.1, float(self.seconds))
            self.index = int(max(0.0, time_s) / step) % len(self.sources)
            return self.sources[self.index]
        embedding = np.asarray(candidate.normed_embedding, dtype=np.float32)
        if self.anchor is None:
            self.anchor = embedding
            return self.sources[self.index % len(self.sources)]
        if cosine_similarity(self.anchor, embedding) < _SCENE_BREAK:
            self.index = (self.index + 1) % len(self.sources)
            self.anchor = embedding
        return self.sources[self.index % len(self.sources)]


def lock_gender(votes: Sequence[int]) -> Optional[int]:
    """Majority of the first samples. 0 female, 1 male."""
    usable = [int(v) for v in votes if int(v) in (0, 1)]
    if len(usable) < GENDER_LOCK_VOTES:
        return None
    sample = usable[:GENDER_LOCK_VOTES]
    males = sum(1 for value in sample if value == 1)
    return 1 if males >= (GENDER_LOCK_VOTES - males) else 0

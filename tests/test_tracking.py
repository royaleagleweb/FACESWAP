"""A locked face keeps its source when the next frame is a weaker match."""

import numpy as np

from faceswap.core import DEFAULT_SIMILARITY, FaceMapping, FaceSwapEngine
from faceswap.face_analyzer import Face, cosine_similarity
from faceswap.video import reset_stats


def _face(embedding: list[float], x: float, width: float = 40.0) -> Face:
    vector = np.array(embedding, dtype=np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    return Face(
        bbox=np.array([x, 10, x + width, 10 + width * 1.3], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        embedding=vector,
        det_score=0.99,
    )


class _Analyzer:
    def __init__(self) -> None:
        self.faces: list[Face] = []

    def analyze(self, _frame):
        return list(self.faces)


class _Swapper:
    def __init__(self) -> None:
        self.sources: list[Face] = []

    def swap(self, frame, target_face, source_face, paste_back=True, coverage="full"):
        self.sources.append(source_face)
        return frame


def _engine(threshold: float = 0.45) -> tuple[FaceSwapEngine, _Analyzer, _Swapper]:
    analyzer = _Analyzer()
    swapper = _Swapper()
    engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper, similarity_threshold=threshold)
    return engine, analyzer, swapper


def test_default_similarity_locks_sooner_than_the_old_cutoff() -> None:
    assert DEFAULT_SIMILARITY == 0.32
    assert DEFAULT_SIMILARITY < 0.45


def test_pose_change_keeps_the_same_source() -> None:
    reference = _face([1, 0, 0], 20)
    source = _face([0, 1, 0], 0)
    first = _face([1, 0, 0], 22)
    # Under the first-lock threshold, which used to drop the frame back to the
    # original, but still close enough to keep a face that has not moved away.
    turned = _face([0.30, 0.9539, 0.0], 28)
    sim = cosine_similarity(reference.normed_embedding, turned.normed_embedding)
    assert sim < 0.45
    assert sim >= 0.29

    engine, analyzer, swapper = _engine(0.45)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    mapping = FaceMapping(source_face=source, reference_face=reference)
    analyzer.faces = [first]
    engine.process_frame(frame, [mapping])
    analyzer.faces = [turned]
    engine.process_frame(frame, [mapping])

    assert swapper.sources == [source, source]
    assert engine.stats.faces_swapped == 2
    assert engine.stats.faces_held == 1
    assert engine.stats.faces_unmatched == 0


def test_running_embedding_holds_below_the_reference_floor() -> None:
    """The lock follows the person even after the original snapshot score drops."""
    reference = _face([1, 0, 0], 20)
    source = _face([0, 1, 0], 0)
    # ~0.62 to the reference: enough to lock, and a base the next frame can stay near.
    locked = _face([0.62, 0.78, 0.08], 22)
    # ~0.25 to the reference (under the 0.30 hold floor) but still this person.
    followed = _face([0.25, 0.78, 0.57], 26)
    assert cosine_similarity(reference.normed_embedding, locked.normed_embedding) >= 0.45
    assert cosine_similarity(reference.normed_embedding, followed.normed_embedding) < 0.30
    assert cosine_similarity(locked.normed_embedding, followed.normed_embedding) >= 0.55

    engine, analyzer, swapper = _engine(0.45)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    mapping = FaceMapping(source_face=source, reference_face=reference)
    analyzer.faces = [locked]
    engine.process_frame(frame, [mapping])
    analyzer.faces = [followed]
    engine.process_frame(frame, [mapping])
    assert swapper.sources == [source, source]
    assert engine.stats.faces_held == 1


def test_stranger_in_the_same_box_is_not_swapped() -> None:
    reference = _face([1, 0, 0], 20)
    source = _face([0, 1, 0], 0)
    engine, analyzer, swapper = _engine(0.45)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    mapping = FaceMapping(source_face=source, reference_face=reference)
    analyzer.faces = [_face([1, 0, 0], 22)]
    engine.process_frame(frame, [mapping])
    stranger = _face([0, 0, 1], 24)
    assert cosine_similarity(reference.normed_embedding, stranger.normed_embedding) < 0.1
    analyzer.faces = [stranger]
    engine.process_frame(frame, [mapping])

    assert swapper.sources == [source]
    assert engine.stats.faces_swapped == 1
    assert engine.stats.faces_unmatched == 1


def test_weak_match_without_overlap_does_not_hold() -> None:
    reference = _face([1, 0, 0], 20)
    source = _face([0, 1, 0], 0)
    engine, analyzer, swapper = _engine(0.45)
    frame = np.zeros((80, 200, 3), dtype=np.uint8)
    mapping = FaceMapping(source_face=source, reference_face=reference)
    analyzer.faces = [_face([1, 0, 0], 10)]
    engine.process_frame(frame, [mapping])
    far = _face([0.30, 0.95, 0.05], 140)
    analyzer.faces = [far]
    engine.process_frame(frame, [mapping])
    assert len(swapper.sources) == 1
    assert engine.stats.faces_unmatched == 1


def test_reset_tracks_requires_a_fresh_lock() -> None:
    reference = _face([1, 0, 0], 20)
    source = _face([0, 1, 0], 0)
    engine, analyzer, swapper = _engine(0.45)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    mapping = FaceMapping(source_face=source, reference_face=reference)
    analyzer.faces = [_face([1, 0, 0], 22)]
    engine.process_frame(frame, [mapping])
    reset_stats(engine)
    analyzer.faces = [_face([0.30, 0.95, 0.05], 24)]
    engine.process_frame(frame, [mapping])
    assert engine.stats.faces_swapped == 0
    assert engine.stats.faces_unmatched == 1
    assert len(swapper.sources) == 1

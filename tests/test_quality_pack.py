"""Object mask, gender lock, preview quality, and source rotation."""

from __future__ import annotations

import inspect

import numpy as np

from faceswap.core import FaceMapping, FaceSwapEngine
from faceswap.coverage import face_axes, template_128
from faceswap.face_analyzer import Face
from faceswap.occlusion import _occ_mask
from faceswap.quality import (
    RESTORE_MIN_PX,
    SourceRotation,
    lock_gender,
    resolve_quality,
    should_restore,
    smooth_delta,
)
from faceswap.swapper import FaceSwapper
from videoswa.project import PROJECT_SUFFIX, gender_mark, load_project, save_project


def _landmarks() -> np.ndarray:
    dst = template_128()
    scale = 2.2
    center = dst.mean(axis=0)
    return (dst - center) * scale + np.array([200.0, 190.0])


def _face(embedding: list[float], x: float, width: float = 40.0, gender: int | None = None) -> Face:
    vector = np.array(embedding, dtype=np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    return Face(
        bbox=np.array([x, 10, x + width, 10 + width * 1.3], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        embedding=vector,
        det_score=0.99,
        gender=gender,
    )


class _Analyzer:
    def __init__(self) -> None:
        self.faces: list[Face] = []
        self.calls = 0

    def analyze(self, _frame):
        self.calls += 1
        return list(self.faces)


class _Swapper:
    def __init__(self) -> None:
        self.sources: list[Face] = []
        self.keys: list[int] = []

    def begin_face(self, key: int) -> None:
        self.keys.append(key)

    def swap(self, frame, target_face, source_face, paste_back=True, coverage="full"):
        self.sources.append(source_face)
        return frame


def test_occ_mask_keeps_a_lollipop_and_a_beard() -> None:
    kps = _landmarks()
    frame = np.full((480, 460, 3), (140, 160, 180), dtype=np.uint8)
    axes = face_axes(kps)
    assert axes is not None
    _eye, mouth, down_u, side_u, em, eye_dist = axes
    beard = mouth + down_u * (0.9 * em)
    y0 = int(beard[1])
    x0 = int(beard[0] - 0.9 * eye_dist)
    x1 = int(beard[0] + 0.9 * eye_dist)
    frame[y0:y0 + 28, x0:x1] = (30, 40, 50)
    cheek = mouth - down_u * (0.55 * em) + side_u * (0.35 * eye_dist)
    cy, cx = int(cheek[1]), int(cheek[0])
    frame[cy:cy + 14, cx:cx + 14] = (255, 40, 40)

    weight = _occ_mask(frame, kps, "full")
    assert weight[0, 0] == 0.0
    assert weight[cy + 6, cx + 6] < 0.05
    assert weight[y0 + 10, int(beard[0])] > 0.9


def test_occ_mask_leaves_a_uniform_face_intact() -> None:
    kps = _landmarks()
    frame = np.full((480, 460, 3), (90, 140, 110), dtype=np.uint8)
    weight = _occ_mask(frame, kps, "full")
    assert weight[0, 0] == 0.0
    inside = weight > 0.95
    assert int(np.count_nonzero(inside)) > 500


def test_neural_keepout_stays_inside_the_face() -> None:
    kps = _landmarks()
    frame = np.full((480, 460, 3), (140, 160, 180), dtype=np.uint8)
    neural = np.ones(frame.shape[:2], dtype=np.float32)
    weight = _occ_mask(frame, kps, "full", neural=neural)
    assert weight.max() == 0.0


def test_swap_calls_occ_mask_for_a_normal_swap() -> None:
    source = inspect.getsource(FaceSwapper.swap)
    assert "_occ_mask(" in source
    assert "if self.object_mask:" in source
    assert "models.xseg is not None" in source
    init = inspect.getsource(FaceSwapper.__init__)
    assert "self.object_mask = True" in init
    assert "self.precise_edges = False" in init


def test_gender_locks_after_five_detections() -> None:
    analyzer = _Analyzer()
    swapper = _Swapper()
    engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper)
    source = _face([0, 1, 0], 0)
    mapping = FaceMapping(source_face=source, reference_face=None)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    male = _face([1, 0, 0], 20, gender=1)
    for _ in range(5):
        analyzer.faces = [male]
        engine.process_frame(frame, [mapping])
    assert engine._tracks[0].locked_gender == 1
    female = _face([1, 0, 0], 20, gender=0)
    analyzer.faces = [female]
    engine.process_frame(frame, [mapping])
    assert engine._tracks[0].locked_gender == 1
    assert female.gender == 1
    assert lock_gender([1, 1, 1, 0, 0]) == 1


def test_cached_frames_do_not_vote_on_gender() -> None:
    analyzer = _Analyzer()
    engine = FaceSwapEngine(analyzer=analyzer, swapper=_Swapper(), detect_stride=2)
    mapping = FaceMapping(source_face=_face([0, 1, 0], 0), reference_face=None)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    person = _face([1, 0, 0], 20, gender=1)
    analyzer.faces = [person]
    for _ in range(4):
        engine.process_frame(frame, [mapping])
    assert analyzer.calls == 2
    assert len(engine._tracks[0].gender_votes) == 2
    assert engine._tracks[0].locked_gender is None


def test_small_faces_are_skipped() -> None:
    analyzer = _Analyzer()
    swapper = _Swapper()
    engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper, min_face_px=80)
    analyzer.faces = [_face([1, 0, 0], 20, width=30)]
    mapping = FaceMapping(source_face=_face([0, 1, 0], 0), reference_face=None)
    engine.process_frame(np.zeros((80, 120, 3), dtype=np.uint8), [mapping])
    assert engine.stats.faces_skipped == 1
    assert engine.stats.faces_swapped == 0
    assert swapper.sources == []


def test_interval_and_scene_rotation() -> None:
    first = _face([1, 0, 0], 0)
    second = _face([0, 1, 0], 40)
    rotation = SourceRotation()
    rotation.mode = "interval"
    rotation.seconds = 5
    rotation.sources = [first, second]
    assert rotation.choose(first, first, 0.0) is first
    assert rotation.choose(first, first, 5.0) is second

    rotation = SourceRotation()
    rotation.mode = "scene"
    rotation.sources = [first, second]
    assert rotation.choose(first, first, 0.0) is first
    assert rotation.choose(first, second, 1.0) is second


def test_fast_draft_keeps_the_object_mask() -> None:
    assert resolve_quality(
        enhance=True,
        precise_edges=True,
        object_mask=False,
        fast_draft=True,
        preview=True,
    ) == (False, False, True)
    assert resolve_quality(
        enhance=True,
        precise_edges=True,
        object_mask=True,
        fast_draft=True,
        preview=False,
    ) == (True, True, True)
    assert should_restore(np.array([0, 0, 95, 10]), True) is False
    assert should_restore(np.array([0, 0, RESTORE_MIN_PX, 10]), True) is True
    eased = smooth_delta(np.zeros(3, dtype=np.float32), np.array([10, 0, 0], dtype=np.float32))
    assert eased[0] < 10


def test_project_roundtrip_keeps_gender_marks(tmp_path) -> None:
    assert gender_mark(1) == "♂"
    assert gender_mark(0) == "♀"
    saved = save_project(
        tmp_path / "clip",
        {"sources": [{"path": "face.png", "gender": "♂", "label": "Face 1 — Male"}]},
    )
    assert saved.suffix == PROJECT_SUFFIX
    loaded = load_project(saved)
    assert loaded["version"] == 1
    assert loaded["sources"][0]["gender"] == "♂"

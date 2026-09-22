"""Detection must return a face when the GPU session is blind, and OpenCV stays pinned."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from faceswap.core import FaceMapping, FaceSwapEngine
from faceswap.face_analyzer import (
    Face,
    FaceAnalyzer,
    prepare_detection_image,
    sensitivity_to_thresh,
)
from videoswa.jobs import MIN_ROI_CHANGE, roi_mean_change
from videoswa.worker import seek_times

REPO = Path(__file__).resolve().parents[1]


class _RawFace:
    def __init__(self) -> None:
        self.bbox = np.array([10, 10, 80, 100], dtype=np.float32)
        self.kps = np.array([[20, 30], [60, 30], [40, 50], [25, 70], [55, 70]], dtype=np.float32)
        self.embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.normed_embedding = self.embedding
        self.det_score = 0.42
        self.gender = 1
        self.age = 30


class _App:
    def __init__(self, hits: bool) -> None:
        self.hits = hits
        self.seen = []

    def get(self, image):
        self.seen.append(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            return []
        if not self.hits:
            return []
        return [_RawFace()]


def _analyzer(gpu_hits: bool, cpu_hits: bool) -> FaceAnalyzer:
    analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
    analyzer.det_size = (640, 640)
    analyzer.det_thresh = 0.30
    analyzer._on_cpu = False
    analyzer._gpu_detector_blind = False
    analyzer._cpu_app = None
    analyzer.last_detect_note = ""
    analyzer.app = _App(gpu_hits)
    analyzer._cpu_detector = lambda: _install_cpu(analyzer, cpu_hits)
    return analyzer


def _install_cpu(analyzer: FaceAnalyzer, hits: bool):
    if analyzer._cpu_app is None:
        analyzer._cpu_app = _App(hits)
    return analyzer._cpu_app


def _synthetic(dtype=np.uint8) -> np.ndarray:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[20:80, 30:70] = (40, 80, 160)
    if dtype == np.uint8:
        return image
    return (image.astype(np.float32) / 255.0)


def test_requirements_pin_opencv_so_gfpgan_cannot_upgrade_it() -> None:
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    enhance = (REPO / "requirements-enhance.txt").read_text(encoding="utf-8")
    assert "opencv-python==4.10.0.84" in requirements
    assert "opencv-python-headless" not in requirements
    assert "opencv-python==4.10.0.84" in enhance
    assert "gfpgan==1.3.8" in enhance


def test_float_and_rgba_frames_still_detect_after_cpu_retry() -> None:
    analyzer = _analyzer(gpu_hits=False, cpu_hits=True)
    rgba = np.dstack([_synthetic(), np.full((96, 96), 255, dtype=np.uint8)])
    for image in (_synthetic(), _synthetic(np.float32), rgba):
        faces = analyzer.analyze(image)
        assert len(faces) >= 1
    assert analyzer._gpu_detector_blind is True
    assert "CPU" in analyzer.last_detect_note
    assert all(frame.dtype == np.uint8 and frame.shape[2] == 3 for frame in analyzer._cpu_app.seen)


def test_prepare_detection_image_makes_contiguous_uint8_bgr() -> None:
    prepared = prepare_detection_image(_synthetic(np.float32))
    assert prepared.dtype == np.uint8
    assert prepared.flags["C_CONTIGUOUS"]
    assert prepared.shape[2] == 3
    assert sensitivity_to_thresh(63) == np.float32(0.298) or abs(sensitivity_to_thresh(63) - 0.30) < 0.01


def test_both_detectors_empty_does_not_mark_the_gpu_blind() -> None:
    analyzer = _analyzer(gpu_hits=False, cpu_hits=False)
    assert analyzer.analyze(_synthetic()) == []
    assert analyzer._gpu_detector_blind is False


def test_seek_samples_nearby_timestamps_and_stops_at_eight() -> None:
    times = seek_times(10.0, 30.0)
    assert times[0] == 10.0
    assert 10.4 in times and 9.6 in times
    assert len(times) <= 8
    assert seek_times(0.0, 0.0) == [0.0]


def test_tiny_roi_change_is_below_the_visible_swap_bar() -> None:
    original = np.full((40, 40, 3), 20, dtype=np.uint8)
    swapped = original.copy()
    swapped[8:24, 8:24] = 24
    change = roi_mean_change(original, swapped, [np.array([8, 8, 24, 24])])
    assert change < MIN_ROI_CHANGE
    swapped[8:24, 8:24] = 200
    assert roi_mean_change(original, swapped, [np.array([8, 8, 24, 24])]) >= MIN_ROI_CHANGE


def test_match_gender_blocks_only_when_enabled() -> None:
    source = Face(
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        det_score=0.9,
        gender=1,
    )
    target = Face(
        bbox=np.array([0, 0, 40, 40], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        det_score=0.9,
        gender=0,
    )

    class _Analyzer:
        def analyze(self, _frame):
            return [target]

    class _Swapper:
        def swap(self, frame, target_face, source_face, paste_back=True, coverage="full"):
            frame = frame.copy()
            frame[:] = 255
            return frame

    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    mapping = FaceMapping(source_face=source, reference_face=target)
    blocked = FaceSwapEngine(analyzer=_Analyzer(), swapper=_Swapper(), similarity_threshold=0.2)
    blocked.match_gender = True
    assert blocked.process_frame(frame, [mapping]).max() == 0
    assert blocked.stats.faces_swapped == 0

    allowed = FaceSwapEngine(analyzer=_Analyzer(), swapper=_Swapper(), similarity_threshold=0.2)
    assert allowed.match_gender is False
    assert allowed.process_frame(frame, [mapping]).max() == 255
    assert len(allowed.last_boxes) == 1

"""A short clip keeps the swap through a detection gap and writes the same timing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from faceswap.core import FaceMapping, FaceSwapEngine, SwapStats
from faceswap.coverage import template_128
from faceswap.face_analyzer import Face
from faceswap.video import process_video


def _write(path: Path, frames: int = 8, fps: float = 10.0, size=(64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), 30 + index, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _face(x: float = 16.0) -> Face:
    kps = template_128() * 0.45 + np.array([x, 8.0])
    return Face(
        bbox=np.array([x, 6, x + 36, 44], dtype=np.float32),
        kps=kps.astype(np.float32),
        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        det_score=0.99,
        gender=1,
    )


class _Analyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, _frame):
        self.calls += 1
        # Two empty frames in the middle. The tracker should hold the swap.
        if self.calls in {4, 5}:
            return []
        return [_face()]


class _Swapper:
    def __init__(self) -> None:
        self.calls = 0
        self.paste_wiped = False
        self.last_canvas = None
        self.last_matrix = None
        self.stage_ms = {}

    def swap(self, frame, target_face, source_face, paste_back=True, coverage="full"):
        self.calls += 1
        self.paste_wiped = False
        out = frame.copy()
        out[:8, :8] = (0, 255, 0)
        self.last_canvas = np.full((128, 128, 3), 255, dtype=np.uint8)
        self.last_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 20.0]], dtype=np.float64)
        return out

    def begin_face(self, _key) -> None:
        return None

    def reset_color_memory(self) -> None:
        return None


def test_swap_holds_through_a_gap_and_keeps_length(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write(path)
    analyzer = _Analyzer()
    swapper = _Swapper()
    engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper, detect_stride=1)
    source = _face(0)
    mapping = FaceMapping(source_face=source, reference_face=None)
    out = process_video(engine, [mapping], path, output_path=tmp_path / "out.mp4", keep_audio=False)
    assert out.exists()
    assert engine.stats.frames == 8
    assert engine.stats.faces_swapped >= 6
    assert engine.stats.coverage_ratio >= 0.75
    assert any("no face detected" in line for line in engine.stats.miss_log) or engine.stats.faces_held >= 1
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 8
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    assert abs(fps - 10.0) < 0.2
    assert int(width) == 64
    assert int(height) == 48
    assert "detect" in engine.stats.stage_line()


def test_trim_keeps_framerate_and_shortens_only_the_range(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write(path, frames=10, fps=10.0)

    class _Engine:
        def __init__(self) -> None:
            self.stats = SwapStats()
            self.calls = 0

        def process_frame(self, frame, mappings, time_s=0.0):
            self.calls += 1
            return frame

        def reset_tracks(self) -> None:
            return None

    engine = _Engine()
    out = process_video(
        engine,
        [],
        path,
        output_path=tmp_path / "trim.mp4",
        keep_audio=False,
        start_s=0.2,
        end_s=0.5,
    )
    assert out.exists()
    assert engine.calls == 3
    cap = cv2.VideoCapture(str(out))
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 10.0) < 0.2
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    cap.release()

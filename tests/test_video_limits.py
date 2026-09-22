"""The 5-minute guard runs before any frame is swapped."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from faceswap.core import SwapStats
from faceswap.video import (
    MAX_VIDEO_SECONDS,
    SwapCancelled,
    VideoTooLongError,
    assert_duration_allowed,
    process_video,
    swap_frame,
)


def _write_video(path: Path, frames: int = 6, fps: float = 10.0) -> None:
    size = (64, 48)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :] = (index * 20, 40, 80)
        writer.write(frame)
    writer.release()


class _DummyEngine:
    def __init__(self) -> None:
        self.stats = SwapStats()
        self.calls = 0

    def process_frame(self, frame, mappings, time_s=0.0):
        self.calls += 1
        return frame


def test_short_video_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "short.mp4"
    _write_video(path)
    info = assert_duration_allowed(path)
    assert info.duration_s <= MAX_VIDEO_SECONDS
    assert info.duration_s > 0


def test_over_five_minutes_is_rejected_before_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "long.mp4"
    _write_video(path, frames=4, fps=10)
    monkeypatch.setattr("faceswap.video._ffprobe_duration", lambda _path: 5 * 60 + 1)
    engine = _DummyEngine()
    with pytest.raises(VideoTooLongError, match="Nothing was processed"):
        process_video(engine, [], path, output_path=tmp_path / "out.mp4")
    assert engine.calls == 0
    assert not (tmp_path / "out.mp4").exists()


def test_exactly_five_minutes_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "edge.mp4"
    _write_video(path)
    monkeypatch.setattr("faceswap.video._ffprobe_duration", lambda _path: float(MAX_VIDEO_SECONDS))
    info = assert_duration_allowed(path)
    assert info.duration_s == pytest.approx(MAX_VIDEO_SECONDS)


def test_process_short_video_writes_mp4(tmp_path: Path) -> None:
    path = tmp_path / "in.mp4"
    _write_video(path, frames=3, fps=10)
    engine = _DummyEngine()
    out = process_video(engine, [], path, output_path=tmp_path / "out.mp4", keep_audio=False)
    assert out.exists()
    assert out.stat().st_size > 0
    assert engine.calls == 3


def test_cancel_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "in.mp4"
    _write_video(path, frames=5, fps=10)
    engine = _DummyEngine()
    cancel = threading.Event()

    def _frame(frame, mappings, time_s=0.0):
        engine.calls += 1
        cancel.set()
        return frame

    engine.process_frame = _frame  # type: ignore[method-assign]
    with pytest.raises(SwapCancelled):
        process_video(
            engine,
            [],
            path,
            output_path=tmp_path / "out.mp4",
            cancel_event=cancel,
        )
    assert not (tmp_path / "out.mp4").exists()
    assert engine.calls == 1


def test_half_resolution_swap_returns_the_original_size() -> None:
    class _Engine:
        def process_frame(self, frame, _mappings, time_s=0.0):
            self.shape = frame.shape
            return frame

    engine = _Engine()
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    out = swap_frame(engine, frame, [], scale=0.5)
    assert out.shape == frame.shape
    assert engine.shape == (20, 30, 3)

"""Desktop window and packaging checks."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from faceswap.cli import main
from faceswap.video import MAX_VIDEO_SECONDS, VideoTooLongError

REPO = Path(__file__).resolve().parents[1]


def _write_video(path: Path, frames: int = 4, fps: float = 10.0) -> None:
    size = (80, 48)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    frame = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _imports_gradio(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "gradio" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "gradio":
                return True
    return False


def test_web_ui_dependency_is_gone() -> None:
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "gradio" not in requirements
    assert "pyside6" in requirements
    for path in REPO.rglob("*.py"):
        if any(part in {".venv", "venv", ".git"} for part in path.parts):
            continue
        assert not _imports_gradio(path)
    assert not (REPO / "ui" / "app.py").exists()
    assert not (REPO / "preview.py").exists()
    assert not (REPO / "app.py").exists()


def test_removed_ui_command_points_at_desktop() -> None:
    assert main(["ui"]) == 2


def test_window_rejects_video_over_five_minutes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QApplication
    from videoswa.desktop import MainWindow

    path = tmp_path / "too_long.mp4"
    _write_video(path)
    monkeypatch.setattr("faceswap.video._ffprobe_duration", lambda _path: float(MAX_VIDEO_SECONDS) + 30)

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        with pytest.raises(VideoTooLongError, match="Nothing was processed"):
            window.load_video(path)
        assert window._video is None
    finally:
        window.close()
        window.worker.wait(3000)


def test_window_loads_short_video(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication
    from videoswa.desktop import MainWindow

    path = tmp_path / "clip.mp4"
    _write_video(path)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    try:
        info = window.load_video(path)
        assert info.duration_s <= MAX_VIDEO_SECONDS
        assert window.windowTitle() == "Videoswa"
        assert "TensorRT" in window.execution.itemText(0)
        assert window.coverage.currentData() == "full"
        assert "beard" in window.coverage.currentText().lower()
        assert window.face_mode.currentData() == "single"
        assert window.apply_all.isChecked() is False
        assert "Running on" in window.provider_banner.text()
        assert window.enhance.isEnabled()
        assert window.speed.currentData() == 1.0
        assert window.execution.findData("tensorrt") >= 0
        assert window.object_mask.isChecked() is True
        assert window.precise_edges.isChecked() is False
        assert window.fast_draft.isChecked() is True
        assert window.detect_every.isChecked() is True
        assert window.min_face.value() == 0
        assert window.compare_slider.value() == 50
        assert window.compare_slider.isEnabled() is False
        assert "XSeg" in window.model_status.text()
        assert window.assignment.text()
        assert window.output_edit.text().endswith("_videoswa.mp4")
        assert window.time_slider.isEnabled()
        assert window.match_gender.isChecked() is True
        assert window.beard.isChecked() is True
        assert window.encoder.currentData() == "auto"
        assert window.keep_audio.isChecked() is True
        assert window.detector_size.value() == 640
        assert window.run_btn.text() == "Swap"
        assert window.object_mask.text().lower().startswith("ai face mask")
        app.processEvents()
    finally:
        window.close()
        window.worker.wait(3000)

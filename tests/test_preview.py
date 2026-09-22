"""One-frame preview uses the export mapping and does not write a video."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from faceswap.face_analyzer import Face
from videoswa.images import side_by_side, wipe_preview
from videoswa.jobs import FACE_MODE_MULTIPLE, SwapRequest
from videoswa.worker import EngineWorker, PreviewRequest


def _face(x: float = 0.0) -> Face:
    return Face(
        bbox=np.array([x, 0, x + 40, 50], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        embedding=np.array([1, 0, 0], dtype=np.float32),
        det_score=0.99,
        gender=1,
    )


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (80, 48))
    assert writer.isOpened()
    frame = np.full((48, 80, 3), 40, dtype=np.uint8)
    for _ in range(3):
        writer.write(frame)
    writer.release()


def test_side_by_side_keeps_both_frames() -> None:
    original = np.zeros((20, 30, 3), dtype=np.uint8)
    swapped = np.full((20, 30, 3), 255, dtype=np.uint8)
    stacked = side_by_side(original, swapped)
    assert stacked.shape == (20, 30 + 8 + 30, 3)
    assert int(original[0, 0, 0]) == 0
    assert stacked[10, 5, 0] < 40
    assert stacked[10, -5, 0] > 200


def test_wipe_shows_swap_on_the_left_and_original_on_the_right() -> None:
    original = np.zeros((20, 40, 3), dtype=np.uint8)
    swapped = np.full((20, 40, 3), 200, dtype=np.uint8)
    wiped = wipe_preview(original, swapped, 0.5)
    assert wiped[10, 4, 0] == 200
    assert wiped[10, 36, 0] == 0
    assert int(wiped[10, 20, 0]) == 255


def test_preview_job_does_not_start_an_export() -> None:
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    worker = EngineWorker()
    seen: dict[str, object] = {}
    worker._preview = lambda job: seen.setdefault("preview", job)  # type: ignore[method-assign]
    worker._swap = lambda job: seen.setdefault("swap", True)  # type: ignore[method-assign]
    request = SwapRequest(
        video_path=Path("clip.mp4"),
        output_path=Path("out.mp4"),
        single_source=Path("face.png"),
        selected_face=_face(),
    )
    worker.request_preview(PreviewRequest(timestamp_s=1.5, swap=request))
    worker.shutdown()
    worker.run()
    assert seen["preview"].timestamp_s == 1.5
    assert "swap" not in seen


def test_preview_button_follows_video_and_source(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from videoswa.desktop import MainWindow
    from videoswa.worker import DetectedPerson

    video = tmp_path / "clip.mp4"
    image = tmp_path / "source.png"
    _write_video(video)
    assert cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    captured: dict[str, PreviewRequest] = {}
    window.worker.request_preview = lambda job: captured.setdefault("job", job)  # type: ignore[method-assign]
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    try:
        assert window.preview_btn.text() == "Preview frame"
        assert window.preview_btn.isEnabled() is False
        window.load_video(video)
        assert window.preview_btn.isEnabled() is False

        window._single_source = image
        window._update_preview_button()
        assert window.preview_btn.isEnabled() is False

        crop = np.zeros((16, 16, 3), dtype=np.uint8)
        window._on_detected([DetectedPerson(crop_bgr=crop, face=_face(), index=0)])
        window._single_source = image
        window._update_preview_button()
        app.processEvents()
        if "job" not in captured:
            assert window.preview_btn.isEnabled() is True
            window.preview_btn.click()
            app.processEvents()
        job = captured["job"]
        assert job.timestamp_s == 0.0
        assert job.swap.single_source == image
        assert job.swap.coverage == "full"
        assert job.swap.selected_face is window._selected_face()
        assert not (tmp_path / "preview-only.mp4").exists()
        assert not list(tmp_path.glob("*videoswa.mp4"))

        window._set_busy(False)
        window.face_mode.setCurrentIndex(1)
        app.processEvents()
        assert window.face_mode.currentData() == FACE_MODE_MULTIPLE
        assert window.preview_btn.isEnabled() is False
        window._cards[0].source_path = image
        window._cards[0].source_changed.emit()
        assert window.preview_btn.isEnabled() is True
    finally:
        window.close()
        window.worker.wait(3000)

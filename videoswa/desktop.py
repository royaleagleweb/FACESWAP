"""Videoswa desktop window."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from faceswap.utils import logger
from faceswap.video import (
    VIDEO_SUFFIXES,
    VideoDurationUnknownError,
    VideoInfo,
    VideoTooLongError,
    assert_duration_allowed,
    format_timestamp,
    read_frame_at,
)
from videoswa.images import bgr_to_qpixmap
from videoswa.jobs import FaceSource, SwapRequest, validate_request
from videoswa.worker import DetectRequest, DetectedPerson, EngineWorker

_VIDEO_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v)"
_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
_PRESETS = [
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
]


def _gfpgan_available() -> bool:
    try:
        import gfpgan  # noqa: F401
    except Exception:
        return False
    return True


class _LogBridge(QWidget):
    message = Signal(str)


class _QtLogHandler(logging.Handler):
    def __init__(self, bridge: _LogBridge) -> None:
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.message.emit(self.format(record))
        except Exception:
            pass


class FaceCard(QFrame):
    """One detected person plus the source image that should replace them."""

    def __init__(self, person: DetectedPerson, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.person = person
        self.source_path: Optional[Path] = None
        self.setObjectName("FaceCard")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        thumb = QLabel()
        thumb.setFixedSize(96, 96)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = bgr_to_qpixmap(person.crop_bgr, max_edge=180)
        thumb.setPixmap(pix.scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(thumb)

        column = QVBoxLayout()
        title = QLabel(f"Person {person.index + 1}")
        title.setObjectName("CardTitle")
        self.source_label = QLabel("No source image — this person stays unchanged")
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("Muted")
        buttons = QHBoxLayout()
        choose = QPushButton("Choose source…")
        clear = QPushButton("Clear")
        choose.clicked.connect(self._choose)
        clear.clicked.connect(self._clear)
        buttons.addWidget(choose)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        column.addWidget(title)
        column.addWidget(self.source_label)
        column.addLayout(buttons)
        layout.addLayout(column, stretch=1)

    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Source face for person {self.person.index + 1}", "", _IMAGE_FILTER)
        if not path:
            return
        self.source_path = Path(path)
        self.source_label.setText(self.source_path.name)

    def _clear(self) -> None:
        self.source_path = None
        self.source_label.setText("No source image — this person stays unchanged")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Videoswa")
        self.resize(1180, 900)
        self._video: Optional[Path] = None
        self._info: Optional[VideoInfo] = None
        self._cards: list[FaceCard] = []
        self._busy = False
        self._single_source: Optional[Path] = None

        self.worker = EngineWorker()
        self.worker.detect_ready.connect(self._on_detected)
        self.worker.detect_failed.connect(self._on_detect_failed)
        self.worker.swap_progress.connect(self._on_progress)
        self.worker.swap_finished.connect(self._on_swap_finished)
        self.worker.swap_failed.connect(self._on_swap_failed)
        self.worker.swap_cancelled.connect(self._on_swap_cancelled)
        self.worker.status.connect(self._set_status)
        self._log_bridge = _LogBridge()
        self._log_handler = _QtLogHandler(self._log_bridge)
        self._log_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._build()
        self._apply_style()
        self._log_bridge.message.connect(self._append_log)
        logger.addHandler(self._log_handler)
        self.worker.start()
        if not shutil.which("ffmpeg"):
            self._set_status("FFmpeg was not found. Audio cannot be copied into the MP4 until it is installed.")

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(8)
        left.setContentsMargins(0, 0, 8, 0)
        title = QLabel("Videoswa")
        title.setObjectName("Title")
        subtitle = QLabel("Map source faces onto people in a video, then write an MP4.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("Muted")
        left.addWidget(title)
        left.addWidget(subtitle)

        target_box = QFrame()
        target_box.setObjectName("Panel")
        target_layout = QVBoxLayout(target_box)
        target_layout.addWidget(QLabel("Target video"))
        browse_row = QHBoxLayout()
        self.video_label = QLabel("No video selected")
        self.video_label.setWordWrap(True)
        self.video_label.setObjectName("Muted")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_video)
        browse_row.addWidget(self.video_label, stretch=1)
        browse_row.addWidget(browse)
        target_layout.addLayout(browse_row)
        self.duration_label = QLabel("Duration: —")
        target_layout.addWidget(self.duration_label)

        self.time_label = QLabel("Sample frame  0:00")
        target_layout.addWidget(self.time_label)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._on_slider)
        target_layout.addWidget(self.time_slider)
        left.addWidget(target_box)

        options = QFrame()
        options.setObjectName("Panel")
        form = QVBoxLayout(options)
        form.addWidget(QLabel("Execution"))
        self.execution = QComboBox()
        self.execution.addItem("Auto (TensorRT → CUDA → CPU)", "auto")
        self.execution.addItem("CUDA only", "cuda")
        self.execution.addItem("CPU only", "cpu")
        form.addWidget(self.execution)

        sim_row = QHBoxLayout()
        sim_row.addWidget(QLabel("Match threshold"))
        self.similarity = QSpinBox()
        self.similarity.setRange(10, 90)
        self.similarity.setValue(45)
        self.similarity.setSuffix(" %")
        self.similarity.setToolTip("Cosine similarity required to treat a detection as the same person.")
        sim_row.addWidget(self.similarity)
        form.addLayout(sim_row)

        form.addWidget(QLabel("Face coverage"))
        self.coverage = QComboBox()
        self.coverage.addItem("Full, including beard", "full")
        self.coverage.addItem("Normal (tight face)", "normal")
        self.coverage.setToolTip(
            "Full replaces the jaw, cheeks, and beard. Normal keeps a tight oval around the inner face."
        )
        form.addWidget(self.coverage)

        self.keep_audio = QCheckBox("Keep original audio")
        self.keep_audio.setChecked(True)
        self.enhance = QCheckBox("Sharpen swapped faces (GFPGAN)")
        self.enhance.setChecked(False)
        if not _gfpgan_available():
            self.enhance.setEnabled(False)
            self.enhance.setToolTip("Optional. Install GFPGAN to enable this (requirements-enhance.txt).")
        form.addWidget(self.keep_audio)
        form.addWidget(self.enhance)

        crf_row = QHBoxLayout()
        crf_row.addWidget(QLabel("Quality (CRF)"))
        self.crf = QSpinBox()
        self.crf.setRange(0, 32)
        self.crf.setValue(18)
        crf_row.addWidget(self.crf)
        form.addLayout(crf_row)

        self.preset = QComboBox()
        self.preset.addItems(_PRESETS)
        self.preset.setCurrentText("medium")
        form.addWidget(QLabel("Encode preset"))
        form.addWidget(self.preset)
        left.addWidget(options)

        output_box = QFrame()
        output_box.setObjectName("Panel")
        output_layout = QVBoxLayout(output_box)
        output_layout.addWidget(QLabel("Output MP4"))
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("output.mp4")
        out_browse = QPushButton("Save as…")
        out_browse.clicked.connect(self._pick_output)
        out_row.addWidget(self.output_edit, stretch=1)
        out_row.addWidget(out_browse)
        output_layout.addLayout(out_row)
        left.addWidget(output_box)

        self.detect_btn = QPushButton("Detect faces")
        self.detect_btn.clicked.connect(self._detect)
        self.run_btn = QPushButton("Run swap")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        left.addWidget(self.detect_btn)
        left.addWidget(self.run_btn)
        left.addWidget(self.cancel_btn)
        ethics = QLabel("Swap only people you have consent to depict. Label synthetic media when you publish it.")
        ethics.setWordWrap(True)
        ethics.setObjectName("Muted")
        left.addWidget(ethics)
        left.addStretch(1)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_wrap)
        left_scroll.setFixedWidth(376)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        outer.addWidget(left_scroll)

        right = QVBoxLayout()
        self.preview = QLabel("Choose a video to preview a frame.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setObjectName("Preview")
        right.addWidget(self.preview)

        header = QHBoxLayout()
        header.addWidget(QLabel("People in this frame"))
        header.addStretch(1)
        self.single_check = QCheckBox("Use one source for every face")
        self.single_check.toggled.connect(self._toggle_single)
        header.addWidget(self.single_check)
        right.addLayout(header)

        self.single_btn = QPushButton("Choose source for everyone…")
        self.single_btn.setVisible(False)
        self.single_btn.clicked.connect(self._pick_single_source)
        self.single_label = QLabel("")
        self.single_label.setObjectName("Muted")
        self.single_label.setVisible(False)
        right.addWidget(self.single_btn)
        right.addWidget(self.single_label)

        self.face_host = QWidget()
        self.face_layout = QVBoxLayout(self.face_host)
        self.face_layout.setContentsMargins(0, 0, 0, 0)
        self.face_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.face_host)
        scroll.setMinimumHeight(180)
        right.addWidget(scroll, stretch=1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        right.addWidget(self.progress)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setFixedHeight(120)
        right.addWidget(self.log)
        outer.addLayout(right, stretch=1)

    def _apply_style(self) -> None:
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(
            """
            QWidget { color: #e6e8eb; background: #16181d; }
            QFrame#Panel, QFrame#FaceCard {
                background: #1f232a;
                border: 1px solid #31363f;
                border-radius: 8px;
            }
            QLabel#Title { font-size: 22px; font-weight: 600; background: transparent; }
            QLabel#Muted, QLabel#CardTitle { background: transparent; }
            QLabel#Muted { color: #9aa3ad; }
            QLabel#CardTitle { font-weight: 600; }
            QLabel#Preview {
                background: #12141a;
                border: 1px solid #31363f;
                border-radius: 8px;
                color: #9aa3ad;
            }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {
                background: #12141a;
                border: 1px solid #31363f;
                border-radius: 6px;
                padding: 4px 6px;
            }
            QPushButton {
                background: #2a303a;
                border: 1px solid #3a414d;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton:hover { background: #343b47; }
            QPushButton:disabled { color: #6d7580; }
            QPushButton#Primary { background: #2f6fed; border-color: #2f6fed; }
            QPushButton#Primary:hover { background: #3d7cf5; }
            QProgressBar {
                background: #12141a;
                border: 1px solid #31363f;
                border-radius: 6px;
                text-align: center;
            }
            QProgressBar::chunk { background: #2f6fed; border-radius: 5px; }
            QScrollArea { border: none; }
            """
        )

    def _set_status(self, text: str) -> None:
        self.status.setText(text)
        self._append_log(text)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _timestamp(self) -> float:
        return self.time_slider.value() / 10.0

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Target video", "", _VIDEO_FILTER)
        if not path:
            return
        video = Path(path)
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            QMessageBox.warning(self, "Not a video", "Choose an mp4, mov, avi, mkv, webm, or m4v file.")
            return
        try:
            self.load_video(video)
        except VideoTooLongError as exc:
            QMessageBox.critical(self, "Video is too long", str(exc))
        except VideoDurationUnknownError as exc:
            QMessageBox.critical(self, "Duration unknown", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Could not open video", str(exc))

    def load_video(self, video: Path) -> VideoInfo:
        """Probe a video and keep it only when it is within the 5-minute limit."""
        info = assert_duration_allowed(video)
        self._video = video
        self._info = info
        self.video_label.setText(video.name)
        self.duration_label.setText(
            f"Duration: {format_timestamp(info.duration_s)}  ·  "
            f"{info.width}×{info.height}  ·  {info.fps:.2f} fps"
        )
        steps = max(0, int(info.duration_s * 10))
        self.time_slider.setEnabled(True)
        self.time_slider.setRange(0, steps)
        self.time_slider.setValue(0)
        self.output_edit.setText(str(video.with_name(f"{video.stem}_videoswa.mp4")))
        self._clear_faces()
        self._refresh_preview()
        self._set_status(f"Loaded {video.name}. Sample a frame, then detect faces.")
        return info

    def _pick_output(self) -> None:
        suggestion = self.output_edit.text() or "videoswa.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Save swapped video", suggestion, "MP4 (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output_edit.setText(path)

    def _on_slider(self, _value: int) -> None:
        self.time_label.setText(f"Sample frame  {format_timestamp(self._timestamp())}")
        self._preview_timer.start(120)

    def _refresh_preview(self) -> None:
        if self._video is None:
            return
        try:
            frame = read_frame_at(self._video, self._timestamp())
        except Exception as exc:
            self.preview.setText(str(exc))
            return
        pix = bgr_to_qpixmap(frame, max_edge=960)
        self.preview.setPixmap(
            pix.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._video is not None and not self._busy:
            self._preview_timer.start(80)

    def _clear_faces(self) -> None:
        self._cards.clear()
        while self.face_layout.count():
            item = self.face_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.face_layout.addStretch(1)

    def _detect(self) -> None:
        if self._busy:
            return
        if self._video is None or self._info is None:
            QMessageBox.information(self, "No video", "Choose a target video first.")
            return
        self._set_busy(True, cancellable=False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Detecting…")
        self.worker.request_detect(
            DetectRequest(
                video_path=self._video,
                timestamp_s=self._timestamp(),
                execution=self.execution.currentData(),
            )
        )

    def _on_detected(self, people: list) -> None:
        self._clear_faces()
        stretch = self.face_layout.takeAt(self.face_layout.count() - 1)
        del stretch
        if not people:
            empty = QLabel("No faces at this timestamp. Move the slider and detect again.")
            empty.setObjectName("Muted")
            self.face_layout.addWidget(empty)
        for person in people:
            if not isinstance(person, DetectedPerson):
                continue
            card = FaceCard(person)
            self._cards.append(card)
            self.face_layout.addWidget(card)
        self.face_layout.addStretch(1)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self._set_busy(False)
        self._set_status(
            f"Detected {len(self._cards)} face(s) at {format_timestamp(self._timestamp())}. "
            "Choose a source image for each person you want to replace."
        )

    def _on_detect_failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self._set_busy(False)
        QMessageBox.critical(self, "Detection failed", message)

    def _toggle_single(self, checked: bool) -> None:
        self.single_btn.setVisible(checked)
        self.single_label.setVisible(checked)
        for card in self._cards:
            card.setEnabled(not checked)

    def _pick_single_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Source face for every person", "", _IMAGE_FILTER)
        if not path:
            return
        self._single_source = Path(path)
        self.single_label.setText(self._single_source.name)

    def _run(self) -> None:
        if self._busy or self._video is None:
            if self._video is None:
                QMessageBox.information(self, "No video", "Choose a target video first.")
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "No output", "Choose where to save the MP4.")
            return
        sources: list[FaceSource] = []
        single: Optional[Path] = None
        if self.single_check.isChecked():
            if self._single_source is None:
                QMessageBox.warning(self, "No source", "Choose the source image that should replace every face.")
                return
            single = self._single_source
        else:
            for card in self._cards:
                if card.source_path is None:
                    continue
                sources.append(
                    FaceSource(
                        face=card.person.face,
                        source_path=card.source_path,
                        label=f"person_{card.person.index + 1}",
                    )
                )
        request = SwapRequest(
            video_path=self._video,
            output_path=Path(output_text),
            execution=str(self.execution.currentData()),
            enhance=self.enhance.isChecked(),
            similarity=self.similarity.value() / 100.0,
            coverage=str(self.coverage.currentData()),
            keep_audio=self.keep_audio.isChecked(),
            crf=self.crf.value(),
            preset=self.preset.currentText(),
            single_source=single,
            face_sources=sources,
        )
        try:
            validate_request(request)
        except VideoTooLongError as exc:
            QMessageBox.critical(self, "Video is too long", str(exc))
            return
        except VideoDurationUnknownError as exc:
            QMessageBox.critical(self, "Duration unknown", str(exc))
            return
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot start", str(exc))
            return

        if request.output_path.exists():
            answer = QMessageBox.question(
                self,
                "Replace file?",
                f"{request.output_path.name} already exists. Replace it?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._set_busy(True, cancellable=True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Starting…")
        self.worker.request_swap(request)

    def _cancel(self) -> None:
        self.cancel_btn.setEnabled(False)
        self._set_status("Cancelling after the current frame…")
        self.worker.request_cancel()

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
            self.progress.setFormat(f"Frame {done}")
            return
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"%v / %m frames")

    def _on_swap_finished(self, path: str, summary: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setFormat("Done")
        self._set_busy(False)
        self._set_status(f"Wrote {path}. {summary}")
        QMessageBox.information(self, "Swap finished", f"Saved\n{path}\n\n{summary}")

    def _on_swap_failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self._set_busy(False)
        QMessageBox.critical(self, "Swap failed", message)

    def _on_swap_cancelled(self) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Cancelled")
        self._set_busy(False)
        self._set_status("Swap cancelled. No output file was written.")

    def _set_busy(self, busy: bool, cancellable: bool = False) -> None:
        self._busy = busy
        self.detect_btn.setEnabled(not busy)
        self.run_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy and cancellable)
        self.execution.setEnabled(not busy)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        logger.removeHandler(self._log_handler)
        self.worker.request_cancel()
        self.worker.shutdown()
        self.worker.wait(3000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Videoswa")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

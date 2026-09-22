"""Videoswa desktop window."""

from __future__ import annotations

import json
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

from faceswap.face_analyzer import face_label
from faceswap.swapper import gfpgan_install_tip
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
from videoswa.images import bgr_to_qpixmap, wipe_preview
from videoswa.jobs import FACE_MODE_MULTIPLE, FACE_MODE_SINGLE, FaceSource, SwapRequest, validate_request
from videoswa.project import PROJECT_SUFFIX, gender_mark, load_project, save_project
from videoswa.worker import DetectRequest, DetectedPerson, EngineWorker, PreviewRequest

_VIDEO_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v)"
_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
_PRESETS = [
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
]


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
    """One detected face, its gender label, and an optional per-face source."""

    clicked = Signal(int)
    source_changed = Signal()

    def __init__(self, person: DetectedPerson, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.person = person
        self.source_path: Optional[Path] = None
        self._selected = False
        self._multi = False
        self.setObjectName("FaceCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        thumb = QLabel()
        thumb.setFixedSize(96, 96)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = bgr_to_qpixmap(person.crop_bgr, max_edge=180)
        thumb.setPixmap(pix.scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(thumb)

        column = QVBoxLayout()
        self.title = QLabel(face_label(person.index, person.face.gender))
        self.title.setObjectName("CardTitle")
        self.hint = QLabel("Click to select")
        self.hint.setObjectName("Muted")
        self.source_label = QLabel("No source image — this person stays unchanged")
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("Muted")
        self.source_label.setVisible(False)
        buttons = QHBoxLayout()
        self.choose_btn = QPushButton("Choose source…")
        self.clear_btn = QPushButton("Clear")
        self.choose_btn.clicked.connect(self._choose)
        self.clear_btn.clicked.connect(self._clear)
        self.choose_btn.setVisible(False)
        self.clear_btn.setVisible(False)
        buttons.addWidget(self.choose_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addStretch(1)
        column.addWidget(self.title)
        column.addWidget(self.hint)
        column.addWidget(self.source_label)
        column.addLayout(buttons)
        layout.addLayout(column, stretch=1)

    def set_multi(self, multi: bool) -> None:
        self._multi = multi
        self.source_label.setVisible(multi)
        self.choose_btn.setVisible(multi)
        self.clear_btn.setVisible(multi)
        if multi:
            self.hint.setVisible(False)
        else:
            self.hint.setVisible(True)
            self._refresh_hint()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.setObjectName("FaceCardSelected" if selected else "FaceCard")
        self.style().unpolish(self)
        self.style().polish(self)
        self._refresh_hint()

    def _refresh_hint(self) -> None:
        if self._multi:
            return
        self.hint.setText("Selected" if self._selected else "Click to select")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.clicked.emit(self.person.index)
        super().mousePressEvent(event)

    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Source face for {face_label(self.person.index, self.person.face.gender)}",
            "",
            _IMAGE_FILTER,
        )
        if not path:
            return
        self.source_path = Path(path)
        self.source_label.setText(self.source_path.name)
        self.source_changed.emit()

    def _clear(self) -> None:
        self.source_path = None
        self.source_label.setText("No source image — this person stays unchanged")
        self.source_changed.emit()


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
        self._selected_index: Optional[int] = None
        self._showing_swap = False
        self._preview_bgr: Optional[object] = None
        self._preview_original = None
        self._preview_swapped = None
        self._rotation_paths: list[Path] = []

        self.worker = EngineWorker()
        self.worker.detect_ready.connect(self._on_detected)
        self.worker.detect_failed.connect(self._on_detect_failed)
        self.worker.preview_ready.connect(self._on_preview_ready)
        self.worker.preview_failed.connect(self._on_preview_failed)
        self.worker.swap_progress.connect(self._on_progress)
        self.worker.swap_finished.connect(self._on_swap_finished)
        self.worker.swap_failed.connect(self._on_swap_failed)
        self.worker.swap_cancelled.connect(self._on_swap_cancelled)
        self.worker.status.connect(self._set_status)
        self.worker.provider.connect(self.show_provider)
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
        subtitle = QLabel("Swap a face in a video, then write an MP4.")
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
        self.preview_btn = QPushButton("Preview swap on this frame")
        self.preview_btn.setEnabled(False)
        self.preview_btn.setToolTip(
            "Swap only the sample frame. Drag Before / after to wipe between "
            "the original and the swapped frame. Does not write the video."
        )
        self.preview_btn.clicked.connect(self._preview_swap)
        target_layout.addWidget(self.preview_btn)
        left.addWidget(target_box)

        options = QFrame()
        options.setObjectName("Panel")
        form = QVBoxLayout(options)
        form.addWidget(QLabel("Execution"))
        self.execution = QComboBox()
        if sys.platform.startswith("win"):
            auto_label = "Auto (TensorRT → CUDA → DirectML → CPU)"
        else:
            auto_label = "Auto (TensorRT → CUDA → CPU)"
        self.execution.addItem(auto_label, "auto")
        self.execution.addItem("TensorRT", "tensorrt")
        self.execution.addItem("CUDA only", "cuda")
        if sys.platform.startswith("win"):
            self.execution.addItem("DirectML", "directml")
        self.execution.addItem("CPU only", "cpu")
        form.addWidget(self.execution)
        self.provider_banner = QLabel("Running on: waiting for models")
        self.provider_banner.setWordWrap(True)
        self.provider_banner.setObjectName("ProviderBanner")
        form.addWidget(self.provider_banner)

        sim_row = QHBoxLayout()
        sim_row.addWidget(QLabel("Match threshold"))
        self.similarity = QSpinBox()
        self.similarity.setRange(10, 90)
        self.similarity.setValue(32)
        self.similarity.setSuffix(" %")
        self.similarity.setToolTip(
            "Similarity required to lock a person the first time. After that, "
            "Videoswa keeps the same source while the face stays in place. "
            "Raise this if the wrong person is swapped."
        )
        sim_row.addWidget(self.similarity)
        form.addLayout(sim_row)
        tips = QLabel(
            "Quality tips: use a sharp frontal source photo, sample a frame where "
            "the face is clear, and leave coverage on Full. Raise the match "
            "threshold only if the wrong person is swapped. If the result looks "
            "soft, enable GFPGAN below — it stays off until you turn it on."
        )
        tips.setWordWrap(True)
        tips.setObjectName("Muted")
        form.addWidget(tips)

        form.addWidget(QLabel("Face coverage"))
        self.coverage = QComboBox()
        self.coverage.addItem("Full, including beard", "full")
        self.coverage.addItem("Normal (tight face)", "normal")
        self.coverage.setToolTip(
            "Full replaces the jaw, cheeks, and beard. Normal keeps a tight oval around the inner face."
        )
        form.addWidget(self.coverage)

        self.object_mask = QCheckBox("Object mask (XSeg)")
        self.object_mask.setChecked(True)
        self.object_mask.setToolTip(
            "On for every normal swap. Keeps lollipops, food, and hands when they "
            "cover the face. Uses models/xseg.onnx when that file is present, and "
            "the built-in mask otherwise. The mask is warped to the face only."
        )
        self.precise_edges = QCheckBox("Precise edges (BiSeNet)")
        self.precise_edges.setChecked(False)
        self.precise_edges.setToolTip(
            "Off by default. Turn on for a tighter hairline when models/bisenet.onnx is installed."
        )
        self.fast_draft = QCheckBox("Fast draft in preview")
        self.fast_draft.setChecked(True)
        self.fast_draft.setToolTip(
            "Preview and playback skip GFPGAN and BiSeNet. Object mask stays on. "
            "Run swap uses the checkboxes above."
        )
        self.detect_every = QCheckBox("Detect every 2nd frame")
        self.detect_every.setChecked(True)
        self.detect_every.setToolTip(
            "Full export runs the detector on every other frame and reuses landmarks between them."
        )
        form.addWidget(self.object_mask)
        form.addWidget(self.precise_edges)
        form.addWidget(self.fast_draft)
        form.addWidget(self.detect_every)
        self.fast_draft_btn = QPushButton("⚡ Fast draft")
        self.fast_draft_btn.setToolTip("Restore off, precise edges off, object mask on.")
        self.fast_draft_btn.clicked.connect(self._apply_fast_draft)
        form.addWidget(self.fast_draft_btn)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min face size (px)"))
        self.min_face = QSpinBox()
        self.min_face.setRange(0, 512)
        self.min_face.setValue(0)
        self.min_face.setToolTip("Skip background faces smaller than this. 0 keeps every detection. Try 64 in a crowd.")
        min_row.addWidget(self.min_face)
        form.addLayout(min_row)

        form.addWidget(QLabel("Source rotation"))
        self.rotation = QComboBox()
        self.rotation.addItem("A different face per person", "per_person")
        self.rotation.addItem("When the face changes", "scene")
        self.rotation.addItem("Every N seconds", "interval")
        form.addWidget(self.rotation)
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Every"))
        self.rotation_seconds = QSpinBox()
        self.rotation_seconds.setRange(1, 120)
        self.rotation_seconds.setValue(5)
        self.rotation_seconds.setSuffix(" s")
        rot_row.addWidget(self.rotation_seconds)
        self.add_rotation_btn = QPushButton("Add rotation source…")
        self.add_rotation_btn.clicked.connect(self._add_rotation_source)
        rot_row.addWidget(self.add_rotation_btn)
        form.addLayout(rot_row)
        self.rotation_label = QLabel("Rotation uses the source images you add here.")
        self.rotation_label.setWordWrap(True)
        self.rotation_label.setObjectName("Muted")
        form.addWidget(self.rotation_label)

        form.addWidget(QLabel("Export speed"))
        self.speed = QComboBox()
        self.speed.addItem("Full quality", 1.0)
        self.speed.addItem("Half resolution (faster)", 0.5)
        self.speed.setToolTip(
            "Full quality is the default export. Half resolution is faster on CPU "
            "and DirectML. The MP4 is still the original size."
        )
        form.addWidget(self.speed)

        self.keep_audio = QCheckBox("Keep original audio")
        self.keep_audio.setChecked(True)
        self.enhance = QCheckBox("Sharpen swapped faces (GFPGAN)")
        self.enhance.setChecked(False)
        install_tip = gfpgan_install_tip()
        self.enhance.setToolTip(
            install_tip
            or "Optional, and off by default. Turn this on when the swap looks soft. "
            "The first enhanced swap downloads GFPGANv1.4.pth into models/."
        )
        self.enhance.toggled.connect(self._on_enhance_toggled)
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
        project_row = QHBoxLayout()
        save_project_btn = QPushButton("Save project")
        load_project_btn = QPushButton("Load project")
        save_project_btn.clicked.connect(self._save_project)
        load_project_btn.clicked.connect(self._load_project)
        project_row.addWidget(save_project_btn)
        project_row.addWidget(load_project_btn)
        output_layout.addLayout(project_row)
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
        self.compare_slider = QSlider(Qt.Orientation.Horizontal)
        self.compare_slider.setRange(0, 100)
        self.compare_slider.setValue(50)
        self.compare_slider.setEnabled(False)
        self.compare_slider.setToolTip("Drag to wipe between the swapped frame and the original.")
        self.compare_slider.valueChanged.connect(self._on_compare)
        right.addWidget(QLabel("Before / after"))
        right.addWidget(self.compare_slider)

        header = QHBoxLayout()
        header.addWidget(QLabel("Swap mode"))
        self.face_mode = QComboBox()
        self.face_mode.addItem("Single face", FACE_MODE_SINGLE)
        self.face_mode.addItem("Multiple faces", FACE_MODE_MULTIPLE)
        self.face_mode.currentIndexChanged.connect(self._on_mode_changed)
        header.addWidget(self.face_mode, stretch=1)
        right.addLayout(header)

        self.mode_hint = QLabel("One source image replaces the selected face.")
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setObjectName("Muted")
        right.addWidget(self.mode_hint)

        self.single_btn = QPushButton("Choose source image…")
        self.single_btn.clicked.connect(self._pick_single_source)
        self.single_label = QLabel("No source image")
        self.single_label.setObjectName("Muted")
        self.apply_all = QCheckBox("Apply this source to every face")
        self.apply_all.setChecked(False)
        self.apply_all.toggled.connect(self._on_apply_all)
        right.addWidget(self.single_btn)
        right.addWidget(self.single_label)
        right.addWidget(self.apply_all)

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
            QFrame#Panel, QFrame#FaceCard, QFrame#FaceCardSelected {
                background: #1f232a;
                border: 1px solid #31363f;
                border-radius: 8px;
            }
            QFrame#FaceCardSelected { border: 2px solid #2f6fed; }
            QLabel#Title { font-size: 22px; font-weight: 600; background: transparent; }
            QLabel#Muted, QLabel#CardTitle { background: transparent; }
            QLabel#Muted { color: #9aa3ad; }
            QLabel#ProviderBanner, QLabel#ProviderSlow {
                font-weight: 600;
                padding: 8px;
                border-radius: 6px;
            }
            QLabel#ProviderBanner { background: #1e3a2f; color: #d5f5e3; }
            QLabel#ProviderSlow { background: #4a3418; color: #ffe0b2; }
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

    def show_provider(self, text: str) -> None:
        self.provider_banner.setText(text)
        slow = "CPU" in text and "DirectML" not in text
        self.provider_banner.setObjectName("ProviderSlow" if slow else "ProviderBanner")
        self.provider_banner.style().unpolish(self.provider_banner)
        self.provider_banner.style().polish(self.provider_banner)

    def _on_enhance_toggled(self, checked: bool) -> None:
        if not checked:
            return
        tip = gfpgan_install_tip()
        if tip is None:
            return
        QMessageBox.information(self, "GFPGAN is not installed", tip)
        self.enhance.blockSignals(True)
        self.enhance.setChecked(False)
        self.enhance.blockSignals(False)

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
        self._showing_swap = False
        self._set_status(f"Loaded {video.name}. Sample a frame, then detect faces.")
        self._update_preview_button()
        return info

    def _pick_output(self) -> None:
        suggestion = self.output_edit.text() or "videoswa.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Save swapped video", suggestion, "MP4 (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output_edit.setText(path)

    def _on_slider(self, _value: int) -> None:
        self._showing_swap = False
        self.compare_slider.setEnabled(False)
        self.time_label.setText(f"Sample frame  {format_timestamp(self._timestamp())}")
        self._preview_timer.start(120)

    def _paint_preview(self, frame) -> None:
        pix = bgr_to_qpixmap(frame, max_edge=960)
        self.preview.setPixmap(
            pix.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _refresh_preview(self) -> None:
        if self._showing_swap and self._preview_bgr is not None:
            self._paint_preview(self._preview_bgr)
            return
        if self._video is None:
            return
        try:
            frame = read_frame_at(self._video, self._timestamp())
        except Exception as exc:
            self.preview.setText(str(exc))
            return
        self._paint_preview(frame)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._video is not None and not self._busy:
            self._preview_timer.start(80)

    def _clear_faces(self) -> None:
        self._cards.clear()
        self._selected_index = None
        while self.face_layout.count():
            item = self.face_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.face_layout.addStretch(1)
        self._update_preview_button()

    def _single_mode(self) -> bool:
        return self.face_mode.currentData() == FACE_MODE_SINGLE

    def _on_mode_changed(self) -> None:
        single = self._single_mode()
        self.single_btn.setVisible(single)
        self.single_label.setVisible(single)
        self.apply_all.setVisible(single)
        if single:
            if self.apply_all.isChecked():
                self.mode_hint.setText("One source image replaces every face in the video.")
            else:
                self.mode_hint.setText("One source image replaces the selected face. Click a thumbnail to change it.")
        else:
            self.mode_hint.setText("Choose a source image on each face you want to replace. Leave a face empty to keep it.")
        for card in self._cards:
            card.set_multi(not single)
            card.set_selected(single and not self.apply_all.isChecked() and card.person.index == self._selected_index)
        self._update_preview_button()

    def _on_apply_all(self, _checked: bool) -> None:
        self._on_mode_changed()

    def _select_face(self, index: int) -> None:
        if not self._single_mode():
            return
        self._selected_index = index
        for card in self._cards:
            card.set_selected(card.person.index == index and not self.apply_all.isChecked())
        self._update_preview_button()

    def _selected_face(self):
        for card in self._cards:
            if card.person.index == self._selected_index:
                return card.person.face
        return None

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
            card.clicked.connect(self._select_face)
            card.source_changed.connect(self._update_preview_button)
            self._cards.append(card)
            self.face_layout.addWidget(card)
        if self._cards:
            primary = max(self._cards, key=lambda card: card.person.face.area)
            self._selected_index = primary.person.index
        self._on_mode_changed()
        self.face_layout.addStretch(1)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self._set_busy(False)
        labels = ", ".join(face_label(card.person.index, card.person.face.gender) for card in self._cards)
        detail = f" ({labels})" if labels else ""
        self._set_status(
            f"Detected {len(self._cards)} face(s) at {format_timestamp(self._timestamp())}{detail}."
        )

    def _on_detect_failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self._set_busy(False)
        QMessageBox.critical(self, "Detection failed", message)

    def _pick_single_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Source face", "", _IMAGE_FILTER)
        if not path:
            return
        self._single_source = Path(path)
        self.single_label.setText(self._single_source.name)
        self._update_preview_button()

    def _preview_inputs_ready(self) -> bool:
        """True when this frame can be swapped without starting an export."""
        if self._busy or self._video is None:
            return False
        if self._single_mode():
            if self._single_source is None:
                return False
            if self.apply_all.isChecked():
                return True
            return self._selected_face() is not None
        return any(card.source_path is not None for card in self._cards)

    def _update_preview_button(self) -> None:
        self.preview_btn.setEnabled(self._preview_inputs_ready())

    def _compose_request(self, output_path: Optional[Path] = None) -> Optional[SwapRequest]:
        """Build the mapping request shared by preview and full export."""
        if self._video is None:
            QMessageBox.information(self, "No video", "Choose a target video first.")
            return None
        if output_path is None:
            output_path = Path(self.output_edit.text().strip())
        sources: list[FaceSource] = []
        single: Optional[Path] = None
        selected = None
        apply_to_all = False
        if self._single_mode():
            if self._single_source is None:
                QMessageBox.warning(self, "No source", "Choose the source image to swap in.")
                return None
            single = self._single_source
            apply_to_all = self.apply_all.isChecked()
            if not apply_to_all:
                selected = self._selected_face()
                if selected is None:
                    QMessageBox.warning(self, "No face selected", "Detect faces, then select the face to replace.")
                    return None
        else:
            for card in self._cards:
                if card.source_path is None:
                    continue
                sources.append(
                    FaceSource(
                        face=card.person.face,
                        source_path=card.source_path,
                        label=face_label(card.person.index, card.person.face.gender),
                    )
                )
            if not sources:
                QMessageBox.warning(self, "No source", "Choose a source image for at least one face.")
                return None
        return SwapRequest(
            video_path=self._video,
            output_path=output_path,
            execution=str(self.execution.currentData()),
            enhance=self.enhance.isChecked(),
            similarity=self.similarity.value() / 100.0,
            coverage=str(self.coverage.currentData()),
            keep_audio=self.keep_audio.isChecked(),
            crf=self.crf.value(),
            preset=self.preset.currentText(),
            scale=float(self.speed.currentData()),
            face_mode=str(self.face_mode.currentData()),
            single_source=single,
            selected_face=selected,
            apply_to_all=apply_to_all,
            face_sources=sources,
            precise_edges=self.precise_edges.isChecked(),
            object_mask=self.object_mask.isChecked(),
            fast_draft_preview=self.fast_draft.isChecked(),
            detect_every_other=self.detect_every.isChecked(),
            min_face_px=self.min_face.value(),
            rotation_mode=str(self.rotation.currentData()),
            rotation_seconds=float(self.rotation_seconds.value()),
            rotation_paths=list(self._rotation_paths),
        )

    def _preview_swap(self) -> None:
        if self._busy:
            return
        request = self._compose_request(Path("preview-only.mp4"))
        if request is None:
            return
        try:
            validate_request(request)
        except VideoTooLongError as exc:
            QMessageBox.critical(self, "Video is too long", str(exc))
            return
        except VideoDurationUnknownError as exc:
            QMessageBox.critical(self, "Duration unknown", str(exc))
            return
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot preview", str(exc))
            self._set_status(str(exc))
            return
        self._set_busy(True, cancellable=False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Previewing this frame…")
        self._set_status("Swapping the sample frame…")
        self.worker.request_preview(PreviewRequest(timestamp_s=self._timestamp(), swap=request))

    def _on_preview_ready(self, original, swapped, note: str, unchanged: bool) -> None:
        self._showing_swap = True
        self._preview_original = original
        self._preview_swapped = swapped
        self.compare_slider.setEnabled(True)
        self._paint_compare()
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if not unchanged else 0)
        self.progress.setFormat("Preview" if not unchanged else "Idle")
        self._set_busy(False)
        self._set_status(note)
        if unchanged:
            QMessageBox.warning(self, "Preview did not swap a face", note)

    def _paint_compare(self) -> None:
        if self._preview_original is None or self._preview_swapped is None:
            return
        image = wipe_preview(
            self._preview_original,
            self._preview_swapped,
            self.compare_slider.value() / 100.0,
        )
        self._preview_bgr = image
        self._paint_preview(image)

    def _on_compare(self, _value: int) -> None:
        if self._showing_swap:
            self._paint_compare()

    def _apply_fast_draft(self) -> None:
        self.enhance.setChecked(False)
        self.precise_edges.setChecked(False)
        self.object_mask.setChecked(True)
        self.fast_draft.setChecked(True)
        self._set_status("Fast draft: restore off, precise edges off, object mask on.")

    def _add_rotation_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Rotation source", "", _IMAGE_FILTER)
        if not path:
            return
        self._rotation_paths.append(Path(path))
        names = ", ".join(item.name for item in self._rotation_paths)
        self.rotation_label.setText(f"Rotation sources: {names}")

    def _project_payload(self) -> dict:
        sources = []
        if self._single_source is not None:
            gender = None
            face = self._selected_face()
            if face is not None:
                gender = face.gender
            sources.append({
                "path": str(self._single_source),
                "role": "single",
                "gender": gender_mark(gender),
                "label": self.single_label.text(),
            })
        for card in self._cards:
            if card.source_path is None:
                continue
            sources.append({
                "path": str(card.source_path),
                "role": "face",
                "index": card.person.index,
                "gender": gender_mark(card.person.face.gender),
                "label": card.title.text(),
            })
        for path in self._rotation_paths:
            sources.append({"path": str(path), "role": "rotation", "gender": "", "label": path.name})
        return {
            "video": str(self._video) if self._video else "",
            "output": self.output_edit.text().strip(),
            "sources": sources,
            "settings": {
                "execution": self.execution.currentData(),
                "similarity": self.similarity.value(),
                "coverage": self.coverage.currentData(),
                "enhance": self.enhance.isChecked(),
                "object_mask": self.object_mask.isChecked(),
                "precise_edges": self.precise_edges.isChecked(),
                "fast_draft_preview": self.fast_draft.isChecked(),
                "detect_every_other": self.detect_every.isChecked(),
                "min_face_px": self.min_face.value(),
                "rotation_mode": self.rotation.currentData(),
                "rotation_seconds": self.rotation_seconds.value(),
                "face_mode": self.face_mode.currentData(),
                "apply_all": self.apply_all.isChecked(),
                "scale": self.speed.currentData(),
                "keep_audio": self.keep_audio.isChecked(),
                "crf": self.crf.value(),
                "preset": self.preset.currentText(),
            },
        }

    def _save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "videoswa.videoswaproj", f"Videoswa project (*{PROJECT_SUFFIX})"
        )
        if not path:
            return
        saved = save_project(Path(path), self._project_payload())
        self._set_status(f"Saved project {saved}")

    def _load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project", "", f"Videoswa project (*{PROJECT_SUFFIX})"
        )
        if not path:
            return
        try:
            data = load_project(Path(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Could not load project", str(exc))
            return
        self._apply_project(data)
        self._set_status(f"Loaded project {path}")

    def _apply_project(self, data: dict) -> None:
        settings = data.get("settings") or {}
        self._set_combo(self.execution, settings.get("execution"))
        self._set_combo(self.coverage, settings.get("coverage"))
        self._set_combo(self.face_mode, settings.get("face_mode"))
        self._set_combo(self.rotation, settings.get("rotation_mode"))
        if settings.get("similarity") is not None:
            self.similarity.setValue(int(settings["similarity"]))
        if settings.get("min_face_px") is not None:
            self.min_face.setValue(int(settings["min_face_px"]))
        if settings.get("rotation_seconds") is not None:
            self.rotation_seconds.setValue(int(settings["rotation_seconds"]))
        if settings.get("crf") is not None:
            self.crf.setValue(int(settings["crf"]))
        if settings.get("preset"):
            self.preset.setCurrentText(str(settings["preset"]))
        self.enhance.setChecked(bool(settings.get("enhance", False)))
        self.object_mask.setChecked(bool(settings.get("object_mask", True)))
        self.precise_edges.setChecked(bool(settings.get("precise_edges", False)))
        self.fast_draft.setChecked(bool(settings.get("fast_draft_preview", True)))
        self.detect_every.setChecked(bool(settings.get("detect_every_other", True)))
        self.apply_all.setChecked(bool(settings.get("apply_all", False)))
        self.keep_audio.setChecked(bool(settings.get("keep_audio", True)))
        if settings.get("scale") is not None:
            self._set_combo(self.speed, settings.get("scale"))
        if data.get("output"):
            self.output_edit.setText(str(data["output"]))
        self._rotation_paths = []
        single = None
        for item in data.get("sources") or []:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            role = item.get("role")
            if role == "rotation":
                self._rotation_paths.append(Path(item["path"]))
            elif role == "single":
                single = Path(item["path"])
        if self._rotation_paths:
            self.rotation_label.setText(
                "Rotation sources: " + ", ".join(path.name for path in self._rotation_paths)
            )
        if single is not None and single.is_file():
            self._single_source = single
            mark = ""
            for item in data.get("sources") or []:
                if isinstance(item, dict) and item.get("role") == "single":
                    mark = item.get("gender") or ""
            self.single_label.setText(f"{single} {mark}".strip())
        video = data.get("video") or ""
        if video and Path(video).is_file():
            try:
                self.load_video(Path(video))
            except Exception as exc:
                self._set_status(str(exc))
        self._update_preview_button()

    @staticmethod
    def _set_combo(box: QComboBox, value) -> None:
        if value is None:
            return
        for index in range(box.count()):
            if box.itemData(index) == value:
                box.setCurrentIndex(index)
                return

    def _on_preview_failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self._set_busy(False)
        self._set_status(message)
        QMessageBox.warning(self, "Preview failed", message)

    def _run(self) -> None:
        if self._busy or self._video is None:
            if self._video is None:
                QMessageBox.information(self, "No video", "Choose a target video first.")
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "No output", "Choose where to save the MP4.")
            return
        request = self._compose_request()
        if request is None:
            return
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
        self._update_preview_button()

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

"""Background thread that owns InsightFace sessions.

ONNX Runtime GPU contexts are easiest to reuse on the thread that created
them, so detection and swapping both run here.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from faceswap.core import FaceSwapEngine, SwapStats
from faceswap.face_analyzer import FaceAnalyzer
from faceswap.providers import format_providers, provider_status_text, tensorrt_missing_note
from faceswap.quality import resolve_quality
from faceswap.swapper import FaceSwapper
from faceswap.utils import logger
from faceswap.video import SwapCancelled, process_video, read_frame_at, reset_stats, swap_frame
from videoswa.images import crop_face
from videoswa.jobs import MIN_ROI_CHANGE, SwapRequest, build_mappings, read_image, roi_mean_change


def _quality_note(stats: SwapStats) -> str:
    """Tell the user why a swap still looks like the original, or looks soft."""
    if stats.faces_swapped <= 0:
        return (
            "No faces were swapped, so the video still looks like the original. "
            "Move the sample slider until a face is listed, use a clearer source photo, "
            "or lower Min face size and the match threshold."
        )
    if stats.faces_detected <= 0:
        return (
            "No faces were detected. Move the slider, or lower Min face size and Detect sensitivity."
        )
    if stats.faces_swapped < 0.7 * stats.faces_detected and stats.faces_unmatched > stats.faces_swapped:
        return (
            "Many frames kept the original face. Lower the match threshold, "
            "or turn off Match gender if the wrong gender tag is blocking the swap. "
            "Raise the threshold if the wrong person is swapped."
        )
    return "Audio is kept when Keep audio is on and FFmpeg can read the file."


def seek_times(current_s: float, duration_s: float, limit: int = 8) -> list[float]:
    """Nearby timestamps to try when the current frame has no face."""
    duration = max(0.0, float(duration_s))
    current = min(max(0.0, float(current_s)), duration)
    offsets = (0.0, 0.4, -0.4, 0.8, -0.8, 1.5, -1.5, 2.5, -2.5, 4.0, -4.0, 8.0, -8.0)
    raw = [current + offset for offset in offsets]
    raw.append(duration * 0.5)
    raw.append(max(0.0, duration - 0.2))
    times: list[float] = []
    for value in raw:
        if value < -1e-3 or value > duration + 1e-3:
            continue
        stamped = round(min(max(0.0, value), duration), 3)
        if stamped not in times:
            times.append(stamped)
        if len(times) >= limit:
            break
    return times or [current]


@dataclass
class DetectRequest:
    video_path: Path
    timestamp_s: float
    execution: str
    duration_s: float = 0.0
    det_size: int = 640
    det_thresh: float = 0.30
    auto_seek: bool = True


@dataclass
class SourceCheckRequest:
    path: Path
    execution: str
    det_size: int = 640
    det_thresh: float = 0.30
    role: str = "library"


@dataclass
class PreviewRequest:
    """One sample frame, using the same mapping settings as a full export."""

    timestamp_s: float
    swap: SwapRequest


@dataclass
class DetectedPerson:
    crop_bgr: object
    face: object
    index: int


class EngineWorker(QThread):
    detect_ready = Signal(object)
    detect_failed = Signal(str)
    preview_ready = Signal(object, object, str, bool)
    preview_failed = Signal(str)
    source_ready = Signal(str, bool, str, str)
    swap_progress = Signal(int, int)
    swap_finished = Signal(str, str)
    swap_failed = Signal(str)
    swap_cancelled = Signal()
    status = Signal(str)
    provider = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._engine: Optional[FaceSwapEngine] = None
        self._engine_key: Optional[tuple] = None

    def request_detect(self, job: DetectRequest) -> None:
        self._queue.put(("detect", job))

    def request_preview(self, job: PreviewRequest) -> None:
        self._queue.put(("preview", job))

    def request_source(self, job: SourceCheckRequest) -> None:
        self._queue.put(("source", job))

    def request_swap(self, job: SwapRequest) -> None:
        self._cancel.clear()
        self._queue.put(("swap", job))

    def request_cancel(self) -> None:
        self._cancel.set()

    def shutdown(self) -> None:
        self._cancel.set()
        self._queue.put(None)

    def run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind, job = item
            try:
                if kind == "detect":
                    self._detect(job)
                elif kind == "preview":
                    self._preview(job)
                elif kind == "source":
                    self._check_source(job)
                else:
                    self._swap(job)
            except SwapCancelled:
                self.swap_cancelled.emit()
            except Exception as exc:
                logger.exception("Videoswa %s failed", kind)
                message = str(exc) or exc.__class__.__name__
                if kind == "detect":
                    self.detect_failed.emit(message)
                elif kind == "preview":
                    self.preview_failed.emit(message)
                elif kind == "source":
                    self.source_ready.emit(str(getattr(job, "path", "")), False, message, getattr(job, "role", "library"))
                else:
                    self.swap_failed.emit(message)

    def _engine_for(self, execution: str, enhance: bool) -> FaceSwapEngine:
        if self._engine is not None and self._engine_key == execution:
            self._bind_cuda_status(self._engine)
            tip = self._engine.swapper.set_enhance(enhance)
            if tip:
                self.status.emit(tip)
            self._emit_provider(self._engine)
            return self._engine
        self.status.emit(
            "Loading InsightFace models. The first run downloads buffalo_l and inswapper_128."
        )
        use_gpu = execution != "cpu"
        analyzer = FaceAnalyzer(use_gpu=use_gpu, execution=execution)
        swapper = FaceSwapper(use_gpu=use_gpu, enhance=enhance, execution=execution)
        self._engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper)
        self._engine_key = execution
        self._bind_cuda_status(self._engine)
        self._emit_provider(self._engine)
        return self._engine

    def _bind_cuda_status(self, engine: FaceSwapEngine) -> None:
        guard = getattr(engine, "cuda_guard", None)
        if guard is None:
            return

        def _report(message: str) -> None:
            self.status.emit(message)
            self.provider.emit(message)

        guard.on_fallback = _report

    def _emit_provider(self, engine: FaceSwapEngine) -> None:
        guard = getattr(engine, "cuda_guard", None)
        note = getattr(guard, "note", None) if guard is not None else None
        if note:
            self.provider.emit(note)
            self.status.emit(note)
            return
        names = getattr(engine.swapper, "active_providers", None) or getattr(engine.swapper, "providers", [])
        text = provider_status_text(names)
        mode = getattr(engine.swapper, "execution", "auto")
        if mode in {"auto", "tensorrt"}:
            text += tensorrt_missing_note(names)
        detail = format_providers(getattr(engine.swapper, "providers", []) or [])
        occlusion = getattr(engine.swapper, "_occlusion", None)
        if occlusion is not None and getattr(occlusion, "status", ""):
            text = f"{text} {occlusion.status}"
        self.provider.emit(text)
        self.status.emit(f"Models ready. {text} ({detail})" if detail else f"Models ready. {text}")

    def _apply_detector(self, engine: FaceSwapEngine, det_size: int, det_thresh: float) -> None:
        setter = getattr(engine.analyzer, "set_detection", None)
        if callable(setter):
            setter((int(det_size), int(det_size)), float(det_thresh))

    def _detect(self, job: DetectRequest) -> None:
        engine = self._engine_for(job.execution, enhance=False)
        self._apply_detector(engine, job.det_size, job.det_thresh)
        self.status.emit("Detecting faces…")
        times = seek_times(job.timestamp_s, job.duration_s) if job.auto_seek else [float(job.timestamp_s)]
        found_time = float(job.timestamp_s)
        frame = None
        faces = []
        for timestamp in times:
            try:
                candidate = read_frame_at(Path(job.video_path), timestamp)
            except Exception:
                continue
            faces = engine.analyzer.analyze(candidate)
            frame = candidate
            if faces:
                found_time = timestamp
                break
        people = []
        if frame is not None:
            people = [
                DetectedPerson(crop_bgr=crop_face(frame, face), face=face, index=index)
                for index, face in enumerate(faces)
            ]
        note = getattr(engine.analyzer, "last_detect_note", "") or ""
        if not people:
            note = (
                "No faces in this part of the video. Move the slider to a clearer shot, "
                "pick a frame where the face is larger, or lower Min face size and Detect sensitivity."
            )
            if "OpenCV" in (getattr(engine.analyzer, "last_detect_note", "") or ""):
                note = f"{note}\n\n{engine.analyzer.last_detect_note}"
        elif abs(found_time - float(job.timestamp_s)) > 0.05:
            parked = f"Moved the slider to {found_time:.1f}s because the current frame had no face."
            note = f"{note} {parked}".strip()
        self.detect_ready.emit({
            "people": people,
            "time_s": found_time,
            "note": note,
            "found": bool(people),
        })
        self.status.emit(note or f"Detected {len(people)} face(s).")

    def _check_source(self, job: SourceCheckRequest) -> None:
        engine = self._engine_for(job.execution, enhance=False)
        self._apply_detector(engine, job.det_size, job.det_thresh)
        path = Path(job.path)
        image = read_image(path)
        face = engine.analyzer.best_face(image)
        if face is None:
            self.source_ready.emit(
                str(path),
                False,
                (
                    f"No face found in {path.name}. Use a clear photo of one person "
                    "(JPG, PNG, or HEIC). If the face is small or turned, lower Min face size "
                    "and Detect sensitivity, then add the photo again."
                ),
                job.role,
            )
            return
        self.source_ready.emit(str(path), True, f"Source ready: {path.name}", job.role)

    def _configure(self, engine: FaceSwapEngine, job: SwapRequest, *, preview: bool) -> None:
        enhance, precise, object_mask = resolve_quality(
            enhance=job.enhance,
            precise_edges=job.precise_edges,
            object_mask=job.object_mask,
            fast_draft=job.fast_draft_preview,
            preview=preview,
        )
        reset_stats(engine)
        engine.similarity_threshold = job.similarity
        engine.coverage = job.coverage
        engine.min_face_px = int(job.min_face_px)
        engine.detect_stride = 1 if preview else (2 if job.detect_every_other else 1)
        engine.rotation.mode = job.rotation_mode
        engine.rotation.seconds = float(job.rotation_seconds)
        engine.rotation.sources = [] if job.rotation_mode == "per_person" else self._rotation_faces(engine, job)
        engine.match_gender = bool(getattr(job, "match_gender", False))
        swapper = engine.swapper
        swapper.object_mask = object_mask
        swapper.precise_edges = precise
        swapper.allow_restore = enhance
        swapper.color_match = float(getattr(job, "color_match", 0.60))
        swapper.sharpen = float(getattr(job, "sharpen", 0.0))
        swapper.restore_strength = float(getattr(job, "restore_strength", 1.0))
        self._apply_detector(engine, int(getattr(job, "det_size", 640)), float(getattr(job, "det_thresh", 0.30)))
        if enhance and hasattr(swapper, "set_enhance"):
            tip = swapper.set_enhance(True)
            if tip:
                self.status.emit(tip)

    def _rotation_faces(self, engine: FaceSwapEngine, job: SwapRequest):
        faces = []
        paths = list(job.rotation_paths)
        if job.single_source is not None:
            paths.insert(0, Path(job.single_source))
        paths.extend(item.source_path for item in job.face_sources)
        seen: set[str] = set()
        for path in paths:
            key = str(Path(path))
            if key in seen:
                continue
            seen.add(key)
            image = read_image(Path(path))
            face = engine.analyzer.best_face(image)
            if face is None:
                raise ValueError(f"No face found in rotation source '{Path(path).name}'.")
            faces.append(face)
        return faces

    def _preview(self, job: PreviewRequest) -> None:
        """Swap one sample frame and return it. Does not encode a video."""
        draft = job.swap.fast_draft_preview
        engine = self._engine_for(job.swap.execution, enhance=False if draft else job.swap.enhance)
        self._configure(engine, job.swap, preview=True)
        self.status.emit("Swapping this frame…")
        mappings = build_mappings(engine, job.swap)
        frame = read_frame_at(Path(job.swap.video_path), job.timestamp_s)
        swapped = swap_frame(
            engine, frame, mappings, scale=job.swap.scale, time_s=job.timestamp_s
        )
        change = roi_mean_change(frame, swapped, getattr(engine, "last_boxes", []))
        unchanged = engine.stats.faces_swapped == 0 or change < MIN_ROI_CHANGE
        if engine.stats.faces_swapped == 0:
            note = (
                "No face was swapped on this frame, so the preview still looks like the original. "
                "Move the slider until a face is listed, use a clearer source photo, "
                "or lower the match threshold and Min face size."
            )
        elif change < MIN_ROI_CHANGE:
            note = (
                "The preview barely changed inside the face, so this is not a usable swap. "
                "Move the slider to a clearer face, pick a clearer source photo, "
                "or lower Min face size and the match threshold."
            )
        else:
            note = (
                "Preview ready. The face identity changed. Drag Before / after, "
                "then press Swap to export the video with audio."
            )
        self.preview_ready.emit(frame, swapped, note, unchanged)

    def _swap(self, job: SwapRequest) -> None:
        engine = self._engine_for(job.execution, job.enhance)
        self._configure(engine, job, preview=False)
        self.status.emit("Building face mappings…")
        mappings = build_mappings(engine, job)
        self.status.emit(f"Swapping {len(mappings)} mapping(s)…")

        def _progress(done: int, total: int) -> None:
            self.swap_progress.emit(done, total)

        output = process_video(
            engine,
            mappings,
            input_path=Path(job.video_path),
            output_path=Path(job.output_path),
            progress=_progress,
            keep_audio=job.keep_audio,
            crf=job.crf,
            preset=job.preset,
            cancel_event=self._cancel,
            scale=job.scale,
            encoder=getattr(job, "encoder", "auto"),
        )
        stats = engine.stats
        summary = (
            f"Frames {stats.frames} · faces detected {stats.faces_detected} · "
            f"swapped {stats.faces_swapped} · unmatched {stats.faces_unmatched}"
        )
        if stats.faces_held:
            summary += f" · held through {stats.faces_held} softer frame(s)"
        summary += " " + _quality_note(stats)
        self.swap_finished.emit(str(output), summary)

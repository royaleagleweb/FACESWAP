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
from videoswa.jobs import SwapRequest, build_mappings, read_image


def _quality_note(stats: SwapStats) -> str:
    """Tell the user why a swap still looks like the original, or looks soft."""
    if stats.faces_detected <= 0:
        return (
            "No faces were detected. Sample a frame where the face is clear and frontal."
        )
    if stats.faces_swapped < 0.7 * stats.faces_detected and stats.faces_unmatched > stats.faces_swapped:
        return (
            "Many frames kept the original face. Use a clear frontal sample, "
            "or lower the match threshold. Raise it if the wrong person is swapped."
        )
    return (
        "If the swap looks soft, enable Sharpen swapped faces (GFPGAN). It stays off until you turn it on."
    )


@dataclass
class DetectRequest:
    video_path: Path
    timestamp_s: float
    execution: str


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

    def _detect(self, job: DetectRequest) -> None:
        engine = self._engine_for(job.execution, enhance=False)
        self.status.emit("Detecting faces…")
        frame = read_frame_at(Path(job.video_path), job.timestamp_s)
        faces = engine.analyzer.analyze(frame)
        people = [
            DetectedPerson(crop_bgr=crop_face(frame, face), face=face, index=index)
            for index, face in enumerate(faces)
        ]
        self.detect_ready.emit(people)
        self.status.emit(f"Detected {len(people)} face(s).")

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
        swapper = engine.swapper
        swapper.object_mask = object_mask
        swapper.precise_edges = precise
        swapper.allow_restore = enhance
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
        unchanged = engine.stats.faces_swapped == 0
        if unchanged:
            note = (
                "No face was swapped on this frame, so the preview still looks like the original. "
                "Use a clearer sample, or lower the match threshold."
            )
        else:
            note = "Preview ready. Drag Before / after to compare. Run swap writes the video."
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

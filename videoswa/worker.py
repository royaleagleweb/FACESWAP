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
from faceswap.providers import format_providers, provider_status_text
from faceswap.swapper import FaceSwapper
from faceswap.utils import logger
from faceswap.video import SwapCancelled, process_video, read_frame_at, reset_stats
from videoswa.images import crop_face
from videoswa.jobs import SwapRequest, build_mappings


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
class DetectedPerson:
    crop_bgr: object
    face: object
    index: int


class EngineWorker(QThread):
    detect_ready = Signal(object)
    detect_failed = Signal(str)
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
                else:
                    self._swap(job)
            except SwapCancelled:
                self.swap_cancelled.emit()
            except Exception as exc:
                logger.exception("Videoswa %s failed", kind)
                message = str(exc) or exc.__class__.__name__
                if kind == "detect":
                    self.detect_failed.emit(message)
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
        detail = format_providers(getattr(engine.swapper, "providers", []) or [])
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

    def _swap(self, job: SwapRequest) -> None:
        engine = self._engine_for(job.execution, job.enhance)
        engine.similarity_threshold = job.similarity
        engine.coverage = job.coverage
        reset_stats(engine)
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

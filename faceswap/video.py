"""Video I/O. Decode/encode through OpenCV and preserve original audio."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from .core import FaceMapping, FaceSwapEngine, SwapStats
from .utils import OUTPUTS_DIR, TEMP_DIR, logger

# Videoswa refuses anything longer than this before a swap starts.
MAX_VIDEO_SECONDS = 5 * 60
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoTooLongError(ValueError):
    """The target video is longer than Videoswa allows."""

    def __init__(self, duration_s: float, limit_s: float = MAX_VIDEO_SECONDS) -> None:
        self.duration_s = duration_s
        self.limit_s = limit_s
        super().__init__(
            f"This video is {format_duration_long(duration_s)} long. "
            f"Videoswa only accepts videos up to {format_duration_long(limit_s)}. "
            "Choose a shorter file. Nothing was processed."
        )


class VideoDurationUnknownError(ValueError):
    """Duration could not be read, so the 5-minute limit could not be applied."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        super().__init__(
            f"Could not determine how long '{self.path.name}' is. "
            "Videoswa will not start a swap without a duration, because videos "
            "longer than 5 minutes are rejected. Install FFmpeg (ffprobe) and "
            "try again, or re-encode the file to MP4."
        )


class SwapCancelled(Exception):
    """The user cancelled a swap before the output file was written."""


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool
    duration_s: float


def format_duration_long(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    if secs or not parts:
        parts.append(f"{secs} second" + ("s" if secs != 1 else ""))
    return " ".join(parts)


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _ffprobe_duration(path: Path) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    if not out or out.upper() == "N/A":
        return None
    try:
        value = float(out)
    except ValueError:
        return None
    return value if value > 0 else None


def _ffprobe_has_audio(path: Path) -> bool:
    if not shutil.which("ffprobe"):
        return False
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0", str(path),
            ],
            stderr=subprocess.STDOUT,
        ).decode().strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False


def _opencv_meta(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return width, height, fps, frame_count


def _duration_seconds(path: Path, fps: float, frame_count: int) -> float:
    probed = _ffprobe_duration(path)
    if probed is not None:
        return probed
    if fps > 1e-3 and frame_count > 0:
        return frame_count / fps
    raise VideoDurationUnknownError(path)


def probe(path: Path) -> VideoInfo:
    width, height, fps, frame_count = _opencv_meta(path)
    duration_s = _duration_seconds(path, fps, frame_count)
    return VideoInfo(
        width=width,
        height=height,
        fps=fps if fps > 1e-3 else 30.0,
        frame_count=frame_count,
        has_audio=_ffprobe_has_audio(path),
        duration_s=duration_s,
    )


def assert_duration_allowed(
    path: Path,
    limit_s: Optional[float] = None,
) -> VideoInfo:
    """Open ``path`` and reject it before any swap work if it is too long."""
    if limit_s is None:
        limit_s = MAX_VIDEO_SECONDS
    info = probe(path)
    # A few milliseconds of container rounding must not reject a 5:00 file.
    if info.duration_s > limit_s + 0.05:
        raise VideoTooLongError(info.duration_s, limit_s)
    return info


def read_frame_at(path: Path, timestamp_s: float) -> np.ndarray:
    """Return one BGR frame at ``timestamp_s``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        target = max(0.0, float(timestamp_s))
        cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(fps * target))))
            ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame at {format_timestamp(timestamp_s)}")
    return frame


def grab_first_frame(path: Path) -> Optional[np.ndarray]:
    try:
        return read_frame_at(path, 0.0)
    except RuntimeError:
        return None


def resolve_encoder(choice: str = "auto") -> str:
    """Prefer NVENC when ffmpeg lists it. Anything else stays on libx264."""
    name = (choice or "auto").strip().lower()
    if name in {"auto", "h264_nvenc", "nvenc"} and _ffmpeg_has_encoder("h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


def _ffmpeg_has_encoder(name: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return name in (result.stdout or "")


def process_video(
    engine: FaceSwapEngine,
    mappings: Sequence[FaceMapping],
    input_path: Path,
    output_path: Optional[Path] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    keep_audio: bool = True,
    crf: int = 18,
    preset: str = "medium",
    cancel_event: Optional[threading.Event] = None,
    limit_s: Optional[float] = None,
    scale: float = 1.0,
    encoder: str = "auto",
) -> Path:
    """Apply face swaps frame-by-frame and re-mux original audio.

    Duration is checked before the first frame is written. ``cancel_event``
    stops the loop between frames and does not leave an output file behind.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = OUTPUTS_DIR / f"{input_path.stem}_videoswa.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if limit_s is None:
        limit_s = MAX_VIDEO_SECONDS
    info = assert_duration_allowed(input_path, limit_s=limit_s)
    logger.info(
        "Video: %dx%d @ %.2ffps, %.2fs, %d frames, audio=%s",
        info.width, info.height, info.fps, info.duration_s, info.frame_count, info.has_audio,
    )

    expected = info.frame_count
    if expected <= 0 and info.fps > 0:
        expected = max(1, int(round(info.duration_s * info.fps)))
    reset_tracks = getattr(engine, "reset_tracks", None)
    if callable(reset_tracks):
        reset_tracks()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    cancelled = False
    with tempfile.TemporaryDirectory(dir=TEMP_DIR) as tmpdir:
        tmp_video = Path(tmpdir) / "swapped_silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_video), fourcc, info.fps, (info.width, info.height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not open video writer (mp4v)")

        bar = tqdm(
            total=expected or None,
            desc="Swapping",
            unit="f",
            disable=progress is not None,
        )
        idx = 0
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                time_s = (idx / info.fps) if info.fps else 0.0
                out = swap_frame(engine, frame, mappings, scale=scale, time_s=time_s)
                if out.shape[1] != info.width or out.shape[0] != info.height:
                    out = cv2.resize(out, (info.width, info.height))
                writer.write(out)
                idx += 1
                bar.update(1)
                if progress is not None:
                    progress(idx, expected)
        finally:
            cap.release()
            writer.release()
            bar.close()

        if cancelled:
            raise SwapCancelled("Swap cancelled. No output file was written.")

        chosen = resolve_encoder(encoder)
        if keep_audio and info.has_audio and shutil.which("ffmpeg"):
            _mux_audio(tmp_video, input_path, output_path, crf=crf, preset=preset, encoder=chosen)
        elif shutil.which("ffmpeg"):
            _reencode(tmp_video, output_path, crf=crf, preset=preset, encoder=chosen)
        else:
            shutil.copy2(tmp_video, output_path)
            logger.warning("ffmpeg not found; output will not contain audio and may be mpeg4.")

    logger.info("Wrote %s", output_path)
    logger.info("Stats: %s", engine.stats)
    return output_path


def swap_frame(
    engine: FaceSwapEngine,
    frame: np.ndarray,
    mappings,
    scale: float = 1.0,
    time_s: float = 0.0,
) -> np.ndarray:
    """Swap one frame. ``scale`` below 1 runs the model on a smaller image.

    The returned frame is always the original size. Half resolution is the
    faster export path when inference is on CPU or DirectML.
    """
    scale = float(scale or 1.0)
    height, width = frame.shape[:2]
    if scale >= 0.99 or width < 4 or height < 4:
        return engine.process_frame(frame, mappings, time_s=time_s)
    small_w = max(2, int(round(width * scale)))
    small_h = max(2, int(round(height * scale)))
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    swapped = engine.process_frame(small, mappings, time_s=time_s)
    if swapped.shape[1] != width or swapped.shape[0] != height:
        swapped = cv2.resize(swapped, (width, height), interpolation=cv2.INTER_LINEAR)
    return swapped


def _video_encode_args(encoder: str, crf: int, preset: str) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-cq", str(crf), "-preset", "p4", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _mux_audio(
    silent_video: Path,
    original: Path,
    dest: Path,
    crf: int,
    preset: str,
    encoder: str = "libx264",
) -> None:
    def _cmd(enc: str) -> list[str]:
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent_video), "-i", str(original),
            "-map", "0:v:0", "-map", "1:a:0?",
            *_video_encode_args(enc, crf, preset),
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(dest),
        ]

    try:
        _run_ffmpeg(_cmd(encoder))
    except subprocess.CalledProcessError:
        if encoder == "libx264":
            raise
        logger.warning("Encoder %s failed. Falling back to libx264.", encoder)
        _run_ffmpeg(_cmd("libx264"))


def _reencode(src: Path, dest: Path, crf: int, preset: str, encoder: str = "libx264") -> None:
    def _cmd(enc: str) -> list[str]:
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            *_video_encode_args(enc, crf, preset),
            str(dest),
        ]

    try:
        _run_ffmpeg(_cmd(encoder))
    except subprocess.CalledProcessError:
        if encoder == "libx264":
            raise
        logger.warning("Encoder %s failed. Falling back to libx264.", encoder)
        _run_ffmpeg(_cmd("libx264"))


def process_image(
    engine: FaceSwapEngine,
    mappings: Sequence[FaceMapping],
    input_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    img = cv2.imread(str(input_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {input_path}")
    out = engine.process_frame(img, mappings)
    if output_path is None:
        output_path = OUTPUTS_DIR / f"{Path(input_path).stem}_videoswa.png"
    cv2.imwrite(str(output_path), out)
    return Path(output_path)


def reset_stats(engine: FaceSwapEngine) -> SwapStats:
    engine.stats = SwapStats()
    reset_tracks = getattr(engine, "reset_tracks", None)
    if callable(reset_tracks):
        reset_tracks()
    return engine.stats

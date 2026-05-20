"""Video I/O. Decode/encode through ffmpeg and preserve original audio."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from .core import FaceMapping, FaceSwapEngine
from .utils import OUTPUTS_DIR, TEMP_DIR, logger


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    has_audio = _ffprobe_has_audio(path)
    return VideoInfo(width=width, height=height, fps=fps, frame_count=n, has_audio=has_audio)


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


def grab_first_frame(path: Path) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def process_video(
    engine: FaceSwapEngine,
    mappings: Sequence[FaceMapping],
    input_path: Path,
    output_path: Optional[Path] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    keep_audio: bool = True,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Apply face swaps frame-by-frame and re-mux original audio."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = OUTPUTS_DIR / f"{input_path.stem}_swapped.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe(input_path)
    logger.info(
        "Video: %dx%d @ %.2ffps, %d frames, audio=%s",
        info.width, info.height, info.fps, info.frame_count, info.has_audio,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    with tempfile.TemporaryDirectory(dir=TEMP_DIR) as tmpdir:
        tmp_video = Path(tmpdir) / "swapped_silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_video), fourcc, info.fps, (info.width, info.height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not open video writer (mp4v)")

        bar = tqdm(total=info.frame_count or None, desc="Swapping", unit="f")
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out = engine.process_frame(frame, mappings)
                writer.write(out)
                idx += 1
                bar.update(1)
                if progress is not None:
                    progress(idx, info.frame_count)
        finally:
            cap.release()
            writer.release()
            bar.close()

        if keep_audio and info.has_audio and shutil.which("ffmpeg"):
            _mux_audio(tmp_video, input_path, output_path, crf=crf, preset=preset)
        elif shutil.which("ffmpeg"):
            _reencode(tmp_video, output_path, crf=crf, preset=preset)
        else:
            shutil.copy2(tmp_video, output_path)
            logger.warning("ffmpeg not found; output will not contain audio.")

    logger.info("Wrote %s", output_path)
    logger.info("Stats: %s", engine.stats)
    return output_path


def _mux_audio(silent_video: Path, original: Path, dest: Path, crf: int, preset: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(silent_video), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


def _reencode(src: Path, dest: Path, crf: int, preset: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


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
        output_path = OUTPUTS_DIR / f"{Path(input_path).stem}_swapped.png"
    cv2.imwrite(str(output_path), out)
    return Path(output_path)

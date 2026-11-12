"""Headless check: detect a photo, swap the first 30 frames, write proof files.

    python -m videoswa --selftest face.jpg clip.mp4
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path

import cv2
import numpy as np

from faceswap.diag import configure_logging, log_file_path
from faceswap.video import probe

logger = logging.getLogger("videoswa.selftest")


def run_selftest(argv: list[str]) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="videoswa --selftest")
    parser.add_argument("photo", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out or (log_file_path().parent / "selftest")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"log: {log_file_path()}"]
    ok = True
    try:
        ok = _run(args.photo, args.video, max(1, int(args.frames)), out_dir, lines) and ok
    except Exception:
        ok = False
        text = traceback.format_exc()
        logger.error("selftest crashed\n%s", text)
        lines.append("CRASH")
        lines.append(text.strip())
    report = "\n".join(lines)
    print(report)
    logger.info("selftest report\n%s", report)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _run(photo: Path, video: Path, frames: int, out_dir: Path, lines: list[str]) -> bool:
    from faceswap.core import FaceSwapEngine
    from faceswap.face_analyzer import FaceAnalyzer
    from faceswap.swapper import FaceSwapper
    from faceswap.video import process_video, read_frame_index
    from videoswa.jobs import build_mappings, read_image
    from videoswa.jobs import SwapRequest

    ok = True
    if not photo.is_file():
        lines.append(f"photo: MISSING {photo}")
        return False
    if not video.is_file():
        lines.append(f"video: MISSING {video}")
        return False

    image = read_image(photo)
    lines.append(f"photo file: {photo} {image.shape[1]}x{image.shape[0]}")
    analyzer = FaceAnalyzer(use_gpu=True, execution="auto")
    logger.info(
        "detector providers: %s",
        getattr(analyzer, "active_providers", None),
    )
    face = analyzer.best_face(image)
    note = getattr(analyzer, "last_detect_note", "") or ""
    if face is None:
        lines.append(f"photo: NO FACE. {note}".strip())
        ok = False
    else:
        span = int(max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]))
        lines.append(f"photo: face score={face.det_score:.3f} size={span}px det={analyzer.det_size[0]}")
        if note:
            lines.append(f"photo note: {note}")

    info = probe(video)
    lines.append(
        f"video: {info.width}x{info.height} @ {info.fps:.2f}fps "
        f"{info.frame_count} frames audio={info.has_audio}"
    )
    print("Preparing GPU, first time only. TensorRT can take several minutes. Later runs use the cache.")
    lines.append("Preparing GPU, first time only. TensorRT can take several minutes. Later runs use the cache.")
    swapper = FaceSwapper(use_gpu=True, enhance=False, execution="auto")
    swapper.pixel_boost = 128
    swapper.precise_edges = False
    logger.info("inswapper providers: %s", getattr(swapper, "active_providers", None))
    logger.info("enhance: off for selftest (device would be logged if GFPGAN were loaded)")
    lines.append(f"provider: {getattr(swapper, 'active_providers', None)}")
    lines.append("enhance: off")
    lines.append("preset: balanced (128px, detect every 3rd frame, video det 320)")
    engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper, detect_stride=3)
    if face is None:
        lines.append("swap: skipped, source photo had no face")
        return False

    request = SwapRequest(
        video_path=video,
        output_path=out_dir / "selftest.mp4",
        single_source=photo,
        apply_to_all=True,
        face_mode="single",
        enhance=False,
        pixel_boost=128,
        detect_every_other=True,
        detect_stride=3,
        video_det_size=320,
        fast_draft_preview=True,
    )
    mappings = build_mappings(engine, request)
    import time as _time

    from faceswap.swapper import warm_models

    warm_started = _time.perf_counter()
    warm_models(engine.swapper)
    det_started = _time.perf_counter()
    try:
        engine.analyzer.set_detection(320)
        det_ms = (_time.perf_counter() - det_started) * 1000.0
        kind = "cache hit" if det_ms < 8000 else "built"
        lines.append(f"warmup_det320: {det_ms:.0f}ms once ({kind})")
        logger.info("video detector 320 %s in %.1fs", kind, det_ms / 1000.0)
    except Exception as exc:
        logger.info("detector 320 warmup skipped: %s", exc)
        lines.append(f"warmup_det320: skipped ({exc})")
    warmup_ms = (_time.perf_counter() - warm_started) * 1000.0
    lines.append(f"warmup: {warmup_ms:.0f}ms once (InSwapper batch-1 + XSeg + det 320, not in per-frame swap)")
    end_s = 0.0
    if info.fps > 0:
        end_s = min(float(info.duration_s), frames / float(info.fps))
    output = process_video(
        engine,
        mappings,
        video,
        output_path=out_dir / "selftest.mp4",
        keep_audio=True,
        end_s=end_s,
        encoder="libx264",
    )
    stats = engine.stats
    lines.append(
        f"swap: frames={stats.frames} detected={stats.faces_detected} "
        f"swapped={stats.faces_swapped} held_locks={stats.faces_held} "
        f"detector_frames={stats.detector_frames} cached_frames={stats.cached_frames} "
        f"unmatched={stats.faces_unmatched} wiped={stats.faces_wiped} "
        f"coverage={stats.coverage_ratio:.2f}"
    )
    lines.append(f"timings: {stats.stage_line()}")
    if stats.miss_log:
        lines.append("misses: " + " | ".join(stats.miss_log[:8]))
    lines.append(f"mp4: {output}")
    if stats.faces_swapped <= 0:
        lines.append("swap: FAIL no frame was swapped")
        ok = False

    original = read_frame_index(video, 0)
    cap = cv2.VideoCapture(str(output))
    grabbed, swapped = cap.read()
    cap.release()
    preview_path = out_dir / "preview.png"
    before_path = out_dir / "before.png"
    after_path = out_dir / "after.png"
    if original is not None:
        cv2.imwrite(str(before_path), original)
    if not grabbed or swapped is None:
        lines.append("preview: FAIL could not read the output")
        return False
    if swapped.shape[:2] != original.shape[:2]:
        swapped = cv2.resize(swapped, (original.shape[1], original.shape[0]))
    cv2.imwrite(str(after_path), swapped)
    gap = np.full((original.shape[0], 8, 3), 255, dtype=np.uint8)
    side = np.concatenate([original, gap, swapped], axis=1)
    cv2.imwrite(str(preview_path), side)
    delta = float(np.mean(np.abs(original.astype(np.int16) - swapped.astype(np.int16))))
    lines.append(f"preview: {preview_path} full-frame mean change {delta:.2f}")
    lines.append(f"before: {before_path}")
    lines.append(f"after: {after_path}")
    if delta < 0.4 and stats.faces_swapped > 0:
        lines.append("preview: WARN the first frame barely changed; a later frame may hold the face")
    return ok

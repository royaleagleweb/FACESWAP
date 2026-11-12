"""Time each preset on a short clip.

    python -m videoswa --bench face.jpg clip.mp4
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from faceswap.diag import configure_logging, log_file_path
from faceswap.quality import quality_preset
from faceswap.video import probe, process_video, reset_stats


def run_bench(argv: list[str]) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="videoswa --bench")
    parser.add_argument("photo", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--presets", default="laptop,balanced")
    args = parser.parse_args(argv)
    if not args.photo.is_file() or not args.video.is_file():
        print("FAIL missing photo or video")
        return 1
    print("Preparing GPU, first time only. TensorRT can take several minutes. Later runs use the cache.")
    try:
        return _run(args.photo, args.video, max(5, int(args.frames)), [part.strip() for part in args.presets.split(",") if part.strip()])
    except Exception:
        print(traceback.format_exc())
        print("FAIL")
        return 1


def _run(photo: Path, video: Path, frames: int, presets: list[str]) -> int:
    from faceswap.core import FaceSwapEngine
    from faceswap.face_analyzer import FaceAnalyzer
    from faceswap.swapper import FaceSwapper, warm_models
    from videoswa.jobs import SwapRequest, build_mappings

    info = probe(video)
    end_s = frames / float(info.fps) if info.fps else 1.0
    analyzer = FaceAnalyzer(use_gpu=True, execution="auto")
    if analyzer.best_face(_image(photo)) is None:
        print("FAIL no face in the photo")
        return 1
    swapper = FaceSwapper(use_gpu=True, enhance=False, execution="auto")
    warm_started = time.perf_counter()
    warm_models(swapper)
    try:
        det_ms = analyzer.warm_size(320)
        print(f"detector 320 forward {det_ms:.0f}ms ({'cache hit' if det_ms < 8000 else 'built'})")
    except Exception as exc:
        print(f"detector 320 skipped: {exc}")
    warm_s = time.perf_counter() - warm_started
    kind = "cache hit" if warm_s < 8.0 else "built"
    print(f"warmup {warm_s:.1f}s once ({kind}), not included in the fps below")
    print(f"log: {log_file_path()}")
    out_dir = log_file_path().parent / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in presets:
        chosen = quality_preset(name)
        swapper.pixel_boost = int(chosen["pixel_boost"])
        swapper.precise_edges = bool(chosen["precise_edges"])
        swapper.allow_restore = False
        swapper.sharpen = 0
        engine = FaceSwapEngine(
            analyzer=analyzer,
            swapper=swapper,
            detect_stride=int(chosen.get("detect_stride", 1)),
        )
        request = SwapRequest(
            video_path=video,
            output_path=out_dir / f"{name}.mp4",
            single_source=photo,
            apply_to_all=True,
            face_mode="single",
            enhance=False,
            pixel_boost=int(chosen["pixel_boost"]),
            detect_every_other=bool(chosen["detect_every_other"]),
            detect_stride=int(chosen.get("detect_stride", 1)),
            video_det_size=320,
        )
        mappings = build_mappings(engine, request)
        reset_stats(engine)
        started = time.perf_counter()
        process_video(
            engine,
            mappings,
            video,
            output_path=out_dir / f"{name}.mp4",
            keep_audio=False,
            end_s=end_s,
            encoder="libx264",
        )
        elapsed = time.perf_counter() - started
        stats = engine.stats
        counted = max(1, int(stats.frames))
        fps = counted / elapsed if elapsed > 0 else 0.0
        pipeline = _pipeline_ms(stats)
        print(
            f"{name}: {pipeline:.1f} ms/frame pipeline, {1000.0 / pipeline:.1f} fps from stages, "
            f"{fps:.1f} fps wall, frames={counted} det={stats.detector_frames} tracked={stats.cached_frames}"
        )
        print(f"  {stats.stage_line()}")
    print("enhance: off in --bench so GFPGAN is not in the number")
    return 0


def _image(photo: Path):
    from videoswa.jobs import read_image

    return read_image(photo)


def _pipeline_ms(stats) -> float:
    frames = max(1, int(stats.frames))
    total = sum(float(stats.stage_ms.get(name, 0.0)) for name in ("detect", "swap", "decode", "encode"))
    return total / frames

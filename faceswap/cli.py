"""Command-line entry point for Videoswa.

Examples
--------
Open the desktop window (also the default when no command is given):
    python -m videoswa
    python run.py

Single source onto every face in a video:
    python run.py swap -t input.mp4 -s alice.jpg -o out.mp4

Two sources mapped to two specific people (reference snapshots from the video):
    python run.py swap -t input.mp4 \\
        --pair alice.jpg=ref_alice.jpg \\
        --pair bob.jpg=ref_bob.jpg \\
        -o out.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .core import FaceSwapEngine
from .coverage import COVERAGE_CHOICES, DEFAULT_COVERAGE
from .face_analyzer import FaceAnalyzer
from .providers import EXECUTION_CHOICES
from .swapper import FaceSwapper
from .utils import logger
from .video import (
    VIDEO_SUFFIXES,
    VideoDurationUnknownError,
    VideoTooLongError,
    assert_duration_allowed,
    process_image,
    process_video,
)


def _parse_pairs(pairs: List[str]) -> List[Tuple[Path, Optional[Path]]]:
    out: List[Tuple[Path, Optional[Path]]] = []
    for raw in pairs:
        if "=" in raw:
            src, ref = raw.split("=", 1)
            out.append((Path(src), Path(ref) if ref else None))
        else:
            out.append((Path(raw), None))
    return out


def _read(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def cmd_swap(args: argparse.Namespace) -> int:
    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")

    pairs: List[Tuple[Path, Optional[Path]]] = []
    if args.source:
        pairs.append((Path(args.source), None))
    pairs.extend(_parse_pairs(args.pair or []))

    if not pairs:
        raise SystemExit("Provide --source or at least one --pair source=reference")

    execution = "cpu" if args.cpu else args.execution
    if target.suffix.lower() in VIDEO_SUFFIXES:
        try:
            assert_duration_allowed(target)
        except (VideoTooLongError, VideoDurationUnknownError) as exc:
            raise SystemExit(str(exc)) from exc

    analyzer = FaceAnalyzer(use_gpu=execution != "cpu", det_thresh=args.det_thresh, execution=execution)
    swapper = FaceSwapper(use_gpu=execution != "cpu", enhance=args.enhance, execution=execution)
    engine = FaceSwapEngine(
        analyzer=analyzer,
        swapper=swapper,
        similarity_threshold=args.similarity,
        coverage=args.coverage,
        apply_to_all_when_no_reference=args.apply_to_all,
    )

    mappings = engine.build_mappings([(_read(s), _read(r) if r else None) for s, r in pairs])

    suffix = target.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        out = process_video(
            engine,
            mappings,
            input_path=target,
            output_path=Path(args.output) if args.output else None,
            keep_audio=not args.no_audio,
            crf=args.crf,
            preset=args.preset,
        )
    else:
        out = process_image(
            engine,
            mappings,
            input_path=target,
            output_path=Path(args.output) if args.output else None,
        )
    print(out)
    return 0


def cmd_desktop(_args: Optional[argparse.Namespace] = None) -> int:
    from videoswa.desktop import main as desktop_main
    return desktop_main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="videoswa",
        description="Videoswa — multi-face video face swap",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("swap", help="Run a faceswap on an image or video")
    s.add_argument("-t", "--target", required=True, help="Target image or video")
    s.add_argument("-s", "--source", help="Single source face image (applied to all faces)")
    s.add_argument(
        "--pair", action="append",
        help="Source=Reference image pair. Repeat for multi-face swaps.",
    )
    s.add_argument("-o", "--output", help="Output path (default: outputs/<name>_videoswa.mp4)")
    s.add_argument("--similarity", type=float, default=0.45, help="Cosine threshold for reference match")
    s.add_argument(
        "--coverage",
        choices=list(COVERAGE_CHOICES),
        default=DEFAULT_COVERAGE,
        help="full covers jaw and beard (default); normal is the tight face oval",
    )
    s.add_argument("--det-thresh", type=float, default=0.5, help="Face detection threshold")
    s.add_argument(
        "--execution",
        choices=list(EXECUTION_CHOICES),
        default="auto",
        help="ONNX Runtime providers: auto is TensorRT, then CUDA, then CPU",
    )
    s.add_argument("--cpu", action="store_true", help="Force CPU inference")
    s.add_argument("--enhance", action="store_true", help="Run GFPGAN on swapped faces (optional extra)")
    s.add_argument("--no-audio", action="store_true", help="Drop original audio track")
    s.add_argument("--crf", type=int, default=18, help="x264 CRF (lower = better quality)")
    s.add_argument("--preset", default="medium", help="x264 preset")
    s.add_argument(
        "--apply-to-all", action="store_true",
        help="Also apply a wildcard source to faces that miss every reference",
    )
    s.set_defaults(func=cmd_swap)

    desktop = sub.add_parser("desktop", help="Open the Videoswa window")
    desktop.set_defaults(func=cmd_desktop)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "ui":
        print(
            "The Gradio interface has been removed. Open the desktop app with:\n"
            "    python -m videoswa",
            file=sys.stderr,
        )
        return 2
    if not argv:
        return cmd_desktop()

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    except (VideoTooLongError, VideoDurationUnknownError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

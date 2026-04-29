"""Command-line entry point.

Examples
--------
Single source onto every face in a video:
    python -m faceswap.cli swap -t input.mp4 -s alice.jpg -o out.mp4

Two sources mapped to two specific people (reference snapshots from the video):
    python -m faceswap.cli swap -t input.mp4 \\
        --pair alice.jpg=ref_alice.jpg \\
        --pair bob.jpg=ref_bob.jpg \\
        -o out.mp4

Image input:
    python -m faceswap.cli swap -t group.jpg -s alice.jpg -o group_swapped.png

Launch the web UI:
    python -m faceswap.cli ui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .core import FaceSwapEngine
from .face_analyzer import FaceAnalyzer
from .swapper import FaceSwapper
from .utils import logger
from .video import process_image, process_video


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

    analyzer = FaceAnalyzer(use_gpu=not args.cpu, det_thresh=args.det_thresh)
    swapper = FaceSwapper(use_gpu=not args.cpu, enhance=args.enhance)
    engine = FaceSwapEngine(
        analyzer=analyzer,
        swapper=swapper,
        similarity_threshold=args.similarity,
        apply_to_all_when_no_reference=args.apply_to_all,
    )

    mappings = engine.build_mappings([(_read(s), _read(r) if r else None) for s, r in pairs])

    suffix = target.suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
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


def cmd_ui(args: argparse.Namespace) -> int:
    from ui.app import launch
    launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faceswap", description="Video multi-face faceswap")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("swap", help="Run a faceswap on an image or video")
    s.add_argument("-t", "--target", required=True, help="Target image/video")
    s.add_argument("-s", "--source", help="Single source face image (applied to all faces)")
    s.add_argument(
        "--pair", action="append",
        help="Source=Reference image pair. Use multiple times for multi-face swaps.",
    )
    s.add_argument("-o", "--output", help="Output path (default: outputs/<name>_swapped.mp4)")
    s.add_argument("--similarity", type=float, default=0.45, help="Cosine threshold for ref match")
    s.add_argument("--det-thresh", type=float, default=0.5, help="Face detection threshold")
    s.add_argument("--cpu", action="store_true", help="Force CPU inference")
    s.add_argument("--enhance", action="store_true", help="Run GFPGAN on swapped faces")
    s.add_argument("--no-audio", action="store_true", help="Drop original audio track")
    s.add_argument("--crf", type=int, default=18, help="x264 CRF (lower = better quality)")
    s.add_argument("--preset", default="medium", help="x264 preset")
    s.add_argument(
        "--apply-to-all", action="store_true",
        help="When wildcard mappings exist alongside specific ones, also apply the wildcard to unmatched faces",
    )
    s.set_defaults(func=cmd_swap)

    u = sub.add_parser("ui", help="Launch the Gradio web UI")
    u.add_argument("--host", default="0.0.0.0")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--share", action="store_true")
    u.set_defaults(func=cmd_ui)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())

"""Videoswa entry point.

Open the desktop window:
    python run.py

Command-line swap:
    python run.py swap -t input.mp4 --pair alice.jpg=ref_alice.jpg -o out.mp4
"""

from faceswap.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

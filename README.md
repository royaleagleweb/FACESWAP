# Multi-Face Video FaceSwap

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/royaleagleweb/FACESWAP/blob/claude/video-faceswap-program-Ak3RC/run_in_colab.ipynb)

A FaceFusion-style video face-swap tool. Detects every face in a target image
or video, lets you map one or more **source identities** onto specific
people, and re-encodes the result while preserving the original audio.

**Try it now in Google Colab** — click the badge above. Free T4 GPU, public
Gradio URL printed in the last cell. ~3 minutes to first frame.

Built on top of:

- [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l`
  detector + `inswapper_128` ONNX swap model
- ONNX Runtime (CPU / CUDA / CoreML / DirectML auto-detected)
- FFmpeg (audio re-mux + H.264 encode)
- Optional GFPGAN post-process for sharper faces
- Gradio web UI

## Features

- **Multi-face mapping** — pair each source identity with a reference snapshot
  of the person to replace. The engine picks the closest match per frame using
  cosine similarity over face embeddings.
- **Wildcard mode** — provide just one source image with no reference, and
  every face in the video gets that identity.
- **Image or video input** (`.mp4 / .mov / .avi / .mkv / .webm / .png / .jpg ...`).
- **Audio preserved** through FFmpeg re-mux, x264 CRF / preset configurable.
- **GPU acceleration** when CUDA / CoreML / DirectML is available, automatic
  fallback to CPU.
- **CLI + Web UI** — pick whichever fits your workflow.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# FFmpeg is required for audio + final encode:
#   macOS:    brew install ffmpeg
#   Ubuntu:   sudo apt install ffmpeg
#   Windows:  choco install ffmpeg
```

The first run downloads two model files into `./models/`:
- `inswapper_128.onnx` (~ 530 MB) — the swap network
- `buffalo_l/*` — InsightFace detector + ArcFace recognizer (auto-fetched by `insightface`)

For GPU inference, install the matching ONNX Runtime build:

```bash
pip install onnxruntime-gpu     # CUDA
# or
pip install onnxruntime-directml  # Windows DirectML
```

## CLI

Single source applied to **every** face in a video:

```bash
python run.py swap -t input.mp4 -s alice.jpg -o out.mp4
```

Map two source identities to two specific people in the video. The
`reference` images are snapshots from the **target** that contain the
person you want to replace:

```bash
python run.py swap -t party.mp4 \
    --pair alice.jpg=ref_alice_in_video.jpg \
    --pair bob.jpg=ref_bob_in_video.jpg \
    -o party_swapped.mp4
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--enhance` | Run GFPGAN on each swapped face (slower, sharper) |
| `--similarity 0.45` | Cosine threshold for the reference→detection match |
| `--cpu` | Force CPU even if a GPU is available |
| `--no-audio` | Drop the original audio track |
| `--crf 18 --preset medium` | x264 quality / speed |

Image input works too:

```bash
python run.py swap -t group_photo.jpg -s me.jpg -o group_swapped.png
```

## Web UI

```bash
python run.py ui --host 0.0.0.0 --port 7860
```

1. Upload a target video.
2. Pick a timestamp where everyone you want to swap is on screen.
3. Click **Detect target faces**.
4. For each detected face, upload the source identity (leave a slot empty to
   keep that person unchanged), or tick **Use single source for all faces**.
5. Click **Run swap**.

Output shows the rendered video and live stats (frames, faces detected,
swapped, unmatched).

## Programmatic API

```python
import cv2
from faceswap import FaceAnalyzer, FaceSwapper, FaceSwapEngine

analyzer = FaceAnalyzer(use_gpu=True)
swapper = FaceSwapper(use_gpu=True, enhance=False)
engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper)

mappings = engine.build_mappings([
    (cv2.imread("alice.jpg"),  cv2.imread("ref_alice.jpg")),
    (cv2.imread("bob.jpg"),    cv2.imread("ref_bob.jpg")),
])

from pathlib import Path
from faceswap.video import process_video
process_video(engine, mappings, Path("input.mp4"), Path("out.mp4"))
```

## Repository layout

```
faceswap/
├── faceswap/
│   ├── __init__.py
│   ├── core.py            # FaceSwapEngine + multi-face mapping logic
│   ├── face_analyzer.py   # buffalo_l detection + embeddings
│   ├── swapper.py         # inswapper_128 wrapper + optional GFPGAN
│   ├── video.py           # decode/encode + ffmpeg audio re-mux
│   ├── utils.py           # model download, providers, paths
│   └── cli.py             # argparse CLI
├── ui/
│   └── app.py             # Gradio web UI
├── run.py                 # `python run.py swap|ui ...`
├── requirements.txt
└── README.md
```

## Ethics

This project ships face-swap technology for legitimate creative, research, and
accessibility uses. **Do not** use it to harass, defame, deceive, or create
sexual content of real people without explicit consent. Many jurisdictions
also require clear "synthetic media" disclosure when published. You are
responsible for how you use it.

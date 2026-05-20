"""Lightweight UI preview.

Mirrors the layout of `ui/app.py` but stubs out the InsightFace + inswapper
pipeline so the UI is viewable without GPU or the 530 MB model. Detection
uses the OpenCV Haar cascade. "Swap" simply blends the source face into
the detected face region — it is NOT a real identity swap, only a UI demo.

Run:  python preview.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import gradio as gr
import numpy as np

MAX_FACES = 6
HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect(frame_bgr: np.ndarray) -> List[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    boxes = HAAR.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
    return [tuple(map(int, b)) for b in boxes][:MAX_FACES]


def _crop(frame: np.ndarray, box, pad: float = 0.25) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x - px); y1 = max(0, y - py)
    x2 = min(w, x + bw + px); y2 = min(h, y + bh + py)
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def _video_frame(path: str, t_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(fps * t_s)))
    ok, f = cap.read(); cap.release()
    if not ok:
        raise gr.Error("Could not extract frame")
    return f


_state: dict[str, object] = {}


def detect(target, t_s: float):
    if target is None:
        raise gr.Error("Upload a target video or image first.")
    p = Path(target if isinstance(target, str) else target.get("name") or target.get("path"))
    if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        frame = _video_frame(str(p), t_s)
    else:
        frame = cv2.imread(str(p))
        if frame is None:
            raise gr.Error(f"Cannot read image: {p}")

    boxes = _detect(frame)
    _state["boxes"] = boxes
    _state["frame"] = frame

    crops: List[Optional[np.ndarray]] = [None] * MAX_FACES
    for i, b in enumerate(boxes):
        crops[i] = _crop(frame, b)

    updates: list = []
    for i in range(MAX_FACES):
        visible = i < len(boxes)
        updates.append(gr.update(value=crops[i], visible=visible))
        updates.append(gr.update(visible=visible))
        updates.append(gr.update(visible=visible, value=f"**Face {i + 1}**"))

    info = (
        f"Detected {len(boxes)} face(s) (Haar cascade — preview only). "
        f"Real engine uses InsightFace `buffalo_l` + `inswapper_128`."
    )
    return [info, *updates]


def _stub_swap_region(frame: np.ndarray, box, src_rgb: np.ndarray) -> np.ndarray:
    """Visual stub: blend the source face into the target box.

    NOT a real face swap — only here so the UI demo produces a visible
    result without needing the inswapper_128 model.
    """
    x, y, bw, bh = box
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR) if src_rgb.ndim == 3 else src_rgb
    resized = cv2.resize(src_bgr, (bw, bh))
    out = frame.copy()
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.ellipse(mask, (bw // 2, bh // 2), (int(bw * 0.45), int(bh * 0.55)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out[y:y + bh, x:x + bw] = (
        out[y:y + bh, x:x + bw].astype(np.float32) * (1 - mask3)
        + resized.astype(np.float32) * mask3
    ).astype(np.uint8)
    return out


def run(target, use_single, single_src, *src_uploads):
    if target is None:
        raise gr.Error("Upload a target first.")
    p = Path(target if isinstance(target, str) else target.get("name") or target.get("path"))

    if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        frame = _video_frame(str(p), 0.0)
    else:
        frame = cv2.imread(str(p))

    boxes = _state.get("boxes") or _detect(frame)
    out = frame
    if use_single:
        if single_src is None:
            raise gr.Error("Provide a single source image, or untick the checkbox.")
        for b in boxes:
            out = _stub_swap_region(out, b, single_src)
    else:
        for i, src in enumerate(src_uploads):
            if i >= len(boxes) or src is None:
                continue
            out = _stub_swap_region(out, boxes[i], src)

    tmp = Path(tempfile.gettempdir()) / "preview_out.png"
    cv2.imwrite(str(tmp), out)
    note = (
        "**Preview mode.** This is a visual demo — the result is NOT a real "
        "identity swap. The full engine on this PR uses InsightFace "
        "`buffalo_l` for detection + 512-d embeddings and the `inswapper_128` "
        "ONNX model for the actual identity swap."
    )
    return str(tmp), note


def build() -> gr.Blocks:
    with gr.Blocks(title="Multi-Face Video FaceSwap (preview)") as demo:
        gr.Markdown(
            "# Multi-Face Video FaceSwap — UI Preview\n"
            "_This is a layout-only preview. Detection uses Haar cascades and the "
            "'swap' is a visual blend, not a real identity swap. The full pipeline "
            "(InsightFace + inswapper_128) is wired up in `ui/app.py` and runs once "
            "the dependencies and model weights are installed._"
        )
        with gr.Row():
            with gr.Column(scale=1):
                target = gr.File(
                    label="Target video or image",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
                                ".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                    type="filepath",
                )
                t_s = gr.Slider(0, 60, value=0.0, step=0.1, label="Snapshot timestamp (s)")
                gr.Checkbox(label="Use GPU if available", value=True, interactive=False)
                gr.Checkbox(label="GFPGAN face enhance (slower)", value=False, interactive=False)
                detect_btn = gr.Button("Detect target faces", variant="secondary")
                info = gr.Markdown()

            with gr.Column(scale=2):
                use_single = gr.Checkbox(label="Use single source for all faces", value=False)
                single_src = gr.Image(label="Single source identity", type="numpy")

                gr.Markdown("### Per-face source mapping")
                refs: list = []
                srcs: list = []
                lbls: list = []
                for i in range(MAX_FACES):
                    with gr.Row():
                        l = gr.Markdown(value=f"**Face {i+1}**", visible=False)
                        r = gr.Image(label=f"Detected face {i+1}", visible=False, interactive=False)
                        s = gr.Image(label=f"Replace with (face {i+1})", visible=False, type="numpy")
                    lbls.append(l); refs.append(r); srcs.append(s)

                with gr.Accordion("Advanced", open=False):
                    gr.Slider(0.1, 0.9, value=0.45, step=0.01, label="Identity match threshold")
                    gr.Checkbox(label="Keep original audio", value=True)
                    gr.Slider(0, 32, value=18, step=1, label="x264 CRF")
                    gr.Dropdown(["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"],
                                value="medium", label="x264 preset")

                run_btn = gr.Button("Run swap", variant="primary")
                stats = gr.Markdown()
                image_out = gr.Image(label="Output (preview blend)")

        outs: list = [info]
        for i in range(MAX_FACES):
            outs.extend([refs[i], srcs[i], lbls[i]])
        detect_btn.click(detect, inputs=[target, t_s], outputs=outs)
        run_btn.click(run, inputs=[target, use_single, single_src, *srcs],
                      outputs=[image_out, stats])
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    build().queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=("--share" in sys.argv),
        inbrowser=False,
    )

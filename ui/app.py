"""Gradio web UI for the multi-face video faceswap.

Workflow:
1. Upload a target video (or image).
2. Click "Detect target faces" — the app extracts a snapshot at the chosen
   timestamp and shows every detected face.
3. For each detected face, upload the source identity you want to swap in
   (leave a slot empty to keep that person unchanged).
4. Click "Run swap".

You can also tick "Apply to all faces" and just upload one source image to
swap every face in the video with that single identity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from faceswap.core import FaceMapping, FaceSwapEngine
from faceswap.face_analyzer import Face, FaceAnalyzer
from faceswap.swapper import FaceSwapper
from faceswap.utils import OUTPUTS_DIR, TEMP_DIR, logger
from faceswap.video import process_image, process_video

MAX_FACES = 6  # number of slots shown in the UI

_engine_cache: dict[Tuple[bool, bool], FaceSwapEngine] = {}


def get_engine(use_gpu: bool, enhance: bool) -> FaceSwapEngine:
    key = (use_gpu, enhance)
    if key not in _engine_cache:
        analyzer = FaceAnalyzer(use_gpu=use_gpu)
        swapper = FaceSwapper(use_gpu=use_gpu, enhance=enhance)
        _engine_cache[key] = FaceSwapEngine(analyzer=analyzer, swapper=swapper)
    return _engine_cache[key]


def _load_video_frame(video_path: str, timestamp_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(fps * timestamp_s)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise gr.Error("Could not extract frame from video")
    return frame


def _crop_face(frame: np.ndarray, face: Face, pad: float = 0.25) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    px = int((x2 - x1) * pad)
    py = int((y2 - y1) * pad)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w, x2 + px); y2 = min(h, y2 + py)
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


_detected_state: dict[str, object] = {}


def detect_faces(target, timestamp: float, use_gpu: bool):
    """Detect faces from the target media and return previews + visibility."""
    if target is None:
        raise gr.Error("Upload a target video or image first.")

    target_path = target if isinstance(target, str) else target.get("name") or target.get("path")
    target_path = Path(target_path)
    if not target_path.exists():
        raise gr.Error(f"Target not found: {target_path}")

    if target_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        frame = _load_video_frame(str(target_path), timestamp)
    else:
        frame = cv2.imread(str(target_path))
        if frame is None:
            raise gr.Error(f"Cannot read image: {target_path}")

    analyzer = get_engine(use_gpu=use_gpu, enhance=False).analyzer
    faces = analyzer.analyze(frame)

    crops: List[Optional[np.ndarray]] = [None] * MAX_FACES
    for i, f in enumerate(faces[:MAX_FACES]):
        crops[i] = _crop_face(frame, f)

    _detected_state["faces"] = faces[:MAX_FACES]
    _detected_state["frame"] = frame

    updates: list = []
    for i in range(MAX_FACES):
        visible = i < len(faces)
        updates.append(gr.update(value=crops[i], visible=visible))     # ref preview
        updates.append(gr.update(visible=visible))                     # source upload
        updates.append(gr.update(visible=visible, value=f"Face {i + 1}"))  # label

    info = (
        f"Detected {len(faces)} face(s). Showing up to {MAX_FACES}. "
        f"Upload one source image per face you want to replace; leave the others empty."
    )
    return [info, *updates]


def run_swap(
    target,
    use_single_source: bool,
    single_source,
    similarity: float,
    use_gpu: bool,
    enhance: bool,
    keep_audio: bool,
    crf: int,
    preset: str,
    *source_uploads,
):
    if target is None:
        raise gr.Error("Upload a target video or image first.")

    target_path = Path(target if isinstance(target, str) else (target.get("name") or target.get("path")))

    engine = get_engine(use_gpu=use_gpu, enhance=enhance)
    engine.similarity_threshold = similarity

    mappings: List[FaceMapping] = []

    if use_single_source:
        if single_source is None:
            raise gr.Error("Upload a single source image, or untick 'Use single source for all'.")
        src_img = _to_bgr(single_source)
        mappings.append(engine.build_mapping(src_img, reference_image_bgr=None, label="all"))
    else:
        faces: List[Face] = _detected_state.get("faces") or []  # type: ignore[assignment]
        if not faces:
            raise gr.Error("Click 'Detect target faces' first.")
        frame = _detected_state["frame"]  # type: ignore[index]
        for i, src in enumerate(source_uploads):
            if i >= len(faces) or src is None:
                continue
            src_img = _to_bgr(src)
            src_face = engine.analyzer.best_face(src_img)
            if src_face is None:
                logger.warning("No face detected in source for face %d, skipping", i + 1)
                continue
            mappings.append(
                FaceMapping(
                    source_face=src_face,
                    reference_face=faces[i],
                    label=f"face_{i+1}",
                )
            )

    if not mappings:
        raise gr.Error("No source images provided.")

    if not mappings:
        raise gr.Error("No face was detected in any of the source images.")

    suffix = target_path.suffix.lower()
    out_path = OUTPUTS_DIR / f"{target_path.stem}_swapped{'.png' if suffix not in {'.mp4','.mov','.avi','.mkv','.webm','.m4v'} else '.mp4'}"
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        out = process_video(
            engine, mappings, input_path=target_path, output_path=out_path,
            keep_audio=keep_audio, crf=crf, preset=preset,
        )
        return str(out), None, _stats_str(engine)
    else:
        out = process_image(engine, mappings, input_path=target_path, output_path=out_path)
        return None, str(out), _stats_str(engine)


def _stats_str(engine: FaceSwapEngine) -> str:
    s = engine.stats
    return (
        f"Frames: {s.frames} | Faces detected: {s.faces_detected} | "
        f"Swapped: {s.faces_swapped} | Unmatched: {s.faces_unmatched}"
    )


def _to_bgr(image) -> np.ndarray:
    """Convert a Gradio Image upload (RGB ndarray or path) to BGR ndarray."""
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise gr.Error(f"Could not read source image: {image}")
        return img
    if isinstance(image, dict) and "path" in image:
        return _to_bgr(image["path"])
    raise gr.Error("Unsupported source image type")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Multi-Face Video FaceSwap") as demo:
        gr.Markdown(
            "# Multi-Face Video FaceSwap\n"
            "Upload a target video, detect the faces, then upload one source identity "
            "per face you want to replace. Or tick 'Use single source for all' to swap "
            "every face with one identity."
        )

        with gr.Row():
            with gr.Column(scale=1):
                target = gr.File(
                    label="Target video or image",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
                                ".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                    type="filepath",
                )
                timestamp = gr.Slider(0, 60, value=0.0, step=0.1, label="Snapshot timestamp (s)")
                use_gpu = gr.Checkbox(label="Use GPU if available", value=True)
                enhance = gr.Checkbox(label="GFPGAN face enhance (slower)", value=False)

                detect_btn = gr.Button("Detect target faces", variant="secondary")
                detect_info = gr.Markdown()

            with gr.Column(scale=2):
                use_single = gr.Checkbox(label="Use single source for all faces", value=False)
                single_source = gr.Image(label="Single source identity", type="numpy")

                gr.Markdown("### Per-face source mapping")
                ref_previews: List[gr.Image] = []
                source_inputs: List[gr.Image] = []
                labels: List[gr.Markdown] = []
                for i in range(MAX_FACES):
                    with gr.Row():
                        lbl = gr.Markdown(value=f"Face {i+1}", visible=False)
                        rp = gr.Image(label=f"Detected face {i+1}", visible=False, interactive=False)
                        si = gr.Image(label=f"Replace with (face {i+1})", visible=False, type="numpy")
                    labels.append(lbl)
                    ref_previews.append(rp)
                    source_inputs.append(si)

                with gr.Accordion("Advanced", open=False):
                    similarity = gr.Slider(0.1, 0.9, value=0.45, step=0.01, label="Identity match threshold")
                    keep_audio = gr.Checkbox(label="Keep original audio", value=True)
                    crf = gr.Slider(0, 32, value=18, step=1, label="x264 CRF (lower = higher quality)")
                    preset = gr.Dropdown(
                        ["ultrafast", "superfast", "veryfast", "faster", "fast",
                         "medium", "slow", "slower", "veryslow"],
                        value="medium", label="x264 preset",
                    )

                run_btn = gr.Button("Run swap", variant="primary")
                stats = gr.Markdown()
                video_out = gr.Video(label="Output video")
                image_out = gr.Image(label="Output image")

        # Wire up detection. detect_faces returns 1 + 3*MAX_FACES updates.
        detect_outputs: list = [detect_info]
        for i in range(MAX_FACES):
            detect_outputs.extend([ref_previews[i], source_inputs[i], labels[i]])
        detect_btn.click(detect_faces, inputs=[target, timestamp, use_gpu], outputs=detect_outputs)

        run_btn.click(
            run_swap,
            inputs=[target, use_single, single_source, similarity, use_gpu, enhance,
                    keep_audio, crf, preset, *source_inputs],
            outputs=[video_out, image_out, stats],
        )

    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False) -> None:
    app = build_app()
    app.queue().launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch()

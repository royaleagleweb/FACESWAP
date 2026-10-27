"""CUDA install pins, pixel boost, presets, NVENC, and the frame pipeline."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from faceswap.coverage import coverage_alpha, template_128
from faceswap.face_analyzer import pad_to_square, prepare_source_image
from faceswap.smooth import OneEuroFilter, landmark_jumped
from faceswap.codeformer import blend_restored, codeformer_install_tip
from faceswap.pixel_boost import join_tiles, pixel_boost_tiles, run_pixel_boost, split_tiles
from faceswap.providers import register_windows_dll_directories, windows_dll_candidates
from faceswap.quality import quality_preset
from faceswap.runtime_check import INSIGHTFACE_WHEEL, environment_problems
from faceswap.utils import CODEFORMER_SHA256, CODEFORMER_URLS
from faceswap.video import SwapCancelled, _video_encode_args, process_video, resolve_encoder
from videoswa.jobs import MAX_IDENTITY_PHOTOS, average_normed_embeddings, read_image


ROOT = Path(__file__).resolve().parents[1]


def test_windows_cuda_requirements_pin_the_lenovo_stack() -> None:
    text = (ROOT / "requirements-windows-cuda.txt").read_text(encoding="utf-8")
    assert "numpy==1.26.4" in text
    assert "opencv-python==4.10.0.84" in text
    assert "opencv-python-headless" in text
    assert INSIGHTFACE_WHEEL in text
    assert "insightface-0.7.3-cp311-cp311-win_amd64.whl" in text
    assert "onnxruntime-gpu==1.22.0" in text
    assert "tensorrt-cu12==10.9.0.34" in text
    assert "nvidia-cudnn-cu12" in text


def test_environment_check_names_numpy2_opencv5_and_headless() -> None:
    assert environment_problems(numpy_version="1.26.4", opencv_version="4.10.0", headless_installed=False) == []
    numpy_hit = environment_problems(numpy_version="2.1.0", opencv_version="4.10.0", headless_installed=False)
    assert any("numpy.dtype size changed" in item for item in numpy_hit)
    opencv_hit = environment_problems(numpy_version="1.26.4", opencv_version="5.0.0", headless_installed=False)
    assert any("OpenCV 5.0.0" in item and "4.10.0.84" in item for item in opencv_hit)
    headless_hit = environment_problems(numpy_version="1.26.4", opencv_version="4.10.0", headless_installed=True)
    assert any("opencv-python-headless" in item for item in headless_hit)


def test_windows_dll_candidates_include_torch_nvidia_and_tensorrt(tmp_path: Path) -> None:
    nvidia = tmp_path / "nvidia" / "cudnn" / "bin"
    nvidia.mkdir(parents=True)
    tensorrt = tmp_path / "tensorrt_libs"
    tensorrt.mkdir()
    torch_lib = tmp_path / "torchlib"
    torch_lib.mkdir()
    found = {path.resolve() for path in windows_dll_candidates(search_roots=[tmp_path], torch_lib=torch_lib)}
    assert nvidia.resolve() in found
    assert tensorrt.resolve() in found
    assert torch_lib.resolve() in found
    assert register_windows_dll_directories() == []


def test_pixel_boost_tile_counts_and_reassembly() -> None:
    assert pixel_boost_tiles(128) == 1
    assert pixel_boost_tiles(256) == 4
    assert pixel_boost_tiles(512) == 16
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    tiles = split_tiles(image)
    assert len(tiles) == 4
    assert tiles[0].shape == (128, 128, 3)
    assert np.array_equal(join_tiles(tiles, 256), image)

    frame = np.zeros((180, 180, 3), dtype=np.uint8)
    kps = template_128() + np.array([20.0, 16.0])
    calls = []

    def _swap(tile):
        calls.append(tuple(tile.shape))
        return np.full_like(tile, 7)

    canvas, matrix = run_pixel_boost(frame, kps, 256, _swap)
    assert len(calls) == 4
    assert canvas.shape == (256, 256, 3)
    assert matrix.shape == (2, 3)
    assert int(canvas[0, 0, 0]) == 7


def test_quality_presets_cover_fast_balanced_and_quality() -> None:
    fast = quality_preset("fast")
    balanced = quality_preset("balanced")
    quality = quality_preset("quality")
    assert fast["pixel_boost"] == 128
    assert fast["precise_edges"] is False
    assert fast["enhance"] is False
    assert fast["fast_draft_preview"] is True
    assert fast["detect_every_other"] is True
    assert balanced["pixel_boost"] == 256
    assert balanced["precise_edges"] is True
    assert balanced["fast_draft_preview"] is True
    assert balanced["detect_every_other"] is True
    assert quality["pixel_boost"] == 512
    assert quality["precise_edges"] is True
    assert quality["fast_draft_preview"] is False
    assert quality["detect_every_other"] is False
    assert quality["restorer"] == "codeformer"


def test_identity_average_uses_five_photos_at_most() -> None:
    assert MAX_IDENTITY_PHOTOS == 5
    first = np.array([1.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0], dtype=np.float32)
    mean = average_normed_embeddings([first, second])
    assert mean.shape == (2,)
    assert abs(float(np.linalg.norm(mean)) - 1.0) < 1e-5
    assert mean[0] == pytest.approx(mean[1])


def test_codeformer_metadata_and_blend_do_not_download() -> None:
    assert CODEFORMER_SHA256 == "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7"
    assert any("codeformer.pth" in url for url in CODEFORMER_URLS)
    tip = codeformer_install_tip()
    assert tip is None or "PyTorch" in tip
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    restored = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert int(blend_restored(original, restored, 0.0)[0, 0, 0]) == 0
    assert int(blend_restored(original, restored, 1.0)[0, 0, 0]) == 100


def test_one_euro_holds_still_and_resets_on_a_jump() -> None:
    filt = OneEuroFilter()
    first = np.zeros((5, 2), dtype=np.float64)
    assert np.allclose(filt(0.0, first), first)
    nudged = first.copy()
    nudged[:, 0] += 1.0
    second = filt(1.0 / 30.0, nudged)
    assert float(np.mean(np.abs(second - first))) < 1.0
    jumped = first.copy()
    jumped[:, 0] += 200.0
    assert landmark_jumped(second, jumped) is True


def test_source_image_handles_exif_alpha_and_size(tmp_path: Path) -> None:
    from PIL import Image

    wide = np.zeros((20, 80, 3), dtype=np.uint8)
    square = prepare_source_image(wide)
    assert square.shape[0] == square.shape[1]
    assert square.shape[0] >= 640
    huge = np.zeros((1800, 3000, 3), dtype=np.uint8)
    fitted = prepare_source_image(huge)
    assert max(fitted.shape[:2]) <= 1600
    assert pad_to_square(wide).shape[0] == 80

    rgba = Image.new("RGBA", (12, 10), (255, 0, 0, 0))
    rgba.putpixel((2, 2), (0, 255, 0, 255))
    rgba_path = tmp_path / "alpha.png"
    rgba.save(rgba_path)
    loaded = read_image(rgba_path)
    assert loaded.shape == (10, 12, 3)
    assert int(loaded[0, 0, 0]) > 240
    assert int(loaded[2, 2, 1]) > 240

    turned = Image.new("RGB", (40, 10), (255, 0, 0))
    exif = turned.getexif()
    exif[274] = 6
    exif_path = tmp_path / "turned.jpg"
    turned.save(exif_path, exif=exif)
    oriented = read_image(exif_path)
    assert oriented.shape[0] > oriented.shape[1]


def test_jaw_boundary_is_feathered() -> None:
    kps = template_128() + np.array([80.0, 70.0])
    alpha = coverage_alpha((240, 240), kps, "full")
    soft = (alpha > 0.15) & (alpha < 0.85)
    assert int(np.count_nonzero(soft)) > 10


def test_resolve_encoder_prefers_h264_nvenc_then_hevc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("faceswap.video._ffmpeg_has_encoder", lambda name: name == "h264_nvenc")
    assert resolve_encoder("auto") == "h264_nvenc"
    monkeypatch.setattr("faceswap.video._ffmpeg_has_encoder", lambda name: name == "hevc_nvenc")
    assert resolve_encoder("auto") == "hevc_nvenc"
    assert resolve_encoder("hevc_nvenc") == "hevc_nvenc"
    monkeypatch.setattr("faceswap.video._ffmpeg_has_encoder", lambda _name: False)
    assert resolve_encoder("auto") == "libx264"
    assert resolve_encoder("h264_nvenc") == "libx264"
    hevc = _video_encode_args("hevc_nvenc", 18, "medium")
    assert "hevc_nvenc" in hevc
    assert "yuv420p" in hevc


def test_pipeline_keeps_frame_order_and_cancel(tmp_path: Path) -> None:
    from faceswap.core import SwapStats

    path = tmp_path / "in.mp4"
    import cv2

    size = (32, 24)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((size[1], size[0], 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    class _Engine:
        def __init__(self) -> None:
            self.stats = SwapStats()
            self.times: list[float] = []

        def process_frame(self, frame, mappings, time_s=0.0):
            self.times.append(float(time_s))
            out = frame.copy()
            out[:, :, 1] = 9
            return out

    engine = _Engine()
    out = process_video(engine, [], path, output_path=tmp_path / "out.mp4", keep_audio=False)
    assert out.exists()
    assert engine.times == sorted(engine.times)
    assert len(engine.times) == 3

    cancel = threading.Event()

    def _cancel(frame, mappings, time_s=0.0):
        engine.times.append(time_s)
        cancel.set()
        return frame

    engine.times.clear()
    engine.process_frame = _cancel  # type: ignore[method-assign]
    with pytest.raises(SwapCancelled):
        process_video(engine, [], path, output_path=tmp_path / "cancelled.mp4", cancel_event=cancel)
    assert not (tmp_path / "cancelled.mp4").exists()
    assert len(engine.times) == 1

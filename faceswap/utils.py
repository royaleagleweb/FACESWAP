"""Common utilities: model download, path helpers, logging."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger("faceswap")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("FACESWAP_MODELS_DIR", PROJECT_ROOT / "models"))
TEMP_DIR = Path(os.environ.get("FACESWAP_TEMP_DIR", PROJECT_ROOT / "temp"))
OUTPUTS_DIR = Path(os.environ.get("FACESWAP_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

for _d in (MODELS_DIR, TEMP_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# Public mirrors for the inswapper_128 ONNX model. The model is the same one
# used by the official InsightFace examples; checksum verifies integrity.
INSWAPPER_URLS = [
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
]
INSWAPPER_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def download_file(urls: list[str], destination: Path, expected_sha256: Optional[str] = None) -> Path:
    """Download a file from the first working mirror, with progress bar + checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            return destination
        logger.warning("Checksum mismatch on %s, re-downloading", destination)
        destination.unlink()

    last_err: Optional[Exception] = None
    for url in urls:
        try:
            logger.info("Downloading %s", url)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                tmp = destination.with_suffix(destination.suffix + ".part")
                with tmp.open("wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=destination.name
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bar.update(len(chunk))
                tmp.replace(destination)
            if expected_sha256 and _sha256(destination) != expected_sha256:
                raise RuntimeError(f"Checksum mismatch for {destination.name}")
            return destination
        except Exception as exc:  # network or checksum issue
            last_err = exc
            logger.warning("Download from %s failed: %s", url, exc)
            if destination.exists():
                destination.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {destination.name}: {last_err}")


def ensure_inswapper() -> Path:
    """Make sure inswapper_128.onnx is present locally and return its path."""
    target = MODELS_DIR / "inswapper_128.onnx"
    return download_file(INSWAPPER_URLS, target, INSWAPPER_SHA256)


def select_providers(use_gpu: bool = True) -> list[str]:
    """Pick onnxruntime execution providers based on what's installed."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required") from exc

    available = set(ort.get_available_providers())
    preferred: list[str] = []
    if use_gpu:
        for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"):
            if p in available:
                preferred.append(p)
    preferred.append("CPUExecutionProvider")
    return preferred

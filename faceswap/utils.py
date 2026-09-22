"""Common utilities: model download, path helpers, logging."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

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


# Public mirrors for InsightFace's inswapper_128 ONNX model. The first two
# serve the canonical 554,253,681-byte file. Later URLs are fallbacks.
# Drop a file at models/inswapper_128.onnx to skip this when its SHA256 is
# in INSWAPPER_SHA256_ALLOWLIST.
INSWAPPER_URLS = [
    "https://huggingface.co/Chuchuwa2/inswap/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/crw-dev/Deepinsightinswapper/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
]
# Canonical InsightFace inswapper_128.onnx (the copy common mirrors serve).
INSWAPPER_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
# Also accepted: an older redistributed copy already present in some installs.
INSWAPPER_SHA256_ALLOWLIST = (
    INSWAPPER_SHA256,
    "a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7",
)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _accepted_hashes(expected_sha256: Optional[Union[str, Sequence[str]]]) -> Optional[tuple[str, ...]]:
    """Normalize one hash or an allowlist. None means do not check."""
    if expected_sha256 is None:
        return None
    if isinstance(expected_sha256, str):
        hashes = (expected_sha256,)
    else:
        hashes = tuple(expected_sha256)
    normalized = tuple(item.strip().lower() for item in hashes if item and item.strip())
    if not normalized:
        return None
    return normalized


def download_file(
    urls: list[str],
    destination: Path,
    expected_sha256: Optional[Union[str, Sequence[str]]] = None,
) -> Path:
    """Download a file from the first mirror whose bytes match an accepted SHA256.

    ``expected_sha256`` may be one hex digest or several. A checksum miss, HTTP
    error, or network error moves on to the next URL. A file already at
    ``destination`` is kept when it matches.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    accepted = _accepted_hashes(expected_sha256)
    if destination.exists() and destination.stat().st_size > 0:
        if accepted is None or _sha256(destination) in accepted:
            return destination
        logger.warning("Checksum mismatch on %s, re-downloading", destination)
        destination.unlink()

    failures: list[str] = []
    for url in urls:
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            logger.info("Downloading %s", url)
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with partial.open("wb") as handle, tqdm(
                    total=total, unit="B", unit_scale=True, desc=destination.name
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bar.update(len(chunk))
                partial.replace(destination)
            if accepted is not None:
                digest = _sha256(destination)
                if digest not in accepted:
                    expected = ", ".join(accepted)
                    raise RuntimeError(
                        f"Checksum mismatch for {destination.name}: "
                        f"got {digest}, expected one of {expected}"
                    )
            return destination
        except Exception as exc:  # network or checksum issue
            failures.append(f"{url}: {exc}")
            logger.warning("Download from %s failed: %s", url, exc)
            destination.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
    expected = "any file" if accepted is None else ", ".join(accepted)
    details = "\n".join(f"  - {item}" for item in failures) or "  - no URLs were provided"
    raise RuntimeError(
        f"Failed to download {destination.name} from {len(urls)} mirror(s). "
        f"Accepted SHA256: {expected}. "
        f"Place a file with one of those hashes at {destination} to skip the download.\n"
        f"{details}"
    )


def ensure_inswapper() -> Path:
    """Make sure inswapper_128.onnx is present locally and return its path."""
    target = MODELS_DIR / "inswapper_128.onnx"
    return download_file(INSWAPPER_URLS, target, INSWAPPER_SHA256_ALLOWLIST)


def select_providers(use_gpu: bool = True, execution: str = "auto"):
    """First ONNX Runtime provider list Videoswa will try.

    Kept for callers that only need the preferred list. Loading code should
    use ``faceswap.providers.provider_attempts`` so a failed TensorRT session
    can fall back to CUDA and then CPU.
    """
    from .providers import provider_attempts

    mode = "cpu" if not use_gpu else execution
    return provider_attempts(mode)[0]

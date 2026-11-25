"""InSwapper download: canonical checksum, mirror order, and failure text."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from faceswap.utils import (
    INSWAPPER_SHA256,
    INSWAPPER_SHA256_ALLOWLIST,
    INSWAPPER_URLS,
    download_file,
    ensure_inswapper,
)

CANONICAL = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
LEGACY = "a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _Response:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 1):
        yield self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def test_canonical_hash_is_preferred_and_mirrors_lead() -> None:
    assert INSWAPPER_SHA256 == CANONICAL
    assert INSWAPPER_SHA256_ALLOWLIST[0] == CANONICAL
    assert LEGACY in INSWAPPER_SHA256_ALLOWLIST
    assert INSWAPPER_URLS[0].startswith("https://huggingface.co/Chuchuwa2/inswap/")
    assert INSWAPPER_URLS[1].startswith("https://huggingface.co/crw-dev/Deepinsightinswapper/")


def test_existing_file_matching_either_known_hash_is_kept(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.onnx"
    canonical.write_bytes(b"canonical-inswapper")
    legacy = tmp_path / "legacy.onnx"
    legacy.write_bytes(b"legacy-inswapper")
    assert download_file([], canonical, _digest(b"canonical-inswapper")) == canonical
    assert download_file([], legacy, (_digest(b"other"), _digest(b"legacy-inswapper"))) == legacy
    assert canonical.read_bytes() == b"canonical-inswapper"


def test_checksum_mismatch_tries_the_next_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = b"good-inswapper"
    bad = b"bad-inswapper"
    calls: list[str] = []

    def fake_get(url, stream=True, timeout=60):
        calls.append(url)
        payload = bad if url.endswith("/bad") else good
        return _Response(payload)

    monkeypatch.setattr("faceswap.utils.requests.get", fake_get)
    destination = tmp_path / "inswapper_128.onnx"
    path = download_file(
        ["https://example.test/bad", "https://example.test/good"],
        destination,
        _digest(good),
    )
    assert path == destination
    assert destination.read_bytes() == good
    assert calls == ["https://example.test/bad", "https://example.test/good"]
    assert not destination.with_suffix(".onnx.part").exists()


def test_all_mirrors_fail_with_hashes_and_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"not-the-model"

    def fake_get(url, stream=True, timeout=60):
        if url.endswith("/down"):
            raise RuntimeError("connection reset")
        return _Response(payload)

    monkeypatch.setattr("faceswap.utils.requests.get", fake_get)
    destination = tmp_path / "inswapper_128.onnx"
    with pytest.raises(RuntimeError, match="Failed to download inswapper_128.onnx") as exc:
        download_file(
            ["https://example.test/mismatch", "https://example.test/down"],
            destination,
            (CANONICAL, LEGACY),
        )
    message = str(exc.value)
    assert CANONICAL in message
    assert LEGACY in message
    assert "https://example.test/mismatch" in message
    assert "https://example.test/down" in message
    assert _digest(payload) in message
    assert "connection reset" in message
    assert str(destination) in message
    assert not destination.exists()


def test_ensure_inswapper_asks_for_the_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_download(urls, destination, expected_sha256=None):
        captured["urls"] = list(urls)
        captured["hashes"] = tuple(expected_sha256)
        destination.write_bytes(b"ok")
        return destination

    monkeypatch.setattr("faceswap.utils.MODELS_DIR", tmp_path)
    monkeypatch.setattr("faceswap.utils.download_file", fake_download)
    path = ensure_inswapper()
    assert path == tmp_path / "inswapper_128.onnx"
    assert captured["urls"] == INSWAPPER_URLS
    assert captured["hashes"][0] == CANONICAL
    assert LEGACY in captured["hashes"]

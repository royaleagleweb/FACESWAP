"""CUDA sessions that die inside Conv fall back to CPU."""

from __future__ import annotations

import os

import numpy as np
import pytest

from faceswap.face_analyzer import FaceAnalyzer
from faceswap.providers import (
    CPU,
    CUDA,
    CUDA_FAILED_STATUS,
    TENSORRT,
    CudaRuntimeGuard,
    configure_cuda_runtime,
    cuda_provider_options,
    is_cuda_runtime_failure,
    provider_attempts,
    provider_names,
)

CHANA_ERROR = (
    "[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running "
    "Conv node. Name:'conv0' Status Message: CUDNN_FE failure 7: "
    "GRAPH_EXECUTION_FAILED ; GPU=0 ; hostname=CHANA ; "
    "onnxruntime\\core\\providers\\cuda\\nn\\conv.cc line=485"
)


def test_chana_cudnn_fe_error_is_a_cuda_runtime_failure() -> None:
    assert is_cuda_runtime_failure(RuntimeError(CHANA_ERROR))
    assert not is_cuda_runtime_failure(RuntimeError("No face found in source image"))


def test_auto_order_stays_tensorrt_then_cuda_then_cpu(tmp_path) -> None:
    attempts = provider_attempts(
        "auto",
        available=[TENSORRT, CUDA, CPU],
        cache_dir=tmp_path,
    )
    assert provider_names(attempts[0]) == [TENSORRT, CUDA, CPU]
    assert provider_names(attempts[1]) == [CUDA, CPU]
    assert provider_names(attempts[2]) == [CPU]


def test_disabled_frontend_selects_default_conv_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORT_DISABLE_CUDNN_FRONTEND", raising=False)
    configure_cuda_runtime()
    assert os.environ["ORT_DISABLE_CUDNN_FRONTEND"] == "1"
    assert cuda_provider_options()["cudnn_conv_algo_search"] == "DEFAULT"
    monkeypatch.setenv("ORT_DISABLE_CUDNN_FRONTEND", "0")
    assert cuda_provider_options()["cudnn_conv_algo_search"] == "HEURISTIC"


class _Session:
    def __init__(self, name: str) -> None:
        self.name = name
        self.providers = [(CUDA, {})]
        self.loads: list[str] = []

    def adopt_cpu(self) -> None:
        self.loads.append("cpu")
        self.providers = [CPU]


def test_runtime_failure_rebuilds_analyzer_and_swapper_on_cpu() -> None:
    analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
    analyzer.providers = [(CUDA, {})]
    calls = {"n": 0}

    def _analyze_impl(_image):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(CHANA_ERROR)
        return ["face"]

    analyzer._analyze_impl = _analyze_impl

    def _adopt_analyzer() -> None:
        analyzer.providers = [CPU]

    analyzer.adopt_cpu = _adopt_analyzer
    swapper = _Session("inswapper")
    guard = CudaRuntimeGuard()
    guard.attach(analyzer)
    guard.attach(swapper)
    seen: list[str] = []
    guard.on_fallback = seen.append

    found = analyzer.analyze(np.zeros((4, 4, 3), dtype=np.uint8))
    assert found == ["face"]
    assert calls["n"] == 2
    assert seen == [CUDA_FAILED_STATUS]
    assert guard.note == "CUDA failed; using CPU"
    assert swapper.loads == ["cpu"]
    assert analyzer.providers == [CPU]


def test_second_cudnn_failure_is_not_retried_forever() -> None:
    analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
    analyzer.providers = [(CUDA, {})]
    analyzer._analyze_impl = lambda _image: (_ for _ in ()).throw(RuntimeError(CHANA_ERROR))
    analyzer.adopt_cpu = lambda: setattr(analyzer, "providers", [CPU]) or None
    guard = CudaRuntimeGuard()
    guard.attach(analyzer)
    with pytest.raises(RuntimeError, match="GRAPH_EXECUTION_FAILED"):
        analyzer.analyze(np.zeros((2, 2, 3), dtype=np.uint8))

"""TensorRT is first when the ONNX Runtime build advertises it."""

from pathlib import Path

from faceswap.providers import (
    CPU,
    CUDA,
    TENSORRT,
    format_providers,
    provider_attempts,
    provider_names,
)


def test_auto_order_is_tensorrt_then_cuda_then_cpu(tmp_path: Path) -> None:
    attempts = provider_attempts(
        "auto",
        available=[TENSORRT, CUDA, CPU, "CoreMLExecutionProvider"],
        cache_dir=tmp_path,
    )
    assert provider_names(attempts[0]) == [TENSORRT, CUDA, CPU]
    assert provider_names(attempts[1]) == [CUDA, CPU]
    assert provider_names(attempts[2]) == [CPU]
    assert "TensorRT" not in format_providers(attempts[1])


def test_cuda_mode_skips_tensorrt(tmp_path: Path) -> None:
    attempts = provider_attempts(
        "cuda",
        available=[TENSORRT, CUDA, CPU],
        cache_dir=tmp_path,
    )
    assert provider_names(attempts[0]) == [CUDA, CPU]
    assert all(TENSORRT not in provider_names(attempt) for attempt in attempts)


def test_cpu_mode_is_only_cpu() -> None:
    attempts = provider_attempts("cpu", available=[TENSORRT, CUDA, CPU])
    assert attempts == [[CPU]]


def test_windows_auto_includes_directml_before_cpu(tmp_path: Path) -> None:
    from faceswap.providers import DIRECTML

    attempts = provider_attempts(
        "auto",
        available=[TENSORRT, CUDA, DIRECTML, CPU],
        cache_dir=tmp_path,
        platform="win32",
    )
    assert provider_names(attempts[0]) == [TENSORRT, CUDA, DIRECTML, CPU]
    assert provider_names(attempts[1]) == [CUDA, DIRECTML, CPU]
    assert provider_names(attempts[2]) == [DIRECTML, CPU]
    assert provider_names(attempts[3]) == [CPU]


def test_linux_auto_does_not_insert_directml_ahead_of_cuda(tmp_path: Path) -> None:
    from faceswap.providers import DIRECTML

    attempts = provider_attempts(
        "auto",
        available=[TENSORRT, CUDA, DIRECTML, CPU],
        cache_dir=tmp_path,
        platform="linux",
    )
    assert provider_names(attempts[0]) == [TENSORRT, CUDA, CPU]
    assert all(DIRECTML not in provider_names(attempt) for attempt in attempts)


def test_directml_when_no_nvidia(tmp_path: Path) -> None:
    attempts = provider_attempts(
        "auto",
        available=["DmlExecutionProvider", CPU],
        cache_dir=tmp_path,
    )
    assert provider_names(attempts[0]) == ["DmlExecutionProvider", CPU]


def test_installed_build_keeps_cpu_last() -> None:
    attempts = provider_attempts("auto")
    names = provider_names(attempts[0])
    assert names[-1] == CPU
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    if TENSORRT in available:
        assert names[0] == TENSORRT
    elif CUDA in available:
        assert names[0] == CUDA

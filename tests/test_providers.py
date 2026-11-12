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


def test_tensorrt_mode_without_nvinfer_keeps_directml(tmp_path: Path) -> None:
    from faceswap.providers import DIRECTML

    attempts = provider_attempts(
        "tensorrt",
        available=[CUDA, DIRECTML, CPU],
        cache_dir=tmp_path,
        platform="win32",
    )
    assert TENSORRT not in provider_names(attempts[0])
    assert provider_names(attempts[0]) == [CUDA, DIRECTML, CPU]
    assert any(DIRECTML in provider_names(attempt) for attempt in attempts)


def test_tensorrt_request_on_directml_only_stays_on_directml(tmp_path: Path) -> None:
    from faceswap.providers import DIRECTML

    attempts = provider_attempts(
        "tensorrt",
        available=[DIRECTML, CPU],
        cache_dir=tmp_path,
        platform="win32",
    )
    assert provider_names(attempts[0]) == [DIRECTML, CPU]
    assert all(TENSORRT not in provider_names(attempt) for attempt in attempts)


def test_mask_session_falls_through_to_directml_when_nvinfer_is_missing(tmp_path: Path, monkeypatch) -> None:
    import onnxruntime

    from faceswap.providers import DIRECTML, open_onnx_session

    calls: list[list[str]] = []

    class _Session:
        def __init__(self, providers) -> None:
            self._providers = [item if isinstance(item, str) else item[0] for item in providers]

        def get_providers(self) -> list[str]:
            return list(self._providers)

    def _session(_path, providers=None):
        names = [item if isinstance(item, str) else item[0] for item in providers]
        calls.append(names)
        if TENSORRT in names or CUDA in names:
            raise RuntimeError("nvinfer_10.dll is not on PATH")
        return _Session(providers)

    monkeypatch.setattr(
        "faceswap.providers.provider_attempts",
        lambda execution, available=None, cache_dir=None, platform=None: [
            [(TENSORRT, {"trt_engine_cache_enable": True}), CUDA, DIRECTML, CPU],
            [CUDA, DIRECTML, CPU],
            [DIRECTML, CPU],
            [CPU],
        ],
    )
    monkeypatch.setattr(onnxruntime, "InferenceSession", _session)
    model = tmp_path / "xseg.onnx"
    model.write_bytes(b"not-a-model")
    _session_obj, _providers, active = open_onnx_session(model, "auto", what="XSeg")
    assert calls[0][0] == TENSORRT
    assert DIRECTML in active
    assert TENSORRT not in active


def test_tensorrt_missing_note_names_nvinfer() -> None:
    from faceswap.providers import tensorrt_missing_note

    note = tensorrt_missing_note([CPU])
    assert "nvinfer_10.dll" in note
    assert tensorrt_missing_note([TENSORRT, CUDA, CPU]) == ""


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

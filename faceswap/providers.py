"""ONNX Runtime execution providers for Videoswa.

NVIDIA Windows preference is TensorRT, then CUDA, then CPU. Session creation
is retried down that list when TensorRT (or CUDA) fails to initialize, which
is the usual result of a missing ``nvinfer`` DLL or a CUDA/TensorRT version
mismatch. CoreML and DirectML are used only when neither TensorRT nor CUDA
is available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

from .utils import MODELS_DIR, logger

ProviderEntry = Union[str, tuple[str, dict]]
ProviderList = list[ProviderEntry]

TENSORRT = "TensorrtExecutionProvider"
CUDA = "CUDAExecutionProvider"
COREML = "CoreMLExecutionProvider"
DIRECTML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"

EXECUTION_AUTO = "auto"
EXECUTION_TENSORRT = "tensorrt"
EXECUTION_CUDA = "cuda"
EXECUTION_CPU = "cpu"
EXECUTION_CHOICES = (
    EXECUTION_AUTO,
    EXECUTION_TENSORRT,
    EXECUTION_CUDA,
    EXECUTION_CPU,
)


def default_trt_cache_dir() -> Path:
    return MODELS_DIR / "trt_cache"


def preload_runtime_libraries() -> None:
    """Load CUDA/cuDNN DLLs shipped as Python wheels, when ORT supports it.

    TensorRT itself is not bundled in the ``onnxruntime-gpu`` wheel. The
    TensorRT ``lib`` directory still has to be on ``PATH`` (Windows) or
    ``LD_LIBRARY_PATH`` (Linux).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return
    preload = getattr(ort, "preload_dlls", None)
    if not callable(preload):
        return
    try:
        preload()
    except Exception as exc:  # library mismatch should not block CPU inference
        logger.warning("onnxruntime.preload_dlls() failed: %s", exc)


def installed_providers() -> list[str]:
    preload_runtime_libraries()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. See the Videoswa README for the "
            "Windows CPU and TensorRT setups."
        ) from exc
    return list(ort.get_available_providers())


def provider_name(entry: ProviderEntry) -> str:
    return entry[0] if isinstance(entry, tuple) else entry


def provider_names(providers: Sequence[ProviderEntry]) -> list[str]:
    return [provider_name(entry) for entry in providers]


def format_providers(providers: Sequence[ProviderEntry]) -> str:
    return " → ".join(provider_names(providers)) or "(none)"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _device_id() -> int:
    try:
        return int(os.environ.get("VIDEOSWA_GPU_DEVICE", "0"))
    except ValueError:
        return 0


def tensorrt_provider_options(cache_dir: Path) -> dict:
    """TensorRT EP options recommended by ONNX Runtime, with an engine cache.

    FP16 is on by default because RTX 4070-class GPUs are built for it.
    Set ``VIDEOSWA_TRT_FP16=0`` to build FP32 engines instead. Workspace
    size is ``VIDEOSWA_TRT_WORKSPACE_MB`` (default 2048).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        workspace_mb = int(os.environ.get("VIDEOSWA_TRT_WORKSPACE_MB", "2048"))
    except ValueError:
        workspace_mb = 2048
    workspace_mb = max(256, workspace_mb)
    return {
        "device_id": _device_id(),
        "trt_max_workspace_size": workspace_mb * 1024 * 1024,
        "trt_fp16_enable": _env_flag("VIDEOSWA_TRT_FP16", True),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_dir),
    }


def cuda_provider_options() -> dict:
    return {
        "device_id": _device_id(),
        "arena_extend_strategy": "kNextPowerOfTwo",
        # HEURISTIC avoids the long exhaustive autotune on every new shape.
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
    }


def _gpu_providers(
    available: Iterable[str],
    *,
    allow_tensorrt: bool,
    allow_cuda: bool,
    cache_dir: Path,
) -> ProviderList:
    present = set(available)
    providers: ProviderList = []
    if allow_tensorrt and TENSORRT in present:
        providers.append((TENSORRT, tensorrt_provider_options(cache_dir)))
    if allow_cuda and CUDA in present:
        providers.append((CUDA, cuda_provider_options()))
    if not providers:
        if DIRECTML in present:
            providers.append(DIRECTML)
        elif COREML in present:
            providers.append(COREML)
    providers.append(CPU)
    return providers


def _dedupe(attempts: list[ProviderList]) -> list[ProviderList]:
    seen: set[tuple] = set()
    unique: list[ProviderList] = []
    for attempt in attempts:
        key = tuple(
            (provider_name(entry),) + (
                tuple(sorted((str(k), repr(v)) for k, v in entry[1].items()))
                if isinstance(entry, tuple) else ()
            )
            for entry in attempt
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(attempt)
    return unique


def provider_attempts(
    execution: str = EXECUTION_AUTO,
    available: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
) -> list[ProviderList]:
    """Provider lists to try, best first.

    ``auto`` and ``tensorrt`` both prefer TensorRT and fall back to CUDA,
    then CPU. ``cuda`` skips TensorRT. ``cpu`` never requests a GPU.
    """
    mode = (execution or EXECUTION_AUTO).strip().lower()
    if mode not in EXECUTION_CHOICES:
        raise ValueError(
            f"Unknown execution mode '{execution}'. "
            f"Choose one of: {', '.join(EXECUTION_CHOICES)}."
        )
    if available is None:
        available = installed_providers()
    cache = cache_dir or default_trt_cache_dir()

    if mode == EXECUTION_CPU:
        return [[CPU]]

    allow_tensorrt = mode in {EXECUTION_AUTO, EXECUTION_TENSORRT}
    allow_cuda = mode in {EXECUTION_AUTO, EXECUTION_TENSORRT, EXECUTION_CUDA}
    present = set(available)

    if mode == EXECUTION_TENSORRT and TENSORRT not in present:
        logger.warning(
            "TensorRT was requested but TensorrtExecutionProvider is not in "
            "this ONNX Runtime build (%s). Falling back to CUDA, then CPU.",
            ", ".join(available) or "no providers",
        )

    attempts: list[ProviderList] = [
        _gpu_providers(
            available,
            allow_tensorrt=allow_tensorrt,
            allow_cuda=allow_cuda,
            cache_dir=cache,
        )
    ]
    if allow_tensorrt and TENSORRT in present:
        attempts.append(
            _gpu_providers(
                available,
                allow_tensorrt=False,
                allow_cuda=allow_cuda,
                cache_dir=cache,
            )
        )
    attempts.append([CPU])
    return _dedupe(attempts)


def uses_gpu(providers: Sequence[ProviderEntry]) -> bool:
    return any(name != CPU for name in provider_names(providers))


def active_providers_from_sessions(sessions: Iterable[object]) -> list[str]:
    found: list[str] = []
    for session in sessions:
        getter = getattr(session, "get_providers", None)
        if not callable(getter):
            continue
        try:
            names = list(getter())
        except Exception:
            continue
        for name in names:
            if name not in found:
                found.append(name)
    return found


def activation_summary(requested: Sequence[ProviderEntry], active: Sequence[str]) -> str:
    requested_names = provider_names(requested)
    active_list = list(active)
    summary = f"requested {format_providers(requested)}"
    if active_list:
        summary += f"; active { ' → '.join(active_list) }"
    if TENSORRT in requested_names and TENSORRT not in active_list:
        summary += ". TensorRT did not stay active, so inference is using the next provider"
    return summary


def run_with_provider_fallback(
    attempts: Sequence[ProviderList],
    factory: Callable[[ProviderList], object],
    *,
    what: str,
) -> tuple[object, ProviderList]:
    """Call ``factory(providers)`` until one attempt initializes.

    ``factory`` may raise to reject a silent GPU fallback (GPU requested,
    session reports CPU only). The final CPU-only attempt is still tried.
    """
    if not attempts:
        raise RuntimeError(f"No execution providers to try for {what}")

    last_error: Optional[BaseException] = None
    for providers in attempts:
        label = format_providers(providers)
        try:
            obj = factory(providers)
        except Exception as exc:
            last_error = exc
            logger.warning("%s failed with %s: %s", what, label, exc)
            continue
        logger.info("%s is using %s", what, label)
        return obj, providers

    raise RuntimeError(
        f"Could not start {what}. On an RTX 4070, install onnxruntime-gpu, "
        "CUDA 12.x, and TensorRT 10.x, and put the TensorRT lib folder on PATH "
        "(see the Videoswa README). CPU-only ONNX Runtime is the fallback when "
        "those libraries are missing."
    ) from last_error

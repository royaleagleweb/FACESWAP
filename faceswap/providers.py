"""ONNX Runtime execution providers for Videoswa.

NVIDIA Windows preference is TensorRT, then CUDA, then DirectML, then CPU.
Session creation is retried down that list when TensorRT or CUDA fails to
initialize. DirectML is the fast Windows path when CUDA's cuDNN frontend
is unhealthy. CoreML is used on macOS when no NVIDIA provider is present.
"""

from __future__ import annotations

import os
import sys
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
EXECUTION_DIRECTML = "directml"
EXECUTION_CPU = "cpu"
CUDA_FAILED_STATUS = "CUDA failed; using CPU"
DIRECTML_FALLBACK_STATUS = "CUDA failed; using DirectML"
EXECUTION_CHOICES = (
    EXECUTION_AUTO,
    EXECUTION_TENSORRT,
    EXECUTION_CUDA,
    EXECUTION_DIRECTML,
    EXECUTION_CPU,
)


def default_trt_cache_dir() -> Path:
    return MODELS_DIR / "trt_cache"


# cuDNN 9's frontend conv path (ORT 1.20+) can create a CUDA session and then
# fail inside Conv with ``CUDNN_FE failure 7: GRAPH_EXECUTION_FAILED``.
# ONNX Runtime does not expose a provider option that turns that frontend off.
# ``ORT_DISABLE_CUDNN_FRONTEND=1`` is the switch newer builds and Windows
# launchers honor; ``cudnn_conv_algo_search=DEFAULT`` is the fallback heuristic
# the CUDA provider does expose.
_CUDA_RUNTIME_MARKERS = (
    "CUDNN_FE",
    "GRAPH_EXECUTION_FAILED",
    "FAILED TO INITIALIZE CUDNN",
    "CUDNN_STATUS",
    "CUDA FAILURE",
    "CUDA_ERROR",
    "CUBLAS_STATUS",
)


def configure_cuda_runtime() -> None:
    """Prefer the cuDNN path that does not use the broken frontend graph.

    Set ``ORT_DISABLE_CUDNN_FRONTEND=0`` before startup to keep the faster
    HEURISTIC search. The variable is applied before ONNX Runtime is imported.
    """
    os.environ.setdefault("ORT_DISABLE_CUDNN_FRONTEND", "1")


def cudnn_frontend_disabled() -> bool:
    configure_cuda_runtime()
    return _env_flag("ORT_DISABLE_CUDNN_FRONTEND", True)


def is_cuda_runtime_failure(exc: BaseException) -> bool:
    """True when a session already started and a CUDA/cuDNN kernel then failed."""
    text = f"{type(exc).__name__}: {exc}".upper()
    return any(marker in text for marker in _CUDA_RUNTIME_MARKERS)


class CudaRuntimeGuard:
    """Rebuild every attached session on CPU after a mid-graph CUDA failure."""

    def __init__(self) -> None:
        self.members: list[object] = []
        self.note: Optional[str] = None
        self.on_fallback: Optional[Callable[[str], None]] = None

    def attach(self, member: object) -> None:
        if member not in self.members:
            self.members.append(member)
        setattr(member, "cuda_guard", self)

    def recover(self, exc: BaseException) -> bool:
        if not is_cuda_runtime_failure(exc):
            return False
        if self.note == CUDA_FAILED_STATUS:
            return False
        if not any(_member_uses_gpu(member) for member in self.members):
            return False
        try:
            available = installed_providers()
        except Exception:
            available = [CPU]
        mode = preferred_fallback_mode(self._current_providers(), available)
        logger.warning("%s. %s", _fallback_note(mode), exc)
        for member in self.members:
            switched = False
            if mode != EXECUTION_CPU:
                adopt_mode = getattr(member, "adopt_execution", None)
                if callable(adopt_mode):
                    adopt_mode(mode)
                    switched = True
            if not switched:
                adopt = getattr(member, "adopt_cpu", None)
                if callable(adopt):
                    adopt()
        self.note = _note_for_members(self.members, mode)
        if self.on_fallback is not None:
            self.on_fallback(self.note)
        return True

    def _current_providers(self) -> Sequence[ProviderEntry]:
        for member in self.members:
            providers = getattr(member, "providers", None)
            if providers:
                return providers
        return [CPU]


def _member_uses_gpu(member: object) -> bool:
    providers = getattr(member, "providers", None)
    if not providers:
        return False
    return uses_gpu(providers)


def run_with_cuda_fallback(guard: Optional[CudaRuntimeGuard], fn: Callable[[], object]) -> object:
    """Run ``fn`` once, and once more on CPU if CUDA fails inside the graph."""
    try:
        return fn()
    except Exception as exc:
        if guard is None or not guard.recover(exc):
            raise
    return fn()


def preload_runtime_libraries() -> None:
    """Load CUDA/cuDNN DLLs shipped as Python wheels, when ORT supports it.

    TensorRT itself is not bundled in the ``onnxruntime-gpu`` wheel. The
    TensorRT ``lib`` directory still has to be on ``PATH`` (Windows) or
    ``LD_LIBRARY_PATH`` (Linux).
    """
    configure_cuda_runtime()
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


def _is_windows(platform: Optional[str] = None) -> bool:
    return (platform or sys.platform).startswith("win")


def preferred_fallback_mode(
    current: Sequence[ProviderEntry],
    available: Sequence[str],
    platform: Optional[str] = None,
) -> str:
    """Next execution mode after TensorRT or CUDA fails while the graph is running.

    Windows prefers DirectML when that provider is installed. Otherwise CPU.
    """
    names = set(provider_names(current))
    if names & {TENSORRT, CUDA} and DIRECTML in set(available) and _is_windows(platform):
        return EXECUTION_DIRECTML
    return EXECUTION_CPU


def _fallback_note(mode: str) -> str:
    if mode == EXECUTION_DIRECTML:
        return DIRECTML_FALLBACK_STATUS
    return CUDA_FAILED_STATUS


def _note_for_members(members: Sequence[object], requested_mode: str) -> str:
    for member in members:
        providers = getattr(member, "providers", None)
        if not providers:
            continue
        device = active_device_name(providers)
        if device == "DirectML":
            return DIRECTML_FALLBACK_STATUS
        if device == "CPU":
            return CUDA_FAILED_STATUS
        return f"CUDA failed; using {device}"
    return _fallback_note(requested_mode)


def active_device_name(providers: Sequence) -> str:
    """Short name of the provider that will actually run."""
    names = list(providers)
    if names and not isinstance(names[0], str):
        names = provider_names(names)
    for key, label in (
        (TENSORRT, "TensorRT"),
        (CUDA, "CUDA"),
        (DIRECTML, "DirectML"),
        (COREML, "CoreML"),
        (CPU, "CPU"),
    ):
        if key in names:
            return label
    return "CPU"


def provider_status_text(providers: Sequence) -> str:
    """Banner copy for the desktop window."""
    name = active_device_name(providers)
    if name == "CPU":
        return (
            "Running on CPU — this is slow. Choose Half resolution for a faster "
            "export, or on Windows install onnxruntime-directml "
            "(requirements-windows-directml.txt)."
        )
    if name == "DirectML":
        return "Running on DirectML."
    return f"Running on {name}."


def cuda_provider_options() -> dict:
    # DEFAULT selects cuDNN frontend HeurMode FALLBACK, which is the least
    # aggressive conv path this ONNX Runtime build exposes. HEURISTIC is
    # faster when the frontend graph actually runs.
    search = "DEFAULT" if cudnn_frontend_disabled() else "HEURISTIC"
    return {
        "device_id": _device_id(),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": search,
        "do_copy_in_default_stream": True,
    }


def _gpu_providers(
    available: Iterable[str],
    *,
    allow_tensorrt: bool,
    allow_cuda: bool,
    allow_directml: bool,
    cache_dir: Path,
    platform: Optional[str] = None,
) -> ProviderList:
    present = set(available)
    providers: ProviderList = []
    if allow_tensorrt and TENSORRT in present:
        providers.append((TENSORRT, tensorrt_provider_options(cache_dir)))
    if allow_cuda and CUDA in present:
        providers.append((CUDA, cuda_provider_options()))
    nvidia = any(provider_name(entry) in {TENSORRT, CUDA} for entry in providers)
    # On Windows, DirectML stays behind CUDA so a broken cuDNN path can fall
    # through without going straight to CPU. Elsewhere DirectML is only used
    # when TensorRT and CUDA are not available.
    if allow_directml and DIRECTML in present and (_is_windows(platform) or not nvidia):
        providers.append(DIRECTML)
    elif not providers and COREML in present:
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
    platform: Optional[str] = None,
) -> list[ProviderList]:
    """Provider lists to try, best first.

    On Windows, ``auto`` is TensorRT, then CUDA, then DirectML, then CPU.
    ``cuda`` skips TensorRT. ``directml`` skips NVIDIA. ``cpu`` is only CPU.
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
    plat = platform or sys.platform
    present = set(available)

    if mode == EXECUTION_CPU:
        return [[CPU]]
    if mode == EXECUTION_DIRECTML:
        if DIRECTML not in present:
            logger.warning(
                "DirectML was requested but DmlExecutionProvider is not installed. "
                "Install onnxruntime-directml (see requirements-windows-directml.txt)."
            )
            return [[CPU]]
        return _dedupe([[DIRECTML, CPU], [CPU]])

    allow_tensorrt = mode in {EXECUTION_AUTO, EXECUTION_TENSORRT}
    allow_cuda = mode in {EXECUTION_AUTO, EXECUTION_TENSORRT, EXECUTION_CUDA}
    allow_directml = mode in {EXECUTION_AUTO, EXECUTION_TENSORRT, EXECUTION_CUDA, EXECUTION_DIRECTML}

    if mode == EXECUTION_TENSORRT and TENSORRT not in present:
        logger.warning(
            "TensorRT was requested but TensorrtExecutionProvider is not in "
            "this ONNX Runtime build (%s). Falling back to CUDA, then DirectML, then CPU.",
            ", ".join(available) or "no providers",
        )

    def _attempt(use_trt: bool, use_cuda: bool) -> ProviderList:
        return _gpu_providers(
            available,
            allow_tensorrt=use_trt,
            allow_cuda=use_cuda,
            allow_directml=allow_directml,
            cache_dir=cache,
            platform=plat,
        )

    attempts: list[ProviderList] = [_attempt(allow_tensorrt, allow_cuda)]
    if allow_tensorrt and TENSORRT in present:
        attempts.append(_attempt(False, allow_cuda))
    if allow_cuda and CUDA in present and allow_directml and DIRECTML in present and _is_windows(plat):
        attempts.append(_attempt(False, False))
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
        "CUDA 12.x, and TensorRT 10.x, and put the TensorRT lib folder on PATH. "
        "If CUDA fails inside a convolution, install onnxruntime-directml instead "
        "(requirements-windows-directml.txt). CPU is the last fallback."
    ) from last_error

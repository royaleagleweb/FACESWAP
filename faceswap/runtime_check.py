"""Startup check for the Windows stack that makes detection return zero faces."""

from __future__ import annotations

from typing import Optional

NUMPY_PIN = "numpy==1.26.4"
OPENCV_PIN = "opencv-python==4.10.0.84"
INSIGHTFACE_WHEEL = (
    "https://github.com/Gourieff/Assets/raw/main/Insightface/"
    "insightface-0.7.3-cp311-cp311-win_amd64.whl"
)


def environment_problems(
    numpy_version: Optional[str] = None,
    opencv_version: Optional[str] = None,
    headless_installed: Optional[bool] = None,
) -> list[str]:
    """Problems that make InsightFace detect nothing.

    Pass versions in tests. When they are omitted, the installed packages are read.
    An empty list means this process can try to detect faces.
    """
    problems: list[str] = []
    numpy_text = numpy_version if numpy_version is not None else _numpy_version()
    major = _major(numpy_text)
    if major is not None and major >= 2:
        problems.append(
            f"numpy {numpy_text} is installed. The Windows InsightFace wheel was built "
            "against numpy 1.x. numpy 2 raises 'numpy.dtype size changed' and detection stops. "
            f"Run: pip install {NUMPY_PIN}"
        )

    opencv_text = opencv_version if opencv_version is not None else _opencv_version()
    if _opencv_is_broken(opencv_text):
        problems.append(
            f"OpenCV {opencv_text} is installed. Face detection needs {OPENCV_PIN}. "
            "OpenCV 4.11 and OpenCV 5, including the 5.0.0 build GFPGAN or albumentations "
            "can pull in, make InsightFace return zero faces. "
            f"Run: pip install {OPENCV_PIN}"
        )

    headless = headless_installed if headless_installed is not None else _headless_installed()
    if headless:
        problems.append(
            "opencv-python-headless is installed. Albumentations pulls headless 5.0, "
            "which breaks detection even when opencv-python is also present. "
            f"Run: pip uninstall -y opencv-python-headless && pip install {OPENCV_PIN}"
        )
    return problems


def environment_message(problems: Optional[list[str]] = None) -> str:
    rows = environment_problems() if problems is None else problems
    if not rows:
        return ""
    body = "\n\n".join(rows)
    return (
        "Videoswa cannot detect faces with this Python environment.\n\n"
        f"{body}\n\n"
        "On the Lenovo CUDA install, use requirements-windows-cuda.txt and the "
        f"prebuilt wheel {INSIGHTFACE_WHEEL}"
    )


def _major(version: Optional[str]) -> Optional[int]:
    if not version:
        return None
    token = str(version).split(".")[0]
    try:
        return int(token)
    except ValueError:
        return None


def _opencv_is_broken(version: Optional[str]) -> bool:
    if not version:
        return False
    parts = str(version).split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return False
    return major > 4 or (major == 4 and minor >= 11)


def _numpy_version() -> Optional[str]:
    try:
        import numpy
    except ImportError:
        return None
    return str(numpy.__version__)


def _opencv_version() -> Optional[str]:
    try:
        import cv2
    except ImportError:
        return None
    return str(cv2.__version__)


def _headless_installed() -> bool:
    try:
        from importlib.metadata import distributions
    except ImportError:
        distributions = None  # type: ignore[assignment]
    if distributions is not None:
        try:
            for dist in distributions():
                name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
                if name == "opencv-python-headless":
                    return True
        except Exception:
            pass
    try:
        import cv2
    except ImportError:
        return False
    path = str(getattr(cv2, "__file__", "")).lower().replace("_", "-")
    return "opencv-python-headless" in path

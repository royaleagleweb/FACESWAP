"""Video multi-face faceswap toolkit."""

from .core import FaceSwapEngine, FaceMapping
from .face_analyzer import FaceAnalyzer, Face
from .swapper import FaceSwapper

__version__ = "0.1.0"
__all__ = ["FaceSwapEngine", "FaceMapping", "FaceAnalyzer", "Face", "FaceSwapper"]

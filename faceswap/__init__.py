"""Videoswa face-swap engine: InsightFace buffalo_l + inswapper_128."""

from .core import FaceMapping, FaceSwapEngine
from .face_analyzer import Face, FaceAnalyzer
from .swapper import FaceSwapper

__version__ = "1.0.0"
__all__ = ["FaceSwapEngine", "FaceMapping", "FaceAnalyzer", "Face", "FaceSwapper"]

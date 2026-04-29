"""Face detection + embedding via InsightFace's buffalo_l pack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .utils import logger, select_providers


@dataclass
class Face:
    """A detected face with its identity embedding and geometry."""
    bbox: np.ndarray            # (4,) xyxy
    kps: np.ndarray             # (5, 2) landmarks
    embedding: np.ndarray       # (512,)
    det_score: float
    gender: Optional[int] = None
    age: Optional[int] = None

    @property
    def normed_embedding(self) -> np.ndarray:
        norm = np.linalg.norm(self.embedding) + 1e-8
        return self.embedding / norm

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class FaceAnalyzer:
    """Wraps insightface.FaceAnalysis for detection + embedding extraction."""

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        use_gpu: bool = True,
        det_thresh: float = 0.5,
    ) -> None:
        from insightface.app import FaceAnalysis  # local import; heavy

        providers = select_providers(use_gpu=use_gpu)
        logger.info("FaceAnalyzer providers: %s", providers)

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size, det_thresh=det_thresh)

    def analyze(self, image_bgr: np.ndarray) -> List[Face]:
        """Detect every face in an image and return them sorted left-to-right."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        raw = self.app.get(image_bgr)
        faces: List[Face] = []
        for f in raw:
            faces.append(
                Face(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32),
                    embedding=np.asarray(f.normed_embedding * np.linalg.norm(f.embedding), dtype=np.float32),
                    det_score=float(getattr(f, "det_score", 0.0)),
                    gender=getattr(f, "gender", None),
                    age=getattr(f, "age", None),
                )
            )
        faces.sort(key=lambda x: x.bbox[0])  # left to right
        return faces

    def best_face(self, image_bgr: np.ndarray) -> Optional[Face]:
        """Return the largest face — useful for source reference images."""
        faces = self.analyze(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

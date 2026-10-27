"""Side and profile faces stay in the list, match, and get a full paste."""

from __future__ import annotations

import inspect

import numpy as np

from faceswap.core import FaceMapping, FaceSwapEngine
from faceswap.coverage import coverage_alpha, face_axes, paste_swapped_face, template_128
from faceswap.face_analyzer import Face, FaceAnalyzer, cosine_similarity
from faceswap.pose import face_yaw, pose_name, repair_landmarks
from faceswap.swapper import FaceSwapper
from faceswap.utils import BISENET_SHA256, GFPGAN_SHA256, XSEG_SHA256, optional_model_report


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    count = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    variance = (src_c ** 2).sum() / count
    cov = (dst_c.T @ src_c) / count
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(singular) @ sign) / variance
    translation = mu_dst - scale * (rotation @ mu_src)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def _frontal() -> np.ndarray:
    dst = template_128()
    scale = 2.2
    center = dst.mean(axis=0)
    return (dst - center) * scale + np.array([220.0, 200.0])


def _profile() -> np.ndarray:
    """Collapse the far eye and mouth, and push the nose out to one side."""
    base = _frontal()
    axes = face_axes(base)
    assert axes is not None
    eye_c, mouth_c, down_u, side_u, em, _eye = axes
    pts = base.copy()
    far = eye_c - side_u * (0.02 * em)
    pts[0] = far
    pts[1] = far + side_u * (0.12 * em)
    pts[2] = eye_c + down_u * (0.35 * em) + side_u * (0.62 * em)
    near = mouth_c + side_u * (0.08 * em)
    pts[4] = near + side_u * (0.12 * em)
    pts[3] = near
    return pts.astype(np.float32)


def _face(embedding: np.ndarray, kps: np.ndarray, x: float = 40.0) -> Face:
    vector = np.asarray(embedding, dtype=np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    return Face(
        bbox=np.array([x, 30, x + 90, 150], dtype=np.float32),
        kps=np.asarray(kps, dtype=np.float32),
        embedding=vector,
        det_score=0.31,
        gender=1,
    )


class _Analyzer:
    def __init__(self) -> None:
        self.faces: list[Face] = []

    def analyze(self, _frame):
        return list(self.faces)


class _Swapper:
    def swap(self, frame, target_face, source_face, paste_back=True, coverage="full"):
        self.target = target_face
        return frame


def test_detector_keeps_a_low_score_profile() -> None:
    assert inspect.signature(FaceAnalyzer.__init__).parameters["det_thresh"].default == 0.30
    person = _face(np.array([1.0, 0.0, 0.0]), _profile())
    assert person.det_score < 0.5
    assert pose_name(face_yaw(person.kps)) == "Profile"


def test_profile_landmarks_repair_and_cover_the_cheek() -> None:
    raw = _profile()
    yaw = face_yaw(raw)
    assert yaw > 0.62
    assert pose_name(yaw) == "Profile"
    assert abs(face_yaw(_frontal())) < 0.28

    repaired = repair_landmarks(raw)
    raw_axes = face_axes(raw)
    fixed_axes = face_axes(repaired)
    assert raw_axes is not None and fixed_axes is not None
    assert fixed_axes[5] > raw_axes[5] * 1.8

    shape = (480, 520)
    mask = coverage_alpha(shape, repaired, "full", yaw=face_yaw(repaired))
    nose = raw[2].astype(int)
    assert mask[nose[1], nose[0]] > 0.5
    jaw = (raw[2] + np.array([0.0, 55.0])).astype(int)
    assert mask[jaw[1], jaw[0]] > 0.5
    assert mask[0, 0] == 0.0

    matrix = _umeyama(repaired, template_128())
    frame = np.full((*shape, 3), 20, dtype=np.uint8)
    swapped = np.full((128, 128, 3), (0, 220, 0), dtype=np.uint8)
    pasted = paste_swapped_face(frame, swapped, matrix, repaired, "full", yaw=face_yaw(repaired))
    assert int(pasted[nose[1], nose[0], 1]) > 150
    assert int(pasted[jaw[1], jaw[0], 1]) > 150
    assert int(pasted[0, 0, 1]) < 40


def test_profile_face_locks_when_a_frontal_match_would_not() -> None:
    reference = _face(np.array([1.0, 0.0, 0.0]), _frontal(), x=40)
    turned = np.array([0.24, 0.97, 0.0], dtype=np.float32)
    assert 0.22 <= cosine_similarity(reference.normed_embedding, turned) < 0.32
    frame = np.zeros((80, 200, 3), dtype=np.uint8)
    mapping = FaceMapping(source_face=reference, reference_face=reference)

    frontal_engine = FaceSwapEngine(analyzer=_Analyzer(), swapper=_Swapper(), similarity_threshold=0.32)
    frontal = _face(turned, _frontal(), x=40)
    frontal_engine.analyzer.faces = [frontal]
    frontal_engine.process_frame(frame, [mapping])
    assert frontal_engine.stats.faces_swapped == 0

    profile_engine = FaceSwapEngine(analyzer=_Analyzer(), swapper=_Swapper(), similarity_threshold=0.32)
    profile = _face(turned, _profile(), x=40)
    profile_engine.analyzer.faces = [profile]
    profile_engine.process_frame(frame, [mapping])
    assert profile_engine.stats.faces_swapped == 1


def test_turning_to_profile_keeps_the_locked_source() -> None:
    reference = _face(np.array([1.0, 0.0, 0.0]), _frontal())
    first = _face(np.array([0.97, 0.24, 0.0]), _frontal())
    # Far from the reference photo, closer to the running track, and in profile.
    turned_vec = np.array([0.12, 0.99, 0.05], dtype=np.float32)
    turned = _face(turned_vec, _profile())
    assert cosine_similarity(reference.normed_embedding, turned.normed_embedding) < 0.20
    assert cosine_similarity(first.normed_embedding, turned.normed_embedding) >= 0.34
    assert pose_name(face_yaw(turned.kps)) == "Profile"

    analyzer = _Analyzer()
    swapper = _Swapper()
    engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper, similarity_threshold=0.32)
    mapping = FaceMapping(source_face=reference, reference_face=reference)
    frame = np.zeros((80, 200, 3), dtype=np.uint8)
    analyzer.faces = [first]
    engine.process_frame(frame, [mapping])
    analyzer.faces = [turned]
    engine.process_frame(frame, [mapping])
    assert engine.stats.faces_swapped == 2
    assert engine.stats.faces_unmatched == 0


def test_swap_repairs_profile_landmarks_before_paste() -> None:
    source = inspect.getsource(FaceSwapper.swap)
    assert "repair_landmarks" in source
    assert "_occ_mask(" in source


def test_optional_models_have_checksums_and_a_status_line() -> None:
    assert len(XSEG_SHA256) == 64
    assert len(BISENET_SHA256) == 64
    assert len(GFPGAN_SHA256) == 64
    report = optional_model_report()
    assert "XSeg" in report
    assert "BiSeNet" in report
    assert "GFPGAN" in report

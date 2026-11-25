"""Each detected person keeps their own source identity."""

import numpy as np
import pytest

from faceswap.core import FaceMapping, FaceSwapEngine
from faceswap.face_analyzer import Face, cosine_similarity


def _face(embedding: list[float], x: float) -> Face:
    return Face(
        bbox=np.array([x, 0, x + 20, 30], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        embedding=np.array(embedding, dtype=np.float32),
        det_score=0.99,
    )


class _Analyzer:
    def __init__(self, faces: list[Face]) -> None:
        self._faces = faces

    def analyze(self, _frame):
        return list(self._faces)


class _Swapper:
    def __init__(self) -> None:
        self.pairs: list[tuple[Face, Face]] = []

    def swap(self, frame, target_face, source_face, paste_back=True, coverage="full"):
        self.pairs.append((target_face, source_face))
        return frame


def test_two_references_swap_independently() -> None:
    alice_ref = _face([1, 0, 0], 0)
    bob_ref = _face([0, 1, 0], 40)
    alice_src = _face([0.2, 0.1, 0], 0)
    bob_src = _face([0.1, 0.3, 0], 0)
    detected = [_face([1, 0, 0], 5), _face([0, 1, 0], 50)]
    swapper = _Swapper()
    engine = FaceSwapEngine(analyzer=_Analyzer(detected), swapper=swapper, similarity_threshold=0.45)
    frame = np.zeros((32, 80, 3), dtype=np.uint8)
    engine.process_frame(
        frame,
        [
            FaceMapping(source_face=alice_src, reference_face=alice_ref, label="alice"),
            FaceMapping(source_face=bob_src, reference_face=bob_ref, label="bob"),
        ],
    )
    assert len(swapper.pairs) == 2
    assert swapper.pairs[0][1] is alice_src
    assert swapper.pairs[1][1] is bob_src
    assert engine.stats.faces_swapped == 2
    assert engine.stats.faces_unmatched == 0


def test_tiny_extra_face_is_skipped_when_another_person_is_present() -> None:
    big = _face([1, 0, 0], 0)
    big.bbox = np.array([0, 0, 90, 110], dtype=np.float32)
    tiny = _face([0, 1, 0], 120)
    swapper = _Swapper()
    engine = FaceSwapEngine(
        analyzer=_Analyzer([big, tiny]),
        swapper=swapper,
        similarity_threshold=0.2,
    )
    engine.skip_tiny_px = 40
    engine.apply_to_all_when_no_reference = True
    engine.process_frame(
        np.zeros((160, 200, 3), dtype=np.uint8),
        [FaceMapping(source_face=_face([0.2, 0.2, 0.2], 0), reference_face=None)],
    )
    assert len(swapper.pairs) == 1
    assert engine.stats.faces_skipped == 1
    assert engine.stats.faces_swapped == 1


def test_identity_strength_mixes_embeddings() -> None:
    from faceswap.pixel_boost import apply_identity_strength

    source = _face([1, 0, 0], 0)
    target = _face([0, 1, 0], 0)
    mixed = apply_identity_strength(source, target, 0.0)
    assert mixed.normed_embedding[1] > 0.9
    same = apply_identity_strength(source, target, 1.0)
    assert same is source


def test_256_paste_changes_the_face_box() -> None:
    from faceswap.coverage import paste_swapped_face, template_128
    from faceswap.pixel_boost import frame_to_crop_matrix
    from faceswap.swapper import _box_mean_abs

    kps = (template_128() - template_128().mean(axis=0)) * 3.0 + np.array([400.0, 300.0])
    frame = np.full((600, 800, 3), 20, dtype=np.uint8)
    matrix = frame_to_crop_matrix(kps, 256)
    swapped = np.full((256, 256, 3), (0, 0, 220), dtype=np.uint8)
    out = paste_swapped_face(frame, swapped, matrix, kps, "full", color_match=0)
    bbox = np.array(
        [
            float(kps[:, 0].min()) - 8,
            float(kps[:, 1].min()) - 8,
            float(kps[:, 0].max()) + 8,
            float(kps[:, 1].max()) + 8,
        ],
        dtype=np.float32,
    )
    assert _box_mean_abs(frame, out, bbox) > 1.5


def test_mask_matrix_scales_a_256_canvas_into_a_128_input() -> None:
    from faceswap.occlusion import _scale_matrix

    matrix = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float64)
    scaled = _scale_matrix(matrix, 256, (128, 128))
    assert scaled[0, 2] == pytest.approx(5.0)
    assert scaled[1, 2] == pytest.approx(10.0)


def test_busy_gpu_does_not_count_as_a_batch_win() -> None:
    from faceswap.swapper import batch_decision

    # The laptop log compared a 27.6 ms batch with a 249.7 ms pair of singles
    # while another swap was using the GPU. That must not be saved as "on".
    assert batch_decision(249.7 / 2.0, 27.6) == "skip"
    assert batch_decision(8.0, 12.0) == "on"
    assert batch_decision(8.0, 20.0) == "off"


def test_face_batch_stays_off_without_tensorrt() -> None:
    from faceswap.swapper import face_batch_session

    class _Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class _Model:
        session = _Session()

    assert face_batch_session(_Model()) is None


def test_unmatched_person_is_left_alone() -> None:
    alice_ref = _face([1, 0, 0], 0)
    alice_src = _face([1, 0, 0], 0)
    stranger = _face([0, 0, 1], 40)
    swapper = _Swapper()
    engine = FaceSwapEngine(
        analyzer=_Analyzer([alice_ref, stranger]),
        swapper=swapper,
        similarity_threshold=0.45,
    )
    engine.process_frame(np.zeros((8, 8, 3), dtype=np.uint8), [
        FaceMapping(source_face=alice_src, reference_face=alice_ref),
    ])
    assert len(swapper.pairs) == 1
    assert engine.stats.faces_unmatched == 1
    assert cosine_similarity(alice_ref.normed_embedding, stranger.normed_embedding) < 0.45

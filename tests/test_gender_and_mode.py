"""Gender labels and the optional single-face vs multi-face mapping path."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from faceswap.core import FaceMapping
from faceswap.face_analyzer import Face, coerce_gender, face_label, format_gender
from videoswa.jobs import FACE_MODE_MULTIPLE, FACE_MODE_SINGLE, FaceSource, SwapRequest, build_mappings, validate_request
from videoswa.worker import DetectedPerson


def _face(embedding: list[float], x: float, gender: int | None = None, width: float = 20, height: float = 30) -> Face:
    return Face(
        bbox=np.array([x, 0, x + width, height], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        embedding=np.array(embedding, dtype=np.float32),
        det_score=0.99,
        gender=gender,
    )


def _write_image(path: Path) -> None:
    assert cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (80, 48))
    assert writer.isOpened()
    frame = np.full((48, 80, 3), 30, dtype=np.uint8)
    for _ in range(4):
        writer.write(frame)
    writer.release()


def test_gender_labels_follow_insightface_codes() -> None:
    assert coerce_gender(0) == 0
    assert coerce_gender(1) == 1
    assert coerce_gender(np.int64(1)) == 1
    assert coerce_gender(np.array([0])) == 0
    assert coerce_gender(None) is None
    assert coerce_gender(2) is None
    assert coerce_gender("male") is None
    assert format_gender(1) == "Male"
    assert format_gender(0) == "Female"
    assert format_gender(None) == "Unknown"
    assert face_label(0, 1) == "Face 1 — Male"
    assert face_label(2, 0) == "Face 3 — Female"
    male = _face([1, 0, 0], 0, gender=1)
    assert male.gender_label == "Male"


class _Analyzer:
    def __init__(self, face: Face) -> None:
        self.face = face

    def best_face(self, _image):
        return self.face


class _Engine:
    def __init__(self, source: Face) -> None:
        self.analyzer = _Analyzer(source)
        self.wildcard_calls = 0

    def build_mapping(self, source_image, reference_image_bgr=None, label=""):
        self.wildcard_calls += 1
        assert reference_image_bgr is None
        return FaceMapping(source_face=self.analyzer.face, reference_face=None, label=label)


def test_single_mode_maps_the_selected_face_only(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    _write_image(image)
    selected = _face([1, 0, 0], 10, gender=1)
    source = _face([0, 1, 0], 0, gender=0)
    engine = _Engine(source)
    request = SwapRequest(
        video_path=tmp_path / "clip.mp4",
        output_path=tmp_path / "out.mp4",
        face_mode=FACE_MODE_SINGLE,
        single_source=image,
        selected_face=selected,
        apply_to_all=False,
    )
    mappings = build_mappings(engine, request)
    assert engine.wildcard_calls == 0
    assert len(mappings) == 1
    assert mappings[0].reference_face is selected
    assert mappings[0].source_face is source
    assert mappings[0].label == "selected"


def test_single_apply_to_all_is_a_wildcard(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    _write_image(image)
    source = _face([0, 1, 0], 0)
    engine = _Engine(source)
    request = SwapRequest(
        video_path=tmp_path / "clip.mp4",
        output_path=tmp_path / "out.mp4",
        face_mode=FACE_MODE_SINGLE,
        single_source=image,
        selected_face=None,
        apply_to_all=True,
    )
    mappings = build_mappings(engine, request)
    assert engine.wildcard_calls == 1
    assert len(mappings) == 1
    assert mappings[0].reference_face is None
    assert mappings[0].label == "all"


def test_multiple_mode_keeps_one_mapping_per_source(tmp_path: Path) -> None:
    alice_img = tmp_path / "alice.png"
    bob_img = tmp_path / "bob.png"
    _write_image(alice_img)
    _write_image(bob_img)
    alice_ref = _face([1, 0, 0], 0, gender=0)
    bob_ref = _face([0, 1, 0], 40, gender=1)
    source = _face([0, 0, 1], 0)
    engine = _Engine(source)
    request = SwapRequest(
        video_path=tmp_path / "clip.mp4",
        output_path=tmp_path / "out.mp4",
        face_mode=FACE_MODE_MULTIPLE,
        face_sources=[
            FaceSource(face=alice_ref, source_path=alice_img, label=face_label(0, 0)),
            FaceSource(face=bob_ref, source_path=bob_img, label=face_label(1, 1)),
        ],
    )
    mappings = build_mappings(engine, request)
    assert engine.wildcard_calls == 0
    assert [item.reference_face for item in mappings] == [alice_ref, bob_ref]
    assert [item.label for item in mappings] == ["Face 1 — Female", "Face 2 — Male"]


def test_validate_single_and_multiple_requirements(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video)
    image = tmp_path / "source.png"
    _write_image(image)
    output = tmp_path / "out.mp4"

    with pytest.raises(ValueError, match="Choose one source image"):
        validate_request(
            SwapRequest(video_path=video, output_path=output, face_mode=FACE_MODE_SINGLE)
        )
    with pytest.raises(ValueError, match="select the face"):
        validate_request(
            SwapRequest(
                video_path=video,
                output_path=output,
                face_mode=FACE_MODE_SINGLE,
                single_source=image,
            )
        )
    validate_request(
        SwapRequest(
            video_path=video,
            output_path=output,
            face_mode=FACE_MODE_SINGLE,
            single_source=image,
            selected_face=_face([1, 0, 0], 0, gender=1),
        )
    )
    with pytest.raises(ValueError, match="at least one person"):
        validate_request(
            SwapRequest(video_path=video, output_path=output, face_mode=FACE_MODE_MULTIPLE)
        )


def test_window_defaults_to_single_face_and_labels_gender(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from videoswa.desktop import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    crop = np.zeros((32, 32, 3), dtype=np.uint8)
    female = _face([1, 0, 0], 0, gender=0, width=20, height=20)
    male = _face([0, 1, 0], 40, gender=1, width=40, height=50)
    try:
        assert window.face_mode.currentData() == FACE_MODE_SINGLE
        assert window.face_mode.currentText() == "Single face"
        assert window.apply_all.isChecked() is False
        assert window.apply_all.isVisible()
        assert window.coverage.currentData() == "full"
        window._on_detected(
            [
                DetectedPerson(crop_bgr=crop, face=female, index=0),
                DetectedPerson(crop_bgr=crop, face=male, index=1),
            ]
        )
        app.processEvents()
        assert [card.title.text() for card in window._cards] == [
            "Face 1 — Female",
            "Face 2 — Male",
        ]
        assert window._selected_index == 1
        assert window._cards[1]._selected
        assert window._cards[0].hint.text() == "Click to select"
        assert window._cards[1].hint.text() == "Selected"
        assert not window._cards[0].choose_btn.isVisible()
        window.face_mode.setCurrentIndex(1)
        app.processEvents()
        assert window.face_mode.currentData() == FACE_MODE_MULTIPLE
        assert window._cards[0].choose_btn.isVisible()
        assert window._cards[1].choose_btn.isVisible()
        assert not window.single_btn.isVisible()
        window.face_mode.setCurrentIndex(0)
        window._select_face(0)
        app.processEvents()
        assert window._selected_index == 0
        assert window._cards[0].hint.text() == "Selected"
        assert window._selected_face() is female
    finally:
        window.close()
        window.worker.wait(3000)

import numpy as np
import pytest

from agegender import AGE_BUCKETS, GENDERS, Prediction, face_boxes, format_label, pad_box, top_label


def _dets(*rows):
    return np.array(rows, dtype=np.float32).reshape(1, 1, len(rows), 7)


def test_face_boxes_threshold_and_scaling():
    dets = _dets([0, 1, 0.95, 0.1, 0.2, 0.5, 0.6],
                 [0, 1, 0.50, 0.0, 0.0, 0.3, 0.3])   # below threshold
    assert face_boxes(dets, 640, 480) == [(64, 96, 320, 288)]


def test_face_boxes_clips_and_drops_degenerate():
    dets = _dets([0, 1, 0.9, -0.2, -0.1, 1.3, 1.2],   # partly outside the frame -> clipped
                 [0, 1, 0.9, 0.5, 0.5, 0.5, 0.9])     # zero width -> dropped
    assert face_boxes(dets, 100, 50) == [(0, 0, 99, 49)]


def test_face_boxes_empty():
    assert face_boxes(np.zeros((1, 1, 0, 7), dtype=np.float32), 10, 10) == []


@pytest.mark.parametrize("box,shape,expected", [
    ((50, 50, 80, 90), (200, 200, 3), (30, 30, 100, 110)),
    ((5, 5, 95, 45), (50, 100, 3), (0, 0, 100, 50)),     # clipped to the image
])
def test_pad_box(box, shape, expected):
    assert pad_box(box, 20, shape) == expected


def test_top_label():
    scores = np.zeros(len(AGE_BUCKETS)); scores[4] = 0.8
    assert top_label(scores, AGE_BUCKETS) == ("(25-32)", pytest.approx(0.8))
    assert top_label(np.array([[0.1, 0.9]]), GENDERS)[0] == "Female"


def test_top_label_length_mismatch():
    with pytest.raises(ValueError):
        top_label(np.array([0.5, 0.5, 0.0]), GENDERS)


def test_format_label():
    assert format_label(Prediction((0, 0, 1, 1), "Male", 0.9, "(25-32)", 0.6)) == "Male, (25-32)"


def test_model_paths_reports_missing(tmp_path):
    from agegender.models import model_paths
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        model_paths(tmp_path)

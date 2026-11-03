"""Pure helpers (NumPy only) used by the detector; unit-tested without models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Label order of the Levi & Hassner (2015) Adience models.
AGE_BUCKETS = ("(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)")
GENDERS = ("Male", "Female")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(frozen=True)
class Prediction:
    box: Box
    gender: str
    gender_confidence: float
    age: str
    age_confidence: float


def face_boxes(detections: np.ndarray, width: int, height: int, conf_threshold: float = 0.7) -> list[Box]:
    """Convert an SSD output blob of shape (1, 1, N, 7) into pixel boxes clipped to the frame."""
    boxes: list[Box] = []
    for det in detections.reshape(-1, 7):
        if float(det[2]) <= conf_threshold:
            continue
        x1, y1, x2, y2 = (det[3:7] * np.array([width, height, width, height])).astype(int)
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        if x2 > x1 and y2 > y1:
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def pad_box(box: Box, padding: int, shape: Sequence[int]) -> Box:
    """Grow a box by ``padding`` pixels on each side, clipped to an image of ``shape``."""
    height, width = shape[0], shape[1]
    x1, y1, x2, y2 = box
    return (max(0, x1 - padding), max(0, y1 - padding),
            min(width, x2 + padding), min(height, y2 + padding))


def top_label(scores: np.ndarray, labels: Sequence[str]) -> tuple[str, float]:
    scores = np.asarray(scores).ravel()
    if scores.size != len(labels):
        raise ValueError(f"expected {len(labels)} scores, got {scores.size}")
    i = int(scores.argmax())
    return labels[i], float(scores[i])


def format_label(pred: Prediction) -> str:
    return f"{pred.gender}, {pred.age}"

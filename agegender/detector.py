"""OpenCV DNN face detection + age/gender classification."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .core import AGE_BUCKETS, GENDERS, MODEL_MEAN_VALUES, Prediction, face_boxes, format_label, pad_box, top_label
from .models import model_paths


class AgeGenderDetector:
    def __init__(self, model_dir: str | Path | None = None, conf_threshold: float = 0.7, padding: int = 20):
        p = model_paths(model_dir)
        self.face_net = cv2.dnn.readNet(str(p["face_model"]), str(p["face_proto"]))
        self.age_net = cv2.dnn.readNet(str(p["age_model"]), str(p["age_proto"]))
        self.gender_net = cv2.dnn.readNet(str(p["gender_model"]), str(p["gender_proto"]))
        self.conf_threshold = conf_threshold
        self.padding = padding

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), swapRB=True, crop=False)  # as in the original notebook
        self.face_net.setInput(blob)
        return face_boxes(self.face_net.forward(), w, h, self.conf_threshold)

    def predict(self, frame: np.ndarray) -> list[Prediction]:
        preds = []
        for box in self.detect_faces(frame):
            x1, y1, x2, y2 = pad_box(box, self.padding, frame.shape)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            self.gender_net.setInput(blob)
            gender, g_conf = top_label(self.gender_net.forward()[0], GENDERS)
            self.age_net.setInput(blob)
            age, a_conf = top_label(self.age_net.forward()[0], AGE_BUCKETS)
            preds.append(Prediction(box, gender, g_conf, age, a_conf))
        return preds

    @staticmethod
    def annotate(frame: np.ndarray, preds: list[Prediction]) -> np.ndarray:
        out = frame.copy()
        thickness = max(1, round(frame.shape[0] / 150))
        for p in preds:
            x1, y1, x2, y2 = p.box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness, cv2.LINE_8)
            cv2.putText(out, format_label(p), (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
        return out

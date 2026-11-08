"""Age and gender estimation from face images with OpenCV DNN (Caffe models)."""
from .core import AGE_BUCKETS, GENDERS, Prediction, face_boxes, format_label, pad_box, top_label

__all__ = ["AGE_BUCKETS", "GENDERS", "Prediction", "face_boxes", "format_label", "pad_box", "top_label"]

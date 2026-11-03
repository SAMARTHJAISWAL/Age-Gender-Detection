"""Locating (and optionally downloading) the pretrained model files."""
from __future__ import annotations

import os
from pathlib import Path

FILES = {
    "face_proto": "opencv_face_detector.pbtxt",
    "face_model": "opencv_face_detector_uint8.pb",
    "age_proto": "age_deploy.prototxt",
    "age_model": "age_net.caffemodel",
    "gender_proto": "gender_deploy.prototxt",
    "gender_model": "gender_net.caffemodel",
}
# Google Drive folder with all six files (the project's original download location).
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1Ubf32g3G-6cxCl6R5yZMDKlhVSlVXzTI"


def model_dir(override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("AGEGENDER_MODEL_DIR", "models")).expanduser()


def model_paths(directory: str | Path | None = None) -> dict[str, Path]:
    base = model_dir(directory)
    paths = {key: base / name for key, name in FILES.items()}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {base.resolve()}: {', '.join(missing)}. "
            "Run `python scripts/download_models.py` or set AGEGENDER_MODEL_DIR.")
    return paths

"""Download the six pretrained model files (~94 MB) into ./models (or AGEGENDER_MODEL_DIR).

Uses gdown to fetch the project's shared Google Drive folder:
    pip install gdown
    python scripts/download_models.py [target_dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agegender.models import DRIVE_FOLDER, FILES, model_dir  # noqa: E402


def main() -> int:
    try:
        import gdown
    except ImportError:
        print("gdown is required: pip install gdown", file=sys.stderr)
        return 1
    target = model_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    target.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(DRIVE_FOLDER, output=str(target), quiet=False)
    missing = [n for n in FILES.values() if not (target / n).exists()]
    if missing:
        print(f"Still missing: {missing}", file=sys.stderr)
        return 1
    print(f"All model files are in {target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

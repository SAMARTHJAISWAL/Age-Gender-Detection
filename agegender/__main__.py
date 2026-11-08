"""CLI: ``python -m agegender --image photo.jpg`` or ``python -m agegender --camera 0``."""
from __future__ import annotations

import argparse
import sys

import cv2

from .core import format_label
from .detector import AgeGenderDetector


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agegender", description="Estimate age bucket and gender of faces.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an image")
    src.add_argument("--camera", type=int, help="webcam index (ESC or q to quit)")
    ap.add_argument("--models", help="folder with the model files (default: $AGEGENDER_MODEL_DIR or ./models)")
    ap.add_argument("--conf", type=float, default=0.7, help="face detection confidence threshold")
    ap.add_argument("--output", help="save the annotated image here (image mode)")
    ap.add_argument("--no-display", action="store_true", help="do not open a window")
    args = ap.parse_args(argv)

    try:
        det = AgeGenderDetector(args.models, conf_threshold=args.conf)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            return 1
        preds = det.predict(frame)
        if not preds:
            print("No face detected")
        for i, p in enumerate(preds, 1):
            print(f"Face {i} at {p.box}: {p.gender} ({p.gender_confidence:.2f}), "
                  f"age {p.age} ({p.age_confidence:.2f})")
        out = det.annotate(frame, preds)
        if args.output:
            cv2.imwrite(args.output, out)
            print(f"Saved {args.output}")
        if not args.no_display:
            cv2.imshow("Age & Gender", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preds = det.predict(frame)
        cv2.imshow("Age & Gender", det.annotate(frame, preds))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

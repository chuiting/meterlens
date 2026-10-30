"""Run the detector on one image and save it with the counter box drawn."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import CONFIG, resolve_path


def detect_counter(image_path: str, out_path: str = None, conf: float = 0.25):
    weights_path = resolve_path(CONFIG["paths"]["yolo_weights"])
    if not weights_path.exists():
        raise FileNotFoundError(f"no trained weights at {weights_path} -- run src/train_yolo.py first")

    model = YOLO(str(weights_path))
    result = model.predict(source=image_path, conf=conf, verbose=False)[0]

    annotated = result.plot()  # box + confidence label

    image_path = Path(image_path)
    if out_path is None:
        out_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
    out_path = Path(out_path)

    cv2.imwrite(str(out_path), annotated)
    print(f"saved annotated image: {out_path}")

    boxes = result.boxes
    if len(boxes) == 0:
        print("no counter detected at this confidence threshold")
    else:
        for box in boxes:
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            confidence = float(box.conf[0])
            print(f"counter bbox (x1,y1,x2,y2): ({x1}, {y1}, {x2}, {y2})  confidence: {confidence:.3f}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trained YOLO model on one image and save the bbox overlay.")
    parser.add_argument("image_path", help="path to the image to test")
    parser.add_argument("--out", default=None, help="path to save the annotated image (default: alongside input, suffixed _detected)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default 0.25)")
    args = parser.parse_args()
    detect_counter(args.image_path, args.out, args.conf)

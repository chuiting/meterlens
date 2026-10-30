"""Run the detector once per split and cache the counter crops under data/recognition/."""

import json
import time

import cv2
import numpy as np

from config import CONFIG, resolve_path
from parse import load_all_annotations

DETECTION_CONF = 0.25


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def predicted_crops_dir(split: str):
    return resolve_path(CONFIG["paths"]["recognition_dir"]) / split / "counters_predicted"


def predicted_index_path(split: str):
    return resolve_path(CONFIG["paths"]["recognition_dir"]) / split / "predicted_index.json"


def build_predicted_crops(split: str, force: bool = False) -> dict:
    """Detect and cache crops for `split`; returns {basename: {detected, bbox, conf, reading}}."""
    index_path = predicted_index_path(split)
    if index_path.exists() and not force:
        with open(index_path) as f:
            return json.load(f)

    from ultralytics import YOLO  # lazy import

    weights_path = resolve_path(CONFIG["paths"]["yolo_weights"])
    if not weights_path.exists():
        raise FileNotFoundError(f"no trained detector at {weights_path} -- run src/train_yolo.py first")
    yolo = YOLO(str(weights_path))
    counter_size = tuple(CONFIG["image_sizes"]["counter_crop"])

    out_dir = predicted_crops_dir(split)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_annotations()
    split_df = df[df["split"] == split].reset_index(drop=True)
    _log(f"running detector over {len(split_df)} {split} images (cached to {out_dir})")

    index = {}
    for i, row in enumerate(split_df.itertuples()):
        if i % 100 == 0:
            _log(f"  {i}/{len(split_df)}")
        basename = row.filename.rsplit(".", 1)[0]
        img_path = resolve_path(row.image_path)
        result = yolo.predict(source=str(img_path), conf=DETECTION_CONF, verbose=False)[0]
        boxes = result.boxes
        entry = {"detected": False, "bbox": None, "conf": None, "reading": row.reading}
        if len(boxes) > 0:
            img = cv2.imread(str(img_path))
            best = int(np.argmax(boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
            crop = cv2.resize(img[y1:y2, x1:x2], counter_size)
            cv2.imwrite(str(out_dir / f"{basename}.jpg"), crop)
            entry.update({"detected": True, "bbox": [x1, y1, x2 - x1, y2 - y1],
                          "conf": float(boxes.conf[best])})
        index[basename] = entry
        del result

    with open(index_path, "w") as f:
        json.dump(index, f, indent=1)
    n_miss = sum(1 for e in index.values() if not e["detected"])
    _log(f"{split}: {len(index) - n_miss} detected, {n_miss} missed")
    return index


def load_predicted_crops(split: str, grayscale: bool = True):
    """Return (basenames, readings, crops); a crop is None where the detector missed."""
    index = build_predicted_crops(split)
    out_dir = predicted_crops_dir(split)
    basenames = sorted(index)
    readings, crops = [], []
    for b in basenames:
        readings.append(index[b]["reading"])
        if index[b]["detected"]:
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            crops.append(cv2.imread(str(out_dir / f"{b}.jpg"), flag))
        else:
            crops.append(None)
    return basenames, readings, crops


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the detector-crop cache for one or more splits.")
    parser.add_argument("splits", nargs="*", default=["validation", "testing"])
    parser.add_argument("--force", action="store_true", help="rerun the detector even if a cache exists")
    args = parser.parse_args()
    for s in args.splits:
        build_predicted_crops(s, force=args.force)

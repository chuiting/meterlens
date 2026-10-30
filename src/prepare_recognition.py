"""Phase 1: build the recognition dataset (counter, digit and margin crops) for the CNN and SVM."""

import json
from pathlib import Path

import cv2

from config import CONFIG, resolve_path
from parse import load_all_annotations

# Same split names as the dataset itself.
SPLITS = ["training", "validation", "testing"]


def crop(img, bbox):
    x, y, w, h = bbox
    x, y = max(0, x), max(0, y)
    return img[y:y + h, x:x + w]


def crop_with_margin(img, bbox, margin_frac):
    """Crop bbox expanded by margin_frac on every side; returns (crop, (x0, y0) offset)."""
    x, y, w, h = bbox
    mx, my = int(round(w * margin_frac)), int(round(h * margin_frac))
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
    return img[y0:y1, x0:x1], (x0, y0)


def shift_bbox(bbox, offset):
    x, y, w, h = bbox
    return [int(x - offset[0]), int(y - offset[1]), int(w), int(h)]


def prepare_recognition():
    recog_dir = resolve_path(CONFIG["paths"]["recognition_dir"])
    counter_w, counter_h = CONFIG["image_sizes"]["counter_crop"]
    digit_w, digit_h = CONFIG["image_sizes"]["digit_crop"]
    margin_frac = CONFIG["augmentation"]["crop_margin_frac"]

    df = load_all_annotations()

    for split in SPLITS:
        counters_dir = recog_dir / split / "counters"
        digits_dir = recog_dir / split / "digits"
        margin_dir = recog_dir / split / "counters_margin"
        counters_dir.mkdir(parents=True, exist_ok=True)
        digits_dir.mkdir(parents=True, exist_ok=True)
        margin_dir.mkdir(parents=True, exist_ok=True)

        labels = {}
        geometry = {}
        split_df = df[df["split"] == split]
        for _, row in split_df.iterrows():
            basename = Path(row["filename"]).stem
            img_path = resolve_path(row["image_path"])
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"cv2 failed to read {img_path}")

            counter_crop = crop(img, row["counter_bbox"])
            counter_crop = cv2.resize(counter_crop, (counter_w, counter_h))
            cv2.imwrite(str(counters_dir / f"{basename}.jpg"), counter_crop)

            reading = row["reading"]
            for i, digit_bbox in enumerate(row["digit_bboxes"]):
                digit_crop = crop(img, digit_bbox)
                digit_crop = cv2.resize(digit_crop, (digit_w, digit_h))
                digit_label = reading[i]
                cv2.imwrite(
                    str(digits_dir / f"{basename}_d{i+1}_{digit_label}.jpg"),
                    digit_crop,
                )

            labels[basename] = list(reading)

            margin_crop, offset = crop_with_margin(img, row["counter_bbox"], margin_frac)
            cv2.imwrite(str(margin_dir / f"{basename}.jpg"), margin_crop)
            geometry[basename] = {
                "counter_bbox": shift_bbox(row["counter_bbox"], offset),
                "digit_bboxes": [shift_bbox(b, offset) for b in row["digit_bboxes"]],
            }

        labels_path = recog_dir / split / "labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        with open(recog_dir / split / "geometry.json", "w") as f:
            json.dump(geometry, f)

        print(f"{split}: wrote {len(split_df)} counter crops + {len(split_df) * 5} digit crops to {recog_dir / split}")


if __name__ == "__main__":
    prepare_recognition()

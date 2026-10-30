"""Phase 1: parse UFPR-AMR annotations. Bboxes are (x, y, w, h) in pixels."""

from pathlib import Path
import re

import pandas as pd
from PIL import Image

from config import CONFIG, PROJECT_ROOT, resolve_path

DIGIT_LINE_RE = re.compile(r"digit (\d): (\d+) (\d+) (\d+) (\d+)")


def parse_annotation(path) -> dict:
    """Parse one annotation file -> {camera, reading (5-char str), counter_bbox, digit_bboxes}."""
    path = Path(path)
    with open(path, "r") as f:
        lines = [line.strip() for line in f.read().splitlines() if line.strip()]

    camera = None
    reading = None
    counter_bbox = None
    digit_bboxes = [None] * 5

    for line in lines:
        # 8 files have a stray character before "reading:" (e.g. "Creading:"), so match by substring.
        if line.startswith("camera:"):
            camera = line.split("camera:", 1)[1].strip()
        elif "reading:" in line:
            reading = line.split("reading:", 1)[1].strip().zfill(5)
        elif line.startswith("position:"):
            x, y, w, h = (int(v) for v in line.split("position:", 1)[1].split())
            counter_bbox = (x, y, w, h)
        else:
            m = DIGIT_LINE_RE.match(line)
            if m:
                idx = int(m.group(1)) - 1
                x, y, w, h = (int(v) for v in m.groups()[1:])
                digit_bboxes[idx] = (x, y, w, h)

    if camera is None or reading is None or counter_bbox is None or any(b is None for b in digit_bboxes):
        raise ValueError(f"Malformed or incomplete annotation: {path}")
    if len(reading) != 5:
        raise ValueError(f"Reading is not 5 digits after zero-pad in {path}: {reading!r}")

    return {
        "camera": camera,
        "reading": reading,
        "counter_bbox": counter_bbox,
        "digit_bboxes": digit_bboxes,
    }


def xywh_to_yolo(bbox, img_w: int, img_h: int):
    """(x, y, w, h) absolute pixels -> (cx, cy, w, h) normalised to [0, 1]."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return (cx, cy, w / img_w, h / img_h)


def yolo_to_xywh(bbox, img_w: int, img_h: int):
    """(cx, cy, w, h) normalised -> (x, y, w, h) absolute pixels."""
    cx, cy, w, h = bbox
    abs_w = w * img_w
    abs_h = h * img_h
    x = cx * img_w - abs_w / 2
    y = cy * img_h - abs_h / 2
    return (int(round(x)), int(round(y)), int(round(abs_w)), int(round(abs_h)))


def load_all_annotations() -> pd.DataFrame:
    """Parse every annotation in data/raw/ into one DataFrame row per image.

    Paths are relative to the project root; open them with `resolve_path()`.
    """
    split_dirs = {
        "training": resolve_path(CONFIG["paths"]["raw_train"]),
        "validation": resolve_path(CONFIG["paths"]["raw_val"]),
        "testing": resolve_path(CONFIG["paths"]["raw_test"]),
    }

    rows = []
    for split, split_dir in split_dirs.items():
        for txt_path in sorted(split_dir.glob("*.txt")):
            img_path = txt_path.with_suffix(".jpg")
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image for annotation: {txt_path}")

            ann = parse_annotation(txt_path)
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            rows.append({
                "filename": img_path.name,
                "split": split,
                "camera": ann["camera"],
                "reading": ann["reading"],
                "counter_bbox": ann["counter_bbox"],
                "digit_bboxes": ann["digit_bboxes"],
                "img_w": img_w,
                "img_h": img_h,
                "image_path": img_path.relative_to(PROJECT_ROOT).as_posix(),
                "annotation_path": txt_path.relative_to(PROJECT_ROOT).as_posix(),
            })

    return pd.DataFrame(rows)


def write_dataset_details(df: pd.DataFrame) -> Path:
    """Write one row per image to results/data/dataset_details.csv.

    Read it back with dtype={"reading": str} to keep leading zeros; bbox columns
    are Python-literal strings (use ast.literal_eval).
    """
    out_path = resolve_path(CONFIG["paths"]["dataset_details"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    df = load_all_annotations()
    print(df["split"].value_counts())
    print(df.head())
    write_dataset_details(df)

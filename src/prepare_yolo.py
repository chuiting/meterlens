"""Phase 1: convert the parsed dataset into YOLO-format detection labels."""

import shutil
from pathlib import Path

from config import CONFIG, resolve_path
from parse import load_all_annotations, xywh_to_yolo

# Same split names as the dataset itself.
SPLITS = ["training", "validation", "testing"]


def prepare_yolo():
    yolo_dir = resolve_path(CONFIG["paths"]["yolo_dir"])

    df = load_all_annotations()

    for split in SPLITS:
        images_dir = yolo_dir / split / "images"
        labels_dir = yolo_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        split_df = df[df["split"] == split]
        for _, row in split_df.iterrows():
            basename = Path(row["filename"]).stem
            dst_img = images_dir / row["filename"]
            shutil.copy(resolve_path(row["image_path"]), dst_img)

            cx, cy, w, h = xywh_to_yolo(row["counter_bbox"], row["img_w"], row["img_h"])
            label_path = labels_dir / f"{basename}.txt"
            with open(label_path, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"{split}: wrote {len(split_df)} images + labels to {images_dir.parent}")

    data_yaml_path = resolve_path(CONFIG["paths"]["yolo_data_yaml"])
    class_names = CONFIG["yolo"]["class_names"]
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {yolo_dir.resolve()}\n")
        f.write("train: training/images\n")
        f.write("val: validation/images\n")
        f.write("test: testing/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    print(f"wrote {data_yaml_path}")


if __name__ == "__main__":
    prepare_yolo()

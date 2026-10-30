"""Phase 2: fine-tune COCO-pretrained YOLOv8n to detect the counter (transfer learning)."""

import argparse
import ast
import json
import random
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from config import CONFIG, resolve_path

# Harmless warning on Apple GPUs (MPS) about non-deterministic ops.
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


# Ultralytics test-split plots to keep
TEST_PLOTS = ("BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png",
              "confusion_matrix.png", "confusion_matrix_normalized.png")


def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def evaluate_per_image(refresh_detections: bool = False, iou_thresh: float = 0.5):
    """Per-image IoU on the test split -> results/yolo/yolo_test_predictions.csv.

    Uses the pipeline's cached detections (predicted_crops.py); set
    refresh_detections=True after retraining.
    """
    from predicted_crops import build_predicted_crops

    index = build_predicted_crops("testing", force=refresh_detections)

    details_path = resolve_path(CONFIG["paths"]["dataset_details"])
    df = pd.read_csv(details_path, dtype={"reading": str})
    df["counter_bbox"] = df["counter_bbox"].apply(ast.literal_eval)
    test_df = df[df["split"] == "testing"].reset_index(drop=True)

    rows = []
    for _, row in test_df.iterrows():
        entry = index[row["filename"].rsplit(".", 1)[0]]
        if not entry["detected"]:
            rows.append({
                "filename": row["filename"],
                "image_path": row["image_path"],
                "detected": False,
                "confidence": None,
                "iou": 0.0,
                "status": "MISS",
            })
            continue

        best_iou = iou(row["counter_bbox"], entry["bbox"])
        rows.append({
            "filename": row["filename"],
            "image_path": row["image_path"],
            "detected": True,
            "confidence": entry["conf"],
            "iou": best_iou,
            "status": "OK" if best_iou >= iou_thresh else "LOW_IOU",
        })

    preds_df = pd.DataFrame(rows).sort_values("iou")
    out_path = resolve_path(CONFIG["paths"]["yolo_test_predictions"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_path, index=False)

    n_miss = (preds_df["status"] == "MISS").sum()
    n_low = (preds_df["status"] == "LOW_IOU").sum()
    n_ok = (preds_df["status"] == "OK").sum()
    print(f"wrote {out_path}")
    print(f"test images: {len(preds_df)} total -- {n_ok} OK, {n_low} LOW_IOU (<{iou_thresh}), {n_miss} MISS (no detection)")
    return preds_df


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _on_epoch_end(trainer):
    try:
        loss = float(trainer.loss.item()) if trainer.loss is not None else None
    except (AttributeError, TypeError, ValueError):
        loss = None
    _log(f"epoch {trainer.epoch + 1}/{trainer.epochs} done (loss={loss})")


def train_yolo(resume: bool = False):
    cfg = CONFIG["yolo"]
    seed = CONFIG["seed"]
    set_seeds(seed)


    data_yaml = resolve_path(CONFIG["paths"]["yolo_data_yaml"])
    runs_dir = resolve_path(CONFIG["paths"]["yolo_runs_dir"])
    device = pick_device()
    _log(f"training on device: {device}")

    if resume:
        last_pt = runs_dir / "train" / "weights" / "last.pt"
        if not last_pt.exists():
            raise FileNotFoundError(
                f"--resume given but no checkpoint found at {last_pt} -- run without --resume first"
            )
        _log(f"resuming from checkpoint {last_pt} (crashed/interrupted run picks up where it left off)")
        model = YOLO(str(last_pt))
        model.add_callback("on_train_epoch_end", _on_epoch_end)
        model.train(resume=True)
    else:
        _log(f"loading base model {cfg['base_model']} ...")
        model = YOLO(cfg["base_model"])
        model.add_callback("on_train_epoch_end", _on_epoch_end)
        _log("base model loaded, scanning dataset and starting training "
             "(first epoch is the slowest -- label/image cache is being built)")

        model.train(
            data=str(data_yaml),
            imgsz=cfg["imgsz"],
            epochs=cfg["epochs"],
            batch=cfg["batch"],
            patience=cfg["patience"],
            workers=cfg.get("workers", 8),
            cache=cfg.get("cache", False),
            seed=seed,
            device=device,
            project=str(runs_dir),
            name="train",
            exist_ok=True,
            verbose=True,
        )
    _log("training finished")

    best_weights = runs_dir / "train" / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"expected best weights at {best_weights}, training may have failed")

    yolo_weights_path = resolve_path(CONFIG["paths"]["yolo_weights"])
    yolo_weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, yolo_weights_path)
    _log(f"copied best weights to {yolo_weights_path}")

    _log("running final evaluation on test split ...")
    best_model = YOLO(yolo_weights_path)
    val_results = best_model.val(
        data=str(data_yaml), split="test", device=device,
        project=str(runs_dir), name="val", exist_ok=True,
    )

    curves_dir = resolve_path(CONFIG["paths"]["yolo_metrics"]).parent
    curves_dir.mkdir(parents=True, exist_ok=True)
    for name in TEST_PLOTS:
        src = Path(val_results.save_dir) / name
        if src.exists():
            shutil.copy(src, curves_dir / name)
    shutil.copy(runs_dir / "train" / "results.csv", resolve_path(CONFIG["paths"]["yolo_training_history"]))
    shutil.rmtree(val_results.save_dir, ignore_errors=True)
    _log(f"copied test-split plots and training history to {curves_dir}")

    metrics = {
        "mAP50": float(val_results.box.map50),
        "mAP50-95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
    }

    metrics_path = resolve_path(CONFIG["paths"]["yolo_metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {metrics_path}")
    print(metrics)

    _log("running per-image test evaluation (misses / low-IoU detections) ...")
    evaluate_per_image(refresh_detections=True)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the run from data/yolo/runs/train/weights/last.pt "
             "after a crash/interruption, instead of restarting from epoch 0",
    )
    parser.add_argument(
        "--per-image-only", action="store_true",
        help="skip training entirely and just (re)run per-image test evaluation against "
             "the cached pipeline detections (predicted_crops.py), writing results/yolo/yolo_test_predictions.csv",
    )
    args = parser.parse_args()
    if args.per_image_only:
        evaluate_per_image()
    else:
        train_yolo(resume=args.resume)

"""Inspector portal API, mounted under /api/inspector."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, UploadFile

from api import main as core
from config import CONFIG, resolve_path
from trace import trace

router = APIRouter(prefix="/api/inspector")


@lru_cache
def sensing_histogram(bins: int = 20) -> dict:
    """Validation-split sensing distances, binned, for the Capture tab's gate chart."""
    df = pd.read_csv(resolve_path(CONFIG["paths"]["sensing_scores"]))
    d = df[df["split"] == "validation"]["distance"].to_numpy()
    counts, edges = np.histogram(d, bins=bins, range=(0, float(d.max())))
    return {"counts": counts.tolist(), "min": 0.0, "max": float(edges[-1])}


def with_meta(out: dict, image_id: str) -> dict:
    if out["detected"] and out["sensing"]["distance"] is not None:
        out["sensing"]["histogram"] = sensing_histogram()
    out.update({"image_id": image_id, "image_url": f"/uploads/{image_id}.jpg"})
    return out


@router.post("/trace")
async def trace_upload(file: UploadFile = File(...)):
    image_id, bgr = await core.save_upload(file)
    return with_meta(trace(core.reader, bgr), image_id)


@router.get("/trace/dataset/{split}/{basename}")
def trace_dataset(split: str, basename: str):
    path = core.PROJECT_ROOT / "data" / "raw" / split / f"{basename}.jpg"
    if split not in ("training", "validation", "testing") or not path.exists():
        raise core.HTTPException(status_code=404, detail="unknown dataset image")
    out = trace(core.reader, core.cv2.imread(str(path)))
    if out["detected"] and out["sensing"]["distance"] is not None:
        out["sensing"]["histogram"] = sensing_histogram()
    out.update({"image_id": basename, "image_url": f"/data/raw/{split}/{basename}.jpg"})
    return out


@router.get("/trace/{image_id}")
def trace_stored(image_id: str):
    return with_meta(trace(core.reader, core.load_upload(image_id)), image_id)


def load_json(key):
    with open(resolve_path(CONFIG["paths"][key])) as f:
        return json.load(f)


@router.get("/models")
def models():
    ev = load_json("evaluation_summary")
    fusion = load_json("fusion_metrics")
    sensing = load_json("sensing_metrics")
    weights = CONFIG["fusion"]["weights"]
    gt, e2e_val, e2e_test = ev["recognition_ground_truth_crops"], ev["end_to_end_validation"], ev["end_to_end_test"]

    def row(name, val, test, role):
        return {"model": name, "val_full": val["full_sequence_accuracy"], "test_full": test["full_sequence_accuracy"],
                "test_digits": float(np.mean(test["per_digit_accuracy"])), "test_d5": test["per_digit_accuracy"][4],
                "test_per_digit": test["per_digit_accuracy"], "role": role}

    rows = [row(n, gt["validation"][n], gt["testing"][n], f"member · w {weights[n]:.3f}" if n in weights else "built, not deployed")
            for n in ("cnn", "cnn_rgb", "svm", "rf")]
    fused_name = "fused (" + " + ".join(gt["testing"]["fused"]["members"]) + ")"
    rows.append(row(fused_name, gt["validation"]["fused"], gt["testing"]["fused"], "deployed"))
    e2e = {"model": "YOLO + " + fused_name,
           "val_full": e2e_val["fused"]["full_sequence_accuracy"], "test_full": e2e_test["fused"]["full_sequence_accuracy"],
           "test_digits": float(np.mean(e2e_test["fused"]["per_digit_accuracy_on_detected"])),
           "test_d5": e2e_test["fused"]["per_digit_accuracy_on_detected"][4],
           "test_per_digit": e2e_test["fused"]["per_digit_accuracy_on_detected"],
           "role": f"{e2e_test['detector_misses']} detector misses / {e2e_test['n_images']}"}
    rows.append(e2e)

    baseline = ev["published_baseline"]
    return {
        "generated": ev["generated"],
        "headline": {"digits": e2e["test_digits"], "full": e2e["test_full"], "full_gt": gt["testing"]["fused"]["full_sequence_accuracy"],
                     "approved_accuracy": e2e_test["decision"]["approved_accuracy"], "flag_rate": e2e_test["decision"]["flag_rate"],
                     "errors_caught": e2e_test["decision"]["errors_caught"], "errors_approved": e2e_test["decision"]["errors_approved"],
                     "baseline_full": baseline["rows"][-3]["full_reading_accuracy"] / 100},
        "rows": rows,
        "overlap": fusion["selected"]["validation_error_overlap"],
        "ablation": load_json("cnn_ablation"),
        "baseline": {"rows": baseline["rows"][-3:], "source": baseline["source"], "caption": baseline["caption"]},
        "sensing": {"components": sensing["pca_n_components"], "clusters": len(sensing["kmeans_cluster_sizes"]),
                    **sensing["flag_rates"]["testing"]["ground_truth_crops"],
                    "flagged_error_rate": sensing["recognition_error_by_flag"]["testing"]["flagged_error_rate"],
                    "passed_error_rate": sensing["recognition_error_by_flag"]["testing"]["passed_error_rate"]},
        "detection": ev["detection"],
        "confusion_accuracy": ev["confusion"]["digit_accuracy"],
        "confusion_top": ev["confusion"]["top_confusions"],
        "dataset": pd.read_csv(resolve_path(CONFIG["paths"]["dataset_details"]), usecols=["split"])["split"].value_counts().to_dict(),
        "figures": {"confusion": "/results/evaluation/confusion_matrix.png",
                    "sensing_histogram": "/results/sensing/sensing_distance_histogram.png",
                    "decision_flow": "/results/decision/decision_layer_effect.png",
                    "iou": "/results/yolo/iou_histogram.png",
                    "recognition": "/results/evaluation/digit_recognition.png",
                    "ablation": "/results/cnn/cnn_ablation.png"},
    }


@lru_cache
def dataset_rows(split: str) -> list:
    key = "end_to_end_predictions" if split == "testing" else "end_to_end_validation_predictions"
    df = pd.read_csv(resolve_path(CONFIG["paths"][key]), dtype=str, keep_default_na=False)
    sens = pd.read_csv(resolve_path(CONFIG["paths"]["sensing_scores"]))
    sens = sens[(sens["split"] == split) & (sens["crop_source"] == "detector")].set_index("basename")
    details = pd.read_csv(resolve_path(CONFIG["paths"]["dataset_details"]), dtype={"reading": str}).set_index("filename")
    rows = []
    for r in df.itertuples():
        members = {m: getattr(r, f"{m}_reading") for m in ("cnn", "cnn_rgb", "svm")}
        detected = r.detected == "True"
        rows.append({
            "basename": r.basename, "truth": r.truth, "detected": detected,
            "camera": details.loc[f"{r.basename}.jpg", "camera"],
            "members": members, "fused": r.fused_reading,
            "confidence": float(r.confidence) if r.confidence else None,
            "decision": r.decision, "correct": r.correct == "True",
            "iou": float(r.iou) if r.iou else 0.0,
            "sensing": float(sens.loc[r.basename, "distance"]) if r.basename in sens.index else None,
            "sensing_flag": bool(sens.loc[r.basename, "flag"]) if r.basename in sens.index else False,
            "crop_url": f"/data/recognition/{split}/counters_predicted/{r.basename}.jpg" if detected else None,
        })
    return rows


def dataset_available(split: str) -> dict:
    """Whether the crops and raw photos for a split are on disk."""
    return {"crops": any((core.PROJECT_ROOT / "data" / "recognition" / split / "counters_predicted").glob("*.jpg")),
            "photos": any((core.PROJECT_ROOT / "data" / "raw" / split).glob("*.jpg"))}


@router.get("/datasets/{split}")
def datasets(split: str):
    if split not in ("validation", "testing"):
        raise core.HTTPException(status_code=404, detail="end-to-end predictions exist for validation and testing only")
    try:
        rows = dataset_rows(split)
    except FileNotFoundError as e:
        raise core.HTTPException(status_code=503, detail=f"evaluation results missing: {e.filename} -- run src/evaluate.py")
    return {**dataset_available(split), "rows": rows}

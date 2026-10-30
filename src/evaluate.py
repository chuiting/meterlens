"""Phase 9: evaluation on the test split. Results go to results/evaluation/.

    python src/evaluate.py [--skip-detector-val]
"""

import argparse
import ast
import json
import shutil
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG, resolve_path
from figures import save_report_figures
from fusion import load_probs, read_probs
from pipeline import MeterReader
from predicted_crops import build_predicted_crops, predicted_crops_dir

NUM_DIGITS = 5

# Laroca et al. (2019), arXiv 1902.09600, Table 7
PUBLISHED_BASELINE = {
    "source": "Laroca et al. 2019, arXiv 1902.09600, Table 7 (UFPR-AMR test split, detector-cropped counters, mean of 10 runs)",
    "rows": [
        {"approach": "Multi-task CNN, original training set", "digit_accuracy": 24.64, "full_reading_accuracy": 0.00},
        {"approach": "Multi-task CNN, with synthesised data", "digit_accuracy": 95.96, "full_reading_accuracy": 87.69},
        {"approach": "CRNN, with synthesised data", "digit_accuracy": 97.87, "full_reading_accuracy": 92.30},
        {"approach": "CR-NET (per-digit detector), with synthesised data", "digit_accuracy": 98.30, "full_reading_accuracy": 94.13},
    ],
}


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def results_path(name):
    out_dir = resolve_path(CONFIG["paths"]["evaluation_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def pipeline_ious(split: str) -> dict:
    """Per-image IoU of the pipeline's box vs the annotation (a miss counts as 0)."""
    details = pd.read_csv(resolve_path(CONFIG["paths"]["dataset_details"]), dtype={"reading": str})
    details = details[details["split"] == split].copy()
    details["basename"] = details["filename"].str.replace(".jpg", "", regex=False)
    gt = {r.basename: ast.literal_eval(r.counter_bbox) for r in details.itertuples()}
    index = build_predicted_crops(split)
    return {b: (_iou(gt[b], e["bbox"]) if e["detected"] else 0.0) for b, e in index.items()}


def evaluate_detection(skip_val: bool):
    if skip_val:
        with open(resolve_path(CONFIG["paths"]["yolo_metrics"])) as f:
            metrics = json.load(f)
        _log("detector mAP reused from results/yolo/yolo_metrics.json (--skip-detector-val)")
    else:
        from ultralytics import YOLO
        _log("running detector validation on the test split ...")
        model = YOLO(str(resolve_path(CONFIG["paths"]["yolo_weights"])))
        r = model.val(data=str(resolve_path(CONFIG["paths"]["yolo_data_yaml"])), split="test",
                      batch=CONFIG["yolo"]["batch"], workers=CONFIG["yolo"]["workers"], verbose=False, plots=False,
                      project=str(resolve_path(CONFIG["paths"]["yolo_runs_dir"])), name="val", exist_ok=True)
        metrics = {"mAP50": float(r.box.map50), "mAP50-95": float(r.box.map),
                   "precision": float(r.box.mp), "recall": float(r.box.mr)}
        shutil.rmtree(r.save_dir, ignore_errors=True)  # plots=False, so nothing worth keeping

    ious = pipeline_ious("testing")
    index = build_predicted_crops("testing")
    iou_arr = np.array(list(ious.values()))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(iou_arr, bins=np.linspace(0, 1, 21), color="#3776ab")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="IoU 0.5")
    ax.set_xlabel("IoU of the highest-confidence box vs the annotated counter box")
    ax.set_ylabel("test images"); ax.set_title(f"Detector IoU on test (mean {iou_arr.mean():.3f})"); ax.legend()
    fig.tight_layout(); fig.savefig(results_path("iou_histogram.png"), dpi=130); plt.close(fig)

    metrics.update({
        "mean_iou": float(iou_arr.mean()),
        "median_iou": float(np.median(iou_arr)),
        "iou_below_0.5": int((iou_arr < 0.5).sum()),
        "misses": int(sum(not e["detected"] for e in index.values())),
        "n_images": len(index),
        "note": "IoU is for the highest-confidence box, the one the pipeline uses; misses count as IoU 0",
    })
    _log(f"detection: mAP50 {metrics['mAP50']:.4f}, mAP50-95 {metrics['mAP50-95']:.4f}, mean IoU {metrics['mean_iou']:.3f}, misses {metrics['misses']}")
    return metrics, ious


def evaluate_recognition():
    members = CONFIG["fusion"]["members"]
    table = {}
    for split, key in (("validation", "val_probs"), ("testing", "test_probs")):
        rows = {}
        for m in members:
            _, truth, probs = load_probs(m[key])
            ok = probs.argmax(-1) == truth
            rows[m["name"]] = {"full_sequence_accuracy": float(ok.all(1).mean()),
                               "per_digit_accuracy": [float(x) for x in ok.mean(0)]}
        _, truth, fused = load_probs(CONFIG["paths"][f"fusion_{'val' if split == 'validation' else 'test'}_probs"])
        ok = fused.argmax(-1) == truth
        rows["fused"] = {"full_sequence_accuracy": float(ok.all(1).mean()),
                         "per_digit_accuracy": [float(x) for x in ok.mean(0)],
                         "members": CONFIG["fusion"]["selected_members"]}
        table[split] = rows
    for name, r in table["testing"].items():
        _log(f"recognition (GT crops, test): {name:8s} full {r['full_sequence_accuracy']:.4f}  per-digit {[round(x, 3) for x in r['per_digit_accuracy']]}")
    return table


def confusion_matrix():
    _, truth, fused = load_probs(CONFIG["paths"]["fusion_test_probs"])
    pred = fused.argmax(-1)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(truth.ravel(), pred.ravel()):
        cm[t, p] += 1
    off = [(int(t), int(p), int(cm[t, p])) for t in range(10) for p in range(10) if t != p and cm[t, p] > 0]
    off.sort(key=lambda x: -x[2])

    fig, ax = plt.subplots(figsize=(7, 6))
    norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for t in range(10):
        for p in range(10):
            ax.text(p, t, str(cm[t, p]), ha="center", va="center", fontsize=8,
                    color="white" if norm[t, p] > 0.5 else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit"); ax.set_ylabel("true digit")
    ax.set_title("Fused reading, test split, ground-truth crops (4,000 digits)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout(); fig.savefig(results_path("confusion_matrix.png"), dpi=130); plt.close(fig)
    with open(results_path("confusion_matrix.json"), "w") as f:
        json.dump({"matrix": cm.tolist(), "top_confusions": off[:10], "labels": list(range(10))}, f, indent=2)
    _log(f"confusion: top confusions (true, pred, count) {off[:6]}")
    return {"top_confusions": off[:10], "digit_accuracy": float(np.trace(cm) / cm.sum())}


def decision_plot():
    threshold = CONFIG["decision"]["confidence_threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    stats = {}
    for ax, (split, key) in zip(axes, (("validation", "fusion_val_probs"), ("testing", "fusion_test_probs"))):
        _, truth, fused = load_probs(CONFIG["paths"][key])
        _, _, conf = read_probs(fused)
        ok = (fused.argmax(-1) == truth).all(1)
        bins = np.linspace(0, 1, 21)
        ax.hist(conf[ok], bins=bins, alpha=0.7, color="#2ca02c", label=f"correct (n={ok.sum()})")
        ax.hist(conf[~ok], bins=bins, alpha=0.7, color="#d62728", label=f"wrong (n={(~ok).sum()})")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold {threshold:.2f}")
        ax.set_title(f"{split}: fused reading confidence"); ax.set_xlabel("min per-digit confidence"); ax.legend()
        flagged = conf < threshold
        stats[split] = {"flag_rate": float(flagged.mean()),
                        "errors_caught": float((flagged & ~ok).sum() / max(1, (~ok).sum())),
                        "approved_accuracy": float(ok[~flagged].mean()),
                        "errors_approved": int((~flagged & ~ok).sum())}
    fig.tight_layout(); fig.savefig(results_path("confidence_vs_correctness.png"), dpi=130); plt.close(fig)
    return {"threshold": threshold, **stats}


def evaluate_end_to_end(split: str, ious: dict, reader: MeterReader):
    """Run every image of `split` end to end, write its CSV, return (summary, DataFrame)."""
    index = build_predicted_crops(split)
    basenames = sorted(index)
    truth = [index[b]["reading"] for b in basenames]
    details = pd.read_csv(resolve_path(CONFIG["paths"]["dataset_details"]), dtype={"reading": str})
    details = details[details["split"] == split]
    img_paths = {r.filename.replace(".jpg", ""): r.image_path for r in details.itertuples()}

    _log(f"end to end ({split}): preparing member inputs for {len(basenames)} photos from the cached detector boxes ...")
    detected = [index[b]["detected"] for b in basenames]
    inputs = {m["name"]: [] for m in reader.members}
    for i, b in enumerate(basenames):
        if i % 200 == 0:
            _log(f"  {i}/{len(basenames)}")
        if not detected[i]:
            continue
        img = cv2.imread(str(resolve_path(img_paths[b])))
        bbox = tuple(index[b]["bbox"])
        for m in reader.members:
            inputs[m["name"]].append(reader.member_input(img, bbox, m))

    probs = {}
    for m in reader.members:
        _log(f"  running {m['name']} on {len(inputs[m['name']])} detected counters ...")
        probs[m["name"]] = reader.member_probs(m, inputs[m["name"]])
    fused = reader.fuse(probs)
    decisions = reader.decide(fused)
    member_readings = {n: read_probs(p)[0] for n, p in probs.items()}

    rows = []
    j = 0
    for i, b in enumerate(basenames):
        row = {"basename": b, "truth": truth[i], "detected": detected[i], "iou": round(ious[b], 4)}
        if detected[i]:
            d = decisions[j]
            row.update({f"{n}_reading": member_readings[n][j] for n in member_readings})
            row.update({"fused_reading": d["reading"], "confidence": round(d["confidence"], 4),
                        "decision": d["decision"], "correct": d["reading"] == truth[i]})
            j += 1
        else:
            row.update({"fused_reading": None, "confidence": 0.0, "decision": "FLAGGED", "correct": False})
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_key = "end_to_end_predictions" if split == "testing" else "end_to_end_validation_predictions"
    csv_path = resolve_path(CONFIG["paths"][csv_key])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    n = len(df)
    det = df[df["detected"]]
    per_member = {}
    for name in member_readings:
        ok = np.array([r == t for r, t in zip(det[f"{name}_reading"], det["truth"])])
        per_member[name] = {"full_sequence_accuracy": float(ok.sum() / n),
                            "per_digit_accuracy": [float(np.mean([r[k] == t[k] for r, t in zip(det[f"{name}_reading"], det["truth"])]))
                                                   for k in range(NUM_DIGITS)]}
    fused_ok = df["correct"].values
    per_digit = [float(np.mean([r[k] == t[k] for r, t in zip(det["fused_reading"], det["truth"])])) for k in range(NUM_DIGITS)]
    flagged = df["decision"].values == "FLAGGED"
    e2e = {
        "n_images": n,
        "detector_misses": int((~df["detected"]).sum()),
        "members": per_member,
        "fused": {"full_sequence_accuracy": float(fused_ok.mean()), "per_digit_accuracy_on_detected": per_digit},
        "decision": {
            "threshold": reader.threshold,
            "flag_rate": float(flagged.mean()),
            "errors_caught": float((flagged & ~fused_ok).sum() / max(1, (~fused_ok).sum())),
            "approved_count": int((~flagged).sum()),
            "approved_accuracy": float(fused_ok[~flagged].mean()),
            "errors_approved": int((~flagged & ~fused_ok).sum()),
        },
        "note": "misses count as wrong, flagged readings; per-member and per-digit rates are over detected images except full_sequence_accuracy, which is over all images",
    }
    _log(f"end to end ({split}, {n} photos): fused full-sequence {e2e['fused']['full_sequence_accuracy']:.4f}; "
         + ", ".join(f"{k} {v['full_sequence_accuracy']:.4f}" for k, v in per_member.items()))
    d = e2e["decision"]
    _log(f"  decision @ {d['threshold']:.2f}: flag rate {d['flag_rate']:.1%}, errors caught {d['errors_caught']:.1%}, "
         f"approved {d['approved_count']} at {d['approved_accuracy']:.3f}")
    return e2e, df


def failure_gallery(df):
    k = CONFIG["evaluation"]["failure_gallery_count"]
    wrong = df[df["detected"] & ~df["correct"]].sort_values("confidence", ascending=False).head(k)
    crops_dir = predicted_crops_dir("testing")
    tiles = []
    for r in wrong.itertuples():
        img = cv2.imread(str(crops_dir / f"{r.basename}.jpg"))
        if img is None:
            continue
        cv2.putText(img, f"true {r.truth}  read {r.fused_reading}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(img, f"conf {r.confidence:.2f} {r.decision}", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(img)
    if tiles:
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(results_path("failure_gallery.jpg")), grid)
    approved_wrong = int(((df["decision"] == "APPROVED") & ~df["correct"]).sum())
    _log(f"failure gallery: {len(tiles)} most confident wrong readings written; {approved_wrong} wrong readings were approved")
    return {"gallery_count": len(tiles), "wrong_readings": int((~df["correct"]).sum()),
            "wrong_and_approved": approved_wrong,
            "listed": [{"basename": r.basename, "truth": r.truth, "read": r.fused_reading, "confidence": float(r.confidence),
                        "decision": r.decision} for r in wrong.itertuples()]}


def run_evaluation(skip_detector_val: bool = False):
    detection, ious = evaluate_detection(skip_detector_val)
    recognition = evaluate_recognition()
    confusion = confusion_matrix()
    decision = decision_plot()
    reader = MeterReader()
    end_to_end_val, _ = evaluate_end_to_end("validation", pipeline_ious("validation"), reader)
    end_to_end, df = evaluate_end_to_end("testing", ious, reader)
    failures = failure_gallery(df)
    report_figures = save_report_figures(recognition["testing"], df, end_to_end["decision"]["threshold"])
    _log(f"report figures written: {', '.join(report_figures)}")

    ours = {
        "fused, ground-truth crops (recognition stage)": {
            "digit_accuracy": 100 * float(np.mean(recognition["testing"]["fused"]["per_digit_accuracy"])),
            "full_reading_accuracy": 100 * recognition["testing"]["fused"]["full_sequence_accuracy"]},
        "fused, detector crops (end to end)": {
            "digit_accuracy": 100 * float(np.mean(end_to_end["fused"]["per_digit_accuracy_on_detected"])),
            "full_reading_accuracy": 100 * end_to_end["fused"]["full_sequence_accuracy"]},
        "from-scratch CNN alone, ground-truth crops": {
            "digit_accuracy": 100 * float(np.mean(recognition["testing"]["cnn"]["per_digit_accuracy"])),
            "full_reading_accuracy": 100 * recognition["testing"]["cnn"]["full_sequence_accuracy"]},
    }
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detection": detection,
        "recognition_ground_truth_crops": recognition,
        "confusion": confusion,
        "decision_ground_truth_crops": decision,
        "end_to_end_validation": end_to_end_val,
        "end_to_end_test": end_to_end,
        "failures": failures,
        "published_baseline": {**PUBLISHED_BASELINE, "ours": ours,
                               "caption": "Published rows are a 10-run mean on detector-cropped counters; ours are single runs. "
                                          "Full-reading accuracy = all five digits correct in both."},
        "figures": ["iou_histogram.png", "confusion_matrix.png", "confidence_vs_correctness.png", "failure_gallery.jpg"],
    }
    out = results_path("evaluation_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"wrote {out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-detector-val", action="store_true",
                        help="reuse results/yolo/yolo_metrics.json instead of re-running the detector validation")
    args = parser.parse_args()
    run_evaluation(skip_detector_val=args.skip_detector_val)

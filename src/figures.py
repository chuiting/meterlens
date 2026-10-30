"""Report figures shared by evaluate.py and the notebooks."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG, resolve_path

MODEL_COLORS = {"cnn": "#3776ab", "cnn_rgb": "#6fa8dc", "svm": "#ff7f0e", "rf": "#9467bd", "fused": "#2ca02c"}


def _load_json(key):
    with open(resolve_path(CONFIG["paths"][key])) as f:
        return json.load(f)


def _save(fig, save_path, **kwargs):
    if save_path is not None:
        fig.savefig(save_path, **kwargs)
    return fig


def plot_iou_histogram(save_path=None):
    """Phase 2: IoU of YOLO's box against the annotated counter box, test split."""
    preds = pd.read_csv(resolve_path(CONFIG["paths"]["yolo_test_predictions"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(preds["iou"], bins=np.linspace(0, 1, 21), color="#E8743B")
    ax.axvline(0.5, color="#d62728", linestyle="--", linewidth=1.5)
    ax.text(0.51, 0.95, "IoU = 0.5", color="#d62728", transform=ax.get_xaxis_transform(), va="top")
    ax.set_xlabel("IoU (YOLO's box vs annotated box)")
    ax.set_ylabel("Test Images")
    ax.set_title("YOLO IoU Performance on Test Set")
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, save_path, dpi=130)


def plot_cnn_ablation(save_path=None):
    """Phase 3: validation full-sequence accuracy after each CNN technique is added."""
    ablation = _load_json("cnn_ablation")
    step_display = {"baseline": "Baseline", "+he_init": "+He init", "+l2_reg": "+L2",
                    "+lr_step_decay": "+LR decay", "+augmentation": "+Augment",
                    "+digit_permutation": "+Synthesis", "+real_finetune": "+Fine-tune"}
    labels = [f"{i+1}\n{step_display.get(r['step'], r['step'])}" for i, r in enumerate(ablation)]
    pct = [r["val_full_sequence_accuracy"] * 100 for r in ablation]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(ablation))
    ax.plot(x, pct, marker="o", color="#E8743B", linewidth=2.5, markersize=7, zorder=3)
    ax.fill_between(x, pct, 0, color="#E8743B", alpha=0.12)
    for i, v in enumerate(pct):
        ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.1f}")
    ax.set_ylabel("Validation full-sequence accuracy [%]")
    ax.set_title(f"CNN Technique Ablation ({len(ablation)} Rounds)", fontsize=14)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, save_path, dpi=130)


def plot_sensing_histogram(save_path=None):
    """Phase 7: anomaly distance of every ground-truth crop, per split, with the threshold."""
    metrics = _load_json("sensing_metrics")
    scores = pd.read_csv(resolve_path(CONFIG["paths"]["sensing_scores"]))
    scores = scores[scores["crop_source"] == "ground_truth"]
    distances = {split: scores.loc[scores["split"] == split, "distance"].to_numpy()
                 for split in ("training", "validation", "testing")}
    val_mean, threshold = metrics["validation_mean_distance"], metrics["sensing_distance_threshold"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, max(d.max() for d in distances.values()) * 1.02, 50)
    for split, color in (("training", "#1f77b4"), ("validation", "#ff7f0e"), ("testing", "#2ca02c")):
        ax.hist(distances[split], bins=bins, alpha=0.5, color=color,
                label=f"{split.capitalize()} (n={len(distances[split])})")
    ax.axvline(val_mean, color="#555555", linestyle=":", linewidth=1.5, label="Mean (μ)")
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label="Threshold (μ + 2σ)")
    # value labels beside the lines
    ax.annotate(f"Mean = {val_mean:.0f}", xy=(val_mean, 0.9), xycoords=("data", "axes fraction"),
                xytext=(5, 0), textcoords="offset points", color="#555555", fontsize=10,
                va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.annotate(f"Threshold = {threshold:.0f}", xy=(threshold, 0.45), xycoords=("data", "axes fraction"),
                xytext=(5, 0), textcoords="offset points", color="#d62728", fontsize=10,
                va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_axisbelow(True)                                   # keep the grid behind the bars
    ax.grid(axis="y", linestyle="--", color="#e6e6e6", linewidth=0.8)
    ax.set_xlabel("Distance to Nearest K-Means Centroid")
    ax.set_ylabel("No. of crops")
    ax.set_title("Anomaly Distance Distribution using PCA and K-Means")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path, dpi=150)


def plot_digit_recognition(recognition, save_path=None):
    """Phase 9: full-sequence and per-position accuracy of every recogniser on ground-truth crops."""
    models = list(recognition)
    full = [recognition[m]["full_sequence_accuracy"] * 100 for m in models]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})
    # left: all five digits right
    a1.bar(models, full, color=[MODEL_COLORS.get(m, "#7f7f7f") for m in models])
    a1.set_ylim(0, 100); a1.set_ylabel("accuracy (%)"); a1.set_title("Full-sequence accuracy\n(all 5 digits right)", fontsize=11)
    for i, v in enumerate(full):
        a1.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    # right: each digit position on its own
    w = 0.8 / len(models); x = np.arange(5)
    for j, m in enumerate(models):
        vals = [v * 100 for v in recognition[m]["per_digit_accuracy"]]
        a2.bar(x + (j - (len(models) - 1) / 2) * w, vals, w, color=MODEL_COLORS.get(m, "#7f7f7f"), label=m)
        for xi, v in zip(x + (j - (len(models) - 1) / 2) * w, vals):
            a2.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=6)
    a2.set_xticks(x); a2.set_xticklabels([f"d{i+1}" for i in x]); a2.set_ylim(0, 108); a2.set_ylabel("accuracy (%)")
    a2.set_title("Per-digit accuracy by position", fontsize=11); a2.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    for a in (a1, a2):
        a.set_axisbelow(True); a.grid(axis="y", linestyle="--", color="#e6e6e6")
    fig.suptitle("Digit Recognition on ground-truth crops", y=1.02, fontsize=12)
    fig.tight_layout()
    return _save(fig, save_path, dpi=130, bbox_inches="tight")


def decision_flow_counts(predictions):
    """Split end-to-end readings into correct/wrong x approved/flagged, in plot_decision_flow's order."""
    ok = predictions["correct"].astype(bool)
    app = predictions["decision"] == "APPROVED"
    return int((ok & app).sum()), int((ok & ~app).sum()), int((~ok & app).sum()), int((~ok & ~app).sum())


def save_report_figures(recognition, test_predictions, threshold):
    """Write every figure the Inspector Models page shows (called from evaluate.py)."""
    from decision import plot_decision_flow

    paths = CONFIG["paths"]
    written = {
        "iou": plot_iou_histogram(resolve_path(paths["yolo_test_predictions"]).parent / "iou_histogram.png"),
        "ablation": plot_cnn_ablation(resolve_path(paths["cnn_ablation"]).parent / "cnn_ablation.png"),
        "sensing": plot_sensing_histogram(resolve_path(paths["sensing_distance_histogram"])),
        "recognition": plot_digit_recognition(recognition, resolve_path(paths["evaluation_summary"]).parent / "digit_recognition.png"),
        "decision": plot_decision_flow(*decision_flow_counts(test_predictions), threshold,
                                       title="How decision layer affects correct and wrong readings?",
                                       save_path=resolve_path(paths["decision_metrics"]).parent / "decision_layer_effect.png"),
    }
    for fig in written.values():
        plt.close(fig)
    return list(written)

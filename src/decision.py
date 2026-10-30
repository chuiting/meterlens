"""Phase 8: decision layer - approve a reading or flag it for review.

APPROVED if confidence >= threshold and no sensing flag, else FLAGGED with reasons.
The threshold is chosen on validation and written to config.yaml.
"""

import json
import re
import time

import numpy as np

from config import CONFIG, CONFIG_PATH, resolve_path
from sensing import SENSING_REASON

NUM_DIGITS = 5
CONFIDENCE_REASON = "Low model confidence -- manual review required"


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_reading(reading: str, confidence: float, sensing_flag: bool = False, cfg: dict = None):
    """Return ("APPROVED", []) or ("FLAGGED", reasons)."""
    cfg = cfg if cfg is not None else CONFIG["decision"]
    threshold = cfg["confidence_threshold"]
    if threshold is None:
        raise ValueError("decision.confidence_threshold is not set -- run src/decision.py first")
    reasons = []
    if sensing_flag:
        reasons.append({"kind": "sensing", "message": SENSING_REASON})
    if confidence < threshold:
        reasons.append({"kind": "confidence", "message": CONFIDENCE_REASON})
    status = "FLAGGED" if reasons else "APPROVED"
    return status, reasons


def load_fused(path):
    z = np.load(resolve_path(path))
    probs = np.stack([z[f"digit_{i + 1}_probs"] for i in range(NUM_DIGITS)], axis=1)
    truth = np.array([[int(c) for c in r] for r in z["readings"]])
    correct = (probs.argmax(-1) == truth).all(axis=1)
    confidence = probs.max(-1).min(axis=1)  # lowest digit confidence
    return confidence, correct


def sweep(confidence, correct, thresholds):
    """Per threshold: flag rate, share of errors caught, approved-set accuracy."""
    rows = []
    n = len(correct)
    n_wrong = int((~correct).sum())
    for t in thresholds:
        flagged = confidence < t
        approved = ~flagged
        rows.append({
            "threshold": float(t),
            "flag_rate": float(flagged.mean()),
            "errors_caught": float((flagged & ~correct).sum() / n_wrong) if n_wrong else 1.0,
            "correct_flagged": float((flagged & correct).sum() / correct.sum()),
            "approved_accuracy": float(correct[approved].mean()) if approved.any() else None,
            "approved_count": int(approved.sum()),
            "errors_approved": int((approved & ~correct).sum()),
            "n": n,
        })
    return rows


def choose_threshold(rows):
    """Maximise errors caught minus correct flagged; ties go to the lower threshold."""
    best = max(rows, key=lambda r: (r["errors_caught"] - r["correct_flagged"], -r["threshold"]))
    return best["threshold"]


def write_threshold_to_config(threshold: float):
    text = CONFIG_PATH.read_text()
    text, n = re.subn(r"(?m)^(  confidence_threshold:).*$", rf"\1 {threshold:.2f}", text, count=1)
    if n != 1:
        raise RuntimeError("could not find decision.confidence_threshold in config.yaml")
    CONFIG_PATH.write_text(text)
    _log(f"config.yaml: decision.confidence_threshold = {threshold:.2f}")


def at_threshold(rows, threshold):
    return next(r for r in rows if abs(r["threshold"] - threshold) < 1e-9)


def run_decision():
    val_conf, val_ok = load_fused(CONFIG["paths"]["fusion_val_probs"])
    test_conf, test_ok = load_fused(CONFIG["paths"]["fusion_test_probs"])
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    val_rows = sweep(val_conf, val_ok, thresholds)
    test_rows = sweep(test_conf, test_ok, thresholds)
    threshold = choose_threshold(val_rows)
    v, t = at_threshold(val_rows, threshold), at_threshold(test_rows, threshold)

    _log(f"validation: {len(val_ok)} fused readings, {int((~val_ok).sum())} wrong; "
         f"median confidence correct {np.median(val_conf[val_ok]):.2f}, wrong {np.median(val_conf[~val_ok]):.2f}")
    _log(f"chosen threshold (max errors_caught - correct_flagged on validation): {threshold:.2f}")
    for name, r in (("validation", v), ("testing", t)):
        _log(f"  {name:10s} flag rate {r['flag_rate']:.1%}, errors caught {r['errors_caught']:.1%}, "
             f"approved set {r['approved_count']} readings at {r['approved_accuracy']:.3f} accuracy "
             f"({r['errors_approved']} errors slip through)")

    # check both branches (approve / flag) on real readings
    cfg = {"confidence_threshold": threshold}
    hi = int(np.argmax(val_conf)); lo = int(np.argmin(val_conf))
    examples = []
    for idx in (hi, lo):
        decision, reasons = evaluate_reading("(validation image %d)" % idx, float(val_conf[idx]), cfg=cfg)
        examples.append({"index": idx, "confidence": float(val_conf[idx]), "correct": bool(val_ok[idx]),
                         "decision": decision, "reasons": reasons})
        _log(f"  example: confidence {val_conf[idx]:.3f} -> {decision}"
             + (f" ({'; '.join(r['message'] for r in reasons)})" if reasons else ""))
    assert {e["decision"] for e in examples} == {"APPROVED", "FLAGGED"}, "both branches must be exercised"

    reference = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    metrics = {
        "selection_rule": "maximise (errors caught - correct readings flagged) on the validation split; ties to the lower threshold",
        "confidence_definition": "minimum of the five per-digit confidences of the fused reading",
        "confidence_threshold": threshold,
        "at_threshold": {"validation": v, "testing": t},
        "reference_thresholds": {
            "validation": [at_threshold(val_rows, x) for x in reference],
            "testing": [at_threshold(test_rows, x) for x in reference],
        },
        "sweep": {"validation": val_rows, "testing": test_rows},
        "median_confidence": {
            "validation": {"correct": float(np.median(val_conf[val_ok])), "wrong": float(np.median(val_conf[~val_ok]))},
            "testing": {"correct": float(np.median(test_conf[test_ok])), "wrong": float(np.median(test_conf[~test_ok]))},
        },
        "branch_examples": examples,
        "plausibility_check": "none -- no meter identity and no reading history in the dataset (scope decision)",
    }
    out = resolve_path(CONFIG["paths"]["decision_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    _log(f"wrote {out}")
    write_threshold_to_config(threshold)
    return metrics


def plot_decision_flow(correct_approved, correct_flagged, wrong_approved, wrong_flagged, threshold,
                       title="", split="test", save_path=None):
    """Flow diagram from all readings (correct/wrong) to approved/flagged. Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    correct, wrong = correct_approved + correct_flagged, wrong_approved + wrong_flagged
    n, approved, flagged = correct + wrong, correct_approved + wrong_approved, correct_flagged + wrong_flagged
    GREEN, RED, GAP = "#2ca02c", "#d62728", 70
    X0, X1 = 0.45, 2.55
    fig, ax = plt.subplots(figsize=(12, 5.6))

    def bar(x, y, h, color, w=0.45):
        ax.add_patch(Rectangle((x, y), w, h, color=color, zorder=3))

    def smooth(x):
        u = (x - X0) / (X1 - X0)
        return u * u * (3 - 2 * u)

    def ribbon(y0a, y0b, y1a, y1b, color):
        xs = np.linspace(X0, X1, 60)
        s = smooth(xs)
        ax.fill_between(xs, y0a + (y1a - y0a) * s, y0b + (y1b - y0b) * s, color=color, alpha=0.35, lw=0, zorder=2)

    # left bar: all readings split into correct / wrong
    bar(0, 0, correct, GREEN)
    bar(0, correct, wrong, RED)
    # right bars: approved (top) and flagged (below, with a gap), each split into correct / wrong
    fl_top = approved + GAP
    bar(2.55, 0, correct_approved, GREEN)
    bar(2.55, correct_approved, wrong_approved, RED)
    bar(2.55, fl_top, correct_flagged, GREEN)
    bar(2.55, fl_top + correct_flagged, wrong_flagged, RED)

    # the four ribbons: (left top, left bottom, right top, right bottom)
    flows = {
        "correct_approved": (0, correct_approved, 0, correct_approved, GREEN),
        "correct_flagged": (correct_approved, correct, fl_top, fl_top + correct_flagged, GREEN),
        "wrong_approved": (correct, correct + wrong_approved, correct_approved, approved, RED),
        "wrong_flagged": (correct + wrong_approved, n, fl_top + correct_flagged, fl_top + flagged, RED),
    }
    for y0a, y0b, y1a, y1b, color in flows.values():
        ribbon(y0a, y0b, y1a, y1b, color)

    # counts on the bars
    for x, count, y_top, height, size in ((0, correct, 0, correct, 11), (0, wrong, correct, wrong, 11),
                                          (2.55, correct_approved, 0, correct_approved, 11),
                                          (2.55, wrong_approved, correct_approved, wrong_approved, 7),
                                          (2.55, correct_flagged, fl_top, correct_flagged, 11),
                                          (2.55, wrong_flagged, fl_top + correct_flagged, wrong_flagged, 11)):
        ax.text(x + 0.225, y_top + height / 2, f"{count}", ha="center", va="center", fontsize=size,
                color="white", fontweight="bold", zorder=4)

    # labels
    ax.text(-0.08, correct / 2, f"Correct\n({correct / n:.1%})", ha="right", va="center",
            fontsize=10, color=GREEN, fontweight="bold")
    ax.text(-0.08, correct + wrong / 2, f"Wrong\n({wrong / n:.1%})", ha="right", va="center",
            fontsize=10, color=RED, fontweight="bold")
    ax.text(0.22, -45, f"All {n} {split} readings", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(1.5, -45, f"Decision layer (threshold {threshold:.2f})", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(3.08, approved / 2, f"APPROVED automatically\n{approved} readings ({approved / n:.1%})\n"
            f"{correct_approved / approved:.1%} are right", ha="left", va="center", fontsize=10.5, fontweight="bold")
    ax.text(3.08, correct_approved + wrong_approved / 2,
            f"{wrong_approved} wrong readings still approved\n(risky: errors go unchecked)",
            ha="left", va="center", fontsize=9.5, color=RED, fontweight="bold")
    ax.text(3.08, fl_top + correct_flagged / 2 - 8,
            f"{correct_flagged} correct but flagged\n(wasted effort)", ha="left", va="center",
            fontsize=9.5, color=GREEN, fontweight="bold")
    ax.text(3.08, fl_top + flagged / 2 + 28, f"FLAGGED for review\n{flagged} readings ({flagged / n:.1%})\n"
            f"holds {wrong_flagged} of the {wrong} wrong readings ({wrong_flagged / wrong:.1%})",
            ha="left", va="center", fontsize=10.5, fontweight="bold")

    ax.set_xlim(-0.9, 4.6)
    ax.set_ylim(fl_top + flagged + 40, -90)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig


if __name__ == "__main__":
    run_decision()

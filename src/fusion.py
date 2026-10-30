"""Phase 6: confidence-weighted fusion of the recognition models.

Per digit: p = sum(w_m * p_m), digit = argmax(p). Reading confidence = min digit confidence.
The best member combination on validation is written to config.yaml.
"""

import itertools
import json
import re
import time

import numpy as np

from config import CONFIG, CONFIG_PATH, resolve_path

NUM_DIGITS = 5
NUM_CLASSES = 10


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fuse(probs_by_member: dict, weights: dict) -> np.ndarray:
    """probs_by_member: {name: (N, 5, 10)}; weights: {name: w}. Returns (N, 5, 10)."""
    names = list(probs_by_member)
    total = sum(weights[n] for n in names)
    fused = sum(weights[n] / total * probs_by_member[n] for n in names)
    return fused


def read_probs(fused: np.ndarray):
    """(N, 5, 10) -> readings, per-digit confidences (N, 5), reading confidence (N,)."""
    digits = fused.argmax(axis=-1)
    per_digit_conf = fused.max(axis=-1)
    readings = ["".join(str(d) for d in row) for row in digits]
    return readings, per_digit_conf, per_digit_conf.min(axis=1)


def score(fused: np.ndarray, truth: np.ndarray):
    """Full-sequence accuracy and the five per-digit accuracies."""
    correct = fused.argmax(axis=-1) == truth
    return float(correct.all(axis=1).mean()), [float(x) for x in correct.mean(axis=0)]


def accuracy_weights(per_digit_val_acc: dict, names) -> dict:
    """Weights proportional to mean validation per-digit accuracy, normalised over names."""
    raw = {n: float(np.mean(per_digit_val_acc[n])) for n in names}
    total = sum(raw.values())
    return {n: raw[n] / total for n in names}


def load_probs(path):
    """Load a saved probability file -> basenames, truth (N, 5), probs (N, 5, 10)."""
    z = np.load(resolve_path(path))
    probs = np.stack([z[f"digit_{i + 1}_probs"] for i in range(NUM_DIGITS)], axis=1).astype(np.float64)
    truth = np.array([[int(c) for c in r] for r in z["readings"]])
    return list(z["basenames"]), truth, probs


def load_members(split_key: str):
    """Load every member's probs for one split (checks the image order matches)."""
    members = CONFIG["fusion"]["members"]
    basenames = truth = None
    probs = {}
    for m in members:
        b, t, p = load_probs(m[split_key])
        if basenames is None:
            basenames, truth = b, t
        else:
            assert b == basenames, f"{m['name']}: image order differs from {members[0]['name']}"
            assert (t == truth).all(), f"{m['name']}: readings differ from {members[0]['name']}"
        probs[m["name"]] = p
    readings = ["".join(str(d) for d in row) for row in truth]
    return basenames, truth, readings, probs


def write_weights_to_config(selected: list, weights: dict):
    """Update only the `selected_members:` and `weights:` lines in config.yaml."""
    text = CONFIG_PATH.read_text()
    sel = "[" + ", ".join(selected) + "]"
    w = "{" + ", ".join(f"{n}: {weights[n]:.4f}" for n in selected) + "}"
    text, n1 = re.subn(r"(?m)^(  selected_members:).*$", rf"\1 {sel}", text, count=1)
    text, n2 = re.subn(r"(?m)^(  weights:).*$", rf"\1 {w}", text, count=1)
    if n1 != 1 or n2 != 1:
        raise RuntimeError("could not find the fusion selected_members/weights lines in config.yaml")
    CONFIG_PATH.write_text(text)
    _log(f"config.yaml: fusion.selected_members = {sel}, fusion.weights = {w}")


def save_fused_probs(path, basenames, readings, fused):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"basenames": np.array(basenames), "readings": np.array(readings)}
    for i in range(NUM_DIGITS):
        data[f"digit_{i + 1}_probs"] = fused[:, i].astype(np.float32)
    np.savez(path, **data)
    _log(f"wrote {path}")


def run_fusion():
    cfg = CONFIG["fusion"]
    names = [m["name"] for m in cfg["members"]]
    required = list(cfg["required_minimum"])
    _log(f"members: {names}; required minimum: {required}")

    val_b, val_t, val_r, val_p = load_members("val_probs")
    test_b, test_t, test_r, test_p = load_members("test_probs")

    # single models
    singles = {}
    for n in names:
        vf, vd = score(val_p[n], val_t)
        tf, td = score(test_p[n], test_t)
        singles[n] = {"validation": {"full_sequence_accuracy": vf, "per_digit_accuracy": vd},
                      "testing": {"full_sequence_accuracy": tf, "per_digit_accuracy": td}}
        _log(f"  {n:8s} val {vf:.4f}  test {tf:.4f}  val per-digit {[round(x, 3) for x in vd]}")
    val_acc = {n: singles[n]["validation"]["per_digit_accuracy"] for n in names}

    # every combination of two or more members
    combos = {}
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            w = accuracy_weights(val_acc, combo)
            vf, vd = score(fuse({n: val_p[n] for n in combo}, w), val_t)
            tf, td = score(fuse({n: test_p[n] for n in combo}, w), test_t)
            key = "+".join(combo)
            combos[key] = {"members": list(combo), "weights": w,
                           "validation": {"full_sequence_accuracy": vf, "per_digit_accuracy": vd},
                           "testing": {"full_sequence_accuracy": tf, "per_digit_accuracy": td}}
            _log(f"  {key:22s} val {vf:.4f}  test {tf:.4f}  weights {{{', '.join(f'{n}: {w[n]:.3f}' for n in combo)}}}")

    # selection on validation: best full-sequence, fewest members on a tie
    best_key = max(combos, key=lambda k: (combos[k]["validation"]["full_sequence_accuracy"], -len(combos[k]["members"])))
    best = combos[best_key]
    best_single = max(names, key=lambda n: singles[n]["validation"]["full_sequence_accuracy"])
    required_key = "+".join(required)

    # error overlap on validation for the selected combination
    wrong = {n: set(np.where(~(val_p[n].argmax(-1) == val_t).all(1))[0]) for n in best["members"]}
    fused_val = fuse({n: val_p[n] for n in best["members"]}, best["weights"])
    fused_test = fuse({n: test_p[n] for n in best["members"]}, best["weights"])
    wrong_fused = set(np.where(~(fused_val.argmax(-1) == val_t).all(1))[0])
    overlap = {
        "wrong_readings_per_member": {n: len(s) for n, s in wrong.items()},
        "wrong_in_every_member": len(set.intersection(*wrong.values())),
        "wrong_after_fusion": len(wrong_fused),
        "fixed_by_fusion": len(set.union(*wrong.values()) - wrong_fused),
        "broken_by_fusion": len(wrong_fused - set.union(*wrong.values())),
    }

    # confidence of correct vs wrong readings
    def conf_stats(fused, truth):
        _, _, conf = read_probs(fused)
        ok = (fused.argmax(-1) == truth).all(1)
        return {"median_confidence_correct": float(np.median(conf[ok])),
                "median_confidence_wrong": float(np.median(conf[~ok])) if (~ok).any() else None}

    metrics = {
        "weight_scheme": "proportional to mean validation per-digit accuracy, normalised over the combined members",
        "reading_confidence": "minimum of the five per-digit confidences",
        "singles": singles,
        "combinations": combos,
        "required_minimum": {"key": required_key, **combos[required_key]},
        "selected": {"key": best_key, **best,
                     "validation_error_overlap": overlap,
                     "confidence": {"validation": conf_stats(fused_val, val_t), "testing": conf_stats(fused_test, test_t)}},
        "best_single_model": {"name": best_single, **singles[best_single]},
    }
    out = resolve_path(CONFIG["paths"]["fusion_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    _log(f"wrote {out}")

    save_fused_probs(CONFIG["paths"]["fusion_val_probs"], val_b, val_r, fused_val)
    save_fused_probs(CONFIG["paths"]["fusion_test_probs"], test_b, test_r, fused_test)
    write_weights_to_config(best["members"], best["weights"])

    print(f"\nselected on validation: {best_key}  ->  val {best['validation']['full_sequence_accuracy']:.4f}, "
          f"test {best['testing']['full_sequence_accuracy']:.4f}")
    print(f"required minimum {required_key}: val {combos[required_key]['validation']['full_sequence_accuracy']:.4f}, "
          f"test {combos[required_key]['testing']['full_sequence_accuracy']:.4f}")
    print(f"best single model {best_single}: val {singles[best_single]['validation']['full_sequence_accuracy']:.4f}, "
          f"test {singles[best_single]['testing']['full_sequence_accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    run_fusion()

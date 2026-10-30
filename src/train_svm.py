"""Phase 4: SVM digit classifier on HOG features.

reference: trained on ground-truth digit crops; deployed: trained on equal-width strips.
"""

import gc
import json
import random
import time
from collections import defaultdict
from itertools import product

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from config import CONFIG, resolve_path
from predicted_crops import load_predicted_crops

NUM_DIGITS = 5
NUM_CLASSES = 10
SOURCES = ("digit_crops", "counter_strips")
ROLES = {"digit_crops": "reference", "counter_strips": "deployed"}


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def extract_hog(gray_digit, cfg):
    return hog(
        gray_digit,
        orientations=cfg["hog_orientations"],
        pixels_per_cell=tuple(cfg["hog_pixels_per_cell"]),
        cells_per_block=tuple(cfg["hog_cells_per_block"]),
    )


def images_to_hog_matrix(images, cfg):
    return np.array([extract_hog(img, cfg) for img in images])


def equal_width_strips(counter_crop, digit_size):
    """Split a 400x106 counter crop into 5 equal-width strips, resized to digit_size."""
    h, w = counter_crop.shape[:2]
    strip_w = w / NUM_DIGITS
    strips = []
    for i in range(NUM_DIGITS):
        x0, x1 = int(round(i * strip_w)), int(round((i + 1) * strip_w))
        strips.append(cv2.resize(counter_crop[:, x0:x1], digit_size))
    return strips


def projection_profile_strips(counter_crop, digit_size):
    """Cut at the column with the least ink near each equal-width boundary (within 30%).

    Strokes are made the foreground whatever the counter's polarity (Otsu, then
    invert if strokes are the majority).
    """
    gray = cv2.cvtColor(counter_crop, cv2.COLOR_BGR2GRAY) if counter_crop.ndim == 3 else counter_crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() > 127:
        binary = 255 - binary
    col_density = binary.sum(axis=0).astype(np.float64)

    h, w = gray.shape
    strip_w = w / NUM_DIGITS
    search_radius = int(strip_w * 0.3)

    cut_points = [0]
    for i in range(1, NUM_DIGITS):
        expected = int(round(i * strip_w))
        lo, hi = max(1, expected - search_radius), min(w - 1, expected + search_radius)
        cut_points.append(lo + int(np.argmin(col_density[lo:hi])))
    cut_points.append(w)

    strips = []
    for i in range(NUM_DIGITS):
        x0, x1 = cut_points[i], cut_points[i + 1]
        if x1 <= x0:  # degenerate valley search -> equal-width for this slice
            x0, x1 = int(round(i * strip_w)), int(round((i + 1) * strip_w))
        strips.append(cv2.resize(counter_crop[:, x0:x1], digit_size))
    return strips


SEGMENTERS = {"equal_width": equal_width_strips, "projection_profile": projection_profile_strips}


def load_digit_crops(split: str):
    """Ground-truth digit crops -> (images, labels, (basename, position) meta)."""
    digits_dir = resolve_path(CONFIG["paths"]["recognition_dir"]) / split / "digits"
    images, labels, meta = [], [], []
    for p in sorted(digits_dir.glob("*.jpg")):
        basename, pos, label = p.stem.split("_")
        images.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        labels.append(int(label))
        meta.append((basename, int(pos[1:])))
    return images, labels, meta


def load_counter_strips(split: str, digit_size):
    """Equal-width strips of the ground-truth counter crops (same layout as load_digit_crops)."""
    recog_dir = resolve_path(CONFIG["paths"]["recognition_dir"]) / split
    with open(recog_dir / "labels.json") as f:
        labels_json = json.load(f)
    images, labels, meta = [], [], []
    for basename in sorted(labels_json):
        gray = cv2.imread(str(recog_dir / "counters" / f"{basename}.jpg"), cv2.IMREAD_GRAYSCALE)
        for pos, strip in enumerate(equal_width_strips(gray, digit_size)):
            images.append(strip)
            labels.append(int(labels_json[basename][pos]))
            meta.append((basename, pos + 1))
    return images, labels, meta


def load_source(split: str, source: str, digit_size):
    if source == "digit_crops":
        return load_digit_crops(split)
    if source == "counter_strips":
        return load_counter_strips(split, digit_size)
    raise ValueError(source)


def prepare_hog_datasets(source: str, hog_cfg, digit_size, max_train_samples, seed):
    """HOG features for all splits, scaler fitted on training. Shared with train_rf.py."""
    _log(f"loading {source} for training/validation/testing")
    train_imgs, train_labels, _ = load_source("training", source, digit_size)
    val_imgs, val_labels, val_meta = load_source("validation", source, digit_size)
    test_imgs, test_labels, test_meta = load_source("testing", source, digit_size)

    if len(train_imgs) > max_train_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_imgs), size=max_train_samples, replace=False)
        train_imgs = [train_imgs[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]
        _log(f"subsampled training samples to {max_train_samples}")

    _log(f"extracting HOG: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    X_train = images_to_hog_matrix(train_imgs, hog_cfg)
    X_val = images_to_hog_matrix(val_imgs, hog_cfg)
    X_test = images_to_hog_matrix(test_imgs, hog_cfg)
    scaler = StandardScaler().fit(X_train)
    del train_imgs, val_imgs, test_imgs
    gc.collect()
    return {
        "X_train": scaler.transform(X_train), "y_train": train_labels,
        "X_val": scaler.transform(X_val), "y_val": val_labels, "val_meta": val_meta,
        "X_test": scaler.transform(X_test), "y_test": test_labels, "test_meta": test_meta,
        "scaler": scaler,
    }


def full_sequence_accuracy(meta, y_true, y_pred, n_readings=None):
    """Per-digit and full-reading accuracy; n_readings counts detector misses as wrong."""
    per_digit_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    by_reading = defaultdict(dict)
    for (basename, pos), t, p in zip(meta, y_true, y_pred):
        by_reading[basename][pos] = (t, p)
    correct = sum(1 for positions in by_reading.values()
                  if len(positions) == NUM_DIGITS and all(t == p for t, p in positions.values()))
    denom = n_readings if n_readings is not None else len(by_reading)
    return per_digit_acc, (correct / denom if denom else 0.0)


def grid_search_svm(X_train, y_train, X_val, y_val, cfg, seed):
    """Grid over C and gamma, selected on validation (probability=True for fusion)."""
    best_model, best_acc, best_params, log = None, -1.0, None, []
    for C, gamma in product(cfg["C_grid"], cfg["gamma_grid"]):
        model = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=seed)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        log.append({"C": C, "gamma": gamma, "val_per_digit_accuracy": acc})
        _log(f"  C={C}, gamma={gamma} -> val per-digit accuracy {acc:.4f}")
        if acc > best_acc:
            best_model, best_acc, best_params = model, acc, {"C": C, "gamma": gamma}
    return best_model, best_params, best_acc, log


def evaluate_on_predicted_crops(model, scaler, hog_cfg, digit_size, split: str):
    """Evaluate on detector crops with both segmenters; misses get uniform probabilities."""
    basenames, readings, crops = load_predicted_crops(split, grayscale=True)
    n_total = len(basenames)
    n_miss = sum(c is None for c in crops)
    out = {}
    for name, segment in SEGMENTERS.items():
        y_true, y_pred, meta = [], [], []
        probs = np.full((n_total, NUM_DIGITS, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
        detected = np.zeros(n_total, dtype=bool)
        strips, owners = [], []
        for i, (b, crop) in enumerate(zip(basenames, crops)):
            if crop is None:
                continue
            detected[i] = True
            for pos, s in enumerate(segment(crop, digit_size)):
                strips.append(s)
                owners.append((i, pos))
        feats = scaler.transform(images_to_hog_matrix(strips, hog_cfg))
        proba = model.predict_proba(feats)
        classes = list(model.classes_)
        for (i, pos), row in zip(owners, proba):
            for c, p in zip(classes, row):
                probs[i, pos, int(c)] = p
            pred = int(classes[int(np.argmax(row))])
            y_true.append(int(readings[i][pos]))
            y_pred.append(pred)
            meta.append((basenames[i], pos + 1))
        per_digit, full_seq = full_sequence_accuracy(meta, y_true, y_pred, n_readings=n_total)
        out[name] = {
            "metrics": {
                "per_digit_accuracy": per_digit,
                "full_sequence_accuracy": full_seq,
                "detector_miss_count": n_miss,
                "detector_miss_rate": n_miss / n_total,
                "n_images": n_total,
            },
            "probs": probs, "detected": detected, "basenames": basenames, "readings": readings,
        }
    return out


def predict_proba_matrix(model, X, meta, n_readings, basenames):
    """Per-digit probabilities (N, 5, 10) from a flat feature matrix."""
    index = {b: i for i, b in enumerate(basenames)}
    probs = np.full((n_readings, NUM_DIGITS, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    for (b, pos), row in zip(meta, proba):
        for c, p in zip(classes, row):
            probs[index[b], pos - 1, int(c)] = p
    return probs


def save_probs(path, basenames, readings, gt_probs, predicted_probs, detected):
    """Save probabilities in the CNN's .npz layout, plus the detector-crop version."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"basenames": np.array(basenames), "readings": np.array(readings), "detected": detected}
    for i in range(NUM_DIGITS):
        data[f"digit_{i + 1}_probs"] = gt_probs[:, i]
        data[f"digit_{i + 1}_probs_predicted"] = predicted_probs[:, i]
    np.savez(path, **data)
    _log(f"wrote {path}")


def run_classifier_experiment(classifier_name, grid_search_fn, grid_cfg, weights_paths,
                              scaler_paths, probs_paths, metrics_path, extra_metrics_fn=None):
    """Train and evaluate both variants of one classifier; shared by the SVM and RF.

    grid_search_fn(X_train, y_train, X_val, y_val, grid_cfg, seed) -> (model, params, acc, log).
    """
    seed = CONFIG["seed"]
    set_seeds(seed)
    hog_cfg = CONFIG["svm"]
    digit_size = tuple(CONFIG["image_sizes"]["digit_crop"])
    max_train = CONFIG["svm"]["max_train_samples"]

    datasets = {s: prepare_hog_datasets(s, hog_cfg, digit_size, max_train, seed) for s in SOURCES}
    n_readings = {"validation": len({b for b, _ in datasets["digit_crops"]["val_meta"]}),
                  "testing": len({b for b, _ in datasets["digit_crops"]["test_meta"]})}

    variants = {}
    for source in SOURCES:
        d = datasets[source]
        _log(f"[{classifier_name} / {source} ({ROLES[source]})] grid search, selection on validation")
        model, params, val_acc, log = grid_search_fn(d["X_train"], d["y_train"], d["X_val"], d["y_val"], grid_cfg, seed)
        _log(f"  best {params} (val per-digit {val_acc:.4f})")

        models_dir = resolve_path(CONFIG["paths"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, resolve_path(CONFIG["paths"][weights_paths[source]]))
        joblib.dump(d["scaler"], resolve_path(CONFIG["paths"][scaler_paths[source]]))

        evaluation = {}
        for split in ("validation", "testing"):
            key_x, key_y, key_m = ("X_val", "y_val", "val_meta") if split == "validation" else ("X_test", "y_test", "test_meta")
            split_eval = {}
            # ground-truth inputs
            for eval_source in SOURCES:
                de = datasets[eval_source]
                # rescale with this variant's scaler
                X_raw = de["scaler"].inverse_transform(de[key_x])
                X = d["scaler"].transform(X_raw)
                pred = model.predict(X)
                per_digit, full_seq = full_sequence_accuracy(de[key_m], de[key_y], pred)
                split_eval[f"ground_truth_{eval_source}"] = {"per_digit_accuracy": per_digit,
                                                             "full_sequence_accuracy": full_seq}
            # detector crops, both segmentations
            predicted = evaluate_on_predicted_crops(model, d["scaler"], hog_cfg, digit_size, split)
            split_eval["predicted_crops"] = {name: r["metrics"] for name, r in predicted.items()}
            evaluation[split] = split_eval
            if ROLES[source] == "deployed":
                variants.setdefault("_predicted", {})[split] = predicted
                de = datasets["counter_strips"]
                variants.setdefault("_gt_probs", {})[split] = predict_proba_matrix(
                    model, de[key_x], de[key_m], n_readings[split], predicted["equal_width"]["basenames"])

        variants[source] = {
            "role": ROLES[source],
            "grid_search": {"log": log, "best_params": params, "best_val_per_digit_accuracy": val_acc},
            "evaluation": evaluation,
        }
        if extra_metrics_fn is not None:
            variants[source].update(extra_metrics_fn(model))

    # choose the segmentation on validation
    dep_val = variants["counter_strips"]["evaluation"]["validation"]["predicted_crops"]
    chosen = max(dep_val, key=lambda k: dep_val[k]["full_sequence_accuracy"])
    reasons = {
        "equal_width": "equal-width strips matched or beat projection-profile cutting on the validation split, "
                       "consistent with the near-uniform digit spacing on these mechanical counters",
        "projection_profile": "projection-profile cutting beat equal-width strips on the validation split",
    }

    # save the deployed variant's probabilities for fusion
    for split in ("validation", "testing"):
        p = variants["_predicted"][split][chosen]
        save_probs(CONFIG["paths"][probs_paths[split]], p["basenames"], p["readings"],
                   variants["_gt_probs"][split], p["probs"], p["detected"])

    metrics = {
        "classifier": classifier_name,
        "hog": {k: hog_cfg[k] for k in ("hog_orientations", "hog_pixels_per_cell", "hog_cells_per_block")},
        "deployed_variant": "counter_strips",
        "variants": {s: variants[s] for s in SOURCES},
        "segmentation_choice": {"chosen": chosen, "selected_on": "validation", "reason": reasons[chosen]},
    }
    metrics_path = resolve_path(CONFIG["paths"][metrics_path])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    _log(f"wrote {metrics_path}")
    print_summary(metrics)
    return metrics


EVALUATION_INPUTS = [
    ("ground-truth digit crops", ("ground_truth_digit_crops",)),
    ("ground-truth counter strips", ("ground_truth_counter_strips",)),
    ("detector crops, equal-width", ("predicted_crops", "equal_width")),
    ("detector crops, projection-profile", ("predicted_crops", "projection_profile")),
]


def evaluation_table(metrics):
    """One row per (training source, evaluation input) on the test split."""
    rows = []
    for source, v in metrics["variants"].items():
        for label, keys in EVALUATION_INPUTS:
            cell = v["evaluation"]["testing"]
            for k in keys:
                cell = cell[k]
            rows.append({"trained on": source, "role": v["role"], "evaluated on": label,
                         "per-digit": round(cell["per_digit_accuracy"], 4),
                         "full-sequence": round(cell["full_sequence_accuracy"], 4)})
    return rows


def print_summary(metrics):
    print(f"\n{metrics['classifier']}: test-split accuracy (per-digit / full-sequence)")
    print(f"{'trained on':16s} {'GT digit crops':>18s} {'GT counter strips':>20s} {'detector equal-width':>22s} {'detector proj-profile':>22s}")
    for source, v in metrics["variants"].items():
        e = v["evaluation"]["testing"]
        cells = [e["ground_truth_digit_crops"], e["ground_truth_counter_strips"],
                 e["predicted_crops"]["equal_width"], e["predicted_crops"]["projection_profile"]]
        print(f"{source:16s} " + " ".join(f"{c['per_digit_accuracy']:.3f} / {c['full_sequence_accuracy']:.3f}".rjust(w)
                                          for c, w in zip(cells, (18, 20, 22, 22))))
    print(f"segmentation used downstream: {metrics['segmentation_choice']['chosen']} (chosen on validation)")


def train_svm():
    return run_classifier_experiment(
        classifier_name="SVM",
        grid_search_fn=grid_search_svm,
        grid_cfg=CONFIG["svm"],
        weights_paths={"digit_crops": "svm_digit_crop_weights", "counter_strips": "svm_weights"},
        scaler_paths={"digit_crops": "hog_scaler_digit_crops", "counter_strips": "hog_scaler"},
        probs_paths={"validation": "svm_val_probs", "testing": "svm_test_probs"},
        metrics_path="svm_metrics",
    )


if __name__ == "__main__":
    train_svm()

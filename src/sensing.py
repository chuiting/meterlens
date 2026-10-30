"""Phase 7: image-quality gate on the counter crop (PCA + KMeans).

Score = distance to the nearest cluster centre; flag if above the threshold
(validation mean + 2*std).
"""

import csv
import json
import re
import time

import cv2
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import CONFIG, CONFIG_PATH, resolve_path

np.random.seed(CONFIG["seed"])

SENSING_REASON = ("This photo looks unusual compared to typical meter photos, which can make the "
                  "reading less reliable")
SPLITS = ("training", "validation", "testing")


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def crop_to_vector(crop):
    """Flatten a counter crop to a grayscale 1D vector."""
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.shape[2] == 3 else crop[..., 0]
    return crop.astype(np.float64).reshape(-1)


def fit_pca_kmeans(vectors, variance_target: float, n_clusters: int, seed: int):
    """Fit PCA (to `variance_target` variance), then KMeans on the projected crops."""
    pca = PCA(n_components=variance_target, random_state=seed).fit(vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(pca.transform(vectors))
    return pca, kmeans


def score_crop(vector, pca, kmeans) -> float:
    """Distance from the crop's PCA projection to the nearest cluster centre."""
    return float(kmeans.transform(pca.transform([vector])).min())


_pca_cache = None
_kmeans_cache = None


def _load_pca():
    global _pca_cache
    if _pca_cache is None:
        _pca_cache = joblib.load(resolve_path(CONFIG["paths"]["pca_weights"]))
    return _pca_cache


def _load_kmeans():
    global _kmeans_cache
    if _kmeans_cache is None:
        _kmeans_cache = joblib.load(resolve_path(CONFIG["paths"]["kmeans_weights"]))
    return _kmeans_cache


def evaluate_sensing(crop, pca=None, kmeans=None, threshold: float = None):
    """(distance, flag) for one counter crop; models and threshold load from disk if not given."""
    pca = pca if pca is not None else _load_pca()
    kmeans = kmeans if kmeans is not None else _load_kmeans()
    if threshold is None:
        threshold = CONFIG["sensing"]["sensing_distance_threshold"]
    if threshold is None:
        raise ValueError("sensing.sensing_distance_threshold is not set -- run src/sensing.py first")
    distance = score_crop(crop_to_vector(crop), pca, kmeans)
    return distance, bool(distance > threshold)


def crop_dir(split: str, source: str = "ground_truth"):
    """source: "ground_truth" (counters/) or "detector" (counters_predicted/)."""
    sub = {"ground_truth": "counters", "detector": "counters_predicted"}[source]
    return resolve_path(CONFIG["paths"]["recognition_dir"]) / split / sub


def load_crops(split: str, source: str = "ground_truth"):
    """Sorted (basenames, grayscale crops) for one split."""
    d = crop_dir(split, source)
    paths = sorted(d.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no crops in {d} -- run src/prepare_recognition.py"
                                + (" and src/predicted_crops.py" if source == "detector" else ""))
    return [p.stem for p in paths], [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]


def vectors_for(crops):
    return np.stack([crop_to_vector(c) for c in crops])


def score_all(crops, pca, kmeans):
    return np.array([score_crop(crop_to_vector(c), pca, kmeans) for c in crops])


def write_threshold_to_config(threshold: float):
    text = CONFIG_PATH.read_text()
    text, n = re.subn(r"(?m)^(  sensing_distance_threshold:).*$", rf"\1 {threshold:.1f}", text, count=1)
    if n != 1:
        raise RuntimeError("could not find sensing.sensing_distance_threshold in config.yaml")
    CONFIG_PATH.write_text(text)
    _log(f"config.yaml: sensing.sensing_distance_threshold = {threshold:.1f}")


def recognition_error_by_flag(split: str, names, distances, threshold: float):
    """Fused-model error rate for flagged vs passed crops (None if fusion has not run)."""
    key = {"validation": "fusion_val_probs", "testing": "fusion_test_probs"}.get(split)
    if key is None:
        return None
    path = resolve_path(CONFIG["paths"][key])
    if not path.exists():
        return None
    z = np.load(path)
    probs = np.stack([z[f"digit_{i + 1}_probs"] for i in range(5)], axis=1)
    truth = np.array([[int(c) for c in r] for r in z["readings"]])
    correct = (probs.argmax(-1) == truth).all(axis=1)
    fused_names = [n.decode() if isinstance(n, bytes) else n for n in z["basenames"]]
    correct_by_name = dict(zip(fused_names, correct))

    mask = np.array([n in correct_by_name for n in names])
    if not mask.any():
        return None
    names_m = np.array(names)[mask]
    distances_m = distances[mask]
    correct_m = np.array([correct_by_name[n] for n in names_m])
    flags_m = distances_m > threshold

    def err(sel):
        return float(1 - correct_m[sel].mean()) if sel.any() else None

    return {
        "flagged_error_rate": err(flags_m),
        "flagged_n": int(flags_m.sum()),
        "passed_error_rate": err(~flags_m),
        "passed_n": int((~flags_m).sum()),
    }


def write_scores_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["basename", "split", "crop_source", "distance", "flag"])
        w.writeheader()
        w.writerows(rows)
    _log(f"wrote {path} ({len(rows)} rows)")


def run_sensing():
    cfg = CONFIG["sensing"]

    names, crops = {}, {}
    for split in SPLITS:
        names[split], crops[split] = load_crops(split)
        _log(f"{split:10s} {len(crops[split])} ground-truth crops loaded")

    # fit on training only
    train_vectors = vectors_for(crops["training"])
    pca, kmeans = fit_pca_kmeans(train_vectors, cfg["pca_variance_target"], cfg["n_clusters"], CONFIG["seed"])
    _log(f"PCA kept {pca.n_components_} components of {train_vectors.shape[1]} "
         f"(target {cfg['pca_variance_target']:.0%} variance)")
    _log(f"KMeans cluster sizes: {np.bincount(kmeans.labels_).tolist()}")

    paths = CONFIG["paths"]
    joblib.dump(pca, resolve_path(paths["pca_weights"]))
    joblib.dump(kmeans, resolve_path(paths["kmeans_weights"]))
    _log(f"wrote {resolve_path(paths['pca_weights'])} and {resolve_path(paths['kmeans_weights'])}")

    distances = {split: score_all(crops[split], pca, kmeans) for split in SPLITS}
    for split, d in distances.items():
        _log(f"{split:10s} distance: min {d.min():.0f}, median {np.median(d):.0f}, "
             f"p95 {np.percentile(d, 95):.0f}, max {d.max():.0f}")

    # threshold from validation only: mean + 2*std (the start of the right tail)
    val_mean, val_std = float(distances["validation"].mean()), float(distances["validation"].std())
    threshold = round(val_mean + 2 * val_std, 1)
    _log(f"validation distance mean {val_mean:.0f}, std {val_std:.0f} -> threshold (mean + 2*std) = {threshold:.0f}")

    # flag rates per split, ground-truth and detector crops
    flag_rates, csv_rows = {}, []
    for split in SPLITS:
        flags = distances[split] > threshold
        flag_rates[split] = {"ground_truth_crops": {"flag_rate": float(flags.mean()), "flagged": int(flags.sum()),
                                                    "n": int(len(flags))}}
        csv_rows += [{"basename": b, "split": split, "crop_source": "ground_truth", "distance": f"{d:.2f}", "flag": int(f)}
                     for b, d, f in zip(names[split], distances[split], flags)]
        try:
            det_names, det_crops = load_crops(split, "detector")
        except FileNotFoundError:
            continue
        det_distances = score_all(det_crops, pca, kmeans)
        det_flags = det_distances > threshold
        flag_rates[split]["detector_crops"] = {"flag_rate": float(det_flags.mean()), "flagged": int(det_flags.sum()),
                                               "n": int(len(det_flags))}
        csv_rows += [{"basename": b, "split": split, "crop_source": "detector", "distance": f"{d:.2f}", "flag": int(f)}
                     for b, d, f in zip(det_names, det_distances, det_flags)]
    for split, r in flag_rates.items():
        parts = [f"{src} {v['flagged']}/{v['n']} ({v['flag_rate']:.1%})" for src, v in r.items()]
        _log(f"  {split:10s} flagged: " + ", ".join(parts))

    # does the flag correlate with the model getting the reading wrong?
    recognition_error = {}
    for split in ("validation", "testing"):
        r = recognition_error_by_flag(split, names[split], distances[split], threshold)
        if r is not None:
            recognition_error[split] = r
            _log(f"  {split:10s} fused-model error rate: flagged {r['flagged_error_rate']:.1%} (n={r['flagged_n']}) "
                 f"vs. passed {r['passed_error_rate']:.1%} (n={r['passed_n']})")

    write_scores_csv(resolve_path(paths["sensing_scores"]), csv_rows)

    metrics = {
        "method": "PCA(n_components=0.95) then KMeans(n_clusters=3) on flattened grayscale training crops; "
                 "anomaly score is Euclidean distance to the nearest cluster centroid",
        "pca_n_components": int(pca.n_components_),
        "pca_input_dim": int(train_vectors.shape[1]),
        "kmeans_cluster_sizes": np.bincount(kmeans.labels_).tolist(),
        "selection_rule": "mean + 2*std of the validation distance distribution, read off the histogram by inspection "
                          "(no grid search, no cross-validation, test split not used)",
        "sensing_distance_threshold": threshold,
        "flag_rates": flag_rates,
        "recognition_error_by_flag": recognition_error,
    }
    out = resolve_path(paths["sensing_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    _log(f"wrote {out}")
    write_threshold_to_config(threshold)
    return metrics


if __name__ == "__main__":
    run_sensing()

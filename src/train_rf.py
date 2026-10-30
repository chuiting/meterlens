"""Phase 5: Random Forest on the same HOG features as the SVM (comparison model)."""

from itertools import product

from sklearn.ensemble import RandomForestClassifier

from config import CONFIG
from train_svm import run_classifier_experiment, _log


def grid_search_rf(X_train, y_train, X_val, y_val, cfg, seed):
    """Grid over n_estimators and max_depth, selected on validation."""
    best_model, best_acc, best_params, log = None, -1.0, None, []
    for n_estimators, max_depth in product(cfg["n_estimators_grid"], cfg["max_depth_grid"]):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        log.append({"n_estimators": n_estimators, "max_depth": max_depth, "val_per_digit_accuracy": acc})
        _log(f"  n_estimators={n_estimators}, max_depth={max_depth} -> val per-digit accuracy {acc:.4f}")
        if acc > best_acc:
            best_model, best_acc, best_params = model, acc, {"n_estimators": n_estimators, "max_depth": max_depth}
    return best_model, best_params, best_acc, log


def train_rf():
    return run_classifier_experiment(
        classifier_name="Random Forest",
        grid_search_fn=grid_search_rf,
        grid_cfg=CONFIG["rf"],
        weights_paths={"digit_crops": "rf_digit_crop_weights", "counter_strips": "rf_weights"},
        scaler_paths={"digit_crops": "hog_scaler_digit_crops", "counter_strips": "hog_scaler"},
        probs_paths={"validation": "rf_val_probs", "testing": "rf_test_probs"},
        metrics_path="rf_metrics",
    )


if __name__ == "__main__":
    train_rf()

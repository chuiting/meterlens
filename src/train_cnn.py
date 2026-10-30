"""Phase 3: multi-task CNN with 5 softmax heads, one per digit, plus the ablation study."""

import argparse
import gc
import json
import random
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import CONFIG, resolve_path
from augment import (augment_image, crop_counter_with_jitter, crop_counter_with_margin,
                     DigitPermutationSynthesizer)

NUM_DIGITS = 5
NUM_CLASSES = 10

# Cumulative: each step keeps every addition from the steps before it.
ABLATION_STEPS = [
    {"name": "baseline",           "he_init": False, "l2_reg": False, "step_decay": False, "augment": False, "permute": False},
    {"name": "+he_init",           "he_init": True,  "l2_reg": False, "step_decay": False, "augment": False, "permute": False},
    {"name": "+l2_reg",            "he_init": True,  "l2_reg": True,  "step_decay": False, "augment": False, "permute": False},
    {"name": "+lr_step_decay",     "he_init": True,  "l2_reg": True,  "step_decay": True,  "augment": False, "permute": False},
    {"name": "+augmentation",      "he_init": True,  "l2_reg": True,  "step_decay": True,  "augment": True,  "permute": False},
    {"name": "+digit_permutation", "he_init": True,  "l2_reg": True,  "step_decay": True,  "augment": True,  "permute": True},
    # Continues from the previous step's model.
    {"name": "+real_finetune",     "he_init": True,  "l2_reg": True,  "step_decay": True,  "augment": True,  "permute": True, "finetune": True},
]


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split):
    """Load counter crops (RGB) and readings for one split."""
    recog_dir = resolve_path(CONFIG["paths"]["recognition_dir"]) / split
    counter_w, counter_h = CONFIG["image_sizes"]["counter_crop"]

    with open(recog_dir / "labels.json") as f:
        labels = json.load(f)

    basenames = sorted(labels.keys())
    images = np.zeros((len(basenames), counter_h, counter_w, 3), dtype=np.uint8)
    readings = []
    for idx, basename in enumerate(basenames):
        img_path = recog_dir / "counters" / f"{basename}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"missing counter crop: {img_path}")
        images[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        readings.append("".join(labels[basename]))

    return images, readings, basenames


def load_margin_data(split, expected_readings):
    """Load the context-margin crops and digit geometry, aligned with load_split."""
    recog_dir = resolve_path(CONFIG["paths"]["recognition_dir"]) / split
    with open(recog_dir / "geometry.json") as f:
        geometry = json.load(f)
    with open(recog_dir / "labels.json") as f:
        labels = json.load(f)

    basenames = sorted(labels.keys())
    margin_images, geoms, readings = [], [], []
    for basename in basenames:
        img = cv2.imread(str(recog_dir / "counters_margin" / f"{basename}.jpg"))
        if img is None:
            raise FileNotFoundError(f"missing margin crop for {basename} -- rerun prepare_recognition.py")
        margin_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        geoms.append(geometry[basename])
        readings.append("".join(labels[basename]))
    assert readings == list(expected_readings), "geometry.json and labels.json are out of sync"
    return margin_images, geoms


def to_one_hot(readings):
    labels = [np.zeros((len(readings), NUM_CLASSES), dtype=np.float32) for _ in range(NUM_DIGITS)]
    for i, reading in enumerate(readings):
        for d in range(NUM_DIGITS):
            labels[d][i, int(reading[d])] = 1.0
    return labels


def to_model_input(images_uint8):
    """uint8 RGB (N, h, w, 3) -> float32 in 0-1, single-channel if cnn.grayscale."""
    if CONFIG["cnn"].get("grayscale", False):
        gray = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images_uint8], axis=0)
        return gray[..., None].astype(np.float32) / 255.0
    return images_uint8.astype(np.float32) / 255.0


def input_channels():
    return 1 if CONFIG["cnn"].get("grayscale", False) else 3


def load_eval_crops(split, margin_frac):
    """Val/test crops cut with a symmetric margin around the ground-truth box."""
    recog_dir = resolve_path(CONFIG["paths"]["recognition_dir"]) / split
    counter_w, counter_h = CONFIG["image_sizes"]["counter_crop"]
    with open(recog_dir / "geometry.json") as f:
        geometry = json.load(f)
    with open(recog_dir / "labels.json") as f:
        labels = json.load(f)
    basenames = sorted(labels.keys())
    images = np.zeros((len(basenames), counter_h, counter_w, 3), dtype=np.uint8)
    for idx, basename in enumerate(basenames):
        img = cv2.imread(str(recog_dir / "counters_margin" / f"{basename}.jpg"))
        if img is None:
            raise FileNotFoundError(f"missing margin crop for {basename} -- rerun prepare_recognition.py")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[idx] = crop_counter_with_margin(img, geometry[basename]["counter_bbox"],
                                               (counter_w, counter_h), margin_frac)
    return images


def make_synthesizer(train_margin, train_geoms, train_readings, cross_counter_prob):
    aug_cfg = CONFIG["augmentation"]
    return DigitPermutationSynthesizer(
        train_margin, train_geoms, train_readings,
        cross_counter_prob=cross_counter_prob,
        transition_prob=aug_cfg["digit_transition_prob"],
        transition_split_range=aug_cfg["digit_transition_split_range"],
        same_polarity=aug_cfg.get("digit_permutation_same_polarity", True),
        feather_px=aug_cfg.get("digit_patch_feather_px", 3),
        contrast_gain_range=aug_cfg.get("digit_contrast_gain_range", (0.5, 2.0)),
    )


class CounterSequence(keras.utils.Sequence):
    """Training batches with fresh augmentation every epoch (training split only).

    Each sample may be replaced by a synthesised counter, then jittered, resized
    and augmented.
    """

    def __init__(self, margin_images, geoms, readings, batch_size, synthesizer=None,
                 synth_prob=0.0, shuffle=True):
        # Build batches in background threads (Keras 3); older Keras ignores this.
        try:
            super().__init__(workers=CONFIG["cnn"]["data_workers"], use_multiprocessing=False,
                             max_queue_size=8)
        except TypeError:
            super().__init__()
        self.margin_images = margin_images
        self.geoms = geoms
        self.readings = readings
        self.batch_size = batch_size
        self.synthesizer = synthesizer
        self.synth_prob = synth_prob if synthesizer is not None else 0.0
        self.out_size = tuple(CONFIG["image_sizes"]["counter_crop"])  # (w, h)
        self.shuffle = shuffle
        self.indices = np.arange(len(readings))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.readings) / self.batch_size))

    def make_sample(self, bi):
        """One augmented (image_uint8_400x106, reading) pair for training index bi."""
        if self.synthesizer is not None and random.random() < self.synth_prob:
            margin_img, reading = self.synthesizer.synthesize(bi)
        else:
            margin_img, reading = self.margin_images[bi], self.readings[bi]
        img = crop_counter_with_jitter(margin_img, self.geoms[bi]["counter_bbox"], self.out_size)
        return augment_image(img), reading

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images, batch_readings = [], []
        for bi in batch_idx:
            img, reading = self.make_sample(bi)
            batch_images.append(img)
            batch_readings.append(reading)
        return to_model_input(np.stack(batch_images)), tuple(to_one_hot(batch_readings))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_model(he_init: bool, l2_reg: float | None):
    init = "he_normal" if he_init else "glorot_uniform"
    reg = keras.regularizers.l2(l2_reg) if l2_reg else None
    counter_w, counter_h = CONFIG["image_sizes"]["counter_crop"]

    inputs = keras.Input(shape=(counter_h, counter_w, input_channels()), name="counter_crop")

    x = keras.layers.Conv2D(128, 5, padding="same", kernel_initializer=init)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(128, 3, padding="same", kernel_initializer=init)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(192, 3, padding="same", kernel_initializer=init)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Flatten()(x)
    # BN after the large Dense layer stops the ReLUs dying early in training.
    x = keras.layers.Dense(512, kernel_initializer=init, kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(CONFIG["cnn"]["dropout"])(x)

    # Heads stay float32 under mixed precision.
    outputs = [
        keras.layers.Dense(
            NUM_CLASSES, activation="softmax",
            kernel_initializer=init, kernel_regularizer=reg,
            name=f"digit_{i + 1}", dtype="float32",
        )(x)
        for i in range(NUM_DIGITS)
    ]

    return keras.Model(inputs=inputs, outputs=outputs, name="multitask_cnn")


def make_step_decay_schedule(initial_lr, drop_factor, epochs_drop, min_lr=0.0):
    def schedule(epoch, lr):
        return max(min_lr, initial_lr * (drop_factor ** np.floor((1 + epoch) / epochs_drop)))
    return schedule


def build_callbacks(use_step_decay: bool, cfg: dict, patience: int):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    # From step 4, a fixed step-decay schedule replaces ReduceLROnPlateau.
    if use_step_decay:
        decay_cfg = cfg["lr_step_decay"]
        # The decay period is not stretched with longer runs; min_lr sets a floor.
        schedule = make_step_decay_schedule(
            cfg["learning_rate"], decay_cfg["drop_factor"], decay_cfg["epochs_drop"], decay_cfg.get("min_lr", 0.0))
        callbacks.append(keras.callbacks.LearningRateScheduler(schedule, verbose=0))
    else:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=cfg["reduce_lr_patience"], factor=0.5, min_lr=1e-6,
            )
        )
    return callbacks


def evaluate_full_sequence(model, images_float, readings):
    probs = model.predict(images_float, verbose=0)  # list of 5 arrays, each (N, 10)
    pred_digits = np.stack([p.argmax(axis=1) for p in probs], axis=1)  # (N, 5)
    pred_readings = ["".join(str(d) for d in row) for row in pred_digits]
    true_digits = np.array([[int(c) for c in r] for r in readings])
    per_digit_acc = [float((pred_digits[:, d] == true_digits[:, d]).mean()) for d in range(NUM_DIGITS)]
    full_seq_acc = float(np.mean([p == r for p, r in zip(pred_readings, readings)]))
    return full_seq_acc, per_digit_acc, pred_readings, probs


def step_epochs(step_cfg):
    """Epoch budget for one ablation step (longer for augmented steps)."""
    cfg = CONFIG["cnn"]
    return cfg["epochs_augmented"] if step_cfg["augment"] else cfg["epochs"]


def step_l2(step_cfg):
    return CONFIG["cnn"]["l2_reg"] if step_cfg["l2_reg"] else None


def to_float32_model(model, step_cfg):
    """Rebuild the model as plain float32 so it runs on any CPU."""
    keras.mixed_precision.set_global_policy("float32")
    fp32 = build_model(he_init=step_cfg["he_init"], l2_reg=step_l2(step_cfg))
    fp32.set_weights(model.get_weights())
    return fp32


def run_ablation_step(step_cfg, train_data, val_images_f, val_labels, epochs, batch_size):
    cfg = CONFIG["cnn"]
    train_images, train_readings, train_margin, train_geoms = train_data
    # Mixed precision roughly halves GPU time per epoch.
    keras.mixed_precision.set_global_policy("mixed_float16" if cfg["mixed_precision"] else "float32")
    model = build_model(he_init=step_cfg["he_init"], l2_reg=step_l2(step_cfg))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=["categorical_crossentropy"] * NUM_DIGITS,
        metrics=[["accuracy"]] * NUM_DIGITS,
    )
    patience = cfg["early_stopping_patience_augmented"] if step_cfg["augment"] else cfg["early_stopping_patience"]
    callbacks = build_callbacks(step_cfg["step_decay"], cfg, patience)

    if step_cfg["augment"]:
        synthesizer = None
        if step_cfg["permute"]:
            synthesizer = make_synthesizer(
                train_margin, train_geoms, train_readings,
                CONFIG["augmentation"]["digit_permutation_cross_counter_prob"])
        train_seq = CounterSequence(
            train_margin, train_geoms, train_readings, batch_size,
            synthesizer=synthesizer, synth_prob=CONFIG["augmentation"]["digit_permutation_prob"],
        )
        validation_steps = int(np.ceil(len(val_images_f) / batch_size))
        history = model.fit(
            train_seq, validation_data=(val_images_f, val_labels),
            epochs=epochs, callbacks=callbacks, verbose=2,
            validation_steps=validation_steps,
        )
    else:
        x_train = train_images.astype(np.float32) / 255.0
        y_train = to_one_hot(train_readings)
        # Explicit step counts: Keras 3 can under-count them for multi-output targets.
        steps_per_epoch = int(np.ceil(len(x_train) / batch_size))
        validation_steps = int(np.ceil(len(val_images_f) / batch_size))
        history = model.fit(
            x_train, y_train, validation_data=(val_images_f, val_labels),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2,
            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        )

    return model, history


def run_finetune_step(model, train_data, val_images_f, val_labels, val_readings, epochs, batch_size):
    """Fine-tune at a low learning rate on mostly real counters; kept only if val improves."""
    ft_cfg = CONFIG["cnn"]["finetune"]
    _, train_readings, train_margin, train_geoms = train_data

    before_full, before_per_digit, _, _ = evaluate_full_sequence(model, val_images_f, val_readings)
    before_weights = [w.copy() for w in model.get_weights()]
    _log(f"fine-tune start: val full-sequence accuracy = {before_full:.4f}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ft_cfg["learning_rate"]),
        loss=["categorical_crossentropy"] * NUM_DIGITS,
        metrics=[["accuracy"]] * NUM_DIGITS,
    )
    synthesizer = make_synthesizer(train_margin, train_geoms, train_readings,
                                   ft_cfg["digit_permutation_cross_counter_prob"])
    train_seq = CounterSequence(
        train_margin, train_geoms, train_readings, batch_size,
        synthesizer=synthesizer, synth_prob=ft_cfg["digit_permutation_prob"],
    )
    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=ft_cfg["early_stopping_patience"], restore_best_weights=True)]
    history = model.fit(
        train_seq, validation_data=(val_images_f, val_labels),
        epochs=epochs, callbacks=callbacks, verbose=2,
        validation_steps=int(np.ceil(len(val_images_f) / batch_size)),
    )

    after_full, after_per_digit, _, _ = evaluate_full_sequence(model, val_images_f, val_readings)
    kept = after_full >= before_full
    if not kept:
        model.set_weights(before_weights)
    _log(f"fine-tune end: val full-sequence accuracy {before_full:.4f} -> {after_full:.4f}; "
         f"{'kept fine-tuned weights' if kept else 'reverted to pre-finetune weights'}")
    info = {
        "before_val_full_sequence_accuracy": before_full,
        "before_val_per_digit_accuracy": before_per_digit,
        "after_val_full_sequence_accuracy": after_full,
        "after_val_per_digit_accuracy": after_per_digit,
        "kept": bool(kept),
    }
    return model, history, info


def load_saved_model_for_finetune(step_cfg):
    """Resume the ablation at the fine-tune step from the saved deployed model."""
    weights_path = resolve_path(CONFIG["paths"]["cnn_weights"])
    if not weights_path.exists():
        raise SystemExit(f"--start-from +real_finetune needs the synthesis-trained model at {weights_path}")
    saved = keras.models.load_model(str(weights_path), compile=False)
    cfg = CONFIG["cnn"]
    keras.mixed_precision.set_global_policy("mixed_float16" if cfg["mixed_precision"] else "float32")
    model = build_model(he_init=step_cfg["he_init"], l2_reg=step_l2(step_cfg))
    model.set_weights(saved.get_weights())
    _log(f"loaded {weights_path} as the starting point for fine-tuning")
    return model


def save_probs(path_key, basenames, readings, probs):
    path = resolve_path(CONFIG["paths"][path_key])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"basenames": np.array(basenames), "readings": np.array(readings)}
    for i in range(NUM_DIGITS):
        data[f"digit_{i + 1}_probs"] = probs[i]
    np.savez(path, **data)
    _log(f"wrote {path}")


def write_model_summary():
    """Write the deployed model's architecture summary to a text file."""
    weights_path = resolve_path(CONFIG["paths"]["cnn_weights"])
    if not weights_path.exists():
        raise SystemExit(f"{weights_path} not found -- train the model first")
    model = keras.models.load_model(str(weights_path), compile=False)
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    summary_path = resolve_path(CONFIG["paths"]["cnn_model_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    _log(f"wrote {summary_path}")


def write_classification_report():
    """Per-class precision/recall/F1 pooled over all 5 positions, from saved probabilities."""
    from sklearn.metrics import classification_report as sk_classification_report

    report = {}
    for split, path_key in [("validation", "cnn_val_probs"), ("testing", "cnn_test_probs")]:
        probs_path = resolve_path(CONFIG["paths"][path_key])
        if not probs_path.exists():
            raise SystemExit(f"{probs_path} not found -- train the model first")
        z = np.load(probs_path)
        true_digits = np.array([[int(c) for c in r] for r in z["readings"]])
        pred_digits = np.stack([z[f"digit_{i + 1}_probs"].argmax(axis=1) for i in range(NUM_DIGITS)], axis=1)
        report[split] = sk_classification_report(
            true_digits.ravel(), pred_digits.ravel(), labels=list(range(10)),
            target_names=[str(d) for d in range(10)], output_dict=True, zero_division=0,
        )
    report_path = resolve_path(CONFIG["paths"]["cnn_classification_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    _log(f"wrote {report_path}")


def train_cnn(start_from: str = None):
    cfg = CONFIG["cnn"]
    set_seeds(CONFIG["seed"])

    step_names = [s["name"] for s in ABLATION_STEPS]
    if start_from is not None and start_from not in step_names:
        raise SystemExit(f"--start-from must be one of {step_names}, got {start_from!r}")

    gpus = tf.config.list_physical_devices("GPU")
    _log(f"training on device: {'GPU' if gpus else 'CPU'} ({len(gpus)} GPU(s) found)")

    _log("loading recognition data ...")
    train_images, train_readings, _ = load_split("training")
    train_margin, train_geoms = load_margin_data("training", train_readings)
    val_images, val_readings, val_basenames = load_split("validation")
    test_images, test_readings, test_basenames = load_split("testing")
    _log(f"loaded {len(train_readings)} training / {len(val_readings)} validation / "
         f"{len(test_readings)} testing counter crops")

    # Evaluate on crops with the eval margin (as the pipeline does); tight crops are logged too.
    eval_margin = float(cfg.get("eval_crop_margin", 0.0))
    _log(f"evaluation crops: ground-truth box +{eval_margin:.0%} margin (tight-crop metrics logged alongside)")
    val_images_f = to_model_input(load_eval_crops("validation", eval_margin))
    test_images_f = to_model_input(load_eval_crops("testing", eval_margin))
    val_tight_f = to_model_input(val_images)
    test_tight_f = to_model_input(test_images)
    val_labels = to_one_hot(val_readings)
    _log(f"model input: {input_channels()} channel(s)")

    ablation_path = resolve_path(CONFIG["paths"]["cnn_ablation"])
    ablation_results = []
    final_model, final_history = None, None

    steps_to_run = ABLATION_STEPS
    if start_from is not None:
        # Resume partway: reuse earlier steps' records (seeded) and retrain from start_from.
        skipped = step_names[:step_names.index(start_from)]
        if ablation_path.exists():
            previous = {r["step"]: r for r in json.load(open(ablation_path))}
            ablation_results = [previous[n] for n in skipped if n in previous]
        missing = [n for n in skipped if n not in {r["step"] for r in ablation_results}]
        if missing:
            _log(f"WARNING: no previous results for skipped steps {missing}; the ablation table will be incomplete")
        steps_to_run = ABLATION_STEPS[step_names.index(start_from):]
        _log(f"resuming ablation from {start_from!r}; keeping {len(ablation_results)} earlier step(s) from {ablation_path}")

    train_data = (train_images, train_readings, train_margin, train_geoms)
    prev_model, finetune_info = None, None
    for step in steps_to_run:
        if step.get("finetune"):
            epochs = cfg["finetune"]["epochs"]
            _log(f"training ablation step: {step['name']} ({epochs} epochs, continuing from the previous model) ...")
            start_model = prev_model if prev_model is not None else load_saved_model_for_finetune(step)
            model, history, finetune_info = run_finetune_step(
                start_model, train_data, val_images_f, val_labels, val_readings, epochs, cfg["batch_size"])
        else:
            epochs = step_epochs(step)
            _log(f"training ablation step: {step['name']} ({epochs} epochs) ...")
            model, history = run_ablation_step(step, train_data, val_images_f, val_labels, epochs, cfg["batch_size"])
        full_seq_acc, per_digit_acc, _, _ = evaluate_full_sequence(model, val_images_f, val_readings)
        tight_full_seq_acc, tight_per_digit_acc, _, _ = evaluate_full_sequence(model, val_tight_f, val_readings)
        record = {
            "step": step["name"],
            "val_full_sequence_accuracy": full_seq_acc,
            "val_per_digit_accuracy": per_digit_acc,
            "val_full_sequence_accuracy_tight_crop": tight_full_seq_acc,
            "val_per_digit_accuracy_tight_crop": tight_per_digit_acc,
            "eval_crop_margin": eval_margin,
            "epochs_trained": len(history.epoch),
        }
        if finetune_info is not None and step.get("finetune"):
            record["finetune"] = finetune_info
        ablation_results.append(record)
        _log(f"{step['name']}: val full-sequence accuracy = {full_seq_acc:.4f} "
             f"(trained {record['epochs_trained']} epochs)")

        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ablation_path, "w") as f:
            json.dump(ablation_results, f, indent=2)

        # Last step is the deployed model, stored as a plain float32 model.
        final_model, final_history = to_float32_model(model, step), history

        # Keep this model if the fine-tune step follows; otherwise free GPU memory.
        next_idx = step_names.index(step["name"]) + 1
        next_is_finetune = next_idx < len(ABLATION_STEPS) and ABLATION_STEPS[next_idx].get("finetune", False)
        if next_is_finetune:
            prev_model = model
        else:
            prev_model = None
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    weights_path = resolve_path(CONFIG["paths"]["cnn_weights"])
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(str(weights_path))
    _log(f"saved deployed model to {weights_path}")

    history_path = resolve_path(CONFIG["paths"]["cnn_history"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in final_history.history.items()}, f, indent=2)
    _log(f"wrote {history_path}")

    _log("evaluating deployed model on validation + testing splits ...")
    val_full_acc, val_per_digit, val_pred_readings, val_probs = evaluate_full_sequence(
        final_model, val_images_f, val_readings)
    test_full_acc, test_per_digit, test_pred_readings, test_probs = evaluate_full_sequence(
        final_model, test_images_f, test_readings)
    val_tight_full, val_tight_per_digit, _, _ = evaluate_full_sequence(final_model, val_tight_f, val_readings)
    test_tight_full, test_tight_per_digit, _, _ = evaluate_full_sequence(final_model, test_tight_f, test_readings)

    # Saved probabilities use the margin crops, matching the pipeline.
    save_probs("cnn_val_probs", val_basenames, val_readings, val_probs)
    save_probs("cnn_test_probs", test_basenames, test_readings, test_probs)

    metrics = {
        "eval_crop_margin": eval_margin,
        "input_channels": input_channels(),
        "validation": {"full_sequence_accuracy": val_full_acc, "per_digit_accuracy": val_per_digit},
        "testing": {"full_sequence_accuracy": test_full_acc, "per_digit_accuracy": test_per_digit},
        "validation_tight_crop": {"full_sequence_accuracy": val_tight_full, "per_digit_accuracy": val_tight_per_digit},
        "testing_tight_crop": {"full_sequence_accuracy": test_tight_full, "per_digit_accuracy": test_tight_per_digit},
    }
    if finetune_info is not None:
        metrics["finetune"] = finetune_info
    metrics_path = resolve_path(CONFIG["paths"]["cnn_metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    _log(f"wrote {metrics_path}")
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-from", type=str, default=None, metavar="STEP",
        help="skip the ablation steps before STEP (e.g. '+augmentation'), reusing their records from an "
             "existing results/cnn/cnn_ablation.json; useful when only the long augmented steps need rerunning",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="skip training entirely; regenerate the model summary and per-digit classification "
             "report from the already-saved model weights and prediction probabilities",
    )
    args = parser.parse_args()
    if args.report_only:
        write_model_summary()
        write_classification_report()
    else:
        train_cnn(start_from=args.start_from)

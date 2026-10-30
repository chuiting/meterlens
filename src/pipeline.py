"""End-to-end inference: photo -> counter box -> sensing -> recognition -> fusion -> decision.

  check()       detect the counter and run the sensing gate
  recognize()   run the members, fuse and decide
  read()        check() then recognize()
"""

import time

import cv2
import numpy as np

from config import CONFIG, resolve_path
from augment import crop_counter_with_margin
from decision import evaluate_reading
from fusion import fuse, read_probs
from sensing import SENSING_REASON, evaluate_sensing

NUM_DIGITS = 5
NUM_CLASSES = 10


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_keras_h5(path):
    """Load a Keras .h5, dropping `quantization_config` if it was saved by a newer Keras."""
    from tensorflow import keras
    try:
        return keras.models.load_model(str(path), compile=False)
    except TypeError as e:
        if "quantization_config" not in str(e):
            raise
    import h5py, json
    with h5py.File(path, "r") as f:
        config = json.loads(f.attrs["model_config"])
    for layer in config["config"]["layers"]:
        layer["config"].pop("quantization_config", None)
    model = keras.Model.from_config(config["config"])
    model.load_weights(str(path))
    return model


class MeterReader:
    def __init__(self, members=None):
        fusion_cfg = CONFIG["fusion"]
        selected = list(fusion_cfg["selected_members"]) if members is None else list(members)
        if not selected:
            raise ValueError("fusion.selected_members is empty -- run src/fusion.py first")
        by_name = {m["name"]: m for m in fusion_cfg["members"]}
        self.members = [by_name[n] for n in selected]
        self.weights = {n: float(fusion_cfg["weights"][n]) for n in selected} if members is None \
            else {n: 1.0 for n in selected}
        self.threshold = CONFIG["decision"]["confidence_threshold"]
        self.counter_size = tuple(CONFIG["image_sizes"]["counter_crop"])   # (w, h)
        self.digit_size = tuple(CONFIG["image_sizes"]["digit_crop"])
        self.detector_conf = CONFIG["pipeline"]["detector_conf"]
        self._detector = None
        self._models = {}
        self._tf_pinned = False


    @property
    def detector(self):
        if self._detector is None:
            from ultralytics import YOLO
            self._detector = YOLO(str(resolve_path(CONFIG["paths"]["yolo_weights"])))
        return self._detector

    def _model(self, member):
        name = member["name"]
        if name not in self._models:
            path = resolve_path(member["weights_file"])
            _log(f"loading {name} from {path}")
            if member["input"]["kind"] == "counter":
                import tensorflow as tf
                if CONFIG["pipeline"].get("tensorflow_cpu_only", True) and not self._tf_pinned:
                    # CPU only: the Apple Silicon GPU plugin gave unstable outputs.
                    try:
                        tf.config.set_visible_devices([], "GPU")
                    except RuntimeError:
                        pass  # devices already initialised
                    self._tf_pinned = True
                self._models[name] = load_keras_h5(path)
            else:
                import joblib
                scaler = joblib.load(resolve_path(CONFIG["paths"]["hog_scaler"]))
                self._models[name] = (joblib.load(path), scaler)
        return self._models[name]


    def detect(self, image_bgr):
        """Best counter box (x, y, w, h) and its confidence, or (None, None)."""
        result = self.detector.predict(source=image_bgr, conf=self.detector_conf, verbose=False)[0]
        boxes = result.boxes
        if len(boxes) == 0:
            return None, None
        best = int(np.argmax(boxes.conf.cpu().numpy()))
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        return (x1, y1, x2 - x1, y2 - y1), float(boxes.conf[best])

    def member_input(self, image_bgr, bbox, member):
        """Build the input one member expects from the photo and counter box."""
        spec = member["input"]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crop = crop_counter_with_margin(rgb, bbox, self.counter_size, float(spec.get("crop_margin", 0.0)))
        if spec["kind"] == "counter":
            if spec.get("grayscale", False):
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)[..., None]
            return crop.astype(np.float32) / 255.0
        if spec["kind"] == "strips":
            from train_svm import equal_width_strips, extract_hog
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            return np.array([extract_hog(s, CONFIG["svm"]) for s in equal_width_strips(gray, self.digit_size)])
        raise ValueError(spec["kind"])

    def member_probs(self, member, inputs):
        """inputs: list of member_input() results for N photos -> (N, 5, 10)."""
        model = self._model(member)
        if member["input"]["kind"] == "counter":
            heads = model.predict(np.stack(inputs), batch_size=CONFIG["pipeline"]["cnn_batch_size"], verbose=0)
            return np.stack(heads, axis=1).astype(np.float64)
        clf, scaler = model
        feats = scaler.transform(np.concatenate(inputs, axis=0))          # (N*5, 324)
        proba = clf.predict_proba(feats)                                   # (N*5, n_classes)
        probs = np.zeros((len(inputs), NUM_DIGITS, NUM_CLASSES))
        classes = [int(c) for c in clf.classes_]
        proba = proba.reshape(len(inputs), NUM_DIGITS, -1)
        for j, c in enumerate(classes):
            probs[:, :, c] = proba[:, :, j]
        return probs

    def fuse(self, probs_by_member):
        return fuse(probs_by_member, self.weights)

    def decide(self, fused, sensing_flags=None):
        """fused (N, 5, 10) -> result dicts. sensing_flags defaults to all False."""
        readings, per_digit_conf, conf = read_probs(fused)
        if sensing_flags is None:
            sensing_flags = [False] * len(readings)
        out = []
        for r, pdc, c, sflag in zip(readings, per_digit_conf, conf, sensing_flags):
            decision, reasons = evaluate_reading(r, float(c), sensing_flag=bool(sflag))
            out.append({"reading": r, "per_digit_confidence": [float(x) for x in pdc],
                        "confidence": float(c), "decision": decision, "reasons": reasons})
        return out


    def sensing_crop(self, image_bgr, bbox):
        """Tight grayscale counter crop, as the sensing gate was fitted on."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crop = crop_counter_with_margin(rgb, bbox, self.counter_size, 0.0)
        return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    def sense(self, image_bgr, bbox):
        """(score, flag) for one photo; (None, False) if the sensing models are missing."""
        try:
            score, flag = evaluate_sensing(self.sensing_crop(image_bgr, bbox))
            return score, flag
        except (FileNotFoundError, ValueError):
            return None, False


    def _boxes(self, bbox, w, h):
        """Counter and five digit boxes as fractions of the image size."""
        x, y, bw, bh = bbox
        strip_w = bw / NUM_DIGITS
        return ({"x": x / w, "y": y / h, "w": bw / w, "h": bh / h},
                [{"x": (x + i * strip_w) / w, "y": y / h, "w": strip_w / w, "h": bh / h} for i in range(NUM_DIGITS)])


    def check(self, image_bgr):
        """Fast phase: detect the counter and run the sensing gate (no recognition)."""
        h, w = image_bgr.shape[:2]
        bbox, det_conf = self.detect(image_bgr)
        if bbox is None:
            return {"detected": False, "sensing_score": None, "sensing_flag": False, "reasons": [],
                    "counter_bbox": None, "digit_bboxes": None}
        sensing_score, sensing_flag = self.sense(image_bgr, bbox)
        counter_bbox, digit_bboxes = self._boxes(bbox, w, h)
        reasons = [{"kind": "sensing", "message": SENSING_REASON}] if sensing_flag else []
        return {"detected": True, "detector_confidence": det_conf, "sensing_score": sensing_score,
                "sensing_flag": sensing_flag, "reasons": reasons,
                "counter_bbox": counter_bbox, "digit_bboxes": digit_bboxes}

    def recognize(self, image_bgr, sensing_flag: bool = False):
        """Slow phase: run every member, fuse and decide, using check()'s sensing flag."""
        h, w = image_bgr.shape[:2]
        bbox, det_conf = self.detect(image_bgr)
        if bbox is None:
            return {"detected": False, "reading": None, "confidence": 0.0,
                    "decision": "FLAGGED",
                    "reasons": [{"kind": "detection", "message": "No counter detected -- manual review required"}],
                    "sensing_flag": False,
                    "counter_bbox": None, "digit_bboxes": None, "members": {}}
        probs = {m["name"]: self.member_probs(m, [self.member_input(image_bgr, bbox, m)]) for m in self.members}
        fused = self.fuse(probs)
        result = self.decide(fused, sensing_flags=[sensing_flag])[0]
        counter_bbox, digit_bboxes = self._boxes(bbox, w, h)
        result.update({
            "detected": True,
            "detector_confidence": det_conf,
            "sensing_flag": sensing_flag,
            "counter_bbox": counter_bbox,
            "digit_bboxes": digit_bboxes,
            "members": {n: read_probs(p)[0][0] for n, p in probs.items()},
        })
        return result

    def read(self, image_bgr):
        """check() then recognize(), reading the photo even if it is sensing-flagged."""
        c = self.check(image_bgr)
        if not c["detected"]:
            return {"detected": False, "reading": None, "confidence": 0.0,
                    "decision": "FLAGGED",
                    "reasons": [{"kind": "detection", "message": "No counter detected -- manual review required"}],
                    "sensing_score": None, "sensing_flag": False,
                    "counter_bbox": None, "digit_bboxes": None, "members": {}}
        result = self.recognize(image_bgr, sensing_flag=c["sensing_flag"])
        result["sensing_score"] = c["sensing_score"]
        return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Read one meter photo end to end.")
    parser.add_argument("image_path")
    args = parser.parse_args()
    img = cv2.imread(args.image_path)
    if img is None:
        raise SystemExit(f"could not read {args.image_path}")
    print(json.dumps(MeterReader().read(img), indent=2))

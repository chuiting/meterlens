"""Run one photo through the pipeline stage by stage for the Inspector."""

import base64
import time

import cv2
import numpy as np
from skimage.feature import hog

from augment import crop_counter_with_margin
from config import CONFIG
from fusion import read_probs
from pipeline import MeterReader
from train_svm import equal_width_strips


def png_b64(img):
    """Image -> PNG data URI."""
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def hog_image(gray_digit, cfg):
    _, vis = hog(gray_digit, orientations=cfg["hog_orientations"],
                 pixels_per_cell=tuple(cfg["hog_pixels_per_cell"]),
                 cells_per_block=tuple(cfg["hog_cells_per_block"]), visualize=True)
    vis = vis / vis.max() if vis.max() > 0 else vis
    return (vis * 255).astype(np.uint8)


def member_summary(probs):
    """(1, 5, 10) -> reading, per-digit confidence, full softmax per digit."""
    readings, per_digit, conf = read_probs(probs)
    return {"reading": readings[0], "per_digit_confidence": [float(c) for c in per_digit[0]],
            "confidence": float(conf[0]), "probs": np.round(probs[0], 4).tolist()}


def trace(reader: MeterReader, image_bgr) -> dict:
    timings = {}

    def timed(name, fn):
        t = time.perf_counter()
        out = fn()
        timings[name] = round((time.perf_counter() - t) * 1000, 1)
        return out

    h, w = image_bgr.shape[:2]
    bbox, det_conf = timed("detect", lambda: reader.detect(image_bgr))
    if bbox is None:
        return {"detected": False, "timings_ms": timings}

    gray = timed("crop", lambda: reader.sensing_crop(image_bgr, bbox))
    rgb = crop_counter_with_margin(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), bbox, reader.counter_size, 0.0)
    score, flag = timed("sensing", lambda: reader.sense(image_bgr, bbox))

    strips = timed("segment", lambda: equal_width_strips(gray, reader.digit_size))
    hog_cfg = CONFIG["svm"]
    hogs = [hog_image(s, hog_cfg) for s in strips]

    probs = {}
    for m in reader.members:
        probs[m["name"]] = timed(m["name"], lambda m=m: reader.member_probs(m, [reader.member_input(image_bgr, bbox, m)]))
    fused = timed("fusion", lambda: reader.fuse(probs))
    result = timed("decision", lambda: reader.decide(fused, sensing_flags=[flag])[0])

    counter_bbox, digit_bboxes = reader._boxes(bbox, w, h)
    return {
        "detected": True,
        "image_size": {"w": w, "h": h},
        "detection": {"bbox": counter_bbox, "digit_bboxes": digit_bboxes, "confidence": float(det_conf)},
        "crop": {"rgb": png_b64(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), "gray": png_b64(gray), "size": list(reader.counter_size)},
        "sensing": {"distance": score, "threshold": CONFIG["sensing"]["sensing_distance_threshold"], "flag": flag},
        "segmentation": {"strips": [png_b64(s) for s in strips], "hog": [png_b64(v) for v in hogs]},
        "members": {name: member_summary(p) for name, p in probs.items()},
        "fusion": {"weights": reader.weights, "probs": np.round(fused[0], 4).tolist(),
                   "reading": result["reading"], "per_digit_confidence": result["per_digit_confidence"],
                   "confidence": result["confidence"]},
        "decision": {"status": result["decision"], "reasons": result["reasons"], "threshold": reader.threshold},
        "timings_ms": {**timings, "total": round(sum(timings.values()), 1)},
    }


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Trace one meter photo through every pipeline stage.")
    parser.add_argument("image_path")
    args = parser.parse_args()
    img = cv2.imread(args.image_path)
    if img is None:
        raise SystemExit(f"could not read {args.image_path}")
    out = trace(MeterReader(), img)
    for k in ("crop", "segmentation"):          # keep the CLI output readable
        out.pop(k, None)
    print(json.dumps(out, indent=2))

"""Phase 1: training-time augmentation (training split only): per-image transforms
and digit-permutation synthesis (Laroca et al., 2019).
"""

import random

import cv2
import numpy as np

from config import CONFIG

_AUG_CFG = CONFIG["augmentation"]


def adjust_brightness(img, factor_range=None):
    lo, hi = factor_range or _AUG_CFG["brightness_range"]
    factor = random.uniform(lo, hi)
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate(img, max_deg=None):
    max_deg = max_deg if max_deg is not None else _AUG_CFG["rotation_deg"]
    angle = random.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def gaussian_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def gaussian_noise(img, std=10):
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_counter_with_jitter(margin_img, counter_bbox, out_size, jitter_range=None):
    """Crop the counter with each edge moved by a random fraction, then resize.

    Mimics an imprecise detector box (-2% to +8%, as in the UFPR-AMR paper).
    """
    lo, hi = jitter_range or _AUG_CFG["crop_jitter_range"]
    x, y, w, h = counter_bbox
    img_h, img_w = margin_img.shape[:2]
    x0 = int(round(max(0, x - random.uniform(lo, hi) * w)))
    x1 = int(round(min(img_w, x + w + random.uniform(lo, hi) * w)))
    y0 = int(round(max(0, y - random.uniform(lo, hi) * h)))
    y1 = int(round(min(img_h, y + h + random.uniform(lo, hi) * h)))
    return cv2.resize(margin_img[y0:y1, x0:x1], tuple(out_size))


def crop_counter_with_margin(margin_img, counter_bbox, out_size, margin_frac):
    """Crop the counter expanded by margin_frac on every side, then resize (no randomness)."""
    x, y, w, h = counter_bbox
    img_h, img_w = margin_img.shape[:2]
    x0 = int(round(max(0, x - margin_frac * w)))
    x1 = int(round(min(img_w, x + w + margin_frac * w)))
    y0 = int(round(max(0, y - margin_frac * h)))
    y1 = int(round(min(img_h, y + h + margin_frac * h)))
    return cv2.resize(margin_img[y0:y1, x0:x1], tuple(out_size))


def _clamped_slice(img, bbox):
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]
    return max(0, y), min(img_h, y + h), max(0, x), min(img_w, x + w)


class DigitPermutationSynthesizer:
    """Build new training counters by refilling each digit slot with a random digit patch.

    - cross_counter_prob: chance of taking the patch from another counter (any class)
      instead of this counter's own digits.
    - transition_prob: chance (per slot) of a mid-roll digit, labelled with the
      outgoing digit as in the dataset.
    - Patches come from same-polarity counters, keep their aspect ratio, and are
      tone-matched and feathered so the seam is not visible.
    """

    def __init__(self, margin_images, geometries, readings, cross_counter_prob=0.0,
                 transition_prob=0.0, transition_split_range=(0.35, 0.8),
                 same_polarity=True, feather_px=3, contrast_gain_range=(0.5, 2.0)):
        assert len(margin_images) == len(geometries) == len(readings)
        self.images = margin_images
        self.geoms = geometries
        self.readings = readings
        self.cross_counter_prob = cross_counter_prob
        n_slots = len(readings[0])
        if np.isscalar(transition_prob):
            transition_prob = [float(transition_prob)] * n_slots
        assert len(transition_prob) == n_slots, "transition_prob needs one value per digit slot"
        self.transition_prob = [float(p) for p in transition_prob]
        self.transition_split_range = tuple(transition_split_range)
        self.feather_px = int(feather_px)
        self.contrast_gain_range = tuple(contrast_gain_range)
        self.polarity = [self._polarity(i) for i in range(len(readings))]
        # class -> [(image_idx, position), ...]
        self.bank = {c: [] for c in range(10)}
        self.bank_by_polarity = {pol: {c: [] for c in range(10)} for pol in (0, 1)}
        for i, reading in enumerate(readings):
            for pos, ch in enumerate(reading):
                self.bank[int(ch)].append((i, pos))
                self.bank_by_polarity[self.polarity[i]][int(ch)].append((i, pos))
        self.same_polarity = same_polarity
        self.min_polarity_bank = 5  # fall back to all donors below this

    def _polarity(self, img_idx):
        """1 = light digits on dark, 0 = dark on light (centre vs border brightness, majority vote)."""
        votes = []
        for pos in range(len(self.readings[img_idx])):
            patch = self._patch(img_idx, pos)
            if patch.size == 0:
                continue
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h, w = gray.shape
            if h < 4 or w < 4:
                continue
            centre = gray[int(h * 0.25):int(h * 0.75), int(w * 0.3):int(w * 0.7)]
            ring = np.concatenate([gray[:int(h * 0.1)].ravel(), gray[int(h * 0.9):].ravel(),
                                   gray[:, :int(w * 0.15)].ravel(), gray[:, int(w * 0.85):].ravel()])
            if centre.size and ring.size:
                votes.append(1 if centre.mean() > ring.mean() else 0)
        return 1 if not votes or np.mean(votes) >= 0.5 else 0

    def _patch(self, img_idx, pos):
        y0, y1, x0, x1 = _clamped_slice(self.images[img_idx], self.geoms[img_idx]["digit_bboxes"][pos])
        return self.images[img_idx][y0:y1, x0:x1]

    def _donor_bank(self, cls, base_idx):
        if self.same_polarity:
            bank = self.bank_by_polarity[self.polarity[base_idx]][cls]
            if len(bank) >= self.min_polarity_bank:
                return bank
        return self.bank[cls]

    def _patch_of_class(self, cls, prefer_idx):
        """A patch of class cls, preferably from counter prefer_idx."""
        same = [(i, p) for i, p in self.bank[cls] if i == prefer_idx]
        return self._patch(*random.choice(same or self._donor_bank(cls, prefer_idx)))

    def _fit_to_slot(self, patch, slot):
        """Scale the patch to the slot keeping its aspect ratio, and centre it."""
        slot_h, slot_w = slot.shape[:2]
        ph, pw = patch.shape[:2]
        scale = min(slot_h / ph, slot_w / pw)
        new_w, new_h = max(1, int(round(pw * scale))), max(1, int(round(ph * scale)))
        resized = cv2.resize(patch, (new_w, new_h))
        if new_w == slot_w and new_h == slot_h:
            return resized
        canvas = slot.copy()
        x0 = (slot_w - new_w) // 2
        y0 = (slot_h - new_h) // 2
        if x0 > 0:
            canvas[:, :x0] = np.repeat(slot[:, :1], x0, axis=1)
        if x0 + new_w < slot_w:
            canvas[:, x0 + new_w:] = np.repeat(slot[:, -1:], slot_w - x0 - new_w, axis=1)
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _match_tone(self, patch, slot):
        """Match the patch's brightness and contrast to the slot."""
        p = patch.astype(np.float32)
        p_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        s_gray = cv2.cvtColor(slot, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gain = s_gray.std() / max(p_gray.std(), 1.0)
        gain = float(np.clip(gain, *self.contrast_gain_range))
        out = (p - p_gray.mean()) * gain + s_gray.mean()
        return np.clip(out, 0, 255).astype(np.uint8)

    def _blend(self, patch, slot):
        """Blend the patch into the slot with a soft border."""
        f = self.feather_px
        h, w = slot.shape[:2]
        if f <= 0 or h <= 2 * f or w <= 2 * f:
            return patch
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[f:h - f, f:w - f] = 1.0
        alpha = cv2.GaussianBlur(alpha, (2 * f + 1, 2 * f + 1), 0)[..., None]
        out = patch.astype(np.float32) * alpha + slot.astype(np.float32) * (1 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _transition_composite(self, outgoing_patch, outgoing_cls, src_idx, slot_w, slot_h):
        """Mid-roll digit: lower part of the outgoing digit above the top of the next."""
        incoming = self._patch_of_class((outgoing_cls + 1) % 10, src_idx)
        if incoming.size == 0:
            return None
        split = random.uniform(*self.transition_split_range)
        top_h = max(1, min(slot_h - 1, int(round(split * slot_h))))
        top = cv2.resize(outgoing_patch, (slot_w, slot_h))[slot_h - top_h:]
        bottom = cv2.resize(incoming, (slot_w, slot_h))[:slot_h - top_h]
        composite = np.vstack([top, bottom])
        seam = slice(max(0, top_h - 1), min(slot_h, top_h + 1))
        composite[seam] = (composite[seam].astype(np.float32) * 0.5).astype(np.uint8)
        return composite

    def synthesize(self, base_idx):
        """Return (image, reading) for a new counter synthesised from counter base_idx."""
        base = self.images[base_idx].copy()
        base_reading = self.readings[base_idx]
        new_reading = []
        original = self.images[base_idx]
        for pos in range(len(base_reading)):
            if random.random() < self.cross_counter_prob:
                target = random.randrange(10)
                src_idx, src_pos = random.choice(self._donor_bank(target, base_idx))
            else:
                src_idx, src_pos = base_idx, random.randrange(len(base_reading))
                target = int(base_reading[src_pos])
            patch = self._patch(src_idx, src_pos)
            y0, y1, x0, x1 = _clamped_slice(base, self.geoms[base_idx]["digit_bboxes"][pos])
            if patch.size == 0 or y1 <= y0 or x1 <= x0:
                new_reading.append(base_reading[pos])  # unusable slot: keep the original
                continue
            slot = original[y0:y1, x0:x1]
            slot_h, slot_w = slot.shape[:2]
            if random.random() < self.transition_prob[pos]:
                composite = self._transition_composite(patch, target, src_idx, slot_w, slot_h)
                if composite is not None:
                    patch = composite
            patch = self._fit_to_slot(patch, slot)
            patch = self._match_tone(patch, slot)
            base[y0:y1, x0:x1] = self._blend(patch, slot)
            new_reading.append(str(target))
        return base, "".join(new_reading)


def augment_image(img):
    out = adjust_brightness(img)
    out = rotate(out)

    roll = random.random()
    if roll < _AUG_CFG["gaussian_blur_prob"]:
        out = gaussian_blur(out)
    elif roll < _AUG_CFG["gaussian_blur_prob"] + _AUG_CFG["gaussian_noise_prob"]:
        out = gaussian_noise(out)

    return out


if __name__ == "__main__":
    # Save a before/after example.
    import glob
    from config import resolve_path

    counters_dir = resolve_path(CONFIG["paths"]["recognition_dir"]) / "training" / "counters"
    sample_paths = sorted(glob.glob(str(counters_dir / "*.jpg")))
    if not sample_paths:
        print(f"No counter crops found in {counters_dir} - run prepare_recognition.py first.")
    else:
        img = cv2.imread(sample_paths[0])
        aug = augment_image(img)
        out_path = resolve_path(CONFIG["paths"]["augment_demo"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), np.hstack([img, aug]))
        print(f"wrote before/after demo to {out_path}")

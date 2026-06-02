"""
subcropper.py — Detect and split multi-image crops flagged during review (Phase 2.5).

For crops tagged "needs_split" in review_meta.json, this module:
  1. Computes a mean-intensity projection of the image along both axes.
  2. Finds valleys in the projection that are deep enough (below mean by
     min_seam_depth_px) and span >= min_seam_span of the perpendicular dimension.
     These are physical dividers between side-by-side or stacked sub-images.
  3. Cuts the image at the detected seams, producing N sub-images.
  4. Saves each sub-image as a new crop file with _s1, _s2, ... suffix.
  5. Updates review_meta.json: the original entry is marked not-approved (replaced),
     and each split sub-crop gets a new entry marked as approved.

Config keys used (from config.yaml → subcropper):
    min_seam_depth_px : how far below the mean a valley must be (default 30)
    min_seam_span     : fraction of perpendicular dimension a seam must cross (default 0.85)

Usage:
    from src.subcropper import process_splits
    results = process_splits(cfg)
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Low-level seam detection
# ---------------------------------------------------------------------------

def _mean_projection(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Mean pixel intensity along rows (axis=1, gives a profile over height)
    or columns (axis=0, gives a profile over width).
    """
    gray = arr.mean(axis=2)          # H×W×3 → H×W
    return gray.mean(axis=axis)      # collapse the chosen axis


def _find_seams(profile: np.ndarray, min_depth: float, min_span: float) -> list:
    """
    Locate valley positions in `profile` that qualify as genuine seams.

    A position qualifies when:
      - Its value is at least `min_depth` below the profile mean.
      - It is a local minimum (not just any low point in a plateau).
      - It falls within the central `min_span` fraction of the profile
        (guards against cutting at true image borders).

    Nearby candidates are merged: within a 5%-of-length window only the
    deepest valley survives.
    """
    n      = len(profile)
    mean   = profile.mean()
    margin = max(1, int(n * (1.0 - min_span) / 2.0))

    raw = []
    for i in range(margin, n - margin):
        if profile[i] < mean - min_depth:
            left  = profile[i - 1] if i > 0     else profile[i] + 1
            right = profile[i + 1] if i < n - 1 else profile[i] + 1
            if profile[i] <= left and profile[i] <= right:
                raw.append(i)

    if len(raw) < 2:
        return raw

    # Merge seams closer than 5% of the profile length
    window  = max(1, int(n * 0.05))
    merged  = [raw[0]]
    for s in raw[1:]:
        if s - merged[-1] < window:
            if profile[s] < profile[merged[-1]]:
                merged[-1] = s
        else:
            merged.append(s)
    return merged


# ---------------------------------------------------------------------------
# Public detection + splitting helpers
# ---------------------------------------------------------------------------

def detect_split_axis(img: Image.Image, cfg: dict) -> tuple:
    """
    Decide whether to split horizontally or vertically (or not at all).

    Returns:
        axis  : "horizontal" | "vertical" | "none"
        seams : list of pixel positions to cut at (empty if axis == "none")
    """
    min_depth = float(cfg.get("subcropper", {}).get("min_seam_depth_px", 30))
    min_span  = float(cfg.get("subcropper", {}).get("min_seam_span",      0.85))

    arr = np.asarray(img.convert("RGB")).astype(float)

    # Horizontal seams divide the image top-to-bottom (cuts along the height axis)
    h_profile = _mean_projection(arr, axis=1)   # profile over height → horizontal cuts
    v_profile = _mean_projection(arr, axis=0)   # profile over width  → vertical cuts

    h_seams = _find_seams(h_profile, min_depth, min_span)
    v_seams = _find_seams(v_profile, min_depth, min_span)

    if not h_seams and not v_seams:
        return "none", []
    # Prefer the axis with more seams; break ties by picking vertical (most common case)
    if len(v_seams) >= len(h_seams):
        return "vertical", v_seams
    return "horizontal", h_seams


def split_image(img: Image.Image, axis: str, seams: list) -> list:
    """
    Cut `img` at the given `seams` along `axis`.

    Returns a list of PIL Images (sub-crops), in order from top-to-bottom
    or left-to-right.
    """
    w, h  = img.size
    cuts  = [0] + seams + ([h] if axis == "horizontal" else [w])
    parts = []
    for i in range(len(cuts) - 1):
        if axis == "horizontal":
            parts.append(img.crop((0, cuts[i], w, cuts[i + 1])))
        else:
            parts.append(img.crop((cuts[i], 0, cuts[i + 1], h)))
    return parts


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def process_splits(cfg: dict) -> dict:
    """
    Process all crops flagged "needs_split" in review_meta.json.

    For each flagged crop:
      - Run detect_split_axis().
      - If seams found: save sub-crops as <stem>_s1.png, <stem>_s2.png, ...;
        mark the original entry as approved=False in the JSON (it is now
        replaced by its sub-crops); add new entries for each sub-crop
        with approved=True.
      - If no seams found: clear the needs_split flag and leave the crop as-is.

    Writes the updated review_meta.json in place.

    RETURNS:
        dict mapping original crop path (str) → list of new sub-crop Paths
    """
    meta_path = Path(cfg["paths"]["logs"]) / "review_meta.json"
    if not meta_path.exists():
        print("No review_meta.json found — run Phase 2 (Review) first.")
        return {}

    with open(meta_path) as f:
        meta = json.load(f)

    crops_dir = Path(cfg["paths"]["crops"])
    fmt       = cfg["extractor"].get("save_format", "PNG").upper()
    ext       = "jpg" if fmt == "JPEG" else "png"
    results   = {}

    for patent_id, pdata in meta.items():
        images_meta = pdata.get("images", {})
        new_entries = {}

        for fname, idata in list(images_meta.items()):
            if not idata.get("needs_split", False):
                continue

            crop_path = crops_dir / patent_id / fname
            if not crop_path.exists():
                print(f"  Missing crop: {crop_path} — skipping")
                continue

            img  = Image.open(crop_path).convert("RGB")
            axis, seams = detect_split_axis(img, cfg)

            if not seams:
                idata["needs_split"] = False
                print(f"  No seams found in {fname} — flag cleared")
                continue

            parts = split_image(img, axis, seams)
            stem  = crop_path.stem
            saved = []

            for i, part in enumerate(parts, start=1):
                sub_name = f"{stem}_s{i}.{ext}"
                sub_path = crop_path.parent / sub_name
                part.save(sub_path, fmt)
                saved.append(sub_path)
                new_entries[sub_name] = {
                    "approved":    True,
                    "is_main":     False,
                    "needs_split": False,
                    "split_from":  fname,
                }

            # Mark original as replaced
            idata["approved"]    = False
            idata["needs_split"] = False
            results[str(crop_path)] = saved
            print(f"  Split {fname} → {len(parts)} sub-crops ({axis})")

        images_meta.update(new_entries)
        pdata["last_updated"] = datetime.now().isoformat(timespec="seconds")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_splits = sum(len(v) for v in results.values())
    print(f"\nSub-crop splitting complete: {len(results)} crops split → {total_splits} new files.")
    return results

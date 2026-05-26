"""
processor.py — Resize and pad accepted figure crops to 224×224 for DINOv2 (Phase 3).

WHY 224×224?
DINOv2 (and most Vision Transformers) expect a fixed square input size.
224×224 is the standard size used during pre-training.

WHY ASPECT-RATIO PADDING INSTEAD OF STRETCHING?
Stretching distorts the figure. Instead:
  1. Apply a light sharpness enhancement BEFORE downscaling (preserves edge detail)
  2. Resize so the longest side = 224 px (aspect ratio preserved)
  3. Detect the dominant background color from the raw crop's corner patches
  4. Place the resized image on a square canvas of that color, centered

WHY DETECT BACKGROUND COLOR?
Raw YOLO crops sometimes have black backgrounds. Padding them with white creates
an unnatural "dual background" (black crop on white padding) that degrades training.
Matching the padding to the crop's own background avoids this artifact.

FILES READ:
    crops/{RecordNumber}/{RecordNumber}_p{page:03d}_c{crop:02d}.png

FILES WRITTEN:
    processed/{RecordNumber}/{RecordNumber}_p{page:03d}_c{crop:02d}.png
    Already-processed files are SKIPPED (safe to re-run).

Usage:
    from src.processor import process_crops
    processed_paths = process_crops(accepted_paths, cfg)
"""

import re
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from pathlib import Path
from tqdm import tqdm


def detect_background_color(img: Image.Image) -> tuple:
    """
    Detect the dominant background color by sampling four 5×5 corner patches.

    Ignores the outer 4 pixels to avoid YOLO bbox edge artifacts.
    Returns the mode (most frequent exact RGB triple) across all 100 sampled pixels.

    Edge cases:
      - Images <= 8px in either dimension → returns white (255, 255, 255)
      - Images too small for the inset+patch combo → falls back to full-image mode

    Ported from the working implementation in Image_Chose_&_Save_PAdding_Enhanced_224x224.ipynb.
    """
    img = img.convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]

    if h <= 8 or w <= 8:
        return (255, 255, 255)

    inset = 4
    patch = 5

    if h < inset * 2 + patch or w < inset * 2 + patch:
        pixels = arr.reshape(-1, 3)
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        return tuple(int(c) for c in colors[np.argmax(counts)])

    top_left     = arr[inset:inset+patch,     inset:inset+patch]
    top_right    = arr[inset:inset+patch,     w-inset-patch:w-inset]
    bottom_left  = arr[h-inset-patch:h-inset, inset:inset+patch]
    bottom_right = arr[h-inset-patch:h-inset, w-inset-patch:w-inset]

    patches = np.vstack([
        top_left.reshape(-1, 3),
        top_right.reshape(-1, 3),
        bottom_left.reshape(-1, 3),
        bottom_right.reshape(-1, 3),
    ])
    colors, counts = np.unique(patches, axis=0, return_counts=True)
    return tuple(int(c) for c in colors[np.argmax(counts)])


def resize_and_pad(img: Image.Image, target_size: int, pad_color: tuple) -> Image.Image:
    """
    Sharpen → resize to fit inside target_size×target_size → pad with pad_color.

    Sharpness is applied BEFORE downscaling so the original high-frequency edge
    detail is preserved through the resize (enhancing after would only affect the
    already-blurred result).

    PARAMETERS:
        img         : PIL Image in RGB mode
        target_size : output square size in pixels (224 for DINOv2)
        pad_color   : RGB tuple — use detect_background_color() to get this
    """
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.20)

    w, h = img.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas   = Image.new("RGB", (target_size, target_size), pad_color)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    return canvas


def process_crops(crop_paths: list, cfg: dict) -> list:
    """
    Apply detect_background_color + resize_and_pad to every accepted crop.

    Called by Phase 3 with the list of accepted paths from review_meta.json.
    Already-processed files are skipped (safe to re-run).

    PARAMETERS (from config.yaml → processor section):
        target_size : 224
        save_format : "PNG"

    RETURNS:
        list of Path objects pointing to the processed output files
    """
    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_size = cfg["processor"]["target_size"]
    fmt         = cfg["processor"]["save_format"].upper()
    ext         = "jpg" if fmt == "JPEG" else "png"

    output_paths = []

    for crop_path in tqdm(crop_paths, desc="Processing crops"):
        crop_path = Path(crop_path)

        record_id  = crop_path.name.split("_p")[0]
        patent_out = processed_dir / record_id
        patent_out.mkdir(parents=True, exist_ok=True)
        out_path   = patent_out / crop_path.with_suffix(f".{ext}").name

        if out_path.exists():
            output_paths.append(out_path)
            continue

        try:
            img       = Image.open(crop_path).convert("RGB")
            pad_color = detect_background_color(img)
            result    = resize_and_pad(img, target_size, pad_color)
            result.save(out_path, fmt)
            output_paths.append(out_path)
        except Exception as e:
            print(f"\n  Error processing {crop_path.name}: {e}")

    print(f"\nProcessing complete: {len(output_paths)} images saved to {processed_dir}")
    return output_paths


# ---------------------------------------------------------------------------
# Organize processed images into the final folder with CPC codes in filenames
# ---------------------------------------------------------------------------

def _build_cpc_map(cfg: dict) -> dict:
    """
    Read the PatSeer Excel and return {record_id: cpc_first_sanitized}.

    The CPC column contains codes separated by ' | ' (e.g. 'B64C29/0041 | B60L50/60').
    We take only the first code and strip '/' so it is filename-safe.
    Patents with no CPC entry get the placeholder 'NOCPC'.
    """
    excel_path = Path(cfg["paths"]["excel"])
    df = pd.read_excel(excel_path, dtype={"Record Number": str})

    cpc_map = {}
    for _, row in df.iterrows():
        record_id = str(row.get("Record Number", "")).strip()
        cpc_raw   = str(row.get("CPC", "") or "").strip()
        if not record_id:
            continue
        first_cpc = cpc_raw.split("|")[0].strip() if cpc_raw else ""
        # Remove all characters that are not alphanumeric → filename-safe
        cpc_clean = re.sub(r"[^A-Za-z0-9]", "", first_cpc) if first_cpc else "NOCPC"
        cpc_map[record_id] = cpc_clean

    return cpc_map


def organize_processed(cfg: dict) -> list:
    """
    Copy every processed 224×224 image into the final/ folder, injecting the
    patent's first CPC code into each filename.

    Input  (processed/):  {patent_id}/{patent_id}_p{page}_c{crop}.png
    Output (final/):      {patent_id}/{patent_id}_{CPC}_p{page}_c{crop}.png

    Example:
        processed/US2022267016A1/US2022267016A1_p001_c01.png
        →  final/US2022267016A1/US2022267016A1_B64C290041_p001_c01.png

    CPC codes are read from the PatSeer Excel (paths.excel → 'CPC' column).
    Patents missing from the Excel receive 'NOCPC' as placeholder.
    Already-present files in final/ are skipped (safe to re-run).

    RETURNS:
        list of Path objects written to final/
    """
    processed_dir = Path(cfg["paths"]["processed"])
    final_dir     = Path(cfg["paths"]["final"])
    final_dir.mkdir(parents=True, exist_ok=True)

    print("Building CPC map from PatSeer Excel …")
    cpc_map = _build_cpc_map(cfg)

    output_paths = []
    patent_folders = sorted(processed_dir.iterdir()) if processed_dir.exists() else []

    for patent_folder in tqdm(patent_folders, desc="Organizing final output"):
        if not patent_folder.is_dir():
            continue

        patent_id = patent_folder.name
        cpc       = cpc_map.get(patent_id, "NOCPC")
        out_dir   = final_dir / patent_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(patent_folder.glob("*.png")) + sorted(patent_folder.glob("*.jpg")):
            # Insert CPC right after the patent ID prefix:
            # "US2022267016A1_p001_c01.png" → "US2022267016A1_B64C290041_p001_c01.png"
            stem = img_path.stem  # e.g. US2022267016A1_p001_c01
            suffix = img_path.suffix
            # Split on the first "_p" to isolate the patent ID from the page/crop part
            parts = stem.split("_p", 1)
            if len(parts) == 2:
                new_name = f"{parts[0]}_{cpc}_p{parts[1]}{suffix}"
            else:
                new_name = f"{stem}_{cpc}{suffix}"

            dest = out_dir / new_name
            if dest.exists():
                output_paths.append(dest)
                continue

            shutil.copy2(img_path, dest)
            output_paths.append(dest)

    print(f"\nOrganize complete: {len(output_paths)} images in {final_dir}")
    return output_paths

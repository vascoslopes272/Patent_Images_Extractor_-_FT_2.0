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

_CAT_ABBR = {"shrouded": "SHR", "open_rotor": "OPN"}


def _build_cpc_map(cfg: dict) -> dict:
    """Return {record_id: cpc_first_clean} from the PatSeer Excel.
    Separator in the CPC column is ';' (not '|'). Removes all non-alphanumeric
    characters so the result is filename-safe (e.g. 'B64C29/0033' → 'B64C290033').
    """
    excel_path = Path(cfg["paths"]["excel"])
    df = pd.read_excel(excel_path, dtype={"Record Number": str})

    cpc_map = {}
    for _, row in df.iterrows():
        record_id = str(row.get("Record Number", "")).strip()
        cpc_raw   = str(row.get("CPC", "") or "").strip()
        if not record_id:
            continue
        first_cpc = re.split(r"\s*;\s*", cpc_raw)[0].strip() if cpc_raw else ""
        cpc_clean = re.sub(r"[^A-Za-z0-9]", "", first_cpc) if first_cpc else "NOCPC"
        cpc_map[record_id] = cpc_clean

    return cpc_map


def _build_category_map(cfg: dict) -> dict:
    """Return {record_id: category} from selected_patents.csv in the experiment folder."""
    csv_path = Path(cfg["paths"]["experiment"]) / "selected_patents.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, dtype={"Record Number": str})
    return {str(r).strip().upper(): str(c).strip()
            for r, c in zip(df["Record Number"], df["category"])}


def organize_processed(cfg: dict) -> list:
    """
    Copy every processed 224×224 image into class subfolders under final/, with
    category abbreviation and CPC code injected into each filename.

    Output structure:
        final/shrouded/   {id}_SHR_{CPC}_p{page:03d}_c{crop:02d}.png
        final/open_rotor/ {id}_OPN_{CPC}_p{page:03d}_c{crop:02d}.png

    Examples:
        US2022267016A1_SHR_B64C2720_p003_c01.png
        US2022267016A1_OPN_B64C290033_p003_c01.png

    Patents not in selected_patents.csv are placed in final/unlabelled/.
    Already-present files are skipped (safe to re-run).
    """
    processed_dir = Path(cfg["paths"]["processed"])
    final_dir     = Path(cfg["paths"]["final"])

    print("Building CPC map from PatSeer Excel …")
    cpc_map = _build_cpc_map(cfg)
    print("Loading categories from selected_patents.csv …")
    cat_map = _build_category_map(cfg)

    output_paths = []
    patent_folders = sorted(processed_dir.iterdir()) if processed_dir.exists() else []

    for patent_folder in tqdm(patent_folders, desc="Organising final output"):
        if not patent_folder.is_dir():
            continue

        patent_id = patent_folder.name
        cpc       = cpc_map.get(patent_id, "NOCPC")
        category  = cat_map.get(patent_id.upper(), "unlabelled")
        cat_abbr  = _CAT_ABBR.get(category, category.upper()[:3])

        out_dir = final_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(patent_folder.glob("*.png")) + sorted(patent_folder.glob("*.jpg")):
            # Input:  US2022267016A1_p003_c01.png
            # Output: US2022267016A1_SHR_B64C2720_p003_c01.png
            stem   = img_path.stem    # e.g. US2022267016A1_p003_c01
            suffix = img_path.suffix
            parts  = stem.split("_p", 1)
            if len(parts) == 2:
                new_name = f"{parts[0]}_{cat_abbr}_{cpc}_p{parts[1]}{suffix}"
            else:
                new_name = f"{stem}_{cat_abbr}_{cpc}{suffix}"

            dest = out_dir / new_name
            if dest.exists():
                output_paths.append(dest)
                continue

            shutil.copy2(img_path, dest)
            output_paths.append(dest)

    by_cat = {}
    for p in output_paths:
        by_cat.setdefault(p.parent.name, 0)
        by_cat[p.parent.name] += 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n} images")
    print(f"Organise complete: {len(output_paths)} images in {final_dir}")
    return output_paths

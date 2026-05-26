"""
processor.py — Resize and pad figure crops to 224×224 for DINOv2 (Phase 2).

WHY 224×224?
DINOv2 (and most Vision Transformers) expect a fixed square input size.
224×224 is the standard size used during pre-training.

WHY PADDING INSTEAD OF STRETCHING?
Stretching would distort the aspect ratio of the figure (a wide diagram would
become squashed into a square). Instead:
  1. Resize the image so its LONGEST side = 224 px (aspect ratio preserved)
  2. Place it on a white 224×224 canvas, centered
  3. The remaining space is filled with white padding

Example: a 400×200 px crop becomes 224×112 px, then centered on a
224×224 white canvas with 56 px of white padding on top and bottom.

FILES READ:
    /home/vasco/data/patents/crops/{RecordNumber}_p{page:03d}_c{crop:02d}.png

FILES WRITTEN:
    /home/vasco/data/patents/processed/{RecordNumber}_p{page:03d}_c{crop:02d}.png
    (same filename, different folder — the processed/ folder is the Phase 2 output)
    Already-processed files are SKIPPED (safe to re-run).

Usage:
    from src.processor import process_crops
    processed_paths = process_crops(crop_paths, cfg)
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm


def resize_and_pad(img: Image.Image, target_size: int, pad_color: tuple) -> Image.Image:
    """
    Resize one image to fit inside target_size×target_size while preserving
    aspect ratio, then pad the remaining space with pad_color.

    Steps:
      1. Find the scaling factor so the longer side equals target_size
      2. Resize both dimensions by that factor (LANCZOS = high-quality downscaling)
      3. Create a blank square canvas filled with pad_color (white = (255,255,255))
      4. Paste the resized image centered on the canvas

    PARAMETERS:
        img         : PIL Image in RGB mode
        target_size : output square size in pixels (224 for DINOv2)
        pad_color   : RGB tuple for padding (255,255,255) = white
    """
    w, h = img.size

    # Scale so the longest side = target_size, shorter side shrinks proportionally
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # LANCZOS gives the best quality when downscaling (avoids aliasing artifacts)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create the white square canvas
    canvas = Image.new("RGB", (target_size, target_size), pad_color)

    # Center the resized image on the canvas
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    return canvas


def process_crops(crop_paths: list, cfg: dict) -> list:
    """
    Apply resize_and_pad to every crop from Phase 1 and save the results.

    SKIP LOGIC:
    If the output file already exists in processed/, it is skipped.
    This means Phase 2 is safe to re-run without reprocessing everything.

    PARAMETERS (from config.yaml → processor section):
        target_size   : 224 (square output size)
        padding_color : [255, 255, 255] (white)
        save_format   : "PNG"

    FILES READ:
        /home/vasco/data/patents/crops/*.png

    FILES WRITTEN:
        /home/vasco/data/patents/processed/*.png
        (same filenames as the crops, same naming convention, different folder)

    RETURNS:
        list of Path objects pointing to the processed output files
    """
    # Output folder — created automatically if missing
    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Read settings from config
    target_size = cfg["processor"]["target_size"]           # 224
    pad_color   = tuple(cfg["processor"]["padding_color"])  # (255, 255, 255)
    fmt         = cfg["processor"]["save_format"].upper()   # "JPEG" or "PNG"
    ext         = "jpg" if fmt == "JPEG" else "png"

    output_paths = []

    for crop_path in tqdm(crop_paths, desc="Processing crops"):
        crop_path = Path(crop_path)

        # Mirror the per-patent subfolder: extract record_id from the filename
        # (everything before the first "_p") and create the same subfolder in processed/
        record_id   = crop_path.name.split("_p")[0]
        patent_out  = processed_dir / record_id
        patent_out.mkdir(parents=True, exist_ok=True)
        out_path    = patent_out / crop_path.with_suffix(f".{ext}").name

        if out_path.exists():
            output_paths.append(out_path)
            continue

        try:
            img    = Image.open(crop_path).convert("RGB")
            result = resize_and_pad(img, target_size, pad_color)
            result.save(out_path, fmt)
            output_paths.append(out_path)
        except Exception as e:
            print(f"\n  Error processing {crop_path.name}: {e}")

    print(f"\nProcessing complete: {len(output_paths)} images saved to {processed_dir}")
    return output_paths

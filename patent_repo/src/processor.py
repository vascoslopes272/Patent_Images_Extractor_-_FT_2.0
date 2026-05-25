"""
processor.py — Resize and pad crops to DINOv2-ready square images.

The strategy:
  1. Keep aspect ratio — resize so the longer side = target_size
  2. Pad the shorter side with white to make it exactly target_size x target_size
  3. Save to processed/ folder with the same filename

This preserves quality and gives DINOv2 clean, consistent input.

Usage:
    from src.processor import process_crops
    process_crops(crop_paths, cfg)
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm


def resize_and_pad(img: Image.Image, target_size: int, pad_color: tuple) -> Image.Image:
    """Resize keeping aspect ratio, then pad to square."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # High-quality resize
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create white square canvas and paste centered
    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))
    return canvas


def process_crops(crop_paths: list, cfg: dict) -> list:
    """
    Resize and pad all crops. Skips already-processed files.
    Returns list of output paths.
    """
    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_size = cfg["processor"]["target_size"]
    pad_color   = tuple(cfg["processor"]["padding_color"])
    fmt         = cfg["processor"]["save_format"]

    output_paths = []

    for crop_path in tqdm(crop_paths, desc="Processing crops"):
        crop_path = Path(crop_path)
        out_path  = processed_dir / crop_path.name

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

    print(f"\nProcessing complete: {len(output_paths)} images ready.")
    return output_paths

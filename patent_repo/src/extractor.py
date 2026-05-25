"""
extractor.py — Run DocLayout-YOLO on PDFs and save cropped figures.

Usage:
    from src.extractor import extract_crops
    extract_crops(pdf_paths, cfg)
"""

import fitz  # pymupdf
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def pdf_to_images(pdf_path: Path, dpi: int = 150):
    """Render each page of a PDF as a PIL Image."""
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def load_model(cfg: dict):
    """Load DocLayout-YOLO model."""
    from doclayout_yolo import YOLOv10
    device = cfg["extractor"]["device"]
    model_path = cfg["extractor"]["model_path"]
    model = YOLOv10(model_path)
    model.to(device)
    return model


def extract_crops(pdf_paths: dict, cfg: dict) -> dict:
    """
    For each PDF in pdf_paths {record_id: path}, run DocLayout-YOLO
    and save cropped figures to cfg['paths']['crops'].

    Returns: {record_id: [list of crop paths]}
    """
    crops_dir = Path(cfg["paths"]["crops"])
    crops_dir.mkdir(parents=True, exist_ok=True)

    dpi          = cfg["extractor"]["pdf_dpi"]
    conf         = cfg["extractor"]["confidence_threshold"]
    target_cls   = set(cfg["extractor"]["target_classes"])
    min_px       = cfg["processor"]["min_crop_pixels"]

    model = load_model(cfg)
    all_crops = {}

    for record_id, pdf_path in tqdm(pdf_paths.items(), desc="Extracting crops"):
        record_crops = []
        try:
            pages = pdf_to_images(pdf_path, dpi=dpi)
        except Exception as e:
            print(f"\n  Error reading {record_id}: {e}")
            all_crops[record_id] = []
            continue

        crop_idx = 0
        for page_num, page_img in enumerate(pages):
            img_np = np.array(page_img)

            # Run YOLO inference
            results = model.predict(
                source=img_np,
                conf=conf,
                device=cfg["extractor"]["device"],
                verbose=False,
            )

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    # Get class name
                    cls_id   = int(box.cls[0])
                    cls_name = model.names[cls_id].lower()

                    if cls_name not in target_cls:
                        continue

                    # Get bounding box (xyxy, absolute pixels)
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    # Skip tiny detections
                    if (x2 - x1) < min_px or (y2 - y1) < min_px:
                        continue

                    # Crop and save
                    crop = page_img.crop((x1, y1, x2, y2))
                    fname = f"{record_id}_p{page_num+1:03d}_c{crop_idx+1:02d}.png"
                    out_path = crops_dir / fname
                    crop.save(out_path, "PNG")
                    record_crops.append(out_path)
                    crop_idx += 1

        all_crops[record_id] = record_crops
        total = sum(len(v) for v in all_crops.values())

    print(f"\nExtraction complete: {total} crops saved.")
    return all_crops

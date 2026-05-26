"""
extractor.py — Convert PDFs to images and extract figure crops with DocLayout-YOLO.

TWO MODES — choose based on whether you need the PDF files afterwards:

  extract_crops_streaming(df, cfg)          ← USE THIS (default, no PDFs saved)
      Downloads each PDF into RAM, processes it immediately, discards it.
      Faster: no disk write/read cycle for PDFs. Saves gigabytes of space.
      Crops are still saved to disk as usual.

  extract_crops(pdf_paths, cfg)             ← use if PDFs are already on disk
      Reads PDFs from disk.
      Use this if you want to keep PDFs around to re-run extraction with
      different YOLO settings without re-downloading.

FILE NAMING CONVENTION (same for both modes):
    crops/{RecordNumber}/{RecordNumber}_p{page:03d}_c{crop:02d}.png
    - Crops are saved in a per-patent subfolder named after the Record Number.
    - page  = page number within the PDF (1-based, zero-padded to 3 digits)
    - crop  = crop counter per patent (1-based, zero-padded to 2 digits,
              counts across all pages — so p001_c01, p001_c02, p003_c03 etc.)
    Examples:
        US2022267016A1/US2022267016A1_p001_c01.png
        US2022267016A1/US2022267016A1_p003_c02.png

FILES WRITTEN:
    crops/{RecordNumber}/{RecordNumber}_p{page:03d}_c{crop:02d}.png
    logs/extraction_status.xlsx   ← one row per patent with URL status + crop count

Usage:
    from src.extractor import extract_crops
    crop_results = extract_crops(pdf_paths, cfg)
    # crop_results = {"US2022267016A1": [Path("...c01.png"), Path("...c02.png")], ...}
"""

import io
import time
import requests
import pandas as pd
import fitz          # PyMuPDF — converts PDF pages to images
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# pdf_to_images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_source, dpi: int = 150):
    """
    Render every page of a PDF as a PIL Image and return them as a list.

    ACCEPTS:
        pdf_source : either a file path (Path / str) OR raw PDF bytes (bytes / BytesIO)
                     — the streaming mode passes bytes so the PDF never touches disk.

    DPI controls the resolution of the rendered images:
      - Higher DPI → sharper image, better YOLO detection, but slower and more RAM
      - 150 DPI is a good balance: a typical A4 page becomes ~1240×1754 px
      - Set in config.yaml → extractor.pdf_dpi

    RETURNS: list of PIL Images, one per page (index 0 = page 1)
    """
    # fitz.open() accepts a file path (str) or raw bytes via stream=
    if isinstance(pdf_source, (bytes, io.BytesIO)):
        raw = pdf_source if isinstance(pdf_source, bytes) else pdf_source.read()
        doc = fitz.open(stream=raw, filetype="pdf")
    else:
        doc = fitz.open(str(pdf_source))

    images = []
    for page in doc:
        # fitz.Matrix scales the page: (dpi/72) because PDFs are defined at 72 DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        # Convert the raw pixel bytes to a PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

def load_model(cfg: dict):
    """
    Load the DocLayout-YOLO model from the .pt weights file.

    The model is loaded ONCE per run (before the patent loop) and then
    reused for all PDFs. Loading takes a few seconds — do not call this
    inside the per-patent loop.

    Model file location: cfg["extractor"]["model_path"]
        → set in config.yaml, resolved to an absolute path by config_loader
        → weights file: pipeline/models/doclayout_yolo_docstructbench_imgsz1024.pt

    Device: cfg["extractor"]["device"]  → "cuda:1" (GPU 1, always)
    """
    from doclayout_yolo import YOLOv10
    device     = cfg["extractor"]["device"]
    model_path = cfg["extractor"]["model_path"]
    model = YOLOv10(model_path)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# extract_crops_streaming  ← the default, no PDFs saved to disk
# ---------------------------------------------------------------------------

def extract_crops_streaming(df, cfg: dict, no_url_df=None) -> dict:
    """
    Download each PDF directly into RAM, run YOLO, save crops. No PDF files written.

    This is faster than the two-step download→extract approach because:
      - No disk write for the PDF (can be 5–20 MB each × 90 patents = up to 1.8 GB)
      - No disk read to open it again immediately after
      - Less disk space needed

    TRADEOFF: if the run crashes halfway, already-processed patents' crops are
    still on disk (the crop skip logic handles this), but PDFs must be
    re-downloaded. For 90 patents at ~10s each that's ~15 min re-download.
    If you prefer to keep PDFs for re-use, use extract_crops() with pre-saved PDFs.

    PARAMETERS:
        df        : DataFrame from get_subset() — patents with valid pdf_url
        cfg       : config dict from load_config()
        no_url_df : optional — the missing_df returned by load_patents(), so patents
                    with no URL at all also appear in extraction_status.xlsx as "no_url"

    FILES WRITTEN:
        crops/{RecordNumber}/{RecordNumber}_p{page:03d}_c{crop:02d}.png
        logs/extraction_status.xlsx  — one row per patent (all statuses in one file)

    RETURNS:
        dict: { record_id: [list of crop Path objects] }
    """
    crops_dir = Path(cfg["paths"]["crops"])
    crops_dir.mkdir(parents=True, exist_ok=True)

    dpi        = cfg["extractor"]["pdf_dpi"]
    conf       = cfg["extractor"]["confidence_threshold"]
    target_cls = set(cfg["extractor"]["target_classes"])
    min_px     = cfg["processor"]["min_crop_pixels"]
    timeout    = cfg["downloader"]["timeout_seconds"]
    retries    = cfg["downloader"]["retry_attempts"]
    delay      = cfg["downloader"]["delay_between_requests"]
    fmt        = cfg["extractor"].get("save_format", "PNG").upper()
    ext        = "jpg" if fmt == "JPEG" else "png"

    model          = load_model(cfg)
    all_crops      = {}
    status_records = []   # one dict per patent → saved to extraction_status.xlsx

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Download + Extract"):
        record_id = str(row["Record Number"])
        url       = row["pdf_url"]

        # ── Per-patent subfolder ──────────────────────────────────────────
        # Each patent gets its own folder: crops_dir / {RecordNumber} /
        patent_dir = crops_dir / record_id
        patent_dir.mkdir(parents=True, exist_ok=True)

        # ── Skip if already processed ─────────────────────────────────────
        existing = list(patent_dir.glob(f"{record_id}_p*.{ext}"))
        if existing:
            all_crops[record_id] = existing
            status_records.append({
                "Record Number": record_id,
                "pdf_url":       url,
                "status":        "skipped (already done)",
                "n_pages":       None,
                "n_crops":       len(existing),
                "error":         None,
            })
            continue

        # ── Download PDF into memory (no disk write) ──────────────────────
        pdf_bytes  = None
        last_error = None
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()
                pdf_bytes = resp.content
                break
            except Exception as e:
                last_error = str(e)
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    print(f"\n  FAILED to download {record_id}: {e}")

        if pdf_bytes is None:
            all_crops[record_id] = []
            status_records.append({
                "Record Number": record_id,
                "pdf_url":       url,
                "status":        "download_failed",
                "n_pages":       None,
                "n_crops":       0,
                "error":         last_error,
            })
            continue

        # ── Render PDF pages to images ────────────────────────────────────
        # Fails here if the URL returned HTML or a non-PDF file instead of a PDF.
        try:
            pages = pdf_to_images(pdf_bytes, dpi=dpi)
        except Exception as e:
            print(f"\n  Not a valid PDF {record_id}: {e}")
            all_crops[record_id] = []
            status_records.append({
                "Record Number": record_id,
                "pdf_url":       url,
                "status":        "not_pdf",
                "n_pages":       None,
                "n_crops":       0,
                "error":         str(e),
            })
            continue

        # ── Run YOLO on each page and save crops ──────────────────────────
        record_crops = []
        crop_idx     = 0

        for page_num, page_img in enumerate(pages):
            img_np  = np.array(page_img)
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
                    cls_name = model.names[int(box.cls[0])].lower()
                    if cls_name not in target_cls:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    if (x2 - x1) < min_px or (y2 - y1) < min_px:
                        continue

                    crop     = page_img.crop((x1, y1, x2, y2))
                    fname    = f"{record_id}_p{page_num+1:03d}_c{crop_idx+1:02d}.{ext}"
                    out_path = patent_dir / fname
                    crop.save(out_path, fmt)
                    record_crops.append(out_path)
                    crop_idx += 1

        all_crops[record_id] = record_crops
        status_records.append({
            "Record Number": record_id,
            "pdf_url":       url,
            "status":        "ok" if record_crops else "ok_no_crops",
            "n_pages":       len(pages),
            "n_crops":       len(record_crops),
            "error":         None,
        })
        time.sleep(delay)

    # ── Add no-URL patents to the status records ──────────────────────────
    # Patents that had no URL in the Excel are passed in via no_url_df.
    # They go at the top of the report so missing URLs are easy to spot.
    if no_url_df is not None and not no_url_df.empty:
        no_url_rows = [
            {
                "Record Number": str(row["Record Number"]),
                "pdf_url":       None,
                "status":        "no_url",
                "n_pages":       None,
                "n_crops":       0,
                "error":         "No PDF URL in Excel",
            }
            for _, row in no_url_df.iterrows()
        ]
        status_records = no_url_rows + status_records

    # ── Save combined extraction status Excel ─────────────────────────────
    # One row per patent: Record Number | pdf_url | status | n_pages | n_crops | error
    # Status values:
    #   "ok"                     → PDF downloaded, figures found and saved
    #   "ok_no_crops"            → PDF downloaded but YOLO found no figures/tables
    #   "download_failed"        → could not reach the URL after all retries
    #   "not_pdf"                → URL returned something that is not a valid PDF
    #   "skipped (already done)" → crops already existed from a previous run
    #   "no_url"                 → no PDF URL in the Excel at all
    logs_dir = Path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = logs_dir / "extraction_status.xlsx"
    pd.DataFrame(status_records).to_excel(status_path, index=False)
    print(f"Status report: {status_path}")

    total = sum(len(v) for v in all_crops.values())
    print(f"Extraction complete: {total} crops saved to {crops_dir}")
    return all_crops


# ---------------------------------------------------------------------------
# extract_crops  ← use only if PDFs are already saved on disk
# ---------------------------------------------------------------------------

def extract_crops(pdf_paths: dict, cfg: dict) -> dict:
    """
    Main Phase 1 function: run DocLayout-YOLO on every PDF and save figure crops.

    For each patent:
      1. Render all PDF pages to images (pdf_to_images)
      2. Run YOLO inference on each page image
      3. For each detected bounding box:
           - Check that the class is in target_classes (e.g. "figure")
           - Check that the box is above the confidence threshold (config: 0.3)
           - Check that the box is larger than min_crop_pixels (config: 50px)
           - If all checks pass: crop that region and save it as PNG

    PARAMETERS (all from cfg / config.yaml):
        extractor.pdf_dpi             resolution for PDF→image rendering
        extractor.confidence_threshold minimum YOLO confidence to keep a detection
        extractor.target_classes      which DocLayout class names to keep (e.g. ["figure"])
        processor.min_crop_pixels     discard crops smaller than this in either dimension

    FILES READ:
        /home/vasco/data/patents/pdfs/{RecordNumber}.pdf

    FILES WRITTEN:
        /home/vasco/data/patents/crops/{RecordNumber}_p{page:03d}_c{crop:02d}.png
        The crops/ folder is created automatically if it doesn't exist.

    RETURNS:
        dict: { record_id: [list of Path objects for each saved crop] }
        Patents that had a read error get an empty list.
        Patents with no detected figures get an empty list.
    """
    # Output folder — created automatically if missing
    crops_dir = Path(cfg["paths"]["crops"])
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Read inference settings from config
    dpi        = cfg["extractor"]["pdf_dpi"]
    conf       = cfg["extractor"]["confidence_threshold"]
    target_cls = set(cfg["extractor"]["target_classes"])  # e.g. {"figure"}
    min_px     = cfg["processor"]["min_crop_pixels"]      # e.g. 50
    fmt        = cfg["extractor"].get("save_format", "JPEG").upper()
    ext        = "jpg" if fmt == "JPEG" else "png"

    # Load the YOLO model once before the loop (slow — only done once)
    model = load_model(cfg)
    all_crops = {}

    for record_id, pdf_path in tqdm(pdf_paths.items(), desc="Extracting crops"):
        record_crops = []

        # Per-patent subfolder
        patent_dir = crops_dir / record_id
        patent_dir.mkdir(parents=True, exist_ok=True)

        try:
            pages = pdf_to_images(pdf_path, dpi=dpi)
        except Exception as e:
            print(f"\n  Error reading {record_id}: {e}")
            all_crops[record_id] = []
            continue

        crop_idx = 0

        for page_num, page_img in enumerate(pages):
            img_np = np.array(page_img)
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
                    cls_name = model.names[int(box.cls[0])].lower()
                    if cls_name not in target_cls:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    if (x2 - x1) < min_px or (y2 - y1) < min_px:
                        continue

                    crop     = page_img.crop((x1, y1, x2, y2))
                    fname    = f"{record_id}_p{page_num+1:03d}_c{crop_idx+1:02d}.{ext}"
                    out_path = patent_dir / fname
                    crop.save(out_path, fmt)
                    record_crops.append(out_path)
                    crop_idx += 1

        all_crops[record_id] = record_crops

    total = sum(len(v) for v in all_crops.values())
    print(f"\nExtraction complete: {total} crops saved to {crops_dir}")
    return all_crops

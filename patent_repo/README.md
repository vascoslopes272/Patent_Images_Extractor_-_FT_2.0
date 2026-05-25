# Patent Images Extractor & FT 2.0

Pipeline to extract, process, and curate patent figures for DINOv2 fine-tuning.

## Pipeline Overview

| Phase | Notebook | What it does |
|-------|----------|-------------|
| 0 (side quest) | `00_patent_filter.ipynb` | Explore & filter the Excel to select your target patents |
| 1 | `01_extraction.ipynb` | Download PDFs + run DocLayout-YOLO to crop figures |
| 2 | `02_processing.ipynb` | Resize + pad crops to 224×224 for DINOv2 |
| 3 | `03_review.ipynb` | Human review UI — accept/reject images |

Run everything at once: `python main.py`

## Data Storage

Data lives **locally** (not in this repo) and is backed up to Google Drive.

| What | Local path (set in config.yaml) |
|------|---------------------------------|
| Excel dataset | `~/data/patents/1629__dataset_22_05_26.xlsx` |
| Downloaded PDFs | `~/data/patents/pdfs/` |
| YOLO crops | `~/data/patents/crops/` |
| Processed images | `~/data/patents/processed/` |
| Human-reviewed | `~/data/patents/reviewed/` |

## Image Naming Convention

`{RecordNumber}_p{page:03d}_c{crop:02d}.png`

Example: `US2022267016A1_p003_c01.png`

## Subset Control

Edit `config.yaml → subset.mode`:
- `"all"` — all 1,629 patents
- `"n_first"` — first N patents (set `n_first: 90`)
- `"filter"` — filter by record_type, legal_status, tech_sub_domain

Or run from CLI: `python main.py --subset n_first --n 90`

## Setup

```bash
conda create -n patents python=3.11
conda activate patents
pip install -r requirements.txt
pip install doclayout-yolo   # or from source
```

Check GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

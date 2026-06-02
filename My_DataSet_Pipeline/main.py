"""
main.py — Pipeline orchestrator.

New pipeline order:
    Phase 1   → Extract: DocLayout-YOLO → crops/
    Phase 2   → Review:  interactive notebook (03_review.ipynb) — writes review_meta.json
    Phase 2.5 → Split:   sub-crop multi-image crops flagged "needs_split"
    Phase 3   → Process: resize + pad only accepted crops → processed/

Run from the terminal:

    python main.py                    # phases 1 + 2.5 + 3 (review is notebook-only)
    python main.py --phases 1         # only extract
    python main.py --phases 2.5       # only split flagged crops (after review)
    python main.py --phases 3         # only process accepted crops (after review)
    python main.py --phases 2.5 3     # split then process
    python main.py --subset n_first --n 20

Phase 2 (Review) is interactive and runs exclusively inside 03_review.ipynb.
"""

import argparse
import json
from pathlib import Path
from src.config_loader import load_config
from src.patents       import load_patents, get_subset
from src.extractor     import extract_crops_streaming
from src.subcropper    import process_splits
from src.processor     import process_crops, organize_processed


def _load_accepted_paths(cfg: dict) -> list:
    """
    Read review_meta.json and return the list of crop Paths that are approved
    and not flagged as needs_split (those are handled by Phase 2.5 first).
    """
    meta_path = Path(cfg["paths"]["logs"]) / "review_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"review_meta.json not found at {meta_path}.\n"
            "Run Phase 2 (Review) inside 03_review.ipynb first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    crops_dir = Path(cfg["paths"]["crops"])
    accepted  = []

    for patent_id, pdata in meta.items():
        if pdata.get("is_duplicate") or pdata.get("review_status") == "DISAPPROVED":
            continue
        for fname, idata in pdata.get("images", {}).items():
            if idata.get("approved") and not idata.get("needs_split"):
                p = crops_dir / patent_id / fname
                if p.exists():
                    accepted.append(p)

    print(f"Accepted crops from review_meta.json: {len(accepted)}")
    return accepted


def run(cfg: dict, phases: list):
    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    crop_index_path = logs / "crop_index.json"

    # ── Phase 1: Extract ──────────────────────────────────────────────────
    if 1 in phases:
        print("\n=== PHASE 1: Download + Extract ===")
        df, missing_df = load_patents(cfg)
        subset         = get_subset(df, cfg)
        crop_results   = extract_crops_streaming(subset, cfg, no_url_df=missing_df)
        crop_index     = {k: [str(p) for p in v] for k, v in crop_results.items()}
        with open(crop_index_path, "w") as f:
            json.dump(crop_index, f, indent=2)
        print(f"Crop index saved: {crop_index_path}")

    # ── Phase 2.5: Sub-crop splitting ─────────────────────────────────────
    if 2.5 in phases:
        print("\n=== PHASE 2.5: Sub-crop Splitting ===")
        results = process_splits(cfg)
        if not results:
            print("No crops were flagged for splitting.")

    # ── Phase 3: Process accepted crops ──────────────────────────────────
    if 3 in phases:
        print("\n=== PHASE 3: Resize + Pad (accepted crops only) ===")
        accepted = _load_accepted_paths(cfg)
        if not accepted:
            print("No accepted crops found — check review_meta.json.")
        else:
            process_crops(accepted, cfg)
            print("\n=== PHASE 3b: Organize final output with CPC codes ===")
            organize_processed(cfg)

    print("\nPipeline done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patent image pipeline")
    parser.add_argument(
        "--phases", nargs="+", type=float, default=[1, 2.5, 3],
        help="Phases to run: 1  2.5  3  (Phase 2 = review notebook only)",
    )
    parser.add_argument("--subset", choices=["all", "n_first", "filter"], default=None)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.subset:
        cfg["subset"]["mode"] = args.subset
    if args.n:
        cfg["subset"]["n_first"] = args.n

    run(cfg, args.phases)

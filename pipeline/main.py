"""
main.py — Pipeline orchestrator.

Run all phases or specific ones from the terminal:

    python main.py                        # runs phases 1 + 2 (default)
    python main.py --phases 1             # only download + extract
    python main.py --phases 2             # only process (needs phase 1 done)
    python main.py --phases 1 2           # explicit
    python main.py --subset all           # override subset mode
    python main.py --subset n_first --n 50
"""

import argparse
import json
from pathlib import Path
from src.config_loader import load_config
from src.patents import load_patents, get_subset
from src.extractor import extract_crops_streaming
from src.processor import process_crops


def run(cfg: dict, phases: list[int]):
    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    crop_index_path = logs / "crop_index.json"

    if 1 in phases:
        print("\n=== PHASE 1: Download + Extract ===")
        df, missing_df = load_patents(cfg)
        subset         = get_subset(df, cfg)
        crop_results   = extract_crops_streaming(subset, cfg, no_url_df=missing_df)
        # Save crop index
        crop_index = {k: [str(p) for p in v] for k, v in crop_results.items()}
        with open(crop_index_path, "w") as f:
            json.dump(crop_index, f, indent=2)
        print(f"Crop index saved: {crop_index_path}")

    if 2 in phases:
        print("\n=== PHASE 2: Resize + Pad ===")
        if not crop_index_path.exists():
            raise FileNotFoundError(f"No crop index found at {crop_index_path}. Run phase 1 first.")
        with open(crop_index_path) as f:
            crop_index = json.load(f)
        all_crops = [p for paths in crop_index.values() for p in paths]
        process_crops(all_crops, cfg)

    print("\n✓ Pipeline done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--subset", choices=["all", "n_first", "filter"], default=None)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()

    # Allow CLI override of subset mode
    if args.subset:
        cfg["subset"]["mode"] = args.subset
    if args.n:
        cfg["subset"]["n_first"] = args.n

    run(cfg, args.phases)

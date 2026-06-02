"""
cleanup_unused_files.py — Identify and remove legacy src/ files that were
copied from My_DataSet_Pipeline but are not needed for the DeepPatent2 pipeline.

Run with --dry-run (default) to preview what would be deleted.
Run with --delete to actually remove the files.

Usage:
    python cleanup_unused_files.py           # dry run (safe, default)
    python cleanup_unused_files.py --delete  # actually delete
"""

import argparse
import sys
from pathlib import Path

# ── Files safe to delete ──────────────────────────────────────────────────────
# These modules all belong to the PDF-extraction / PatSeer-download pipeline
# (My_DataSet_Pipeline) and have no role in the DeepPatent2 workflow.

PIPELINE_DIR = Path(__file__).resolve().parent

SAFE_TO_DELETE = [
    # PDF download & YOLO extraction — DeepPatent2 images already exist
    PIPELINE_DIR / "src" / "extractor.py",
    # PatSeer Excel loading — DeepPatent2 uses JSON labels, not PatSeer
    PIPELINE_DIR / "src" / "patents.py",
    # YOLO crop resize / padding — not needed for pre-cropped DeepPatent2 images
    PIPELINE_DIR / "src" / "processor.py",
    # Human review UI — DeepPatent2 is pre-labelled; no manual review needed
    PIPELINE_DIR / "src" / "reviewer.py",
    # Patent selector notebook UI — not applicable to DeepPatent2
    PIPELINE_DIR / "src" / "selector.py",
    # Multi-image crop splitter — not needed for DeepPatent2 single-page PNGs
    PIPELINE_DIR / "src" / "subcropper.py",
]

# ── Files to KEEP ─────────────────────────────────────────────────────────────
KEEP_REASON = {
    "zero_shot.py":          "core DINOv2 embedding + clustering logic",
    "deeppatent_loader.py":  "DeepPatent2-specific dataset indexer (new)",
    "config_loader.py":      "config file loader (updated for config_deeppatent.yaml)",
    "__init__.py":            "package marker",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete the files (default: dry run)")
    args = parser.parse_args()

    dry_run = not args.delete

    print("=" * 65)
    print(f"Deepatent2_Pipeline — src/ cleanup  ({'DRY RUN' if dry_run else 'DELETE'})")
    print("=" * 65)

    # ── Files to delete ───────────────────────────────────────────────────────
    print("\nFiles scheduled for REMOVAL:")
    to_remove = []
    for path in SAFE_TO_DELETE:
        exists = path.exists()
        status = "✓ exists" if exists else "— already gone"
        print(f"  {status}  {path.relative_to(PIPELINE_DIR)}")
        if exists:
            to_remove.append(path)

    # ── Files to keep ─────────────────────────────────────────────────────────
    print("\nFiles to KEEP:")
    for name, reason in KEEP_REASON.items():
        p = PIPELINE_DIR / "src" / name
        exists = "✓" if p.exists() else "✗ MISSING"
        print(f"  {exists}  src/{name:<30s}  ({reason})")

    # ── Confirm & execute ─────────────────────────────────────────────────────
    print()
    if not to_remove:
        print("Nothing to delete — all target files already removed.")
        return

    if dry_run:
        print(f"DRY RUN: would delete {len(to_remove)} file(s).")
        print("Re-run with --delete to actually remove them.")
        return

    confirm = input(f"\nAbout to DELETE {len(to_remove)} file(s). Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        sys.exit(0)

    for path in to_remove:
        path.unlink()
        print(f"  Deleted: {path.relative_to(PIPELINE_DIR)}")

    print(f"\nDone. Removed {len(to_remove)} file(s).")


if __name__ == "__main__":
    main()

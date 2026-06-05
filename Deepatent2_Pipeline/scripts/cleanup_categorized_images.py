"""
cleanup_categorized_images.py — Run after reorganize_patent_images.py to fix the
existing Aicraft related Images/ folder.

Three problems fixed:
  1. Folder naming — old script created "UAV /Drone/" (nested) due to "/" in names.
     This script moves images to flat "UAV - Drone/" folders.
  2. False-positive images — re-classifies every image with the updated keyword
     taxonomy and deletes those that no longer match their platform folder.
  3. Other/ folder — deletes the 1M+ non-aviation images in the Other/ dump folder.

Usage:
  python cleanup_categorized_images.py --dry-run    # preview only, no changes
  python cleanup_categorized_images.py --run        # apply all three fixes
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
AICRAFT_ROOT = Path("/mnt/storage_11tb/Drive_files_to_syncronize/Aicraft related Images")
JSON_DIR     = Path("/mnt/storage_11tb/Drive_files_to_syncronize/DeepPatent2/Json_Files")

YEAR_JSON_FILES = {
    "2007": "segmentation_2007.json",
    "2008": "design2008.json", "2009": "design2009.json",
    "2010": "design2010.json", "2011": "design2011.json",
    "2012": "design2012.json", "2013": "design2013.json",
    "2014": "design2014.json", "2015": "design2015.json",
    "2016": "design2016.json", "2017": "design2017.json",
    "2018": "design2018.json", "2019": "design2019.json",
    "2019part3": "design2019part3.json", "2020": "design2020.json",
}

# ── Import taxonomy ───────────────────────────────────────────────────────────
_ANALYST_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_ANALYST_DIR))
from categories_refactored import PLATFORM_ARCHITECTURES, safe_folder_name  # type: ignore

# ── Old nested paths that the broken slash created ────────────────────────────
# Maps canonical label → (old nested path, new flat path)
OLD_TO_NEW: list[tuple[str, Path, Path]] = []
for cat in PLATFORM_ARCHITECTURES:
    if "/" in cat:
        parts = cat.split("/", 1)
        old_path = AICRAFT_ROOT / (parts[0].strip() + " ") / (" " + parts[1].strip())
        new_path = AICRAFT_ROOT / safe_folder_name(cat)
        OLD_TO_NEW.append((cat, old_path, new_path))

_TEXT_FIELDS = ("object", "object_title", "caption")


# ── Compile updated platform patterns ─────────────────────────────────────────
def _compile_patterns() -> dict[str, re.Pattern]:
    out = {}
    for platform, keywords in PLATFORM_ARCHITECTURES.items():
        sorted_kws = sorted(keywords, key=len, reverse=True)
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in sorted_kws) + r")\b",
            re.IGNORECASE,
        )
        out[platform] = pattern
    return out


_PATTERNS = _compile_patterns()


def classify_platform(text: str) -> str:
    for platform, pattern in _PATTERNS.items():
        if pattern.search(text):
            return platform
    return "Other"


# ── Load patent metadata from JSON files ──────────────────────────────────────
def load_patent_texts() -> dict[str, str]:
    print("Loading patent metadata from JSON files (not touching them)...")
    patent_text: dict[str, str] = {}
    for year, json_name in YEAR_JSON_FILES.items():
        json_path = JSON_DIR / json_name
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            pid = rec.get("patentID", "")
            if not pid:
                continue
            parts = [str(rec.get(field, "")) for field in _TEXT_FIELDS if rec.get(field)]
            patent_text[pid] = patent_text.get(pid, "") + " " + " ".join(parts)
    print(f"  Loaded metadata for {len(patent_text):,} patents.")
    return patent_text


def patent_id_from_filename(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"^(.+?)-D\d{5}$", stem)
    return m.group(1) if m else stem


# ── Phase 1: Fix nested folder paths ──────────────────────────────────────────
def fix_folder_paths(dry_run: bool) -> int:
    print("\n" + "─" * 60)
    print("Phase 1: Fix nested folder paths (slash bug)")
    moved = 0
    for label, old_path, new_path in OLD_TO_NEW:
        if not old_path.exists():
            print(f"  [skip] {label}: old path not found — {old_path}")
            continue
        pngs = list(old_path.glob("*.png"))
        print(f"  {label}: {len(pngs):,} images  {old_path} → {new_path}")
        if not dry_run:
            new_path.mkdir(parents=True, exist_ok=True)
            for p in pngs:
                shutil.move(str(p), new_path / p.name)
            # Remove the now-empty nested dirs
            try:
                old_path.rmdir()
                old_path.parent.rmdir()
            except OSError:
                pass
        moved += len(pngs)
    print(f"  {'Would move' if dry_run else 'Moved'}: {moved:,} images total.")
    return moved


# ── Phase 2: Delete false-positive images in platform folders ─────────────────
def delete_false_positives(patent_texts: dict[str, str], dry_run: bool) -> int:
    print("\n" + "─" * 60)
    print("Phase 2: Re-classify images with updated keywords, delete false positives")
    deleted = 0
    by_folder: dict[str, dict] = defaultdict(lambda: {"kept": 0, "deleted": 0, "bad_titles": []})

    for cat in PLATFORM_ARCHITECTURES:
        folder = AICRAFT_ROOT / safe_folder_name(cat)
        if not folder.exists():
            continue
        pngs = list(folder.glob("*.png"))
        for img_path in pngs:
            pid = patent_id_from_filename(img_path.name)
            text = patent_texts.get(pid, "")
            new_label = classify_platform(text)
            if new_label != cat:
                # False positive — no longer matches with updated keywords
                by_folder[safe_folder_name(cat)]["deleted"] += 1
                by_folder[safe_folder_name(cat)]["bad_titles"].append(img_path.name[:50])
                deleted += 1
                if not dry_run:
                    img_path.unlink()
            else:
                by_folder[safe_folder_name(cat)]["kept"] += 1

    for folder_name, stats in sorted(by_folder.items()):
        print(f"\n  [{folder_name}]")
        print(f"    kept   : {stats['kept']:,}")
        print(f"    {'would delete' if dry_run else 'deleted'}: {stats['deleted']:,}")
        if stats["bad_titles"]:
            for t in stats["bad_titles"][:5]:
                print(f"      ✗ {t}")
            if len(stats["bad_titles"]) > 5:
                print(f"      ... and {len(stats['bad_titles']) - 5} more")

    print(f"\n  Total {'would delete' if dry_run else 'deleted'}: {deleted:,} false-positive images.")
    return deleted


# ── Phase 3: Delete the Other/ folder ─────────────────────────────────────────
def delete_other_folder(dry_run: bool) -> int:
    print("\n" + "─" * 60)
    print("Phase 3: Delete Other/ folder (non-aviation dump)")
    other_dir = AICRAFT_ROOT / "Other"
    if not other_dir.exists():
        print("  Other/ folder not found — already gone or never created.")
        return 0
    count = sum(1 for _ in other_dir.rglob("*.png"))
    print(f"  {'Would delete' if dry_run else 'Deleting'} {count:,} images in {other_dir}")
    if not dry_run:
        shutil.rmtree(other_dir)
        print("  Deleted.")
    return count


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Clean up Aicraft related Images/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="Preview all changes without touching any files.")
    group.add_argument("--run", action="store_true",
                       help="Apply all three cleanup phases.")
    args = parser.parse_args()
    dry_run = args.dry_run

    if dry_run:
        print("=" * 60)
        print("DRY RUN — no files will be modified")
        print("=" * 60)
    else:
        print("=" * 60)
        print("RUNNING cleanup — files WILL be moved/deleted")
        print("=" * 60)

    patent_texts = load_patent_texts()
    n_moved   = fix_folder_paths(dry_run)
    n_deleted = delete_false_positives(patent_texts, dry_run)
    n_other   = delete_other_folder(dry_run)

    print("\n" + "=" * 60)
    print("Summary")
    print(f"  Images fixed (path rename)  : {n_moved:,}")
    print(f"  Images deleted (false pos.) : {n_deleted:,}")
    print(f"  Other/ images removed       : {n_other:,}")
    if dry_run:
        print("\nRe-run with  --run  to apply these changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()

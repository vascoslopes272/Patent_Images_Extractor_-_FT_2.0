"""
reorganize_patent_images.py — ONE-TIME script. Run once, then delete.

What it does:
  1. Creates /mnt/storage_11tb/Drive_files_to_syncronize/Aicraft related Images/
     with one subfolder per PLATFORM_ARCHITECTURES category.
  2. Reads DeepPatent2 JSON annotations (unchanged, left in place).
  3. Classifies each patent by platform via keyword matching.
  4. COPIES (does not move) matching PNG images to:
        Aicraft related Images/{platform_name}/{original_filename}
  5. JSON files are never touched.

After running, flip  use_categorized_layout: true  in config_deeppatent.yaml
to make the notebook use the new layout.
"""

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DEEPPATENT2_ROOT = Path("/mnt/storage_11tb/Drive_files_to_syncronize/DeepPatent2")
JSON_DIR         = DEEPPATENT2_ROOT / "Json_Files"
OUTPUT_ROOT      = Path("/mnt/storage_11tb/Drive_files_to_syncronize/Aicraft related Images")

# ── Import taxonomy ───────────────────────────────────────────────────────────
_ANALYST_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_ANALYST_DIR))
from categories_refactored import PLATFORM_ARCHITECTURES, CATEGORIES, safe_folder_name  # type: ignore

# Year → image subdirectory (mirrors config_deeppatent.yaml)
YEAR_IMAGE_SUBDIRS: dict[str, str] = {
    "2007": "Original/2007/Original",
    "2008": "Original/2008/Original",
    "2009": "Original/2009/Original",
    "2010": "Original",
    "2011": "Original",
    "2012": "Original_2012/Original",
    "2013": "Original",
    "2014": "Original_2014/Original",
    "2015": "Original_2015/Original",
    "2016": "Original",
    "2017": "Original_2017/Original",
    "2018": "Original",
    "2019": "Original",
    "2019part3": "Original",
    "2020": "Original_2020/Original",
}

YEAR_JSON_FILES: dict[str, str] = {
    "2007": "segmentation_2007.json",
    "2008": "design2008.json",
    "2009": "design2009.json",
    "2010": "design2010.json",
    "2011": "design2011.json",
    "2012": "design2012.json",
    "2013": "design2013.json",
    "2014": "design2014.json",
    "2015": "design2015.json",
    "2016": "design2016.json",
    "2017": "design2017.json",
    "2018": "design2018.json",
    "2019": "design2019.json",
    "2019part3": "design2019part3.json",
    "2020": "design2020.json",
}

# ── Compile platform patterns (same logic as deeppatent_loader.py) ────────────
def _compile_patterns(arch_dict: dict) -> dict[str, re.Pattern]:
    out = {}
    for platform, keywords in arch_dict.items():
        sorted_kws = sorted(keywords, key=len, reverse=True)
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in sorted_kws) + r")\b",
            re.IGNORECASE,
        )
        out[platform] = pattern
    return out

_PLATFORM_PATTERNS = _compile_patterns(PLATFORM_ARCHITECTURES)
_TEXT_FIELDS = ("object", "object_title", "caption")


def classify_platform(text: str) -> str:
    for platform, pattern in _PLATFORM_PATTERNS.items():
        if pattern.search(text):
            return platform
    return "Other"


# ── Step 1: Create output directories ─────────────────────────────────────────
def create_output_dirs() -> None:
    print("─" * 60)
    print("Step 1: Creating output directories")
    for category in CATEGORIES:
        folder = OUTPUT_ROOT / safe_folder_name(category)
        folder.mkdir(parents=True, exist_ok=True)
    # No "Other" folder — non-aviation patents are not copied
    print(f"  Created {len(CATEGORIES)} category folders under:\n  {OUTPUT_ROOT}")


# ── Step 2: Build image index (filename → Path) ───────────────────────────────
def build_image_index() -> dict[str, Path]:
    print("─" * 60)
    print("Step 2: Indexing images across year folders")
    index: dict[str, Path] = {}
    for year, subdir in YEAR_IMAGE_SUBDIRS.items():
        year_dir = DEEPPATENT2_ROOT / year / subdir
        if not year_dir.is_dir():
            print(f"  [skip] {year_dir} — not extracted yet")
            continue
        found = list(year_dir.glob("*.png"))
        for p in found:
            index[p.name] = p
        print(f"  {year:>12s} : {len(found):>6,} images")
    print(f"  Total indexed: {len(index):,}")
    return index


# ── Step 3: Build patent → platform map from JSON metadata ───────────────────
def build_patent_platform_map() -> dict[str, str]:
    print("─" * 60)
    print("Step 3: Classifying patents by platform (reading JSONs, not moving them)")
    patent_text: dict[str, str] = {}
    for year, json_name in YEAR_JSON_FILES.items():
        json_path = JSON_DIR / json_name
        if not json_path.exists():
            print(f"  [skip] {json_path.name} — not found")
            continue
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            pid = rec.get("patentID", "")
            if not pid:
                continue
            parts = [str(rec.get(field, "")) for field in _TEXT_FIELDS if rec.get(field)]
            patent_text[pid] = patent_text.get(pid, "") + " " + " ".join(parts)

    platform_map: dict[str, str] = {
        pid: classify_platform(text)
        for pid, text in patent_text.items()
    }

    counts: dict[str, int] = defaultdict(int)
    for lbl in platform_map.values():
        counts[lbl] += 1
    print(f"  Patents classified: {len(platform_map):,}")
    for plat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {plat:<40s}: {n:>6,}")
    return platform_map


# ── Step 4: Copy images to category folders ───────────────────────────────────
def _patent_id_from_filename(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"^(.+?)-D\d{5}$", stem)
    return m.group(1) if m else stem


def copy_images(
    image_index: dict[str, Path],
    patent_platform: dict[str, str],
) -> None:
    print("─" * 60)
    print("Step 4: Copying images → category folders (originals are NOT deleted)")

    counts: dict[str, int] = defaultdict(int)
    skipped = 0

    for fname, src_path in image_index.items():
        pid = _patent_id_from_filename(fname)
        platform = patent_platform.get(pid, "Other")
        if platform == "Other":
            continue  # skip non-aviation patents — no "Other" dump folder
        dest_dir = OUTPUT_ROOT / safe_folder_name(platform)
        dest_path = dest_dir / fname

        if dest_path.exists():
            skipped += 1
            continue

        shutil.copy2(src_path, dest_path)
        counts[platform] += 1

    print(f"  Already present (skipped) : {skipped:>6,}")
    print(f"  Newly copied              : {sum(counts.values()):>6,}")
    for plat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {plat:<40s}: {n:>6,} files")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("DeepPatent2 image reorganiser")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"JSON source : {JSON_DIR}  ← unchanged")
    print("=" * 60)

    create_output_dirs()
    image_index      = build_image_index()
    patent_platform  = build_patent_platform_map()
    copy_images(image_index, patent_platform)

    print("─" * 60)
    print("Done. Next step:")
    print("  Set  use_categorized_layout: true  in config_deeppatent.yaml")
    print("  to switch the notebook to the new layout.")
    print("=" * 60)

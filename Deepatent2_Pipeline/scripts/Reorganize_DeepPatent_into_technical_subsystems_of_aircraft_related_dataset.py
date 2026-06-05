"""
classify_technical_subsystems.py — Run after reorganize_patent_images.py.

Classifies already-filtered aviation images by TECHNICAL_SUBSYSTEMS and copies
them into subfolders inside the existing Aicraft related Images/ directory:

  Aicraft related Images/
    Technical Subsystems/
      Propulsion System/
      Wing Systems/
      Rotor & Blade Systems/
      Landing Gear & Undercarriage/
      ...

SOURCE  — images already filtered to aviation platforms in:
              Aicraft related Images/{platform}/

MAPPING — one image can match MULTIPLE subsystems (a "landing gear for UAV"
          ends up in both UAV - Drone/ AND Technical Subsystems/Landing Gear/).
          Images with no subsystem match are skipped.

JSON files are read but never moved or modified.

Usage:
    python3 scripts/classify_technical_subsystems.py
"""

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PLATFORM_ROOT  = Path("/mnt/storage_11tb/Drive_files_to_syncronize/Aicraft related Images")
SUBSYSTEM_ROOT = PLATFORM_ROOT / "Technical Subsystems"
JSON_DIR       = Path("/mnt/storage_11tb/Drive_files_to_syncronize/DeepPatent2/Json_Files")

YEAR_JSON_FILES = {
    "2007": "design_2007.json",
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
from categories_refactored import TECHNICAL_SUBSYSTEMS, safe_folder_name  # type: ignore

_TEXT_FIELDS = ("object", "object_title", "caption")


# ── Compile one pattern per subsystem ────────────────────────────────────────
def _compile_subsystem_patterns() -> dict[str, re.Pattern]:
    out = {}
    for subsystem, keywords in TECHNICAL_SUBSYSTEMS.items():
        sorted_kws = sorted(keywords, key=len, reverse=True)
        out[subsystem] = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in sorted_kws) + r")\b",
            re.IGNORECASE,
        )
    return out

_SUBSYSTEM_PATTERNS = _compile_subsystem_patterns()


def classify_subsystems(text: str) -> list[str]:
    """Return ALL matching subsystem labels for a text blob (can be multiple)."""
    return [sub for sub, pat in _SUBSYSTEM_PATTERNS.items() if pat.search(text)]


def patent_id_from_filename(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"^(.+?)-D\d{5}$", stem)
    return m.group(1) if m else stem


# ── Step 1: Create output directories ─────────────────────────────────────────
def create_output_dirs() -> None:
    print("─" * 60)
    print("Step 1: Creating subsystem directories")
    for subsystem in TECHNICAL_SUBSYSTEMS:
        (SUBSYSTEM_ROOT / safe_folder_name(subsystem)).mkdir(parents=True, exist_ok=True)
    print(f"  Created {len(TECHNICAL_SUBSYSTEMS)} subsystem folders under:\n  {SUBSYSTEM_ROOT}")


# ── Step 2: Load patent metadata ──────────────────────────────────────────────
def load_patent_texts() -> dict[str, str]:
    print("─" * 60)
    print("Step 2: Loading JSON metadata (files are not modified)")
    patent_text: dict[str, str] = {}
    for year, json_name in YEAR_JSON_FILES.items():
        json_path = JSON_DIR / json_name
        if not json_path.exists():
            print(f"  [skip] {json_name} — not found")
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


# ── Step 3: Classify and copy ─────────────────────────────────────────────────
def classify_and_copy(patent_texts: dict[str, str]) -> None:
    print("─" * 60)
    print("Step 3: Classifying aviation images by subsystem and copying")

    # Collect all PNGs from platform folders
    source_images: list[Path] = []
    for folder in sorted(PLATFORM_ROOT.iterdir()):
        if folder.is_dir():
            source_images.extend(folder.glob("*.png"))
    print(f"  Source aviation images: {len(source_images):,}")

    counts: dict[str, int] = defaultdict(int)
    no_match = 0
    skipped  = 0

    for img_path in source_images:
        pid  = patent_id_from_filename(img_path.name)
        text = patent_texts.get(pid, "")
        matched_subsystems = classify_subsystems(text)

        if not matched_subsystems:
            no_match += 1
            continue

        for subsystem in matched_subsystems:
            dest = SUBSYSTEM_ROOT / safe_folder_name(subsystem) / img_path.name
            if dest.exists():
                skipped += 1
                continue
            shutil.copy2(img_path, dest)
            counts[subsystem] += 1

    print(f"\n  No subsystem match (skipped)  : {no_match:,}")
    print(f"  Already present (skipped)     : {skipped:,}")
    print(f"  Newly copied                  : {sum(counts.values()):,}")
    print()
    for subsystem, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {n:>6,}  {subsystem}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Aircraft Subsystem classifier")
    print(f"Source  : {PLATFORM_ROOT}")
    print(f"Output  : {SUBSYSTEM_ROOT}")
    print(f"JSONs   : {JSON_DIR}  ← unchanged")
    print("=" * 60)

    create_output_dirs()
    patent_texts = load_patent_texts()
    classify_and_copy(patent_texts)

    print("─" * 60)
    print("Done.")
    print("=" * 60)

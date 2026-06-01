"""
Core parsing, matching, and aggregation logic for patent JSON analysis.
All functions are pure and side-effect free — easy to test in isolation.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from categories_refactored import CATEGORIES, PLATFORM_ARCHITECTURES, TECHNICAL_SUBSYSTEMS


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json_file(path: Path) -> list[dict[str, Any]]:
    """
    Load a single JSON file and return its records as a list.
    Handles both top-level lists and dicts (returns dict values if needed).
    Returns an empty list on any parse/IO error without crashing.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Some dumps wrap the list under a key — flatten all values
            records: list[dict[str, Any]] = []
            for v in data.values():
                if isinstance(v, list):
                    records.extend(v)
                elif isinstance(v, dict):
                    records.append(v)
            return records
        return []
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []


def scan_data_dir(data_dir: Path) -> list[Path]:
    """Return sorted list of all .json files found in data_dir (non-recursive)."""
    return sorted(data_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

_TEXT_FIELDS = ("object", "object_title", "caption", "title", "description")


def extract_text(record: dict[str, Any]) -> str:
    """Concatenate all relevant text fields from a record into one string."""
    parts: list[str] = []
    for field in _TEXT_FIELDS:
        val = record.get(field)
        if isinstance(val, str) and val:
            parts.append(val)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def _build_pattern(keywords: list[str]) -> re.Pattern:
    """Compile a single regex that matches any keyword as a word boundary."""
    # Sort longest first to avoid short prefix swallowing longer matches
    sorted_kws = sorted(keywords, key=len, reverse=True)
    escaped = [re.escape(kw) for kw in sorted_kws]
    joined = "|".join(escaped)
    # Use word boundaries so "car" doesn't match "cardiac"
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


_COMPILED: dict[str, re.Pattern] = {
    cat: _build_pattern(kws) for cat, kws in CATEGORIES.items()
}

# Separate patterns for platforms and subsystems
_COMPILED_PLATFORMS: dict[str, re.Pattern] = {
    cat: _build_pattern(kws) for cat, kws in PLATFORM_ARCHITECTURES.items()
}

_COMPILED_SUBSYSTEMS: dict[str, re.Pattern] = {
    cat: _build_pattern(kws) for cat, kws in TECHNICAL_SUBSYSTEMS.items()
}


def match_categories(text: str) -> list[str]:
    """Return list of category names whose keywords appear in text."""
    return [cat for cat, pat in _COMPILED.items() if pat.search(text)]


def match_platforms(text: str) -> list[str]:
    """Return platform architectures that match the text."""
    return [cat for cat, pat in _COMPILED_PLATFORMS.items() if pat.search(text)]


def match_subsystems(text: str) -> list[str]:
    """Return technical subsystems that match the text."""
    return [cat for cat, pat in _COMPILED_SUBSYSTEMS.items() if pat.search(text)]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def categorize_records(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    For each record, extract text and match against all categories.
    A record can belong to multiple categories.
    Returns a dict mapping category → list of matching records.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        text = extract_text(record)
        for cat in match_categories(text):
            buckets[cat].append(record)
    return dict(buckets)


def run_analysis(data_dir: Path) -> dict[str, Any]:
    """
    Full pipeline: scan → load → categorize → aggregate.

    Returns
    -------
    dict with keys:
        files_found     : int
        total_records   : int
        category_counts : dict[str, int]  — category → number of matching records
        category_records: dict[str, list] — category → full record list
        uncategorized   : int
        file_errors     : list[str]
        platform_counts : dict[str, int]  — platform → count
        subsystem_counts: dict[str, int]  — subsystem → count
        platform_subsystem_matrix: dict[str, dict[str, int]]
    """
    json_files = scan_data_dir(data_dir)
    files_found = len(json_files)

    all_records: list[dict[str, Any]] = []
    file_errors: list[str] = []

    for path in json_files:
        loaded = load_json_file(path)
        if not loaded and path.stat().st_size > 0:
            file_errors.append(path.name)
        all_records.extend(loaded)

    total_records = len(all_records)
    category_records = categorize_records(all_records)

    # Count records that matched NO category
    matched_ids: set[int] = set()
    for recs in category_records.values():
        matched_ids.update(id(r) for r in recs)
    uncategorized = sum(1 for r in all_records if id(r) not in matched_ids)

    # NEW: Separate platform and subsystem counts
    platform_counts: dict[str, int] = defaultdict(int)
    subsystem_counts: dict[str, int] = defaultdict(int)
    platform_subsystem_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in all_records:
        text = extract_text(record)
        platforms = match_platforms(text)
        subsystems = match_subsystems(text)

        for plat in platforms:
            platform_counts[plat] += 1
            for subsys in subsystems:
                platform_subsystem_matrix[plat][subsys] += 1

        for subsys in subsystems:
            subsystem_counts[subsys] += 1

    return {
        "files_found": files_found,
        "total_records": total_records,
        "category_counts": {cat: len(recs) for cat, recs in category_records.items()},
        "category_records": category_records,
        "uncategorized": uncategorized,
        "file_errors": file_errors,
        # NEW fields
        "platform_counts": dict(platform_counts),
        "subsystem_counts": dict(subsystem_counts),
        "platform_subsystem_matrix": {k: dict(v) for k, v in platform_subsystem_matrix.items()},
    }


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_csv_reports(results: dict[str, Any], output_dir: Path) -> None:
    """Export 3 CSV files: platforms, subsystems, and matrix."""
    import csv
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV 1: Platform Distribution
    platforms_csv = output_dir / "01_platforms_distribution.csv"
    with open(platforms_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Platform Architecture", "Patents", "Percentage"])
        total = sum(results['platform_counts'].values()) if results['platform_counts'] else 1
        for name in sorted(results['platform_counts'].keys()):
            count = results['platform_counts'][name]
            pct = (count / total * 100) if total > 0 else 0
            writer.writerow([name, count, f"{pct:.2f}%"])
    print(f"✓ {platforms_csv.name}")

    # CSV 2: Subsystems Distribution
    subsystems_csv = output_dir / "02_subsystems_distribution.csv"
    with open(subsystems_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Technical Subsystem", "Patents"])
        for name in sorted(results['subsystem_counts'].keys()):
            count = results['subsystem_counts'][name]
            writer.writerow([name, count])
    print(f"✓ {subsystems_csv.name}")

    # CSV 3: Platform-Subsystem Matrix
    matrix_csv = output_dir / "03_platform_subsystem_matrix.csv"
    subsystems = sorted(results['subsystem_counts'].keys())
    platforms = sorted(results['platform_counts'].keys())
    with open(matrix_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Platform"] + subsystems)
        for platform in platforms:
            row = [platform]
            for subsystem in subsystems:
                count = results['platform_subsystem_matrix'].get(platform, {}).get(subsystem, 0)
                row.append(count)
            writer.writerow(row)
    print(f"✓ {matrix_csv.name}")

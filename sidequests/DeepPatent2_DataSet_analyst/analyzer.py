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

from categories import CATEGORIES


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


def match_categories(text: str) -> list[str]:
    """Return list of category names whose keywords appear in text."""
    return [cat for cat, pat in _COMPILED.items() if pat.search(text)]


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

    return {
        "files_found": files_found,
        "total_records": total_records,
        "category_counts": {cat: len(recs) for cat, recs in category_records.items()},
        "category_records": category_records,
        "uncategorized": uncategorized,
        "file_errors": file_errors,
    }

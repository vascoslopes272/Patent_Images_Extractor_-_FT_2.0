"""
Compare patent codes between DeepPatent2 analysis results (JSON) and external Excel dataset.
Identifies overlaps, categories, and generates visualizations.

Handles multiple patent ID formats:
  - Design patents: USD0836880-20190101 (USD + number + date)
  - US Utility: US2022267016A1 (US + number + letter)
  - International: WO2021155385A1, CA3096221A1, etc.
"""

from __future__ import annotations

import pandas as pd
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Patent ID Normalization
# ---------------------------------------------------------------------------

def normalize_patent_id(patent_id: str) -> str:
    """
    Extract CORE patent number, ignoring format variations and dates.

    Handles:
      - USD0836880-20190101 → 0836880        (design + date, removes date)
      - US2022267016A1      → 267016         (utility + year, removes year + type)
      - US11787551B1        → 11787551       (utility, removes type)
      - WO2021155385A1      → 155385         (world + year, removes year + type)
      - CA3096221A1         → 3096221        (canadian, removes type)

    Format breakdown:
      - US2022267016A1: US (country) + 2022 (year) + 267016 (core) + A1 (type)
      - CA3096221A1: CA (country) + 3096221 (core) + A1 (type)
      - USD0836880-20190101: USD (design) + 0836880 (core) + 20190101 (date)

    Parameters
    ----------
    patent_id : str
        Raw patent ID

    Returns
    -------
    str
        Core sequential number only (no country, year, date, or type)
    """
    patent_id = patent_id.strip().upper()

    # Remove country/type prefixes: USD, US, WO, CA, EP, etc.
    # Keep only the numeric and hyphen/dash parts
    numeric_part = re.sub(r"^[A-Z]+", "", patent_id)

    # Remove trailing type letters and hyphens (like A1, B1, B2, -20190101, etc.)
    numeric_part = re.sub(r"[-_][0-9A-Z]+$", "", numeric_part)  # Remove trailing -XXXX or -XXXXXXXX
    numeric_part = re.sub(r"[A-Z]\d+$", "", numeric_part)        # Remove trailing A1, B1, etc.

    if not numeric_part:
        return patent_id  # Fallback if nothing left

    # Check if first 4 characters are a year (1950-2099) and remove it
    # This handles: US2022267016 → 267016, WO2021155385 → 155385
    if len(numeric_part) >= 8 and numeric_part[:4].isdigit():
        first_four = int(numeric_part[:4])
        if 1950 <= first_four <= 2099:
            # Year detected, return the rest
            return numeric_part[4:]

    # If no year pattern, return what we have
    return numeric_part


def normalize_patent_id_set(patent_ids: set[str]) -> dict[str, str]:
    """
    Create mapping: normalized_id → original_id for all patents.

    Returns
    -------
    dict
        {normalized: original, ...}
    """
    mapping = {}
    for pid in patent_ids:
        normalized = normalize_patent_id(pid)
        mapping[normalized] = pid
    return mapping


# ---------------------------------------------------------------------------
# Patent Loading & Extraction
# ---------------------------------------------------------------------------

def load_excel_patents(excel_path: str | Path) -> pd.DataFrame:
    """
    Load Excel file and return DataFrame with patent codes.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Record Number, Tech Domain, Tech Sub Domain, Industry, Title, CPC
    """
    df = pd.read_excel(excel_path)

    # Select relevant columns
    cols_to_keep = ["Record Number", "Tech Domain", "Tech Sub Domain", "Industry", "Title", "CPC"]
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols].copy()

    # Clean patent codes: strip whitespace, convert to string
    df["Record Number"] = df["Record Number"].astype(str).str.strip()

    # Add normalized version for robust comparison
    df["Record Number (Normalized)"] = df["Record Number"].apply(normalize_patent_id)

    return df


def extract_patents_from_analysis(analysis_results: dict[str, Any]) -> set[str]:
    """
    Extract patent IDs from the analysis results dict.
    Assumes results contain 'category_records' with nested dicts of patent records.

    Parameters
    ----------
    analysis_results : dict
        Output from analyzer.run_analysis()

    Returns
    -------
    set[str]
        Set of unique patent IDs from the analysis.
    """
    patent_ids: set[str] = set()

    # Collect from all categorized records
    for category, records in analysis_results.get("category_records", {}).items():
        for record in records:
            if "patentID" in record:
                patent_ids.add(str(record["patentID"]).strip())

    return patent_ids


def compare_patents(
    json_patent_ids: set[str],
    excel_df: pd.DataFrame,
    categorization_column: str = "Tech Domain",
) -> dict[str, Any]:
    """
    Compare patent IDs from JSON analysis with Excel dataset.
    Uses normalized IDs to handle format variations.

    Parameters
    ----------
    json_patent_ids : set[str]
        Patent IDs from DeepPatent2 analysis.
    excel_df : pd.DataFrame
        DataFrame loaded from Excel (must have 'Record Number (Normalized)' column).
    categorization_column : str
        Which column to use for categorization. Default: "Tech Domain".

    Returns
    -------
    dict with keys:
        total_json_patents      : int
        total_excel_patents     : int
        overlap_count           : int (exact + normalized matches)
        overlap_percentage      : float
        overlapping_ids         : set[str] (original IDs from Excel)
        excel_records_overlap   : pd.DataFrame
        category_breakdown      : dict[str, int]
        categorization_column   : str
        match_details           : dict  (how many matched by which method)
    """
    # Normalize JSON IDs
    json_normalized = {normalize_patent_id(pid): pid for pid in json_patent_ids}
    excel_normalized = set(excel_df["Record Number (Normalized)"].unique())

    # Find overlaps using normalized IDs
    overlapping_normalized = json_normalized.keys() & excel_normalized

    # Map back to original IDs
    overlapping_ids = set(excel_df[
        excel_df["Record Number (Normalized)"].isin(overlapping_normalized)
    ]["Record Number"].unique())

    # Get the records from Excel that overlap
    overlap_records = excel_df[excel_df["Record Number"].isin(overlapping_ids)].copy()

    # Count by category
    if categorization_column in overlap_records.columns:
        category_counts = overlap_records[categorization_column].value_counts().to_dict()
    else:
        category_counts = {}

    return {
        "total_json_patents": len(json_patent_ids),
        "total_excel_patents": len(excel_df),
        "overlap_count": len(overlapping_ids),
        "overlap_percentage": round(100 * len(overlapping_ids) / len(excel_df), 2),
        "overlapping_ids": overlapping_ids,
        "excel_records_overlap": overlap_records,
        "category_breakdown": category_counts,
        "categorization_column": categorization_column,
        "match_details": {
            "method": "normalized core numbers",
            "normalized_matches": len(overlapping_normalized),
        },
    }


def print_comparison_summary(comparison_results: dict[str, Any]) -> None:
    """Pretty-print a summary of the comparison results."""
    print("=" * 70)
    print("PATENT COMPARISON SUMMARY (with normalized ID matching)")
    print("=" * 70)
    print(f"Total patents in JSON analysis      : {comparison_results['total_json_patents']:,}")
    print(f"Total patents in Excel file         : {comparison_results['total_excel_patents']:,}")
    print(f"Overlapping patents found           : {comparison_results['overlap_count']:,}")
    print(f"Overlap percentage (vs Excel)       : {comparison_results['overlap_percentage']:.2f}%")
    print()
    print(f"Match method: {comparison_results['match_details']['method']}")
    print(f"Normalized core IDs that matched    : {comparison_results['match_details']['normalized_matches']:,}")
    print()
    print(f"Categorization by '{comparison_results['categorization_column']}':")
    print("-" * 70)
    if comparison_results["category_breakdown"]:
        for category, count in sorted(
            comparison_results["category_breakdown"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = 100 * count / comparison_results["overlap_count"]
            print(f"  {category:45} {count:>6,}  ({pct:>6.2f}%)")
    else:
        print("  (No categorization data available)")
    print("=" * 70)


def save_overlap_to_csv(
    comparison_results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the overlapping patents and their details to CSV."""
    output_path = Path(output_path)
    comparison_results["excel_records_overlap"].to_csv(output_path, index=False)
    print(f"Overlap data saved to: {output_path}")

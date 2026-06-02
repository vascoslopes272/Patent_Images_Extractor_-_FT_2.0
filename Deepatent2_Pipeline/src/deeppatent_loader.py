"""
deeppatent_loader.py — Dataset indexer for DeepPatent2.

Builds a unified image-path list and metadata index from the DeepPatent2
directory structure (yearly subfolders → Original/) and the flat JSON labels
directory.  Optionally filters to specific platform architecture categories
using the keyword taxonomy from DeepPatent2_DataSet_analyst/categories_refactored.py.

Usage in notebook
-----------------
    from src.config_loader import load_config
    from src.deeppatent_loader import DeepPatent2Dataset

    cfg     = load_config(config_name="config_deeppatent.yaml")
    dataset = DeepPatent2Dataset(cfg)

    image_paths = dataset.get_image_paths()   # list[Path]
    img_df      = dataset.build_dataframe()   # pd.DataFrame — one row per image

    # Platform label for UMAP ground-truth colouring
    labels = [dataset.get_platform_label(p) for p in image_paths]

PUBLIC API
──────────
    DeepPatent2Dataset(cfg)
        .get_image_paths()            → list[Path]
        .get_metadata_for_image(path) → dict
        .get_platform_label(path)     → str
        .build_dataframe()            → pd.DataFrame

    patent_id_from_path(path)         → str   (USD0850055-20190604)
    year_from_path(path)              → str   (e.g. "2019part3")
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── Load keyword taxonomy from DeepPatent2_DataSet_analyst ───────────────────
# The categories_refactored module lives in a sibling directory.  We inject it
# into sys.path at import time so callers do not need to configure PYTHONPATH.
_ANALYST_DIR = Path(__file__).resolve().parent.parent / "DeepPatent2_DataSet_analyst"
if str(_ANALYST_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYST_DIR))

try:
    from categories_refactored import PLATFORM_ARCHITECTURES  # type: ignore
    _HAVE_TAXONOMY = True
except ImportError:
    PLATFORM_ARCHITECTURES = {}
    _HAVE_TAXONOMY = False


# ── Filename helpers ──────────────────────────────────────────────────────────

def patent_id_from_path(path: Path) -> str:
    """Extract patent ID from a DeepPatent2 filename.

    USD0850055-20190604-D00001.png  →  USD0850055-20190604
    Falls back to the full stem if the pattern does not match.
    """
    stem = Path(path).stem
    m = re.match(r"^(.+?)-D\d{5}$", stem)
    return m.group(1) if m else stem


def year_from_path(path: Path) -> str:
    """Infer the year key from the image path by walking up to the first
    directory component that starts with a 4-digit year.

    .../DeepPatent2/2019part3/Original/USD...png  →  "2019part3"
    .../DeepPatent2/2020/Original_2020/Original/USD...png  →  "2020"
    """
    for part in reversed(Path(path).parts):
        if re.match(r"^\d{4}", part):
            return part
    return ""


# ── Platform keyword classifier ───────────────────────────────────────────────

_TEXT_FIELDS = ("object", "object_title", "caption")


def _compile_platform_patterns() -> dict[str, re.Pattern]:
    """Build one compiled regex pattern per platform architecture."""
    patterns = {}
    for platform, keywords in PLATFORM_ARCHITECTURES.items():
        sorted_kws = sorted(keywords, key=len, reverse=True)
        escaped    = [re.escape(kw) for kw in sorted_kws]
        joined     = "|".join(escaped)
        patterns[platform] = re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)
    return patterns


# Compiled once at import time (empty if taxonomy unavailable)
_PLATFORM_PATTERNS: dict[str, re.Pattern] = (
    _compile_platform_patterns() if _HAVE_TAXONOMY else {}
)


def classify_platform(text: str) -> str:
    """Return the first matching platform architecture label for a text blob.

    Checks platforms in PLATFORM_ARCHITECTURES definition order and returns
    the first match.  Returns "Other" if no pattern matches.
    """
    for platform, pattern in _PLATFORM_PATTERNS.items():
        if pattern.search(text):
            return platform
    return "Other"


# ── Dataset class ─────────────────────────────────────────────────────────────

class DeepPatent2Dataset:
    """
    Index all original images, map them to JSON annotations, and optionally
    restrict to specific platform architecture categories.

    The platform label per patent is determined by keyword matching on the
    object/caption text in the JSON annotations, using the taxonomy defined in
    DeepPatent2_DataSet_analyst/categories_refactored.py.

    Parameters
    ----------
    cfg : dict from load_config(config_name="config_deeppatent.yaml")

    Key config keys used
    --------------------
    paths.images_root, paths.json_labels
    dataset.year_image_subdirs, dataset.year_json_files
    dataset.active_years          (None → auto-detect from filesystem)
    dataset.active_platforms      (None → all; e.g. ["UAV / Drone", "VTOL / Advanced Air Mobility"])
    dataset.max_samples           (None → all images)
    dataset.sample_seed
    """

    def __init__(self, cfg: dict):
        self._images_root    = Path(cfg["paths"]["images_root"])
        self._json_dir       = Path(cfg["paths"]["json_labels"])
        self._year_subdirs   = cfg["dataset"]["year_image_subdirs"]
        self._year_json_map  = cfg["dataset"]["year_json_files"]
        self._active_years   = cfg["dataset"].get("active_years")
        self._active_plats   = cfg["dataset"].get("active_platforms")  # None → all
        self._max_samples    = cfg["dataset"].get("max_samples")
        self._sample_seed    = cfg["dataset"].get("sample_seed", 42)

        self._image_index:    dict[str, Path]  = {}  # filename → path
        self._meta_index:     dict[str, list]  = {}  # figure_file → [annotations]
        self._patent_text:    dict[str, str]   = {}  # patentID → concatenated text
        self._patent_platform: dict[str, str]  = {}  # patentID → platform label
        self._filtered_names: set[str]         = set()  # figure_filenames to keep

        self._build_image_index()
        self._build_metadata_index()
        self._classify_platforms()
        self._apply_platform_filter()

    # ── Index builders ────────────────────────────────────────────────────────

    def _active_year_keys(self) -> list[str]:
        all_keys = list(self._year_subdirs.keys())
        if self._active_years:
            return [y for y in all_keys if y in self._active_years]
        # Auto-detect: only include years where the image directory exists
        return [
            year for year in all_keys
            if (self._images_root / year / self._year_subdirs[year]).is_dir()
        ]

    def _build_image_index(self) -> None:
        """Scan each extracted year folder and register filename → Path."""
        years = self._active_year_keys()
        total = 0
        for year in years:
            year_dir = self._images_root / year / self._year_subdirs[year]
            for p in year_dir.glob("*.png"):
                self._image_index[p.name] = p
                total += 1
        print(f"[DeepPatent2Dataset] Indexed {total} images across "
              f"{len(years)} year(s): {', '.join(years)}")

    def _build_metadata_index(self) -> None:
        """Load JSON annotation files and build figure_file → [annotations] map.
        Also accumulates per-patent text for platform classification.
        """
        years     = self._active_year_keys()
        total_ann = 0
        for year in years:
            json_name = self._year_json_map.get(year)
            if not json_name:
                continue
            json_path = self._json_dir / json_name
            if not json_path.exists():
                print(f"  [warning] JSON not found: {json_path}")
                continue
            with open(json_path, encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                ff  = rec.get("figure_file", "")
                pid = rec.get("patentID", "")
                if not ff or ff not in self._image_index:
                    continue
                self._meta_index.setdefault(ff, []).append(rec)
                # Accumulate text for platform classification
                if pid:
                    text_parts = [
                        str(rec.get(field, ""))
                        for field in _TEXT_FIELDS
                        if rec.get(field)
                    ]
                    self._patent_text[pid] = (
                        self._patent_text.get(pid, "") + " " + " ".join(text_parts)
                    )
                total_ann += 1
        print(f"[DeepPatent2Dataset] Metadata: {total_ann} annotations matched "
              f"to {len(self._meta_index)} figure files, "
              f"{len(self._patent_text)} unique patents.")

    def _classify_platforms(self) -> None:
        """Assign a platform architecture label to every patent."""
        if not _HAVE_TAXONOMY:
            print("  [warning] categories_refactored not found; platform labels set to 'Other'.")
        for pid, text in self._patent_text.items():
            self._patent_platform[pid] = classify_platform(text)

        if self._active_plats:
            counts = defaultdict(int)
            for lbl in self._patent_platform.values():
                counts[lbl] += 1
            for plat in self._active_plats:
                print(f"  Platform '{plat}': {counts.get(plat, 0)} patents")

    def _apply_platform_filter(self) -> None:
        """Restrict _image_index to images whose patent matches active_platforms."""
        if not self._active_plats:
            self._filtered_names = set(self._image_index.keys())
            return

        target_patents = {
            pid for pid, lbl in self._patent_platform.items()
            if lbl in self._active_plats
        }

        # Also include images with no annotations but whose filename encodes a
        # matching patent ID (best-effort fallback for unannotated images)
        kept = 0
        for fname in list(self._image_index.keys()):
            annotations = self._meta_index.get(fname, [])
            if annotations:
                pid = annotations[0].get("patentID", "")
            else:
                pid = patent_id_from_path(fname)
            if pid in target_patents:
                self._filtered_names.add(fname)
                kept += 1

        n_removed = len(self._image_index) - kept
        print(f"[DeepPatent2Dataset] Platform filter {self._active_plats}: "
              f"kept {kept} images, removed {n_removed}.")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_image_paths(self) -> list[Path]:
        """Return filtered and optionally sampled image paths (sorted)."""
        paths = sorted(
            p for fn, p in self._image_index.items()
            if fn in self._filtered_names
        )
        if self._max_samples and len(paths) > self._max_samples:
            rng   = random.Random(self._sample_seed)
            paths = sorted(rng.sample(paths, self._max_samples))
            print(f"[DeepPatent2Dataset] Sampled {len(paths)} images "
                  f"(seed={self._sample_seed})")
        return paths

    def get_metadata_for_image(self, path: Path) -> dict:
        """Return aggregated annotation metadata for a single image path."""
        filename    = Path(path).name
        annotations = self._meta_index.get(filename, [])
        pid         = patent_id_from_path(path)
        yr          = year_from_path(path)

        if not annotations:
            return {
                "patentID":       pid,
                "year":           yr,
                "object_title":   "unknown",
                "aspects":        [],
                "bbox_count":     0,
                "caption_sample": "",
            }

        objects      = [a.get("object_title", "") for a in annotations if a.get("object_title")]
        object_title = max(set(objects), key=objects.count) if objects else "unknown"
        aspects      = sorted({a["aspect"] for a in annotations if a.get("aspect")})
        caption_sample = next(
            (a["caption"] for a in annotations if a.get("caption")), ""
        )
        return {
            "patentID":       annotations[0].get("patentID", pid),
            "year":           annotations[0].get("patentdate", "")[:4] or yr,
            "object_title":   object_title,
            "aspects":        aspects,
            "bbox_count":     len(annotations),
            "caption_sample": caption_sample,
        }

    def get_platform_label(self, path: Path) -> str:
        """Return the platform architecture label for the patent of this image.

        Useful as a ground-truth colour in UMAP scatter plots:
            gt_labels = [dataset.get_platform_label(p) for p in image_paths]
        """
        filename    = Path(path).name
        annotations = self._meta_index.get(filename, [])
        if annotations:
            pid = annotations[0].get("patentID", "")
        else:
            pid = patent_id_from_path(path)
        return self._patent_platform.get(pid, "Other")

    def build_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame with one row per (filtered) image including metadata.

        Columns: figure_file, file_path, patentID, year, platform,
                 object_title, aspects, bbox_count, caption_sample
        """
        rows = []
        for img_path in self.get_image_paths():
            meta     = self.get_metadata_for_image(img_path)
            platform = self.get_platform_label(img_path)
            rows.append({
                "figure_file":    img_path.name,
                "file_path":      str(img_path),
                "patentID":       meta["patentID"],
                "year":           meta["year"],
                "platform":       platform,
                "object_title":   meta["object_title"],
                "aspects":        "; ".join(meta["aspects"]),
                "bbox_count":     meta["bbox_count"],
                "caption_sample": meta["caption_sample"],
            })
        df = pd.DataFrame(rows)
        print(f"[DeepPatent2Dataset] DataFrame: {len(df)} rows, "
              f"{df['patentID'].nunique()} unique patents, "
              f"{df['platform'].value_counts().to_dict()}")
        return df

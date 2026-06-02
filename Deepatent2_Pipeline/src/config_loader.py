"""
config_loader.py — single entry point for all configuration.

Every notebook and script imports this instead of reading config.yaml directly.
The key benefit: all path strings become absolute Path objects, so the code
works correctly regardless of which directory the notebook is opened from.

Usage:
    from src.config_loader import load_config
    cfg = load_config()

    cfg["paths"]["excel"]            # absolute Path to the Excel file
    cfg["paths"]["crops"]            # absolute Path to /home/vasco/data/patents/crops/
    cfg["extractor"]["device"]       # "cuda:1"
    cfg["subset"]["mode"]            # "n_first" / "all" / "filter"
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_config(config_name: str = "config.yaml") -> Path:
    """
    Walk up from this file's location until config_name is found.

    This means load_config() works whether called from:
      - pipeline/notebooks/   (Jupyter)
      - pipeline/             (main.py, terminal)
      - pipeline/src/         (direct imports)
    """
    here = Path(__file__).resolve().parent          # pipeline/src/
    for candidate in [here, here.parent, here.parent.parent]:
        p = candidate / config_name
        if p.exists():
            return p
    raise FileNotFoundError(f"{config_name} not found starting from " + str(here))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    config_name: str = "config.yaml",
) -> dict:
    """
    Load a config YAML file and return everything as a plain Python dict.

    Steps performed:
      1. Find and parse the config file with yaml.safe_load
      2. Convert every value in cfg["paths"] from a string → absolute Path
      3. Resolve cfg["extractor"]["model_path"] if present
      4. Read .env and inject DRIVE_PATH as cfg["paths"]["drive_root"]

    Parameters
    ----------
    config_path : optional — pass an explicit path to the config file.
                  If None, searches upward from this file automatically.
    config_name : filename to search for when config_path is None.
                  Default "config.yaml"; pass "config_deeppatent.yaml" for
                  the DeepPatent2 pipeline.

    Returns
    -------
    dict with all settings. Path values are pathlib.Path objects, not strings.
    """

    # ── Step 1: locate and parse config file ─────────────────────────────
    if config_path is None:
        config_path = _find_config(config_name)
    config_path = Path(config_path).resolve()

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # pipeline/ is the anchor for relative paths defined in config.yaml
    root = config_path.parent

    # ── Step 2: resolve cfg["paths"] → absolute Path objects ──────────────
    # Example: "data/pdfs" → /home/vasco/data/patents/pdfs
    # Absolute paths (starting with /) are kept as-is.
    for key, val in cfg.get("paths", {}).items():
        if val:
            p = Path(val).expanduser()
            cfg["paths"][key] = p if p.is_absolute() else (root / p).resolve()

    # ── Step 3: resolve model_path ────────────────────────────────────────
    # model_path lives under cfg["extractor"], not cfg["paths"], so it needs
    # its own resolution. Without this, a notebook in pipeline/notebooks/
    # would look for the model in notebooks/models/ instead of pipeline/models/.
    extractor = cfg.get("extractor", {})
    if extractor.get("model_path"):
        mp = Path(extractor["model_path"])
        extractor["model_path"] = str(mp if mp.is_absolute() else (root / mp).resolve())

    # ── Step 4: read .env and inject DRIVE_PATH ───────────────────────────
    # .env sits one level above pipeline/ at the project root.
    # DRIVE_PATH is the Google Drive mount path for backup.
    load_dotenv(config_path.parent.parent / ".env")
    drive = os.getenv("DRIVE_PATH")
    if drive:
        cfg["paths"]["drive_root"] = Path(drive)

    return cfg

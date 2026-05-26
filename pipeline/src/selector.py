"""
selector.py — Interactive patent browser and selector UI (Phase 0 / pre-extraction).

Loads all patents from the PatSeer Excel, displays per-patent metadata and any
existing crops (if Phase 1 has already run), and lets the user flag which patents
to include in the pipeline.

Every click is persisted to logs/selected_patents.json — crash-safe, resumable.

selected_patents.json structure:
{
  "total_selected": 45,
  "last_updated": "2026-05-26T14:30:00",
  "patents": {
    "US2022267016A1": {
      "selected": true,
      "title": "ELECTRICALLY POWERED ROTARY-WING AIRCRAFT",
      "cpc_first": "B64C29/0041",
      "record_type": "Patent",
      "legal_status": "ALIVE",
      "tech_domain": "TRANSPORT",
      "last_updated": "2026-05-26T14:30:00"
    },
    ...
  }
}

Downstream usage — set in config.yaml:
    subset:
      mode: "selected"   ← get_subset() reads selected_patents.json

Usage inside 01_patent_selector.ipynb:
    from src.selector import PatentSelectorUI
    ui = PatentSelectorUI(cfg)
    ui.show()
    ui.export()    # writes selected_patents.xlsx alongside the JSON
"""

import json
from io import BytesIO
from pathlib import Path
from datetime import datetime

import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import pandas as pd

from src.patents import load_patents


_N = lambda pid: str(pid).strip().upper()


class PatentSelectorUI:
    """
    Paginated patent browser for manually curating the dataset before extraction.

    Each patent card shows:
      - Patent ID, Title, CPC, Record Type, Legal Status, Tech Domain
      - Clickable PDF link
      - Horizontal thumbnail strip of existing crops (if Phase 1 already ran)
      - Select / Deselect button

    Navigate with ← → buttons or by searching a Patent ID.
    All selections are saved to logs/selected_patents.json after every click.
    """

    def __init__(self, cfg: dict, filters: dict | None = None):
        """
        filters (optional, set in the notebook — no need to touch config.yaml):
            {
              "cpc_first":    ["B64C29/0041", "B64C29/0083"],  # None = all
              "legal_status": "ALIVE",                          # None = all
              "record_type":  "Patent",                         # None = all
            }
        All keys are optional; omit or set to None to skip that filter.
        """
        self.cfg       = cfg
        self.crops_dir = Path(cfg["paths"]["crops"])
        self.meta_path = Path(cfg["paths"]["logs"]) / "selected_patents.json"
        self.thumb_max = cfg.get("reviewer", {}).get("thumbnail_max_px", 400)

        print("Loading patents from Excel…")
        df, _ = load_patents(cfg)
        if filters:
            df = self._apply_filters(df, filters)
        self._df    = df.reset_index(drop=True)
        self._total = len(self._df)

        self._thumb_cache = {}
        self._selected    = self._load_selections()
        self._idx         = 0

        self._build_widgets()

    # ── Filters ───────────────────────────────────────────────────────────

    @staticmethod
    def _apply_filters(df, filters: dict):
        if filters.get("record_type"):
            df = df[df["Record Type"] == filters["record_type"]]

        if filters.get("legal_status"):
            df = df[df["Family Legal Status(Dead/Alive)"] == filters["legal_status"]]

        if filters.get("cpc_first"):
            allowed   = [c.strip() for c in filters["cpc_first"]]
            first_cpc = df["CPC"].fillna("").str.split(r"\s*\|\s*", n=1).str[0].str.strip()
            df = df[first_cpc.isin(allowed)]

        print(f"  Filters applied: {len(df)} patents match.")
        return df

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_selections(self) -> set:
        if not self.meta_path.exists():
            return set()
        with open(self.meta_path) as f:
            data = json.load(f)
        return {_N(pid) for pid, v in data.get("patents", {}).items() if v.get("selected")}

    def _persist(self):
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                data = json.load(f)

        patents = data.setdefault("patents", {})

        for _, row in self._df.iterrows():
            pid_n = _N(str(row.get("Record Number", "")))
            if not pid_n:
                continue
            entry = patents.setdefault(pid_n, {})
            entry["selected"]     = pid_n in self._selected
            entry["title"]        = str(row.get("Title", "") or "")
            entry["cpc_first"]    = str(row.get("CPC", "") or "").split("|")[0].strip()
            entry["record_type"]  = str(row.get("Record Type", "") or "")
            entry["legal_status"] = str(row.get("Family Legal Status(Dead/Alive)", "") or "")
            entry["tech_domain"]  = str(row.get("Tech Sub Domain", "") or "")
            entry["last_updated"] = datetime.now().isoformat(timespec="seconds")

        data["total_selected"] = len(self._selected)
        data["last_updated"]   = datetime.now().isoformat(timespec="seconds")

        with open(self.meta_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Data helpers ──────────────────────────────────────────────────────

    def _row(self):
        return self._df.iloc[self._idx]

    def _crops_for(self, record_id: str) -> list:
        folder = self.crops_dir / record_id
        if not folder.exists():
            return []
        return sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))

    def _thumb_bytes(self, p: Path) -> bytes:
        key = str(p)
        if key not in self._thumb_cache:
            with Image.open(p) as img:
                img = img.convert("RGB")
                img.thumbnail((160, 160))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=65)
            self._thumb_cache[key] = buf.getvalue()
        return self._thumb_cache[key]

    # ── Widget construction ───────────────────────────────────────────────

    def _build_widgets(self):
        nav_layout = widgets.Layout(width="40px", height="100%", min_height="560px")
        self._w_prev = widgets.Button(description="<", button_style="info", layout=nav_layout)
        self._w_next = widgets.Button(description=">", button_style="info", layout=nav_layout)

        self._w_search = widgets.Text(
            placeholder="Patent ID (e.g. US2022267016A1)",
            layout=widgets.Layout(width="340px"),
        )
        self._w_go = widgets.Button(description="Go", button_style="primary")

        self._w_counter   = widgets.HTML()
        self._w_meta      = widgets.HTML()
        self._w_link      = widgets.HTML()
        self._w_sel_badge = widgets.HTML()

        self._w_select   = widgets.Button(
            description="✓  Select", button_style="success",
            layout=widgets.Layout(width="150px"),
        )
        self._w_deselect = widgets.Button(
            description="✗  Deselect", button_style="danger",
            layout=widgets.Layout(width="150px"),
        )

        self._w_crops_box = widgets.VBox(
            [], layout=widgets.Layout(width="100%")
        )

        self._w_hint = widgets.HTML(
            "<small>Use ← → to navigate. Click <b>Select</b> / <b>Deselect</b> per patent. "
            "Results are saved to <code>logs/selected_patents.json</code> immediately.</small>"
        )

        # Wire events
        self._w_prev.on_click(lambda _: self._move(-1))
        self._w_next.on_click(lambda _: self._move(+1))
        self._w_go.on_click(self._search)
        self._w_search.on_submit(self._search)
        self._w_select.on_click(self._on_select)
        self._w_deselect.on_click(self._on_deselect)

    def _assemble_layout(self) -> widgets.Widget:
        top_row     = widgets.HBox([self._w_search, self._w_go])
        action_row  = widgets.HBox([self._w_select, self._w_deselect])

        center = widgets.VBox([
            top_row,
            self._w_counter,
            self._w_meta,
            self._w_link,
            self._w_sel_badge,
            action_row,
            widgets.HTML("<hr style='margin:6px 0; border-color:#e0e0e0'>"),
            self._w_crops_box,
            self._w_hint,
        ], layout=widgets.Layout(width="100%", padding="0 8px"))

        return widgets.HBox(
            [self._w_prev, center, self._w_next],
            layout=widgets.Layout(align_items="stretch", width="100%"),
        )

    # ── Display refresh ───────────────────────────────────────────────────

    def _refresh(self):
        row    = self._row()
        pid    = str(row.get("Record Number", ""))
        pid_n  = _N(pid)
        title  = str(row.get("Title", "") or "—")
        cpc    = str(row.get("CPC", "") or "—")
        rtype  = str(row.get("Record Type", "") or "—")
        status = str(row.get("Family Legal Status(Dead/Alive)", "") or "—")
        domain = str(row.get("Tech Sub Domain", "") or "—")
        url    = str(row.get("pdf_url", "") or "")

        is_selected = pid_n in self._selected

        # Counter
        self._w_counter.value = (
            f"<b>Patent {self._idx + 1}&nbsp;/&nbsp;{self._total}</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Selected: <b>{len(self._selected)}</b>&nbsp;/&nbsp;{self._total}"
        )

        # Metadata block
        self._w_meta.value = (
            "<div style='line-height:1.6; margin:4px 0;'>"
            f"<span style='font-size:1.1em; font-weight:bold;'>{pid}</span><br>"
            f"<b>Title:</b> {title}<br>"
            f"<b>CPC:</b> {cpc}<br>"
            f"<b>Type:</b> {rtype}"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;<b>Status:</b> {status}"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;<b>Domain:</b> {domain}"
            "</div>"
        )

        # PDF link
        if url:
            self._w_link.value = (
                f"<a href='{url}' target='_blank' style='font-size:0.95em;'>"
                f"📄 Open PDF</a>"
            )
        else:
            self._w_link.value = "<span style='color:#9e9e9e; font-size:0.9em;'>No PDF link available</span>"

        # Selection badge
        if is_selected:
            self._w_sel_badge.value = (
                "<span style='color:#2e7d32; font-weight:bold; font-size:1em;'>"
                "● SELECTED</span>"
            )
        else:
            self._w_sel_badge.value = (
                "<span style='color:#9e9e9e; font-size:1em;'>"
                "○ Not selected</span>"
            )

        # Crops strip
        self._refresh_crops(pid)

    def _refresh_crops(self, record_id: str):
        crops = self._crops_for(record_id)

        if not crops:
            self._w_crops_box.children = (
                widgets.HTML(
                    "<i style='color:#9e9e9e;'>No crops yet — run Phase 1 (01_extraction.ipynb) first.</i>"
                ),
            )
            return

        # Show up to 20 thumbnails in a horizontal scroll strip
        display_crops = crops[:20]
        overflow_note = f" (+{len(crops) - 20} more)" if len(crops) > 20 else ""

        header = widgets.HTML(
            f"<b>Extracted crops: {len(crops)}{overflow_note}</b>",
            layout=widgets.Layout(margin="0 0 4px 0"),
        )

        thumb_cards = []
        for p in display_crops:
            img_w = widgets.Image(
                value=self._thumb_bytes(p),
                format="jpeg",
                layout=widgets.Layout(width="160px", height="160px", object_fit="contain"),
            )
            label = widgets.HTML(
                f"<small style='word-break:break-all;'>{p.name}</small>",
                layout=widgets.Layout(width="160px"),
            )
            thumb_cards.append(
                widgets.VBox(
                    [img_w, label],
                    layout=widgets.Layout(
                        align_items="center",
                        margin="0 6px 0 0",
                        min_width="168px",
                    ),
                )
            )

        strip = widgets.HBox(
            thumb_cards,
            layout=widgets.Layout(
                overflow_x="auto",
                flex_flow="row nowrap",
                width="100%",
                padding="4px 0",
            ),
        )
        self._w_crops_box.children = (header, strip)

    # ── Event handlers ────────────────────────────────────────────────────

    def _move(self, step: int):
        self._idx = max(0, min(self._idx + step, self._total - 1))
        self._refresh()

    def _search(self, _=None):
        query = self._w_search.value.strip().upper()
        if not query:
            return
        idx_map = {_N(str(v)): i for i, v in enumerate(self._df["Record Number"])}
        if query in idx_map:
            self._idx = idx_map[query]
            self._refresh()

    def _on_select(self, _=None):
        pid_n = _N(str(self._row().get("Record Number", "")))
        self._selected.add(pid_n)
        self._persist()
        self._refresh()

    def _on_deselect(self, _=None):
        pid_n = _N(str(self._row().get("Record Number", "")))
        self._selected.discard(pid_n)
        self._persist()
        self._refresh()

    # ── Public API ────────────────────────────────────────────────────────

    def show(self):
        """Launch the selector UI in the current Jupyter cell."""
        if self._total == 0:
            print("No patents loaded from Excel.")
            return
        print(
            f"Loaded {self._total} patents | "
            f"{len(self._selected)} already selected | "
            f"Saved to: {self.meta_path}"
        )
        root = self._assemble_layout()
        display(root)
        self._refresh()

    def export(self) -> Path:
        """
        Persist the current selections and write a companion Excel file.

        Files written:
            logs/selected_patents.json  (always kept up to date on every click)
            logs/selected_patents.xlsx  (one row per selected patent, for inspection)
        """
        self._persist()

        selected_ids = self._selected
        cols = ["Record Number", "Title", "CPC", "Record Type",
                "Family Legal Status(Dead/Alive)", "Tech Sub Domain"]
        out_cols = [c for c in cols if c in self._df.columns]

        filtered  = self._df[
            self._df["Record Number"].apply(lambda x: _N(str(x)) in selected_ids)
        ].copy()

        excel_out = Path(self.cfg["paths"]["logs"]) / "selected_patents.xlsx"
        filtered[out_cols].to_excel(excel_out, index=False)

        print(f"selected_patents.json : {self.meta_path}")
        print(f"selected_patents.xlsx : {excel_out}")
        print(f"Total selected        : {len(selected_ids)} / {self._total}")
        return self.meta_path

    def summary(self) -> dict:
        """Print and return a selection summary."""
        print(f"Selected: {len(self._selected)} / {self._total} patents")
        return {"selected": len(self._selected), "total": self._total}

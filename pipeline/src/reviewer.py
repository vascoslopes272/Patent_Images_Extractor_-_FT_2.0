"""
reviewer.py — ipywidgets gallery UI for human review of raw figure crops (Phase 2).

Reads raw crops from crops/{record_id}/ (Phase 1 output).
Writes ALL decisions to logs/review_meta.json after every single action —
no files are copied here; the processing phase reads the JSON.

review_meta.json structure (one entry per patent):
{
  "US2022267016A1": {
    "review_status": "SAVED",          # PENDING | SAVED | DISAPPROVED | DUPLICATE
    "architecture_mode": 1,            # 1 = single arch, 2 = two distinct architectures
    "architecture_stage": 0,           # 0 = not saved, 1 = arch1 saved, 2 = both saved
    "is_duplicate": false,
    "duplicate_of": null,
    "note": "interesting VTOL design",
    "main_image": "US2022267016A1_p001_c02.png",
    "images": {
      "US2022267016A1_p001_c01.png": {
        "approved": true,
        "is_main": false,
        "needs_split": false,
        "architecture": 1              # 1 or 2 (only set when architecture_mode == 2)
      }
    },
    "last_updated": "2026-05-26T14:30:00"
  }
}

Usage inside 03_review.ipynb:
    from src.reviewer import ReviewUI
    from src.config_loader import load_config
    cfg = load_config()
    ui = ReviewUI(cfg)
    ui.show()
    # After reviewing: ui.summary()
"""

import json
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from PIL import Image


_N = lambda pid: str(pid).strip().upper()   # normalize patent ID


class ReviewUI:
    """
    Interactive patent image gallery review widget for Jupyter notebooks.

    Each patent is shown as a grid of clickable thumbnail cards.
    Clicking a thumbnail toggles it between APPROVED and DISAPPROVED.
    Use the arrow buttons or search to navigate between patents.
    Every action persists to review_meta.json immediately (crash-safe).
    """

    def __init__(self, cfg: dict, patent_ids: set | None = None):
        self.cfg       = cfg
        self.crops_dir = Path(cfg["paths"]["crops"])
        self.meta_path = Path(cfg["paths"]["logs"]) / "review_meta.json"
        self.thumb_max = cfg.get("reviewer", {}).get("thumbnail_max_px", 400)

        # Optional allowlist — when set, only these patent IDs appear in the UI
        self._allowed_ids = (
            {str(p).strip().upper() for p in patent_ids} if patent_ids else None
        )

        self._patents     = self._discover_patents()
        self._thumb_cache = {}
        self._s           = self._load_state()

        self._build_widgets()
        self._recompute_totals()

    # ── Discovery ─────────────────────────────────────────────────────────

    def _discover_patents(self) -> list:
        """Find per-patent subdirectories in crops/ that contain images.
        If self._allowed_ids is set, only folders whose name matches are included."""
        patents = []
        if not self.crops_dir.exists():
            return patents
        for folder in sorted(self.crops_dir.iterdir()):
            if not folder.is_dir():
                continue
            if self._allowed_ids is not None and folder.name.upper() not in self._allowed_ids:
                continue
            imgs = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
            if imgs:
                patents.append({"patent_id": folder.name, "images": imgs})
        return patents

    # ── State init & persistence ──────────────────────────────────────────

    def _load_state(self) -> dict:
        """
        Build the working state dict from review_meta.json (if it exists).
        All sets/dicts here are keyed by normalised patent IDs.
        """
        meta = {}
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                meta = json.load(f)

        s = {
            "patent_idx":          0,
            "active_idx":          0,
            "rejected_patents":    set(),        # pid_n
            "duplicated_patents":  {},           # pid_n → original_pid_n
            "rejected_images":     set(),        # str(Path)
            "main_image_by_patent":{},           # pid_n → str(Path)
            "architecture_modes":  {},           # pid_n → 1 or 2
            "architecture_stages": {},           # pid_n → 0, 1, or 2
            "arch_selections":     {},           # pid_n → {slot: set of str(Path)}
            "needs_split":         set(),        # (pid_n, fname)
            "notes":               {},           # pid_n → str
            "saved":               False,
            "cards":               [],
            "totals":              {"total": 0, "approved": 0, "rejected": 0},
        }

        # Rebuild working state from persisted JSON
        for pid, pdata in meta.items():
            pid_n = _N(pid)

            status = pdata.get("review_status", "PENDING")
            if status == "DISAPPROVED":
                s["rejected_patents"].add(pid_n)
            if pdata.get("is_duplicate"):
                s["duplicated_patents"][pid_n] = _N(pdata.get("duplicate_of") or "")

            mode  = pdata.get("architecture_mode", 1)
            stage = pdata.get("architecture_stage", 0)
            s["architecture_modes"][pid_n]  = int(mode)
            s["architecture_stages"][pid_n] = int(stage)

            if pdata.get("note"):
                s["notes"][pid_n] = pdata["note"]

            main_fname = pdata.get("main_image")
            if main_fname:
                for pat in self._patents:
                    if _N(pat["patent_id"]) == pid_n:
                        for p in pat["images"]:
                            if p.name == main_fname:
                                s["main_image_by_patent"][pid_n] = str(p)
                        break

            for fname, idata in pdata.get("images", {}).items():
                if not idata.get("approved", True):
                    for pat in self._patents:
                        if _N(pat["patent_id"]) == pid_n:
                            for p in pat["images"]:
                                if p.name == fname:
                                    s["rejected_images"].add(str(p))
                            break
                if idata.get("needs_split"):
                    s["needs_split"].add((pid_n, fname))

                # Restore arch selections
                arch_slot = idata.get("architecture")
                if arch_slot in (1, 2) and idata.get("approved", False):
                    s["arch_selections"].setdefault(pid_n, {}).setdefault(arch_slot, set())
                    for pat in self._patents:
                        if _N(pat["patent_id"]) == pid_n:
                            for p in pat["images"]:
                                if p.name == fname:
                                    s["arch_selections"][pid_n][arch_slot].add(str(p))
                            break

        # Patents not yet in review_meta.json start fully disapproved.
        # The user clicks to approve, rather than clicking to disapprove.
        seen_pids = {_N(pid) for pid in meta}
        for pat in self._patents:
            if _N(pat["patent_id"]) not in seen_pids:
                for img_path in pat["images"]:
                    s["rejected_images"].add(str(img_path))

        return s

    def _persist(self):
        """Write the full decision state to review_meta.json. Called after every action."""
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing file to preserve any keys we don't manage (e.g. split_from)
        meta = {}
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                meta = json.load(f)

        for pat in self._patents:
            pid    = pat["patent_id"]
            pid_n  = _N(pid)
            images = pat["images"]

            pdata = meta.setdefault(pid_n, {})

            # Patent-level fields
            if self._is_dup(pid_n):
                pdata["review_status"] = "DUPLICATE"
                pdata["is_duplicate"]  = True
                pdata["duplicate_of"]  = self._s["duplicated_patents"].get(pid_n, "")
            elif self._is_rejected(pid_n):
                pdata["review_status"] = "DISAPPROVED"
                pdata["is_duplicate"]  = False
                pdata["duplicate_of"]  = None
            else:
                if pdata.get("review_status") not in ("SAVED",):
                    pdata["review_status"] = "PENDING"
                pdata["is_duplicate"] = False
                pdata["duplicate_of"] = None

            pdata["architecture_mode"]  = self._arch_mode(pid_n)
            pdata["architecture_stage"] = self._arch_stage(pid_n)
            pdata["note"]               = self._s["notes"].get(pid_n, "")

            main_src = self._s["main_image_by_patent"].get(pid_n, "")
            pdata["main_image"] = Path(main_src).name if main_src else None

            # Per-image fields
            img_meta = pdata.setdefault("images", {})
            mode = self._arch_mode(pid_n)

            for img_path in images:
                fname = img_path.name
                entry = img_meta.setdefault(fname, {})

                approved = (
                    str(img_path) not in self._s["rejected_images"]
                    and not self._is_rejected(pid_n)
                    and not self._is_dup(pid_n)
                )
                entry["approved"]    = approved
                entry["is_main"]     = (main_src != "" and str(img_path) == main_src)
                entry["needs_split"] = (pid_n, fname) in self._s["needs_split"]

                # Architecture slot tagging (only relevant when mode == 2)
                if mode == 2:
                    arch_sels = self._s["arch_selections"].get(pid_n, {})
                    if str(img_path) in arch_sels.get(1, set()):
                        entry["architecture"] = 1
                    elif str(img_path) in arch_sels.get(2, set()):
                        entry["architecture"] = 2
                    else:
                        entry["architecture"] = None
                else:
                    entry["architecture"] = 1 if approved else None

            pdata["last_updated"] = datetime.now().isoformat(timespec="seconds")

        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── State helpers ─────────────────────────────────────────────────────

    def _cur(self) -> dict:
        return self._patents[self._s["patent_idx"]]

    def _pid(self) -> str:
        return _N(self._cur()["patent_id"])

    def _is_rejected(self, pid_n: str) -> bool:
        return pid_n in self._s["rejected_patents"]

    def _is_dup(self, pid_n: str) -> bool:
        return pid_n in self._s["duplicated_patents"]

    def _is_img_rejected(self, path) -> bool:
        return str(path) in self._s["rejected_images"]

    def _arch_mode(self, pid_n: str) -> int:
        return int(self._s["architecture_modes"].get(pid_n, 1))

    def _arch_stage(self, pid_n: str) -> int:
        return int(self._s["architecture_stages"].get(pid_n, 0))

    # ── Thumbnail helper ──────────────────────────────────────────────────

    def _thumb_bytes(self, image_path: Path) -> bytes:
        key = str(image_path)
        if key not in self._thumb_cache:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((self.thumb_max, self.thumb_max))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=65)
            self._thumb_cache[key] = buf.getvalue()
        return self._thumb_cache[key]

    # ── Totals ────────────────────────────────────────────────────────────

    def _recompute_totals(self):
        total = approved = rejected = 0
        for pat in self._patents:
            pid_n = _N(pat["patent_id"])
            for img in pat["images"]:
                total += 1
                if self._is_rejected(pid_n) or self._is_dup(pid_n) or self._is_img_rejected(img):
                    rejected += 1
                else:
                    approved += 1
        self._s["totals"] = {"total": total, "approved": approved, "rejected": rejected}

    # ── Widget construction ───────────────────────────────────────────────

    def _build_widgets(self):
        self._w_info       = widgets.HTML()
        self._w_style      = widgets.HTML()
        self._w_out        = widgets.Output()
        self._w_gallery    = widgets.GridBox(
            [],
            layout=widgets.Layout(
                grid_template_columns="repeat(auto-fill, minmax(245px, 1fr))",
                gap="6px",
                max_height="72vh",
                overflow_y="auto",
                align_content="flex-start",
            ),
        )

        # Navigation
        nav_layout = widgets.Layout(height="100%", min_height="1400px", width="35px")
        self._w_prev   = widgets.Button(description="<", button_style="info",  layout=nav_layout)
        self._w_next   = widgets.Button(description=">", button_style="info",  layout=nav_layout)
        self._w_search = widgets.Text(placeholder="Enter Patent ID",
                                      layout=widgets.Layout(width="340px"))
        self._w_go     = widgets.Button(description="Go / Search", button_style="primary")

        # Patent-level actions
        self._w_disapprove = widgets.Button(description="Disapprove Patent", button_style="danger")
        self._w_reapprove  = widgets.Button(description="Reapprove Patent",  button_style="success")
        self._w_duplicate  = widgets.Button(description="Mark Duplicate",    button_style="warning")

        self._w_dup_input  = widgets.Text(placeholder="Original patent ID",
                                          layout=widgets.Layout(width="300px"))
        self._w_dup_next   = widgets.Button(description="Confirm & Next", button_style="warning")
        self._w_dup_row    = widgets.HBox([self._w_dup_input, self._w_dup_next],
                                          layout=widgets.Layout(display="none"))

        # Image-level actions
        self._w_needs_split = widgets.Button(description="Needs Split",            button_style="warning")
        self._w_set_main    = widgets.Button(description="Set Active as Main Image", button_style="primary")

        # Architecture mode
        self._w_arch_toggle = widgets.Button(description="1 architecture in this patent",
                                             button_style="info")
        self._w_save_arch1  = widgets.Button(description="Save Arch 1", button_style="success",
                                             layout=widgets.Layout(display="none"))
        self._w_save_arch2  = widgets.Button(description="Save Arch 2", button_style="warning",
                                             layout=widgets.Layout(display="none"))
        self._w_arch_hint   = widgets.HTML()

        # Note
        self._w_note = widgets.Text(placeholder="Patent note (single line)",
                                    layout=widgets.Layout(width="100%"),
                                    continuous_update=False)

        # Save & advance
        self._w_save = widgets.Button(description="Save & Advance",
                                      button_style="success",
                                      layout=widgets.Layout(width="200px"))

        self._w_hint = widgets.HTML(
            value=(
                "<small>Click a thumbnail to toggle it approved / disapproved. "
                "Use ← → to navigate patents. "
                "All decisions are saved to <code>review_meta.json</code> after every click.</small>"
            )
        )

        # Wire events
        self._w_prev.on_click(lambda _: self._move(-1))
        self._w_next.on_click(lambda _: self._move(+1))
        self._w_go.on_click(self._search)
        self._w_search.on_submit(self._search)

        self._w_disapprove.on_click(self._disapprove)
        self._w_reapprove.on_click(self._reapprove)
        self._w_duplicate.on_click(self._mark_dup)
        self._w_dup_next.on_click(self._confirm_dup)

        self._w_needs_split.on_click(self._toggle_needs_split)
        self._w_set_main.on_click(lambda _: self._set_main(self._s["active_idx"]))

        self._w_arch_toggle.on_click(self._toggle_arch)
        self._w_save_arch1.on_click(lambda _: self._save_arch(1))
        self._w_save_arch2.on_click(lambda _: self._save_arch(2))

        self._w_save.on_click(self._save_and_advance)
        self._w_note.observe(self._on_note_change, names="value")

    def _assemble_layout(self) -> widgets.Widget:
        row_search    = widgets.HBox([self._w_search, self._w_go])
        row_decisions = widgets.HBox([self._w_disapprove, self._w_reapprove, self._w_duplicate])
        row_img       = widgets.HBox([self._w_needs_split, self._w_set_main])
        row_arch      = widgets.HBox([self._w_arch_toggle, self._w_save_arch1, self._w_save_arch2])

        center = widgets.VBox([
            self._w_style,
            row_search,
            self._w_info,
            row_decisions,
            self._w_dup_row,
            row_img,
            row_arch,
            self._w_arch_hint,
            self._w_note,
            self._w_hint,
            self._w_gallery,
            self._w_save,
            self._w_out,
        ], layout=widgets.Layout(width="100%"))

        return widgets.HBox(
            [self._w_prev, center, self._w_next],
            layout=widgets.Layout(align_items="stretch", width="100%"),
        )

    # ── Header refresh ────────────────────────────────────────────────────

    def _refresh_header(self):
        pat    = self._cur()
        pid_n  = _N(pat["patent_id"])
        images = pat["images"]
        t      = self._s["totals"]
        mode   = self._arch_mode(pid_n)

        active_name = images[self._s["active_idx"]].name if images else "—"
        main_src    = self._s["main_image_by_patent"].get(pid_n, "")
        main_name   = Path(main_src).name if main_src else "None"

        if self._is_dup(pid_n):
            orig   = self._s["duplicated_patents"].get(pid_n, "pending")
            status = f"DUPLICATE (source: {orig})"
        elif self._is_rejected(pid_n):
            status = "DISAPPROVED"
        else:
            status = "APPROVED"

        self._w_info.value = (
            "<div style='line-height:1.45; margin:0; padding:0;'>"
            f"<b>Patent {self._s['patent_idx']+1} / {len(self._patents)}:</b> "
            f"{pat['patent_id']}<br>"
            f"Images: {len(images)} &nbsp;|&nbsp; Status: <b>{status}</b> "
            f"&nbsp;|&nbsp; Active: {active_name}<br>"
            f"Main image: <b>{main_name}</b><br>"
            f"Global — Total: {t['total']} &nbsp;|&nbsp; "
            f"Approved: {t['approved']} &nbsp;|&nbsp; Rejected: {t['rejected']}<br>"
            f"Architecture mode: <b>{mode}</b>"
            "</div>"
        )

        # Sync note field
        note_val = self._s["notes"].get(pid_n, "")
        if self._w_note.value != note_val:
            self._w_note.value = note_val

        # Duplicate input row visibility
        if self._is_dup(pid_n):
            self._w_dup_row.layout.display = ""
            self._w_dup_input.value = self._s["duplicated_patents"].get(pid_n, "")
        else:
            self._w_dup_row.layout.display = "none"
            self._w_dup_input.value = ""

        self._refresh_arch_controls()
        self._refresh_action_buttons()

    def _refresh_action_buttons(self):
        pid_n  = self._pid()
        images = self._cur()["images"]
        if not images:
            self._w_needs_split.disabled = True
            self._w_set_main.disabled    = True
            return
        src      = images[self._s["active_idx"]]
        approved = (
            not self._is_rejected(pid_n)
            and not self._is_dup(pid_n)
            and not self._is_img_rejected(src)
        )
        self._w_needs_split.disabled = not approved
        self._w_set_main.disabled    = self._is_dup(pid_n)

    def _refresh_arch_controls(self):
        pid_n = self._pid()
        mode  = self._arch_mode(pid_n)
        stage = self._arch_stage(pid_n)

        if self._is_dup(pid_n):
            self._w_save.disabled       = True
            self._w_save_arch1.disabled = True
            self._w_save_arch2.disabled = True
            self._w_arch_hint.value     = "<small>Duplicate patent — saving skipped.</small>"
            return

        if mode == 2:
            self._w_arch_toggle.description  = "2 architectures in this patent"
            self._w_arch_toggle.button_style = "warning"
            self._w_save.layout.display      = "none"
            self._w_save_arch1.layout.display = ""
            self._w_save_arch2.layout.display = ""
            self._w_save_arch1.disabled = False
            self._w_save_arch2.disabled = (stage < 1)
            self._w_arch_hint.value = (
                "<small>Two-arch mode: select images for Arch 1 → Save Arch 1, "
                "then re-select for Arch 2 → Save Arch 2.</small>"
            )
        else:
            self._w_arch_toggle.description  = "1 architecture in this patent"
            self._w_arch_toggle.button_style = "info"
            self._w_save.layout.display      = ""
            self._w_save_arch1.layout.display = "none"
            self._w_save_arch2.layout.display = "none"
            self._w_save.disabled    = False
            self._w_arch_hint.value  = "<small>Normal mode — one set of approved images.</small>"

    # ── Gallery ───────────────────────────────────────────────────────────

    def _build_gallery(self):
        pat    = self._cur()
        images = pat["images"]

        self._s["active_idx"] = max(0, min(self._s["active_idx"], len(images) - 1)) if images else 0
        self._s["cards"]      = []

        css      = [
            ".tbtn button { border:0!important; box-shadow:none!important; }",
            ".tbtn button:hover { filter:brightness(0.93); }",
        ]
        card_wgs = []

        for idx, img_path in enumerate(images):
            b64 = base64.b64encode(self._thumb_bytes(img_path)).decode("ascii")

            btn = widgets.Button(description="", tooltip=img_path.name)
            btn.add_class("tbtn")
            btn.add_class(f"tbtn-{idx}")
            btn.layout             = widgets.Layout(width="220px", height="220px",
                                                    padding="0px", margin="0px")
            btn.style.button_color = "transparent"
            btn.on_click(lambda _, i=idx: self._toggle_image(i))

            css.append(
                f".tbtn-{idx}{{background:url('data:image/jpeg;base64,{b64}')"
                f" center/contain no-repeat!important;}}"
            )

            name_w   = widgets.HTML(value=f"<small>{img_path.name}</small>")
            status_w = widgets.HTML(value="<b>APPROVED</b>")
            main_btn = widgets.Button(description="Set as Main", button_style="primary",
                                      layout=widgets.Layout(width="220px"))
            main_btn.on_click(lambda _, i=idx: self._set_main(i))

            card = widgets.VBox(
                [btn, name_w, status_w, main_btn],
                layout=widgets.Layout(
                    width="245px", margin="2px", padding="6px",
                    border="1px solid #cfd8dc",
                    align_items="center", background_color="#ffffff",
                ),
            )
            self._s["cards"].append({"path": img_path, "card": card, "status_w": status_w})
            card_wgs.append(card)

        self._w_style.value   = "<style>" + "".join(css) + "</style>"
        self._w_gallery.children = tuple(card_wgs)
        self._refresh_cards()
        self._refresh_header()

    def _refresh_cards(self):
        for i in range(len(self._s["cards"])):
            self._refresh_card(i)

    def _refresh_card(self, idx: int):
        pat   = self._cur()
        pid_n = _N(pat["patent_id"])
        model = self._s["cards"][idx]
        path  = model["path"]

        rejected = (
            self._is_rejected(pid_n)
            or self._is_dup(pid_n)
            or self._is_img_rejected(path)
        )
        active   = bool(pat["images"]) and path == pat["images"][self._s["active_idx"]]
        main_src = self._s["main_image_by_patent"].get(pid_n, "")
        is_main  = (main_src != "" and str(path) == main_src)
        is_split = (pid_n, path.name) in self._s["needs_split"]

        if active and rejected:
            border = "3px solid #bf360c"
        elif active:
            border = "3px solid #1565c0"
        elif rejected:
            border = "2px solid #c62828"
        else:
            border = "1px solid #cfd8dc"

        parts = []
        if self._is_dup(pid_n):
            parts.append("DUPLICATE PATENT")
        elif rejected:
            parts.append("DISAPPROVED")
        else:
            parts.append("APPROVED")
        if is_main:
            parts.append("MAIN IMAGE")
        if is_split:
            parts.append("NEEDS SPLIT")

        model["status_w"].value             = "<b>" + " &nbsp;|&nbsp; ".join(parts) + "</b>"
        model["card"].layout.border           = border
        model["card"].layout.background_color = "#fff1f1" if rejected else "#ffffff"

    # ── Event handlers ────────────────────────────────────────────────────

    def _move(self, step: int):
        new = max(0, min(self._s["patent_idx"] + step, len(self._patents) - 1))
        if new != self._s["patent_idx"]:
            self._s["patent_idx"] = new
            self._s["active_idx"] = 0
            self._s["saved"]      = False
            self._build_gallery()

    def _search(self, _=None):
        query = self._w_search.value.strip().upper()
        if not query:
            return
        idx_map = {_N(p["patent_id"]): i for i, p in enumerate(self._patents)}
        if query not in idx_map:
            with self._w_out:
                self._w_out.clear_output(wait=True)
                print(f"Patent not found: {query}")
            return
        self._s["patent_idx"] = idx_map[query]
        self._s["active_idx"] = 0
        self._build_gallery()

    def _toggle_image(self, idx: int):
        pat   = self._cur()
        pid_n = _N(pat["patent_id"])
        self._s["active_idx"] = idx

        if self._is_rejected(pid_n) or self._is_dup(pid_n):
            self._refresh_cards()
            self._refresh_header()
            return

        path = pat["images"][idx]
        key  = str(path)
        if key in self._s["rejected_images"]:
            self._s["rejected_images"].discard(key)
        else:
            self._s["rejected_images"].add(key)
            # Auto-clear main mapping if the main image is rejected
            if self._s["main_image_by_patent"].get(pid_n, "") == key:
                self._s["main_image_by_patent"].pop(pid_n, None)

        self._s["saved"] = False
        self._recompute_totals()
        self._refresh_cards()
        self._refresh_header()
        self._persist()

    def _disapprove(self, _=None):
        pid_n = self._pid()
        if self._is_dup(pid_n):
            return
        self._s["rejected_patents"].add(pid_n)
        self._s["saved"] = False
        self._recompute_totals()
        self._refresh_cards()
        self._refresh_header()
        self._persist()
        self._move(+1)

    def _reapprove(self, _=None):
        pid_n = self._pid()
        self._s["rejected_patents"].discard(pid_n)
        self._s["duplicated_patents"].pop(pid_n, None)
        self._s["saved"] = False
        self._recompute_totals()
        self._refresh_cards()
        self._refresh_header()
        self._persist()

    def _mark_dup(self, _=None):
        pid_n = self._pid()
        self._s["rejected_patents"].discard(pid_n)
        self._s["duplicated_patents"].setdefault(pid_n, "")
        self._s["saved"] = False
        self._recompute_totals()
        self._refresh_cards()
        self._refresh_header()
        self._persist()

    def _confirm_dup(self, _=None):
        pid_n    = self._pid()
        original = self._w_dup_input.value.strip().upper()
        if not original:
            return
        self._s["duplicated_patents"][pid_n] = original
        self._s["saved"] = True
        self._persist()
        self._move(+1)

    def _toggle_needs_split(self, _=None):
        pat   = self._cur()
        pid_n = _N(pat["patent_id"])
        if not pat["images"]:
            return
        src = pat["images"][self._s["active_idx"]]
        if self._is_rejected(pid_n) or self._is_dup(pid_n) or self._is_img_rejected(src):
            return
        key = (pid_n, src.name)
        if key in self._s["needs_split"]:
            self._s["needs_split"].discard(key)
        else:
            self._s["needs_split"].add(key)
        self._s["saved"] = False
        self._refresh_cards()
        self._refresh_header()
        self._persist()

    def _set_main(self, idx: int):
        pat   = self._cur()
        pid_n = _N(pat["patent_id"])
        if not pat["images"] or self._is_dup(pid_n):
            return
        idx      = max(0, min(idx, len(pat["images"]) - 1))
        selected = pat["images"][idx]
        self._s["active_idx"] = idx
        # Auto-approve the chosen image and its patent
        self._s["rejected_patents"].discard(pid_n)
        self._s["rejected_images"].discard(str(selected))
        self._s["main_image_by_patent"][pid_n] = str(selected)
        self._s["saved"] = False
        self._recompute_totals()
        self._refresh_cards()
        self._refresh_header()
        self._persist()

    def _toggle_arch(self, _=None):
        pid_n = self._pid()
        if self._is_dup(pid_n):
            return
        new_mode = 1 if self._arch_mode(pid_n) == 2 else 2
        self._s["architecture_modes"][pid_n]  = new_mode
        self._s["architecture_stages"][pid_n] = 0
        self._refresh_arch_controls()
        self._refresh_header()
        self._persist()

    def _reset_all_to_disapproved(self):
        """Between Arch1 and Arch2 save: clean slate for the second selection pass."""
        for pat in self._patents:
            for img in pat["images"]:
                self._s["rejected_images"].add(str(img))
        self._recompute_totals()
        self._refresh_cards()
        self._refresh_header()

    def _save_arch(self, slot: int):
        pid_n = self._pid()
        if self._is_dup(pid_n) or self._arch_mode(pid_n) != 2:
            return
        if slot == 2 and self._arch_stage(pid_n) < 1:
            with self._w_out:
                self._w_out.clear_output(wait=True)
                print("Save Architecture 1 before Architecture 2.")
            return

        # Record which images are approved under this slot
        pat    = self._cur()
        images = pat["images"]
        sels   = self._s["arch_selections"].setdefault(pid_n, {})
        sels[slot] = {str(img) for img in images if str(img) not in self._s["rejected_images"]}

        self._s["architecture_stages"][pid_n] = slot
        self._persist()

        if slot == 1:
            self._reset_all_to_disapproved()
            self._refresh_arch_controls()
            with self._w_out:
                self._w_out.clear_output(wait=True)
                print(f"Architecture 1 saved ({len(sels[1])} images). "
                      "Re-select images for Architecture 2, then click Save Arch 2.")
        else:
            self._s["saved"] = True
            self._s["architecture_modes"][pid_n] = 1  # reset for next visit
            with self._w_out:
                self._w_out.clear_output(wait=True)
                print(f"Architecture 2 saved ({len(sels[2])} images). Moving to next patent.")
            self._move(+1)

    def _save_and_advance(self, _=None):
        pid_n = self._pid()
        pat   = self._cur()

        self._s["saved"] = True
        self._persist()

        # Stamp review_status = SAVED directly in the file
        with open(self.meta_path) as f:
            meta = json.load(f)
        if pid_n in meta:
            meta[pid_n]["review_status"] = "SAVED"
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        images   = pat["images"]
        approved = sum(
            1 for img in images
            if not self._is_img_rejected(img) and not self._is_rejected(pid_n)
        )
        with self._w_out:
            self._w_out.clear_output(wait=True)
            print(f"{pat['patent_id']}: {approved}/{len(images)} approved → review_meta.json")
        self._move(+1)

    def _on_note_change(self, change):
        if change.get("name") != "value":
            return
        self._s["notes"][self._pid()] = change.get("new", "")
        self._persist()

    # ── Public API ────────────────────────────────────────────────────────

    def show(self):
        """Launch the review UI in the current Jupyter cell. Call once."""
        if not self._patents:
            print(f"No patent folders with images found in:\n  {self.crops_dir}")
            return
        print(f"Found {len(self._patents)} patents | "
              f"review_meta.json: {self.meta_path}")
        self._recompute_totals()
        root = self._assemble_layout()
        display(root)
        self._build_gallery()

    def summary(self) -> dict:
        """Print and return a summary of the current review state."""
        total = approved = rejected = splits = dups = 0
        for pat in self._patents:
            pid_n = _N(pat["patent_id"])
            if self._is_dup(pid_n):
                dups += 1
            for img in pat["images"]:
                total += 1
                if self._is_rejected(pid_n) or self._is_dup(pid_n) or self._is_img_rejected(img):
                    rejected += 1
                else:
                    approved += 1
                if (pid_n, img.name) in self._s["needs_split"]:
                    splits += 1
        print(
            f"Review summary\n"
            f"  Total images : {total}\n"
            f"  Approved     : {approved}\n"
            f"  Rejected     : {rejected}\n"
            f"  Needs split  : {splits}\n"
            f"  Duplicates   : {dups} patents"
        )
        return {
            "total": total, "approved": approved,
            "rejected": rejected, "needs_split": splits, "duplicates": dups,
        }


# ---------------------------------------------------------------------------
# Standalone export — can be called without instantiating ReviewUI
# ---------------------------------------------------------------------------

def export_review_excel(cfg: dict) -> Path:
    """
    Read review_meta.json and write a two-sheet Excel file to the logs folder.

    Sheet 1 — Patents (one row per patent):
        patent_id | review_status | architecture_mode | is_duplicate | duplicate_of
        main_image | note | n_images | n_approved | n_rejected | n_needs_split

    Sheet 2 — Images (one row per image):
        patent_id | filename | approved | is_main | needs_split | architecture | split_from

    Returns the Path to the written Excel file.
    """
    meta_path = Path(cfg["paths"]["logs"]) / "review_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"review_meta.json not found at {meta_path}.\n"
            "Run the Review UI (02_review.ipynb) first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    patent_rows = []
    image_rows  = []

    for patent_id, pdata in meta.items():
        images_meta = pdata.get("images", {})

        n_approved   = sum(1 for v in images_meta.values() if v.get("approved"))
        n_rejected   = sum(1 for v in images_meta.values() if not v.get("approved"))
        n_needs_split = sum(1 for v in images_meta.values() if v.get("needs_split"))

        patent_rows.append({
            "patent_id":        patent_id,
            "review_status":    pdata.get("review_status", "PENDING"),
            "architecture_mode":pdata.get("architecture_mode", 1),
            "is_duplicate":     pdata.get("is_duplicate", False),
            "duplicate_of":     pdata.get("duplicate_of") or "",
            "main_image":       pdata.get("main_image") or "",
            "note":             pdata.get("note") or "",
            "n_images":         len(images_meta),
            "n_approved":       n_approved,
            "n_rejected":       n_rejected,
            "n_needs_split":    n_needs_split,
            "last_updated":     pdata.get("last_updated", ""),
        })

        for fname, idata in images_meta.items():
            image_rows.append({
                "patent_id":   patent_id,
                "filename":    fname,
                "approved":    idata.get("approved", False),
                "is_main":     idata.get("is_main", False),
                "needs_split": idata.get("needs_split", False),
                "architecture":idata.get("architecture"),
                "split_from":  idata.get("split_from") or "",
            })

    df_patents = pd.DataFrame(patent_rows)
    df_images  = pd.DataFrame(image_rows)

    out_path = Path(cfg["paths"]["logs"]) / "review_export.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_patents.to_excel(writer, sheet_name="Patents", index=False)
        df_images.to_excel(writer,  sheet_name="Images",  index=False)

        # Auto-size columns on both sheets for readability
        for sheet_name, df in [("Patents", df_patents), ("Images", df_images)]:
            ws = writer.sheets[sheet_name]
            for col_idx, col_name in enumerate(df.columns, start=1):
                max_len = max(
                    len(str(col_name)),
                    df[col_name].astype(str).str.len().max() if len(df) else 0,
                )
                ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 2, 60)

    print(f"Excel exported → {out_path}")
    print(f"  Patents sheet : {len(df_patents)} rows")
    print(f"  Images sheet  : {len(df_images)} rows")
    return out_path

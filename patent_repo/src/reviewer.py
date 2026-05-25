"""
reviewer.py — ipywidgets UI for human review of processed crops.

Shows each image with Accept / Reject buttons.
Approved images are moved/copied to cfg['paths']['reviewed'].

Usage in notebook:
    from src.reviewer import ReviewUI
    ui = ReviewUI(processed_paths, cfg)
    ui.show()
"""

import shutil
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output


class ReviewUI:
    def __init__(self, image_paths: list, cfg: dict):
        self.paths    = [Path(p) for p in image_paths]
        self.cfg      = cfg
        self.reviewed = Path(cfg["paths"]["reviewed"])
        self.reviewed.mkdir(parents=True, exist_ok=True)

        self.index    = 0
        self.accepted = []
        self.rejected = []

        # Widgets
        self.img_widget    = widgets.Image(layout=widgets.Layout(width="300px"))
        self.label_widget  = widgets.Label()
        self.counter_widget = widgets.Label()
        self.accept_btn    = widgets.Button(description="✅ Accept",
                                            button_style="success",
                                            layout=widgets.Layout(width="140px"))
        self.reject_btn    = widgets.Button(description="❌ Reject",
                                            button_style="danger",
                                            layout=widgets.Layout(width="140px"))
        self.accept_btn.on_click(self._on_accept)
        self.reject_btn.on_click(self._on_reject)

        self.output = widgets.Output()

    def _load_current(self):
        if self.index >= len(self.paths):
            return False
        p = self.paths[self.index]
        with open(p, "rb") as f:
            self.img_widget.value = f.read()
        self.label_widget.value  = p.name
        self.counter_widget.value = (
            f"Image {self.index + 1} / {len(self.paths)}  |  "
            f"Accepted: {len(self.accepted)}  Rejected: {len(self.rejected)}"
        )
        return True

    def _on_accept(self, _):
        p = self.paths[self.index]
        dest = self.reviewed / p.name
        shutil.copy2(p, dest)
        self.accepted.append(p)
        self.index += 1
        self._advance()

    def _on_reject(self, _):
        self.rejected.append(self.paths[self.index])
        self.index += 1
        self._advance()

    def _advance(self):
        with self.output:
            clear_output(wait=True)
            if not self._load_current():
                print(f"Review complete!\n"
                      f"  Accepted: {len(self.accepted)}\n"
                      f"  Rejected: {len(self.rejected)}")
            else:
                display(self._build_ui())

    def _build_ui(self):
        return widgets.VBox([
            self.counter_widget,
            self.label_widget,
            self.img_widget,
            widgets.HBox([self.accept_btn, self.reject_btn]),
        ])

    def show(self):
        if not self.paths:
            print("No images to review.")
            return
        self._load_current()
        with self.output:
            display(self._build_ui())
        display(self.output)

    def summary(self):
        print(f"Accepted: {len(self.accepted)} images → {self.reviewed}")
        print(f"Rejected: {len(self.rejected)} images")
        return {"accepted": self.accepted, "rejected": self.rejected}

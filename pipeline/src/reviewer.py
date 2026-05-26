"""
reviewer.py — ipywidgets UI for human review of processed 224×224 crops (Phase 3).

Shows one image at a time with Accept / Reject buttons.
  - Accept → the image is COPIED to the reviewed/ folder and kept for DINOv2
  - Reject → the image is skipped (stays in processed/, never deleted)

FILES READ:
    /home/vasco/data/patents/processed/*.png   (the 224×224 padded images from Phase 2)

FILES WRITTEN:
    /home/vasco/data/patents/reviewed/*.png    (only accepted images, same filename)

Usage inside 03_review.ipynb:
    from src.reviewer import ReviewUI
    ui = ReviewUI(processed_paths, cfg)
    ui.show()       # launches the widget UI in the notebook cell output
    ui.summary()    # after review: prints accepted/rejected counts
"""

import shutil
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output


class ReviewUI:
    """
    Interactive image review widget for Jupyter notebooks.

    The UI shows:
      - A progress counter: "Image 12 / 450 | Accepted: 8  Rejected: 3"
      - The filename of the current image
      - The image itself (displayed at 300px wide)
      - Two buttons: ✅ Accept  ❌ Reject

    Clicking Accept copies the file to reviewed/ and moves to the next image.
    Clicking Reject moves to the next image without copying anything.
    When all images have been reviewed, a summary is printed.
    """

    def __init__(self, image_paths: list, cfg: dict):
        """
        Set up the review session.

        PARAMETERS:
            image_paths : list of paths to the processed 224×224 images
            cfg         : config dict from load_config()
                          → cfg["paths"]["reviewed"] is where accepted images are saved
        """
        self.paths    = [Path(p) for p in image_paths]   # all images to review
        self.cfg      = cfg
        self.reviewed = Path(cfg["paths"]["reviewed"])    # output folder for accepted images
        self.reviewed.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

        # Current position in the image list
        self.index = 0

        # Running lists of accepted and rejected image paths
        self.accepted = []
        self.rejected = []

        # ── Widgets ───────────────────────────────────────────────────────
        # Image display widget — value is set to raw PNG bytes
        self.img_widget     = widgets.Image(layout=widgets.Layout(width="300px"))
        # Label showing the current filename
        self.label_widget   = widgets.Label()
        # Label showing progress and running counts
        self.counter_widget = widgets.Label()

        # Accept button → green
        self.accept_btn = widgets.Button(
            description="✅ Accept",
            button_style="success",
            layout=widgets.Layout(width="140px")
        )
        # Reject button → red
        self.reject_btn = widgets.Button(
            description="❌ Reject",
            button_style="danger",
            layout=widgets.Layout(width="140px")
        )

        # Wire buttons to their handlers
        self.accept_btn.on_click(self._on_accept)
        self.reject_btn.on_click(self._on_reject)

        # Output widget — used to refresh the display without flickering
        self.output = widgets.Output()

    # ── Internal methods ──────────────────────────────────────────────────

    def _load_current(self):
        """
        Load the image at self.index into the display widget and update labels.
        Returns False if we have gone past the last image (review complete).
        """
        if self.index >= len(self.paths):
            return False   # no more images

        p = self.paths[self.index]

        # Load raw bytes into the image widget (faster than going through PIL)
        with open(p, "rb") as f:
            self.img_widget.value = f.read()

        self.label_widget.value   = p.name
        self.counter_widget.value = (
            f"Image {self.index + 1} / {len(self.paths)}  |  "
            f"Accepted: {len(self.accepted)}  Rejected: {len(self.rejected)}"
        )
        return True

    def _on_accept(self, _):
        """
        Called when the Accept button is clicked.
        Copies the current image to reviewed/ and advances to the next.
        """
        p    = self.paths[self.index]
        dest = self.reviewed / p.name
        shutil.copy2(p, dest)          # copy2 preserves file metadata
        self.accepted.append(p)
        self.index += 1
        self._advance()

    def _on_reject(self, _):
        """
        Called when the Reject button is clicked.
        Does NOT delete or move anything — just skips to the next image.
        """
        self.rejected.append(self.paths[self.index])
        self.index += 1
        self._advance()

    def _advance(self):
        """
        Refresh the output widget with the next image, or print the
        completion message if all images have been reviewed.
        """
        with self.output:
            clear_output(wait=True)
            if not self._load_current():
                # All images done
                print(
                    f"Review complete!\n"
                    f"  Accepted: {len(self.accepted)} images → {self.reviewed}\n"
                    f"  Rejected: {len(self.rejected)} images"
                )
            else:
                display(self._build_ui())

    def _build_ui(self):
        """Assemble the widget layout for the current image."""
        return widgets.VBox([
            self.counter_widget,
            self.label_widget,
            self.img_widget,
            widgets.HBox([self.accept_btn, self.reject_btn]),
        ])

    # ── Public methods ────────────────────────────────────────────────────

    def show(self):
        """
        Launch the review UI in the current notebook cell output.
        Call this once after creating a ReviewUI instance.

        If image_paths was empty, prints a message and returns immediately.
        """
        if not self.paths:
            print("No images to review.")
            return

        # Load the first image before displaying the output widget
        self._load_current()
        with self.output:
            display(self._build_ui())
        display(self.output)

    def summary(self):
        """
        Print a summary of the review session and return the results.
        Call this after review is complete.

        RETURNS:
            dict with keys "accepted" and "rejected", each a list of Path objects
        """
        print(f"Accepted: {len(self.accepted)} images → {self.reviewed}")
        print(f"Rejected: {len(self.rejected)} images")
        return {"accepted": self.accepted, "rejected": self.rejected}

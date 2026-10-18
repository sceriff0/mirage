"""Canonical publication figure style (Python / matplotlib).

Source of truth: ~/.claude/plotstyle/plotstyle.py
Copied into repos as `plotstyle.py`; do not edit the copy, edit the source.

Usage
-----
    import plotstyle as ps
    ps.apply_style()
    fig, axes = ps.figure(width="double", nrows=1, ncols=3)
    colors = ps.lock_palette(["Tumor", "Normal", "Stroma"], figdir="figures")
    ...
    ps.panel_labels(axes)
    ps.save(fig, "figures/fig2_expression", data=plotted_df)
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

# --------------------------------------------------------------------------
# Palette — Okabe & Ito (2008), colourblind-safe, greyscale-legible.
# Locked order. Never reorder: category->colour stability depends on it.
# --------------------------------------------------------------------------
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]
GREY = "#4D4D4D"
LIGHT_GREY = "#BFBFBF"
SEQUENTIAL = "viridis"      # continuous, perceptually uniform
DIVERGING = "RdBu_r"        # centred data only, always symmetric about the centre

# --------------------------------------------------------------------------
# Canvas geometry — millimetres, matching journal column widths.
# --------------------------------------------------------------------------
MM = 1.0 / 25.4
WIDTHS_MM = {"single": 85.0, "onehalf": 114.0, "double": 180.0}
MAX_HEIGHT_MM = 230.0       # one printed page minus caption space

# Font sizes are FINAL-SIZE points. Figures are drawn at 1:1 and never rescaled.
BASE_PT = 7
TICK_PT = 6
PANEL_PT = 8


def apply_style() -> None:
    """Install the canonical rcParams. Call once, before any plotting."""
    mpl.rcParams.update({
        # fonts — sans-serif, embedded as Type42 so editors cannot substitute
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": BASE_PT,
        "axes.labelsize": BASE_PT,
        "axes.titlesize": BASE_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": TICK_PT,
        "legend.title_fontsize": BASE_PT,
        "figure.titlesize": PANEL_PT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # strokes
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.5,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # frame — data first, chrome second
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        # colour cycle
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "image.cmap": SEQUENTIAL,
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,
        # layout — constrained_layout is what keeps multi-panel grids aligned
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.04,
        "figure.constrained_layout.wspace": 0.04,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": None,          # constrained_layout owns the margins
        "savefig.transparent": False,
        "savefig.facecolor": "white",
    })


def figure(width="single", height_mm=None, nrows=1, ncols=1, aspect=0.75, **kw):
    """Create a correctly sized figure.

    width      : "single" (85mm), "onehalf" (114mm), "double" (180mm), or a number in mm.
    height_mm  : explicit height; if None, derived from `aspect` per panel row.
    aspect     : panel height / panel width, used only when height_mm is None.
    Returns (fig, axes) with axes always a flat list, even for 1x1.
    """
    w_mm = WIDTHS_MM.get(width, width) if isinstance(width, str) else float(width)
    if height_mm is None:
        height_mm = (w_mm / ncols) * aspect * nrows
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(
            f"height {height_mm:.0f}mm exceeds one page ({MAX_HEIGHT_MM:.0f}mm); "
            "split into a main figure plus a supplementary figure"
        )
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(w_mm * MM, height_mm * MM), squeeze=False, **kw
    )
    return fig, list(axes.ravel())


def panel_labels(axes, labels=None, dx_mm=-6.0, dy_mm=1.5, size=PANEL_PT):
    """Stamp bold panel letters at a fixed offset from each axes' top-left corner.

    The offset is in millimetres of paper, not axes fractions, so letters land in
    the same place regardless of panel size — this is what stops the ragged
    A/B/C placement you get from `ax.text(-0.1, 1.05, ...)`.
    Skipped entirely when there is only one panel.
    """
    if len(axes) < 2:
        return
    labels = labels or list(string.ascii_lowercase)
    for ax, lab in zip(axes, labels):
        fig = ax.get_figure()
        offset = ScaledTranslation(dx_mm * MM, dy_mm * MM, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, lab, transform=ax.transAxes + offset,
                fontsize=size, fontweight="bold", va="bottom", ha="left")


def lock_palette(categories, figdir="figures", name="default"):
    """Return {category: hex}, persisted so the same category keeps its colour
    in every figure of the project, forever.

    First call writes <figdir>/.palette.json. Later calls reuse it and append
    only genuinely new categories. Raises past 8 categories — beyond that,
    colour stops being readable and the encoding must change (facet, shape,
    or explicit grouping).
    """
    path = Path(figdir) / ".palette.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    mapping = store.get(name, {})
    for cat in categories:
        key = str(cat)
        if key not in mapping:
            if len(mapping) >= len(OKABE_ITO):
                raise ValueError(
                    f"{len(mapping) + 1} categories exceeds the 8-colour palette; "
                    "facet, group, or encode with shape instead of adding colours"
                )
            mapping[key] = OKABE_ITO[len(mapping)]
    store[name] = mapping
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True))
    return {c: mapping[str(c)] for c in categories}


def save(fig, path, data=None, formats=("pdf", "png", "svg"), close=True):
    """Write the figure in every required format plus its source data.

    path : stem without extension, e.g. "figures/fig2_expression".
    data : DataFrame (or anything with .to_csv) of the exact plotted values;
           written as <stem>_source_data.csv for journal source-data
           requirements and figure auditability.
    """
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out)
        written.append(out)
    if data is not None:
        csv = stem.parent / f"{stem.name}_source_data.csv"
        data.to_csv(csv, index=False)
        written.append(csv)
    if close:
        plt.close(fig)
    return written

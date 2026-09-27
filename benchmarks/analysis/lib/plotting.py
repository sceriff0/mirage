"""Paper-ready matplotlib helpers: consistent theme + vector export (PDF+SVG).

Plot styles consolidated from notebooks/resources.ipynb (scaling scatter) and
notebooks/rTRE.ipynb (before/after boxplots).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THEME = {
    "figure.figsize": (5.0, 3.5),
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
}


def set_paper_theme() -> None:
    plt.rcParams.update(_THEME)


def save_fig(fig, path_stem, formats=("pdf", "svg")) -> list[Path]:
    """Save `fig` in each requested vector/raster format. Returns the paths.

    Default is paper-ready vectors (pdf+svg); pass formats=("png",) for web docs.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def _split(x, y, placeholder):
    """(real_x, real_y, synth_x, synth_y). `placeholder` is a bool mask or None."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ph = (
        np.zeros(x.shape, dtype=bool)
        if placeholder is None
        else np.asarray(placeholder, dtype=bool)
    )
    return x[~ph], y[~ph], x[ph], y[ph]


def _scatter_points(ax, x, y, placeholder, s):
    """Measured points filled; synthetic ones (lib/placeholders.py) hollow grey dashed."""
    rx, ry, px, py = _split(x, y, placeholder)
    ax.scatter(rx, ry, s=s, alpha=0.85, label="measured" if px.size else None)
    if px.size:
        ax.scatter(
            px,
            py,
            s=s,
            facecolors="none",
            edgecolors="0.5",
            linewidths=1.3,
            linestyle="--",
            label="PLACEHOLDER (synthetic)",
        )


def scatter_with_fit(x, y, slope, intercept, xlabel, ylabel, title, placeholder=None):
    """`slope`/`intercept` may be None (no real points to fit): no line is drawn.
    The fit is the caller's, computed on MEASURED rows only; placeholder points are
    drawn hollow and never enter it."""
    x = np.asarray(x, dtype=float)
    fig, ax = plt.subplots()
    _scatter_points(ax, x, y, placeholder, s=18)
    if slope is not None and intercept is not None:
        xs = np.linspace(x.min(), x.max(), 50) if x.size else np.array([0, 1])
        ax.plot(
            xs,
            slope * xs + intercept,
            color="C3",
            lw=1.5,
            label=f"y = {slope:.2f}x + {intercept:.2f}",
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig


def scatter(x, y, xlabel, ylabel, title, labels=None, placeholder=None):
    """A scatter with NO fitted line.

    Deliberately separate from scatter_with_fit above rather than passing it a
    None slope: the resource figures fit peak_rss ~ input_gb because there is a
    model behind it (a marginal cost per input GiB plus a fixed overhead).
    Accuracy against cost has no such model, and drawing a line through it would
    assert one. `labels` annotates each point with the method it belongs to,
    which is the comparison the figure is for.
    """
    fig, ax = plt.subplots()
    _scatter_points(ax, x, y, placeholder, s=24)
    if placeholder is not None and np.asarray(placeholder, dtype=bool).any():
        ax.legend()
    if labels is not None:
        for xi, yi, li in zip(x, y, labels):
            ax.annotate(
                str(li), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points"
            )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig


def bar_by_run(runs, values, ylabel, title, placeholder=None):
    """One bar per run. Synthetic bars (lib/placeholders.py) are hatched, pale and
    edged in grey, so a preview cannot pass for a measured cost."""
    runs = [str(r) for r in runs]
    values = np.asarray(values, dtype=float)
    ph = (
        np.zeros(len(runs), dtype=bool)
        if placeholder is None
        else np.asarray(placeholder, dtype=bool)
    )
    fig, ax = plt.subplots(figsize=(max(5.0, 0.35 * len(runs) + 1.5), 3.5))
    xs = np.arange(len(runs))
    ax.bar(xs[~ph], values[~ph], color="C0", label="measured" if ph.any() else None)
    if ph.any():
        ax.bar(
            xs[ph],
            values[ph],
            color="0.9",
            edgecolor="0.45",
            hatch="///",
            linestyle="--",
            label="PLACEHOLDER (synthetic)",
        )
        ax.legend()
    ax.set_xticks(xs)
    ax.set_xticklabels(runs, rotation=60, ha="right", fontsize=7)
    ax.set(ylabel=ylabel, title=title)
    return fig


def strip_by_run(frame, run_col, metrics, titles, placeholder=None):
    """One panel per metric; x = run, one point per row (e.g. per patient x moving
    slide). Synthetic points hollow grey, measured filled; a small deterministic
    jitter separates points of one run."""
    runs = list(dict.fromkeys(frame[run_col].astype(str)))
    pos = {r: i for i, r in enumerate(runs)}
    ph = (
        np.zeros(len(frame), dtype=bool)
        if placeholder is None
        else np.asarray(placeholder, dtype=bool)
    )
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(max(5.0, 0.35 * len(runs) + 1.5), 2.6 * len(metrics)),
        sharex=True,
        squeeze=False,
    )
    x = frame[run_col].astype(str).map(pos).to_numpy(dtype=float)
    x = x + (np.arange(len(frame)) % 7 - 3) * 0.05
    for ax, m, t in zip(axes[:, 0], metrics, titles):
        _scatter_points(ax, x, frame[m].to_numpy(dtype=float), ph, s=16)
        ax.set(ylabel=t)
    axes[0, 0].set_title("Registration accuracy per run (final stage)")
    if ph.any():
        axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xticks(range(len(runs)))
    axes[-1, 0].set_xticklabels(runs, rotation=60, ha="right", fontsize=7)
    return fig


def before_after_box(df, cols, ylabel, title, log_scale=True):
    fig, ax = plt.subplots()
    ax.boxplot([df[c].dropna().to_numpy() for c in cols])
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels(cols)
    if log_scale:
        ax.set_yscale("log")
    ax.set(ylabel=ylabel, title=title)
    return fig

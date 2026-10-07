"""Paper-ready matplotlib helpers: consistent theme + vector export (PDF+SVG).

Plot styles consolidated from notebooks/resources.ipynb (scaling scatter) and
notebooks/rTRE.ipynb (before/after boxplots).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

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


_TIER_ORDER = ("low", "medium", "high")
# Keyed on the plan's `method` (quality._family). `tiled` only appears for a plan without
# a `method` column, which is STARE.
_BACKEND_TITLE = {
    "valis": "VALIS",
    "stare": "STARE",
    "tiled": "STARE (tiled)",
}
_DEPTH_TITLE = {
    "valis": "micro-reg depth",
    "stare": "stride (px)",
    "tiled": "stride (px)",
}


def cost_by_tier(frame, metrics, ylabels):
    """Registration cost per backend x tier: one row of panels per metric, one column
    per backend, x = tier (low -> high), one marker series per refinement depth
    (VALIS reg_micro_reg, STARE reg_tiled_stride), dodged
    so depths never overlap. Rows share y, so the backends read on one scale."""
    backends = [b for b in _BACKEND_TITLE if b in set(frame["backend"])]
    fig, axes = plt.subplots(
        len(metrics),
        len(backends),
        figsize=(3.2 * len(backends) + 0.8, 2.3 * len(metrics)),
        sharey="row",
        squeeze=False,
    )
    markers = "osD^v"
    for j, b in enumerate(backends):
        sub = frame[frame["backend"] == b]
        depths = sorted(set(sub["depth"]), key=lambda d: (len(d), d))
        off = np.linspace(-0.18, 0.18, len(depths)) if len(depths) > 1 else [0.0]
        for i, (m, lab) in enumerate(zip(metrics, ylabels)):
            ax = axes[i, j]
            for k, d in enumerate(depths):
                dd = sub[sub["depth"] == d]
                x = [
                    _TIER_ORDER.index(t) + off[k] if t in _TIER_ORDER else np.nan
                    for t in dd["tier"]
                ]
                ax.plot(
                    x,
                    dd[m].to_numpy(dtype=float),
                    markers[k % len(markers)],
                    color=f"C{k}",
                    label=d,
                    ms=5,
                )
            ax.set_xticks(range(len(_TIER_ORDER)))
            ax.set_xticklabels(_TIER_ORDER)
            ax.set_xlim(-0.5, len(_TIER_ORDER) - 0.5)
            if j == 0:
                ax.set_ylabel(lab)
        axes[0, j].set_title(_BACKEND_TITLE.get(b, b))
        axes[0, j].legend(
            title=_DEPTH_TITLE.get(b, "depth"),
            fontsize=7,
            title_fontsize=7,
        )
        axes[-1, j].set_xlabel("tier")
    # A cost axis starts at zero, or tiers look further apart. Set ONCE PER ROW, after
    # every backend is drawn: set_ylim turns autoscaling off, and the row shares y, so
    # doing it inside the loop froze the axis at the first backend's range and every
    # larger value of a later backend was drawn outside the panel.
    for i in range(len(metrics)):
        axes[i, 0].set_ylim(bottom=0)
    return fig


# Okabe-Ito: distinguishable under the common colour-vision deficiencies and in grey.
_PHASE_COLOUR = {
    "REGISTER": "#0072B2",
    "TILED_COARSE": "#E69F00",
    "TILED_REG_TILE": "#56B4E9",
    "TILED_SOLVE": "#009E73",
    "TILED_STITCH": "#CC79A7",
}
_TOTAL_COLOUR = "#B8C4CE"
_PHASE_LABEL = {
    "REGISTER": "register (one task)",
    "TILED_COARSE": "coarse alignment",
    "TILED_REG_TILE": "tile registration",
    "TILED_SOLVE": "solve",
    "TILED_STITCH": "stitch",
}


def _arm_axes(arms, labels, legend: bool = False):
    # a legend sits in its own strip to the right of the axes, never over the data
    fig, ax = plt.subplots(
        figsize=(1.7 * len(arms) + 2.2 + (1.6 if legend else 0), 3.4)
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([(labels or {}).get(a, a) for a in arms])
    ax.set_xlim(-0.6, len(arms) - 0.4)
    return fig, ax


def _box(ax, x, values, width, colour):
    """One box of the patients' values: median line, quartile box, whiskers to the
    furthest patient within 1.5 IQR, patients beyond them as small open circles."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return
    ax.boxplot(
        [v],
        positions=[x],
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor=colour, edgecolor="#222222", linewidth=0.8),
        medianprops=dict(color="#222222", linewidth=1.4),
        whiskerprops=dict(color="#222222", linewidth=0.8),
        capprops=dict(color="#222222", linewidth=0.8),
        flierprops=dict(marker="o", ms=3, mfc="none", mec="#222222", mew=0.7),
        manage_ticks=False,
    )


def cost_by_phase(frame, phase_order, value, ylabel, labels=None):
    """One cost of a few arms side by side as boxplots over the patients, per phase.

    ``frame`` is quality.registration_cost_by_patient's long table restricted to the arms
    to draw (in the order of first appearance) and ``value`` one of its per-slide columns.
    Each arm has a wide grey box for the patients' TOTALS and, when it has more than one
    phase, a narrow coloured box per phase beside it; a one-phase arm's single box takes
    that phase's colour, being both.
    """
    arms = list(dict.fromkeys(frame["run_id"]))
    phases = [p for p in phase_order if p in set(frame["phase"])]
    colour = {p: _PHASE_COLOUR.get(p, f"C{i}") for i, p in enumerate(phases)}
    fig, ax = _arm_axes(arms, labels, legend=True)
    total_drawn = False
    for i, arm in enumerate(arms):
        sub = frame[frame["run_id"] == arm]
        mine = [p for p in phases if p in set(sub["phase"])]
        totals = sub.groupby("patient_id")[value].sum(min_count=1)
        if len(mine) <= 1:
            _box(ax, i, totals, 0.4, colour[mine[0]] if mine else _TOTAL_COLOUR)
            continue
        total_drawn = True
        slots = np.linspace(-0.36, 0.36, len(mine) + 1)
        w = 0.72 / (len(mine) + 1) * 0.8
        _box(ax, i + slots[0], totals, w, _TOTAL_COLOUR)
        for x, ph in zip(slots[1:], mine):
            _box(ax, i + x, sub[sub["phase"] == ph][value], w, colour[ph])
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colour[p], ec="#222222") for p in phases]
    names = [_PHASE_LABEL.get(p, p) for p in phases]
    if total_drawn:
        handles.insert(0, plt.Rectangle((0, 0), 1, 1, fc=_TOTAL_COLOUR, ec="#222222"))
        names.insert(0, "total (all phases)")
    ax.legend(
        handles=handles,
        labels=names,
        fontsize=7,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    return fig


def cost_peak_memory(frame, ylabel, labels=None):
    """The largest single task's peak memory per arm, as a boxplot over the patients."""
    arms = list(dict.fromkeys(frame["run_id"]))
    per = frame.groupby(["run_id", "patient_id"])["peak_rss_gb"].max().reset_index()
    fig, ax = _arm_axes(arms, labels)
    for i, arm in enumerate(arms):
        _box(ax, i, per[per["run_id"] == arm]["peak_rss_gb"], 0.4, _TOTAL_COLOUR)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


_ARM_COLOUR = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def time_on_cores(frame, labels=None):
    """Estimated time to register one patient against the cores available.

    ``frame`` is quality.time_on_cores: per arm the line is the MEDIAN patient and the
    band spans the patients (min to max). Both axes are logarithmic: a method whose time
    halves when the cores double falls on a straight diagonal, a method that cannot use
    more cores is flat.
    """
    arms = list(dict.fromkeys(frame["run_id"]))
    fig, ax = plt.subplots(figsize=(4.6 + 1.9, 3.4))
    ax.spines[["top", "right"]].set_visible(False)
    for i, arm in enumerate(arms):
        wide = frame[frame["run_id"] == arm].pivot(
            index="n_cores", columns="patient_id", values="est_hours"
        )
        wide = wide.dropna(how="all")
        if wide.empty:
            continue
        c = _ARM_COLOUR[i % len(_ARM_COLOUR)]
        x = wide.index.to_numpy(dtype=float)
        ax.fill_between(
            x, wide.min(axis=1), wide.max(axis=1), color=c, alpha=0.18, lw=0
        )
        ax.plot(
            x,
            wide.median(axis=1),
            "-o",
            color=c,
            ms=4,
            label=(labels or {}).get(arm, arm).replace("\n", " "),
        )
    cores = sorted(set(frame["n_cores"]))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(cores)
    ax.set_xticklabels([str(c) for c in cores])
    # hours written out (0.5, 1, 2, 5): a log axis left to itself labels only 10^0
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("CPU cores available")
    ax.set_ylabel("Estimated time per patient (hours)")
    ax.legend(
        fontsize=7,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="line: median patient\nband: all patients",
        title_fontsize=6,
    )
    fig.tight_layout()
    return fig

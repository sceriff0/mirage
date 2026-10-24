"""What the publication figures share: the method labels, the box style of Fig. 3, the
paired test, the choice of a representative case, and the image display.

Every number a figure shows is read from the regbench tables or recomputed with the
harness's own code (``benchmarks.regbench.pairs``); nothing here re-defines a score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotstyle as ps

# Locked in this order in every figure, so STARE keeps one colour across the paper.
HIGHLIGHT = ("VALIS", "STARE")
INITIAL = "initial"  # regbench's unregistered pose
# The two image channels: an image's own colours (reference cyan, moving magenta), not
# categories, so they do not come from the palette.
CYAN, MAGENTA = (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)


def palette(figdir):
    """``{label: colour}`` of the highlighted method; every other method is grey."""
    return ps.lock_palette(list(HIGHLIGHT), figdir=str(figdir))


def point_colour(label, colors):
    return colors["STARE"] if label == "STARE" else ps.LIGHT_GREY


def parse_methods(spec):
    """``"Unregistered=initial,VALIS=valis120_hires,STARE=stare"`` -> ``{label: variant}``,
    in the order given (the order of the boxes)."""
    out = {}
    for part in spec.split(","):
        label, _, variant = part.partition("=")
        if not variant:
            raise SystemExit(f"--methods: {part!r} is not LABEL=variant")
        out[label.strip()] = variant.strip()
    return out


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


# ── statistics ────────────────────────────────────────────────────────────────
def wilcoxon_holm(wide, against, others):
    """Paired two-sided Wilcoxon signed-rank of every ``others`` column against
    ``against`` on the rows both have, Holm-adjusted across the comparisons.

    Returns one dict per comparison: n, median difference, raw and adjusted p."""
    from scipy.stats import wilcoxon

    rows = []
    for o in others:
        both = wide[[against, o]].dropna()
        d = both[against] - both[o]
        p = math.nan
        if len(both) >= 2 and (d != 0).any():
            p = float(wilcoxon(both[against], both[o], alternative="two-sided").pvalue)
        rows.append(
            {
                "against": against,
                "method": o,
                "n_pairs": len(both),
                "median_difference": float(d.median()) if len(d) else math.nan,
                "p": p,
            }
        )
    # Holm: the k-th smallest p times (m - k + 1), kept monotone
    order = sorted(
        (i for i in range(len(rows)) if rows[i]["p"] == rows[i]["p"]),
        key=lambda i: rows[i]["p"],
    )
    m, running = len(order), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, rows[i]["p"] * (m - k)))
        rows[i]["p_holm"] = running
    for r in rows:
        r.setdefault("p_holm", math.nan)
        r["stars"] = stars(r["p_holm"])
    return rows


def stars(p):
    if p != p:
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# ── the boxes of Fig. 3 ───────────────────────────────────────────────────────
BOX = dict(color="black", linewidth=0.7)


def fmt_median(v):
    if v != v:
        return "–"
    return f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"


def boxes(ax, groups, colors, rng, width=0.6, label_fmt=fmt_median, log=False):
    """One box per ``(x, label, values)``: black outline, no fliers, every value a
    jittered point (STARE in its colour, the rest grey), the median under the box.
    Returns the plotted points (for the source data)."""
    drawn = []
    for x, label, values in groups:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v) & ((v > 0) if log else True)]
        if len(v):
            ax.boxplot(
                v,
                positions=[x],
                widths=width,
                showfliers=False,
                boxprops=BOX,
                medianprops=dict(color="black", linewidth=0.9),
                whiskerprops=BOX,
                capprops=BOX,
            )
            jx = x + rng.uniform(-width * 0.3, width * 0.3, len(v))
            ax.scatter(
                jx, v, s=5, color=point_colour(label, colors), zorder=3, linewidths=0
            )
            drawn += [
                {"x": float(a), "label": label, "value": float(b)}
                for a, b in zip(jx, v)
            ]
        ax.annotate(
            label_fmt(float(np.median(v))) if len(v) else "–",
            (x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5,
            color=ps.GREY,
        )
    return drawn


def brackets(ax, xs, tests, top, step, log=False):
    """Fig. 3's brackets: from each compared box to the reference box, stacked upward,
    the Holm-adjusted stars on each. ``top`` is the first bracket's height."""
    y = top
    # the shortest bracket lowest, as in Fig. 3
    for t in sorted(tests, key=lambda r: abs(xs[r["against"]] - xs[r["method"]])):
        if not t["stars"]:
            continue
        x0, x1 = sorted((xs[t["method"]], xs[t["against"]]))
        tick = y / 1.04 if log else y - step * 0.3
        ax.plot([x0, x0, x1, x1], [tick, y, y, tick], color="black", lw=0.6)
        ax.text((x0 + x1) / 2, y, t["stars"], ha="center", va="bottom", fontsize=5)
        y = y * step if log else y + step
    return y


def plain_ticks(ax, log, data_max=None):
    """Fig. 3's ticks: on a log axis the 1-2-5 values as plain numbers (0.5, 1, 2, 5,
    10, ...); on a linear one no tick above the data (the brackets' headroom is not a
    value)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    lo, hi = ax.get_ylim()
    if log:
        vals = [
            m * 10.0**e
            for e in range(
                math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1
            )
            for m in (1, 2, 5)
        ]
        vals = [v for v in vals if lo <= v <= hi]
        if len(vals) > 8:  # a wide range: 1-3-10 like Fig. 3e
            vals = [v for v in vals if str(v).lstrip("0.").startswith(("1", "3", "5"))]
        ax.yaxis.set_major_locator(FixedLocator(vals))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    elif data_max is not None:
        ticks = [t for t in ax.get_yticks() if lo <= t <= data_max * 1.0001]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_ylim(lo, hi)


def room_for_medians(ax, log, frac=0.08):
    lo, hi = ax.get_ylim()
    if log:
        ax.set_ylim(lo / (hi / lo) ** frac, hi)
    else:
        ax.set_ylim(lo - (hi - lo) * frac, hi)


# ── which case is representative ──────────────────────────────────────────────
def representative(wide, methods, by="group", min_rounds=1):
    """The unit (a patient: ``by="group"``, or a case) whose scores are closest to the
    cohort median of EVERY method in ``methods``: for each case, the sum over methods of
    ``|score - median| / IQR``; per patient the mean over its rounds. Patients with
    fewer than ``min_rounds`` scored rounds are passed over when any other has enough.
    Deterministic: ties go to the smallest id. Returns ``(id, table)``."""
    sub = wide.dropna(subset=list(methods)).copy()
    if sub.empty:
        raise SystemExit(f"no case was scored for all of {list(methods)}")
    dist = 0.0
    for m in methods:
        q1, med, q3 = np.percentile(sub[m], [25, 50, 75])
        dist = dist + (sub[m] - med).abs() / max(q3 - q1, 1e-9)
    sub["distance_to_median"] = dist
    table = (
        sub.groupby(by)
        .agg(distance_to_median=("distance_to_median", "mean"), n_rounds=(by, "size"))
        .reset_index()
    )
    enough = table[table["n_rounds"] >= min_rounds]
    pool = enough if len(enough) else table
    pool = pool.sort_values(["distance_to_median", by])
    return str(pool.iloc[0][by]), table.sort_values("distance_to_median")


# ── regbench reading ──────────────────────────────────────────────────────────
def cells_table(tables, dataset, which="outlines"):
    df = pd.read_csv(Path(tables) / "cells.csv", low_memory=False)
    df = df[(df["dataset"].astype(str) == dataset) & (df["set"].astype(str) == which)]
    return df[~df["excluded_from_summaries"].map(truthy)]


def run_json(out, variant, case):
    from benchmarks.regbench.cases import result_dir

    p = result_dir(out, variant, case) / "run.json"
    return json.loads(p.read_text()) if p.exists() else {}


def written_nuclear(out, variant, case):
    """``(path, channel, origin_xy)`` of a result's written image, as the tissue scorer
    reads it (``regbench.tissue.reduce_native``): the image's pixel (x, y) is the
    reference canvas' (x + ox, y + oy)."""
    from stare.slide_io import open_lazy

    from benchmarks.regbench.tissue import native_channel

    run = run_json(out, variant, case)
    native = run.get("native") or {}
    files = native.get("files") or []
    path = native.get("nuclear_file") or (files[0] if files else None)
    if not run.get("ok") or not path or not Path(path).exists():
        raise SystemExit(
            f"{variant} has no written image for case {case.case_id} "
            f"(ok={run.get('ok')}, file={path})"
        )
    arr, _, close = open_lazy(path)
    try:
        channel = native_channel(native, case, int(arr.shape[0]))
    finally:
        close()
    ox, oy = (float(v) for v in (native.get("origin_xy") or (0.0, 0.0)))
    return path, channel, (ox, oy)


# ── images ────────────────────────────────────────────────────────────────────
def region(arr, channel, y0, x0, h, w):
    """An ``h`` x ``w`` window of one channel of a lazy ``(C, H, W)`` array at (y0, x0),
    zero outside the image."""
    out = np.zeros((h, w), dtype=np.float32)
    H, W = arr.shape[1:]
    ya, xa, yb, xb = max(y0, 0), max(x0, 0), min(y0 + h, H), min(x0 + w, W)
    if yb > ya and xb > xa:
        out[ya - y0 : yb - y0, xa - x0 : xb - x0] = np.asarray(
            arr[channel, ya:yb, xa:xb]
        )
    return out


def stretch(a, clip=(1.0, 99.8), mask=None):
    """Linear window to the ``clip`` percentiles (of ``mask``, else of the non-zero
    pixels), to [0, 1]."""
    v = a[mask] if mask is not None and mask.any() else a[a > 0]
    if not v.size:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(v, clip)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)


def crisp(a, clip=(1.0, 99.8), clahe=0.01, sharpen=0.6, upsample=3):
    """The zoom display: the linear window computed inside the zoom, CLAHE (clip limit
    ``clahe``, tile 1/8 of the zoom), an unsharp mask (radius 1 px, ``sharpen``), then a
    cubic ``upsample`` so the PDF viewer's own smoothing does not blur the nuclei.
    Each step is off at 0. A DISPLAY adjustment: the legend must state it."""
    from scipy.ndimage import zoom

    out = stretch(a, clip)
    if clahe > 0:
        from skimage.exposure import equalize_adapthist

        k = max(min(out.shape) // 8, 8)
        out = equalize_adapthist(out, kernel_size=k, clip_limit=clahe).astype(
            np.float32
        )
    if sharpen > 0:
        from skimage.filters import unsharp_mask

        out = unsharp_mask(out, radius=1.0, amount=sharpen, preserve_range=True)
    if upsample > 1:
        out = zoom(out, upsample, order=3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fixed_figure(width_mm, height_mm):
    """A figure whose axes are placed by hand, in figure fractions: no layout engine.
    ``layout="none"`` is not enough under plotstyle's rcParams: the second save falls back
    to constrained layout, which moves nothing here but warns on every file."""
    from matplotlib.layout_engine import PlaceHolderLayoutEngine

    fig = ps.plt.figure(figsize=(width_mm * ps.MM, height_mm * ps.MM))
    fig.set_layout_engine(
        PlaceHolderLayoutEngine(adjust_compatible=True, colorbar_gridspec=True)
    )
    return fig


def composite(ref, mov):
    """Reference cyan, moving magenta, grey-to-white where they agree: (R, G, B) =
    (moving, reference, the brighter of the two). A plain sum of cyan and magenta would
    double the blue and draw agreement lavender."""
    return np.clip(np.stack([mov, ref, np.maximum(ref, mov)], axis=-1), 0.0, 1.0)


def on_canvas(a, hw):
    """``a`` padded or cropped at the origin to ``hw``: the unregistered placement."""
    out = np.zeros(hw, dtype=np.float32)
    h, w = min(hw[0], a.shape[0]), min(hw[1], a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def nice_length(target):
    """The 1-2-5 length at or below ``target``."""
    e = 10 ** math.floor(math.log10(target))
    return max(m * e for m in (1, 2, 5) if m * e <= target)


def scale_bar(ax, img_w, um_per_px, frac=0.2, pad=0.05, length_um=None):
    """A white bar in the bottom-left corner, its length above it."""
    length = length_um or nice_length(img_w * um_per_px * frac)
    px = length / um_per_px
    h = ax.get_ylim()[0]  # imshow: ylim = (H, 0)
    x0, y = img_w * pad, h * (1 - pad)
    ax.plot([x0, x0 + px], [y, y], color="white", lw=1.2, solid_capstyle="butt")
    text = f"{length / 1000:g} mm" if length >= 1000 else f"{length:g} µm"
    ax.text(
        x0 + px / 2,
        y - h * 0.015,
        text,
        color="white",
        ha="center",
        va="bottom",
        fontsize=ps.TICK_PT,
    )
    return length

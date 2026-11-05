"""Peak memory and CPU time against input size, per patient: VALIS vs STARE.

Two panels in the style of Fig. 3d-e: at each input size one box per method (black
outline, every patient a jittered point, STARE in its colour, VALIS grey, the median under
the box), log y.

    python -m benchmarks.figures.fig_synthetic_cost <REGBENCH_OUT>/tables \
        --valis valis120_hires --stare stare --outdir figures/cost

The benchmark has no synthetic set any more (``regbench/paper/collect.py``: the synthetic
sweep is removed from the paper), so the default ``--dataset`` is the cohort, ``multiplex``:
its patients differ in size by nature, and are log-binned below.

READS the scorer's own tables, nothing re-derived:

  summary_resources.csv  one row per (dataset, patient, method): ``peak_rss_gb`` (the max
                         over its rounds), ``cpu_h_used`` (measured CPU time, the paper's
                         headline: the sum over rounds for a pair method, the launch total
                         for a per-patient one), ``complete``
  resources.csv          one row per round: ``moving_gpx`` of EVERY round, ok or not, so a
                         patient that failed still has a size

x = the patient's whole workload: the sum of its rounds' moving pixels (fig. 2c's
``input_gpx``): each distinct size its own box when there are at most ``--max-levels`` of
them (a designed size series), otherwise ``--max-levels`` log-spaced bins (a cohort).

ONLY COMPLETE PATIENTS ARE DRAWN: the peak of a run that died is where it died, not what
the registration needs. The others are COUNTED at the top of their size ("3 failed"),
with their causes on stdout and in <stem>_failed.csv: never hidden.

Outputs: <outdir>/cost_vs_size_<dataset>.pdf/.png/.svg, _source_data.csv, _failed.csv.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps

PANELS = (
    ("peak_rss_gb", "Peak RSS (GB, log10)"),
    ("cpu_h_used", "CPU time per patient (h, log10)"),
)
METHODS = ("VALIS", "STARE")


def size_label(gpx):
    """The side of a square image of that many pixels: '4k²', or '900²' below 1024."""
    side = math.sqrt(gpx * 1e9)
    return f"{side / 1024:.0f}k²" if side >= 1024 else f"{side:.0f}²"


def _read(folder, dataset):
    summ = pd.read_csv(Path(folder) / "summary_resources.csv", low_memory=False)
    res = pd.read_csv(Path(folder) / "resources.csv", low_memory=False)
    for df in (summ, res):
        for c in ("dataset", "method", "group"):
            df[c] = df[c].astype(str)
    return summ[summ["dataset"] == dataset], res[res["dataset"] == dataset]


def load(tables, dataset, names):
    """Every method from the FIRST of ``tables`` that has it (its per-patient summary and
    its per-round sizes from that one folder), so methods scored in different
    submissions can share a figure."""
    read = [
        (str(t), *_read(t, dataset))
        for t in ([tables] if isinstance(tables, str) else tables)
    ]
    parts_s, parts_r, have = [], [], {}
    for label, name in names.items():
        src = next(((t, s, r) for t, s, r in read if (s["method"] == name).any()), None)
        if src is None:
            seen = sorted({m for _, s, _ in read for m in s["method"]})
            raise SystemExit(
                f"{label} = {name!r} has no {dataset} row in summary_resources.csv of "
                f"{[t for t, _, _ in read]}; there are: {seen}"
            )
        t, s, r = src
        have[name] = t
        parts_s.append(s[s["method"] == name])
        parts_r.append(r[r["method"] == name])
        print(f"{label} = {name}: {dataset} rows from {t}")
    summ, res = pd.concat(parts_s), pd.concat(parts_r)
    res["moving_gpx"] = pd.to_numeric(res["moving_gpx"], errors="coerce")
    size = res.groupby(["method", "group"])["moving_gpx"].sum(min_count=1)
    cause = (
        res[~res["ok"].map(C.truthy)]
        .groupby(["method", "group"])["cause"]
        .agg(lambda s: ";".join(sorted(set(map(str, s)))))
    )
    rows = []
    for label, name in names.items():
        for _, r in summ[summ["method"] == name].iterrows():
            key = (name, r["group"])
            rows.append(
                {
                    "method": label,
                    "variant": name,
                    "patient": r["group"],
                    "size_gpx": size.get(key, np.nan),
                    "complete": C.truthy(r["complete"]),
                    "cause": cause.get(key, ""),
                    **{k: pd.to_numeric(r.get(k), errors="coerce") for k, _ in PANELS},
                }
            )
    return pd.DataFrame(rows)


def levels(df, max_levels):
    """The size category of every row: the design sizes, or log bins past max_levels."""
    sizes = np.sort(df["size_gpx"].dropna().unique())
    if len(sizes) <= max_levels:
        lab = {s: size_label(s) for s in sizes}
        df["level"] = df["size_gpx"].map(lab)
        return list(dict.fromkeys(lab[s] for s in sizes))
    edges = np.geomspace(sizes.min(), sizes.max() * 1.0001, max_levels + 1)
    idx = np.clip(np.digitize(df["size_gpx"], edges) - 1, 0, max_levels - 1)
    names = [f"{edges[i]:.2g}–{edges[i + 1]:.2g} Gpx" for i in range(max_levels)]
    # a patient without a size is in no bin (digitize would put NaN in the last one)
    df["level"] = [
        names[i] if np.isfinite(v) else None for i, v in zip(idx, df["size_gpx"])
    ]
    return [n for n in names if (df["level"] == n).any()]


def draw(df, order, out, colors, seed):
    rng = np.random.default_rng(seed)
    fig, axes = ps.figure(width="double", nrows=1, ncols=2, aspect=0.8)
    width, gap = 0.32, 0.1
    drawn = []
    for ax, (key, ylabel) in zip(axes, PANELS):
        groups, failed = [], []
        for i, lev in enumerate(order):
            for j, method in enumerate(METHODS):
                x = i + (j - 0.5) * (width + gap)
                here = df[(df["level"] == lev) & (df["method"] == method)]
                groups.append((x, method, here.loc[here["complete"], key]))
                failed.append((x, int((~here["complete"]).sum())))
        pts = C.boxes(ax, groups, colors, rng, width=width, log=True)
        drawn += [{**p, "panel": key} for p in pts]
        for x, n in failed:
            if n:
                ax.annotate(
                    f"{n} failed",
                    (x, 1),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=4.5,
                    color=ps.GREY,
                    rotation=90,
                )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(order)))
        counts = [
            df[df["level"] == lev].groupby("method")["patient"].nunique()
            for lev in order
        ]
        ax.set_xticklabels(
            [
                f"{lev}\n{int(n.get('VALIS', 0))} | {int(n.get('STARE', 0))}"
                for lev, n in zip(order, counts)
            ]
        )
        ax.set_xlabel("Moving pixels per patient\n(n patients: VALIS | STARE)")
        ax.set_xlim(-0.6, len(order) - 0.4)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 6.0)  # room for the failure counts
        C.room_for_medians(ax, True)
        C.plain_ticks(ax, True)
    handles = [
        ps.plt.Line2D([], [], ls="", marker="o", ms=3, color=C.point_colour(m, colors))
        for m in METHODS
    ]
    fig.legend(
        handles,
        METHODS,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    ps.panel_labels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ps.save(fig, out, data=pd.DataFrame(drawn))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "tables",
        nargs="+",
        help="<REGBENCH_OUT>/tables; several: each method from the first that has it",
    )
    ap.add_argument(
        "--valis", required=True, help="the VALIS variant, e.g. valis120_hires"
    )
    ap.add_argument("--stare", default="stare")
    ap.add_argument("--dataset", default="multiplex")
    ap.add_argument("--max-levels", type=int, default=8)
    ap.add_argument("--outdir", default="figures/cost")
    ap.add_argument("--seed", type=int, default=0, help="the jitter's seed")
    a = ap.parse_args(argv)

    ps.apply_style()
    colors = C.palette(a.outdir)
    df = load(a.tables, a.dataset, {"VALIS": a.valis, "STARE": a.stare})
    order = levels(df, a.max_levels)
    unsized = df[df["level"].isna()]
    if len(unsized):
        print(
            f"{unsized['patient'].nunique()} patient(s) without a moving_gpx in "
            f"resources.csv, in no size bin: {sorted(unsized['patient'].unique())}"
        )
    for method in METHODS:
        m = df[df["method"] == method]
        bad = m[~m["complete"]]
        causes = ", ".join(sorted(set(bad["cause"]) - {""}))
        print(
            f"{method} ({m['variant'].iloc[0]}): {m['patient'].nunique()} patient(s), "
            f"{len(m) - len(bad)} complete, {len(bad)} failed"
            + (f" ({causes})" if causes else "")
        )
    out = Path(a.outdir) / f"cost_vs_size_{a.dataset}"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[~df["complete"]].to_csv(f"{out}_failed.csv", index=False)
    draw(df, order, str(out), colors, a.seed)
    print(f"-> {out}.pdf/.png/.svg, _source_data.csv, _failed.csv")


if __name__ == "__main__":
    main()

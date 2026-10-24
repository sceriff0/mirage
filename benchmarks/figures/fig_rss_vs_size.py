"""Peak RSS against slide size: STARE per task, VALIS per slide, on the synthetic size
series, with the cohort's slides at native size overlaid and the node's memory drawn.

    python -m benchmarks.figures.fig_rss_vs_size \
        --synthetic <SYNTH_OUT>/tables --cohort <REGBENCH_OUT>/tables \
        --valis valis120 --stare stare --node-gb 512 --outdir figures/cost

READS ``resources.csv`` (one row per round), nothing re-derived:

  peak_rss_gb  STARE: the peak of its LARGEST single stage job (COARSE, one REG_TILE per
               tile, SOLVE are separate scheduler jobs, so their peaks do not add;
               ``largest_job`` names it): the per-task peak. VALIS: the peak of the
               process that registered the slide (one launch).
  moving_gpx   the moving slide's pixels (one plane), so x = sqrt(moving_gpx) is the
               side of the square slide of that size: exact on the synthetic squares,
               the equivalent side on the cohort's rectangles.

On the synthetic set a case named ``scale_<n>`` is taken at its design size ``n``; any
other synthetic case at sqrt(moving_gpx). Sizes outside ``--min-px``..``--max-px`` are
left out. A per-patient launch (``unit == patient``: several rounds, ONE peak) is one
point, at its largest slide.

FAILED VALIS RUNS ARE DRAWN, with their own symbol (an x), never dropped: the figure's
claim is where VALIS stops fitting the node. A run that died of memory (``cause``
containing ``oom``/``mem``) or left no peak is drawn ON the node-memory line (it needed
at least that); any other failure at the peak it recorded. ``--failed-at peak`` draws
every one at its recorded peak. A failed STARE round is counted on stdout and drawn the
same way.

Cohort points (``--cohort``) are open symbols in the method's colour. Each method is
taken from the FIRST tables folder that has it, as in fig_synthetic_cost.

Prints the numbers the legend needs (n per size, STARE's range, the smallest size where
VALIS fails or crosses the node memory). Outputs: <outdir>/rss_vs_size.pdf/.png/.svg,
_source_data.csv (every point drawn, with its cause), _caption.txt.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps

MEMORY = re.compile(r"oom|mem", re.I)


def _resources(folder):
    p = Path(folder)
    p = p if p.suffix == ".csv" else p / "resources.csv"
    df = pd.read_csv(p, low_memory=False)
    for c in ("dataset", "method", "group", "case_id"):
        if c in df:
            df[c] = df[c].astype(str)
    df["source"] = str(p)
    return df


def pick(folders, dataset, names):
    """``{label: rows}``: each variant's rows of ``dataset`` from the first folder with
    any; ``names`` maps a label to comma-separated alternatives, first found wins."""
    read = [_resources(f) for f in folders]
    out = {}
    for label, alts in names.items():
        for name in [n.strip() for n in alts.split(",") if n.strip()]:
            hit = next(
                (
                    r[(r["dataset"] == dataset) & (r["method"] == name)]
                    for r in read
                    if ((r["dataset"] == dataset) & (r["method"] == name)).any()
                ),
                None,
            )
            if hit is not None:
                print(
                    f"{label} = {name}: {len(hit)} {dataset} rows from {hit['source'].iloc[0]}"
                )
                out[label] = hit.assign(label=label, variant=name)
                break
        else:
            seen = sorted(
                {m for r in read for m in r.loc[r["dataset"] == dataset, "method"]}
            )
            print(f"{label}: none of {alts!r} has a {dataset} row; there are: {seen}")
    return out


def points(rows, design_sizes):
    """One row per point: per round, or per launch for a per-patient launch."""
    d = rows.copy()
    d["ok"] = d["ok"].map(C.truthy)
    d["peak_rss_gb"] = pd.to_numeric(d["peak_rss_gb"], errors="coerce")
    d["moving_gpx"] = pd.to_numeric(d["moving_gpx"], errors="coerce")
    side = np.sqrt(d["moving_gpx"] * 1e9)
    if design_sizes:
        n = d["case_id"].str.extract(r"scale_0*(\d+)", expand=False).astype(float)
        side = n.fillna(side)
    d["side_px"] = side
    d["cause"] = d["cause"].fillna("").astype(str)
    if "largest_job" not in d:
        d["largest_job"] = ""
    unit = d.get("unit", pd.Series("", index=d.index)).fillna("").astype(str)
    launch = d.get("launch_id", pd.Series("", index=d.index)).fillna("").astype(str)
    d["key"] = np.where(
        (unit == "patient") & (launch != ""), "L:" + launch, "R:" + d["case_id"]
    )
    agg = d.groupby("key").agg(
        label=("label", "first"),
        variant=("variant", "first"),
        dataset=("dataset", "first"),
        group=("group", "first"),
        case_id=("case_id", lambda s: ";".join(sorted(set(s)))),
        side_px=("side_px", "max"),
        peak_rss_gb=("peak_rss_gb", "max"),
        ok=("ok", "all"),
        cause=("cause", lambda s: ";".join(sorted(set(s) - {""}))),
        largest_job=("largest_job", "first"),
        source=("source", "first"),
    )
    return agg.reset_index(drop=True)


def place_failed(df, node_gb, mode):
    """Where a failed run is drawn (``y``) and why (``y_rule``)."""
    df["y"] = df["peak_rss_gb"]
    df["y_rule"] = np.where(df["ok"], "measured", "recorded peak at failure")
    if mode == "node" and node_gb:
        hit = ~df["ok"] & (
            df["cause"].str.contains(MEMORY) | ~np.isfinite(df["peak_rss_gb"])
        )
        df.loc[hit, "y"] = node_gb
        df.loc[hit, "y_rule"] = "memory failure: drawn at node memory"
    return df


def caption_numbers(syn, node_gb):
    lines = []
    for label, d in syn.groupby("label"):
        n = d.groupby("side_px").size()
        lines.append(
            f"{label} n per size: "
            + ", ".join(f"{int(s)} px: {k}" for s, k in n.items())
        )
    st = syn[(syn["label"] == "STARE") & syn["ok"]]["peak_rss_gb"].dropna()
    if len(st):
        lines.append(f"STARE peak RSS per task: {st.min():.2f}-{st.max():.2f} GB")
    va = syn[syn["label"] == "VALIS"]
    if len(va):
        bad = va[~va["ok"]]
        over = va[va["ok"] & (va["peak_rss_gb"] > node_gb)] if node_gb else va.iloc[0:0]
        first_fail = bad["side_px"].min() if len(bad) else math.nan
        first_over = over["side_px"].min() if len(over) else math.nan
        lines.append(
            f"VALIS: first size with a failed run {first_fail:.0f} px "
            f"({', '.join(sorted(set(bad['cause']) - {''})) or 'no cause recorded'}); "
            f"first size measured above the node memory {first_over:.0f} px"
        )
        ok = va[va["ok"]].groupby("side_px")["peak_rss_gb"].median()
        lines.append(
            "VALIS median peak per size: "
            + ", ".join(f"{int(s)} px: {g:.1f} GB" for s, g in ok.items())
        )
    return lines


def draw(syn, coh, colors, node_gb, out, logy):
    fig, axes = ps.figure(width="single", aspect=0.85)
    ax = axes[0]
    col = {"STARE": colors["STARE"], "VALIS": ps.GREY}
    drawn = []
    for label in ("VALIS", "STARE"):
        d = syn[syn["label"] == label]
        if d.empty:
            continue
        ok, bad = d[d["ok"]], d[~d["ok"]]
        jit = 2 ** (np.random.default_rng(0).uniform(-0.06, 0.06, len(ok)))
        ax.scatter(
            ok["side_px"] * jit,
            ok["y"],
            s=10,
            color=col[label],
            alpha=0.7,
            lw=0,
            zorder=3,
        )
        med = ok.groupby("side_px")["y"].median()
        ax.plot(med.index, med.values, color=col[label], lw=0.8, zorder=2)
        if len(bad):
            ax.scatter(
                bad["side_px"],
                bad["y"],
                s=18,
                marker="x",
                color=col[label],
                lw=0.8,
                zorder=4,
            )
        drawn.append(d.assign(layer="synthetic"))
    for label in ("VALIS", "STARE"):
        d = coh[coh["label"] == label] if len(coh) else coh
        if d is None or d.empty:
            continue
        ok, bad = d[d["ok"]], d[~d["ok"]]
        ax.scatter(
            ok["side_px"],
            ok["y"],
            s=16,
            marker="D",
            facecolor="none",
            edgecolor=col[label],
            lw=0.7,
            zorder=5,
        )
        if len(bad):
            ax.scatter(
                bad["side_px"],
                bad["y"],
                s=18,
                marker="x",
                color=col[label],
                lw=0.8,
                zorder=5,
            )
        drawn.append(d.assign(layer="cohort"))
    if node_gb:
        ax.axhline(node_gb, ls="--", lw=0.6, color=ps.GREY, zorder=1)
        ax.text(
            0.8,
            node_gb,
            f"node memory, {node_gb:g} GB",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="right",
            fontsize=ps.TICK_PT,
            color=ps.GREY,
        )
    ax.set_xscale("log", base=2)
    sizes = sorted(syn["side_px"].dropna().unique()) if len(syn) else []
    if sizes:
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{int(s):,}" for s in sizes], rotation=45, ha="right")
        ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
        C.plain_ticks(ax, True)
    else:
        ax.set_ylim(bottom=0)
    if node_gb:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, node_gb * (1.6 if logy else 1.1)))
    ax.set_xlabel("Slide size (px per side)")
    ax.set_ylabel("Peak RSS (GB" + (", log10)" if logy else ")"))
    L = ps.plt.Line2D
    handles = [
        L([], [], ls="-", marker="o", ms=3, lw=0.8, color=col["STARE"]),
        L([], [], ls="-", marker="o", ms=3, lw=0.8, color=col["VALIS"]),
        L([], [], ls="", marker="D", ms=3, mfc="none", mec=ps.GREY, mew=0.7),
        L([], [], ls="", marker="x", ms=4, color=ps.GREY, mew=0.8),
    ]
    names = ["STARE (per task)", "VALIS (per slide)", "cohort slides", "failed run"]
    ax.legend(
        handles,
        names,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.9),
        frameon=False,
        fontsize=ps.TICK_PT,
    )
    data = pd.concat(drawn, ignore_index=True) if drawn else pd.DataFrame()
    ps.save(fig, out, data=data)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--synthetic",
        nargs="+",
        required=True,
        help="tables folder(s) or resources.csv holding the synthetic set",
    )
    ap.add_argument(
        "--cohort", nargs="*", default=[], help="tables folder(s) of the cohort"
    )
    ap.add_argument("--synthetic-dataset", default="synthetic")
    ap.add_argument("--cohort-dataset", default="multiplex")
    ap.add_argument(
        "--valis",
        default="valis120,valis,valis_mirage",
        help="VALIS variant; comma-separated alternatives, first found wins",
    )
    ap.add_argument("--stare", default="stare", help="STARE variant (high tier)")
    ap.add_argument("--node-gb", type=float, help="node memory: the dashed line")
    ap.add_argument("--min-px", type=float, default=4096)
    ap.add_argument("--max-px", type=float, default=65536)
    ap.add_argument("--failed-at", choices=("node", "peak"), default="node")
    ap.add_argument("--linear-y", action="store_true")
    ap.add_argument("--outdir", default="figures/cost")
    a = ap.parse_args(argv)

    ps.apply_style()
    colors = C.palette(a.outdir)
    names = {"VALIS": a.valis, "STARE": a.stare}
    syn = pick(a.synthetic, a.synthetic_dataset, names)
    if not syn:
        raise SystemExit("no synthetic rows for either method")
    syn = pd.concat([points(r, True) for r in syn.values()], ignore_index=True)
    keep = syn["side_px"].between(a.min_px * 0.99, a.max_px * 1.01)
    if (~keep).any():
        print(
            f"left out {int((~keep).sum())} synthetic point(s) outside {a.min_px:g}-{a.max_px:g} px"
        )
    syn = place_failed(syn[keep].copy(), a.node_gb, a.failed_at)
    coh = pd.DataFrame()
    if a.cohort:
        c = pick(a.cohort, a.cohort_dataset, names)
        if c:
            coh = place_failed(
                pd.concat([points(r, False) for r in c.values()], ignore_index=True),
                a.node_gb,
                a.failed_at,
            )
            print(f"cohort: {len(coh)} point(s), {coh['group'].nunique()} patient(s)")
    if not a.node_gb:
        print(
            "no --node-gb: no node-memory line, and failed runs at their recorded peak"
        )
    out = Path(a.outdir) / "rss_vs_size"
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(syn, coh, colors, a.node_gb, str(out), not a.linear_y)
    lines = caption_numbers(syn, a.node_gb)
    if len(coh):
        lines.append(
            f"cohort: {len(coh)} points; sides {coh['side_px'].min():.0f}-{coh['side_px'].max():.0f} px (equivalent square)"
        )
    Path(f"{out}_caption.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}.pdf/.png/.svg, _source_data.csv, _caption.txt")


if __name__ == "__main__":
    main()

"""Fig. 2c of the manuscript, as published: peak RSS against synthetic image size, STARE
vs VALIS 1.2.0 (its defaults and the high-resolution example of its documentation).

    python benchmarks/figures/fig2c_peak_rss.py <REGBENCH_OUT>/tables/synthetic <out dir> [--no-hires]

Reads ``stare_stage_peaks.csv`` (STARE: the largest of anchor, tile task and slide-wide
solve, over all rounds of a synthetic patient) and ``synthetic_cost.csv`` (VALIS: the peak of
the one run that registers all rounds; rows ``valis120`` and ``valis120_hires``). The x axis is
log2, the y axis LINEAR; the dashed line marks 200 GB. ``--no-hires`` leaves out the
high-resolution example and writes ``Figure_2c_nohires.*``. Writes ``Figure_2c.{pdf,png,svg}``
and ``Figure_2c_source_data.csv``.

Copied unchanged (2026-11-06) from the script that drew the submitted panel; only this
docstring is new. ``fig_rss_vs_size.py`` is an earlier, different rendering (log y axis).
"""

import collections
import csv
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import plotstyle as ps

tables, out = Path(sys.argv[1]), Path(sys.argv[2])
NO_HIRES = "--no-hires" in sys.argv[3:]
stem = "Figure_2c_nohires" if NO_HIRES else "Figure_2c"
out.mkdir(parents=True, exist_ok=True)


def num(v):
    return float(v) if v not in ("", "NA", "nan", None) else math.nan


def size(p):
    return int(p.lstrip("s"))


rows = []
# STARE: the largest registration job (anchor, tile task or slide-wide solve) over all rounds
by = collections.defaultdict(list)
for r in csv.DictReader(open(tables / "stare_stage_peaks.csv")):
    if r["row"] == "stare":
        jobs = [
            num(r[k]) for k in ("coarse_peak_gb", "tile_peak_gb_max", "solve_peak_gb")
        ]
        by[r["patient_id"]].append(max(x for x in jobs if not math.isnan(x)))
for p, v in by.items():
    rows.append(("STARE (largest task)", size(p), max(v), len(v)))
# VALIS: per-patient peak of one run registering all rounds
names = {
    "valis120": "VALIS 1.2.0, defaults",
    "valis120_hires": "VALIS 1.2.0, high-res. example",
}
for r in csv.DictReader(open(tables / "synthetic_cost.csv")):
    if r["run"] in names and r["status"] == "complete":
        rows.append((names[r["run"]], size(r["patient_id"]), num(r["peak_rss_gb"]), 1))
df = pd.DataFrame(
    rows, columns=["method", "px_per_side", "peak_rss_gb", "n_rounds_or_runs"]
).sort_values(["method", "px_per_side"])

ps.apply_style()
order = [
    "STARE (largest task)",
    "VALIS 1.2.0, defaults",
    "VALIS 1.2.0, high-res. example",
]
colors = ps.lock_palette(order, figdir=str(out))  # same colours in both versions
if NO_HIRES:
    order = order[:2]
    df = df[df.method.isin(order)]
fig, ax = ps.figure(width="single", aspect=0.75)
ax = ax[0] if isinstance(ax, list) else ax
for m, mk in zip(order, ("o", "s", "^")):
    d = df[df.method == m]
    ax.plot(
        d.px_per_side,
        d.peak_rss_gb,
        marker=mk,
        ms=3.5,
        lw=1.0,
        color=colors[m],
        label=m,
    )
if not NO_HIRES:  # the 200 GB mark only means something next to the high-res. curve
    ax.axhline(200, ls="--", lw=0.7, color="0.4")
    ax.text(4096 * 1.05, 200 + 4, "200 GB", fontsize=6, color="0.4", va="bottom")
ax.set_xscale("log", base=2)
# ax.set_yscale("log")
sizes = sorted(df.px_per_side.unique())
ax.set_xticks(sizes)
ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=45, ha="right")
ax.minorticks_off()
# linear y axis
if NO_HIRES:
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
else:
    ax.set_ylim(0, 230)
    ax.set_yticks([0, 50, 100, 150, 200])
ax.set_xlabel("Image size (px per side, log2)")
ax.set_ylabel("Peak RSS (GB)")
if NO_HIRES:  # the empty band between the STARE and VALIS curves
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.45), frameon=False, fontsize=6)
else:
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.84), frameon=False, fontsize=6)
ps.save(fig, str(out / stem), data=df)
print(df.to_string(index=False))

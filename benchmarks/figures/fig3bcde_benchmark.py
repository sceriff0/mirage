# ruff: noqa: E712
# E712: `== True` kept from the submitted script; displacement_readable can be an object column.
"""Fig. 3b-e of the manuscript, as published, from the regbench cohort tables.

    python benchmarks/figures/fig3bcde_benchmark.py <REGBENCH_OUT>/tables <out dir>

b, c: matched-pair nuclear Dice and residual centroid distance (per-pair median) on the 64
round pairs of the nine-round protocol, every pair and every method, the unregistered
condition included (ruling 2026-11-05: no readable-displacement filter); brackets: two-sided
Wilcoxon signed-rank of STARE against each of the five others, Holm over the five. d, e: peak
RSS and CPU time per case of the runs as performed (the two extra stains included), the
image writer excluded for every method. Writes ``Figure_3bcde.{pdf,png,svg}`` and
``Figure_3bcde_source_data.csv``; prints the Holm P values.

Copied unchanged (2026-11-06) from the script that drew the submitted panels; only this
docstring is new. ``fig_cohort_scores.py`` is an earlier, different rendering (78 pairs,
centroid distance tested against four methods). The numbers of the text are printed by
``benchmarks/regbench/paper/accuracy_stats.py``, which applies the same tests.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import plotstyle as ps  # noqa: E402

tables, out = Path(sys.argv[1]), Path(sys.argv[2])
ORDER = ["Unregistered", "ASHLAR", "DeeperHistReg", "PALOM", "VALIS 1.2.0", "STARE"]
M = {
    "initial": "Unregistered",
    "ashlar": "ASHLAR",
    "deeperhistreg_hr_fluo_cpu": "DeeperHistReg",
    "palom": "PALOM",
    "valis120": "VALIS 1.2.0",
    "stare": "STARE",
}

cells = pd.read_csv(tables / "cells.csv", low_memory=False)
cells = cells[cells.method.isin(M)].assign(m=lambda d: d.method.map(M))
# protocol rounds only: drop the two stains outside the nine-round protocol (64 of 78 pairs)
cells = cells[~cells.panel.str.match(r"^DAPI_(PANCK11C_SMA11C|FSP1_L1CAM|FSP1)$")]
res = pd.read_csv(tables / "resources.csv", low_memory=False)
res = res[res.method.isin(M)].assign(m=lambda d: d.method.map(M))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def pair_tests(metric, others, readable):
    s = cells[cells.m == "STARE"].set_index("case_id")
    raw = []
    for o in others:
        y = cells[cells.m == o].set_index("case_id")
        j = s[[metric, "displacement_readable"]].join(
            y[[metric, "displacement_readable"]], rsuffix="_o", how="inner"
        )
        if readable:
            j = j[
                (j.displacement_readable == True) & (j.displacement_readable_o == True)
            ]
        raw.append(wilcoxon(j[metric] - j[metric + "_o"], method="approx").pvalue)
    return dict(zip(others, holm(raw)))


# per-case cost: pair-unit methods summed (CPU) / maxed (peak) over pairs; case-unit rows repeat the case value
def per_case(col, how):
    rows = []
    for m, d in res.groupby("m"):
        agg = d.groupby("patient_id")[col]
        v = agg.sum() if (how == "sum" and d.unit.iloc[0] == "pair") else agg.max()
        rows += [(m, p, x) for p, x in v.items()]
    return pd.DataFrame(rows, columns=["method", "case", col])


ps.apply_style()
mpl.rcParams["axes.unicode_minus"] = False
colors = ps.lock_palette(["STARE", "Other methods"], figdir=str(out))
fig, axes = ps.figure(width="double", nrows=2, ncols=2, aspect=0.62)
rng = np.random.default_rng(0)
data_out = []


def boxes(ax, groups, order, ylabel, fmt, log=False, points_only=False):
    for i, m in enumerate(order):
        v = np.asarray(groups[m], float)
        col = colors["STARE"] if m == "STARE" else colors["Other methods"]
        x = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v, s=5, color=col, alpha=0.8, lw=0, zorder=1)
        if points_only:  # n < 10: points with a median bar, never a box
            ax.plot(
                [i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1.2, zorder=3
            )
        else:
            ax.boxplot(
                [v],
                positions=[i],
                widths=0.55,
                whis=1.5,
                showfliers=False,
                zorder=2,
                medianprops=dict(color="k", lw=1.2),
                boxprops=dict(facecolor="none", lw=0.7),
                whiskerprops=dict(lw=0.7),
                capprops=dict(lw=0.7),
                patch_artist=True,
            )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [f"{m}\n(n={len(groups[m])})" for m in order], rotation=35, ha="right"
    )
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(ylabel)
    if log:  # plain tick numbers (0.1, 1, 10): no 10^x mathtext, which editors re-flow into gibberish
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    tr = ax.get_xaxis_transform()
    for i, m in enumerate(order):
        ax.text(
            i,
            0.02,
            fmt.format(np.median(groups[m])),
            transform=tr,
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="0.35",
        )


def brackets(ax, order, pvals, top, step):
    s = order.index("STARE")
    for k, (o, p) in enumerate(sorted(pvals.items(), key=lambda t: order.index(t[0]))):
        y = top + k * step
        i = order.index(o)
        ax.plot(
            [i, i, s, s], [y - step * 0.25, y, y, y - step * 0.25], color="k", lw=0.6
        )
        ax.text((i + s) / 2, y, stars(p), ha="center", va="bottom", fontsize=6)


# b: Dice
ax = axes[0]
g = {m: cells[cells.m == m].dice_matched.dropna().values for m in ORDER}
boxes(ax, g, ORDER, "Matched-pair nuclear Dice", "{:.2f}")
pb = pair_tests("dice_matched", [m for m in ORDER if m != "STARE"], readable=False)
brackets(ax, ORDER, pb, top=0.95, step=0.045)
ax.set_ylim(0.38, 0.95 + 5 * 0.045 + 0.02)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
data_out += [("b", m, x) for m in ORDER for x in g[m]]

# c: centroid distance, all 64 round pairs, unregistered included (user ruling 2026-11-05)
ax = axes[1]
oc = ORDER
g = {m: cells[cells.m == m].displacement_um_p50.dropna().values for m in oc}
boxes(ax, g, oc, "Residual centroid distance (µm)", "{:.2f}")
pc = pair_tests("displacement_um_p50", [m for m in oc if m != "STARE"], readable=False)
top = max(np.max(v) for v in g.values()) * 1.05
brackets(ax, oc, pc, top=top, step=top * 0.1)
ax.set_ylim(-0.35, top * (1 + 5 * 0.1) + 0.15)
ax.set_yticks([0, 1, 2, 3, 4])
data_out += [("c", m, x) for m in oc for x in g[m]]

# d: peak RSS per case
oc = ORDER[1:]
ax = axes[2]
pk = per_case("peak_rss_gb", "max")
g = {m: pk[pk.method == m].peak_rss_gb.values for m in oc}
boxes(ax, g, oc, "Peak RSS per case (GB, log10)", "{:.1f}", log=True, points_only=True)
ax.set_ylim(0.15, 100)
data_out += [("d", m, x) for m in oc for x in g[m]]

# e: CPU time per case
ax = axes[3]
cp = per_case("cpu_h", "sum")
g = {m: cp[cp.method == m].cpu_h.values for m in oc}
boxes(ax, g, oc, "CPU time per case (h, log10)", "{:.1f}", log=True, points_only=True)
ax.set_ylim(0.03, 200)
data_out += [("e", m, x) for m in oc for x in g[m]]

fig.tight_layout(h_pad=1.2, w_pad=2.0, rect=(0, 0, 1, 0.97))
ps.panel_labels(axes, labels=["b", "c", "d", "e"])
df = pd.DataFrame(data_out, columns=["panel", "method", "value"])
ps.save(fig, str(out / "Figure_3bcde"), data=df)

print("Dice Holm P:", {k: f"{v:.2g}" for k, v in pb.items()})
print("Centroid Holm P:", {k: f"{v:.2g}" for k, v in pc.items()})
print(
    df.groupby(["panel", "method"]).value.agg(["count", "median"]).round(3).to_string()
)

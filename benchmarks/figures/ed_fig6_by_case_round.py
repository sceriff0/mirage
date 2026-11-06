"""Extended Data Fig. 6 of the manuscript, as published: the 64 protocol round pairs of
Fig. 3b,c (the two stains outside the nine-round protocol, FSP1/L1CAM and PANCK/SMA, are
dropped), VALIS 1.2.0 at register().

a  Dice by case          b  Dice by panel pair (moving panel)
c  centroid by case      d  centroid by panel pair      (all 64 round pairs, as Fig. 3c)

    python benchmarks/figures/ed_fig6_by_case_round.py <REGBENCH_OUT>/tables <out dir>

Boxes only (no points overlaid). Writes ``Figure_S6.{pdf,png,svg}`` and
``Figure_S6_source_data.csv`` (file names kept from the submitted figure, which was drafted as
Supplementary Fig. S6), and prints the counts quoted in the legend.

Copied unchanged (2026-11-06) from the script that drew the submitted panels; only this
docstring is new. ``fig_by_case_round.py`` is an earlier, different rendering.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import plotstyle as ps  # noqa: E402

tables, out = Path(sys.argv[1]), Path(sys.argv[2])
M = {
    "initial": "Unregistered",
    "ashlar": "ASHLAR",
    "deeperhistreg_hr_fluo_cpu": "DeeperHistReg",
    "palom": "PALOM",
    "valis120": "VALIS 1.2.0",
    "stare": "STARE",
}
ORDER = list(M.values())
# colours of the original S6 legend (edge colour; boxes are filled with it at alpha 0.8)
COL = {
    "Unregistered": "#A6A5A4",
    "ASHLAR": "#888787",
    "DeeperHistReg": "#9474B4",
    "PALOM": "#57A396",
    "VALIS 1.2.0": "#5878AB",
    "STARE": "#E18333",
}
# panel-pair labels and order of the original S6b,d, protocol rounds only
PAIR = {
    "DAPI_PDL1_ARID1A": "ARID1A/PD-L1",
    "DAPI_CD56_CD14": "CD14/CD56",
    "DAPI_CD45_CD163": "CD163/CD45",
    "DAPI_GZMB_CD74": "CD74/GZMB",
    "DAPI_CD4_CD8": "CD4/CD8",
    "DAPI_VIMENTIN_FOXP3": "FOXP3/VIM",
    "DAPI_CD3_P53": "CD3/P53",
    "DAPI_PD1_wrongL1CAM": "L1CAM/PD-1",
    "DAPI_PD1_L1CAM": "L1CAM/PD-1",
}
PAIRS = list(dict.fromkeys(PAIR.values()))

c = pd.read_csv(tables / "cells.csv", low_memory=False)
c = c[c.method.isin(M) & c.panel.isin(PAIR)].assign(
    m=lambda d: d.method.map(M),
    pair=lambda d: d.panel.map(PAIR),
    case=lambda d: d.patient_id.str.replace("__2", "", regex=False),
)
assert all(len(c[c.m == m]) == 64 for m in ORDER), c.m.value_counts()
CASES = sorted(c.case.unique())  # string order, as the original: 046 052 10338 ... 5456
rd = c  # all 64 round pairs, as Fig. 3c (user ruling 2026-11-05: no readable filter)

ps.apply_style()
mpl.rcParams["axes.unicode_minus"] = False
fig, axes = ps.figure(width="double", nrows=2, ncols=2, aspect=0.69)
rows = []


def boxes(ax, d, key, groups, metric, methods):
    w = 0.8 / len(
        ORDER
    )  # slot width fixed by the six-method layout, as in the original
    for gi, g in enumerate(groups):
        for m in methods:
            v = d[(d[key] == g) & (d.m == m)][metric].dropna().values
            if not len(v):
                continue
            x = gi - 0.4 + w * (ORDER.index(m) + 0.5)
            ax.boxplot(
                [v],
                positions=[x],
                widths=w * 0.85,
                whis=1.5,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(
                    facecolor=mpl.colors.to_rgba(COL[m], 0.8), edgecolor=COL[m], lw=0.6
                ),
                whiskerprops=dict(color=COL[m], lw=0.6),
                capprops=dict(color=COL[m], lw=0.6),
                medianprops=dict(color="k", lw=1.0),
            )
            rows.extend((metric, key, g, m, x_) for x_ in v)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(groups) - 0.5)
    for gi in range(1, len(groups)):
        ax.axvline(gi - 0.5, color="0.85", lw=0.5, zorder=0)


boxes(axes[0], c, "case", CASES, "dice_matched", ORDER)
boxes(axes[1], c, "pair", PAIRS, "dice_matched", ORDER)
boxes(axes[2], rd, "case", CASES, "displacement_um_p50", ORDER)
boxes(axes[3], rd, "pair", PAIRS, "displacement_um_p50", ORDER)
for ax in axes[:2]:
    ax.set_ylabel("Matched-pair nuclear Dice")
    ax.set_ylim(0.45, 0.95)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
for ax in axes[2:]:
    ax.set_ylabel("Residual centroid distance (µm)")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 5)
    ax.set_yticks([0.25, 0.5, 1, 2, 4])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
for ax in (axes[0], axes[2]):
    ax.set_xlabel("Case")
for ax in (axes[1], axes[3]):
    ax.set_xlabel("Panel pair (moving panel)")

handles = [
    Patch(facecolor=mpl.colors.to_rgba(COL[m], 0.8), edgecolor=COL[m], lw=0.6, label=m)
    for m in ORDER
]
fig.legend(
    handles=handles,
    loc="upper center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, 1.0),
    handlelength=1.2,
    columnspacing=1.6,
)
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.5, w_pad=2.0)
ps.panel_labels(axes)
ps.save(
    fig,
    str(out / "Figure_S6"),
    data=pd.DataFrame(rows, columns=["metric", "grouping", "group", "method", "value"]),
)

# numbers quoted in the S6 legend
w = c.pivot_table(index="case", columns="m", values="dice_matched", aggfunc="median")
print(
    "STARE highest median Dice in cases:",
    (w.drop(columns="Unregistered").idxmax(axis=1) == "STARE").sum(),
    "of",
    len(w),
)
wr = c.pivot_table(index="pair", columns="m", values="dice_matched", aggfunc="median")
print(
    "STARE highest median Dice in panel pairs:",
    (wr.drop(columns="Unregistered").idxmax(axis=1) == "STARE").sum(),
    "of",
    len(wr),
)
x = c.pivot_table(index=["case", "pair"], columns="m", values="dice_matched")
print(
    "STARE highest Dice per round pair (5 methods):",
    (x.drop(columns="Unregistered").idxmax(axis=1) == "STARE").sum(),
    "of",
    len(x),
)
dc = rd.pivot_table(
    index="case", columns="m", values="displacement_um_p50", aggfunc="median"
)
print(
    "STARE lowest median centroid in cases:",
    (dc.idxmin(axis=1) == "STARE").sum(),
    "of",
    len(dc),
)
dp = rd.pivot_table(
    index="pair", columns="m", values="displacement_um_p50", aggfunc="median"
)
print(
    "STARE lowest median centroid in panel pairs:",
    (dp.idxmin(axis=1) == "STARE").sum(),
    "of",
    len(dp),
)
print(
    "VALIS 1.2.0 Dice range in 10338:",
    c[(c.case == "10338") & (c.m == "VALIS 1.2.0")]
    .dice_matched.agg(["min", "max"])
    .round(3)
    .tolist(),
)
print("readable pairs per method:", rd.groupby("m").size().to_dict())

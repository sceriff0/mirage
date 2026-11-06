"""Extended Data Fig. 6 (an earlier rendering; submitted: ed_fig6_by_case_round.py): registration accuracy by case and by round pair, per method.

    python -m benchmarks.figures.fig_by_case_round <REGBENCH_OUT>/tables \
        --cases <REGBENCH_CASES> --outdir figures/by_case_round

Each panel is its own file (you compose them): <outdir>/by_case_round_{a,b,c,d}.{pdf,png,
svg}, each with its _source_data.csv, plus by_case_round_medians.csv.

  a  ``cells.csv: dice_matched`` per method, grouped by CASE (the patient, ``group``):
     one box per method and case over its round pairs
  b  the same Dice grouped by MOVING ROUND: the moving slide's markers (``case.json``:
     its channel names less the nuclear one, as fig_crop_grid titles its columns), one
     box per method and round over the cases
  c  ``displacement_um_p50`` (the residual centroid distance, per-pair median, µm) grouped
     as in a; log y
  d  the same distance grouped as in b; log y

The rows are the ones Fig. 3b,c read (``common.cells_table``: the outlines set, the rows
the scorer's summaries keep, a round the METHOD lost scored at the unregistered pose).
``Unregistered`` is left out of c and d, as from Fig. 3c (``fig_cohort_scores``): a
distance between pairs found under the gate says nothing without registration.

Boxes as in Fig. 3: centre line the median, box the quartiles, whiskers 1.5 x IQR, no
fliers, every round pair a jittered point in its method's colour (VALIS and STARE keep
the colours of every other figure, ``common.palette``; Unregistered grey).

by_case_round_medians.csv: per panel, group (a case or a round: ``grouped_by``) and method
the median and n, and which method
has the best median of the group (highest Dice, lowest distance), so a legend's "in 8 of 8
cases" can be checked against the numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps

# the methods of the legend, in box order (make_figures.sh passes the same)
DEFAULT_METHODS = (
    "Unregistered=initial,STARE=stare,VALIS=valis120,PALOM=palom,"
    "DeeperHistReg=deeperhistreg_hr_fluo_cpu,ASHLAR=ashlar"
)
PANELS = {
    "a": ("dice_matched", "group", "Matched-pair nuclear Dice", False),
    "b": ("dice_matched", "round", "Matched-pair nuclear Dice", False),
    "c": (
        "displacement_um_p50",
        "group",
        "Residual centroid distance (µm, log10)",
        True,
    ),
    "d": (
        "displacement_um_p50",
        "round",
        "Residual centroid distance (µm, log10)",
        True,
    ),
}
GROUP_TITLE = {"group": "Case", "round": "Moving round (markers)"}


def _case_order(g):
    """Numeric ids in numeric order, then the rest by name (as ``list_cases`` sorts)."""
    return (0, int(g), "") if g.isdigit() else (1, 0, g)


def round_labels(cases_root, dataset):
    """``{case_id: the moving round's markers}`` from the cases' own ``case.json``, and
    the rounds in the order they first appear (cases sorted as ``list_cases`` sorts)."""
    from benchmarks.regbench.cases import list_cases

    from .fig_crop_grid import markers

    labels, order = {}, []
    for case in list_cases(cases_root, dataset):
        lab = markers(case)
        labels[str(case.case_id)] = lab
        if lab not in order:
            order.append(lab)
    return labels, order


def long_table(tables, dataset, which, methods, rounds):
    """One row per (case, method) of ``methods``: group, case_id, round, label, and the
    two scores. A case with no round label (not among the cases) is said and dropped."""
    df = C.cells_table(tables, dataset, which)
    df = df[df["method"].astype(str).isin(methods.values())].copy()
    df["label"] = df["method"].astype(str).map({v: k for k, v in methods.items()})
    df["case_id"] = df["case_id"].astype(str)
    df["group"] = df["group"].astype(str)
    df["round"] = df["case_id"].map(rounds)
    lost = sorted(df.loc[df["round"].isna(), "case_id"].unique())
    if lost:
        print(f"  !! {len(lost)} case(s) of the tables are not among the cases: {lost}")
    keep = ["group", "case_id", "round", "label", "dice_matched", "displacement_um_p50"]
    return df[df["round"].notna()][keep]


def colours(labels, figdir):
    """The highlighted methods keep their colour (``common.palette``), the others get the
    palette's next ones, Unregistered is grey."""
    others = [m for m in labels if m not in C.HIGHLIGHT and m != "Unregistered"]
    got = ps.lock_palette([*C.HIGHLIGHT, *others], figdir=str(figdir))
    return {m: (ps.LIGHT_GREY if m == "Unregistered" else got[m]) for m in labels}


def medians(df, name, key, by, labels, higher_is_better):
    """The panel's per-group medians, n and the group's best method."""
    rows = []
    for g, sub in df.groupby(by, sort=False):
        med = {m: float(sub.loc[sub["label"] == m, key].median()) for m in labels}
        finite = {m: v for m, v in med.items() if v == v}
        best = (
            (max if higher_is_better else min)(finite, key=finite.get) if finite else ""
        )
        for m in labels:
            vals = sub.loc[sub["label"] == m, key].dropna()
            rows.append({"panel": name, "grouped_by": by, "group": g, "method": m,
                         "median": med[m], "n": int(len(vals)),
                         "best_in_group": m == best})  # fmt: skip
    return rows


def panel(name, df, labels, groups, colors, seed, outdir):
    key, by, ylabel, log = PANELS[name]
    rng = np.random.default_rng(seed)
    shown = [
        m for m in labels if not (key == "displacement_um_p50" and m == "Unregistered")
    ]
    m = len(shown)
    width = 0.8 / max(m, 1)
    fig, (ax,) = ps.figure(width="double", aspect=0.32)
    drawn = []
    for i, g in enumerate(groups):
        for j, lab in enumerate(shown):
            v = df.loc[(df[by] == g) & (df["label"] == lab), key].to_numpy(float)
            v = v[np.isfinite(v) & ((v > 0) if log else True)]
            x = i + (j - (m - 1) / 2) * width
            if len(v):
                ax.boxplot(v, positions=[x], widths=width * 0.85, showfliers=False,
                           boxprops=C.BOX, medianprops=dict(color="black", linewidth=0.8),
                           whiskerprops=C.BOX, capprops=C.BOX)  # fmt: skip
                jx = x + rng.uniform(-width * 0.25, width * 0.25, len(v))
                ax.scatter(jx, v, s=3, color=colors[lab], zorder=3, linewidths=0)
                drawn += [{"panel": name, by: g, "method": lab, "x": float(a),
                           key: float(b)} for a, b in zip(jx, v)]  # fmt: skip
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45 if by == "round" else 0,
                       ha="right" if by == "round" else "center")  # fmt: skip
    ax.set_xlim(-0.6, len(groups) - 0.4)
    for i in range(1, len(groups)):
        ax.axvline(i - 0.5, color=ps.LIGHT_GREY, lw=0.3, zorder=0)
    ax.set_xlabel(GROUP_TITLE[by])
    ax.set_ylabel(ylabel)
    C.plain_ticks(
        ax, log, float(np.nanmax(df[key].to_numpy(float))) if len(df) else None
    )
    handles = [ps.plt.Line2D([], [], ls="", marker="o", ms=3, color=colors[lab], label=lab)
               for lab in shown]  # fmt: skip
    ax.legend(handles=handles, ncol=len(shown), loc="upper center",
              bbox_to_anchor=(0.5, 1.12), frameon=False, handletextpad=0.2,
              columnspacing=0.8)  # fmt: skip
    data = pd.DataFrame(drawn, columns=["panel", by, "method", "x", key])
    ps.save(fig, str(Path(outdir) / f"by_case_round_{name}"), data=data)
    return shown


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Supp. Fig. S6: registration accuracy by case and by round pair"
    )
    ap.add_argument("tables", help="<REGBENCH_OUT>/tables")
    ap.add_argument(
        "--cases", required=True, help="REGBENCH_CASES: the rounds' markers"
    )
    ap.add_argument(
        "--methods", default=DEFAULT_METHODS, help="LABEL=variant,... in box order"
    )
    ap.add_argument("--dataset", default="multiplex")
    ap.add_argument("--set", default="outlines", choices=("outlines", "nuclei"))
    ap.add_argument("--panels", default="a,b,c,d")
    ap.add_argument("--outdir", default="figures/by_case_round")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    ps.apply_style()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = C.parse_methods(a.methods)
    labels = list(methods)
    rounds, round_order = round_labels(a.cases, a.dataset)
    df = long_table(a.tables, a.dataset, a.set, methods, rounds)
    if df.empty:
        raise SystemExit(f"no row of {list(methods.values())} in {a.tables}/cells.csv")
    missing = [m for m in labels if m not in set(df["label"])]
    if missing:
        print(f"  !! no rows for {missing} (check --methods against the tables)")
    colors = colours(labels, outdir)
    order = {
        "group": sorted(df["group"].unique(), key=_case_order),
        "round": [r for r in round_order if r in set(df["round"])],
    }
    summary = []
    for name in a.panels.split(","):
        key, by, _, _ = PANELS[name]
        shown = panel(name, df, labels, order[by], colors, a.seed, outdir)
        summary += medians(df, name, key, by, shown, key == "dice_matched")
    med = pd.DataFrame(summary)
    med.to_csv(outdir / "by_case_round_medians.csv", index=False)
    for name in a.panels.split(","):
        sub = med[(med["panel"] == name) & med["best_in_group"]]
        print(
            f"{name}: best median per group: {sub['method'].value_counts().to_dict()}"
        )
    print(f"-> {outdir}/by_case_round_{{a,b,c,d}}.pdf/.png/.svg, _source_data.csv, "
          "by_case_round_medians.csv")  # fmt: skip


if __name__ == "__main__":
    main()

# ruff: noqa: E712, E731
# E712: `== True` is kept on purpose: displacement_readable can be read as an object
# column (strings / NaN), where a bare truth test would select different rows than the
# published run did. E731: the lambdas are the original script's, kept verbatim.
"""Every accuracy number of the manuscripts (261105 and 261106), from the harness's tables.

    python -m benchmarks.regbench.paper.accuracy_stats <REGBENCH_OUT>/tables

Reads ``cells.csv`` (outlines: StarDist nuclei expanded by 10 px), ``cells_nuclei.csv``
(bare nuclei) and ``ashlar_discard.csv``, and prints, on all 78 registered round pairs and on
the 64 pairs of the nine-round protocol (the reference-round re-stain and the FSP1 / L1CAM
stains removed by ``EXTRA``):

* the Supplementary Table 2 accuracy columns of every reported row (n, matched fraction,
  matched-pair nuclear Dice and residual centroid distance with IQRs over ALL pairs, p90;
  the centroid median over the pairs whose displacement is readable is printed beside it);
* the STARE gain over its rigid anchor M0, and the matched-fraction excess;
* bare-nuclei, mutual-nearest-neighbour and exact-component Dice (Supplementary Note 6);
* per-case winners and margins, and the pairs where STARE is highest of the five methods;
* the paired tests of Fig. 3b,c and Supplementary Note 6: two-sided Wilcoxon signed-rank
  per round pair (normal approximation) and on the per-case medians (exact), Holm-corrected.
  Both metrics are compared against the same FIVE (VALIS, PALOM, ASHLAR, DeeperHistReg,
  Unregistered), on every pair (ruling 2026-11-05: no readable-displacement filter; this is
  what ``benchmarks/figures/fig3bcde_benchmark.py`` tests too);
* the fraction of moving ASHLAR fields replaced by its stage model, and the ASHLAR rounds
  that stayed near the unregistered Dice.

It was kept outside the repository until 2026-11-06 and wrapped in ``main`` then. Its
centroid analysis followed an earlier rule (readable pairs only, four comparisons); on
2026-11-06 it was brought to the manuscripts' rule above, which reproduces their numbers
(pair-level Holm P 1.8e-11 and case-level 0.039 for both metrics).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXTRA = re.compile(r"^DAPI_(PANCK11C_SMA11C|FSP1_L1CAM|FSP1)$")
ROWS = [
    "initial",
    "stare",
    "valis120",
    "palom",
    "deeperhistreg_hr_fluo_cpu",
    "ashlar",
    "stare_rigid",
    "valis120_rigid",
    "valis120_micro",
    "valis120_hires_rigid",
    "valis120_hires_nonrigid",
    "valis120_hires",
    "affine_ashlar",
    "palom_affine",
    "deeperhistreg_hr_cpu",
    "deeperhistreg_hr_gpu",
    "deeperhistreg_hr_fluo_gpu",
]
N = {
    "initial": "Unregistered",
    "stare": "STARE",
    "valis120": "VALIS",
    "palom": "PALOM",
    "ashlar": "ASHLAR",
    "deeperhistreg_hr_fluo_cpu": "DHR",
}
f2 = lambda x: f"{x:.2f}"


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    a = np.empty(len(p))
    r = 0
    for i, j in enumerate(o):
        r = max(r, (len(p) - i) * p[j])
        a[j] = min(1, r)
    return a


def run(sub, title):
    c_, cn_ = sub(c), sub(cn)
    g = lambda m, d=c_: d[d.method == m].set_index("case_id")
    print(f"\n################ {title}")
    print(
        "TABLE 2: row | n | matched frac | Dice med (IQR) | centroid med (IQR) all | readable: med (IQR) [n] | p90 all | p90 readable"
    )
    for m in ROWS:
        x = g(m)
        r = x[x.displacement_readable == True]
        q = lambda s: (
            f"{f2(s.median())} ({f2(s.quantile(0.25))}–{f2(s.quantile(0.75))})"
        )
        print(
            f"  {m:28s} {len(x):3d} {f2(x.pair_fraction.median())} {q(x.dice_matched)} {q(x.displacement_um_p50)} | {q(r.displacement_um_p50) if len(r) else '-'} [{len(r)}] "
            f"p90 {f2(x.displacement_um_p90.median())} / {f2(r.displacement_um_p90.median()) if len(r) else '-'}"
        )
    s = g("stare")
    print(
        "STARE gain over M0:",
        f"{g('stare_rigid').dice_matched.median():.3f} -> {s.dice_matched.median():.3f} ({s.dice_matched.median() - g('stare_rigid').dice_matched.median():.3f})",
        "M0 matched-fraction excess",
        f2(g("stare_rigid").pair_fraction_excess.median()),
        "unreg excess",
        f2(g("initial").pair_fraction_excess.median()),
    )
    print(
        "VALIS 10338: STARE",
        f2(s[s.patient_id == "10338"].dice_matched.median()),
        "VALIS",
        f2(g("valis120")[g("valis120").patient_id == "10338"].dice_matched.median()),
        "VALIS range",
        f2(g("valis120")[g("valis120").patient_id == "10338"].dice_matched.min()),
        f2(g("valis120")[g("valis120").patient_id == "10338"].dice_matched.max()),
    )
    others = ["valis120", "palom", "ashlar", "deeperhistreg_hr_fluo_cpu", "initial"]
    print(
        "bare nuclei Dice:",
        {N[m]: f2(g(m, cn_).dice_matched.median()) for m in ["stare"] + others},
    )
    print(
        "MNN Dice:",
        {N[m]: f2(g(m).dice_matched_mnn.median()) for m in ["stare"] + others[:-1]},
    )
    print(
        "exact Dice:",
        {N[m]: f2(g(m).dice_matched_exact.median()) for m in ["stare"] + others[:-1]},
        "full_lsa",
        {
            N[m]: f2(g(m).dice_matched_full_lsa.median())
            for m in ["stare"] + others[:-1]
        },
    )
    # per-case winner + margin
    meth = ["stare"] + others[:-1]
    W = pd.DataFrame(
        {m: g(m).groupby("patient_id").dice_matched.median() for m in meth}
    )
    D = pd.DataFrame(
        {m: g(m).groupby("patient_id").displacement_um_p50.median() for m in meth}
    )
    marg = W.stare - W.drop(columns="stare").max(axis=1)
    print(
        "cases STARE best Dice",
        (W.idxmax(axis=1) == "stare").sum(),
        "lowest centroid",
        (D.idxmin(axis=1) == "stare").sum(),
        "margin",
        f"{marg.min():.3f}-{marg.max():.3f}",
    )
    P = pd.DataFrame({m: g(m).dice_matched for m in meth})
    print(
        "pairs where STARE highest of five:",
        (P.idxmax(axis=1) == "stare").sum(),
        "/",
        len(P),
    )
    for metric, oth in [("dice_matched", others), ("displacement_um_p50", others)]:
        pp, pc, txt = [], [], []
        for o in oth:
            j = s[[metric, "patient_id"]].join(
                g(o)[[metric]], rsuffix="_o", how="inner"
            )
            d = j[metric] - j[metric + "_o"]
            b = (d > 0).sum() if metric == "dice_matched" else (d < 0).sum()
            pp.append(wilcoxon(d, method="approx").pvalue)
            cm = d.groupby(j.patient_id).median()
            pc.append(wilcoxon(cm, method="exact").pvalue)
            txt.append(
                f"{N[o]}: better {b}/{len(d)}, diff {d.median():+.3f} ({d.quantile(0.25):+.3f} to {d.quantile(0.75):+.3f}), cases {int(((cm > 0) if metric == 'dice_matched' else (cm < 0)).sum())}/8"
            )
        for t, a, b in zip(txt, holm(pp), holm(pc)):
            print(f"  {metric[:8]} {t}; pair Holm {a:.2g}; case Holm {b:.2g}")
    a = ad[ad.row.isin(["ashlar", "affine_ashlar"])].copy()
    a = a[~a.panel.str.match(EXTRA)] if title.startswith("PROTO") else a
    print(
        "ASHLAR replaced fraction median:",
        a.groupby("row").discard_frac.median().round(3).to_dict(),
        "n",
        a.groupby("row").size().to_dict(),
    )
    x = g("ashlar")
    u = g("initial").dice_matched
    near = x[(x.dice_matched - u.reindex(x.index)).abs() < 0.05]
    print(
        "ASHLAR rounds near unregistered (|dDice|<0.05) by case:",
        near.groupby("patient_id").size().to_dict(),
    )
    print(
        "readable-not counts:",
        {
            m: int((g(m).displacement_readable != True).sum())
            for m in [
                "initial",
                "stare_rigid",
                "ashlar",
                "affine_ashlar",
                "palom",
                "stare",
                "valis120",
                "deeperhistreg_hr_fluo_cpu",
            ]
        },
    )
    print("per-case n:", s.groupby("patient_id").size().to_dict())


def main(argv=None):
    """Read the three tables from ``tables_dir`` and print every number, 78 then 64 pairs."""
    global c, cn, ad
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tables_dir", type=Path, help="<REGBENCH_OUT>/tables")
    T = ap.parse_args(argv).tables_dir
    c = pd.read_csv(T / "cells.csv", low_memory=False)
    cn = pd.read_csv(T / "cells_nuclei.csv", low_memory=False)
    ad = pd.read_csv(T / "ashlar_discard.csv", low_memory=False)
    run(lambda d: d, "ALL 78")
    run(lambda d: d[~d.panel.str.match(EXTRA)], "PROTOCOL 64")


if __name__ == "__main__":
    main()

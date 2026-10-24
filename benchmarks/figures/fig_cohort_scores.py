"""Fig. 3b-e: matched-pair Dice, residual centroid distance, peak memory and CPU time
per method on the cohort, one box per method, every case (or patient) a point.

    python -m benchmarks.figures.fig_cohort_scores <REGBENCH_OUT>/tables \
        --methods "Unregistered=initial,ASHLAR=ashlar,...,VALIS=valis120_hires,STARE=stare" \
        --outdir figures/cohort

Each panel is its own file (you compose them): <outdir>/cohort_{dice,distance,
peak_rss,cpu}.{pdf,png,svg}, each with its _source_data.csv, plus cohort_tests.csv.

  dice      ``cells.csv: dice_matched`` (the pipeline's pairing, the headline), one point
            per case (round pair), the rows the scorer's summaries read
            (``excluded_from_summaries`` false: a round the METHOD lost is kept, scored at
            the unregistered pose, as in every table)
  distance  ``displacement_um_p50``, the same rows, ``Unregistered`` left out (a distance
            between pairs found under the gate says nothing without registration)
  peak_rss  ``summary_resources.csv: peak_rss_gb`` per PATIENT (a per-patient method has
            one launch), complete patients only; log y
  cpu       ``cpu_h_used`` (measured CPU time, the paper's headline) per patient, the same

Brackets (dice, distance): every method against ``--against`` (default STARE), paired
two-sided Wilcoxon signed-rank on the cases both scored, Holm-adjusted across the
comparisons of the panel; *** < 0.001, ** < 0.01, * < 0.05, ns. The numbers are in
cohort_tests.csv. Medians are printed under the boxes. Jitter: ``--seed``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps

PANELS = {
    "dice": ("dice_matched", "Matched-pair nuclear Dice", False),
    "distance": ("displacement_um_p50", "Residual centroid distance (µm)", False),
    "peak_rss": ("peak_rss_gb", "Peak RSS (GB, log10)", True),
    "cpu": ("cpu_h_used", "CPU time per patient (h, log10)", True),
}


def per_case(tables, dataset, which, methods, key):
    df = C.cells_table(tables, dataset, which)
    df = df[df["method"].astype(str).isin(methods.values())]
    wide = df.pivot_table(index=["group", "case_id"], columns="method", values=key)
    return wide.rename(columns={v: k for k, v in methods.items()}).reset_index()


def per_patient(tables, dataset, methods, key):
    s = pd.read_csv(Path(tables) / "summary_resources.csv", low_memory=False)
    s = s[(s["dataset"].astype(str) == dataset) & s["complete"].map(C.truthy)]
    s = s[s["method"].astype(str).isin(methods.values())]
    wide = s.pivot_table(index="group", columns="method", values=key)
    return wide.rename(columns={v: k for k, v in methods.items()}).reset_index()


def panel(name, wide, labels, against, colors, seed, outdir):
    key, ylabel, log = PANELS[name]
    rng = np.random.default_rng(seed)
    labels = [m for m in labels if m in wide.columns]
    fig, (ax,) = ps.figure(width=max(40.0, 12.0 * len(labels) + 18.0), aspect=1.0)
    xs = {m: i for i, m in enumerate(labels)}
    drawn = C.boxes(ax, [(xs[m], m, wide[m]) for m in labels], colors, rng, log=log)
    tests = []
    if name in ("dice", "distance") and against in labels:
        tests = C.wilcoxon_holm(wide, against, [m for m in labels if m != against])
        for t in tests:
            t["panel"] = name
        hi = float(np.nanmax(wide[labels].to_numpy(float)))
        span = hi - float(np.nanmin(wide[labels].to_numpy(float)))
        C.brackets(ax, xs, tests, hi + 0.06 * span, 0.07 * span)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    n = {m: int(wide[m].notna().sum()) for m in labels}
    ax.set_xticklabels([f"{m}\n(n={n[m]})" for m in labels], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    C.room_for_medians(ax, log)
    C.plain_ticks(ax, log, float(np.nanmax(wide[labels].to_numpy(float))))
    data = pd.DataFrame(drawn).rename(columns={"label": "method", "value": key})
    ps.save(fig, str(Path(outdir) / f"cohort_{name}"), data=data)
    return tests


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tables", help="<REGBENCH_OUT>/tables")
    ap.add_argument("--methods", required=True, help="LABEL=variant,... in box order")
    ap.add_argument(
        "--against", default="STARE", help="the label every test is against"
    )
    ap.add_argument("--dataset", default="multiplex")
    ap.add_argument("--set", default="outlines", choices=("outlines", "nuclei"))
    ap.add_argument("--panels", default="dice,distance,peak_rss,cpu")
    ap.add_argument("--outdir", default="figures/cohort")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    ps.apply_style()
    colors = C.palette(a.outdir)
    methods = C.parse_methods(a.methods)
    labels = list(methods)
    all_tests = []
    for name in a.panels.split(","):
        key, _, _ = PANELS[name]
        if name in ("dice", "distance"):
            wide = per_case(a.tables, a.dataset, a.set, methods, key)
            shown = [
                m
                for m in labels
                if not (name == "distance" and methods[m] == C.INITIAL)
            ]
        else:
            run = {k: v for k, v in methods.items() if v != C.INITIAL}
            wide = per_patient(a.tables, a.dataset, run, key)
            shown = list(run)
        missing = [m for m in shown if m not in wide.columns]
        if missing:
            print(
                f"  !! {name}: no rows for {missing} (check --methods against the tables)"
            )
        all_tests += panel(name, wide, shown, a.against, colors, a.seed, a.outdir)
        med = {m: round(float(wide[m].median()), 3) for m in shown if m in wide}
        print(f"{name}: medians {med}")
    if all_tests:
        pd.DataFrame(all_tests).to_csv(Path(a.outdir) / "cohort_tests.csv", index=False)
        for t in all_tests:
            print(
                f"  {t['panel']}: {t['method']} vs {t['against']}: n={t['n_pairs']}, "
                f"p_holm={t['p_holm']:.3g} {t['stars']}"
            )
    print(f"-> {a.outdir}/cohort_*.pdf/.png/.svg, _source_data.csv, cohort_tests.csv")


if __name__ == "__main__":
    main()

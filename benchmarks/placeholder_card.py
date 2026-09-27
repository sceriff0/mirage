#!/usr/bin/env python3
"""placeholder_card.py -- mark the image-composite figures that have not been rendered yet.

The composites (mosaic, overlay, zoom, crop, channel) are pictures of REAL slides. Unlike a
score, a slide cannot be synthesised honestly, so nothing here draws an image of tissue.
What it draws instead is a CARD: a grey, hatched panel that says, in the figure itself,
which composite belongs in that slot and why it is not there yet. The layout of a figure
set can then be reviewed while runs are still going, and no card can be mistaken for data.

    fill   for every row of submit_figures.sh's plan (``.launch/figure_plan.tsv``) whose
           output directory holds no rendered file, write
           ``<ROOT>/<out>/PLACEHOLDER_<kind>_row<N>.<fmt>`` plus, at the root,
           ``PLACEHOLDER_COMPOSITES.csv`` (one line per card) and ``PLACEHOLDER_DATA.txt``.
    clear  delete every PLACEHOLDER_* file under ROOT. submit_all_figures.sh runs it before
           every composites pass, so a card never outlives the run that replaces it.

Opt-in: submit_all_figures.sh calls ``fill`` only with PLACEHOLDER_MISSING=1.

Known approximation: several plan rows may write into ONE directory (patients, variants).
A directory holding any rendered file counts as rendered for every row that targets it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PREFIX = "PLACEHOLDER_"
INDEX = "PLACEHOLDER_COMPOSITES.csv"
MARKER = "PLACEHOLDER_DATA.txt"
RENDERED_SUFFIXES = {".png", ".pdf", ".svg", ".tif", ".tiff", ".jpg", ".jpeg"}


def read_plan(plan: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(plan.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(
                f"{plan}:{n}: expected <kind> <run> <out> <args>, got {line!r}"
            )
        rows.append({"row": n, "kind": parts[0], "run": parts[1], "out": parts[2]})
    return rows


def is_rendered(outdir: Path) -> bool:
    if not outdir.is_dir():
        return False
    return any(
        p.is_file()
        and p.suffix.lower() in RENDERED_SUFFIXES
        and not p.name.startswith(PREFIX)
        for p in outdir.rglob("*")
    )


def draw_card(
    path_stem: Path, kind: str, run: str, out: str, formats: list[str]
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0, 0), 1, 1, facecolor="#e6e6e6", edgecolor="#7f7f7f", hatch="//", lw=2
        )
    )
    ax.text(
        0.5,
        0.66,
        "PLACEHOLDER",
        ha="center",
        va="center",
        fontsize=30,
        color="#b22222",
        weight="bold",
    )
    ax.text(
        0.5, 0.45, f"{kind} not rendered yet", ha="center", va="center", fontsize=14
    )
    ax.text(
        0.5,
        0.30,
        f"run: {run}",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
    )
    ax.text(
        0.5,
        0.20,
        f"slot: {out}",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
    )
    ax.text(
        0.5,
        0.07,
        "no image data -- layout preview only",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        color="#555555",
    )
    written = []
    for fmt in formats:
        p = path_stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=100)
        written.append(p)
    plt.close(fig)
    return written


def fill(plan: Path, root: Path, formats: list[str]) -> int:
    cards = []
    for r in read_plan(plan):
        outdir = root / r["out"]
        if is_rendered(outdir):
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        stem = outdir / f"{PREFIX}{r['kind']}_row{r['row']}"
        for p in draw_card(stem, r["kind"], r["run"], r["out"], formats):
            cards.append({**r, "path": str(p.relative_to(root))})
    with open(root / INDEX, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "kind", "run", "out", "path"])
        w.writeheader()
        w.writerows(cards)
    n_slots = len({c["row"] for c in cards})
    (root / MARKER).write_text(
        f"PLACEHOLDER — {n_slots} composite slot(s) under this directory are grey cards, "
        f"not figures.\nThey are listed in {INDEX}. Re-run without PLACEHOLDER_MISSING=1 "
        "once the runs finish; that pass deletes every card first.\n"
    )
    print(
        f"[placeholders] {n_slots} composite slot(s) carded; listed in {root / INDEX}"
    )
    return n_slots


def clear(root: Path) -> int:
    n = 0
    if not root.is_dir():
        return 0
    for p in root.rglob(f"{PREFIX}*"):
        if p.is_file():
            p.unlink()
            n += 1
    if n:
        print(f"[placeholders] removed {n} stale placeholder file(s) under {root}")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fill")
    f.add_argument("--plan", required=True, type=Path)
    f.add_argument("--root", required=True, type=Path)
    f.add_argument("--formats", default="png,pdf")
    c = sub.add_parser("clear")
    c.add_argument("--root", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "fill":
        fill(a.plan, a.root, [x for x in a.formats.split(",") if x])
    else:
        clear(a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

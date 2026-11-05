"""Fig. 3a: one patient's rounds side by side, the same crop unregistered and after each
method, reference DAPI cyan and moving DAPI magenta (white = they agree), each crop with
its own matched-pair Dice.

    python -m benchmarks.figures.fig_crop_grid --cases <REGBENCH_CASES> \
        --out <REGBENCH_OUT> --tables <REGBENCH_OUT>/tables \
        --methods "STARE=stare,VALIS=valis120_hires" --outdir figures/crops

ROWS: ``Unregistered`` (the two native slides on one pixel grid, as the QC composite places
them), then one per ``--methods`` entry: the reference against that result's written
image, on the reference canvas at its ``origin_xy`` (as regbench's tissue scorer reads it).
COLUMNS: the patient's rounds (``--columns``), titled with the moving round's markers.

THE PATIENT (``--patient``, else chosen): the one whose rounds are closest to the cohort
median Dice of every method in ``--methods`` (``common.representative``; ``cells.csv``),
among the patients with at least ``--columns`` rounds: typical, not the best. The table
of that choice is written beside the figure.

THE CROP (``--crop X,Y`` in reference px, its centre, else chosen): one square of
``--crop-um``, the same reference coordinates in every panel (the rounds share their
reference slide): the window with the most reference nuclei, ties to the top-left.

THE DICE OF A CROP is the harness's ``dice_matched``, restricted to the crop: the pairs of
the pipeline's pairing on the WHOLE slide (``cell_pairs.match_cells``, ``lsa``, the gate of
the case's native sets), kept when their reference nucleus lies in the crop, then
2 x sum(intersection) / sum(area) over them, as ``pairs._pair_scores`` computes it. The
whole case's value is printed beside it as a check against ``cells.csv``.

DISPLAY: ``common.crisp`` per crop and channel (window inside the crop, CLAHE, unsharp,
x3 cubic). A display adjustment: state it in the legend.

Outputs: <outdir>/crops_<patient>.pdf/.png/.svg, _source_data.csv (every panel: case,
method, crop, Dice in the crop and in the whole case), _patient_choice.csv, and each
panel as <stem>_<row>_<case>.png for Affinity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps

UNREG = "Unregistered"


def crop_dice(case, out, variant, x0, y0, side):
    """``(dice in the crop, dice of the whole case, pairs in the crop)``."""
    from benchmarks.regbench import pairs
    from benchmarks.regbench.cases import load_points, polys, result_dir
    from benchmarks.regbench.mapping import MOV_TO_REF

    points = load_points(case)
    ref, mov = polys(points, "ref"), polys(points, "mov")
    geo = pairs.Geometry()
    radius, _ = pairs.gate_px(ref, mov, geo)
    a, b = ref, mov
    if variant != C.INITIAL:
        with np.load(result_dir(out, variant, case) / "warped.npz") as z:
            xy, direction = z["cells_xy"].astype(float), str(z["direction"])
        if direction == MOV_TO_REF:
            b = pairs.with_xy(mov, xy)
        else:
            a = pairs.with_xy(ref, xy)
    area_a, cent_a = pairs.area_centroid(a, geo)
    area_b, cent_b = pairs.area_centroid(b, geo)
    ia, ib, _, _ = pairs.cp.match_cells(cent_a, cent_b, radius, method="lsa")
    iou, scored = pairs._pair_iou(a, b, ia, ib, None)
    _, native = pairs.area_centroid(ref, geo)  # the crop is on the reference's nuclei
    c = native[ia]
    inside = (
        (c[:, 0] >= x0)
        & (c[:, 0] < x0 + side)
        & (c[:, 1] >= y0)
        & (c[:, 1] < y0 + side)
    )

    def dice(keep):
        k = keep & scored
        tot = area_a[ia][k] + area_b[ib][k]
        return (
            float(2.0 * (iou[k] * tot / (1.0 + iou[k])).sum() / tot.sum())
            if tot.sum()
            else np.nan
        )

    return dice(inside), dice(np.ones(len(ia), bool)), int((inside & scored).sum())


def densest_window(case, side):
    """Top-left (x0, y0) of the ``side`` square holding the most reference nuclei."""
    from benchmarks.regbench import pairs
    from benchmarks.regbench.cases import load_points, polys

    _, cent = pairs.area_centroid(polys(load_points(case), "ref"))
    cent = cent[np.isfinite(cent).all(1)]
    h, w = case.ref_hw
    best, at = -1, (0, 0)
    step = max(side // 4, 1)
    for y0 in range(0, max(h - side, 0) + 1, step):
        for x0 in range(0, max(w - side, 0) + 1, step):
            n = int(
                (
                    (cent[:, 0] >= x0)
                    & (cent[:, 0] < x0 + side)
                    & (cent[:, 1] >= y0)
                    & (cent[:, 1] < y0 + side)
                ).sum()
            )
            if n > best:
                best, at = n, (x0, y0)
    return at


def markers(case):
    names = list(case.extra.get("channels_mov") or [])
    if isinstance(names, str):
        names = names.split("|")
    nuc = int(case.extra.get("mov_channel", 0))
    rest = [n for i, n in enumerate(names) if i != nuc]
    return " / ".join(rest) if rest else str(case.case_id)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tables", help="<out>/tables: to choose the patient by median Dice"
    )
    ap.add_argument("--dataset", default="multiplex")
    ap.add_argument("--methods", required=True, help="LABEL=variant,... one row each")
    ap.add_argument("--patient", help="pin the patient (else the median-Dice one)")
    ap.add_argument("--columns", type=int, default=4, help="rounds shown")
    ap.add_argument("--crop-um", type=float, default=150.0)
    ap.add_argument("--crop", help="X,Y: the crop's centre in reference px")
    ap.add_argument("--clip", default="1,99.8")
    ap.add_argument("--clahe", type=float, default=0.01)
    ap.add_argument("--sharpen", type=float, default=0.6)
    ap.add_argument("--upsample", type=int, default=3)
    ap.add_argument("--outdir", default="figures/crops")
    a = ap.parse_args(argv)

    from stare.slide_io import open_lazy

    from benchmarks.regbench.cases import list_cases

    ps.apply_style()
    methods = C.parse_methods(a.methods)
    cases = list_cases(a.cases, a.dataset)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    patient = a.patient
    if not patient:
        if not a.tables:
            raise SystemExit("give --patient, or --tables to choose it by median Dice")
        cells = C.cells_table(a.tables, a.dataset)
        wide = cells.pivot_table(
            index=["group", "case_id"], columns="method", values="dice_matched"
        ).reset_index()
        wide["group"] = wide["group"].astype(str)
        patient, table = C.representative(
            wide, list(methods.values()), by="group", min_rounds=a.columns
        )
        table.to_csv(outdir / f"crops_{patient}_patient_choice.csv", index=False)
        print(f"patient: {patient}, the closest to the median Dice of {list(methods)}")
    rounds = [c for c in cases if str(c.group) == patient][: a.columns]
    if not rounds:
        raise SystemExit(f"no {a.dataset} case of patient {patient!r}")
    um_px = float(rounds[0].pixel_size_um)
    side = int(round(a.crop_um / um_px))
    if a.crop:
        cx, cy = (int(float(v)) for v in a.crop.split(","))
        x0, y0 = cx - side // 2, cy - side // 2
    else:
        x0, y0 = densest_window(rounds[0], side)
    show = dict(
        clip=tuple(float(v) for v in a.clip.split(",")),
        clahe=a.clahe,
        sharpen=a.sharpen,
        upsample=a.upsample,
    )

    rows = [(UNREG, C.INITIAL), *methods.items()]
    stem = outdir / f"crops_{patient}"
    panels, records = {}, []
    for case in rounds:
        ref_arr, _, close_ref = open_lazy(case.ref_nuclear)
        try:
            r = C.crisp(C.region(ref_arr, 0, y0, x0, side, side), **show)
        finally:
            close_ref()
        for label, variant in rows:
            if variant == C.INITIAL:
                path, ch, (ox, oy) = case.mov_nuclear, 0, (0.0, 0.0)
            else:
                path, ch, (ox, oy) = C.written_nuclear(a.out, variant, case)
            arr, _, close = open_lazy(path)
            try:
                m = C.region(
                    arr, ch, y0 - int(round(oy)), x0 - int(round(ox)), side, side
                )
            finally:
                close()
            rgb = C.composite(r, C.crisp(m, **show))
            dice, whole, n = crop_dice(case, a.out, variant, x0, y0, side)
            panels[(label, str(case.case_id))] = (rgb, dice)
            ps.plt.imsave(f"{stem}_{label.replace(' ', '_')}_{case.case_id}.png", rgb)
            records.append(
                {
                    "patient": patient,
                    "case_id": case.case_id,
                    "markers": markers(case),
                    "row": label,
                    "variant": variant,
                    "crop_x0_px": x0,
                    "crop_y0_px": y0,
                    "crop_side_px": side,
                    "crop_side_um": a.crop_um,
                    "pairs_in_crop": n,
                    "dice_in_crop": dice,
                    "dice_whole_case": whole,
                }
            )
            print(
                f"  {case.case_id} {label:13}: Dice {dice:.3f} in the crop ({n} pairs), "
                f"{whole:.3f} whole case"
            )

    # the grid: square panels, the row names left, the markers on top
    n_r, n_c = len(rows), len(rounds)
    cell, gap, left, top = 27.0, 0.8, 5.0, 5.0  # mm
    width, height = (
        left + n_c * cell + (n_c - 1) * gap,
        top + n_r * cell + (n_r - 1) * gap,
    )
    fig = C.fixed_figure(width, height)
    from matplotlib.patheffects import withStroke

    for i, (label, _) in enumerate(rows):
        for j, case in enumerate(rounds):
            x, y = left + j * (cell + gap), top + i * (cell + gap)
            ax = fig.add_axes(
                (x / width, 1 - (y + cell) / height, cell / width, cell / height)
            )
            rgb, dice = panels[(label, str(case.case_id))]
            ax.imshow(rgb, interpolation="none")
            ax.set_axis_off()
            ax.text(
                0.97,
                0.97,
                f"Dice = {dice:.2f}",
                color="white",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=ps.TICK_PT,
                path_effects=[withStroke(linewidth=1.0, foreground="black")],
            )
            if i == 0:
                ax.set_title(markers(case), fontsize=ps.TICK_PT, pad=2)
            if j == 0:
                ax.text(
                    -0.06,
                    0.5,
                    label,
                    rotation=90,
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=ps.TICK_PT,
                )
            if i == n_r - 1 and j == 0:
                C.scale_bar(ax, rgb.shape[1], um_px * side / rgb.shape[1], frac=0.35)
    ps.save(fig, str(stem), data=pd.DataFrame(records))
    print(f"-> {stem}.pdf/.png/.svg, _source_data.csv, one PNG per panel")


if __name__ == "__main__":
    main()

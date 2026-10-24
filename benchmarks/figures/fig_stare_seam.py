"""Two-colour DAPI overlay of one round pair after STARE, with an inset across a tile
boundary and single-cell insets (reference cyan, registered moving magenta).

    python -m benchmarks.figures.fig_stare_seam --cases <REGBENCH_CASES> \
        --out <REGBENCH_OUT> --case-id <case> --variant stare --tile 2048 \
        --outdir figures/seam

THE IMAGE is the result's own written slide on the reference canvas at its ``origin_xy``
(``common.written_nuclear``, as regbench's tissue scorer reads it), against the case's
reference nuclear channel: the same placement as fig_overlay's "After".

THE TILES. STARE partitions the reference canvas into square cores of ``--tile`` px from
(0, 0) (``stare.tile_grid``: edges at 0, tile, 2 x tile, ...); each core is registered by
its own REG_TILE task. So the seams are the lines x = k x tile and y = k x tile of the
reference canvas. ``--tile`` must be the tier the result ran with (``high`` = 2048,
``methods/stare.py`` TIERS); it is not read from the result.

THE BOUNDARY INSET (``--inset-um``, the square) is centred on a CORNER where four tiles
meet, so both seams cross it. The corner chosen is the interior one with the most
reference nuclei inside the inset (the case's own nuclei, ``points.npz``); ties go to the
top-left. ``--corner X,Y`` (reference px) pins it. Every candidate is in
``<stem>_candidates.csv``. Small ticks on the inset's frame mark where the seams cross it;
nothing is drawn over the pixels unless ``--seam-lines``.

THE CELL INSETS (``--cells``, ``--cell-um``): reference nuclei inside the boundary inset,
one per tile of the four where possible, each the nucleus closest to a seam whose
nearest neighbour is at least half a cell inset away (an isolated nucleus reads as one
cell). They are numbered on the boundary inset. ``--cell X,Y`` (repeatable) pins them.

DISPLAY as fig_overlay: the overview linearly windowed to the (0.5, 99.8) tissue
percentiles; insets per channel: window inside the inset, CLAHE, unsharp mask, cubic
upsampling (``common.crisp``). Scale bars: the overview ``--overview-bar-um`` (1 mm), the
insets a 1-2-5 length printed on stdout and in the source data for the legend.

Outputs in --outdir, <stem> = seam_<case>_<variant>: .pdf/.png/.svg, _source_data.csv
(corner, cells, scale bars), _candidates.csv, and _overview.png, _boundary.png,
_cell<k>.png at their own resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps


def ref_centroids(case):
    """Reference nucleus centroids (x, y px): the mean vertex of each outer ring."""
    from benchmarks.regbench.cases import load_points

    pts = load_points(case)
    for prefix in ("nuclei", "cells"):
        xy = pts.get(f"{prefix}_ref_xy")
        off = pts.get(f"{prefix}_ref_ring_off")
        if xy is None or off is None or not len(xy):
            continue
        hole = pts.get(f"{prefix}_ref_ring_hole")
        off = np.asarray(off)
        if off.size and off[-1] != len(xy):
            off = np.append(off, len(xy))
        sums = np.add.reduceat(xy, off[:-1], axis=0)
        cnt = np.diff(off)[:, None]
        c = sums / np.maximum(cnt, 1)
        if hole is not None and len(hole) == len(c):
            c = c[~np.asarray(hole, bool)]
        return c
    raise SystemExit(f"case {case.case_id} has no reference nuclei in points.npz")


def corners(hw, tile, cen, win):
    """Every interior tile corner and the number of reference nuclei in the inset."""
    H, W = hw
    rows = []
    for y in range(tile, H, tile):
        for x in range(tile, W, tile):
            inside = (np.abs(cen[:, 0] - x) < win / 2) & (
                np.abs(cen[:, 1] - y) < win / 2
            )
            rows.append({"corner_x": x, "corner_y": y, "n_nuclei": int(inside.sum())})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            f"the slide ({W} x {H} px) has no interior corner of {tile} px tiles"
        )
    return df.sort_values(
        ["n_nuclei", "corner_y", "corner_x"], ascending=[False, True, True]
    )


def pick_cells(cen, cx, cy, win, cell, n):
    """Up to ``n`` isolated nuclei inside the inset, one per quadrant first, each the
    closest to a seam."""
    from scipy.spatial import cKDTree

    m = (np.abs(cen[:, 0] - cx) < win / 2 - cell / 2) & (
        np.abs(cen[:, 1] - cy) < win / 2 - cell / 2
    )
    c = cen[m]
    if not len(c):
        return []
    d, _ = cKDTree(cen).query(c, k=2)
    c = c[d[:, 1] >= cell / 2]
    seam = np.minimum(np.abs(c[:, 0] - cx), np.abs(c[:, 1] - cy))
    quad = (c[:, 0] >= cx).astype(int) + 2 * (c[:, 1] >= cy).astype(int)
    order = np.argsort(seam)
    chosen, used = [], set()
    for rnd in (0, 1):
        for i in order:
            if len(chosen) >= n:
                break
            if any(np.hypot(*(c[i] - c[j])) < cell for j in chosen):
                continue
            if rnd == 0 and quad[i] in used:
                continue
            chosen.append(i)
            used.add(quad[i])
    return [
        (float(c[i, 0]), float(c[i, 1]), float(seam[i]), int(quad[i])) for i in chosen
    ]


def compose(
    ov, factor, boundary, cells, cell_xy, corner, win, cell_px, tile, um_px, a, bars
):
    from matplotlib.patches import Rectangle
    from matplotlib.patheffects import withStroke

    gap, width, n = 2.0, 180.0, max(len(cells), 1)
    aspect = ov.shape[0] / ov.shape[1]
    k = 1 + 1 / n
    w = ((width - gap) * k + gap / n) / (aspect + k)
    z = width - gap - w
    c = (z - (n - 1) * gap) / n
    h = max(w * aspect, z + gap + c)
    if h > ps.MAX_HEIGHT_MM:
        raise SystemExit(f"the figure would be {h:.0f} mm tall")
    fig = C.fixed_figure(width, h)

    def at(x, y, ww, hh):
        return (x / width, 1 - (y + hh) / h, ww / width, hh / h)

    stroke = [withStroke(linewidth=1.2, foreground="black")]
    ax = fig.add_axes(at(0, 0, w, w * aspect))
    ax.imshow(ov, interpolation="lanczos")
    ax.set_axis_off()
    span_um = ov.shape[1] * um_px * factor
    want = a.overview_bar_um if a.overview_bar_um <= 0.6 * span_um else None
    if want is None:
        print(
            f"a {a.overview_bar_um:g} µm bar does not fit a {span_um:.0f} µm slide: a shorter one"
        )
    bars["overview_um"] = C.scale_bar(ax, ov.shape[1], um_px * factor, length_um=want)
    ax.set_xlim(-0.5, ov.shape[1] - 0.5)
    ax.set_ylim(ov.shape[0] - 0.5, -0.5)
    if a.grid:
        H, W = ov.shape[:2]
        for x in np.arange(tile, W * factor, tile) / factor:
            ax.axvline(x, color="white", lw=0.25, ls=":", alpha=0.6)
        for y in np.arange(tile, H * factor, tile) / factor:
            ax.axhline(y, color="white", lw=0.25, ls=":", alpha=0.6)
    for txt, col, yy in (
        ("reference DAPI", C.CYAN, 0.06),
        (f"moving DAPI after {a.label}", C.MAGENTA, 0.02),
    ):
        ax.text(
            0.985,
            yy,
            txt,
            color=col,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=ps.BASE_PT,
        )
    cx, cy = corner
    bx, by = (cx - win / 2) / factor, (cy - win / 2) / factor
    ax.add_patch(
        Rectangle((bx, by), win / factor, win / factor, fill=False, ec="white", lw=0.6)
    )

    axb = fig.add_axes(at(w + gap, 0, z, z))
    axb.imshow(boundary, interpolation="none")
    axb.set_axis_off()
    axb.add_patch(
        Rectangle((0, 0), 1, 1, transform=axb.transAxes, fill=False, ec="white", lw=0.8)
    )
    up = boundary.shape[1] / win
    bars["boundary_um"] = C.scale_bar(axb, boundary.shape[1], um_px / up, frac=0.25)
    mid = boundary.shape[1] / 2
    tick = boundary.shape[1] * 0.035
    for s in (-1, 1):  # seam ticks on the frame, outside the tissue
        e = 0 if s < 0 else boundary.shape[1]
        axb.plot([mid, mid], [e, e - s * tick], color="white", lw=1.0)
        axb.plot([e, e - s * tick], [mid, mid], color="white", lw=1.0)
    if a.seam_lines:
        axb.axvline(mid, color="white", lw=0.4, ls=":")
        axb.axhline(mid, color="white", lw=0.4, ls=":")
    axb.text(
        0.04,
        0.96,
        "tile boundary",
        color="white",
        transform=axb.transAxes,
        ha="left",
        va="top",
        fontsize=ps.BASE_PT,
        path_effects=stroke,
    )
    for i, (x, y, *_) in enumerate(cell_xy):
        px0 = (x - (cx - win / 2) - cell_px / 2) * up
        py0 = (y - (cy - win / 2) - cell_px / 2) * up
        axb.add_patch(
            Rectangle(
                (px0, py0), cell_px * up, cell_px * up, fill=False, ec="white", lw=0.5
            )
        )
        axb.text(
            px0,
            py0 - 2,
            str(i + 1),
            color="white",
            ha="left",
            va="bottom",
            fontsize=ps.TICK_PT,
            path_effects=stroke,
        )
    for i, rgb in enumerate(cells):
        axc = fig.add_axes(at(w + gap + i * (c + gap), z + gap, c, c))
        axc.imshow(rgb, interpolation="none")
        axc.set_axis_off()
        axc.add_patch(
            Rectangle(
                (0, 0), 1, 1, transform=axc.transAxes, fill=False, ec="white", lw=0.6
            )
        )
        b = C.scale_bar(axc, rgb.shape[1], um_px * cell_px / rgb.shape[1], frac=0.4)
        bars["cell_um"] = b
        axc.text(
            0.06,
            0.94,
            str(i + 1),
            color="white",
            transform=axc.transAxes,
            ha="left",
            va="top",
            fontsize=ps.TICK_PT,
            path_effects=stroke,
        )
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True, help="REGBENCH_CASES")
    ap.add_argument("--out", required=True, help="REGBENCH_OUT (holds the results)")
    ap.add_argument("--dataset", default="multiplex")
    ap.add_argument(
        "--case-id", required=True, help="the round pair (the panels of Fig. 1c)"
    )
    ap.add_argument("--variant", default="stare", help="the STARE result (high tier)")
    ap.add_argument("--label", default="STARE", help="the method's name on the figure")
    ap.add_argument(
        "--tile", type=int, default=2048, help="STARE core size of that tier (px)"
    )
    ap.add_argument("--inset-um", type=float, default=250.0)
    ap.add_argument("--cells", type=int, default=4)
    ap.add_argument("--cell-um", type=float, default=20.0)
    ap.add_argument("--corner", help="X,Y: pin the tile corner (reference px)")
    ap.add_argument(
        "--cell", action="append", default=[], help="X,Y: pin a cell (reference px)"
    )
    ap.add_argument("--overview-px", type=int, default=3000)
    ap.add_argument("--overview-bar-um", type=float, default=1000.0)
    ap.add_argument("--clip", default="1,99.8")
    ap.add_argument("--clahe", type=float, default=0.01)
    ap.add_argument("--sharpen", type=float, default=0.6)
    ap.add_argument("--upsample", type=int, default=3)
    ap.add_argument("--cell-upsample", type=int, default=8)
    ap.add_argument(
        "--grid", action="store_true", help="dotted tile grid on the overview"
    )
    ap.add_argument(
        "--seam-lines", action="store_true", help="dotted seams on the inset"
    )
    ap.add_argument("--outdir", default="figures/seam")
    a = ap.parse_args(argv)

    from stare.slide_io import decimation_factor, open_lazy, read_decimated

    from benchmarks.regbench.cases import list_cases

    ps.apply_style()
    case = next(
        (c for c in list_cases(a.cases, a.dataset) if str(c.case_id) == a.case_id), None
    )
    if case is None:
        raise SystemExit(f"no {a.dataset} case {a.case_id!r}")
    um_px = float(case.pixel_size_um or 0)
    if not um_px:
        raise SystemExit(f"case {case.case_id} has no pixel size")
    win = int(round(a.inset_um / um_px))
    cell_px = int(round(a.cell_um / um_px))
    clip = tuple(float(v) for v in a.clip.split(","))
    show = dict(clip=clip, clahe=a.clahe, sharpen=a.sharpen, upsample=a.upsample)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"seam_{case.case_id}_{a.variant}"

    path, ch, (ox, oy) = C.written_nuclear(a.out, a.variant, case)
    ref_arr, _, close_ref = open_lazy(case.ref_nuclear)
    mov_arr, _, close_mov = open_lazy(path)
    try:
        hw = tuple(int(v) for v in ref_arr.shape[1:])
        cen = ref_centroids(case)
        cand = corners(hw, a.tile, cen, win)
        cand.to_csv(f"{stem}_candidates.csv", index=False)
        if a.corner:
            corner = tuple(float(v) for v in a.corner.split(","))
        else:
            r = cand.iloc[0]
            corner = (float(r["corner_x"]), float(r["corner_y"]))
        cx, cy = corner
        if a.cell:
            cell_xy = [
                (*(float(v) for v in spec.split(",")), np.nan, -1) for spec in a.cell
            ]
        else:
            cell_xy = pick_cells(cen, cx, cy, win, cell_px, a.cells)
        print(f"corner ({cx:.0f}, {cy:.0f}) px, {len(cell_xy)} cell inset(s)")

        def pair(x0, y0, side, disp):
            r = C.region(ref_arr, 0, y0, x0, side, side)
            m = C.region(
                mov_arr, ch, y0 - int(round(oy)), x0 - int(round(ox)), side, side
            )
            return C.composite(C.crisp(r, **disp), C.crisp(m, **disp))

        boundary = pair(int(cx - win / 2), int(cy - win / 2), win, show)
        cshow = {**show, "upsample": a.cell_upsample, "clahe": 0.0}
        cells = [
            pair(int(x - cell_px / 2), int(y - cell_px / 2), cell_px, cshow)
            for x, y, *_ in cell_xy
        ]

        f = decimation_factor([hw, tuple(mov_arr.shape[1:])], a.overview_px)
        ref_t = read_decimated(ref_arr, 0, f)
        mov_t = read_decimated(mov_arr, ch, f)
        sy, sx = int(round(oy / f)), int(round(ox / f))
        mov_t = C.on_canvas(
            np.pad(mov_t, ((max(sy, 0), 0), (max(sx, 0), 0)))[
                max(-sy, 0) :, max(-sx, 0) :
            ],
            ref_t.shape,
        )
        tissue = ref_t > np.percentile(ref_t, 50)
        ov = C.composite(
            C.stretch(ref_t, (0.5, 99.8), tissue), C.stretch(mov_t, (0.5, 99.8), tissue)
        )
    finally:
        close_mov()
        close_ref()

    ps.plt.imsave(f"{stem}_overview.png", ov)
    ps.plt.imsave(f"{stem}_boundary.png", boundary)
    for i, rgb in enumerate(cells):
        ps.plt.imsave(f"{stem}_cell{i + 1}.png", rgb)
    bars = {}
    fig = compose(
        ov, f, boundary, cells, cell_xy, corner, win, cell_px, a.tile, um_px, a, bars
    )
    rows = [
        {
            "item": "corner",
            "x_px": cx,
            "y_px": cy,
            "side_um": a.inset_um,
            "tile_px": a.tile,
        }
    ]
    rows += [
        {
            "item": f"cell{i + 1}",
            "x_px": x,
            "y_px": y,
            "side_um": a.cell_um,
            "dist_to_seam_um": d * um_px if d == d else np.nan,
            "quadrant": q,
        }
        for i, (x, y, d, q) in enumerate(cell_xy)
    ]
    rows += [{"item": f"scale_bar_{k}", "length_um": v} for k, v in bars.items()]
    for r in rows:
        r.update(case_id=case.case_id, variant=a.variant, origin_x=ox, origin_y=oy)
    ps.save(fig, str(stem), data=pd.DataFrame(rows))
    print("scale bars (µm): " + ", ".join(f"{k} {v:g}" for k, v in bars.items()))
    print(
        f"-> {stem}.pdf/.png/.svg, _source_data.csv, _candidates.csv, _overview.png, _boundary.png, _cell*.png"
    )


if __name__ == "__main__":
    main()

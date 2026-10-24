"""DAPI overlay of one round pair (reference cyan, moving magenta) with three zooms
chosen automatically: before registration, and optionally after (Fig. 4b).

    python -m benchmarks.figures.fig_overlay --cases <REGBENCH_CASES> --out <REGBENCH_OUT> \
        --tables <REGBENCH_OUT>/tables --zones-from valis120_hires --after stare \
        --outdir figures/overlay

BEFORE: both slides on the SAME pixel grid with no registration (the moving slide padded
or cropped at the origin, as the pipeline's QC composite places it), so every
misalignment on the page is one a method must remove. White = the two agree.
AFTER (``--after VARIANT``): the reference against that result's written image, placed on
the reference canvas at its ``origin_xy`` (as regbench's tissue scorer reads it); the same
layout and the same zones, so the two files sit side by side.

Layout: the whole pair on the left; three square zooms stacked on the right, each boxed
on the overview and numbered like it (``--connectors`` adds thin lines), and labelled:

  Translation  the window where the slides are offset rigidly (large shift, little
               rotation or non-rigid residual)
  Rotation     the window where a local rotation dominates (near the rotation's centre,
               where the shift is small and the misalignment grows from the middle out)
  Deformation  the window where what no rigid motion explains dominates

THE CASE (``--case-id``, else chosen): the round pair closest to the cohort median Dice of
every result in ``--median-of`` (``cells.csv``; ``common.representative``): a typical
case, not the best one. The choice and its table are in the source data.

THE ZONES. The pair has no ground truth, so the misalignment field is a registration
RESULT's (``--zones-from``): its ``warped.npz`` holds where it moved every nucleus outline,
so ``u = moved centroid - native centroid`` is that method's estimate of the offset at
every nucleus. On a grid of windows the size of a zoom (``--zoom-um``, step half a window,
at least ``--min-cells`` nuclei) a rigid fit (Kabsch) gives three displacements in px:

  T = |mean u|                the shift at the window
  R = |angle| x window / 2    what the window's rotation moves its edge by
  D = RMS residual of the fit what no rigid motion explains (non-rigid + scale)

A window's score for a kind is that kind's displacement minus the largest of the other two
(it must DOMINATE); among the tissue-rich windows (at least the median nucleus count),
the best per kind is taken, no two overlapping. Every candidate
is in ``<stem>_candidates.csv``; ``--zone rotation=X,Y`` (reference px, the centre) pins
one. The labels give the numbers in µm and degrees: the chosen result's estimate.

DISPLAY. Overview: each channel linearly windowed to its (0.5, 99.8) percentiles over the
tissue. Zooms (``common.crisp``): the window computed inside each zoom (``--clip``), CLAHE
(``--clahe``), an unsharp mask (``--sharpen``) and a cubic x``--upsample`` for print,
identically on both channels and in both files. Non-linear: say so in the legend.

Outputs in --outdir, <stem> = overlay_<case>[_after_<variant>]: .pdf/.png/.svg, the
_source_data.csv (the zones), _candidates.csv, and for Affinity _overview.png and
_zoom<k>_<kind>.png at their own resolution. Run in the regbench scorer's image.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps

KINDS = ("translation", "rotation", "deformation")
TITLES = {
    "translation": "Translation",
    "rotation": "Rotation",
    "deformation": "Deformation",
}


# ── the misalignment field ────────────────────────────────────────────────────
def field_from_result(out, case, result, which="outlines"):
    """``(p, u)``: native centroids of the side the result moved, and its offsets."""
    from benchmarks.regbench import pairs
    from benchmarks.regbench.cases import load_points, polys, result_dir
    from benchmarks.regbench.mapping import MOV_TO_REF

    d = result_dir(out, result, case)
    if not (d / "warped.npz").exists():
        raise SystemExit(
            f"no {d}/warped.npz: {result} has no result for {case.case_id}"
        )
    with np.load(d / "warped.npz") as z:
        warped = {k: z[k] for k in z.files}
    key = "cells_xy" if which == "outlines" else "nuclei_xy"
    native = polys(
        load_points(case),
        "mov" if str(warped["direction"]) == MOV_TO_REF else "ref",
        which,
    )
    moved = pairs.with_xy(native, warped[key].astype(float))
    _, p = pairs.area_centroid(native)
    _, q = pairs.area_centroid(moved)
    ok = np.isfinite(p).all(1) & np.isfinite(q).all(1)
    return p[ok], q[ok] - p[ok]


def rigid_fit(p, q):
    """Kabsch p -> q: ``(angle_rad, rms_residual)``."""
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    if np.linalg.det(vt.T @ u.T) < 0:
        vt[-1] *= -1
    r = vt.T @ u.T
    res = qc - pc @ r.T
    return math.atan2(r[1, 0], r[0, 0]), float(np.sqrt((res**2).sum(1).mean()))


def candidates(p, u, hw, win, min_cells):
    """Every window of the grid: centre, n cells, T, R, D (px), angle, the three scores."""
    from scipy.spatial import cKDTree

    tree, rows, half = cKDTree(p), [], win / 2
    for cy in range(win // 2, hw[0] - win // 2 + 1, max(win // 2, 1)):
        for cx in range(win // 2, hw[1] - win // 2 + 1, max(win // 2, 1)):
            idx = np.array(tree.query_ball_point([cx, cy], half, p=np.inf), dtype=int)
            if len(idx) < min_cells:
                continue
            ang, d = rigid_fit(p[idx], p[idx] + u[idx])
            rows.append(
                dict(
                    cx=cx,
                    cy=cy,
                    n_cells=len(idx),
                    T_px=float(np.hypot(*u[idx].mean(0))),
                    R_px=abs(ang) * half,
                    D_px=d,
                    angle_deg=math.degrees(ang),
                )
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            f"no {win} px window holds {min_cells} nuclei: lower --min-cells"
        )
    # a zoom shows tissue: at least the median nucleus count of the windows that have any
    df["tissue_rich"] = df["n_cells"] >= df["n_cells"].median()
    parts = {"translation": "T_px", "rotation": "R_px", "deformation": "D_px"}
    for kind, col in parts.items():
        df[f"score_{kind}"] = df[col] - df[[c for c in parts.values() if c != col]].max(
            1
        )
    return df


def choose(df, win, pinned):
    """The best non-overlapping window per kind, the pinned ones first."""
    chosen = {}
    for kind, (x, y) in pinned.items():
        near = df.loc[((df["cx"] - x) ** 2 + (df["cy"] - y) ** 2).idxmin()].to_dict()
        chosen[kind] = {**near, "cx": x, "cy": y, "pinned": True}
    for kind in KINDS:
        if kind in chosen:
            continue
        rich = df[df["tissue_rich"]]
        for _, r in rich.sort_values(
            [f"score_{kind}", "cy", "cx"], ascending=[False, True, True]
        ).iterrows():
            if all(
                abs(r["cx"] - c["cx"]) >= win or abs(r["cy"] - c["cy"]) >= win
                for c in chosen.values()
            ):
                chosen[kind] = {**r.to_dict(), "pinned": False}
                break
    missing = [k for k in KINDS if k not in chosen]
    if missing:
        raise SystemExit(
            f"no non-overlapping window left for {missing}: lower --zoom-um"
        )
    return chosen


# ── the figure ────────────────────────────────────────────────────────────────
def compose(ov, factor, zooms, zones, win, um_px, label, connectors=False):
    from matplotlib.patches import Rectangle
    from matplotlib.patheffects import withStroke

    gap, width = 2.0, 180.0  # mm
    aspect = ov.shape[0] / ov.shape[1]
    # overview width w, zoom side z: w + gap + z = width and w * aspect = 3 z + 2 gap
    z = (width - gap - 2 * gap / aspect) / (1 + 3 / aspect)
    w = width - gap - z
    h = w * aspect
    if h > ps.MAX_HEIGHT_MM:
        raise SystemExit(f"the overview would be {h:.0f} mm tall: too tall a slide")
    fig = C.fixed_figure(width, h)

    def at(x, y, ww, hh):  # mm from the top-left
        return (x / width, 1 - (y + hh) / h, ww / width, hh / h)

    ax_ov = fig.add_axes(at(0, 0, w, h))
    ax_ov.imshow(ov, interpolation="lanczos")
    ax_ov.set_axis_off()
    C.scale_bar(ax_ov, ov.shape[1], um_px * factor)
    ax_ov.text(
        0.015,
        0.985,
        label,
        color="white",
        transform=ax_ov.transAxes,
        ha="left",
        va="top",
        fontsize=ps.BASE_PT,
        fontweight="bold",
    )
    ax_ov.text(
        0.985,
        0.06,
        "reference DAPI",
        color=C.CYAN,
        transform=ax_ov.transAxes,
        ha="right",
        va="bottom",
        fontsize=ps.BASE_PT,
    )
    ax_ov.text(
        0.985,
        0.02,
        "moving DAPI",
        color=C.MAGENTA,
        transform=ax_ov.transAxes,
        ha="right",
        va="bottom",
        fontsize=ps.BASE_PT,
    )
    to_fig = ax_ov.transData + fig.transFigure.inverted()
    for k, kind in enumerate(KINDS):
        zone, rgb = zones[kind], zooms[kind]
        ax = fig.add_axes(at(w + gap, k * (z + gap), z, z))
        ax.imshow(rgb, interpolation="none")
        ax.set_axis_off()
        ax.add_patch(
            Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes, fill=False, ec="white", lw=0.8
            )
        )
        C.scale_bar(ax, rgb.shape[1], um_px * win / rgb.shape[1], frac=0.3)
        val = (
            f"{abs(zone['angle_deg']):.1f}°"
            if kind == "rotation"
            else f"{zone['T_px' if kind == 'translation' else 'D_px'] * um_px:.1f} µm"
        )
        ax.text(
            0.04,
            0.96,
            f"{k + 1}  {TITLES[kind]}\n    {val}",
            color="white",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ps.BASE_PT,
            path_effects=[withStroke(linewidth=1.2, foreground="black")],
        )
        bx, by = (zone["cx"] - win / 2) / factor, (zone["cy"] - win / 2) / factor
        bw = win / factor
        ax_ov.add_patch(Rectangle((bx, by), bw, bw, fill=False, ec="white", lw=0.6))
        ax_ov.text(
            bx + bw * 0.06,
            by + bw * 0.06,
            str(k + 1),
            color="white",
            ha="left",
            va="top",
            fontsize=ps.TICK_PT,
        )
        if (
            connectors
        ):  # thin lines from the box to its zoom (the numbers pair them too)
            (x1, y1), (x2, y2) = to_fig.transform([(bx + bw, by), (bx + bw, by + bw)])
            pos = ax.get_position()
            for (xa, ya), (xb, yb) in (
                ((x1, y1), (pos.x0, pos.y1)),
                ((x2, y2), (pos.x0, pos.y0)),
            ):
                fig.lines.append(
                    ps.plt.Line2D(
                        [xa, xb],
                        [ya, yb],
                        transform=fig.transFigure,
                        color=ps.LIGHT_GREY,
                        lw=0.4,
                    )
                )
    return fig


def pick_case(a, cases):
    if a.case_id:
        case = next((c for c in cases if str(c.case_id) == a.case_id), None)
        if case is None:
            raise SystemExit(f"no {a.dataset} case {a.case_id!r}")
        return case, None
    if not a.tables:
        raise SystemExit("give --case-id, or --tables to choose the median-Dice case")
    labels = [v.strip() for v in a.median_of.split(",")]
    cells = C.cells_table(a.tables, a.dataset)
    wide = cells.pivot_table(
        index=["group", "case_id"], columns="method", values="dice_matched"
    ).reset_index()
    wide["case_id"] = wide["case_id"].astype(str)
    cid, table = C.representative(wide, labels, by="case_id")
    print(f"case: {cid}, the closest to the median Dice of {labels}")
    return next(c for c in cases if str(c.case_id) == cid), table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True, help="REGBENCH_CASES")
    ap.add_argument("--out", required=True, help="REGBENCH_OUT (holds the results)")
    ap.add_argument("--tables", help="<out>/tables: to choose the case by median Dice")
    ap.add_argument("--dataset", default="multiplex")
    ap.add_argument("--case-id", help="pin the case (else the median-Dice one)")
    ap.add_argument(
        "--median-of",
        default="valis120_hires,stare",
        help="the results whose median Dice the case is chosen by",
    )
    ap.add_argument(
        "--zones-from",
        default="valis120_hires",
        help="the result whose moved nuclei give the misalignment field",
    )
    ap.add_argument("--after", help="also draw the pair after this result")
    ap.add_argument("--zoom-um", type=float, default=150.0, help="zoom side (µm)")
    ap.add_argument("--min-cells", type=int, default=40)
    ap.add_argument("--zone", action="append", default=[], help="KIND=X,Y (ref px)")
    ap.add_argument("--overview-px", type=int, default=3000)
    ap.add_argument("--clip", default="1,99.8", help="zoom window, percentiles LO,HI")
    ap.add_argument("--clahe", type=float, default=0.01, help="0 = off")
    ap.add_argument(
        "--sharpen", type=float, default=0.6, help="unsharp amount; 0 = off"
    )
    ap.add_argument("--upsample", type=int, default=3)
    ap.add_argument(
        "--connectors", action="store_true", help="thin lines from each box to its zoom"
    )
    ap.add_argument("--outdir", default="figures/overlay")
    a = ap.parse_args(argv)

    from stare.slide_io import decimation_factor, open_lazy, read_decimated

    from benchmarks.regbench.cases import list_cases

    ps.apply_style()
    case, table = pick_case(a, list_cases(a.cases, a.dataset))
    um_px = float(case.pixel_size_um or 0)
    if not um_px:
        raise SystemExit(f"case {case.case_id} has no pixel size")
    win = int(round(a.zoom_um / um_px))
    show = dict(
        clip=tuple(float(v) for v in a.clip.split(",")),
        clahe=a.clahe,
        sharpen=a.sharpen,
        upsample=a.upsample,
    )
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem0 = f"overlay_{case.case_id}"
    if table is not None:
        table.to_csv(outdir / f"{stem0}_case_choice.csv", index=False)

    ref_arr, _, close_ref = open_lazy(case.ref_nuclear)
    try:
        hw = tuple(ref_arr.shape[1:])
        p, u = field_from_result(a.out, case, a.zones_from)
        cand = candidates(p, u, hw, win, a.min_cells)
        pinned = {}
        for spec in a.zone:
            kind, _, xy = spec.partition("=")
            if kind not in KINDS:
                raise SystemExit(f"--zone {spec}: the kind is one of {KINDS}")
            pinned[kind] = tuple(float(v) for v in xy.split(","))
        zones = choose(cand, win, pinned)
        cand.to_csv(outdir / f"{stem0}_candidates.csv", index=False)
        rows = []
        for k, kind in enumerate(KINDS):
            zn = zones[kind]
            rows.append(
                {
                    "zoom": k + 1,
                    "kind": kind,
                    "case_id": case.case_id,
                    "zones_from": a.zones_from,
                    "centre_x_px": zn["cx"],
                    "centre_y_px": zn["cy"],
                    "side_px": win,
                    "side_um": a.zoom_um,
                    "n_cells": zn["n_cells"],
                    "shift_um": zn["T_px"] * um_px,
                    "rotation_deg": zn["angle_deg"],
                    "rotation_edge_um": zn["R_px"] * um_px,
                    "nonrigid_rms_um": zn["D_px"] * um_px,
                    "pinned": zn["pinned"],
                    **{f"display_{k2}": v for k2, v in show.items()},
                }
            )
            print(
                f"  zoom {k + 1} {kind:11}: centre ({zn['cx']:.0f}, {zn['cy']:.0f}) px, "
                f"shift {zn['T_px'] * um_px:.1f} µm, rotation {zn['angle_deg']:.2f}°, "
                f"non-rigid {zn['D_px'] * um_px:.1f} µm"
                + ("  (pinned)" if zn["pinned"] else "")
            )

        sides = [("Before", case.mov_nuclear, 0, (0.0, 0.0), stem0)]
        if a.after:
            path, ch, origin = C.written_nuclear(a.out, a.after, case)
            sides.append(
                (f"After ({a.after})", path, ch, origin, f"{stem0}_after_{a.after}")
            )
        for label, path, ch, (ox, oy), stem in sides:
            mov_arr, _, close_mov = open_lazy(path)
            try:
                f = decimation_factor([hw, tuple(mov_arr.shape[1:])], a.overview_px)
                ref_t = read_decimated(ref_arr, 0, f)
                mov_t = read_decimated(mov_arr, ch, f)
                # on the reference canvas: shifted by the written image's origin
                sy, sx = int(round(oy / f)), int(round(ox / f))
                mov_t = C.on_canvas(
                    np.pad(mov_t, ((max(sy, 0), 0), (max(sx, 0), 0)))[
                        max(-sy, 0) :, max(-sx, 0) :
                    ],
                    ref_t.shape,
                )
                tissue = ref_t > np.percentile(ref_t, 50)
                ov = C.composite(
                    C.stretch(ref_t, (0.5, 99.8), tissue),
                    C.stretch(mov_t, (0.5, 99.8), tissue),
                )
                zooms = {}
                for kind in KINDS:
                    y0 = int(zones[kind]["cy"]) - win // 2
                    x0 = int(zones[kind]["cx"]) - win // 2
                    r = C.region(ref_arr, 0, y0, x0, win, win)
                    m = C.region(
                        mov_arr, ch, y0 - int(round(oy)), x0 - int(round(ox)), win, win
                    )
                    zooms[kind] = C.composite(C.crisp(r, **show), C.crisp(m, **show))
            finally:
                close_mov()
            for k, kind in enumerate(KINDS):
                ps.plt.imsave(outdir / f"{stem}_zoom{k + 1}_{kind}.png", zooms[kind])
            ps.plt.imsave(outdir / f"{stem}_overview.png", ov)
            fig = compose(ov, f, zooms, zones, win, um_px, label, a.connectors)
            ps.save(fig, str(outdir / stem), data=pd.DataFrame(rows))
            print(
                f"-> {outdir / stem}.pdf/.png/.svg, _source_data.csv, _overview.png, "
                "_zoom*.png"
            )
    finally:
        close_ref()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""reg_zoom.py -- whole-slide DAPI overview with a zoom-in on segmented cells.

One figure per patient, from a mirage ``--outdir`` that ran segmentation:

    left   the reference slide's DAPI, whole slide, white, a white box on the zoom region
    right  that region at full resolution: DAPI in white with the pipeline's segmented
           cells outlined (colour and width settable), in a white frame, joined to the box
           by a translucent funnel
    bottom right  the channel names in their colours ("DAPI", the cells label)
    scale bars on both panels (µm, or mm from 1 mm up)

Inputs come from the run's ``csv/segmented.csv`` (lib/Checkpoint.groovy): the reference
row's ``registered_image`` (DAPI found by OME channel name) and ``cell_mask`` -- or
``nuclei_mask`` with ``--mask nuclei`` -- a label image on the same reference canvas.

Outputs in OUTDIR:
    <patient>_zoom.png/.pdf   the figure
    <patient>_zoom.json       zoom region (reference frame, full-res px), files, settings

    python -m benchmarks.reg_zoom results/segmentation -o figs/zoom --field-um 400 \\
        --outline-color yellow --outline-width 2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from benchmarks import reg_mosaic as rm

log = logging.getLogger("reg_zoom")

BAR_GREY = (0.8, 0.8, 0.8)
DAPI_WHITE = (1.0, 1.0, 1.0)


def read_segmented(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"{path}: not found -- did the run reach segmentation?")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = {
        "patient_id",
        "registered_image",
        "is_reference",
        "cell_mask",
        "nuclei_mask",
    }
    missing = need - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
    return rows


def reference_row(rows: list[dict], patient: str | None) -> dict:
    patients = sorted({r["patient_id"] for r in rows})
    pid = patient or (patients[0] if len(patients) == 1 else None)
    if pid is None:
        raise SystemExit(
            f"{len(patients)} patients in segmented.csv; pass --patient ({patients})"
        )
    refs = [
        r
        for r in rows
        if r["patient_id"] == pid
        and r["is_reference"].strip().lower() in ("true", "1", "yes")
    ]
    if len(refs) != 1:
        raise SystemExit(
            f"{pid}: expected one reference row in segmented.csv, found {len(refs)}"
        )
    return refs[0]


def outlines(labels: np.ndarray, width: int) -> np.ndarray:
    """Boolean cell contours, ``width`` px thick, inside each cell (neighbours stay apart)."""
    from skimage.segmentation import find_boundaries

    edge = find_boundaries(labels, mode="inner")
    if width > 1:
        edge = ndimage.binary_dilation(edge, iterations=width - 1) & (labels > 0)
    return edge


def white(plane: np.ndarray, pmin: float, pmax: float, gamma: float) -> np.ndarray:
    g = rm.stretch(plane, rm.percentile_limits(plane, pmin, pmax), gamma)
    return np.repeat(g[..., None], 3, axis=2)


def render(opt) -> dict:
    run = Path(opt.run)
    rows = read_segmented(opt.segmented_csv or run / "csv" / "segmented.csv")
    row = reference_row(rows, opt.patient)
    pid = row["patient_id"]
    image = rm.published_file(row["registered_image"])
    mask = Path(row["cell_mask" if opt.mask == "cell" else "nuclei_mask"])
    src, msrc = rm.TiffSource(image), rm.TiffSource(mask)
    ci = src.nuclear_index()
    if ci is None:
        raise SystemExit(
            f"{image.name}: no DAPI/Hoechst/CellTox among {src.channel_names}"
        )
    if msrc.shape != src.shape:
        raise SystemExit(
            f"{mask.name} is {msrc.shape}, {image.name} is {src.shape}: the mask is not on the "
            "reference canvas"
        )
    H, W = src.shape
    px = opt.pixel_size_um or rm._float_or_none(row.get("pixel_size")) or src.px
    if not px:
        raise SystemExit("pixel size unknown; pass --pixel-size-um")

    low, factor = src.overview(ci, opt.overview_px)
    field_px = min(int(round(opt.field_um / px)), H, W)
    if field_px > opt.max_zoom_px:
        raise SystemExit(
            f"a {opt.field_um:g} um zoom is {field_px} px at {px} um/px; the zoom is drawn at full "
            f"resolution to show cells, so keep it under --max-zoom-px {opt.max_zoom_px} "
            f"(~{opt.max_zoom_px * px:.0f} um)"
        )
    if opt.roi:
        y, x = (int(v) for v in opt.roi.split(",")[:2])
    else:
        picks = rm.select_rois(low, factor, field_px, 1, 0.0, 0.0, (H, W))
        if not picks:
            raise SystemExit(
                f"{pid}: no tissue region of {field_px} px found; pass --roi"
            )
        y, x = picks[0]
    log.info("%s: zoom y=%d x=%d size=%d px (%s um/px)", pid, y, x, field_px, px)

    dapi = src.read_patch(ci, y, x, field_px, field_px)
    labels = msrc.read_patch(0, y, x, field_px, field_px)
    zoom = white(dapi, opt.pmin, opt.pmax, opt.gamma)
    edge = outlines(labels, opt.outline_width)
    zoom[edge] = _rgb(opt.outline_color)
    over = white(low, opt.pmin, opt.pmax, opt.gamma)

    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"{pid}_zoom"
    formats = [f.strip() for f in opt.formats.split(",") if f.strip()]
    bars = draw_figure(over, factor, zoom, (y, x, field_px), px, opt, stem, formats)
    manifest = {
        "patient": pid,
        "image": str(image),
        "mask": str(mask),
        "mask_kind": opt.mask,
        "pixel_size_um": px,
        "zoom": {"y": y, "x": x, "size_px": field_px, "size_um": field_px * px},
        "overview_px_per_output_px": factor,
        "outline": {"color": opt.outline_color, "width_px": opt.outline_width},
        "cells_in_zoom": int(len(np.setdiff1d(np.unique(labels), [0]))),
        "scalebars": bars,
        "stretch": {"pmin": opt.pmin, "pmax": opt.pmax, "gamma": opt.gamma},
    }
    (outdir / f"{pid}_zoom.json").write_text(json.dumps(manifest, indent=2))
    src.close()
    msrc.close()
    return manifest


def _rgb(color) -> tuple[float, float, float]:
    import matplotlib.colors

    return tuple(float(c) for c in matplotlib.colors.to_rgb(color))


def draw_figure(over, factor, zoom, box, px, opt, stem: Path, formats) -> dict:
    """Overview left, framed zoom top right, funnel between, legend bottom right."""
    plt = rm._mpl()
    from matplotlib.patches import Polygon, Rectangle

    ho, wo = over.shape[:2]
    z = zoom.shape[0]
    gap = int(0.06 * z)
    # the zoom panel's size in the figure: a fraction of the overview's height, as in a
    # classic overview/zoom panel; drawn with nearest-neighbour, so outlines stay crisp
    zoom_disp = int(opt.zoom_fraction * ho)
    fig_w, fig_h = (
        wo + gap + zoom_disp + gap,
        max(ho, zoom_disp + int(0.35 * zoom_disp)),
    )
    dpi = opt.dpi
    fig = plt.figure(figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi, facecolor="black")

    ax_o = fig.add_axes([0, (fig_h - ho) / fig_h, wo / fig_w, ho / fig_h])
    ax_o.imshow(over, interpolation="none")
    ax_o.set_axis_off()
    zx = (wo + gap) / fig_w
    zy = (fig_h - gap * 0.5 - zoom_disp) / fig_h
    ax_z = fig.add_axes([zx, zy, zoom_disp / fig_w, zoom_disp / fig_h])
    ax_z.imshow(zoom, interpolation="none")
    ax_z.set_xticks([])
    ax_z.set_yticks([])
    for spine in ax_z.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(max(1.5, zoom_disp / 400))

    y, x, size = box
    bx, by, bs = x / factor, y / factor, size / factor
    ax_o.add_patch(
        Rectangle((bx, by), bs, bs, fill=False, ec="white", lw=max(1.5, ho / 800))
    )
    # funnel: the box's left corners to the zoom frame's left corners, in figure coordinates
    to_fig = fig.transFigure.inverted()
    tl = to_fig.transform(ax_o.transData.transform((bx, by)))
    bl = to_fig.transform(ax_o.transData.transform((bx, by + bs)))
    tr = to_fig.transform(ax_o.transData.transform((bx + bs, by)))
    br = to_fig.transform(ax_o.transData.transform((bx + bs, by + bs)))
    fig.patches.append(
        Polygon(
            [tl, tr, (zx, zy + zoom_disp / fig_h), (zx, zy), br, bl],
            closed=True,
            transform=fig.transFigure,
            facecolor=(1, 1, 1, opt.funnel_alpha),
            edgecolor="none",
            zorder=0.5,
        )
    )

    font = max(10.0, fig_h / dpi * 3.2)
    bar_over = rm.auto_scalebar_um(wo * factor * px)
    rm.draw_scalebar(
        ax_o,
        ho,
        wo,
        bar_over / (factor * px),
        rm.scalebar_label(bar_over),
        font * 0.8,
        thick=0.006,
        color=BAR_GREY,
    )
    bar_zoom = rm.auto_scalebar_um(z * px)
    rm.draw_scalebar(
        ax_z,
        z,
        z,
        bar_zoom / px,
        rm.scalebar_label(bar_zoom),
        font * 0.6,
        thick=0.008,
        color=BAR_GREY,
    )
    legend_ax = fig.add_axes([zx, 0.0, zoom_disp / fig_w, max(0.02, zy - 0.02)])
    legend_ax.set_axis_off()
    rm.draw_legend(
        legend_ax,
        [("DAPI", DAPI_WHITE), (opt.cells_label, _rgb(opt.outline_color))],
        font * 1.3,
        x=1.0,
        y=0.05,
    )
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=dpi, facecolor="black")
    plt.close(fig)
    return {"overview_um": bar_over, "zoom_um": bar_zoom}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run",
        type=Path,
        metavar="RUN_DIR",
        help="a mirage --outdir that ran segmentation",
    )
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument(
        "--segmented-csv",
        type=Path,
        default=None,
        help="default: RUN_DIR/csv/segmented.csv",
    )
    ap.add_argument("--patient", default=None)
    ap.add_argument(
        "--mask",
        choices=("cell", "nuclei"),
        default="cell",
        help="which label image to outline",
    )
    ap.add_argument("--field-um", type=float, default=300.0, help="zoom side in µm")
    ap.add_argument(
        "--roi",
        default=None,
        metavar="Y,X",
        help="zoom top-left, reference frame, full-res px",
    )
    ap.add_argument("--max-zoom-px", type=int, default=4096)
    ap.add_argument(
        "--overview-px", type=int, default=2400, help="overview long side in output px"
    )
    ap.add_argument(
        "--outline-color",
        default="#ffd400",
        help="any matplotlib colour: yellow, #00ff00, ...",
    )
    ap.add_argument(
        "--outline-width", type=int, default=1, help="outline thickness in image px"
    )
    ap.add_argument(
        "--cells-label", default="cells", help="legend name for the outlines"
    )
    ap.add_argument("--funnel-alpha", type=float, default=0.25)
    ap.add_argument(
        "--zoom-fraction",
        type=float,
        default=0.62,
        help="zoom panel side as a fraction of the overview height",
    )
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.8)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--pixel-size-um", type=float, default=None)
    ap.add_argument(
        "--dpi", type=int, default=100, help="output px per inch (image px are 1:1)"
    )
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None) -> int:
    opt = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if opt.verbose else logging.INFO, format="%(message)s"
    )
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    m = render(opt)
    log.info(
        "wrote %s_zoom.{%s} (%d cells in the zoom) in %s",
        m["patient"],
        opt.formats,
        m["cells_in_zoom"],
        opt.outdir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

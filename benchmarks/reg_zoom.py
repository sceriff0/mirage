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
``nuclei_mask`` with ``--mask nuclei``, or BOTH with ``--mask both``, which outlines the
cells in ``--outline-color`` and the nuclei in ``--nuclei-color`` in one image (nuclei
drawn last, so the inner outline wins where they touch). Each is a label image on the same
reference canvas. Which backend produced them is the RUN's business, not this tool's:
submit_zoom.sh's SEG_METHOD (stardist | instantseg | cellsam) chooses it.

Outputs in OUTDIR:
    <patient>_zoom.png/.pdf   the figure
    <patient>_zoom.json       zoom region (reference frame, full-res px), files, settings
    <patient>_crop.png/.pdf   --crop also|only: the outlined crop ALONE, no overview, no
                              funnel, no legend, at exactly --crop-px square (default: its
                              own pixels), for a figure that supplies its own layout.
                              --field-um sets how much TISSUE it covers, --crop-px how big
                              the file is; --crop-plain drops even the scale bar.

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

BAR_GREY = rm.BAR_GREY  # one definition; the layout lives in reg_mosaic
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
    # --mask both outlines the cell AND the nuclear label image, each in its own colour;
    # segmented.csv carries them side by side, both on the reference canvas
    kinds = ("cell", "nuclei") if opt.mask == "both" else (opt.mask,)
    masks = {}
    for kind in kinds:
        col = f"{'cell' if kind == 'cell' else 'nuclei'}_mask"
        if not row.get(col):
            raise SystemExit(
                f"{pid}: segmented.csv has no {col} for the reference row; "
                f"--mask {opt.mask} needs it"
            )
        masks[kind] = Path(row[col])
    src = rm.TiffSource(image)
    msrcs = {k: rm.TiffSource(v) for k, v in masks.items()}
    mask = masks[kinds[0]]
    msrc = msrcs[kinds[0]]
    csv_channels = [c for c in (row.get("channels") or "").split("|") if c]
    ci = src.nuclear_index(csv_channels)
    if ci is None:
        raise SystemExit(
            f"{image.name}: no DAPI/Hoechst/CellTox among "
            f"{src.channel_names or csv_channels}"
        )
    for kind, m in msrcs.items():
        if m.shape != src.shape:
            raise SystemExit(
                f"{masks[kind].name} is {m.shape}, {image.name} is {src.shape}: the mask is "
                "not on the reference canvas"
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
    zoom = white(dapi, opt.pmin, opt.pmax, opt.gamma)
    colors = {"cell": opt.outline_color, "nuclei": opt.nuclei_color}
    counts, drawn = {}, {}
    # cells first, nuclei over them: a nucleus lies inside its cell, so the inner outline
    # must win where the two touch
    for kind in kinds:
        lab = msrcs[kind].read_patch(0, y, x, field_px, field_px)
        counts[kind] = int(len(np.setdiff1d(np.unique(lab), [0])))
        zoom[outlines(lab, opt.outline_width)] = _rgb(colors[kind])
        drawn[kind] = {"file": str(masks[kind]), "color": colors[kind]}
    over = white(low, opt.pmin, opt.pmax, opt.gamma)

    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"{pid}_zoom"
    formats = [f.strip() for f in opt.formats.split(",") if f.strip()]
    legend = [
        (opt.cells_label if kind == "cell" else opt.nuclei_label, _rgb(colors[kind]))
        for kind in kinds
    ]
    crop_px = None
    if opt.crop != "none":
        # the outlined crop on its own, for a figure that supplies its own layout -- but
        # still labelled: the method top left, the channel and the objects bottom right
        crop_px = write_crop(
            zoom,
            px,
            outdir / f"{pid}_crop",
            formats,
            opt.dpi,
            opt.crop_px,
            not opt.crop_plain,
            legend=() if opt.crop_plain else [(opt.channel_label, DAPI_WHITE), *legend],
            title="" if opt.crop_plain else opt.title,
        )
    bars = (
        None
        if opt.crop == "only"
        else draw_figure(
            over, factor, zoom, (y, x, field_px), px, opt, stem, formats, legend
        )
    )
    manifest = {
        "patient": pid,
        "image": str(image),
        "mask": str(mask),
        "mask_kind": opt.mask,
        "masks": drawn,
        # which plane was drawn, and where its name came from: the slide's own OME header, or
        # segmented.csv when the slide is anonymous (TiffSource.nuclear_index)
        "channel_index": ci,
        "channel_names_from": "ome" if src.channel_names else "checkpoint",
        "pixel_size_um": px,
        "zoom": {"y": y, "x": x, "size_px": field_px, "size_um": field_px * px},
        "overview_px_per_output_px": factor,
        "outline": {"color": opt.outline_color, "width_px": opt.outline_width},
        "cells_in_zoom": counts[kinds[0]],
        "objects_in_zoom": counts,
        "crop_only_file": None
        if opt.crop == "none"
        else {
            "stem": f"{pid}_crop",
            "output_px": crop_px,
            "scalebar": not opt.crop_plain,
            "title": "" if opt.crop_plain else opt.title,
            "labelled": not opt.crop_plain,
        },
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


def write_crop(
    img,
    px: float,
    out_stem: Path,
    formats,
    dpi: int,
    size_px=None,
    scalebar=True,
    legend=(),
    title: str = "",
) -> int:
    """The outlined crop ALONE, at an exact output size -- no overview, no funnel.

    For a figure that supplies its own layout: the file is exactly ``size_px`` square (the
    crop's own pixels when it is not given), drawn nearest-neighbour so a one-pixel outline
    stays one pixel. It is annotated the way every other figure here is -- the method top
    left, the channel and the outlined objects in their colours bottom right, a scale bar --
    so a crop lifted into a panel still says what it shows. ``--crop-plain`` drops all of it.
    Returns the side actually written.
    """
    plt = rm._mpl()
    n = int(size_px or img.shape[0])
    fig = plt.figure(figsize=(n / dpi, n / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    font = max(8.0, n / dpi * 3.0)
    if title:
        rm._outline(
            ax.text(
                0.03,
                0.97,
                title,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=font * 1.15,
                color="white",
                fontweight="bold",
            )
        )
    if legend:
        # the crop has no margin (the axes fill the figure), so the stack starts a little
        # higher than the default 0.03: at 1024 px the bottom entry sat on the edge
        rm.draw_legend(ax, list(legend), font, y=0.045)
    if scalebar:
        # in the IMAGE's data coordinates, not the output size: draw_scalebar places the bar
        # at 0.05 x w of the axes' data range, so passing n instead put it off the image
        # entirely whenever --crop-px differed from the crop (measured at 1024 px on a 461 px
        # crop, 2026-09-20). Matplotlib scales it to the output for us.
        h, w = img.shape[:2]
        bar_um = rm.auto_scalebar_um(h * px)
        rm.draw_scalebar(
            ax,
            h,
            w,
            bar_um / px,
            rm.scalebar_label(bar_um),
            font,
            thick=0.008,
            color=BAR_GREY,
        )
    for fmt in formats:
        # bbox/pad pinned, not inherited: benchmarks/analysis/lib/plotting.py's paper theme
        # sets savefig.bbox="tight" globally, and anything that has called it in this process
        # would otherwise trim and pad the file -- measured 522 px for --crop-px 512.
        fig.savefig(
            f"{out_stem}.{fmt}",
            dpi=dpi,
            facecolor="black",
            bbox_inches=None,
            pad_inches=0,
        )
    plt.close(fig)
    return n


def draw_figure(over, factor, zoom, box, px, opt, stem: Path, formats, legend) -> dict:
    """Overview left, framed zoom top right, funnel between, legend bottom right.

    The layout itself is reg_mosaic.draw_overview_zoom, shared with reg_overlay's --zoom-um
    panels so the two figures keep one style; this supplies the segmentation figure's own
    legend (DAPI + the outlined cells).
    """
    return rm.draw_overview_zoom(
        over,
        zoom,
        box,
        factor,
        px,
        stem,
        formats,
        dpi=opt.dpi,
        zoom_fraction=opt.zoom_fraction,
        funnel_alpha=opt.funnel_alpha,
        title=opt.title,
        legend=[(opt.channel_label, DAPI_WHITE), *legend],
    )


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
        choices=("cell", "nuclei", "both"),
        default="cell",
        help="which label image to outline: the cell mask, the nuclear one, or BOTH in one "
        "image (cells in --outline-color, nuclei in --nuclei-color)",
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
        "--cells-label", default="cells", help="legend name for the cell outlines"
    )
    ap.add_argument(
        "--nuclei-color",
        default="#00e5ff",
        help="colour of the nuclear outlines under --mask both (default #00e5ff); with "
        "--mask nuclei the outlines use --outline-color as before",
    )
    ap.add_argument(
        "--nuclei-label", default="nuclei", help="legend name for the nuclear outlines"
    )
    ap.add_argument(
        "--channel-label",
        default="DAPI",
        help="legend name for the grey channel (default DAPI)",
    )
    ap.add_argument(
        "--title",
        default="",
        help="method name, drawn top left on the figure and on the crop "
        "(submit_zoom.sh passes SEG_METHOD); empty = no title",
    )
    ap.add_argument(
        "--crop",
        choices=("none", "also", "only"),
        default="none",
        help="write the outlined crop ALONE as <pid>_crop.* -- 'also' beside the "
        "overview+zoom figure, 'only' instead of it (default none)",
    )
    ap.add_argument(
        "--crop-px",
        type=int,
        default=0,
        help="side of that file in output px (default 0 = the crop's own pixels, 1:1); "
        "--field-um sets how much TISSUE it covers, this sets how big the file is",
    )
    ap.add_argument(
        "--crop-plain",
        action="store_true",
        help="no scale bar on the crop-alone file: bare pixels",
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
    written = ([] if opt.crop == "only" else [f"{m['patient']}_zoom"]) + (
        [] if opt.crop == "none" else [f"{m['patient']}_crop"]
    )
    log.info(
        "wrote %s.{%s} (%d cells in the zoom) in %s",
        ", ".join(written),
        opt.formats,
        m["cells_in_zoom"],
        opt.outdir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

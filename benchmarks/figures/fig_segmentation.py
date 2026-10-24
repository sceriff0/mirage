"""Fig. 4c: a slide's DAPI with the pipeline's segmentation drawn over a zoomed inset.

    python -m benchmarks.figures.fig_segmentation --image <run>/pyramid/<patient>.ome.tif \
        --geojson <run>/geojson/<patient>_cells.geojson --outdir figures/segmentation

The whole slide in grey (the DAPI channel, ``--channel``: a name or an index; default the
first channel named DAPI/Hoechst), and in its top-right corner a zoom of ``--zoom-um``
with every cell's outline from the pipeline's GeoJSON (``bin/export_geojson.py``):
the nucleus (``nucleusGeometry``, StarDist's own polygon) by default, the cell
(``geometry``) with ``--outline cell``. A box on the slide marks the zoom, joined to it by
a grey connector, as in Fig. 4b.

THE ZOOM (``--zoom X,Y`` in image px, its centre, else chosen): the window holding the
most cells, ties to the top-left. The GeoJSON's coordinates are the full-resolution
pixels of the image they were made on.

DISPLAY: overview linear (0.5, 99.8) percentiles over the tissue; zoom ``common.crisp``
(window inside the zoom, CLAHE, unsharp, x3 cubic). A display adjustment: say so.

Outputs: <outdir>/<stem>.pdf/.png/.svg, _source_data.csv (zoom box, cells drawn,
display settings), _overview.png and _zoom.png for Affinity.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps

# the outlines' colour: the palette's yellow, as Fig. 4c's
OUTLINE = ps.OKABE_ITO[6]
NUCLEAR = re.compile(r"dapi|hoechst", re.IGNORECASE)


def rings(feature, which):
    """The exterior rings of one feature's polygon(s), as (N, 2) arrays."""
    geom = feature.get("nucleusGeometry" if which == "nucleus" else "geometry")
    if geom is None and which == "nucleus":
        geom = feature.get("geometry")
    if not geom:
        return []
    polys = (
        geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    )
    return [np.asarray(p[0], dtype=float) for p in polys if p and len(p[0]) >= 3]


def densest(centres, hw, side):
    best, at = -1, (0, 0)
    step = max(side // 4, 1)
    for y0 in range(0, max(hw[0] - side, 0) + 1, step):
        for x0 in range(0, max(hw[1] - side, 0) + 1, step):
            n = int(
                (
                    (centres[:, 0] >= x0)
                    & (centres[:, 0] < x0 + side)
                    & (centres[:, 1] >= y0)
                    & (centres[:, 1] < y0 + side)
                ).sum()
            )
            if n > best:
                best, at = n, (x0, y0)
    return at


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True, help="the slide (OME-TIFF)")
    ap.add_argument("--geojson", required=True, help="the pipeline's cells GeoJSON")
    ap.add_argument("--channel", help="DAPI channel: name or index (default: by name)")
    ap.add_argument("--pixel-size-um", type=float, help="default: the OME header")
    ap.add_argument("--outline", choices=("nucleus", "cell"), default="nucleus")
    ap.add_argument("--label", default="StarDist cells")
    ap.add_argument("--zoom-um", type=float, default=150.0)
    ap.add_argument("--zoom", help="X,Y: the zoom's centre in image px")
    ap.add_argument("--overview-px", type=int, default=3000)
    ap.add_argument("--clip", default="1,99.8")
    ap.add_argument("--clahe", type=float, default=0.01)
    ap.add_argument("--sharpen", type=float, default=0.6)
    ap.add_argument("--upsample", type=int, default=3)
    ap.add_argument("--outdir", default="figures/segmentation")
    ap.add_argument("--stem", help="default: the image's name")
    a = ap.parse_args(argv)

    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.patheffects import withStroke
    from stare.slide_io import decimation_factor, open_lazy, read_decimated

    from benchmarks.regbench.imageio import channel_names, pixel_size_um

    ps.apply_style()
    names = channel_names(a.image)
    if a.channel is None:
        ch = next((i for i, n in enumerate(names) if NUCLEAR.search(n)), 0)
    elif a.channel.isdigit():
        ch = int(a.channel)
    else:
        ch = names.index(a.channel)
    um_px = a.pixel_size_um or pixel_size_um(a.image)
    if not um_px:
        raise SystemExit("no pixel size in the image's header: pass --pixel-size-um")
    side = int(round(a.zoom_um / um_px))
    show = dict(
        clip=tuple(float(v) for v in a.clip.split(",")),
        clahe=a.clahe,
        sharpen=a.sharpen,
        upsample=a.upsample,
    )

    features = json.loads(Path(a.geojson).read_text())
    features = (
        features.get("features", features) if isinstance(features, dict) else features
    )
    outlines = [r for f in features for r in rings(f, a.outline)]
    if not outlines:
        raise SystemExit(f"no {a.outline} polygon in {a.geojson}")
    centres = np.array([r.mean(0) for r in outlines])

    arr, _, close = open_lazy(a.image)
    try:
        hw = tuple(arr.shape[1:])
        if a.zoom:
            cx, cy = (int(float(v)) for v in a.zoom.split(","))
            x0, y0 = cx - side // 2, cy - side // 2
        else:
            x0, y0 = densest(centres, hw, side)
        f = decimation_factor([hw], a.overview_px)
        thumb = read_decimated(arr, ch, f)
        zoom = C.crisp(C.region(arr, ch, y0, x0, side, side), **show)
    finally:
        close()
    ov = C.stretch(thumb, (0.5, 99.8), thumb > np.percentile(thumb, 50))
    inside = (
        (centres[:, 0] >= x0)
        & (centres[:, 0] < x0 + side)
        & (centres[:, 1] >= y0)
        & (centres[:, 1] < y0 + side)
    )

    width = 180.0
    height = width * ov.shape[0] / ov.shape[1]
    if height > ps.MAX_HEIGHT_MM:
        width, height = width * ps.MAX_HEIGHT_MM / height, ps.MAX_HEIGHT_MM
    fig = C.fixed_figure(width, height)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(ov, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    ax.set_axis_off()
    C.scale_bar(ax, ov.shape[1], um_px * f)
    # the inset: the top-right third of the slide's width, square
    iw = 0.36
    ih = iw * width / height
    inset = fig.add_axes((1 - iw - 0.01, 1 - ih - 0.01, iw, ih))
    inset.imshow(zoom, cmap="gray", vmin=0, vmax=1, interpolation="none")
    up = a.upsample
    for r, keep in zip(outlines, inside):
        if keep:
            inset.add_patch(
                Polygon(
                    (r - [x0, y0]) * up, closed=True, fill=False, ec=OUTLINE, lw=0.35
                )
            )
    inset.set_xlim(0, zoom.shape[1])
    inset.set_ylim(zoom.shape[0], 0)
    inset.set_axis_off()
    inset.add_patch(
        Rectangle(
            (0, 0), 1, 1, transform=inset.transAxes, fill=False, ec="white", lw=0.8
        )
    )
    C.scale_bar(inset, zoom.shape[1], um_px / up, frac=0.3)
    bx, by, bw = x0 / f, y0 / f, side / f
    ax.add_patch(Rectangle((bx, by), bw, bw, fill=False, ec="white", lw=0.6))
    to_fig = ax.transData + fig.transFigure.inverted()
    # the connector: from the box's right edge to the inset's left edge, drawn over the
    # slide and under the inset (Fig. 4c)
    (x1, y1), (x2, y2) = to_fig.transform([(bx + bw, by), (bx + bw, by + bw)])
    p = inset.get_position()
    ax.set_zorder(0)
    inset.set_zorder(2)
    fig.patches.append(
        Polygon(
            [(x1, y1), (p.x0, p.y1), (p.x0, p.y0), (x2, y2)],
            closed=True,
            transform=fig.transFigure,
            fc=ps.GREY,
            ec="none",
            alpha=0.35,
            zorder=1,
        )
    )
    stroke = [withStroke(linewidth=1.2, foreground="black")]
    ax.text(
        0.985,
        0.08,
        names[ch] if names else "DAPI",
        color="white",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ps.BASE_PT + 2,
        path_effects=stroke,
    )
    ax.text(
        0.985,
        0.03,
        a.label,
        color=OUTLINE,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ps.BASE_PT + 2,
        path_effects=stroke,
    )

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / (a.stem or Path(a.image).name.split(".")[0])
    ps.plt.imsave(f"{stem}_overview.png", ov, cmap="gray", vmin=0, vmax=1)
    ps.plt.imsave(f"{stem}_zoom.png", zoom, cmap="gray", vmin=0, vmax=1)
    ps.save(
        fig,
        str(stem),
        data=pd.DataFrame(
            [
                {
                    "image": a.image,
                    "geojson": a.geojson,
                    "channel": ch,
                    "outline": a.outline,
                    "zoom_x0_px": x0,
                    "zoom_y0_px": y0,
                    "zoom_side_px": side,
                    "zoom_side_um": a.zoom_um,
                    "cells_in_zoom": int(inside.sum()),
                    "cells_total": len(outlines),
                    **{f"display_{k}": v for k, v in show.items()},
                }
            ]
        ),
    )
    print(
        f"zoom at ({x0}, {y0}) px, {int(inside.sum())} of {len(outlines)} cells drawn"
    )
    print(f"-> {stem}.pdf/.png/.svg, _source_data.csv, _overview.png, _zoom.png")


if __name__ == "__main__":
    main()

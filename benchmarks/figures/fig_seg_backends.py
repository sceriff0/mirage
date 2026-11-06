"""Supplementary Fig. S7: one region of a registered slide segmented by each backend, and
the pairwise agreement of their whole-cell masks.

    python -m benchmarks.figures.fig_seg_backends --image <run>/pyramid/<patient>.ome.tif \
        --backend CellSAM=<cellsam run>/geojson/<patient>_cells.geojson \
        --backend InstanSeg=<instanseg run>/geojson/<patient>_cells.geojson \
        --backend StarDist=<stardist run>/geojson/<patient>_cells.geojson \
        --outdir figures/seg_backends

READS the pipeline's published GeoJSON of each backend (``bin/export_geojson.py``: a
feature per cell, ``geometry`` the whole cell, ``nucleusGeometry`` the nucleus), not its
masks: the GeoJSON is published at every ``--cleanup_level``, the masks only at ``none``.
The image is the slide the three runs segmented (the same registered pyramid), its DAPI
channel (``--channel``: a name or an index; default the first named DAPI/Hoechst).

  a  rows: the backends in the order given (``--backend``, top to bottom); left the
     whole-cell outlines (yellow), right the nuclear outlines (cyan), over the region's
     DAPI; a scale bar of ``--bar-um`` in every panel
  b  heat map of the pairwise Dice of the backends' WHOLE-CELL masks inside the region:
     each backend's cell polygons filled into one binary mask, 2|A n B| / (|A| + |B|)

THE REGION (``--region X,Y`` in image px, its centre, else chosen): a ``--region-um``
square, the window holding the most cells of the FIRST backend, ties to the top-left
(``fig_segmentation.densest``). A polygon is drawn and filled clipped to the region.

DISPLAY: ``common.crisp`` (window inside the region, CLAHE, unsharp, x``--upsample``
cubic), as every zoom of the paper. A display adjustment: say so in the legend.

Outputs: <outdir>/seg_backends_a.* and seg_backends_b.* (pdf, png, svg), each with its
_source_data.csv (a: the region, the cells drawn per backend and the display settings;
b: one row per pair of backends with its Dice and mask areas).
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import plotstyle as ps
from .fig_segmentation import NUCLEAR, densest, rings

CELL = ps.OKABE_ITO[6]  # yellow, as Fig. 4c's outlines
NUCLEUS = C.CYAN


def load_features(path):
    got = json.loads(Path(path).read_text())
    return got.get("features", got) if isinstance(got, dict) else got


def outlines(features, which):
    """Every exterior ring of the features' ``cell`` or ``nucleus`` polygons."""
    return [r for f in features for r in rings(f, which)]


def in_region(rs, x0, y0, side):
    """The rings whose centre lies in the square region."""
    out = []
    for r in rs:
        cx, cy = r.mean(0)
        if x0 <= cx < x0 + side and y0 <= cy < y0 + side:
            out.append(r)
    return out


def fill_mask(rs, x0, y0, side):
    """The rings filled into one binary mask of the region (clipped to it)."""
    from skimage.draw import polygon

    mask = np.zeros((side, side), dtype=bool)
    for r in rs:
        rr, cc = polygon(r[:, 1] - y0, r[:, 0] - x0, shape=mask.shape)
        mask[rr, cc] = True
    return mask


def dice(a, b):
    total = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / total if total else float("nan")


def pairwise(masks):
    """``(matrix, rows)``: the Dice of every pair of backends' masks (1 on the diagonal)
    and one source-data row per pair."""
    names = list(masks)
    mat = np.eye(len(names))
    rows = []
    for i, j in combinations(range(len(names)), 2):
        d = dice(masks[names[i]], masks[names[j]])
        mat[i, j] = mat[j, i] = d
        rows.append({"backend_a": names[i], "backend_b": names[j], "dice": d,
                     "area_a_px": int(masks[names[i]].sum()),
                     "area_b_px": int(masks[names[j]].sum())})  # fmt: skip
    return mat, rows


def parse_backends(items):
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--backend: {item!r} is not NAME=geojson")
        out[name.strip()] = path.strip()
    if len(out) < 2:
        raise SystemExit("--backend: give at least two backends")
    return out


def panel_a(dapi, drawn, side, um_px, upsample, bar_um, outdir):
    from matplotlib.patches import Polygon

    names = list(drawn)
    fig, axes = ps.figure(width="single", nrows=len(names), ncols=2, aspect=1.0)
    for row, name in enumerate(names):
        for col, (which, colour) in enumerate((("cell", CELL), ("nucleus", NUCLEUS))):
            ax = axes[row * 2 + col]
            ax.imshow(dapi, cmap="gray", vmin=0, vmax=1, interpolation="none")
            for r in drawn[name][which]:
                ax.add_patch(Polygon(r * upsample, closed=True, fill=False, ec=colour,
                                     lw=0.35))  # fmt: skip
            ax.set_xlim(0, dapi.shape[1])
            ax.set_ylim(dapi.shape[0], 0)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name)
            if row == 0:
                ax.set_title("Whole cell" if which == "cell" else "Nucleus")
            C.scale_bar(ax, dapi.shape[1], um_px / upsample, length_um=bar_um)
    ps.save(fig, str(Path(outdir) / "seg_backends_a"), data=None)


def panel_b(mat, names, rows, outdir):
    fig, (ax,) = ps.figure(width=60.0, aspect=0.9)
    im = ax.imshow(mat, cmap=ps.SEQUENTIAL, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=ps.TICK_PT,
                    color="black" if v > 0.6 else "white")  # fmt: skip
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Whole-cell mask Dice")
    ps.save(fig, str(Path(outdir) / "seg_backends_b"), data=pd.DataFrame(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Supp. Fig. S7: one region segmented by each backend, and their Dice"
    )
    ap.add_argument("--image", required=True, help="the registered slide (OME-TIFF)")
    ap.add_argument("--backend", action="append", required=True,
                    help="NAME=<cells GeoJSON>, once per backend, top row first")  # fmt: skip
    ap.add_argument("--channel", help="DAPI channel: name or index (default: by name)")
    ap.add_argument("--pixel-size-um", type=float, help="default: the OME header")
    ap.add_argument("--region-um", type=float, default=150.0)
    ap.add_argument("--region", help="X,Y: the region's centre in image px")
    ap.add_argument("--bar-um", type=float, default=50.0)
    ap.add_argument("--clip", default="1,99.8")
    ap.add_argument("--clahe", type=float, default=0.01)
    ap.add_argument("--sharpen", type=float, default=0.6)
    ap.add_argument("--upsample", type=int, default=3)
    ap.add_argument("--outdir", default="figures/seg_backends")
    a = ap.parse_args(argv)

    from stare.slide_io import open_lazy

    from benchmarks.regbench.imageio import channel_names, pixel_size_um

    ps.apply_style()
    backends = parse_backends(a.backend)
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
    side = int(round(a.region_um / um_px))
    lo, hi = (float(v) for v in a.clip.split(","))
    show = {"clip": (lo, hi), "clahe": float(a.clahe), "sharpen": float(a.sharpen),
            "upsample": int(a.upsample)}  # fmt: skip

    feats = {n: load_features(p) for n, p in backends.items()}
    cells = {n: outlines(f, "cell") for n, f in feats.items()}
    first = next(iter(backends))
    if not cells[first]:
        raise SystemExit(f"no cell polygon in {backends[first]}")
    arr, _, close = open_lazy(a.image)
    try:
        hw = tuple(arr.shape[1:])
        if a.region:
            cx, cy = (int(float(v)) for v in a.region.split(","))
            x0, y0 = cx - side // 2, cy - side // 2
        else:
            x0, y0 = densest(np.array([r.mean(0) for r in cells[first]]), hw, side)
        dapi = C.crisp(C.region(arr, ch, y0, x0, side, side), clip=(lo, hi),
                       clahe=float(a.clahe), sharpen=float(a.sharpen),
                       upsample=int(a.upsample))  # fmt: skip
    finally:
        close()

    drawn, masks, info = {}, {}, []
    for n, f in feats.items():
        cell = in_region(cells[n], x0, y0, side)
        nuc = in_region(outlines(f, "nucleus"), x0, y0, side)
        drawn[n] = {"cell": [r - [x0, y0] for r in cell],
                    "nucleus": [r - [x0, y0] for r in nuc]}  # fmt: skip
        masks[n] = fill_mask(cell, x0, y0, side)
        info.append({"backend": n, "geojson": backends[n], "cells_drawn": len(cell),
                     "nuclei_drawn": len(nuc), "cells_total": len(cells[n])})  # fmt: skip

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    panel_a(dapi, drawn, side, um_px, a.upsample, a.bar_um, outdir)
    region = {"image": a.image, "channel": ch, "region_x0_px": x0, "region_y0_px": y0,
              "region_side_px": side, "region_side_um": a.region_um,
              **{f"display_{k}": v for k, v in show.items()}}  # fmt: skip
    pd.DataFrame([{**region, **r} for r in info]).to_csv(
        outdir / "seg_backends_a_source_data.csv", index=False
    )
    mat, rows = pairwise(masks)
    panel_b(mat, list(masks), rows, outdir)
    for r in rows:
        print(f"{r['backend_a']} vs {r['backend_b']}: whole-cell Dice {r['dice']:.3f}")
    print(f"region at ({x0}, {y0}) px, {side} px; -> {outdir}/seg_backends_{{a,b}}.*")


if __name__ == "__main__":
    main()

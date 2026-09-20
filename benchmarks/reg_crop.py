#!/usr/bin/env python3
"""reg_crop.py -- a clean crop of ONE named channel, at an exact output size.

For the panels a figure is assembled from: no overview, no funnel, no outlines -- one
channel, cropped, scaled so the background is black and the signal is readable, written at
exactly the pixel size asked for.

    python -m benchmarks.reg_crop results/valis_high --channel CD3 \\
        --field-um 150 --crop-px 1024 -o figs/crops

THE CHANNEL is named, not indexed: ``--channel CD3`` is matched against the slide's own OME
channel names, falling back to the ``channels`` column of the checkpoint CSV (the samplesheet's
list) when the file carries none -- the same rule the other tools use for DAPI. An exact
case-insensitive match wins; failing that a unique substring; "CD3" against both "CD3" and
"CD31" is refused rather than guessed. The slide is the run's REGISTERED output by default
(``csv/registered.csv``, or ``csv/segmented.csv``), so every channel is on one canvas and the
same --roi names the same tissue in all of them.

THE SCALING is the point. A percentile stretch (--pmin/--pmax, what the mosaic uses for a
pair of DAPI planes) puts the black point inside the background distribution: the camera
offset and its noise survive it, and the crop reads as a grey field, or -- once it is
coloured -- as a wash of that colour. ``--autoscale clean`` (the default) estimates the
background instead and puts the black point above its noise, which is what QuPath's auto
contrast does; see reg_mosaic.clean_limits for the rule and its two knobs (--sat, --bg-k).
``--autoscale percentile`` restores the old behaviour, and --vmin/--vmax pin it outright.

Outputs in OUTDIR, one set per channel and ROI:
    <patient>_<channel>_crop.png/.pdf   the crop, exactly --crop-px square
    <patient>_<channel>_crop.json       limits, region, and how they were chosen

Requires numpy, scipy, scikit-image, tifffile and matplotlib (requirements/segeval.txt).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from benchmarks import reg_mosaic as rm
from benchmarks.reg_zoom import read_segmented, reference_row

log = logging.getLogger("reg_crop")


def _rgb(color) -> tuple[float, float, float]:
    import matplotlib.colors

    return tuple(float(c) for c in matplotlib.colors.to_rgb(color))


def colorize(gray01: np.ndarray, color) -> np.ndarray:
    """One channel as an RGB image: grey scaled by `color` (white leaves it grey)."""
    return np.clip(gray01[..., None] * np.asarray(_rgb(color), np.float32), 0.0, 1.0)


def limits_for(plane, opt) -> tuple[tuple[float, float], str]:
    if opt.vmin is not None and opt.vmax is not None:
        return (float(opt.vmin), float(opt.vmax)), "pinned"
    if opt.autoscale == "percentile":
        return rm.percentile_limits(plane, opt.pmin, opt.pmax), "percentile"
    return rm.clean_limits(plane, opt.sat, opt.bg_k), "clean"


def patient_rows(opt):
    """Every registered slide of the patient, reference first, and the CSV they came from.

    A channel lives on the slide that carries it -- CD3 is not on the reference -- and every
    registered slide sits on the SAME reference canvas, so one --roi names the same tissue in
    all of them. Hence: search them all, do not assume the reference row.
    """
    run = Path(opt.run)
    csv_path = opt.csv or next(
        (
            p
            for p in (run / "csv" / "registered.csv", run / "csv" / "segmented.csv")
            if p.is_file()
        ),
        run / "csv" / "registered.csv",
    )
    rows = read_segmented(
        csv_path, need=("patient_id", "registered_image", "is_reference")
    )
    ref = reference_row(rows, opt.patient)
    pid = ref["patient_id"]
    mine = [r for r in rows if r["patient_id"] == pid]
    mine.sort(key=lambda r: str(r.get("is_reference", "")).lower() != "true")
    return pid, mine, csv_path


def find_channel(rows, name: str, sources: dict):
    """(row, TiffSource, index) of the slide carrying `name`, or None with what was seen."""
    seen: list[str] = []
    for row in rows:
        path = rm.published_file(row["registered_image"])
        if path not in sources:
            sources[path] = rm.TiffSource(path)
        src = sources[path]
        csv_channels = [c for c in (row.get("channels") or "").split("|") if c]
        seen += src.channel_names or csv_channels
        ci = src.channel_index(name, csv_channels)
        if ci is not None:
            return row, src, ci
    return None, None, sorted(set(seen))


def render(opt) -> list[dict]:
    pid, rows, csv_path = patient_rows(opt)
    sources: dict = {}
    ref = rows[0]
    ref_src = rm.TiffSource(rm.published_file(ref["registered_image"]))
    sources[rm.published_file(ref["registered_image"])] = ref_src
    px = opt.pixel_size_um or rm._float_or_none(ref.get("pixel_size")) or ref_src.px
    if not px:
        raise SystemExit("pixel size unknown; pass --pixel-size-um")
    H, W = ref_src.shape

    field_px = min(int(round(opt.field_um / px)), H, W)
    if field_px > opt.max_px:
        raise SystemExit(
            f"a {opt.field_um:g} um crop is {field_px} px at {px} um/px; it is read at full "
            f"resolution, so keep it under --max-px {opt.max_px} (~{opt.max_px * px:.0f} um)"
        )
    if opt.roi:
        y, x = (int(v) for v in opt.roi.split(",")[:2])
    else:
        ref_channels = [c for c in (ref.get("channels") or "").split("|") if c]
        low, factor = ref_src.overview(
            ref_src.nuclear_index(ref_channels) or 0, opt.overview_px
        )
        picks = rm.select_rois(low, factor, field_px, 1, 0.0, 0.0, (H, W))
        if not picks:
            raise SystemExit(
                f"{pid}: no tissue region of {field_px} px found; pass --roi"
            )
        y, x = picks[0]
    y = min(max(y, 0), max(H - field_px, 0))
    x = min(max(x, 0), max(W - field_px, 0))

    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in opt.formats.split(",") if f.strip()]
    colors = [c.strip() for c in opt.colors.split(",") if c.strip()] or ["white"]
    out = []
    for i, name in enumerate(opt.channel):
        row, src, ci = find_channel(rows, name, sources)
        if row is None:
            raise SystemExit(
                f"{pid}: no channel {name!r} on any registered slide; the run has "
                f"{ci or '(unnamed channels)'}"
            )
        image = rm.published_file(row["registered_image"])
        plane = src.read_patch(ci, y, x, field_px, field_px)
        (lo, hi), how = limits_for(plane, opt)
        color = colors[i % len(colors)]
        img = colorize(rm.stretch(plane, (lo, hi), opt.gamma), color)
        stem = outdir / f"{pid}_{name}_crop"
        n = rm.write_crop(
            img,
            px,
            stem,
            formats,
            opt.dpi,
            opt.crop_px,
            not opt.plain,
            legend=() if opt.plain else [(opt.label or name, _rgb(color))],
            title="" if opt.plain else opt.title,
        )
        log.info(
            "%s %s: y=%d x=%d %d px = %g um -> %d px, limits [%.4g, %.4g] (%s)",
            pid,
            name,
            y,
            x,
            field_px,
            field_px * px,
            n,
            lo,
            hi,
            how,
        )
        meta = {
            "patient": pid,
            "channel": name,
            "channel_index": ci,
            "image": str(image),
            "checkpoint": str(csv_path),
            "pixel_size_um": px,
            "crop": {"y": y, "x": x, "size_px": field_px, "size_um": field_px * px},
            "output_px": n,
            "color": color,
            "limits": {"lo": lo, "hi": hi, "how": how, "gamma": opt.gamma},
            "autoscale": {"mode": opt.autoscale, "sat": opt.sat, "bg_k": opt.bg_k},
            "title": "" if opt.plain else opt.title,
        }
        Path(f"{stem}.json").write_text(json.dumps(meta, indent=2))
        out.append(meta)
    for src in sources.values():
        src.close()
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=Path, metavar="RUN_DIR", help="a mirage --outdir")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument(
        "--channel",
        action="append",
        required=True,
        metavar="NAME",
        help="channel to crop, by name (repeatable); must exist on the slide or in the "
        "checkpoint's channels column",
    )
    ap.add_argument(
        "--csv", type=Path, default=None, help="default: RUN_DIR/csv/registered.csv"
    )
    ap.add_argument("--patient", default=None)
    ap.add_argument(
        "--field-um", type=float, default=150.0, help="crop side in um (tissue)"
    )
    ap.add_argument(
        "--crop-px",
        type=int,
        default=0,
        help="output side in px (0 = the crop's own pixels, 1:1): --field-um sets how much "
        "TISSUE, this how big the file is",
    )
    ap.add_argument(
        "--roi", default=None, metavar="Y,X", help="crop top-left, full-res px"
    )
    ap.add_argument(
        "--autoscale",
        choices=("clean", "percentile"),
        default="clean",
        help="clean (default): background estimated and put below black, as QuPath's auto "
        "contrast; percentile: the plain --pmin/--pmax stretch",
    )
    ap.add_argument(
        "--sat",
        type=float,
        default=0.35,
        help="percent of the SIGNAL allowed to saturate (default 0.35)",
    )
    ap.add_argument(
        "--bg-k",
        type=float,
        default=3.0,
        help="black point = background median + this many robust sigmas (default 3)",
    )
    ap.add_argument(
        "--vmin", type=float, default=None, help="pin the black point outright"
    )
    ap.add_argument(
        "--vmax", type=float, default=None, help="pin the white point outright"
    )
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.8)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument(
        "--colors",
        default="white",
        help="comma-separated colour per --channel, cycled (default white = grey)",
    )
    ap.add_argument(
        "--label", default=None, help="legend text (default: the channel name)"
    )
    ap.add_argument(
        "--title", default="", help="method name, drawn top left; empty = none"
    )
    ap.add_argument(
        "--plain",
        action="store_true",
        help="bare pixels: no title, legend or scale bar",
    )
    ap.add_argument("--max-px", type=int, default=4096)
    ap.add_argument("--overview-px", type=int, default=2400)
    ap.add_argument("--pixel-size-um", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=100)
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
    metas = render(opt)
    log.info("wrote %d crop(s) in %s", len(metas), opt.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

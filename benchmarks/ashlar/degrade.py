#!/usr/bin/env python3
"""Give the whole slides the SAME defect the synthetic ASHLAR tiles carry.

``retile.py`` hands ASHLAR raw tiles with a stage error and per-tile sensor noise. A method
that registers whole slides never sees a stage error -- a stitched slide is what the scanner
made of it -- but it should not get cleaner pixels than ASHLAR either. This writes, for
every slide of a ``preprocessed.csv``, a copy with sensor noise of the same strength
(``retile.noise_sd``: the same fraction of each channel's own range, measured the same way
on the same clean slide), and a ``preprocessed.csv`` naming the copies. VALIS and STARE
are then run from that checkpoint, ASHLAR from tiles of the clean slides plus its per-tile
noise, so all three register equally noisy pixels.

The copies are written with the pipeline's own converter settings (``bin/utils/ome_io``:
tiled OME-TIFF, one page per channel, channel names and pixel size in the header), under
the same file names, so everything keyed on a slide's name still matches.
"""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bin" / "utils"))

from benchmarks.ashlar.retile import noise_sd  # noqa: E402

logger = logging.getLogger(__name__)

BAND_ROWS = 4096


def degrade_slide(src, dst, channels, pixel_size_um, noise_frac, seed):
    """Write ``src`` plus sensor noise to ``dst``; returns the per-channel s.d. used.

    One plane in memory at a time, noise drawn in bands of rows from one generator per
    channel, so the result depends only on (``seed``, channel) and the slide's shape.
    ``seed`` is one integer or a sequence of them.
    """
    import ome_io
    from tiled_io import open_lazy

    seeds = [int(v) for v in np.atleast_1d(seed)]
    arr, dtype, close = open_lazy(src)
    try:
        n_channels, height, width = arr.shape
        if channels is not None and len(channels) != n_channels:
            raise ValueError(
                f"{src}: {n_channels} planes but {len(channels)} channel names {channels}"
            )
        sd = noise_sd(arr, noise_frac)
        tile = ome_io.CONVERT_TIFF_TILE
        integer = np.issubdtype(dtype, np.integer)
        info = np.iinfo(dtype) if integer else None

        def planes():
            for c in range(n_channels):
                plane = np.asarray(arr[c, :, :])  # the lazy view takes 3-tuples only
                if sd[c] > 0:
                    rng = np.random.default_rng([*seeds, 2, c])
                    out = np.empty_like(plane)
                    for y in range(0, height, BAND_ROWS):
                        band = plane[y : y + BAND_ROWS].astype(np.float32)
                        band += rng.normal(0.0, sd[c], band.shape).astype(np.float32)
                        if integer:
                            band = np.clip(np.rint(band), info.min, info.max)
                        out[y : y + BAND_ROWS] = band.astype(dtype)
                    plane = out
                yield plane

        def tiles():
            for plane in planes():
                for y in range(0, height, tile):
                    for x in range(0, width, tile):
                        yield plane[y : y + tile, x : x + tile]

        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".partial")
        with ome_io.ome_tiff_writer(tmp, bigtiff=True, ome=True) as writer:
            writer.write(
                tiles(),
                shape=(n_channels, height, width),
                dtype=dtype,
                metadata=ome_io.ome_metadata(channels, pixel_size_um, axes="CYX"),
                photometric="minisblack",
                tile=(tile, tile),
            )
        tmp.replace(
            dst
        )  # a reader never sees half a slide; a rerun redoes a broken one
    finally:
        close()
    return sd


def degrade_checkpoint(csv_path, out_root, noise_frac, seed, pixel_size_um=None):
    """Degrade every slide of ``csv_path`` into ``out_root``; returns the new csv's path.

    Layout as a preprocessing run's: ``<out_root>/<patient>/preprocessed/<same name>`` and
    ``<out_root>/csv/preprocessed.csv``. A slide already written is kept, so an
    interrupted run continues.
    """
    out_root = Path(out_root)
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "preprocessed_image" not in rows[0]:
        raise SystemExit(f"{csv_path}: no preprocessed_image column")
    for k, row in enumerate(rows):
        src = Path(row["preprocessed_image"])
        dst = out_root / row["patient_id"] / "preprocessed" / src.name
        names = [c for c in (row.get("channels") or "").split("|") if c] or None
        px = pixel_size_um
        if px is None:
            try:
                px = float(row.get("pixel_size") or "")
            except ValueError:
                px = None
        if dst.is_file():
            logger.info("%s: exists, kept", dst)
        else:
            # one seed per slide: the same slide always gets the same noise, and two
            # slides never share a realisation
            sd = degrade_slide(src, dst, names, px, noise_frac, [int(seed), k])
            logger.info("%s: noise s.d. %s", dst, [round(v, 2) for v in sd])
        row["preprocessed_image"] = str(dst)
    out_csv = out_root / "csv" / "preprocessed.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--csv", required=True, help="a preprocessing run's preprocessed.csv"
    )
    ap.add_argument("--out-root", required=True, help="the degraded run's directory")
    ap.add_argument("--noise-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pixel-size-um", type=float, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = degrade_checkpoint(a.csv, a.out_root, a.noise_frac, a.seed, a.pixel_size_um)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

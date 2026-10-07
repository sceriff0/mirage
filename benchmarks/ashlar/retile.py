#!/usr/bin/env python3
"""Cut a stitched cycle OME-TIFF into the uniform tile grid ASHLAR's FilePatternReader reads.

ASHLAR needs raw, unstitched tiles; mirage's registration inputs are already-stitched
whole-slide OME-TIFFs. This module synthesizes the tile input, and it must match ASHLAR's
own position arithmetic exactly, because ASHLAR does not read positions from anywhere --
it *computes* them. ``ashlar/filepattern.py``::

    def tile_position(self, i):
        row, col = self.tile_rc(i)
        return [row, col] * self.tile_size(i) * (1 - self.overlap)

so a tile's stage position is implied entirely by its ``{row}``/``{col}`` and the grid
constants. Writing ``x = col * tile_size * (1 - overlap)`` here makes the fabricated
positions exact by construction; there is no manifest to keep in sync and no drift.

TWO CONSEQUENCES THAT ARE NOT OPTIONAL.

**Tiles must be UNIFORM.** ``tile_size(i)`` returns ``self._tile_size``, read from the
FIRST matching tile only and then applied to every tile in the grid. An edge tile cropped
to its own valid extent is therefore declared at full size and read back short, and the
shape mismatch surfaces inside ``EdgeAligner``/``paste`` rather than here. So every tile is
zero-PADDED to ``tile_size`` square. The prior implementation of this module cropped edge
tiles ("never padded") and could not have worked against the real reader.

**No positions.csv.** ASHLAR has four readers -- bioformats, filepattern, fileseries, zen --
and not one of them consumes a positions file. The prior implementation wrote one; the
reader's ``_enumerate_tiles`` simply fails to regex-match it and skips it. It is written
here as ``grid.json`` instead, for ``ashlar_solve.py``'s benefit, never ASHLAR's.

Tiles carry ALL channels and the filename pattern carries NO ``{channel}`` field, which is
what selects ``FilePatternMetadata``'s ``multi_channel_tiles`` branch: with one regex
channel group and a 3-D image it re-reads the channel map off axis 0. The nuclear/fiducial
channel therefore stays at the same index in every tile without being named.

The grid math (``tile_grid``, ``grid_shape``, ``TilePos``) has no heavy dependencies, so
importing this module for planning alone stays cheap; the image IO imports lazily.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys
from collections import namedtuple
from pathlib import Path

# RELOCATED from bin/ to benchmarks/: ashlar is a BENCHMARK COMPARATOR, not a pipeline
# backend. dev removed the backend at :fire: 6a54479 ("v1.0.0 ships valis and tiled"),
# and tests/test_ashlar_backend_removed.py keeps it out. The comparator still needs a
# driver, so it lives here, harness-owned, and reaches back into the pipeline's
# bin/utils/ for the SHARED manifest/pixel-size code -- reused rather than copied, so
# the manifest ashlar emits cannot drift from the one STARE emits and the single
# predict_from_manifest reads both.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bin" / "utils"))

logger = logging.getLogger(__name__)

# (row, col) grid index; (x, y) fabricated top-left stage position in px; (w, h) the VALID
# data extent, which is smaller than tile_size on the right/bottom edge. The written tile is
# always tile_size square -- w/h say how much of it is real, and grid.json carries them so
# the solve step never mistakes padding for signal.
TilePos = namedtuple("TilePos", "row col x y w h")

# Repo-wide fallback pixel size (um/px) when OME-XML PhysicalSizeX is absent -- matches the
# convention in bin/pad_image.py.
DEFAULT_PIXEL_SIZE_UM = 0.325

TILE_PATTERN = "r{row:03}_c{col:03}.tif"


def _stride(tile_size, overlap_fraction):
    """Grid pitch in px. Mirrors ashlar's ``tile_size * (1 - overlap)`` exactly."""
    return max(1, int(round(tile_size * (1.0 - overlap_fraction))))


def grid_shape(width, height, tile_size, overlap_fraction):
    stride = _stride(tile_size, overlap_fraction)
    n_cols = max(1, math.ceil(width / stride))
    n_rows = max(1, math.ceil(height / stride))
    return (n_rows, n_cols)


def tile_grid(width, height, tile_size, overlap_fraction):
    """The full rectangular grid. ``w``/``h`` are clamped; ``x``/``y`` never are.

    ASHLAR requires a *full rectangular* grid -- ``_enumerate_tiles`` raises "Tiles do not
    form a full rectangular grid" if ``n != len(rows) * len(cols) * len(channels)`` -- so
    every (row, col) cell is emitted even when its valid extent is a sliver.
    """
    stride = _stride(tile_size, overlap_fraction)
    n_rows, n_cols = grid_shape(width, height, tile_size, overlap_fraction)
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * stride
            y = r * stride
            tiles.append(
                TilePos(
                    r,
                    c,
                    x,
                    y,
                    max(0, min(tile_size, width - x)),
                    max(0, min(tile_size, height - y)),
                )
            )
    return tiles


def _pixel_size_um(image_path: Path) -> float:
    """Read ``PhysicalSizeX`` from OME-XML **in µm**; fall back to ``DEFAULT_PIXEL_SIZE_UM``.

    Only genuine absence falls back silently: no OME-XML block, or no ``PhysicalSizeX``
    attribute. A malformed-XML parse failure or an unparseable value is logged naming the
    file and the fallback used, so a corrupted calibration cannot masquerade as "no
    metadata" -- which would silently rescale ASHLAR's ``--maximum-shift``.

    ``PhysicalSizeXUnit`` is applied, via the one conversion table in
    ``bin/utils/pixel_size.py``. This used to return ``float(PhysicalSizeX)``
    raw: a scanner writing ``PhysicalSizeX="325" PhysicalSizeXUnit="nm"`` --
    legal, and common -- was read as 325 µm/px rather than 0.325, a 1000x scale
    error going straight into ASHLAR's ``--maximum-shift``, which is expressed
    in µm and converted to pixels using exactly this number.

    An unrecognised unit RAISES rather than falling back, which is the one place
    this reader's policy differs from ``read_ome_pixel_size``'s (see
    ``unit_to_um``'s docstring). An unknown unit means the file HAS a
    calibration that we failed to interpret, not that it has none; substituting
    ``DEFAULT_PIXEL_SIZE_UM`` for it would discard real information silently,
    and this number has no more-authoritative counterpart to be checked against.
    """
    import tifffile
    from pixel_size import unit_to_um

    with tifffile.TiffFile(str(image_path)) as tif:
        ome_metadata = getattr(tif, "ome_metadata", None)
    if not ome_metadata:
        return DEFAULT_PIXEL_SIZE_UM

    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(ome_metadata)
    except ET.ParseError as e:
        logger.warning(
            "%s: malformed OME-XML (%s); falling back to %s um/px",
            image_path,
            e,
            DEFAULT_PIXEL_SIZE_UM,
        )
        return DEFAULT_PIXEL_SIZE_UM

    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    pixels = root.find(".//ome:Pixels", ns)
    if pixels is None:
        pixels = root.find(".//{*}Pixels")  # namespace-agnostic (other schema versions)
    val = pixels.get("PhysicalSizeX") if pixels is not None else None
    if not val:
        return DEFAULT_PIXEL_SIZE_UM
    try:
        value = float(val)
    except ValueError:
        logger.warning(
            "%s: PhysicalSizeX=%r is not a number; falling back to %s um/px",
            image_path,
            val,
            DEFAULT_PIXEL_SIZE_UM,
        )
        return DEFAULT_PIXEL_SIZE_UM

    raw_unit = pixels.get("PhysicalSizeXUnit")
    factor = unit_to_um(raw_unit)
    if factor is None:
        raise ValueError(
            f"{image_path}: PhysicalSizeXUnit={raw_unit!r} is not a length unit "
            f"this reader knows, so PhysicalSizeX={val!r} cannot be converted to "
            f"um/px. Refusing to guess: this scale is what ASHLAR's "
            f"--maximum-shift is converted by, so a wrong factor is a silent "
            f"registration failure, and falling back to "
            f"{DEFAULT_PIXEL_SIZE_UM} um/px would throw away a calibration the "
            f"file actually has. Add the unit to bin/utils/pixel_size.py's table."
        )
    return value * factor


def _region(arr, n_channels, t, tile_size, dtype, offset_yx=(0, 0)):
    """One zero-padded ``(C, tile_size, tile_size)`` tile read from a lazy (C, H, W) view.

    ``offset_yx`` moves the window off the tile's nominal corner: the tile then holds the
    slide content at ``(t.y + dy, t.x + dx)`` while its file name still says ``(row, col)``,
    which is what a stage that did not land exactly where it reported looks like.
    """
    import numpy as np

    out = np.zeros((n_channels, tile_size, tile_size), dtype=dtype)
    height, width = arr.shape[-2:]
    y0, x0 = t.y + int(offset_yx[0]), t.x + int(offset_yx[1])
    ys, ye = max(y0, 0), min(y0 + tile_size, height)
    xs, xe = max(x0, 0), min(x0 + tile_size, width)
    if ye <= ys or xe <= xs:
        return out
    win = np.asarray(arr[:, ys:ye, xs:xe])
    if (
        win.ndim == 2
    ):  # open_lazy presents a 2-D source as C=1 but returns it un-promoted
        win = win[np.newaxis, ...]
    out[:, ys - y0 : ye - y0, xs - x0 : xe - x0] = win
    return out


def stage_jitter(n_tiles, jitter_px, seed):
    """``(n_tiles, 2)`` integer ``(dy, dx)`` stage errors, uniform in ``[-j, j]`` px.

    A real stage does not land where it reports: ASHLAR exists to find those errors from
    the overlaps (its --maximum-shift is their budget). Whole pixels, so a jittered tile
    is still the slide's own pixels and nothing is resampled.
    """
    import numpy as np

    j = int(round(jitter_px))
    if j <= 0:
        return np.zeros((n_tiles, 2), dtype=int)
    return np.random.default_rng(seed).integers(-j, j + 1, size=(n_tiles, 2))


def noise_sd(arr, noise_frac, max_samples=4_000_000):
    """Per-channel sensor-noise s.d. in grey levels: ``noise_frac`` of the channel's own
    1st-99th percentile range, measured on a strided sample of the slide."""
    import numpy as np

    n_channels, height, width = arr.shape
    if noise_frac <= 0:
        return [0.0] * n_channels
    step = max(1, int(math.ceil(math.sqrt(height * width / max_samples))))
    out = []
    for c in range(n_channels):
        sample = np.asarray(arr[c, ::step, ::step], dtype=np.float32)
        lo, hi = np.percentile(sample, (1.0, 99.0))
        out.append(float(noise_frac * max(hi - lo, 1.0)))
    return out


def add_noise(tile, sd, rng):
    """``tile`` (C, H, W) plus independent Gaussian noise of s.d. ``sd[c]``, in its dtype.

    Each tile is a separate exposure, so the noise is drawn per tile: the SAME tissue seen
    in two neighbouring tiles' overlap is no longer the same pixels. That is also what
    lets ASHLAR's edge registration run at all on tiles cut from one image -- on identical
    overlaps its error metric fails on a rounding difference (job 6844139).
    """
    import numpy as np

    if not any(v > 0 for v in sd):
        return tile
    out = tile.astype(np.float32)
    for c, v in enumerate(sd):
        if v > 0:
            out[c] += rng.normal(0.0, v, out[c].shape).astype(np.float32)
    if np.issubdtype(tile.dtype, np.integer):
        info = np.iinfo(tile.dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(tile.dtype)
    return out.astype(tile.dtype)


def _yx_shape(image_path) -> tuple[int, int]:
    """(height, width) of a TIFF's first series, from its tags -- no pixels decoded."""
    import tifffile

    with tifffile.TiffFile(str(image_path)) as tif:
        s = tif.series[0]
        return int(s.shape[s.axes.index("Y")]), int(s.shape[s.axes.index("X")])


def write_tiles(
    image_path,
    outdir,
    tile_size,
    overlap_fraction,
    cycle=0,
    canvas_like=(),
    pixel_size_um=None,
    stage_jitter_um=0.0,
    noise_frac=0.0,
    seed=0,
    exact_overlap=False,
):
    """Cut ``image_path`` into a uniform padded tile grid + ``grid.json``. Returns its path.

    SYNTHETIC RAW TILES (``stage_jitter_um`` / ``noise_frac``, both off by default). A
    stitched slide cut on a perfect grid is not what a microscope hands ASHLAR: real tiles
    sit a few microns off their reported stage position and each is its own noisy
    exposure. With these set, every tile is cut ``stage_jitter`` px off its nominal corner
    (see :func:`stage_jitter`) and gets its own sensor noise (:func:`add_noise`), seeded by
    (``seed``, ``cycle``, tile), and ``grid.json`` records where each tile REALLY came from
    (``true_positions_yx``) -- the ground truth ASHLAR is not told. ``exact_overlap``
    records the overlap as ``1 - stride / tile_size`` (the pitch is a whole pixel, the
    requested fraction usually is not), so the positions ASHLAR computes from row, col and
    overlap are exactly the nominal corners.

    Streams via ``tiled_io.open_lazy``, whose zarr view fetches only the OME-TIFF tiles a
    region touches, so peak memory is one output tile rather than the whole slide --
    ``load_channels`` would pull a 60k x 40k plane into RAM.

    ``canvas_like``: the grid is laid on the largest height and width among these images
    (and this one), the image anchored at the origin and zero-padded right and bottom. Pass
    EVERY slide of a patient, reference included, to every call: LayerAligner matches tiles
    one-for-one, so all cycles must share n_rows x n_cols. Retiled on their own extents,
    two slides of different size give different grids and ASHLAR_SOLVE refuses the pair
    ("grids disagree on n_rows (35 vs 31)", real run 2026-09-16). ``orig_shape`` stays the
    image's REAL shape -- the solve sizes the stitched output from the reference's -- and
    ``canvas_shape`` records the padded one.

    ``pixel_size_um``: the run's pixel size. When given it wins over the file's OME header,
    which can carry the scanner's own calibration (0.3453 on a real ND2 run given 0.325);
    it is what ASHLAR converts --maximum-shift by.
    """
    import numpy as np
    import tifffile
    from tiled_io import open_lazy

    image_path = Path(image_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if pixel_size_um is None:
        pixel_size_um = _pixel_size_um(image_path)
    arr, dtype, close = open_lazy(image_path)
    try:
        n_channels, height, width = arr.shape
        shapes = [(height, width)] + [_yx_shape(p) for p in canvas_like]
        canvas_h, canvas_w = max(h for h, _ in shapes), max(w for _, w in shapes)
        tiles = [
            t._replace(
                w=max(0, min(t.w, width - t.x)), h=max(0, min(t.h, height - t.y))
            )
            for t in tile_grid(canvas_w, canvas_h, tile_size, overlap_fraction)
        ]
        px = float(pixel_size_um)
        jitter = stage_jitter(len(tiles), stage_jitter_um / px, [seed, cycle, 0])
        sd = noise_sd(arr, noise_frac)
        for k, t in enumerate(tiles):
            tile = _region(arr, n_channels, t, tile_size, dtype, jitter[k])
            tile = add_noise(
                tile, sd, np.random.default_rng([seed, cycle, 1, t.row, t.col])
            )
            name = TILE_PATTERN.format(row=t.row, col=t.col)
            tifffile.imwrite(outdir / name, tile, photometric="minisblack")
    finally:
        close()

    n_rows, n_cols = grid_shape(canvas_w, canvas_h, tile_size, overlap_fraction)
    stride = _stride(tile_size, overlap_fraction)
    grid = {
        "cycle": cycle,
        "pattern": TILE_PATTERN,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "tile_size": tile_size,
        "overlap": (1.0 - stride / tile_size) if exact_overlap else overlap_fraction,
        "stride": stride,
        "pixel_size_um": pixel_size_um,
        "n_channels": int(n_channels),
        "orig_shape": [int(height), int(width)],
        "canvas_shape": [int(canvas_h), int(canvas_w)],
        "source": image_path.name,
        # Valid (unpadded) extent per tile, row-major, so nothing downstream reads padding
        # as signal.
        "valid_extent": [[t.row, t.col, t.w, t.h] for t in tiles],
        # Where each tile's pixels REALLY came from in the slide, (y, x) px, row-major:
        # the nominal corner plus the stage jitter. Equal to the nominal corner when no
        # jitter was asked for. ASHLAR never sees this; the scorer needs it.
        "true_positions_yx": [
            [int(t.y + jitter[k][0]), int(t.x + jitter[k][1])]
            for k, t in enumerate(tiles)
        ],
        "stage_jitter_um": float(stage_jitter_um),
        "noise_frac": float(noise_frac),
        "noise_sd": sd,
        "seed": int(seed),
    }
    grid_path = outdir / "grid.json"
    grid_path.write_text(json.dumps(grid, indent=2))
    return grid_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Cut a stitched cycle OME-TIFF into an ASHLAR-readable uniform tile grid."
    )
    ap.add_argument("--image", required=True, help="stitched cycle OME-TIFF")
    ap.add_argument("--outdir", required=True, help="output dir for tiles + grid.json")
    ap.add_argument(
        "--cycle", type=int, default=0, help="cycle index recorded in grid.json"
    )
    ap.add_argument("--tile-size", type=int, required=True)
    ap.add_argument("--overlap", type=float, required=True, dest="overlap_fraction")
    ap.add_argument(
        "--canvas-like",
        nargs="*",
        default=[],
        help="every slide of the patient: the grid covers the largest of them, so all "
        "cycles share one grid (ASHLAR matches tiles one-for-one)",
    )
    ap.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="the run's pixel size; wins over the OME header",
    )
    ap.add_argument(
        "--stage-jitter-um",
        type=float,
        default=0.0,
        help="synthetic raw tiles: cut each tile up to this far (um, either way, both "
        "axes) off its nominal corner, as a real stage's positioning error (default 0)",
    )
    ap.add_argument(
        "--noise-frac",
        type=float,
        default=0.0,
        help="synthetic raw tiles: per-tile sensor noise, s.d. as a fraction of each "
        "channel's 1st-99th percentile range (default 0)",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed of the jitter and noise")
    ap.add_argument(
        "--exact-overlap",
        action="store_true",
        help="record the overlap as 1 - stride/tile_size, so ASHLAR's computed positions "
        "are the whole-pixel nominal corners",
    )
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    grid_path = write_tiles(
        a.image,
        a.outdir,
        a.tile_size,
        a.overlap_fraction,
        cycle=a.cycle,
        canvas_like=a.canvas_like,
        pixel_size_um=a.pixel_size_um,
        stage_jitter_um=a.stage_jitter_um,
        noise_frac=a.noise_frac,
        seed=a.seed,
        exact_overlap=a.exact_overlap,
    )
    grid = json.loads(Path(grid_path).read_text())
    logger.info(
        "wrote %d x %d tiles of %d px (%s um/px) to %s",
        grid["n_rows"],
        grid["n_cols"],
        grid["tile_size"],
        grid["pixel_size_um"],
        a.outdir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

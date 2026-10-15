"""Any Bio-Formats raw (OME-TIFF series, RareCyte ``.rcpnl``, ...) -> one tile-series cycle,
read through ASHLAR's own ``BioformatsReader``.

Runs INSIDE the ASHLAR image (``REGBENCH_SIF_ASHLAR``, which ships the Bio-Formats jar); the
import is lazy so the module loads where ASHLAR is not installed.

Units of ``reader.metadata.positions`` (ashlar 1.20.0, ``reg.py:349-389``
``BioformatsMetadata.tile_position`` ``[FULL: the method, read from the sdist]``): the stage
position of each series in micrometres (Y, X) with the y-axis flipped
(``position_microns *= [-1, 1]``, except InCell 1000/2000), divided by ``pixel_size``. So
they are PIXELS, in (row, col) order: ``y_px = positions[i, 0]``, ``x_px = positions[i, 1]``.
They are written as given (ASHLAR's own frame, including its flip).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from .. import tiles


def export(path, dest, nuclear_channel: int, channels=None, ffp=None, dfp=None) -> tiles.Cycle:
    """``ffp``/``dfp``: the cycle's flat-/dark-field profiles (MCMICRO ships one pair per
    cycle beside the raw files), recorded in ``cycle.json`` as given; the stitching
    front-ends correct every tile with them (``frontends.Illumination``)."""
    reg = importlib.import_module("ashlar.reg")
    reader = reg.BioformatsReader(str(path))
    md = reader.metadata
    positions = np.asarray(md.positions, dtype=float)  # (n, 2): (row, col), pixels
    n_ch = int(md.num_channels)
    if not 0 <= int(nuclear_channel) < n_ch:
        raise ValueError(f"nuclear_channel {nuclear_channel} but the file has {n_ch} channels")
    names = list(channels) if channels else [f"C{c + 1}" for c in range(n_ch)]

    def planes():
        for i in range(len(positions)):
            plane = reader.read(series=i, c=int(nuclear_channel))  # one tile
            yield i + 1, plane, positions[i, 1], positions[i, 0]

    return tiles.write_cycle(
        Path(dest),
        planes(),
        pixel_size_um=float(md.pixel_size),
        channels=names,
        nuclear_channel=int(nuclear_channel),
        ffp=ffp,
        dfp=dfp,
        extra={"source": str(path), "reader": "bioformats_tiles", "ashlar": "positions px (row, col)"},
    )

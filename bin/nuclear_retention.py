#!/usr/bin/env python3
"""Per-cell nuclear-channel median of ONE registered moving slide (NUCLEAR_RETENTION).

CELL_QC divides this by the reference's own nuclear median to get per-round nuclear
retention. The median comes from quantification's own compute_compartment_intensities,
so "median inside the nucleus" means exactly what `<nuclear>: Nucleus: Median` means in
merged_quant.csv. Only the nuclear plane is read.

Assumes the moving slide is a re-stained round of the SAME section as the reference
(cyclic IF); on serial sections the value is meaningless (spec §3.1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from logger import configure_logging, get_logger  # noqa: E402
from measurements import measurement_key  # noqa: E402
from metadata import extract_channel_names_from_ome, is_nuclear  # noqa: E402
from quantify import _load_mask, compute_compartment_intensities  # noqa: E402
from tiled_io import open_lazy  # noqa: E402

logger = get_logger(__name__)
_NAME = "nuclear"


def nuclear_channel_index(names: List[str], nuclear_markers: List[str]) -> Optional[int]:
    """Index of the first channel matching one of ``nuclear_markers``, via the shared rule.

    Delegates to ``metadata.is_nuclear`` (case-insensitive substring match), so
    the answer agrees with every other consumer of ``params.nuclear_markers``.

    Parameters
    ----------
    names : list of str
        Channel names, in image/channel order.
    nuclear_markers : list of str
        Configured nuclear/fiducial marker names.

    Returns
    -------
    int or None
        Index of the first matching channel, or ``None`` if no marker matches.
    """
    for i, name in enumerate(names):
        if is_nuclear(name, nuclear_markers):
            return i
    return None


def measure_nuclear(cell_mask, nuclei_mask, plane) -> pd.DataFrame:
    """Per-cell nuclear-channel median, plus the whole-cell median for context.

    Delegates to ``quantify.compute_compartment_intensities`` and keeps only the
    ``Median`` statistic for the ``Nucleus`` and ``Cell`` compartments, so the
    values mean exactly what the corresponding ``<nuclear>: <Compartment>:
    Median`` columns mean in ``merged_quant.csv``.

    Parameters
    ----------
    cell_mask : ndarray, shape (Y, X)
        Whole-cell instance mask (background = 0).
    nuclei_mask : ndarray or None, shape (Y, X)
        Nuclear instance mask. When ``None``, only ``Cell`` is reported.
    plane : ndarray, shape (Y, X)
        The nuclear-channel intensity plane.

    Returns
    -------
    DataFrame
        Columns ``label``, ``Nucleus`` (when ``nuclei_mask`` is given) and ``Cell``.
    """
    df = compute_compartment_intensities(cell_mask, nuclei_mask, plane, _NAME)
    out = pd.DataFrame({"label": df["label"].to_numpy()})
    for comp in ("Nucleus", "Cell"):
        key = measurement_key(_NAME, comp, "Median")
        if key in df.columns:
            out[comp] = df[key].to_numpy()
    return out


def parse_args(argv=None):
    """Parse CLI arguments for this script.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]`` (argparse's own default).

    Returns
    -------
    argparse.Namespace
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", required=True)
    p.add_argument("--mask_file", required=True)
    p.add_argument("--nuclei_mask_file", default=None)
    p.add_argument("--nuclear-markers", nargs="+", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Entry point: write the per-cell nuclear-retention CSV for one slide.

    Reads the nuclear-marker channel named by ``--nuclear-markers`` out of
    ``--image`` and measures it against ``--mask_file`` (and, if given,
    ``--nuclei_mask_file``). Writes an empty ``label``-only table instead of
    failing when no channel matches, so a slide missing the nuclear channel
    simply gets no retention key downstream.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector, forwarded to :func:`parse_args`.

    Returns
    -------
    int
        Process exit code; always ``0``.
    """
    configure_logging()
    args = parse_args(argv)
    names = extract_channel_names_from_ome(args.image) or []
    idx = nuclear_channel_index(names, args.nuclear_markers)
    if idx is None:
        logger.warning(
            "%s: no channel matches --nuclear-markers %s (channels: %s); writing an "
            "empty table, so this round gets no retention key",
            args.image, args.nuclear_markers, names,
        )
        pd.DataFrame(columns=["label"]).to_csv(args.output, index=False)
        return 0
    arr, _dtype, close = open_lazy(args.image)
    try:
        plane = np.asarray(arr[idx, :, :])
    finally:
        close()
    if np.issubdtype(plane.dtype, np.signedinteger) or np.issubdtype(plane.dtype, np.floating):
        plane = np.clip(plane, 0, None)
    cell_mask = _load_mask(args.mask_file)
    nuclei_mask = _load_mask(args.nuclei_mask_file) if args.nuclei_mask_file else None
    measure_nuclear(cell_mask, nuclei_mask, plane).to_csv(args.output, index=False)
    logger.info("wrote %s (nuclear channel %r)", args.output, names[idx])
    return 0


if __name__ == "__main__":
    sys.exit(main())

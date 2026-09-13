#!/usr/bin/env python3
"""Cell quantification for single-channel processing.

Designed for Nextflow pipelines where each channel is processed separately
and results are merged afterward. Supports both CPU and GPU processing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# Import from lib for DRY principle
from image_utils import ensure_dir, load_image
from logger import configure_logging, get_logger
from measurements import COMPARTMENTS

logger = get_logger(__name__)


__all__ = [
    "compute_compartment_intensities",
    "quantify_single_channel",
    "run_quantification",
]

# Per-compartment quantification: compartment names in the order they are emitted.
# These become part of the QuPath measurement key "<marker>: <Compartment>: <Stat>".
# (single source of truth: bin/utils/measurements.py)
COMPARTMENT_NAMES = COMPARTMENTS


def _safe_mean(sums: NDArray, counts: NDArray) -> NDArray:
    """Element-wise sums / counts, NaN where counts is 0.

    NaN, not 0.0: a compartment with no pixels was not measured, and the mean of nothing is not
    a number. 0.0 here would be indistinguishable from a compartment that WAS measured and was
    dark, which downstream reads as a negative call. bin/export_geojson.py already treats NaN as
    "missing" and omits the measurement rather than writing a bare NaN literal (which is not
    valid JSON); this makes the producer agree with that policy.
    Guarded by tests/test_absent_compartment_is_nan.py.
    """
    out = np.full_like(sums, np.nan, dtype=np.float64)
    return np.divide(sums, counts, out=out, where=counts > 0)


def _median_per_label_from_sorted(vals_sorted, labs_sorted, valid_labels):
    """Median per label from arrays already sorted by (label, value).

    ``vals_sorted`` and ``labs_sorted`` must be co-ordered by a single sort whose primary
    key is the label and whose secondary key is the value. The median of a label is then a
    positional lookup into its contiguous run: the middle element for an odd count, the mean
    of the two middle elements for an even one -- which is exactly ``np.median``'s definition.
    A label with no pixels in this compartment yields NaN, matching the ``default=np.nan``
    the labeled_comprehension calls this replaces passed (see the NaN-vs-0.0 rationale in
    compute_compartment_intensities, guarded by tests/test_absent_compartment_is_nan.py).
    """
    start = np.searchsorted(labs_sorted, valid_labels, side="left")
    stop = np.searchsorted(labs_sorted, valid_labels, side="right")
    count = stop - start
    out = np.full(valid_labels.size, np.nan, dtype=np.float64)
    present = count > 0
    if not present.any():
        return out
    s = start[present]
    c = count[present]
    lo = s + (c - 1) // 2
    hi = s + c // 2  # == lo when c is odd
    out[present] = 0.5 * (vals_sorted[lo] + vals_sorted[hi])
    return out


def compute_compartment_intensities(
    cell_mask: NDArray,
    nuclei_mask: Optional[NDArray],
    channel: NDArray,
    channel_name: str,
    expanded: bool = False,
) -> pd.DataFrame:
    """Compute per-cell signal for one channel — Median-first.

    The default per-channel statistic is **Median** and is ALWAYS emitted. With
    ``expanded=True`` it additionally emits **Mean** and **Sum** (integrated
    density). This statistic policy is independent of compartments:

    - ``nuclei_mask is None``  -> only the whole-cell ``Cell`` compartment.
    - ``nuclei_mask`` supplied  -> additionally ``Nucleus`` and ``Cytoplasm``.

    When a nuclear mask is supplied, the nuclear signal of cell ``C`` is measured
    over the pixels that belong to cell ``C`` **and** to any nucleus, i.e.
    ``(cell_mask == C) & (nuclei_mask > 0)`` — keyed by the *cell* label, so no
    nucleus->cell pairing is required and the masks need not share label IDs.
    Cytoplasm = ``Cell - Nucleus`` by subtraction (non-negative; the nuclear region
    is a subset of the cell). Works for any backend exposing both masks.

    Output columns are the *final QuPath measurement keys*, so they flow unchanged
    through merge_quant_csvs -> export_geojson -> GeoJSON:

    - ``label``
    - ``<channel>``  (== whole-cell **mean**, kept for backward compatibility and as
      FlowPath's bare-key fallback — that fast path is hard-wired to whole-cell mean)
    - ``<channel>: <Compartment>: Median``  (default statistic, always)
    - with ``expanded=True``, additionally ``: <Compartment>: Mean`` and ``: Sum``

    where ``<Compartment>`` is ``Cell`` (always) plus ``Nucleus``/``Cytoplasm`` when
    a nuclear mask is supplied.

    Parameters
    ----------
    cell_mask : ndarray, shape (Y, X)
        Whole-cell instance mask (background = 0).
    nuclei_mask : ndarray or None, shape (Y, X)
        Nuclear instance mask. When None, only the whole-cell compartment is emitted.
    channel : ndarray, shape (Y, X)
        Single-channel intensity image.
    channel_name : str
        Marker name used to build measurement keys.
    expanded : bool
        If True, also emit Mean and Sum (integrated density) per compartment.

    Returns
    -------
    DataFrame
        One row per (whole-cell) label. Empty DataFrame if no cells.
    """
    cell_mask = np.ascontiguousarray(cell_mask.squeeze())
    channel = channel.squeeze()
    has_nuclei = nuclei_mask is not None
    if has_nuclei:
        nuclei_mask = np.ascontiguousarray(nuclei_mask.squeeze())
        if cell_mask.shape != nuclei_mask.shape:
            raise ValueError(
                f"cell/nuclei mask shape mismatch: {cell_mask.shape} vs {nuclei_mask.shape}"
            )
    if cell_mask.shape != channel.shape:
        raise ValueError(
            f"mask/channel shape mismatch: {cell_mask.shape} vs {channel.shape}"
        )

    n = int(cell_mask.max()) + 1
    flat_cell_all = cell_mask.ravel()
    # Restrict every per-pixel array to cell foreground up front: background pixels are never
    # read downstream (bin 0 of every bincount below is discarded via `valid_labels != 0`), so
    # widening the WHOLE plane to float64 and later sorting it was pure waste. `flat_raw` keeps
    # the source dtype (uint16, written by SPLIT_CHANNELS) -- it feeds the composite-key sort
    # below, which needs the raw integer values, not the float64 weights.
    fg = flat_cell_all != 0
    flat_cell = flat_cell_all[fg]
    flat_raw = channel.ravel()[fg]
    flat_val = flat_raw.astype(np.float64)

    # Whole-cell sums/counts keyed by cell label.
    cell_sum = np.bincount(flat_cell, weights=flat_val, minlength=n)
    cell_count = np.bincount(flat_cell, minlength=n)

    # The label set comes from the bincount that was needed anyway: its non-zero bins ARE the
    # labels present, ascending. Output is identical to `np.unique(cell_mask)` -- np.unique
    # returns ascending distinct values and np.nonzero(counts) returns ascending indices with at
    # least one pixel -- at one linear pass instead of a comparison sort over every pixel
    # (1.6e9 on a 40k x 40k slide). The dtype is kept as the mask's so the published `label`
    # column is unchanged.
    #
    # flat_cell/flat_val are foreground-only, so bin 0 of these bincounts (and of the
    # Nucleus/Cytoplasm ones below) does not accumulate background. Only bin 0 differs, and it is
    # discarded by `valid_labels != 0` immediately below and read by nothing else.
    # Guarded by tests/test_no_full_mask_unique.py.
    valid_labels = np.nonzero(cell_count)[0]
    valid_labels = valid_labels[valid_labels != 0].astype(cell_mask.dtype, copy=False)
    if len(valid_labels) == 0:
        logger.warning("[WARN] No cells found in cell mask")
        return pd.DataFrame()
    sums = {"Cell": cell_sum}
    counts = {"Cell": cell_count}

    nuc_fg = None
    if has_nuclei:
        # Nuclear region (cell ∩ nucleus) keyed by *cell* label — label-pairing free.
        nuc_fg = nuclei_mask.ravel()[fg] > 0
        nuc_cell_labels = np.where(nuc_fg, flat_cell, 0)
        nuc_sum = np.bincount(nuc_cell_labels, weights=flat_val, minlength=n)
        nuc_count = np.bincount(nuc_cell_labels, minlength=n)
        sums["Nucleus"] = nuc_sum
        counts["Nucleus"] = nuc_count
        # Cytoplasm = whole-cell minus nuclear region (per cell, by subtraction).
        sums["Cytoplasm"] = cell_sum - nuc_sum
        counts["Cytoplasm"] = cell_count - nuc_count

    # Canonical order, restricted to the compartments actually computed.
    compartments = [c for c in COMPARTMENT_NAMES if c in sums]

    out: dict = {"label": valid_labels}
    # Backward-compatible bare marker column (== whole-cell mean). FlowPath's
    # bare-key fast path is hard-wired to (WHOLE_CELL, Mean), so this stays mean;
    # the Median default reaches FlowPath via the structured "<marker>: Cell: Median"
    # key below plus FlowPath defaulting its statistic selector to Median.
    out[channel_name] = _safe_mean(cell_sum, cell_count)[valid_labels]

    # Default statistic: MEDIAN (always). One composite-key sort replaces three
    # scipy.ndimage.labeled_comprehension calls, each of which was a Python-level callback
    # invoked once per label per compartment — 180,000 interpreter round-trips for a 60k-cell
    # slide, repeated per marker. Sorting `label << 16 | value` yields label-major, value-minor
    # order: each label's pixels land in one contiguous, already-value-sorted run, so its
    # median is a positional lookup (`_median_per_label_from_sorted`) rather than a callback.
    # Filtering the sorted arrays by the `nuc_fg` boolean mask preserves the ordering of the
    # survivors, so Nucleus and Cytoplasm come off the SAME sort instead of needing one each.
    #
    # This composite key is lexicographic ONLY while pixel values fit its low 16 bits, so it
    # requires a uint16 source (what SPLIT_CHANNELS writes). Any other dtype falls back to the
    # labeled_comprehension path, kept reachable and tested rather than left as dead code --
    # see tests/test_quantify_median.py::test_non_uint16_fallback_matches_fast_path.
    #
    # A label absent from a compartment (e.g. a cell with no nuclear overlap, so an empty
    # Nucleus/Cytoplasm compartment) yields NaN, deliberately. It used to be 0.0, reproducing
    # the older `.reindex(valid_labels).fillna(0.0)` path -- but 0.0 makes "this cell has no
    # nucleus" indistinguishable from "this cell's nucleus was measured and is dark", and
    # downstream gating reads the second. Nothing can separate them after the fact.
    # bin/export_geojson.py:78 already states the opposite policy for its own data
    # ("preserve NaN for cells that had missing raw data") and omits NaN measurements from the
    # GeoJSON rather than emitting a bare NaN literal, which is not valid JSON. This makes the
    # producer agree with its consumer.
    #
    # `Sum` stays 0.0 further down: the sum over an empty set is zero, which is a real answer.
    # Guarded by tests/test_absent_compartment_is_nan.py.
    if flat_raw.dtype == np.uint16:
        # In-place composition: Build `key` from a single astype and mutate in place
        # (`<<=`, `|=`) rather than composing naively as `flat_cell.astype(int64) << 16 |
        # flat_raw.astype(int64)`. The naive form allocates four int64 arrays in sequence,
        # but numpy's ufunc temporary elision (since 1.13) collapses the two astype calls
        # onto refcount-1 temporaries, leaving two live arrays at peak — 2.00× of one
        # int64 array. In-place composition halves that to 1.00× (610.4 MB → 305.2 MB at
        # 40M foreground pixels). `flat_raw` is uint16, so `key |= flat_raw` promotes
        # the operand for the operation without materializing a persistent int64 copy;
        # numpy casts through a small internal buffer instead. See
        # docs/perf/2026-08-26-rss.md.
        key = flat_cell.astype(np.int64)
        key <<= 16
        key |= flat_raw
        order = np.argsort(key)
        labs_sorted = flat_cell[order]
        vals_sorted = flat_val[order]
        medians = {
            "Cell": _median_per_label_from_sorted(
                vals_sorted, labs_sorted, valid_labels
            )
        }
        if has_nuclei:
            nuc_fg_sorted = nuc_fg[order]
            medians["Nucleus"] = _median_per_label_from_sorted(
                vals_sorted[nuc_fg_sorted], labs_sorted[nuc_fg_sorted], valid_labels
            )
            medians["Cytoplasm"] = _median_per_label_from_sorted(
                vals_sorted[~nuc_fg_sorted], labs_sorted[~nuc_fg_sorted], valid_labels
            )
    else:
        medians = {
            "Cell": ndimage.labeled_comprehension(
                flat_val, flat_cell, valid_labels, np.median, np.float64, np.nan
            )
        }
        if has_nuclei:
            cyto_cell_labels = np.where(nuc_fg, 0, flat_cell)
            medians["Nucleus"] = ndimage.labeled_comprehension(
                flat_val, nuc_cell_labels, valid_labels, np.median, np.float64, np.nan
            )
            medians["Cytoplasm"] = ndimage.labeled_comprehension(
                flat_val, cyto_cell_labels, valid_labels, np.median, np.float64, np.nan
            )
    for comp in compartments:
        out[f"{channel_name}: {comp}: Median"] = medians[comp]

    if expanded:
        for comp in compartments:
            out[f"{channel_name}: {comp}: Mean"] = _safe_mean(sums[comp], counts[comp])[
                valid_labels
            ]
        for comp in compartments:
            out[f"{channel_name}: {comp}: Sum"] = sums[comp][valid_labels]

    return pd.DataFrame(out)


def quantify_single_channel(
    mask: NDArray,
    channel_image: NDArray,
    channel_name: str,
    nuclei_mask: Optional[NDArray] = None,
    expanded: bool = False,
) -> pd.DataFrame:
    """Quantify a single channel (intensity only).

    Morphology is computed separately by EXTRACT_CELL_PROPERTIES.

    Delegates to :func:`compute_compartment_intensities`, which is Median-first:
    the default statistic (Median) is always computed, ``expanded`` adds Mean and
    Sum, and ``nuclei_mask`` only controls whether the Nucleus/Cytoplasm compartments
    are added alongside the whole-cell ``Cell`` compartment (so the statistic set is
    the same with or without a nuclear mask).

    Parameters
    ----------
    mask : ndarray, shape (Y, X)
        Whole-cell segmentation mask.
    channel_image : ndarray, shape (Y, X)
        Channel intensity image.
    channel_name : str
        Name of the channel/marker.
    nuclei_mask : ndarray, optional
        Nuclear segmentation mask. If given, adds Nucleus/Cytoplasm compartments.
    expanded : bool
        Also emit Mean and Sum per compartment (Median is always emitted).

    Returns
    -------
    DataFrame
        Intensity per cell: ``label``, bare ``{channel_name}`` (whole-cell mean),
        and the per-compartment measurement keys (Median always; Mean/Sum expanded).
    """
    mask = np.ascontiguousarray(mask.squeeze())
    return compute_compartment_intensities(
        mask, nuclei_mask, channel_image, channel_name, expanded=expanded
    )


def _load_mask(mask_path: str) -> NDArray:
    """Load a segmentation mask from .npy or .tif, squeezed to 2D."""
    if mask_path.endswith(".npy"):
        return np.load(mask_path).squeeze()
    loaded, _ = load_image(mask_path)
    return loaded.squeeze()


def run_quantification(
    mask_path: str,
    channel_path: str,
    output_path: str,
    channel_name: str = None,
    nuclei_mask_path: Optional[str] = None,
    expanded: bool = False,
) -> pd.DataFrame:
    """Run quantification for a single channel.

    Parameters
    ----------
    mask_path : str
        Path to whole-cell segmentation mask (.npy or .tif file).
    channel_path : str
        Path to channel image file (.tif).
    output_path : str
        Path to save output CSV file.
    channel_name : str, optional
        Explicit channel/marker name. If not provided, will be parsed
        from the channel file basename.
    nuclei_mask_path : str, optional
        Path to the nuclear mask. If provided, per-compartment quantification
        (Nucleus / Cytoplasm / Cell) is performed instead of whole-cell only.
    expanded : bool
        With ``nuclei_mask_path``, also emit Mean and Sum per compartment.

    Returns
    -------
    pd.DataFrame
        Quantification results with columns: label, {channel_name}.
        Morphology is computed separately by EXTRACT_CELL_PROPERTIES.

    Raises
    ------
    FileNotFoundError
        If mask_path or channel_path does not exist.
    ValueError
        If mask and channel have incompatible shapes.
    """
    # Use provided channel name, or extract from filename as fallback
    if channel_name is None:
        channel_name = Path(channel_path).stem.split("_")[-1]
        logger.info(f"Channel name parsed from filename: {channel_name}")

    logger.info("=" * 60)
    logger.info(f"Quantifying channel: {channel_name}")
    logger.info("=" * 60)

    # Load segmentation mask
    logger.info(f"Loading mask: {mask_path}")
    try:
        if mask_path.endswith(".npy"):
            mask = np.load(mask_path).squeeze()
        else:
            mask, _ = load_image(mask_path)
            mask = mask.squeeze()
        logger.debug(f"  Mask shape: {mask.shape}")
    except FileNotFoundError:
        logger.error(f"[FAIL] Segmentation mask not found: {mask_path}")
        raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load mask from {mask_path}: {e}")
        raise ValueError(f"Failed to load mask from {mask_path}: {e}") from e

    # Load channel image
    logger.info(f"Loading channel: {channel_path}")
    try:
        channel_image, _ = load_image(channel_path)
        logger.debug(f"  Channel shape: {channel_image.shape}")
    except FileNotFoundError:
        logger.error(f"[FAIL] Channel image not found: {channel_path}")
        raise FileNotFoundError(f"Channel image not found: {channel_path}")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load channel from {channel_path}: {e}")
        raise ValueError(f"Failed to load channel from {channel_path}: {e}") from e

    # Validate shapes match
    if mask.shape != channel_image.squeeze().shape:
        logger.error(
            f"[FAIL] Shape mismatch: mask {mask.shape} vs channel {channel_image.squeeze().shape}"
        )
        raise ValueError(
            f"Shape mismatch: mask {mask.shape} vs channel {channel_image.squeeze().shape}. "
            f"Ensure the mask and channel images have the same spatial dimensions."
        )

    # Optionally load the nuclear mask for per-compartment quantification
    nuclei_mask = None
    if nuclei_mask_path:
        logger.info(f"Loading nuclei mask: {nuclei_mask_path}")
        try:
            nuclei_mask = _load_mask(nuclei_mask_path)
        except FileNotFoundError:
            logger.error(f"[FAIL] Nuclei mask not found: {nuclei_mask_path}")
            raise FileNotFoundError(f"Nuclei mask not found: {nuclei_mask_path}")
        except Exception as e:
            logger.error(
                f"[FAIL] Failed to load nuclei mask from {nuclei_mask_path}: {e}"
            )
            raise ValueError(
                f"Failed to load nuclei mask from {nuclei_mask_path}: {e}"
            ) from e
        mode = "Median + Mean + Sum" if expanded else "Median only"
        logger.info(
            f"Per-compartment quantification enabled: Nucleus/Cytoplasm/Cell, {mode}"
        )

    # Quantify
    result_df = quantify_single_channel(
        mask,
        channel_image,
        channel_name,
        nuclei_mask=nuclei_mask,
        expanded=expanded,
    )

    # Save
    if not result_df.empty:
        result_df.to_csv(output_path, index=False)
        logger.info(f"[OK] Saved {len(result_df)} cells to {output_path}")
    else:
        logger.warning("[WARN] No results to save")
        # Empty CSV with the SAME columns a non-empty result would have, so the
        # downstream per-channel merge stays consistent when a channel's mask
        # contains no cells. Mirrors compute_compartment_intensities: bare column,
        # Median per compartment (always), + Mean/Sum when expanded. Compartments =
        # Cell only without a nuclear mask, else Nucleus/Cytoplasm/Cell.
        comps = list(COMPARTMENT_NAMES) if nuclei_mask is not None else ["Cell"]
        cols = ["label", channel_name]
        cols += [f"{channel_name}: {c}: Median" for c in comps]
        if expanded:
            cols += [f"{channel_name}: {c}: Mean" for c in comps]
            cols += [f"{channel_name}: {c}: Sum" for c in comps]
        empty_df = pd.DataFrame(columns=cols)
        empty_df.to_csv(output_path, index=False)

    logger.info("Quantification complete")

    return result_df


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Single-channel cell quantification (Nextflow compatible)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--channel_tiff",
        type=str,
        required=True,
        help="Path to single channel TIFF image",
    )
    parser.add_argument(
        "--mask_file",
        type=str,
        required=True,
        help="Path to whole-cell segmentation mask (.npy or .tif)",
    )
    parser.add_argument(
        "--nuclei_mask_file",
        type=str,
        default=None,
        help="Path to nuclear mask (.npy or .tif). If given, enables per-compartment "
        "quantification (Nucleus/Cytoplasm/Cell).",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Also emit Mean and Sum per compartment (Median is always emitted as "
        "the default statistic).",
    )
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output CSV filename (default: {channel}_quant.csv)",
    )
    parser.add_argument(
        "--channel-name",
        type=str,
        default=None,
        help="Explicit channel name (if not provided, will parse from filename)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)

    args = parse_args()

    # Ensure output directory exists
    ensure_dir(args.outdir)

    # Derive effective channel name once for consistency
    effective_channel_name = (
        args.channel_name or Path(args.channel_tiff).stem.split("_")[-1]
    )

    # Determine output path
    if args.output_file:
        output_path = os.path.join(args.outdir, args.output_file)
    else:
        output_path = os.path.join(args.outdir, f"{effective_channel_name}_quant.csv")

    # Run quantification (CPU mode)
    # Note: GPU mode removed - use GPU container if GPU acceleration needed
    logger.info("Running CPU quantification")
    run_quantification(
        mask_path=args.mask_file,
        channel_path=args.channel_tiff,
        output_path=output_path,
        channel_name=effective_channel_name,
        nuclei_mask_path=args.nuclei_mask_file,
        expanded=args.expanded,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

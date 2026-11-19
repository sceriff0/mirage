#!/usr/bin/env python3
"""Cell quantification for one channel.

Each channel is quantified separately and the results are merged afterwards.
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

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from image_utils import ensure_dir, load_image
from logger import configure_logging, get_logger
from measurements import COMPARTMENTS

logger = get_logger(__name__)


__all__ = [
    "compute_compartment_intensities",
    "quantify_single_channel",
    "run_quantification",
]

# Compartment names in emitted order; part of the key "<marker>: <Compartment>: <Stat>".
COMPARTMENT_NAMES = COMPARTMENTS


def _safe_mean(sums: NDArray, counts: NDArray) -> NDArray:
    """Element-wise sums / counts, NaN where counts is 0.

    NaN, not 0.0: an empty compartment was not measured, and 0.0 would read as a dark,
    negative cell. export_geojson.py omits NaN measurements.
    """
    out = np.full_like(sums, np.nan, dtype=np.float64)
    return np.divide(sums, counts, out=out, where=counts > 0)


def _median_per_label_from_sorted(vals_sorted, labs_sorted, valid_labels):
    """Median per label from arrays already sorted by (label, value).

    Each label's values form one contiguous sorted run, so its median is a positional
    lookup (same definition as ``np.median``). A label with no pixels yields NaN.
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
    """Compute per-cell signal for one channel.

    Median is always emitted; ``expanded=True`` adds Mean and Sum. Compartments are
    ``Cell``, plus ``Nucleus`` and ``Cytoplasm`` when ``nuclei_mask`` is given.

    The nuclear region of cell ``C`` is ``(cell_mask == C) & (nuclei_mask > 0)``, keyed
    by the cell label, so the masks need not share label IDs. Cytoplasm = Cell - Nucleus.

    Columns are the final QuPath measurement keys: ``label``, bare ``<channel>``
    (whole-cell mean; FlowPath's bare-key fallback reads it), and
    ``<channel>: <Compartment>: <Stat>``.

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
    # Foreground only: background is never read. `flat_raw` keeps the source dtype for the
    # composite-key sort below.
    fg = flat_cell_all != 0
    flat_cell = flat_cell_all[fg]
    flat_raw = channel.ravel()[fg]
    flat_val = flat_raw.astype(np.float64)

    # Whole-cell sums/counts keyed by cell label.
    cell_sum = np.bincount(flat_cell, weights=flat_val, minlength=n)
    cell_count = np.bincount(flat_cell, minlength=n)

    # Non-zero bins are the labels present, ascending: same as np.unique(cell_mask) without
    # sorting every pixel. Bin 0 is dropped; the mask dtype is kept for the `label` column.
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
    # Bare marker column = whole-cell mean; FlowPath's bare-key fast path expects a mean.
    out[channel_name] = _safe_mean(cell_sum, cell_count)[valid_labels]

    # Default statistic: MEDIAN, via one sort on `label << 16 | value`. Each label's
    # pixels form a contiguous, value-sorted run, and masking by `nuc_fg` keeps that
    # order, so Nucleus and Cytoplasm reuse the same sort. The key needs uint16 values;
    # other dtypes use labeled_comprehension.
    # An empty compartment gives a NaN median (not measured); its Sum stays 0.0.
    if flat_raw.dtype == np.uint16:
        # Built in place to keep one int64 array live instead of two at peak.
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

    Delegates to :func:`compute_compartment_intensities`.

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
    if channel_name is None:
        channel_name = Path(channel_path).stem.split("_")[-1]
        logger.info(f"Channel name parsed from filename: {channel_name}")

    logger.info("=" * 60)
    logger.info(f"Quantifying channel: {channel_name}")
    logger.info("=" * 60)

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

    if mask.shape != channel_image.squeeze().shape:
        logger.error(
            f"[FAIL] Shape mismatch: mask {mask.shape} vs channel {channel_image.squeeze().shape}"
        )
        raise ValueError(
            f"Shape mismatch: mask {mask.shape} vs channel {channel_image.squeeze().shape}. "
            f"Ensure the mask and channel images have the same spatial dimensions."
        )

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

    result_df = quantify_single_channel(
        mask,
        channel_image,
        channel_name,
        nuclei_mask=nuclei_mask,
        expanded=expanded,
    )

    if not result_df.empty:
        result_df.to_csv(output_path, index=False)
        logger.info(f"[OK] Saved {len(result_df)} cells to {output_path}")
    else:
        logger.warning("[WARN] No results to save")
        # Same columns as a non-empty result, so the per-channel merge stays consistent.
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

    ensure_dir(args.outdir)

    effective_channel_name = (
        args.channel_name or Path(args.channel_tiff).stem.split("_")[-1]
    )

    if args.output_file:
        output_path = os.path.join(args.outdir, args.output_file)
    else:
        output_path = os.path.join(args.outdir, f"{effective_channel_name}_quant.csv")

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

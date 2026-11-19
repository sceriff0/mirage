#!/usr/bin/env python3
"""
Split multichannel TIFF images into individual single-channel TIFF files.
The nuclear/fiducial channel is only saved from the reference image.

The nuclear channel is chosen by ``--nuclear-markers`` through the shared
``utils.metadata.is_nuclear`` rule. ``lib/MarkerUtils.groovy`` applies the same rule to
pre-compute each patient's ``groupTuple`` size, so the two must agree.

Usage:
    python split_multichannel.py input.ome.tiff output_folder --is-reference
    python split_multichannel.py input.ome.tiff output_folder  # skips the nuclear channel
    python split_multichannel.py in.tiff out --nuclear-markers CELLTOX
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from image_utils import ensure_dir, normalize_image_dimensions
from logger import configure_logging, get_logger
from metadata import extract_channel_names_from_ome, is_nuclear
from ome_io import write_tiff
from pixel_size import (
    read_ome_pixel_size,
    resolve_pixel_size,
    warn_on_pixel_size_mismatch,
)
from tiled_io import open_lazy
from validation import StreamingNegativeClip, clip_negative_values

logger = get_logger(__name__)

__all__ = ["main"]

#: TIFF tile size (px) of each single-channel output. Untiled TIFFs force downstream
#: region reads (MERGE_AND_PYRAMID, quantification) to decode the whole plane. Not
#: imported from convert_image.py, which runs in a different container.
SPLIT_TIFF_TILE = 2048


def split_multichannel_tiff(
    input_path,
    output_dir,
    is_reference=False,
    channel_names=None,
    nuclear_markers=None,
    pixel_size=None,
    keep_channels=None,
):
    """
    Split a multichannel TIFF into single-channel TIFFs.

    ``keep_channels``, when supplied, decides which channels are written; otherwise
    the nuclear channel is saved only if ``is_reference=True``.

    Parameters
    ----------
    input_path : str
        Path to the input multichannel TIFF file.
    output_dir : str
        Directory to save the single-channel TIFFs.
    is_reference : bool
        If True, saves the nuclear channel. If False, skips it.
    channel_names : list of str, optional
        Names for each channel. If None, tries to read from OME metadata.
    keep_channels : sequence of str, optional
        Exact channel names this slide should emit (from
        ``CsvUtils.resolveKeptChannelsPerSlide``), matched case-insensitively. When
        non-empty it replaces the ``is_reference`` nuclear drop.
    nuclear_markers : sequence of str, optional
        Marker names that identify the nuclear/fiducial channel
        (``params.nuclear_markers``). Defaults to ``DEFAULT_NUCLEAR_MARKERS``.

    Returns
    -------
    list of str
        Paths to the saved files.
    """
    # Read lazily, one channel at a time, to bound peak memory. Every channel is still
    # read once, because the negative-clip stats cover skipped channels too.
    logger.info(f"Reading: {input_path}")
    logger.info(f"  Is reference: {is_reference}")

    lazy_arr, lazy_dtype, close_lazy = open_lazy(input_path)
    closed = False
    try:
        raw_shape = lazy_arr.shape

        # (Y, X, C) on disk, by the smallest-dimension heuristic. open_lazy assumes
        # channel-first, so this falls back to a whole-stack read. The pipeline's own
        # producers write channel-first.
        channel_last = raw_shape[2] < raw_shape[0] and raw_shape[2] < raw_shape[1]

        if channel_last:
            close_lazy()
            closed = True
            logger.info(f"  Image shape: {raw_shape}, dtype: {lazy_dtype}")

            img = tifffile.imread(input_path)
            img = clip_negative_values(img, logger, stage_name="split_multichannel")
            data = normalize_image_dimensions(img)
            n_channels = data.shape[0]
            logger.info(f"  Normalized to (C, H, W) with {n_channels} channels")

            def read_channel(i):
                return data[i]

        else:
            logger.info(f"  Image shape: {raw_shape}, dtype: {lazy_dtype}")
            n_channels = raw_shape[0]
            logger.info(f"  Normalized to (C, H, W) with {n_channels} channels")

            clip_stats = StreamingNegativeClip(
                lazy_dtype, logger, stage_name="split_multichannel"
            )

            def read_channel(i):
                channel_data = np.asarray(lazy_arr[i, :, :])
                return clip_stats.process(channel_data)

        if channel_names is None:
            channel_names = extract_channel_names_from_ome(input_path) or None

        if channel_names is None:
            channel_names = [f"Channel_{i}" for i in range(n_channels)]
            logger.info("  Using default channel names")
        else:
            logger.info(f"  Channel names from metadata: {len(channel_names)} channels")

        if len(channel_names) != n_channels:
            logger.warning(
                f"  Warning: {len(channel_names)} names vs {n_channels} channels"
            )
            if len(channel_names) < n_channels:
                channel_names.extend(
                    [f"Channel_{i}" for i in range(len(channel_names), n_channels)]
                )
            else:
                channel_names = channel_names[:n_channels]

        ensure_dir(output_dir)

        # Scale goes in standard TIFF resolution tags, not an OME header: an OME header
        # would be a second source of channel names for extract_channel_names_from_ome.
        detected = read_ome_pixel_size(input_path)
        warn_on_pixel_size_mismatch(
            detected, pixel_size, source=Path(input_path).name, logger=logger
        )
        # CENTIMETER, so resolution is pixels-per-cm: 1e4 µm/cm divided by µm/px.
        resolution = (1e4 / pixel_size, 1e4 / pixel_size)

        saved_paths = []
        skipped_count = 0

        for i, name in enumerate(channel_names):
            # Read every channel, including skipped ones: they count toward the clip stats.
            channel_data = read_channel(i)

            # Case-insensitive substring match, mirrored by lib/MarkerUtils.groovy.
            is_nuclear_channel = is_nuclear(name, nuclear_markers)

            # The keep-set, when supplied, is the whole decision; it keeps channels_count
            # equal to the files that arrive. The fallback serves callers without one
            # (add_cycle's SPLIT_PRIOR_PYRAMID, manual runs). Must stay after
            # read_channel(i), or the clip stats change.
            if keep_channels:
                if name.upper() not in {k.upper() for k in keep_channels}:
                    logger.info(f"  Skipping: {name} (not in keep-set)")
                    skipped_count += 1
                    continue
            elif is_nuclear_channel and not is_reference:
                logger.info(f"  Skipping: {name} (nuclear channel from non-reference)")
                skipped_count += 1
                continue

            clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

            # Sanitising can collide (e.g. "CD3-105" and "CD3_105").
            candidate_path = os.path.join(output_dir, f"{clean_name}.tiff")
            if os.path.exists(candidate_path):
                suffix = 2
                while os.path.exists(
                    os.path.join(output_dir, f"{clean_name}_{suffix}.tiff")
                ):
                    suffix += 1
                logger.warning(
                    f"  Filename collision: '{name}' sanitized to '{clean_name}' which already exists, "
                    f"using '{clean_name}_{suffix}' instead"
                )
                clean_name = f"{clean_name}_{suffix}"

            output_path = os.path.join(output_dir, f"{clean_name}.tiff")

            # maxworkers left at default: one plane per call, so the encode queue does not
            # grow with channel count (unlike the generator-fed multi-channel writers).
            write_tiff(
                output_path,
                channel_data,
                bigtiff=True,
                compression="zlib",
                resolution=resolution,
                resolutionunit="CENTIMETER",
                tile=(SPLIT_TIFF_TILE, SPLIT_TIFF_TILE),
            )

            saved_paths.append(output_path)
            status = " (nuclear channel from reference)" if is_nuclear_channel else ""
            logger.info(f"  Saved: {clean_name}.tiff{status}")

        if not channel_last:
            clip_stats.finalize()

        logger.info(f"✓ Saved {len(saved_paths)} channels, skipped {skipped_count}")
        return saved_paths
    finally:
        if not closed:
            close_lazy()


def main():
    """CLI entry point: split a multichannel TIFF into single-channel TIFFs."""
    parser = argparse.ArgumentParser(
        description="Split multichannel TIFF into single-channel TIFFs "
        "(the nuclear channel is kept only for the reference image)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", type=str, help="Path to input multichannel TIFF file")

    parser.add_argument(
        "output_dir", type=str, help="Directory to save single-channel TIFFs"
    )

    parser.add_argument(
        "--is-reference",
        action="store_true",
        help="This is the reference image (save the nuclear channel)",
    )

    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channel names (optional, will try to read from OME metadata)",
    )

    parser.add_argument(
        "--nuclear-markers",
        nargs="+",
        default=None,
        help="Marker names identifying the nuclear/fiducial channel, which is kept "
        "only on the reference image. SPLIT_CHANNELS always passes "
        "params.nuclear_markers; the default is only for standalone use.",
    )

    parser.add_argument(
        "--keep-channels",
        nargs="+",
        default=None,
        help="Exact channel names this slide should emit, resolved once from the "
        "samplesheet by CsvUtils.resolveKeptChannelsPerSlide and carried on the meta "
        "map as keep_channels. When given it fully replaces the is-reference nuclear "
        "drop. Omitted only by callers that read channel names from OME-XML at "
        "runtime (dev's add_cycle alias SPLIT_PRIOR_PYRAMID).",
    )

    parser.add_argument(
        "--pixel-size",
        dest="pixel_size",
        type=str,
        default=None,
        help="Configured scale in µm/px (SPLIT_CHANNELS passes params.pixel_size). "
        "Stamped on each single-channel TIFF as resolution tags, and the value the "
        "input's own scale is checked against.",
    )

    args = parser.parse_args()

    # 'auto' reads this slide's OME header; CONVERT_IMAGE stamped the resolved scale on it.
    args.pixel_size = resolve_pixel_size(
        args.pixel_size, args.input, source=Path(args.input).name
    )

    configure_logging(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("SPLIT MULTICHANNEL TIFF")
    logger.info("=" * 80)

    split_multichannel_tiff(
        args.input,
        args.output_dir,
        args.is_reference,
        args.channels,
        args.nuclear_markers,
        args.pixel_size,
        args.keep_channels,
    )

    logger.info("=" * 80)
    logger.info("✓ COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    raise SystemExit(main())

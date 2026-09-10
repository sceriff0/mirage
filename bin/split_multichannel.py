#!/usr/bin/env python3
"""
Split multichannel TIFF images into individual single-channel TIFF files.
The nuclear/fiducial channel is only saved from the reference image.

Which channel counts as nuclear comes from ``--nuclear-markers`` (SPLIT_CHANNELS passes
``params.nuclear_markers``), resolved by the shared ``utils.metadata.is_nuclear`` rule --
not by a hardcoded ``"DAPI"``. That matters beyond configurability: Nextflow pre-computes
each patient's ``groupTuple`` size from the SAME rule in ``lib/MarkerUtils.groovy``
(via ``CsvUtils.countChannelsPerPatient``), so if this script kept a channel Groovy
expected to be dropped the group would over-fill and a per-patient consumer would run
twice against one published path.

Usage:
    python split_multichannel.py input.ome.tiff output_folder --is-reference
    python split_multichannel.py input.ome.tiff output_folder  # skips the nuclear channel
    python split_multichannel.py in.tiff out --nuclear-markers CELLTOX
"""

from __future__ import annotations

import argparse
import logging
import os

# Add parent directory to path to import lib modules
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

#: TIFF tile size (px) for each single-channel output this script writes -- a TIFF LAYOUT
#: choice, deliberately owned by THIS script rather than imported from
#: ``convert_image.CONVERT_TIFF_TILE`` (a different constant for a different writer, in a
#: different container image -- a cross-script import would couple their dependency sets).
#: tifffile's zarr view of a STRIPED (untiled) TIFF reports ``chunks=(1, H, W)``, one chunk
#: per whole plane, so every "read just this region" call downstream -- MERGE_AND_PYRAMID
#: and quantification, both consumers of these single-channel files -- decodes the entire
#: plane to slice it. Measured on a 6000x6000 uint16 plane: a single 2048^2 region read
#: peaks at 76.7 MiB striped vs 16.0 MiB tiled. Pinned by
#: ``tests/test_split_multichannel_lazy_read.py``.
SPLIT_TIFF_TILE = 2048


# The streaming negative-clip aggregator used to be defined here, privately. It now
# lives in bin/utils/validation.py beside the whole-array clip_negative_values it
# mirrors, because bin/apply_basic_profiles.py needs the same thing for the same
# reason -- and two copies of a log-contract shim is exactly how the two copies drift.


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

    Which channels are written is decided by ``keep_channels`` when it is supplied,
    and otherwise by the legacy rule: the nuclear channel is saved only if
    ``is_reference=True``.

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
        Exact channel names this slide should emit, resolved once per slide by
        ``CsvUtils.resolveKeptChannelsPerSlide`` and delivered as ``--keep-channels``.
        When non-empty it fully replaces the ``is_reference`` nuclear drop, so a
        non-reference slide can keep a nuclear-list marker the reference never carried.
        Matched case-insensitively against the channel names.
    nuclear_markers : sequence of str, optional
        Marker names that identify the nuclear/fiducial channel
        (``params.nuclear_markers``). Defaults to ``DEFAULT_NUCLEAR_MARKERS``.

    Returns
    -------
    list of str
        Paths to the saved files.
    """
    # Read the image lazily and process/write one channel at a time, instead of
    # tifffile.imread'ing the whole (C, H, W) stack up front. This is a peak-MEMORY
    # optimization: the write loop below always operated on a single plane, so once a
    # channel is written its array can be freed instead of sitting alongside every
    # other channel for the rest of the run. It is NOT a read-size reduction --
    # clip_negative_values' whole-image stats (below) need every channel's pixels
    # regardless of whether that channel ends up written, including a nuclear channel
    # skipped from output, so every channel is still read exactly once, same as the
    # old single tifffile.imread call decoded every page exactly once.
    logger.info(f"Reading: {input_path}")
    logger.info(f"  Is reference: {is_reference}")

    lazy_arr, lazy_dtype, close_lazy = open_lazy(input_path)
    closed = False
    try:
        raw_shape = lazy_arr.shape

        # (Y, X, C) on disk -- the same smallest-dimension heuristic
        # normalize_image_dimensions uses. open_lazy always presents the array
        # channel-FIRST (a 2-D image is promoted to C=1, matching
        # normalize_image_dimensions' own 2-D branch exactly), so it has no notion of
        # a channel-last layout on its own. This script's actual producers --
        # REGISTER's VALIS output and merge_channels_pyramid.py's prior-pyramid output
        # (data.shape unpacked as (C, H, W) at merge_channels_pyramid.py:357) -- both
        # write channel-first, so this branch is a defensive fallback this pipeline
        # does not exercise in production. The original whole-stack read handled it
        # regardless, so equivalence is preserved for it too via the exact old path.
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

        # Get channel names
        if channel_names is None:
            channel_names = extract_channel_names_from_ome(input_path) or None

        if channel_names is None:
            channel_names = [f"Channel_{i}" for i in range(n_channels)]
            logger.info("  Using default channel names")
        else:
            logger.info(f"  Channel names from metadata: {len(channel_names)} channels")

        # Validate channel count
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

        # Create output directory
        ensure_dir(output_dir)

        # These single-channel TIFFs used to be written with no scale at all -- a bare
        # imwrite, no resolution tags, no OME header -- so anything that opened one directly
        # saw 1 px = 1 unit, and MERGE_AND_PYRAMID had nothing to read even in principle.
        # They now carry the configured scale as standard TIFF resolution tags. Standard
        # tags rather than an OME header on purpose: an OME header here would be a second,
        # channel-less source of channel names for extract_channel_names_from_ome to find.
        detected = read_ome_pixel_size(input_path)
        warn_on_pixel_size_mismatch(
            detected, pixel_size, source=Path(input_path).name, logger=logger
        )
        # CENTIMETER, so resolution is pixels-per-cm: 1e4 µm/cm divided by µm/px.
        resolution = (1e4 / pixel_size, 1e4 / pixel_size)

        # Save each channel (skip the nuclear channel if not reference)
        saved_paths = []
        skipped_count = 0

        for i, name in enumerate(channel_names):
            # Read (and, on the lazy path, clip + fold into the running
            # negative-value stats) every channel, INCLUDING ones skipped below: the
            # original whole-image clip_negative_values call ran before any channel
            # was selected/dropped, so a skipped nuclear channel's pixels still have
            # to count toward the aggregate stats to match its output exactly.
            channel_data = read_channel(i)

            # The one nuclear-marker rule: case-insensitive SUBSTRING match against the
            # configured markers, shared with convert_image.py (which already moved this
            # channel to index 0) and mirrored by lib/MarkerUtils.groovy. An exact match
            # would miss names like 'DAPI_nuclear'; a hardcoded 'DAPI' would miss a
            # CELLTOX-configured run and keep a channel Groovy already counted as dropped.
            is_nuclear_channel = is_nuclear(name, nuclear_markers)

            # The keep-set, when supplied, IS the decision -- is_reference and
            # nuclear_markers are not consulted. CsvUtils.resolveKeptChannelsPerSlide
            # resolves it once per slide at samplesheet read so that each marker name is
            # emitted exactly once per patient, which is what keeps Nextflow's
            # pre-computed channels_count equal to BOTH the number of files that arrive
            # and the number of distinct markers. Deciding here instead would mean a
            # per-slide process guessing what a sibling slide emitted.
            #
            # The fallback below is not dead: dev's add_cycle SPLIT_PRIOR_PYRAMID reads channel names
            # from OME-XML at runtime and so cannot be handed a precomputed list, and
            # this script stays runnable by hand.
            #
            # NOTE the position: this sits AFTER read_channel(i), exactly where the
            # nuclear skip already sat, and must stay there. Every channel is read even
            # when it is dropped, so a skipped channel's pixels still count toward the
            # streaming negative-clip statistics -- matching the original whole-image
            # clip_negative_values call. Hoisting the skip above the read to "save work"
            # would silently change those stats.
            if keep_channels:
                if name.upper() not in {k.upper() for k in keep_channels}:
                    logger.info(f"  Skipping: {name} (not in keep-set)")
                    skipped_count += 1
                    continue
            elif is_nuclear_channel and not is_reference:
                logger.info(f"  Skipping: {name} (nuclear channel from non-reference)")
                skipped_count += 1
                continue

            # Clean the channel name for use as filename
            clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

            # Detect filename collisions from sanitization (e.g. "CD3-105" and "CD3_105")
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

            # `maxworkers` is deliberately left at tifffile's default here, unlike the
            # generator-fed writes in bin/apply_basic_profiles.py and
            # bin/merge_channels_pyramid.py:826-847. Those pin it because a single write
            # spans EVERY channel through one generator, so the encode pool's queued
            # tiles are a term that grows with the channel count -- the exact thing
            # those rewrites exist to remove. This call is neither: `channel_data` is one
            # already-resident PLANE, and this `imwrite` -- ONE whole-array call, opening
            # and closing its own TiffWriter -- runs once per channel with nothing carried
            # over to the next iteration, so there is no cross-channel accumulation to pin
            # against. Measured (tracemalloc, isolated from the resident plane) on a
            # zlib-tiled whole-array write of this shape: the encode-queue overhead is
            # roughly one plane's worth at 4096^2 and falls to a fraction of a plane at
            # 8192^2 (bounded by cpu_count() tiles in flight, not by plane size), at both
            # tifffile 2023.4.12 and 2025.5.10. SPLIT_CHANNELS' memory closure reserves
            # 32-128 GB from the whole REGISTERED slide's size, not from one channel's
            # plane, so that bounded overhead sits well inside the existing floor even on
            # the largest real slide. Pinning would only cost wall clock (serialising the
            # per-tile compression this call would otherwise parallelise) for no memory
            # benefit, so it is left unpinned.
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

    # 'auto' resolves off this slide's own OME header. Safe at this stage: CONVERT_IMAGE
    # stamps the scale it resolved onto its output, so the file reaching us carries one.
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

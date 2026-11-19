#!/usr/bin/env python3
"""Apply nf-core BASICPY's illumination profiles and reassemble the slide.

nf-core's ``basicpy`` module computes profiles only, so this script applies them::

    corrected = (image - darkfield) / flatfield

per channel, per pseudo-FOV, then reassembles the slide from the sidecar
``bin/tile_for_basic.py`` wrote.

* Fiducial skip: decided once in ``tile_for_basic`` (``corrected_channels`` in the
  sidecar); channels not listed are copied through uncorrected.
* Negative clip: one aggregate log line per image via ``utils.validation``'s
  ``StreamingNegativeClip``. Only real pixels are folded in, never the padding of an
  edge tile. A degenerate profile (NaN/inf) is not caught by the clip; that is
  ``_read_profile_stack``'s job.
* Storage dtype: clip to the dtype's range, round half-to-even, then cast (same rule as
  ``merge_channels_pyramid.to_uint16``, which lives in a different image).

Memory: the correction streams one output write-tile at a time, so peak is one tile
buffer plus the profile planes (``2 x C x preproc_tile_size^2 x 8`` bytes, float64).
The APPLY_PROFILES memory closure in ``conf/modules.config`` sizes for that term.

The write-tile grid (``WRITE_TILE``) and the pseudo-FOV grid
(``params.preproc_tile_size``) do not align. A tile can straddle up to four FOVs and
profiles are indexed in FOV-local coordinates, so each tile is corrected piecewise over
the sub-windows ``fov_tiling.fov_overlaps`` reports; they partition the tile exactly.

The profile stack's C axis is indexed by position in ``profile_channels`` (fitted
channels only), not by source channel index. A one-channel stack reads back 2-D and is
normalised; a stack whose tile shape does not match the sidecar is refused.

At upstream BASICPY defaults ``get_darkfield`` is False, so ``*-dfp.ome.tif`` is present
and all zeros. That case is detected and logged as a flatfield-only correction; a
missing darkfield file is treated the same way.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "utils"))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from fov_tiling import fov_overlaps  # noqa: E402
from logger import configure_logging, get_logger  # noqa: E402
from ome_io import ome_metadata, ome_tiff_writer  # noqa: E402
from pixel_size import (  # noqa: E402
    read_ome_pixel_size,
    resolve_pixel_size,
    warn_on_pixel_size_mismatch,
)
from tiled_io import open_lazy  # noqa: E402
from validation import StreamingImageStats, StreamingNegativeClip  # noqa: E402

logger = get_logger(__name__)

__all__ = ["apply_basic_profiles", "main"]

#: Sidecar layouts this script understands; an unknown one is refused rather than
#: misread.
SUPPORTED_SIDECAR_VERSIONS = (1,)

STAGE = "apply_basic_profiles"

#: Output write-tile size in pixels: the TIFF tile layout and the streaming unit.
#: Independent of params.preproc_tile_size (see module docstring). The APPLY_PROFILES
#: memory formula in conf/modules.config is derived from this value.
WRITE_TILE = 2048


def _read_profile_stack(path, n_expected, tile_shape, label, require_positive=False):
    """Read a ``/opt/main.py`` profile file as ``(n_expected, tile_h, tile_w)`` float64.

    Validates the values as well as the shape: a NaN or inf profile would pass every
    downstream check (``nan < 0`` is False) and yield plausible-looking garbage pixels.

    ``require_positive`` is for the flatfield, a normalised gain that is strictly positive
    by construction. A zero or negative entry means the fit failed or the flatfield and
    darkfield were swapped (the default darkfield is all zeros).
    """
    data = np.asarray(tifffile.imread(str(path)))
    if data.ndim == 2:
        # tifffile drops the singleton leading axis of a one-channel stack.
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError(
            f"{path}: expected a 2-D or 3-D {label} profile stack, got shape {data.shape}"
        )
    if data.shape[0] != n_expected:
        raise ValueError(
            f"{path}: {label} profile stack has {data.shape[0]} channel(s) but the "
            f"sidecar says {n_expected} channel(s) were fitted. The profiles do not "
            "belong to this tile stack."
        )
    if tuple(data.shape[1:]) != tuple(tile_shape):
        raise ValueError(
            f"{path}: {label} profile tile shape {tuple(data.shape[1:])} does not match "
            f"the sidecar's tile shape {tuple(tile_shape)}. The profiles belong to a "
            "different tile grid; refusing to broadcast them onto this one."
        )

    data = data.astype(np.float64, copy=False)

    if not np.all(np.isfinite(data)):
        n_bad = int(np.count_nonzero(~np.isfinite(data)))
        raise ValueError(
            f"{path}: {n_bad} value(s) in the {label} profile are not finite (NaN or "
            "inf). Applying them would produce NaN pixels that every downstream check "
            "passes -- `nan < 0` is False, so the negative clip never sees them."
        )

    if require_positive and data.min() <= 0:
        raise ValueError(
            f"{path}: the {label} profile must be strictly positive (min is "
            f"{data.min():.6g}); it is a normalised gain and dividing by a zero or "
            "negative entry is silent. If this is BASICPY's *-dfp file, the flatfield "
            "and darkfield have been swapped: at upstream defaults the darkfield is all "
            "zeros."
        )

    return data


def _to_storage_dtype(data, dtype):
    """Clip to the dtype's range, round, then cast; never a bare ``.astype()``.

    ``.astype()`` truncates toward zero, a one-sided bias of mean -0.5 LSB. Clipping comes
    before rounding because 65535.6 rounds to 65536, which wraps to 0 in uint16.
    """
    if data.dtype == dtype:
        return data
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.round(np.clip(data, info.min, info.max)).astype(dtype)
    return data.astype(dtype)


def apply_basic_profiles(
    image_path,
    sidecar_path,
    flatfield_path,
    darkfield_path,
    output_path,
    pixel_size=None,
):
    """Correct ``image_path`` with the fitted profiles and write the reassembled slide."""
    manifest = json.loads(Path(sidecar_path).read_text())
    version = manifest.get("format_version")
    if version not in SUPPORTED_SIDECAR_VERSIONS:
        raise ValueError(
            f"{sidecar_path}: format_version {version!r} is not one of "
            f"{SUPPORTED_SIDECAR_VERSIONS}. Refusing to guess what its fields mean."
        )

    channel_names = list(manifest["channel_names"])
    profile_channels = list(manifest["profile_channels"])
    corrected_channels = set(manifest["corrected_channels"])
    positions = [tuple(p) for p in manifest["positions"]]
    height, width = manifest["image_shape"]
    n_fovs_y, n_fovs_x = manifest["n_fovs_y"], manifest["n_fovs_x"]
    storage_dtype = np.dtype(manifest["source_dtype"])

    tile_shape = tuple(manifest["tile_shape"])

    flatfields = _read_profile_stack(
        flatfield_path,
        len(profile_channels),
        tile_shape,
        "flatfield",
        require_positive=True,
    )
    if darkfield_path is None or not Path(darkfield_path).exists():
        # The vendored module always declares the file, but a hand-run caller may have none.
        darkfields = np.zeros((len(profile_channels), *tile_shape), dtype=np.float64)
        logger.info(
            "No darkfield file supplied -- correction is flatfield-only "
            "(image / flatfield), no additive offset is removed."
        )
    else:
        darkfields = _read_profile_stack(
            darkfield_path, len(profile_channels), tile_shape, "darkfield"
        )
        if not np.any(darkfields):
            # Default BASICPY config: the darkfield file is all zeros. Log it as such rather
            # than implying an offset was subtracted.
            logger.info(
                f"{Path(darkfield_path).name} is all zeros (BASICPY ran with "
                "get_darkfield=False, its default) -- no darkfield was estimated, so the "
                "correction is flatfield-only (image / flatfield)."
            )
        else:
            logger.info(
                f"Subtracting a fitted darkfield (min={darkfields.min():.4g}, "
                f"max={darkfields.max():.4g}) before dividing by the flatfield."
            )

    # Where each source channel's profile sits on the profile stack's C axis.
    profile_index = {source: k for k, source in enumerate(profile_channels)}

    detected_x, detected_y = read_ome_pixel_size(image_path)
    warn_on_pixel_size_mismatch(
        (detected_x, detected_y),
        pixel_size,
        source=Path(image_path).name,
        logger=logger,
    )
    pixel_size_x = detected_x if detected_x is not None else pixel_size
    pixel_size_y = detected_y if detected_y is not None else pixel_size

    logger.info(
        f"Applying profiles to {len(corrected_channels)}/{len(channel_names)} channel(s) "
        f"over {n_fovs_y}x{n_fovs_x} pseudo-FOVs, streaming in {WRITE_TILE}px write-tiles"
    )

    n_channels = len(channel_names)
    # float32 working dtype per tile, matching the whole-array computation so the
    # streamed output is bit-identical to it.
    clip_stats = StreamingNegativeClip(np.dtype(np.float32), logger, stage_name=STAGE)
    out_stats = StreamingImageStats(
        storage_dtype, (n_channels, height, width), "after_basic_correction", logger
    )

    def _correct_tile(crop, index, y0, x0):
        """One write-tile's pixels, corrected, as float32.

        The uint16 crop is cast to float32, the float64 profiles promote the arithmetic to
        float64, and the result is cast back to float32.
        """
        if index not in corrected_channels:
            # Copied through, but still cast to float32 so every channel takes the same
            # clip and storage cast.
            return crop.astype(np.float32)

        k = profile_index[index]
        out = np.empty(crop.shape, dtype=np.float32)
        for fov_index, window, profile in fov_overlaps(
            positions, y0, x0, crop.shape[0], crop.shape[1]
        ):
            out[window] = (
                (crop[window].astype(np.float32) - darkfields[k][profile])
                / flatfields[k][profile]
            ).astype(np.float32)
        return out

    def _tiles():
        """Yield the output in tifffile's tile order (channel, row, col)."""
        for index in range(n_channels):
            if index not in corrected_channels:
                logger.debug(
                    f"  [SKIP] {channel_names[index]!r} (nuclear/fiducial, or nothing fitted)"
                )
            else:
                logger.debug(f"  [OK] {channel_names[index]!r} corrected")
            for y0 in range(0, height, WRITE_TILE):
                for x0 in range(0, width, WRITE_TILE):
                    th = min(WRITE_TILE, height - y0)
                    tw = min(WRITE_TILE, width - x0)
                    crop = np.asarray(
                        arr[index, slice(y0, y0 + th), slice(x0, x0 + tw)]
                    )
                    valid = _correct_tile(crop, index, y0, x0)
                    # Only the valid sub-array is folded into the aggregates; the pad below is a
                    # TIFF layout artefact.
                    valid = clip_stats.process(valid)
                    valid = _to_storage_dtype(valid, storage_dtype)
                    out_stats.process(valid)

                    # A fresh buffer per tile: the writer consumes lazily, so a reused buffer
                    # would be overwritten before it was encoded.
                    tile = np.zeros((WRITE_TILE, WRITE_TILE), dtype=storage_dtype)
                    tile[:th, :tw] = valid
                    yield tile

    arr, _dtype, close = open_lazy(image_path)
    try:
        # Metadata dict and writer come from bin/utils/ome_io.py.
        metadata = ome_metadata(list(channel_names), (pixel_size_x, pixel_size_y))
        with ome_tiff_writer(str(output_path), bigtiff=True, ome=True) as tw:
            tw.write(
                _tiles(),
                shape=(n_channels, height, width),
                dtype=storage_dtype,
                photometric="minisblack",
                metadata=metadata,
                compression="zlib",
                tile=(WRITE_TILE, WRITE_TILE),
                # maxworkers=1: with tifffile's default (os.cpu_count()), encoded tiles of a
                # compressed, generator-fed write queue ahead of the writer, and peak memory
                # grows with channel count. One worker keeps it flat. See
                # bin/merge_channels_pyramid.py for the same fix.
                maxworkers=1,
            )
    finally:
        close()

    # After the last tile, so both aggregates cover the whole image.
    clip_stats.finalize()
    out_stats.finalize()

    logger.info(f"[OK] Saved OME-TIFF with {n_channels} channels to {output_path}")
    # Returns the output path, not the pixels: returning the assembled slide would
    # reintroduce the slide-sized peak.
    return Path(output_path)


def parse_args(argv=None):
    """Parse the BASICPY-apply CLI.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        ``image``, ``sidecar``, ``flatfield``, ``darkfield`` (optional),
        ``output``, ``pixel_size`` (may be the literal ``'auto'``), ``log_level``.
    """
    parser = argparse.ArgumentParser(
        description="Apply BASICPY flatfield/darkfield profiles and reassemble the slide."
    )
    parser.add_argument("--image", required=True, help="Input OME-TIFF (C, Y, X).")
    parser.add_argument("--sidecar", required=True, help="Tile-position JSON.")
    parser.add_argument(
        "--flatfield", required=True, help="*-ffp.ome.tif from BASICPY."
    )
    # Optional: at upstream defaults BASICPY estimates no darkfield.
    parser.add_argument(
        "--darkfield", default=None, help="*-dfp.ome.tif from BASICPY (optional)."
    )
    parser.add_argument("--output", required=True, help="Corrected OME-TIFF to write.")
    parser.add_argument("--pixel-size", dest="pixel_size", type=str, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point: apply the flatfield/darkfield profiles and reassemble the slide.

    Returns
    -------
    int
        0 on success. Failures propagate as exceptions.
    """
    args = parse_args(argv)
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    apply_basic_profiles(
        args.image,
        args.sidecar,
        args.flatfield,
        args.darkfield,
        args.output,
        pixel_size=resolve_pixel_size(args.pixel_size, args.image, source=args.image),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

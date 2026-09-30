#!/usr/bin/env python3
"""Apply nf-core BASICPY's illumination profiles and reassemble the slide.

nf-core's ``basicpy`` module computes PROFILES ONLY. Upstream, mcmicro applies them
downstream inside ASHLAR; mirage has no ASHLAR, so this script is the missing half::

    corrected = (image - darkfield) / flatfield

per channel, per pseudo-FOV, followed by reassembly from the sidecar
``bin/tile_for_basic.py`` wrote.

WHAT IS PRESERVED FROM THE IN-PROCESS BaSiC PATH, and where it comes from:

* **The fiducial skip.** Not re-decided here. ``tile_for_basic`` resolved it once through
  ``utils.metadata.is_nuclear`` and recorded ``corrected_channels`` in the sidecar; this
  script applies profiles to exactly those channels and copies the rest through. Making
  the decision in one place is what stops the two halves drifting -- the drift that
  ``bin/preprocess.py``'s comment records (an earlier version tested ``"DAPI" in
  name.upper()`` and silently corrected a configured CELLTOX fiducial).

* **The negative clip.** One aggregate line, with its percentage computed against the
  WHOLE image, from ``bin/utils/validation.py``'s ``StreamingNegativeClip``. Not forked
  and not re-implemented: that class is the chunked counterpart of ``clip_negative_values``
  and folds per-tile counts into exactly the numbers, and the exact log lines, a single
  whole-array call would have produced. ``bin/split_multichannel.py`` uses the same class
  for the same reason. This script used to call the whole-array function directly, on the
  ``np.stack`` the in-process path built before casting; it no longer holds a whole array
  to call it on. Only real pixels are folded in -- a partial write-tile at the slide edge
  is passed as its valid sub-array, never as the padded buffer, or the padding would enter
  the reported percentage.

  Its RATIONALE has changed and the old one is not carried over. ``bin/preprocess.py``
  explained the clip by BaSiC's darkfield exceeding a pixel value; the pipeline now runs
  BASICPY at upstream defaults, where ``get_darkfield`` is False, so no additive offset is
  subtracted and that specific mechanism is largely gone. What is true today: dividing by
  a fitted flatfield cannot make a non-negative pixel negative, so on the shipped
  configuration this clip is expected to find nothing -- it is kept because the contract
  is what downstream reads, and because this script still subtracts whatever darkfield it
  is handed (a real one when run standalone, or if the defaults decision is revisited).

  What it does NOT do is catch a degenerate profile. `nan < 0` is False, so
  ``detect_negative_values`` never sees one; that is ``_read_profile_stack``'s job, and it
  rejects non-finite entries and a non-positive flatfield BEFORE any arithmetic runs.

* **The storage dtype rule.** Clip to the dtype's range, ROUND (half-to-even), then cast.
  ``.astype()`` truncates toward zero, which is a one-sided -0.5 LSB bias on every
  non-integral pixel and never averages out. Same rule, same words, as
  ``merge_channels_pyramid.to_uint16``; written out here rather than imported because
  that module lives in a different container image and pulls a different dependency set.

NOTHING SLIDE-SIZED IS EVER RESIDENT, and this is the half of the path where that used
to be false. The correction is applied one OUTPUT WRITE-TILE at a time: read that tile's
region through the lazy zarr view, correct it, clip it, cast it, hand it to ``tifffile``'s
iterator-fed writer, drop it. Peak is one write-tile buffer -- constant in slide size,
the same property ``bin/tiled_stitch.py`` has on the STARE path -- PLUS the profile
planes, which are held for the whole run and are NOT constant in channel count: BASICPY
returns one flatfield and one darkfield plane per fitted channel, at full pseudo-FOV
resolution, held as float64, i.e. ``2 x C x preproc_tile_size^2 x 8`` bytes. That is a
deliberate, small, accepted term -- ``conf/modules.config``'s APPLY_PROFILES memory
closure states it and sizes for it (allowing up to 16 channels explicitly; a wider panel
is absorbed by the retry ramp) rather than pretending it is not there.

An earlier version accumulated every corrected channel into a list, ``np.stack``-ed it,
clipped the stack, and cast the stack -- with the list still referenced, so the stack copy
and the cast's temporaries were live at the same time. That is ~8 x C x H x W bytes, about
115 GB on an 8-channel 40000x30000 uint16 slide, and it is why the memory ask here was
size-linear.

THE WRITE-TILE GRID AND THE FOV GRID DO NOT ALIGN, and must not be made to. Write tiles are
``WRITE_TILE`` px (a TIFF layout choice); pseudo-FOVs are ``params.preproc_tile_size`` px (a
BaSiC sampling choice). Any output tile can therefore straddle up to four FOVs, and an
illumination profile is indexed in FOV-LOCAL coordinates -- so each tile is corrected
PIECEWISE, over the sub-windows ``fov_tiling.fov_overlaps`` reports. Those sub-windows
partition the tile exactly, so every pixel is corrected once: not twice at a seam, and not
zero times in a gap.

THE PROFILE STACK'S SHAPE. ``/opt/main.py`` builds ``np.array(flatfields)`` -- one 2-D
plane per FITTED channel, in tile-stack order -- and writes it with ``ome=True``. So its
C axis is indexed by position in ``profile_channels``, not by source channel index, and
the two differ exactly when a fiducial was skipped. tifffile drops a singleton leading
axis on read, so a single-channel profile comes back 2-D; ``_read_profile_stack``
normalises that rather than letting it broadcast silently, and checks the tile shape --
a 128x128 working-size profile would otherwise broadcast or fail with a NumPy shape error
instead of saying that the profiles belong to a different tile grid.

THE DARKFIELD IS ZERO ON THE SHIPPED CONFIGURATION. ``conf/modules.config`` runs BASICPY
at upstream defaults, where ``get_darkfield`` is False. basicpy then never updates its
``D_R``/``D_Z`` terms from their zero initialisation, and ``fit`` still does
``self.darkfield = skimage_resize(D, images.shape[1:])`` -- so ``*-dfp.ome.tif`` is
PRESENT, correctly shaped, and all zeros. That is detected and logged; a missing file is
also accepted and treated as zero. Neither is assumed: the correction is reported as
flatfield-only rather than described as though an offset had been removed.
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

#: The sidecar layouts this script understands. Refusing an unknown one beats misreading
#: a renamed field and silently correcting the wrong channels.
SUPPORTED_SIDECAR_VERSIONS = (1,)

STAGE = "apply_basic_profiles"

#: Output write-tile size, in pixels -- both the TIFF's tile layout and the streaming
#: unit, deliberately the same number so the correction is computed exactly once per
#: tile written. It is a TIFF layout choice and is INDEPENDENT of
#: params.preproc_tile_size, the pseudo-FOV size BaSiC fits on; see the module
#: docstring on why the two grids must not be conflated. conf/modules.config's memory
#: formula for APPLY_PROFILES is derived from this value.
WRITE_TILE = 2048


def _read_profile_stack(path, n_expected, tile_shape, label, require_positive=False):
    """Read a ``/opt/main.py`` profile file as ``(n_expected, tile_h, tile_w)`` float64.

    Validates the VALUES, not only the shape. A degenerate profile is silent all the way
    to the published slide: ``x / 0`` is ``inf``, ``0 / 0`` and ``nan - nan`` are ``nan``,
    and neither is caught downstream -- ``clip_negative_values`` -> ``detect_negative_values``
    tests ``data < 0``, which is False for both, and ``np.round(np.clip(nan)).astype(uint16)``
    is undefined behaviour that yields a plausible-looking integer. So the run stays green
    and the pixels are garbage.

    ``require_positive`` is for the flatfield: it is a normalised gain, strictly positive
    by construction (basicpy divides ``S`` by its own mean and raises outright if
    ``S.max() == 0``). A zero or negative entry means the fit failed, or that the
    flatfield and darkfield arrived the wrong way round -- at upstream defaults the
    darkfield is ALL ZEROS, so a swapped pair lands here as an all-zero flatfield and is
    refused rather than dividing every pixel by zero.
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
    """Clip to the dtype's range, ROUND, then cast -- never a bare ``.astype()``.

    ``.astype()`` truncates toward zero. On intensity data that later becomes a measured
    per-cell statistic that is a systematic downward bias of up to one count, mean -0.5
    LSB, one-sided, and it never averages out. Clipping stays ahead of the round because
    65535.6 rounds to 65536, which wraps to 0 in uint16.
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
        # Handled, not assumed. The vendored module always declares the file as an output,
        # but the script is hand-runnable and a caller may legitimately have none.
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
            # The SHIPPED case: conf/modules.config runs BASICPY at upstream defaults,
            # where get_darkfield is False, so basicpy resizes its zero-initialised
            # darkfield to the tile shape and the file is all zeros. Say so, rather than
            # logging a correction that implies an offset was subtracted.
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
    # float32 is what the whole-array path produced: np.stack over a mix of corrected
    # float32 planes and untouched integer planes promoted everything to float32, and the
    # clip then ran on that. Keeping the per-tile working dtype identical is what makes
    # the streamed output bit-identical to the assembled one.
    clip_stats = StreamingNegativeClip(np.dtype(np.float32), logger, stage_name=STAGE)
    out_stats = StreamingImageStats(
        storage_dtype, (n_channels, height, width), "after_basic_correction", logger
    )

    def _correct_tile(crop, index, y0, x0):
        """One write-tile's pixels, corrected, as float32.

        The arithmetic is elementwise-identical to the whole-array path's
        ``(fov_stack.astype(float32) - darkfield) / flatfield`` then ``.astype(float32)``:
        the uint16 crop is cast to float32, the float64 profiles promote the subtraction
        and the division to float64, and the result is cast back down. Only the ORDER the
        pixels are visited in has changed.
        """
        if index not in corrected_channels:
            # Copied through, but still cast: np.stack promoted the skipped integer
            # channels to float32 alongside the corrected ones, so the clip saw float32
            # for every channel and the storage cast ran from float32 for every channel.
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
                    # Only the VALID sub-array is folded into the aggregates; the pad
                    # below is a TIFF layout artefact and is not part of the image.
                    valid = clip_stats.process(valid)
                    valid = _to_storage_dtype(valid, storage_dtype)
                    out_stats.process(valid)

                    # A fresh full-size buffer per tile, as bin/tiled_stitch.py does: the
                    # writer consumes lazily, so a reused buffer would be overwritten
                    # before it was encoded.
                    tile = np.zeros((WRITE_TILE, WRITE_TILE), dtype=storage_dtype)
                    tile[:th, :tw] = valid
                    yield tile

    arr, _dtype, close = open_lazy(image_path)
    try:
        # The dict AND the writer come from bin/utils/ome_io.py. This file used to
        # rebuild both -- one of the four independent copies of the same six keys.
        # The header's attribute ORDER now follows ome_io.ome_metadata's own order
        # (the order convert_image.py already used), not this file's former order;
        # pixels and values are unchanged.
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
                # ONE COMPRESSOR THREAD, PINNED -- same defect and same fix as
                # bin/merge_channels_pyramid.py:826-847 (read that comment for the full
                # mechanism). This write is channel-major and tile-major out of `_tiles()`
                # above, so it is exactly as generator-fed as that one; tifffile's default
                # `maxworkers=os.cpu_count()` queues encoded tiles ahead of the writer for
                # a COMPRESSED tiled write, and the queue is an unbounded, CHANNEL-COUNT
                # -scaled term in the peak. Measured with the same harness shape as that
                # file's (4096^2 uint16, tile=2048, zlib), peak in units of one plane:
                #
                #                       C=2   C=4   C=8   C=12   C=16
                #   tifffile 2023.4.12  2.24  2.23  2.23  2.23   2.23  (default maxworkers)
                #   tifffile 2025.5.10  3.76  5.75  9.18  12.37  15.55 (default maxworkers)
                #
                # containers/preprocess/Dockerfile pins 2025.5.10 -- the same version
                # `conf/modules.config`'s APPLY_PROFILES memory closure was NOT re-derived
                # against, because it was written on this file's 2023.4.12 development
                # host. `maxworkers=1` removes the slope at both versions: 2023.4.12 goes
                # to a flat 1.11 planes, 2025.5.10 to a flat 0.86 planes, regardless of C.
                # Written bytes are unaffected -- tests/test_apply_basic_profiles.py's
                # pixel- and byte-level assertions pass at both versions.
                maxworkers=1,
            )
    finally:
        close()

    # After the last tile, so both aggregates cover the whole image -- one line each, with
    # the same numbers the whole-array calls printed.
    clip_stats.finalize()
    out_stats.finalize()

    logger.info(f"[OK] Saved OME-TIFF with {n_channels} channels to {output_path}")
    # Returns the output PATH, not the pixels. The whole-array version returned the
    # assembled slide, which is precisely the object this function now exists not to
    # build; handing one back would reintroduce the peak at every call site.
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
    # Optional: at upstream defaults BASICPY estimates no darkfield, and while it still
    # writes an all-zero *-dfp.ome.tif, a caller with none is a legitimate case.
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

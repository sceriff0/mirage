#!/usr/bin/env python3
"""Write a stitched slide as a multi-SITE OME-TIFF so nf-core's BASICPY will accept it.

WHY THIS EXISTS. nf-core's ``basicpy`` module runs
``labsyspharm/basicpy-docker-mcmicro``'s ``/opt/main.py``. That script reads the input
through Bio-Formats, builds its field-of-view axis as::

    istack = istack.stack(I=('M', 'T', 'Z')).transpose('C', 'I', 'Y', 'X')
    if len(istack.coords['I']) < 2 and not args.ignore_single_image_error:
        raise RuntimeError("The image is single sited. Was it saved in the correct way?")

and then fits one profile per channel (``for c, channel_stack in enumerate(istack, 1)``).
A mirage slide is a single stitched plane per channel -- ``SizeM = SizeT = SizeZ = 1`` --
so it is refused outright. mirage's in-process BaSiC path never hit this because it
tiled the slide into pseudo-FOVs itself and handed the tile STACK straight to
``BaSiC.fit_transform``.

This module does the same tiling, but writes the result to disk on the ``Z`` axis:
``len(I) = SizeM * SizeT * SizeZ`` and mirage writes no M and no T, so every site has to
come from Z. Axis order is ``CZYX`` and not ``ZCYX`` because the module iterates
CHANNELS and fits one profile per channel; putting tiles on C would fit one profile per
tile and mix every marker together.

TILING IS NOT REIMPLEMENTED. ``count_fovs`` / ``fov_positions`` / ``iter_padded_fovs``
come from ``bin/utils/fov_tiling.py`` -- the same non-overlapping, remainder-distributing
grid the deleted in-process path used, including its zero-padding of edge tiles to the
grid's maximum tile size. That padding is not a leftover: BaSiCPy's ``fit()`` takes one
``(N, Y, X)`` ndarray and has no ragged-input path, so tiles that differ by a pixel MUST
be padded to a common shape or they cannot be passed at all. The fit input is byte-for-byte
what the in-process path produced.

NOTHING SLIDE-SIZED IS EVER RESIDENT. The grid is laid out by ``fov_positions``, which is
pure arithmetic on ``(H, W, n_fovs_y, n_fovs_x)`` and reads no pixels; each tile is then
read as its own region through the lazy zarr view and written straight out. Peak is ONE
padded tile, independent of both slide size and channel count. This is the same discipline
as ``bin/tiled_stitch.py`` on the STARE path, and ``conf/modules.config``'s memory formula
for this process is derived from the tile size accordingly.

An earlier version read and split a whole channel purely to learn ``positions`` and
``tile_shape``, discarding the pixels -- geometry that ``fov_positions`` now returns for
free. That is why ``fov_tiling`` separates the grid from the array at all.

THE FIDUCIAL SKIP IS MADE HERE, ONCE. The removed in-process path skipped BaSiC for the
nuclear/fiducial channel, and its comment recorded that an earlier version tested
``"DAPI" in name.upper()`` directly and so silently corrected a configured CELLTOX
fiducial. That decision now happens exactly once -- here, through
``utils.metadata.is_nuclear``, the shared rule -- and is RECORDED in the sidecar.
``bin/apply_basic_profiles.py`` reads the answer instead of re-deciding it, so the two
halves cannot drift.

THE SIDECAR. A JSON file next to the tile stack, carrying the tile positions verbatim as
``split_image_into_fovs`` returned them, so the reassembly step never re-derives the
grid from the FOV size. See ``docs/basic_illumination.md``.
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
from fov_tiling import count_fovs, fov_positions, iter_padded_fovs  # noqa: E402
from logger import configure_logging, get_logger  # noqa: E402
from metadata import is_nuclear  # noqa: E402
from ome_io import write_tiff  # noqa: E402
from tiled_io import open_lazy  # noqa: E402

logger = get_logger(__name__)

__all__ = ["tile_for_basic", "main"]

#: Bumped whenever the sidecar's meaning changes. ``apply_basic_profiles`` refuses a
#: version it does not know rather than misreading a field.
SIDECAR_FORMAT_VERSION = 1


def _channel_axis_length(image_path, channel_names):
    """(n_channels, height, width) for a 2-D or (C, H, W) OME-TIFF, without decoding it."""
    with tifffile.TiffFile(str(image_path)) as tif:
        series = tif.series[0]
        shape = tuple(series.shape)
        ndim = series.ndim
        dtype = series.dtype

    if ndim == 2:
        return 1, shape[0], shape[1], dtype
    if ndim == 3:
        if shape[2] == len(channel_names) and shape[0] != len(channel_names):
            raise ValueError(
                f"{image_path}: channel-last (Y, X, C) layout is not supported here. "
                "CONVERT_IMAGE and APPLY_PROFILES both emit axes='CYX'; a channel-last "
                "slide has not been through them."
            )
        return shape[0], shape[1], shape[2], dtype
    raise ValueError(
        f"{image_path}: expected a 2-D or 3-D (C, Y, X) image, got {shape}"
    )


def _resolve_channel_names(channel_names, n_channels):
    """Pad or truncate the supplied names to the image's own channel count."""
    names = list(channel_names or [])
    if len(names) < n_channels:
        logger.warning(
            f"Image has {n_channels} channels but {len(names)} channel names were "
            "supplied. Naming the remainder Channel_N."
        )
        names += [f"Channel_{i}" for i in range(len(names), n_channels)]
    elif len(names) > n_channels:
        logger.warning(
            f"{len(names)} channel names supplied for {n_channels} channels. Truncating."
        )
        names = names[:n_channels]
    return names


def tile_for_basic(
    image_path,
    output_path,
    sidecar_path,
    channel_names,
    fov_size=(1950, 1950),
    skip_nuclear=True,
    nuclear_markers=None,
):
    """Write ``image_path``'s pseudo-FOV tiles as a ``CZYX`` OME-TIFF plus a JSON sidecar.

    Returns the sidecar dict (also written to ``sidecar_path``).
    """
    n_channels, height, width, dtype = _channel_axis_length(image_path, channel_names)
    names = _resolve_channel_names(channel_names, n_channels)

    n_fovs_y, n_fovs_x = count_fovs((height, width), tuple(fov_size))
    n_tiles = n_fovs_y * n_fovs_x
    if n_tiles < 2:
        # /opt/main.py refuses a single-sited stack, and it is right to: BaSiC estimates
        # shading across a POPULATION of fields, so one field is not an estimate at all.
        # Fail here, naming the knob, rather than inside a vendored container.
        largest = max(1, min(height, width) // 2)
        raise ValueError(
            f"{image_path}: a {height}x{width} image tiled at fov_size={tuple(fov_size)} "
            f"gives {n_tiles} tile(s). BaSiC needs at least 2 fields to fit a profile and "
            "nf-core's basicpy module refuses a single-sited image. Lower "
            f"--preproc_tile_size to at most {largest}."
        )

    correctable = [
        i
        for i, name in enumerate(names)
        if not (skip_nuclear and is_nuclear(name, nuclear_markers))
    ]
    skipped = [i for i in range(n_channels) if i not in correctable]

    # A panel whose every channel is the fiducial (a CELLTOX-only panel is a supported
    # input) leaves nothing to correct -- but an OME-TIFF with zero channels is not a
    # thing, and the module still has to run. So the STACK carries every channel while
    # `corrected_channels` stays empty: profiles get fitted and then applied to nothing.
    profile_channels = correctable or list(range(n_channels))
    if not correctable:
        logger.warning(
            "Every channel is a configured nuclear/fiducial marker, so no channel will "
            "be corrected. Tiling all channels anyway so the module has an input; the "
            "fitted profiles are discarded downstream."
        )

    logger.info(
        f"Tiling {n_channels} channel(s) of {height}x{width} into "
        f"{n_fovs_y}x{n_fovs_x} = {n_tiles} pseudo-FOVs "
        f"(fitting {len(profile_channels)}, skipping {len(skipped)})"
    )

    # Pure geometry: no pixels read, nothing image-sized allocated. Identical to what
    # split_image_into_fovs would have returned for this grid
    # (tests/test_fov_tiling.py::test_pure_geometry_matches_the_materialising_split).
    positions, tile_shape = fov_positions((height, width), n_fovs_y, n_fovs_x)

    arr, _dtype, close = open_lazy(image_path)
    try:

        def _read_region(source_index):
            """Lazy region reader bound to one source channel.

            Bound through a factory rather than closing over the loop variable. This is
            NOT currently load-bearing: ``_planes`` below drives each channel's FOVs to
            exhaustion with ``yield from`` before ``source_index`` advances to the next
            channel, so a bare in-loop closure over the same late-binding name would read
            the correct channel too -- verified by swapping the factory for one and
            running the BaSiC-path suite unchanged. The factory is kept anyway: it is a
            correct, harmless habit that stops the read from becoming dependent on
            ``_planes``'s specific "one channel fully exhausted before the next starts"
            order, which is exactly the kind of implicit invariant a later change to
            interleave channels could break silently.
            """

            def read(y0, x0, h, w):
                return np.asarray(
                    arr[source_index, slice(y0, y0 + h), slice(x0, x0 + w)]
                )

            return read

        def _planes():
            # tifffile writes plane-by-plane from an iterator given an explicit shape, so
            # ONE padded tile is the largest thing resident at a time -- not a channel,
            # and not the stack. Same pattern as bin/tiled_stitch.py:stream_tiles.
            for source_index in profile_channels:
                yield from iter_padded_fovs(
                    _read_region(source_index), positions, tile_shape, dtype
                )

        # WRITTEN UNTILED -- generator-fed, but no `tile=` argument, a deliberate departure
        # from "every producer ahead of a lazy reader writes tiled" (see
        # tests/test_slide_io_seam.py's inventory, which records it). Not a gap: every
        # plane on this stack IS already a pseudo-FOV, at most preproc_tile_size px per
        # side, so there is no larger window a downstream reader would need to avoid
        # decoding. Its only reader is BASICPY's /opt/main.py, which loads the whole
        # (C, I, Y, X) stack into one ndarray regardless of tiling -- so a tile layout here
        # would add TIFF overhead for a windowed read that never happens.
        write_tiff(
            str(output_path),
            _planes(),
            shape=(len(profile_channels), n_tiles, tile_shape[0], tile_shape[1]),
            dtype=dtype,
            photometric="minisblack",
            metadata={
                "axes": "CZYX",
                "Channel": {"Name": [names[i] for i in profile_channels]},
            },
            ome=True,
            bigtiff=True,
        )
    finally:
        close()

    manifest = {
        "format_version": SIDECAR_FORMAT_VERSION,
        "source_image": Path(image_path).name,
        "source_dtype": str(np.dtype(dtype)),
        "image_shape": [int(height), int(width)],
        "fov_size": [int(fov_size[0]), int(fov_size[1])],
        "n_fovs_y": int(n_fovs_y),
        "n_fovs_x": int(n_fovs_x),
        "tile_shape": [int(tile_shape[0]), int(tile_shape[1])],
        "channel_names": names,
        # Source channel indices, IN THE ORDER THEY OCCUPY THE TILE STACK'S C AXIS --
        # this is the mapping from a fitted profile back to a slide channel.
        "profile_channels": [int(i) for i in profile_channels],
        # Source channel indices the profiles are actually applied to. Normally identical
        # to profile_channels; empty for an all-fiducial panel.
        "corrected_channels": [int(i) for i in correctable],
        "skipped_channels": [int(i) for i in skipped],
        "positions": [[int(v) for v in pos] for pos in positions],
    }
    Path(sidecar_path).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(f"[OK] Wrote {n_tiles} tiles to {output_path} and {sidecar_path}")
    return manifest


def parse_args(argv=None):
    """Parse the pseudo-FOV tiling CLI.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        ``image``, ``output``, ``sidecar``, ``channels``, ``fov_size``,
        ``skip_nuclear``, ``nuclear_markers``, ``log_level``.
    """
    parser = argparse.ArgumentParser(
        description="Write a slide's pseudo-FOV tiles as a multi-site CZYX OME-TIFF."
    )
    parser.add_argument("--image", required=True, help="Input OME-TIFF (C, Y, X).")
    parser.add_argument("--output", required=True, help="Tile-stack OME-TIFF to write.")
    parser.add_argument("--sidecar", required=True, help="Tile-position JSON to write.")
    parser.add_argument("--channels", nargs="+", required=True, help="Channel names.")
    parser.add_argument(
        "--fov_size", type=int, required=True, help="Square pseudo-FOV size in pixels."
    )
    parser.add_argument(
        "--skip_nuclear",
        action="store_true",
        help="Leave the nuclear/fiducial channel(s) uncorrected.",
    )
    parser.add_argument(
        "--nuclear-markers",
        dest="nuclear_markers",
        nargs="+",
        default=None,
        help="Marker names that identify the nuclear/fiducial channel.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point: write the slide's pseudo-FOV tiles and their position sidecar.

    Returns
    -------
    int
        0 on success.
    """
    args = parse_args(argv)
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    tile_for_basic(
        args.image,
        args.output,
        args.sidecar,
        channel_names=args.channels,
        fov_size=(args.fov_size, args.fov_size),
        skip_nuclear=args.skip_nuclear,
        nuclear_markers=args.nuclear_markers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

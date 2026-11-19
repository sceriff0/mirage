#!/usr/bin/env python3
"""Write a stitched slide as a multi-SITE OME-TIFF so nf-core's BASICPY will accept it.

nf-core's ``basicpy`` module (``basicpy-docker-mcmicro``'s ``/opt/main.py``) stacks
``M``, ``T`` and ``Z`` into its field-of-view axis and refuses a single-sited image. A
mirage slide is one stitched plane per channel, so this tiles it into pseudo-FOVs and
writes them on ``Z``. Axes are ``CZYX`` because the module fits one profile per channel.

Tiling comes from ``bin/utils/fov_tiling.py``: a non-overlapping grid with edge tiles
zero-padded to a common shape, because BaSiCPy's ``fit()`` takes one ``(N, Y, X)`` array.
Each tile is read lazily and written straight out, so peak memory is one padded tile.

The nuclear/fiducial skip is decided here, once, via ``utils.metadata.is_nuclear``, and
recorded in the JSON sidecar with the tile positions; ``bin/apply_basic_profiles.py``
reads both instead of re-deriving them. See ``docs/basic_illumination.md``.
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
    """(n_channels, height, width, dtype) for a 2-D or (C, H, W) OME-TIFF, without decoding it."""
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
        # BaSiC needs a population of fields; fail here, naming the knob, not in the container.
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

    # All-fiducial panel: tile every channel so the module has input, correct none.
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

    # Pure geometry: no pixels read.
    positions, tile_shape = fov_positions((height, width), n_fovs_y, n_fovs_x)

    arr, _dtype, close = open_lazy(image_path)
    try:

        def _read_region(source_index):
            """Lazy region reader bound to one source channel.

            A factory, not a loop closure, so it stays correct if channels are interleaved.
            """

            def read(y0, x0, h, w):
                return np.asarray(
                    arr[source_index, slice(y0, y0 + h), slice(x0, x0 + w)]
                )

            return read

        def _planes():
            # Fed to tifffile plane by plane: one padded tile resident at a time.
            for source_index in profile_channels:
                yield from iter_padded_fovs(
                    _read_region(source_index), positions, tile_shape, dtype
                )

        # Untiled on purpose: each plane is already one pseudo-FOV, and BASICPY loads the
        # whole stack anyway.
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
        # Source channel index of each C slot in the tile stack (profile -> channel).
        "profile_channels": [int(i) for i in profile_channels],
        # Channels the profiles are applied to; empty for an all-fiducial panel.
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

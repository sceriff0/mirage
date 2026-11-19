"""Single-channel extraction and tiled label expansion shared by the segmentation backends.

``segment.py`` (StarDist) and ``segment_cellsam.py`` (CellSAM) both read the nuclear
channel from a multichannel OME-TIFF and expand nuclei into cells
(``expand_labels_tiled``). Kept apart from both so it is testable without their ML stacks.

Reads through ``tiled_io.open_lazy`` (a tifffile zarr view), which decodes only the
requested channel's tiles. ``TiffFile.asarray(out="memmap")`` would instead decode the
whole compressed image into a full-size temporary file first.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import dask.array as da
import numpy as np
import tifffile
from numpy.typing import NDArray
from skimage import segmentation
from tiled_io import open_lazy
from validation import clip_negative_values

__all__ = ["expand_labels_tiled", "extract_dapi_channel"]


def extract_dapi_channel(
    multichannel_image_path: str,
    dapi_channel_index: int = 0,
    *,
    logger: Optional[logging.Logger] = None,
) -> Tuple[NDArray, Dict]:
    """Extract one channel from a multichannel OME-TIFF via a lazy zarr read.

    Parameters
    ----------
    multichannel_image_path : str
        Path to multichannel OME-TIFF file (e.g., from VALIS registration).
    dapi_channel_index : int, optional
        Index of the channel to extract. Default is 0 (first channel).
    logger : logging.Logger, optional
        Logger to log progress to. Defaults to this module's logger.

    Returns
    -------
    dapi_image : ndarray, shape (Y, X)
        The extracted channel, negative-clipped (see ``clip_negative_values``).
    metadata : dict
        Image metadata from OME-TIFF (``{"ome": <xml str>}`` if present).

    Raises
    ------
    ValueError
        If the image has unexpected dimensions (not 2-D or 3-D), or
        ``dapi_channel_index`` is out of range for a 3-D (C, Y, X) image.
    """
    log = logger or logging.getLogger(__name__)
    log.info(f"Loading multichannel image: {multichannel_image_path}")

    # Shape/dtype/OME metadata only -- no pixel data decoded here.
    with tifffile.TiffFile(multichannel_image_path) as tif:
        if tif.series:
            source = tif.series[0]
        else:
            # Non-OME or corrupt OME metadata: fall back to pages.
            log.warning("No OME series found, falling back to raw pages")
            if not tif.pages:
                raise ValueError(
                    f"TIFF file appears corrupted: {multichannel_image_path}"
                )
            source = tif.pages[0]

        image_shape = source.shape
        image_dtype = source.dtype

        metadata: Dict = {}
        if hasattr(tif, "ome_metadata") and tif.ome_metadata:
            metadata["ome"] = tif.ome_metadata
            log.info("  ✓ OME metadata found")

        log.info(f"  Image shape: {image_shape}")
        log.info(f"  Image dtype: {image_dtype}")

    if len(image_shape) not in (2, 3):
        raise ValueError(
            f"Unexpected image dimensions: {image_shape}. "
            f"Expected 2D (Y, X) or 3D (C, Y, X)"
        )

    # Lazy read: only the requested channel's tiles are decoded.
    lazy_view, _dtype, close = open_lazy(multichannel_image_path)
    try:
        if len(image_shape) == 2:
            # Single channel image, assume it's DAPI
            log.info("  - Single channel image (assuming DAPI)")
            dapi_image = np.array(lazy_view[0, :, :], copy=True)
        else:
            n_channels = image_shape[0]
            log.info(f"  - Multichannel image with {n_channels} channels")

            if dapi_channel_index >= n_channels:
                raise ValueError(
                    f"DAPI channel index {dapi_channel_index} out of range "
                    f"for image with {n_channels} channels"
                )

            log.info(
                f"  - Extracting DAPI channel (index {dapi_channel_index}) "
                f"- memory efficient (only this channel's tiles are decoded)"
            )
            dapi_image = np.array(lazy_view[dapi_channel_index, :, :], copy=True)
            log.info(f"  - Extracted DAPI channel (index {dapi_channel_index})")
    finally:
        close()

    dapi_image = clip_negative_values(dapi_image, log, stage_name="extract_dapi")

    log.info(f"  DAPI channel shape: {dapi_image.shape}")
    log.info(f"  DAPI dtype: {dapi_image.dtype}")
    log.info(f"  DAPI value range: [{dapi_image.min()}, {dapi_image.max()}]")

    return dapi_image, metadata


def expand_labels_tiled(
    label_image: NDArray, distance: int = 1, tile_size: int = 1024
) -> NDArray:
    """Expand labels by `distance` pixels, tile by tile, via ``dask.array.map_overlap``.

    Peak memory scales with the tile; the result equals a full-image
    ``skimage.segmentation.expand_labels``. That requires ``depth >= distance``, or tile
    boundaries grow seams.

    A tile too small for the overlap (``tile_size <= 2 * overlap``) falls back to the
    full-image call.
    """
    overlap = distance + 1

    if tile_size <= 2 * overlap:
        return segmentation.expand_labels(label_image, distance=distance)

    dask_labels = da.from_array(label_image, chunks=tile_size)

    def _expand_tile(tile: NDArray) -> NDArray:
        return segmentation.expand_labels(tile, distance=distance).astype(np.uint32)

    expanded = da.map_overlap(
        _expand_tile, dask_labels, depth=overlap, boundary="none", dtype=np.uint32
    )

    return expanded.compute()

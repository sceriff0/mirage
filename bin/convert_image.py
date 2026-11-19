#!/usr/bin/env python3
"""Convert microscopy images to standardized OME-TIFF format.

Supports:
- ND2 (Nikon) via bioio-nd2
- CZI (Zeiss) via bioio-czi
- LIF (Leica) via bioio-lif
- TIFF/OME-TIFF via bioio-tifffile/bioio-ome-tiff
- NDPI/NDPIS (Hamamatsu) via tifffile
- HDF5 (.h5, .hdf5) via h5py
- SVS/QPTIFF/VSI/SCN/MRXS/BIF/IMS (Aperio/Vectra/Olympus/Leica/3DHistech/Ventana/Imaris)
  via bioio-bioformats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from jvm_cache import point_jvm_cache_off_readonly_home
from logger import configure_logging, get_logger
from metadata import DEFAULT_NUCLEAR_MARKERS, pick_nuclear_index
from ome_io import (
    CONVERT_TIFF_TILE,
    detect_reader,
    parse_ndpis,
    require_reader,
    write_ome_tiff,
)
from pixel_size import resolve_pixel_size, warn_on_pixel_size_mismatch

logger = get_logger(__name__)

__all__ = ["main"]


def read_image_bioio(file_path: Path) -> Tuple[Any, dict]:
    """Open a slide with BioIO (ND2, CZI, LIF, TIFF) WITHOUT decoding it.

    Returns ``img.dask_data``, not ``img.data``: a lazy dask array with the same shape
    and dimension order, whose planes are decoded only when something asks for them.
    ``convert_to_ome_tiff``'s ``write_ome_tiff`` is what finally does, one plane at a
    time -- see that function for why both halves had to change together.

    The returned object is therefore NOT an ``np.ndarray``. Everything applied to it
    downstream (``np.squeeze`` here; ``np.transpose``/``np.take``/basic indexing in
    ``convert_to_ome_tiff``) dispatches through dask's ``__array_function__`` and stays
    lazy; ``tests/test_convert_lazy_read.py`` pins that the handle is still an
    unmaterialised graph when it leaves this function.
    """
    from bioio import BioImage

    logger.info(f"Reading with BioIO: {file_path.name}")

    img = BioImage(file_path)

    logger.info(f"Dimensions: {img.dims}")
    logger.info(f"Shape: {img.shape}")
    logger.info(f"Dimension order: {img.dims.order}")

    # Determine channel count - handle 'S' (Samples) as channels if 'C' is not used
    dim_order = img.dims.order
    num_channels = img.dims.C

    # Check if 'S' dimension should be treated as channels
    # This happens when image has 'S' but 'C' is 1 or not meaningful
    image_data = None  # Will be set below (always a lazy handle, never a decoded array)
    if "S" in dim_order and (num_channels == 1 or "C" not in dim_order):
        s_count = img.dims.S
        if s_count > 1:
            logger.info(
                f"Detected 'S' dimension ({s_count}) used as channels instead of 'C' ({num_channels})"
            )
            num_channels = s_count

            # If there's a singleton C dimension, squeeze it out to avoid
            # having two 'C' characters in the dimension order
            if "C" in dim_order and img.dims.C == 1:
                c_pos = dim_order.index("C")
                # np.squeeze on a dask array stays lazy (dispatched via __array_function__).
                image_data = np.squeeze(img.dask_data, axis=c_pos)
                dim_order = dim_order[:c_pos] + dim_order[c_pos + 1 :]
                logger.info(f"Squeezed singleton C dimension at position {c_pos}")
            else:
                image_data = img.dask_data

            # Remap dimension order for downstream processing (treat S as C)
            dim_order = dim_order.replace("S", "C")
            logger.info(f"Remapped dimension order: {dim_order}")

    # Take the lazy handle if the S-as-C branch above did not already
    if image_data is None:
        image_data = img.dask_data

    ps = img.physical_pixel_sizes
    # None means "the file did not say". The fallback lives in convert_to_ome_tiff so
    # warn_on_pixel_size_mismatch can tell a missing scale from one equal to the default.
    pixel_size_x = ps.X
    pixel_size_y = ps.Y
    pixel_size_z = ps.Z

    logger.info(
        f"Pixel sizes - X: {pixel_size_x}, Y: {pixel_size_y}, Z: {pixel_size_z}"
    )
    logger.info(f"Channel names from file: {img.channel_names}")

    metadata = {
        "num_channels": num_channels,
        "physical_pixel_size_x": pixel_size_x,
        "physical_pixel_size_y": pixel_size_y,
        "physical_pixel_size_z": pixel_size_z,
        "channel_names_from_file": img.channel_names,
        "original_dims": dim_order,
    }

    return image_data, metadata


def read_single_ndpi(
    file_path: Path,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """Read a single NDPI file using tifffile, return image and pixel sizes."""
    with tifffile.TiffFile(file_path) as tif:
        series = tif.series[0]
        image_data = series.asarray()

        logger.info(f"  {file_path.name}: shape={image_data.shape}, axes={series.axes}")

        # Extract pixel size from TIFF tags (None until a tag actually provides one)
        pixel_size_x = None
        pixel_size_y = None

        page = tif.pages[0]
        if page.tags.get("XResolution") and page.tags.get("YResolution"):
            x_res = page.tags["XResolution"].value
            y_res = page.tags["YResolution"].value
            res_unit = page.tags.get("ResolutionUnit")

            if x_res and y_res:
                x_res_val = x_res[0] / x_res[1] if isinstance(x_res, tuple) else x_res
                y_res_val = y_res[0] / y_res[1] if isinstance(y_res, tuple) else y_res

                if res_unit and res_unit.value == 3:  # centimeters
                    pixel_size_x = 10000.0 / x_res_val
                    pixel_size_y = 10000.0 / y_res_val
                elif res_unit and res_unit.value == 2:  # inches
                    pixel_size_x = 25400.0 / x_res_val
                    pixel_size_y = 25400.0 / y_res_val

        # Convert to grayscale if RGB (NDPI fluorescence channels are often stored as RGB)
        if image_data.ndim == 3 and image_data.shape[-1] == 3:
            # Check if R, G, B channels are identical
            if np.array_equal(
                image_data[..., 0], image_data[..., 1]
            ) and np.array_equal(image_data[..., 1], image_data[..., 2]):
                logger.info("    RGB channels identical, taking first channel")
            else:
                logger.warning("    RGB channels differ! Taking first channel anyway")
            image_data = image_data[..., 0]
        elif image_data.ndim == 3 and image_data.shape[-1] == 4:
            # RGBA - check RGB channels
            if np.array_equal(
                image_data[..., 0], image_data[..., 1]
            ) and np.array_equal(image_data[..., 1], image_data[..., 2]):
                logger.info("    RGBA channels identical, taking first channel")
            else:
                logger.warning("    RGBA channels differ! Taking first channel anyway")
            image_data = image_data[..., 0]

        return image_data, pixel_size_x, pixel_size_y


def read_image_tifffile(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read NDPI/NDPIS image using tifffile."""
    logger.info(f"Reading with tifffile: {file_path.name}")

    suffix = file_path.suffix.lower()

    if suffix == ".ndpis":
        # NDPIS is a manifest file pointing to multiple NDPI files
        ndpi_files = parse_ndpis(file_path)

        if not ndpi_files:
            raise ValueError(f"No NDPI files found in NDPIS manifest: {file_path}")

        # Read all NDPI files and stack them
        channel_images = []
        pixel_size_x = None
        pixel_size_y = None

        for ndpi_path in ndpi_files:
            if not ndpi_path.exists():
                raise FileNotFoundError(f"NDPI file not found: {ndpi_path}")
            img, px_x, px_y = read_single_ndpi(ndpi_path)
            channel_images.append(img)
            pixel_size_x = px_x
            pixel_size_y = px_y

        # Stack channels: each image becomes a channel (CYX)
        image_data = np.stack(channel_images, axis=0)

    else:
        # Single NDPI file
        image_data, pixel_size_x, pixel_size_y = read_single_ndpi(file_path)

        # Add channel dimension
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]

    num_channels = image_data.shape[0]
    axes = "CYX"

    logger.info(f"Final shape: {image_data.shape}, axes: {axes}")
    logger.info(f"Pixel sizes - X: {pixel_size_x}, Y: {pixel_size_y}")

    metadata = {
        "num_channels": num_channels,
        "physical_pixel_size_x": pixel_size_x,
        "physical_pixel_size_y": pixel_size_y,
        "physical_pixel_size_z": None,
        "channel_names_from_file": None,  # Channel names must come from args
        "original_dims": axes,
    }

    return image_data, metadata


def _find_first_image_dataset(group):
    """Recursively find the first image-like dataset in an HDF5 group."""
    import h5py

    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            if item.ndim >= 2 and np.issubdtype(item.dtype, np.number):
                return item
        elif isinstance(item, h5py.Group):
            result = _find_first_image_dataset(item)
            if result is not None:
                return result
    return None


def _extract_h5_pixel_sizes(
    h5obj,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract pixel size metadata from HDF5 attributes, or Nones if there are none."""
    px_x = None
    px_y = None
    px_z = None

    for attr_name in ("element_size_um", "pixel_size", "resolution", "pixelSize"):
        if attr_name in h5obj.attrs:
            val = h5obj.attrs[attr_name]
            if hasattr(val, "__len__"):
                if len(val) >= 2:
                    px_y, px_x = float(val[-2]), float(val[-1])
                if len(val) >= 3:
                    px_z = float(val[-3])
            else:
                px_x = px_y = float(val)
            break

    return px_x, px_y, px_z


def read_image_h5(file_path: Path) -> Tuple[np.ndarray, dict]:
    """Read HDF5 microscopy image (single stacked dataset, typically CYX)."""
    import h5py

    logger.info(f"Reading with h5py: {file_path.name}")

    with h5py.File(file_path, "r") as f:
        ds = _find_first_image_dataset(f)
        if ds is None:
            raise ValueError(f"No image dataset found in HDF5 file: {file_path}")

        logger.info(f"Found dataset '{ds.name}': shape={ds.shape}, dtype={ds.dtype}")
        image_data = ds[()]

        # Extract pixel sizes from dataset attrs, then root attrs as fallback
        px_x, px_y, px_z = _extract_h5_pixel_sizes(ds)
        if px_x is None:
            px_x, px_y, px_z = _extract_h5_pixel_sizes(f)

        # Extract channel names from attributes if available
        channel_names = None
        for obj in (ds, f):
            if "channel_names" in obj.attrs:
                cn = obj.attrs["channel_names"]
                if hasattr(cn, "__len__") and len(cn) > 0:
                    channel_names = [str(n) for n in cn]
                    break

        # Determine axes from dimensionality
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]
            axes = "CYX"
        elif image_data.ndim == 3:
            # Heuristic: if last dim is small relative to first, it's YXC
            if (
                image_data.shape[-1] <= 10
                and image_data.shape[0] > image_data.shape[-1]
            ):
                image_data = np.moveaxis(image_data, -1, 0)
            axes = "CYX"
        elif image_data.ndim == 4:
            axes = "ZCYX"
        else:
            axes = "TCZYX"

        num_channels = image_data.shape[axes.index("C")]

    logger.info(f"Final shape: {image_data.shape}, axes: {axes}")
    logger.info(f"Pixel sizes - X: {px_x}, Y: {px_y}, Z: {px_z}")
    if channel_names:
        logger.info(f"Channel names from file: {channel_names}")

    metadata = {
        "num_channels": num_channels,
        "physical_pixel_size_x": px_x,
        "physical_pixel_size_y": px_y,
        "physical_pixel_size_z": px_z,
        "channel_names_from_file": channel_names,
        "original_dims": axes,
    }

    return image_data, metadata


def read_image(file_path: Path) -> Tuple[Any, dict]:
    """Read `file_path` with whichever reader `ome_io.detect_reader` names.

    The BioIO branch returns a lazy dask array; the tifffile and HDF5 branches return a
    decoded ``np.ndarray``. ``ome_io.write_ome_tiff`` accepts both. An unclaimed extension
    raises ``UnsupportedFormatError`` from ``detect_reader``.

    ``bioio`` and ``bioio-bioformats`` share this branch. ``require_reader`` runs first so a
    missing plugin is named. For ``bioio-bioformats``, ``point_jvm_cache_off_readonly_home()``
    runs before ``BioImage`` is constructed, because the Bio-Formats Maven resolve otherwise
    writes under the cluster's read-only ``$HOME`` (see ``bin/utils/jvm_cache.py``).
    """
    reader = detect_reader(file_path)
    logger.info(f"Detected reader: {reader}")
    require_reader(reader)

    if reader == "tifffile":
        return read_image_tifffile(file_path)
    if reader == "hdf5":
        return read_image_h5(file_path)
    if reader == "bioio-bioformats":
        point_jvm_cache_off_readonly_home()
    return read_image_bioio(file_path)


def convert_to_ome_tiff(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    channel_names: Optional[List[str]] = None,
    pixel_size_um: str | float | None = None,
    nuclear_markers: Optional[List[str]] = None,
) -> Tuple[Path, List[str]]:
    """Convert image to OME-TIFF with the nuclear/fiducial channel in channel 0.

    The nuclear channel is resolved by marker name from the declared/metadata
    channel names (never the filename), using ``nuclear_markers`` as an ordered
    preference list (default ``('DAPI', 'CELLTOX')``).
    """

    # Read image first to get metadata
    image_data, metadata = read_image(input_path)
    original_dims = metadata.get("original_dims", "TCZYX")

    # Use channel names from file if not specified
    if channel_names is None:
        channel_names = metadata.get("channel_names_from_file")
        if channel_names is None:
            raise ValueError(
                "No channel names provided and none found in file metadata"
            )
        logger.info(f"Using channel names from file: {channel_names}")

    # Validate channel count
    num_channels_in_image = metadata["num_channels"]
    if num_channels_in_image != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: image has {num_channels_in_image}, "
            f"specified {len(channel_names)}: {channel_names}"
        )

    # Resolve the nuclear/fiducial channel by marker name (ordered preference).
    # Identity comes from the declared/metadata channel names, never the filename.
    markers = (
        list(nuclear_markers) if nuclear_markers else list(DEFAULT_NUCLEAR_MARKERS)
    )
    nuclear_index = pick_nuclear_index(channel_names, markers)

    if nuclear_index is None:
        if len(channel_names) == 1:
            # Genuinely single-channel image: treat the sole channel as nuclear.
            logger.warning(
                f"No nuclear marker {markers} in single-channel image; "
                f"assuming '{channel_names[0]}' is the nuclear channel"
            )
            nuclear_index = 0
        else:
            raise ValueError(
                f"None of nuclear_markers {markers} found in channels {channel_names}"
            )

    nuclear_name = channel_names[nuclear_index]

    # Reorder: nuclear channel first. Downstream (segmentation, tiled registration
    # fiducial) trusts the invariant "channel 0 = nuclear marker".
    output_channels = channel_names.copy()
    if nuclear_index != 0:
        logger.info(
            f"Moving nuclear channel '{nuclear_name}' from position {nuclear_index} to 0"
        )
        nuclear_ch = output_channels.pop(nuclear_index)
        output_channels.insert(0, nuclear_ch)
    else:
        logger.info(f"Nuclear channel '{nuclear_name}' already in position 0")

    channels_str = "_".join(output_channels)
    output_filename = output_dir / f"{patient_id}_{channels_str}.ome.tif"

    logger.info(f"Converting: {input_path.name}")
    logger.info(f"Input channels: {channel_names}")
    logger.info(f"Output channels: {output_channels}")

    # Single fallback point. params.pixel_size drives every downstream µm conversion,
    # so a file claiming a different scale changes nothing, but the operator is warned.
    det_x = metadata.get("physical_pixel_size_x")
    det_y = metadata.get("physical_pixel_size_y")
    # 'auto' uses the scale read off the vendor file, the only place it is still
    # available (read_ome_pixel_size cannot open .czi/.ndpi/.svs). Absent metadata raises.
    configured = resolve_pixel_size(
        pixel_size_um, detected=(det_x, det_y), source=input_path.name, logger=logger
    )
    warn_on_pixel_size_mismatch(
        (det_x, det_y), configured, source=input_path.name, logger=logger
    )
    px_x = det_x if det_x is not None else configured
    px_y = det_y if det_y is not None else configured
    px_z = metadata.get("physical_pixel_size_z")

    # Normalize dimensions to standard OME-TIFF order (C before spatial dims)
    # Handle case where C is at the end (e.g., TZYXC from S dimension remapping)
    if "C" in original_dims and original_dims.endswith("C"):
        # Move C from end to before spatial dimensions
        c_pos = original_dims.index("C")

        # Transpose: move C axis to position just before Y
        axes_list = list(range(image_data.ndim))
        dims_list = list(original_dims)
        axes_list.remove(c_pos)
        dims_list.remove("C")
        new_y_pos = dims_list.index("Y")
        axes_list.insert(new_y_pos, c_pos)
        dims_list.insert(new_y_pos, "C")
        image_data = np.transpose(image_data, axes_list)
        original_dims = "".join(dims_list)
        logger.info(
            f"Transposed C to standard position: {original_dims}, shape: {image_data.shape}"
        )

    if original_dims == "TCZYX":
        c_axis = 1
        if image_data.shape[0] == 1:
            image_data = image_data[0]
            original_dims = "CZYX"
            c_axis = 0
    elif original_dims == "TZCYX":
        c_axis = 2
        # Squeeze T if singleton
        if image_data.shape[0] == 1:
            image_data = image_data[0]
            original_dims = "ZCYX"
            c_axis = 1
    elif original_dims == "ZCYX":
        c_axis = 1
    elif original_dims in ("CZYX", "CYX"):
        c_axis = 0
    else:
        # Try to find C axis position
        c_axis = original_dims.index("C") if "C" in original_dims else 0

    # Squeeze Z if singleton
    if original_dims == "CZYX" and image_data.shape[1] == 1:
        image_data = image_data[:, 0, :, :]
        original_dims = "CYX"
    elif original_dims == "ZCYX" and image_data.shape[0] == 1:
        image_data = image_data[0]
        original_dims = "CYX"
        c_axis = 0

    # Rearrange channels
    if channel_names != output_channels:
        # Check for duplicate channel names (index() would silently return first match)
        if len(set(channel_names)) != len(channel_names):
            logger.warning(f"Duplicate channel names detected: {channel_names}")
            # Build index map using enumerate to handle duplicates correctly
            name_to_indices = {}
            for i, ch in enumerate(channel_names):
                name_to_indices.setdefault(ch, []).append(i)
            indices = []
            used = {ch: 0 for ch in channel_names}
            for ch in output_channels:
                idx = name_to_indices[ch][used[ch]]
                used[ch] += 1
                indices.append(idx)
        else:
            indices = [channel_names.index(ch) for ch in output_channels]
        image_data = np.take(image_data, indices, axis=c_axis)
        logger.info(f"Rearranged channels: {channel_names} -> {output_channels}")

    logger.info(f"Final shape: {image_data.shape}, axes: {original_dims}")
    logger.info(f"Pixel size: X={px_x}, Y={px_y}, Z={px_z}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # ome_io owns the metadata dict and the write. bigtiff=True unconditionally, as
    # tests/test_convert_streaming_write.py's reference writer expects.
    logger.info(f"Writing: {output_filename.name}")

    write_ome_tiff(
        output_filename,
        image_data,
        channels=output_channels,
        pixel_size_um=(px_x, px_y),
        tile=CONVERT_TIFF_TILE,
        bigtiff=True,
        axes=original_dims,
        pixel_size_z_um=px_z,
    )

    logger.info(f"Saved: {output_filename.name}")

    # Verify
    with tifffile.TiffFile(output_filename) as tif:
        if tif.ome_metadata:
            logger.info("OME-XML metadata present")
        else:
            logger.warning("No OME metadata found in saved file")

    return output_filename, output_channels


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Convert microscopy images to OME-TIFF"
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--patient_id", type=str, required=True)
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel names (optional, reads metadata when omitted)",
    )
    # str, not float: may be the literal 'auto'. No default: see nextflow.config's
    # pixel_size.
    parser.add_argument("--pixel_size", type=str, default=None)
    parser.add_argument(
        "--nuclear-markers",
        nargs="+",
        default=list(DEFAULT_NUCLEAR_MARKERS),
        help="Ordered preference of nuclear/fiducial marker names; the first present "
        "(resolved from channel metadata) is moved to channel 0.",
    )
    return parser.parse_args()


def main() -> int:
    """Run conversion CLI."""
    configure_logging()
    args = parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    channel_names = None
    if args.channels:
        channel_names = [ch.strip() for ch in args.channels.split(",")]

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info("=" * 70)
    logger.info("Image Converter")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    if channel_names:
        logger.info(f"Channels (override): {channel_names}")
    else:
        logger.info("Channels: read from image metadata")
    logger.info("=" * 70)

    try:
        output_path, output_channels = convert_to_ome_tiff(
            input_path,
            output_dir,
            args.patient_id,
            channel_names,
            args.pixel_size,
            args.nuclear_markers,
        )
    except Exception as exc:
        logger.error(f"Conversion failed: {exc}")
        return 1

    logger.info("=" * 70)
    logger.info(f"Output: {output_path}")
    logger.info(f"Channel order: {output_channels}")
    logger.info("=" * 70)

    # Write channels file for Nextflow metadata propagation
    channels_file = output_dir / f"{args.patient_id}_channels.txt"
    channels_file.write_text(",".join(output_channels))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

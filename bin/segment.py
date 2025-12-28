#!/usr/bin/env python3
"""Cell segmentation using StarDist on DAPI channel from multichannel images.

Simplified segmentation pipeline:
- Loads multichannel OME-TIFF (from registration output)
- Extracts DAPI channel (first channel)
- Normalizes using CSBDeep normalize
- Segments using StarDist whole-image processing
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

# Add parent directory to path to import lib modules
import sys
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from logger import get_logger, configure_logging
from typing import Tuple

import numpy as np
import tifffile
from csbdeep.utils import normalize
from skimage import segmentation
from stardist.models import StarDist2D
from numpy.typing import NDArray

from image_utils import ensure_dir

logger = get_logger(__name__)

__all__ = [
    "extract_dapi_channel",
    "normalize_dapi",
    "load_stardist_model",
    "segment_nuclei",
    "run_segmentation",
]


def extract_dapi_channel(
    multichannel_image_path: str,
    dapi_channel_index: int = 0
) -> Tuple[NDArray, dict]:
    """
    Extract DAPI channel from multichannel OME-TIFF image using memory-mapped I/O.

    Parameters
    ----------
    multichannel_image_path : str
        Path to multichannel OME-TIFF file (e.g., from VALIS registration).
    dapi_channel_index : int, optional
        Index of DAPI channel. Default is 0 (first channel).

    Returns
    -------
    dapi_image : ndarray, shape (Y, X)
        DAPI channel image.
    metadata : dict
        Image metadata from OME-TIFF.

    Raises
    ------
    ValueError
        If image has wrong dimensions or DAPI channel doesn't exist.
    """
    logger.info(f"Loading multichannel image: {multichannel_image_path}")

    # Use memory-mapped I/O to avoid loading entire multichannel image
    with tifffile.TiffFile(multichannel_image_path) as tif:
        # Get shape and metadata WITHOUT loading data
        if len(tif.series[0].shape) == 2:
            # Single channel
            image_shape = tif.series[0].shape
            n_channels = 1
        else:
            # Multichannel
            image_shape = tif.series[0].shape
            n_channels = image_shape[0] if len(image_shape) == 3 else 1

        image_dtype = tif.series[0].dtype

        # Extract OME metadata if available
        metadata = {}
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            metadata['ome'] = tif.ome_metadata
            logger.info(f"  ✓ OME metadata found")

        logger.info(f"  Image shape: {image_shape}")
        logger.info(f"  Image dtype: {image_dtype}")

        # Memory-map the file (doesn't load into RAM)
        image_memmap = tif.asarray(out='memmap')

        # Handle different image formats
        if image_memmap.ndim == 2:
            # Single channel image, assume it's DAPI
            logger.info("  - Single channel image (assuming DAPI)")
            # Load single channel into RAM
            dapi_image = np.array(image_memmap, copy=True)
        elif image_memmap.ndim == 3:
            # Multichannel image (C, Y, X) format
            logger.info(f"  - Multichannel image with {n_channels} channels")

            if dapi_channel_index >= n_channels:
                raise ValueError(
                    f"DAPI channel index {dapi_channel_index} out of range "
                    f"for image with {n_channels} channels"
                )

            # Extract ONLY the DAPI channel into RAM (not all channels)
            logger.info(f"  - Extracting DAPI channel (index {dapi_channel_index}) - memory efficient")
            dapi_image = np.array(image_memmap[dapi_channel_index, :, :], copy=True)
            logger.info(f"  - Extracted DAPI channel (index {dapi_channel_index})")
        else:
            raise ValueError(
                f"Unexpected image dimensions: {image_shape}. "
                f"Expected 2D (Y, X) or 3D (C, Y, X)"
            )

    logger.info(f"  DAPI channel shape: {dapi_image.shape}")
    logger.info(f"  DAPI dtype: {dapi_image.dtype}")
    logger.info(f"  DAPI value range: [{dapi_image.min()}, {dapi_image.max()}]")

    return dapi_image, metadata


def normalize_dapi(
    dapi_image: NDArray,
    pmin: float = 1.0,
    pmax: float = 99.8
) -> NDArray:
    """
    Normalize DAPI image using CSBDeep percentile normalization.

    Parameters
    ----------
    dapi_image : ndarray, shape (Y, X)
        DAPI channel image.
    pmin : float, optional
        Lower percentile for normalization. Default is 1.0.
    pmax : float, optional
        Upper percentile for normalization. Default is 99.8.

    Returns
    -------
    normalized : ndarray, shape (Y, X)
        Normalized DAPI image with values in [0, 1].

    Notes
    -----
    Uses CSBDeep's normalize function which clips to percentiles
    and scales to [0, 1] range. Converts to float32 to save memory.
    """
    logger.info("Normalizing DAPI channel...")
    logger.info(f"  Percentiles: [{pmin}, {pmax}]")

    # Convert to float32 if needed (saves memory vs float64)
    if dapi_image.dtype != np.float32:
        logger.info(f"  Converting from {dapi_image.dtype} to float32 to save memory")
        dapi_image = dapi_image.astype(np.float32)

    # CSBDeep normalize clips to percentiles and scales to [0, 1]
    normalized = normalize(dapi_image, pmin, pmax, axis=(0, 1))

    # Ensure float32 output
    if normalized.dtype != np.float32:
        normalized = normalized.astype(np.float32)

    logger.info(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")

    return normalized


def load_stardist_model(
    model_dir: str,
    model_name: str,
    use_gpu: bool = True
) -> StarDist2D:
    """
    Load pre-trained StarDist model.

    Parameters
    ----------
    model_dir : str
        Directory containing the model.
    model_name : str
        Name of the model (e.g., '2D_versatile_fluo').
    use_gpu : bool, optional
        Use GPU acceleration if available. Default is True.

    Returns
    -------
    model : StarDist2D
        Loaded StarDist model.
    """
    logger.info(f"Loading StarDist model...")
    logger.info(f"  Model name: {model_name}")
    logger.info(f"  Model directory: {model_dir}")
    logger.info(f"  GPU enabled: {use_gpu}")

    # Verify the model path exists
    model_path = Path(model_dir) / model_name
    config_file = model_path / "config.json"

    logger.info(f"  Expected model path: {model_path}")
    logger.info(f"  Expected config file: {config_file}")
    logger.info(f"  Model path exists: {model_path.exists()}")
    logger.info(f"  Config file exists: {config_file.exists()}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    model = StarDist2D(None, name=model_name, basedir=model_dir)

    if hasattr(model, 'config'):
        model.config.use_gpu = use_gpu

    logger.info("  ✓ Model loaded successfully")

    return model


def segment_nuclei(
    normalized_dapi: NDArray,
    model: StarDist2D,
    n_tiles: Tuple[int, int] = (24, 24),
    expand_distance: int = 10
) -> Tuple[NDArray, NDArray]:
    """
    Segment nuclei and create whole-cell masks.

    Parameters
    ----------
    normalized_dapi : ndarray, shape (Y, X)
        Normalized DAPI image.
    model : StarDist2D
        Loaded StarDist model.
    n_tiles : tuple of int, optional
        Number of tiles for tiled processing. Default is (16, 16).
        Larger images benefit from tiling.
    expand_distance : int, optional
        Distance (pixels) to expand nuclei labels to create whole-cell masks.
        Default is 10.

    Returns
    -------
    nuclei_mask : ndarray, shape (Y, X), dtype uint32
        Nuclei segmentation mask with unique cell labels.
    cell_mask : ndarray, shape (Y, X), dtype uint32
        Whole-cell segmentation mask (expanded from nuclei).

    Notes
    -----
    - Uses StarDist predict_instances with tiling for memory efficiency
    - Expands nuclei labels using skimage.segmentation.expand_labels
    - Label 0 is background, labels ≥1 are cells
    """
    logger.info(f"Segmenting nuclei on whole image...")
    logger.info(f"  Image shape: {normalized_dapi.shape}")
    logger.info(f"  Tiling: {n_tiles}")
    logger.info(f"  Cell expansion distance: {expand_distance}px")

    start_time = time.time()

    # Predict nuclei instances using StarDist
    nuclei_labels, _ = model.predict_instances(
        normalized_dapi,
        n_tiles=n_tiles,
        show_tile_progress=False,
        verbose=False
    )

    logger.info(f"  ✓ Nuclei detection complete")

    # Expand nuclei labels to create whole-cell masks
    cell_labels = segmentation.expand_labels(
        nuclei_labels,
        distance=expand_distance
    )

    elapsed = time.time() - start_time

    n_nuclei = len(np.unique(nuclei_labels)) - 1  # Subtract background (0)
    n_cells = len(np.unique(cell_labels)) - 1

    logger.info(f"  Nuclei detected: {n_nuclei}")
    logger.info(f"  Cells (after expansion): {n_cells}")
    logger.info(f"  Segmentation time: {elapsed:.2f}s")

    # Ensure uint32 for large label counts
    return nuclei_labels.astype(np.uint32), cell_labels.astype(np.uint32)


def run_segmentation(
    image_path: str,
    output_dir: str,
    model_dir: str,
    model_name: str,
    dapi_channel_index: int = 0,
    use_gpu: bool = True,
    n_tiles: Tuple[int, int] = (24, 24),
    expand_distance: int = 10,
    pmin: float = 1.0,
    pmax: float = 99.8
) -> Tuple[str, str]:
    """
    Run complete segmentation pipeline on multichannel image.

    Parameters
    ----------
    image_path : str
        Path to multichannel OME-TIFF image (e.g., from VALIS registration).
    output_dir : str
        Output directory for segmentation masks.
    model_dir : str
        StarDist model directory.
    model_name : str
        StarDist model name.
    dapi_channel_index : int, optional
        Index of DAPI channel in multichannel image. Default is 0.
    use_gpu : bool, optional
        Use GPU acceleration. Default is True.
    n_tiles : tuple of int, optional
        Number of tiles for processing. Default is (16, 16).
    expand_distance : int, optional
        Distance to expand nuclei for whole-cell masks. Default is 10.
    pmin : float, optional
        Lower percentile for normalization. Default is 1.0.
    pmax : float, optional
        Upper percentile for normalization. Default is 99.8.

    Returns
    -------
    nuclei_mask_path : str
        Path to saved nuclei segmentation mask.
    cell_mask_path : str
        Path to saved whole-cell segmentation mask.
    """
    logger.info("=" * 80)
    logger.info("CELL SEGMENTATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input image: {image_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Create output directory
    ensure_dir(output_dir)

    # 1. Extract DAPI channel from multichannel image
    dapi_image, metadata = extract_dapi_channel(image_path, dapi_channel_index)

    # 2. Normalize using CSBDeep
    normalized_dapi = normalize_dapi(dapi_image, pmin=pmin, pmax=pmax)

    # 3. Load StarDist model
    model = load_stardist_model(model_dir, model_name, use_gpu=use_gpu)

    # 4. Segment nuclei and create cell masks
    nuclei_mask, cell_mask = segment_nuclei(
        normalized_dapi,
        model,
        n_tiles=n_tiles,
        expand_distance=expand_distance
    )

    # 5. Save outputs
    basename = Path(image_path).stem
    nuclei_mask_path = Path(output_dir) / f"{basename}_nuclei_mask.tif"
    cell_mask_path = Path(output_dir) / f"{basename}_cell_mask.tif"

    logger.info("")
    logger.info("Saving segmentation masks...")
    logger.info(f"  Nuclei mask: {nuclei_mask_path.name}")
    tifffile.imwrite(nuclei_mask_path, nuclei_mask, compression='zlib')

    logger.info(f"  Cell mask: {cell_mask_path.name}")
    tifffile.imwrite(cell_mask_path, cell_mask, compression='zlib')

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ SEGMENTATION COMPLETE")
    logger.info("=" * 80)

    return str(nuclei_mask_path), str(cell_mask_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell segmentation using StarDist on multichannel images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to multichannel OME-TIFF image (e.g., registered image from VALIS)'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory containing StarDist model'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='StarDist model name (e.g., "2D_versatile_fluo")'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./segmentation_output',
        help='Output directory for segmentation masks'
    )

    parser.add_argument(
        '--dapi-channel',
        type=int,
        default=0,
        help='Index of DAPI channel in multichannel image'
    )

    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=False,
        help='Use GPU acceleration (if available)'
    )

    parser.add_argument(
        '--n-tiles',
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=('Y', 'X'),
        help='Number of tiles for processing (Y X)'
    )

    parser.add_argument(
        '--expand-distance',
        type=int,
        default=10,
        help='Distance (pixels) to expand nuclei labels for whole-cell masks'
    )

    parser.add_argument(
        '--pmin',
        type=float,
        default=1.0,
        help='Lower percentile for normalization'
    )

    parser.add_argument(
        '--pmax',
        type=float,
        default=99.8,
        help='Upper percentile for normalization'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)

    args = parse_args()

    run_segmentation(
        image_path=args.image,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        model_name=args.model_name,
        dapi_channel_index=args.dapi_channel,
        use_gpu=args.use_gpu,
        n_tiles=tuple(args.n_tiles),
        expand_distance=args.expand_distance,
        pmin=args.pmin,
        pmax=args.pmax
    )

    return 0


if __name__ == '__main__':
    exit(main())

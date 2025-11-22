#!/usr/bin/env python3
"""Cell type classification using deepcell-types.

This script performs cell type classification on multiplexed imaging data using
the deepcell-types library. It combines quantification and phenotyping into a
single step using pre-trained language-informed vision models.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile

logger = logging.getLogger(__name__)


def load_multichannel_image(image_path: str) -> tuple[np.ndarray, list[str]]:
    """Load multichannel OME-TIFF image and extract channel names.

    Parameters
    ----------
    image_path : str
        Path to merged multichannel OME-TIFF image

    Returns
    -------
    img : np.ndarray
        Image array with shape (C, H, W) as uint16
    channel_names : list of str
        List of channel/marker names
    """
    logger.info(f"Loading image: {image_path}")

    # Load image
    img = tifffile.imread(image_path)

    # Ensure channels-first format (C, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim == 3:
        # If shape is (H, W, C), transpose to (C, H, W)
        if img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
            img = np.transpose(img, (2, 0, 1))

    # Get channel names from OME-TIFF metadata or filename
    with tifffile.TiffFile(image_path) as tif:
        try:
            # Try to get channel names from OME metadata
            if tif.ome_metadata:
                # Parse OME-XML for channel names
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                channels = root.findall('.//ome:Channel', ns)
                channel_names = [ch.get('Name', f'Channel_{i}') for i, ch in enumerate(channels)]
            else:
                channel_names = None
        except:
            channel_names = None

    # Fallback: parse from filename
    if not channel_names or len(channel_names) != img.shape[0]:
        logger.warning("Could not extract channel names from OME metadata, parsing from filename")
        channel_names = parse_channel_names_from_filename(image_path, img.shape[0])

    logger.info(f"Image shape: {img.shape}")
    logger.info(f"Channel names: {channel_names}")

    # Convert to uint16 if needed
    if img.dtype != np.uint16:
        logger.info(f"Converting image from {img.dtype} to uint16")
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Normalize to 0-65535 range
            img = ((img - img.min()) / (img.max() - img.min()) * 65535).astype(np.uint16)
        else:
            img = img.astype(np.uint16)

    return img, channel_names


def parse_channel_names_from_filename(image_path: str, n_channels: int) -> list[str]:
    """Parse channel names from filename.

    Expected filename format: merged_all.ome.tif
    Falls back to generic names if parsing fails.

    Parameters
    ----------
    image_path : str
        Path to image file
    n_channels : int
        Number of channels in image

    Returns
    -------
    list of str
        Channel names
    """
    # For merged files, we can't reliably parse channel names from filename
    # Return generic names
    return [f"Channel_{i}" for i in range(n_channels)]


def load_segmentation_mask(mask_path: str) -> np.ndarray:
    """Load segmentation mask.

    Parameters
    ----------
    mask_path : str
        Path to segmentation mask TIFF

    Returns
    -------
    np.ndarray
        Mask array with shape (H, W) as uint32
    """
    logger.info(f"Loading segmentation mask: {mask_path}")

    mask = tifffile.imread(mask_path)

    # Ensure 2D
    if mask.ndim > 2:
        logger.warning(f"Mask has {mask.ndim} dimensions, taking first channel")
        mask = mask[..., 0] if mask.shape[-1] < mask.shape[0] else mask[0]

    # Convert to uint32
    if mask.dtype != np.uint32:
        mask = mask.astype(np.uint32)

    n_cells = len(np.unique(mask)) - 1  # Exclude background (0)
    logger.info(f"Mask shape: {mask.shape}")
    logger.info(f"Number of cells: {n_cells}")

    return mask


def classify_cells(
    image: np.ndarray,
    mask: np.ndarray,
    channel_names: list[str],
    pixel_size_um: float = 0.325,
    model_name: str = "deepcell-types_2025-06-09",
    device: str = "cuda:0",
    num_workers: int = 4
) -> list[str]:
    """Classify cell types using deepcell-types.

    Parameters
    ----------
    image : np.ndarray
        Multichannel image with shape (C, H, W) as uint16
    mask : np.ndarray
        Segmentation mask with shape (H, W) as uint32
    channel_names : list of str
        Channel/marker names
    pixel_size_um : float, optional
        Pixel size in microns per pixel. Default: 0.325
    model_name : str, optional
        Model name. Default: "deepcell-types_2025-06-09"
    device : str, optional
        Device for inference ("cuda:0" or "cpu"). Default: "cuda:0"
    num_workers : int, optional
        Number of data loader threads. Default: 4

    Returns
    -------
    list of str
        Cell type predictions (length = number of cells)
    """
    logger.info("Running cell type classification with deepcell-types...")

    try:
        import deepcell_types
    except ImportError:
        raise ImportError(
            "deepcell-types is not installed. "
            "Install with: pip install git+https://github.com/vanvalenlab/deepcell-types@master"
        )

    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Pixel size: {pixel_size_um} μm/px")

    # Run prediction
    cell_types = deepcell_types.predict(
        image,
        mask,
        channel_names,
        pixel_size_um,
        model_name=model_name,
        device_num=device,
        num_workers=num_workers,
    )

    logger.info(f"Classified {len(cell_types)} cells")

    # Log cell type distribution
    # Note: cell_types might be a list of lists, so we need to handle that
    try:
        # Convert to pandas Series for value_counts
        # If elements are lists, convert to tuples (hashable)
        if len(cell_types) > 0 and isinstance(cell_types[0], (list, np.ndarray)):
            cell_types_hashable = [tuple(ct) if isinstance(ct, (list, np.ndarray)) else ct for ct in cell_types]
        else:
            cell_types_hashable = cell_types

        type_counts = pd.Series(cell_types_hashable).value_counts()
        logger.info("Cell type distribution:")
        for cell_type, count in type_counts.head(10).items():  # Show top 10
            logger.info(f"  {cell_type}: {count}")

        logger.info(f"Unique cell types: {len(type_counts)}")
    except Exception as e:
        logger.warning(f"Could not log cell type distribution: {e}")
        logger.warning(f"Cell types sample: {cell_types[:10]}")

    return cell_types


def create_cell_dataframe(
    mask: np.ndarray,
    image: np.ndarray,
    channel_names: list[str],
    cell_types: list[str]
) -> pd.DataFrame:
    """Create dataframe with cell information.

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask (H, W)
    image : np.ndarray
        Multichannel image (C, H, W)
    channel_names : list of str
        Channel names
    cell_types : list of str
        Cell type predictions

    Returns
    -------
    pd.DataFrame
        Cell data with label, cell_type, and channel intensities
    """
    from skimage import measure

    logger.info("Extracting cell properties...")

    # Get cell labels (exclude background 0)
    cell_labels = np.unique(mask)[1:]

    # Compute region properties
    props = measure.regionprops(mask)

    # Build dataframe
    data = []
    for i, (prop, cell_type) in enumerate(zip(props, cell_types)):
        row = {
            'label': prop.label,
            'cell_type': cell_type,
            'centroid_y': prop.centroid[0],
            'centroid_x': prop.centroid[1],
            'area': prop.area,
        }

        # Add mean intensity for each channel
        for ch_idx, ch_name in enumerate(channel_names):
            # Get pixel intensities for this cell in this channel
            cell_mask = mask == prop.label
            intensities = image[ch_idx][cell_mask]
            row[ch_name] = np.mean(intensities)

        data.append(row)

    df = pd.DataFrame(data)

    logger.info(f"Created dataframe with {len(df)} cells and {len(df.columns)} columns")

    return df


def run_classification_pipeline(
    image_path: str,
    mask_path: str,
    output_csv: str,
    pixel_size_um: float = 0.325,
    model_name: str = "deepcell-types_2025-06-09",
    device: str = "cuda:0",
    num_workers: int = 4
) -> int:
    """Run complete classification pipeline.

    Parameters
    ----------
    image_path : str
        Path to merged multichannel image
    mask_path : str
        Path to segmentation mask
    output_csv : str
        Output CSV path
    pixel_size_um : float, optional
        Pixel size in microns per pixel. Default: 0.325
    model_name : str, optional
        DeepCell Types model name
    device : str, optional
        Device for inference
    num_workers : int, optional
        Number of data loader threads

    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Load data
    image, channel_names = load_multichannel_image(image_path)
    mask = load_segmentation_mask(mask_path)

    # Validate dimensions match
    if image.shape[1:] != mask.shape:
        raise ValueError(
            f"Image spatial dimensions {image.shape[1:]} don't match mask dimensions {mask.shape}"
        )

    # Classify cells
    cell_types = classify_cells(
        image,
        mask,
        channel_names,
        pixel_size_um=pixel_size_um,
        model_name=model_name,
        device=device,
        num_workers=num_workers
    )

    # Create output dataframe
    df = create_cell_dataframe(mask, image, channel_names, cell_types)

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    logger.info(f"✓ Saved results to: {output_csv}")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell type classification using deepcell-types',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to merged multichannel image'
    )

    parser.add_argument(
        '--mask',
        type=str,
        required=True,
        help='Path to segmentation mask'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV path'
    )

    parser.add_argument(
        '--pixel-size',
        type=float,
        default=0.325,
        help='Pixel size in microns per pixel'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='deepcell-types_2025-06-09',
        help='DeepCell Types model name'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for inference (cuda:0 or cpu)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader threads'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    try:
        return run_classification_pipeline(
            args.image,
            args.mask,
            args.output,
            args.pixel_size,
            args.model,
            args.device,
            args.num_workers
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

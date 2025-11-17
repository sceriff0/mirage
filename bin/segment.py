#!/usr/bin/env python3
"""Cell segmentation utilities using StarDist.

This module exposes small, well-typed functions for loading DAPI images,
preprocessing and segmentation. Functions follow NumPy docstring conventions.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import tifffile
from csbdeep.utils import normalize
from skimage import segmentation, morphology, filters
from stardist.models import StarDist2D
from numpy.typing import NDArray

from utils.io import save_pickle
from utils.image_ops import (
    generate_crop_positions,
    extract_crop,
    reconstruct_image_from_crops,
    apply_gamma_correction
)

logger = logging.getLogger(__name__)

__all__ = [
    "load_dapi_image",
    "preprocess_dapi",
    "normalize_image",
    "load_stardist_model",
    "segment_whole_image",
    "segment_with_cropping",
    "remap_labels",
    "run_segmentation",
]


def load_dapi_image(
    image_path: str
) -> Tuple[NDArray, dict]:
    """
    Load DAPI image from file.

    Parameters
    ----------
    image_path : str
        Path to image file.

    Returns
    -------
    image : ndarray, shape (Y, X)
        DAPI image.
    metadata : dict
        Image metadata including pixel sizes.
    """
    logger.info(f"Loading DAPI image: {image_path}")

    # Load image using tifffile; metadata is best-effort (OME XML if present).
    image_data = tifffile.imread(image_path)
    metadata = {}
    try:
        with tifffile.TiffFile(image_path) as tf:
            ome = tf.ome_metadata
            if ome:
                metadata['ome'] = ome
    except Exception:
        pass

    logger.info(f"  Shape: {image_data.shape}")

    return image_data, metadata


def preprocess_dapi(
    image: NDArray,
    gamma: float = 0.6,
    tophat_radius: int = 50,
    gaussian_sigma: float = 1.0
) -> NDArray:
    """
    Preprocess DAPI image for segmentation.

    Parameters
    ----------
    image : ndarray
        Input DAPI image.
    gamma : float, optional
        Gamma correction value. Default is 0.6.
    tophat_radius : int, optional
        White top-hat morphological radius. Default is 50.
    gaussian_sigma : float, optional
        Gaussian smoothing sigma. Default is 1.0.

    Returns
    -------
    preprocessed : ndarray
        Preprocessed image.

    Notes
    -----
    Pipeline: gamma correction → white top-hat → gaussian smoothing
    """
    # Gamma correction
    preprocessed = apply_gamma_correction(image, gamma)

    # White top-hat for background correction
    footprint = morphology.disk(tophat_radius)
    preprocessed = morphology.white_tophat(preprocessed, footprint=footprint)

    # Gaussian smoothing
    preprocessed = filters.gaussian(
        preprocessed.astype(np.float32),
        sigma=gaussian_sigma
    )

    return preprocessed


def normalize_image(
    image: NDArray,
    pmin: float = 1.0,
    pmax: float = 99.8
) -> NDArray:
    """
    Normalize image intensity using percentiles.

    Parameters
    ----------
    image : ndarray
        Input image.
    pmin : float, optional
        Lower percentile. Default is 1.0.
    pmax : float, optional
        Upper percentile. Default is 99.8.

    Returns
    -------
    normalized : ndarray
        Normalized image with values in [0, 1].
    """
    return normalize(image, pmin, pmax, axis=(0, 1))


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
        Name of the model.
    use_gpu : bool, optional
        Use GPU acceleration. Default is True.

    Returns
    -------
    model : StarDist2D
        Loaded model.
    """
    logger.info(f"Loading StarDist model: {model_name}")
    logger.info(f"  Model directory: {model_dir}")
    logger.info(f"  Use GPU: {use_gpu}")

    model = StarDist2D(None, name=model_name, basedir=model_dir)
    model.config.use_gpu = use_gpu

    return model


def segment_whole_image(
    image: NDArray,
    model: StarDist2D,
    n_tiles: Tuple[int, int] = (16, 16)
) -> NDArray:
    """
    Segment entire image without cropping.

    Parameters
    ----------
    image : ndarray, shape (Y, X)
        Preprocessed and normalized image.
    model : StarDist2D
        Loaded StarDist model.
    n_tiles : tuple of int, optional
        Number of tiles for processing. Default is (16, 16).

    Returns
    -------
    mask : ndarray, shape (Y, X)
        Segmentation mask with cell labels.
    """
    logger.info(f"Segmenting whole image (shape: {image.shape})")

    start_time = time.time()

    # Predict instances
    labels, _ = model.predict_instances(image, n_tiles=n_tiles, verbose=False)

    # Expand labels to fill gaps
    labels_expanded = segmentation.expand_labels(
        labels,
        distance=10,
        spacing=1
    )

    elapsed = time.time() - start_time

    n_cells = len(np.unique(labels_expanded)) - 1  # Subtract background
    logger.info(f"  Cells detected: {n_cells}")
    logger.info(f"  Time: {elapsed:.2f}s")

    return labels_expanded


def segment_with_cropping(
    image: NDArray,
    model: StarDist2D,
    crop_size: int = 8000,
    overlap: int = 500
) -> Tuple[NDArray, List[Tuple[int, int, int, int]]]:
    """
    Segment image using overlapping crops.

    Parameters
    ----------
    image : ndarray, shape (Y, X)
        Preprocessed and normalized image.
    model : StarDist2D
        Loaded StarDist model.
    crop_size : int, optional
        Size of crops. Default is 8000.
    overlap : int, optional
        Overlap between crops. Default is 500.

    Returns
    -------
    mask : ndarray, shape (Y, X)
        Stitched segmentation mask.
    positions : list of tuple
        Crop positions used.

    Notes
    -----
    This function handles label remapping to ensure unique cell IDs
    across all crops.
    """
    logger.info(f"Segmenting with cropping")
    logger.info(f"  Crop size: {crop_size}")
    logger.info(f"  Overlap: {overlap}")

    # Generate crop positions
    positions = generate_crop_positions(
        image_shape=image.shape,
        crop_size=crop_size,
        overlap=overlap
    )

    logger.info(f"  Number of crops: {len(positions)}")

    # Segment each crop
    segmented_crops = []
    max_label = 0

    for idx, pos in enumerate(positions):
        logger.info(f"  Processing crop {idx + 1}/{len(positions)}")

        # Extract crop
        crop = extract_crop(image, pos)

        # Segment
        labels, _ = model.predict_instances(crop, verbose=False)

        # Remap labels to ensure uniqueness
        if max_label > 0:
            labels[labels > 0] += max_label

        # Expand labels
        labels_expanded = segmentation.expand_labels(
            labels,
            distance=10,
            spacing=1
        )

        max_label = np.max(labels_expanded)

        segmented_crops.append(labels_expanded)

        n_cells = len(np.unique(labels_expanded)) - 1
        logger.info(f"    Cells in crop: {n_cells}")

    # Reconstruct full image
    logger.info("Stitching crops...")
    mask = reconstruct_image_from_crops(
        crops=segmented_crops,
        positions=positions,
        image_shape=image.shape,
        overlap=overlap,
        blend_mode='max'  # Use max for segmentation masks
    )

    # Remap to consecutive labels
    mask = remap_labels(mask)

    n_cells = len(np.unique(mask)) - 1
    logger.info(f"Total cells after stitching: {n_cells}")

    return mask, positions


def remap_labels(mask: NDArray) -> NDArray:
    """
    Remap mask labels to consecutive integers starting from 1.

    Parameters
    ----------
    mask : ndarray
        Segmentation mask with potentially non-consecutive labels.

    Returns
    -------
    remapped : ndarray
        Mask with consecutive labels [0, 1, 2, ..., N].

    Examples
    --------
    >>> mask = np.array([[0, 5, 5], [0, 10, 10]])
    >>> remapped = remap_labels(mask)
    >>> np.unique(remapped)
    array([0, 1, 2])
    """
    unique_labels = np.unique(mask[mask > 0])

    remapped = np.zeros_like(mask)

    for new_label, old_label in enumerate(unique_labels, start=1):
        remapped[mask == old_label] = new_label

    return remapped


def run_segmentation(
    dapi_path: str,
    output_dir: str,
    model_dir: str,
    model_name: str,
    use_gpu: bool = True,
    process_whole_image: bool = True,
    crop_size: int = 8000,
    overlap: int = 500,
    gamma: float = 0.6,
    tophat_radius: int = 50,
    gaussian_sigma: float = 1.0,
    pmin: float = 1.0,
    pmax: float = 99.8,
    log_file: str = None
) -> Tuple[str, str]:
    """
    Run complete segmentation pipeline.

    Parameters
    ----------
    dapi_path : str
        Path to DAPI image.
    output_dir : str
        Output directory.
    model_dir : str
        StarDist model directory.
    model_name : str
        StarDist model name.
    use_gpu : bool, optional
        Use GPU. Default is True.
    process_whole_image : bool, optional
        Process whole image without cropping. Default is True.
    crop_size : int, optional
        Crop size if using cropping. Default is 8000.
    overlap : int, optional
        Overlap if using cropping. Default is 500.
    gamma : float, optional
        Gamma correction. Default is 0.6.
    tophat_radius : int, optional
        Top-hat radius. Default is 50.
    gaussian_sigma : float, optional
        Gaussian sigma. Default is 1.0.
    pmin : float, optional
        Normalization lower percentile. Default is 1.0.
    pmax : float, optional
        Normalization upper percentile. Default is 99.8.
    log_file : str, optional
        Log file path. Default is None.

    Returns
    -------
    mask_path : str
        Path to saved segmentation mask.
    positions_path : str
        Path to saved crop positions.
    """
    # Setup logging.
    if log_file:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info("=" * 80)
    logger.info("Starting Segmentation Pipeline")
    logger.info("=" * 80)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load image
    dapi_image, metadata = load_dapi_image(dapi_path)

    # Preprocess
    logger.info("Preprocessing DAPI image...")
    preprocessed = preprocess_dapi(
        dapi_image,
        gamma=gamma,
        tophat_radius=tophat_radius,
        gaussian_sigma=gaussian_sigma
    )

    # Normalize
    logger.info("Normalizing image...")
    normalized = normalize_image(preprocessed, pmin=pmin, pmax=pmax)

    # Load model
    model = load_stardist_model(model_dir, model_name, use_gpu=use_gpu)

    # Segment
    if process_whole_image:
        mask = segment_whole_image(normalized, model)
        positions = generate_crop_positions(
            image_shape=dapi_image.shape,
            crop_size=crop_size,
            overlap=overlap
        )
    else:
        mask, positions = segment_with_cropping(
            normalized,
            model,
            crop_size=crop_size,
            overlap=overlap
        )

    # Save outputs
    basename = Path(dapi_path).stem
    mask_path = Path(output_dir) / f"{basename}_segmentation_mask.npy"
    positions_path = Path(output_dir) / f"{basename}_positions.pkl"

    logger.info(f"Saving segmentation mask: {mask_path.name}")
    np.save(mask_path, mask)

    logger.info(f"Saving crop positions: {positions_path.name}")
    save_pickle(positions, str(positions_path))

    logger.info("=" * 80)
    logger.info("Segmentation Complete")
    logger.info("=" * 80)

    return str(mask_path), str(positions_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell segmentation using StarDist',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        '--dapi_file',
        type=str,
        required=True,
        help='Path to DAPI image'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='StarDist model directory'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='StarDist model name'
    )

    # Optional
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Output directory'
    )

    parser.add_argument(
        '--use_gpu',
        action='store_true',
        default=True,
        help='Use GPU acceleration'
    )

    parser.add_argument(
        '--whole_image',
        action='store_true',
        help='Process whole image (no cropping)'
    )

    parser.add_argument(
        '--crop_size',
        type=int,
        default=8000,
        help='Crop size if using cropping'
    )

    parser.add_argument(
        '--overlap',
        type=int,
        default=500,
        help='Overlap between crops'
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.6,
        help='Gamma correction value'
    )

    parser.add_argument(
        '--tophat_radius',
        type=int,
        default=50,
        help='Top-hat morphological filter radius'
    )

    parser.add_argument(
        '--gaussian_sigma',
        type=float,
        default=1.0,
        help='Gaussian filter sigma for smoothing'
    )

    parser.add_argument(
        '--pmin',
        type=float,
        default=1.0,
        help='Normalization lower percentile'
    )

    parser.add_argument(
        '--pmax',
        type=float,
        default=99.8,
        help='Normalization upper percentile'
    )

    parser.add_argument(
        '--log_file',
        type=str,
        help='Log file path'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    run_segmentation(
        dapi_path=args.dapi_file,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        model_name=args.model_name,
        use_gpu=args.use_gpu,
        process_whole_image=args.whole_image,
        crop_size=args.crop_size,
        overlap=args.overlap,
        gamma=args.gamma,
        tophat_radius=args.tophat_radius,
        gaussian_sigma=args.gaussian_sigma,
        pmin=args.pmin,
        pmax=args.pmax,
        log_file=args.log_file
    )

    return 0


if __name__ == '__main__':
    exit(main())

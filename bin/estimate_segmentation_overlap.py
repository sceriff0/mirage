#!/usr/bin/env python3
"""Estimate segmentation overlap between reference and registered images using DeepCell.

This script provides robust, biologically meaningful measures of registration quality
by segmenting nuclei (DAPI channel) in both reference and registered images using
DeepCell deep learning model, then computing overlap metrics:
- Intersection over Union (IoU/Jaccard)
- Dice Coefficient
- Cell count correlation
- Spatial overlap statistics

This approach is complementary to feature-based TRE and provides dense, interpretable
quality metrics that directly relate to biological structure alignment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import tifffile

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# Import from shared library modules
from logger import log_progress


def load_dapi_channel(image_path: str, max_dim: Optional[int] = None) -> np.ndarray:
    """Load DAPI channel from image using memory-mapped I/O.

    Parameters
    ----------
    image_path : str
        Path to the image file
    max_dim : int, optional
        Maximum dimension for downsampling

    Returns
    -------
    np.ndarray
        DAPI channel as numpy array (H, W)
    """
    log_progress(f"Loading DAPI channel: {Path(image_path).name}")

    with tifffile.TiffFile(image_path) as tif:
        image_shape = tif.series[0].shape
        image_dtype = tif.series[0].dtype

        log_progress(f"  Image shape: {image_shape}")
        log_progress(f"  Image dtype: {image_dtype}")

        image_memmap = tif.asarray(out='memmap')

        if image_memmap.ndim == 2:
            log_progress("  Single channel image (assuming DAPI)")
            dapi_image = np.array(image_memmap, copy=True)
        elif image_memmap.ndim == 3:
            n_channels = image_memmap.shape[0]
            log_progress(f"  Multichannel image with {n_channels} channels")
            log_progress(f"  Extracting DAPI channel (index 0)")
            dapi_image = np.array(image_memmap[0, :, :], copy=True)
        else:
            raise ValueError(
                f"Unexpected image dimensions: {image_shape}. "
                f"Expected 2D (Y, X) or 3D (C, Y, X)"
            )

    if max_dim is not None and max(dapi_image.shape) > max_dim:
        from skimage.transform import resize
        scale = max_dim / max(dapi_image.shape)
        new_shape = (int(dapi_image.shape[0] * scale), int(dapi_image.shape[1] * scale))
        log_progress(f"  Downsampling from {dapi_image.shape} to {new_shape}")
        dapi_image = resize(dapi_image, new_shape, preserve_range=True, anti_aliasing=True).astype(dapi_image.dtype)

    log_progress(f"  DAPI channel shape: {dapi_image.shape}")
    log_progress(f"  DAPI value range: [{dapi_image.min()}, {dapi_image.max()}]")

    return dapi_image


def segment_nuclei_deepcell(
    img: np.ndarray,
    min_nucleus_size: int = 100,
    max_nucleus_size: int = 5000
) -> np.ndarray:
    """Segment nuclei using DeepCell deep learning model."""
    log_progress("  Segmenting nuclei using DeepCell...")

    try:
        from deepcell.applications import NuclearSegmentation
        from skimage import morphology
    except ImportError as e:
        log_progress(f"  ERROR: Failed to import DeepCell: {e}")
        log_progress("  Falling back to simple thresholding...")
        return segment_nuclei_threshold(img, min_nucleus_size, max_nucleus_size)

    if img.ndim == 2:
        img_4d = img[np.newaxis, :, :, np.newaxis]
    else:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_4d = (img_4d - img_min) / (img_max - img_min)
    else:
        log_progress("  WARNING: Image has constant intensity, using threshold fallback")
        return segment_nuclei_threshold(img, min_nucleus_size, max_nucleus_size)

    img_4d = img_4d.astype(np.float32)

    app = NuclearSegmentation()
    labels = app.predict(img_4d)
    labels = labels[0, :, :, 0].astype(np.uint32)

    labels = morphology.remove_small_objects(labels, min_size=min_nucleus_size)
    labels = morphology.remove_large_objects(labels, max_size=max_nucleus_size)
    labels = morphology.label(labels > 0)

    n_nuclei = labels.max()
    log_progress(f"    Detected {n_nuclei} nuclei")

    return labels.astype(np.uint32)


def segment_nuclei_threshold(
    img: np.ndarray,
    min_nucleus_size: int = 100,
    max_nucleus_size: int = 5000
) -> np.ndarray:
    """Fallback threshold-based segmentation."""
    from skimage import filters, morphology

    threshold = filters.threshold_otsu(img)
    binary = img > threshold

    binary = morphology.remove_small_holes(binary, area_threshold=min_nucleus_size)
    binary = morphology.remove_small_objects(binary, min_size=min_nucleus_size)

    labels = morphology.label(binary)
    labels = morphology.remove_large_objects(labels, max_size=max_nucleus_size)

    n_nuclei = labels.max()
    log_progress(f"    Detected {n_nuclei} nuclei (threshold method)")

    return labels.astype(np.uint32)


def compute_segmentation_metrics(
    mask_ref: np.ndarray,
    mask_reg: np.ndarray
) -> Dict:
    """Compute overlap metrics between reference and registered segmentation masks."""
    log_progress("  Computing overlap metrics...")

    binary_ref = mask_ref > 0
    binary_reg = mask_reg > 0

    intersection = np.logical_and(binary_ref, binary_reg).sum()
    union = np.logical_or(binary_ref, binary_reg).sum()

    iou = intersection / union if union > 0 else 0.0
    sum_masks = binary_ref.sum() + binary_reg.sum()
    dice = 2 * intersection / sum_masks if sum_masks > 0 else 0.0

    ref_pixels = binary_ref.sum()
    reg_pixels = binary_reg.sum()

    false_positive_pixels = np.logical_and(binary_reg, ~binary_ref).sum()
    fp_rate = false_positive_pixels / ref_pixels if ref_pixels > 0 else 0.0

    false_negative_pixels = np.logical_and(binary_ref, ~binary_reg).sum()
    fn_rate = false_negative_pixels / ref_pixels if ref_pixels > 0 else 0.0

    n_cells_ref = mask_ref.max()
    n_cells_reg = mask_reg.max()
    cell_count_diff = abs(n_cells_ref - n_cells_reg)
    cell_count_ratio = min(n_cells_ref, n_cells_reg) / max(n_cells_ref, n_cells_reg) if max(n_cells_ref, n_cells_reg) > 0 else 0.0

    coverage = intersection / ref_pixels if ref_pixels > 0 else 0.0

    log_progress(f"    IoU: {iou:.4f}")
    log_progress(f"    Dice: {dice:.4f}")
    log_progress(f"    Coverage: {coverage:.4f}")

    return {
        'iou': float(iou),
        'dice': float(dice),
        'coverage': float(coverage),
        'intersection_pixels': int(intersection),
        'union_pixels': int(union),
        'reference_pixels': int(ref_pixels),
        'registered_pixels': int(reg_pixels),
        'false_positive_rate': float(fp_rate),
        'false_negative_rate': float(fn_rate),
        'n_cells_reference': int(n_cells_ref),
        'n_cells_registered': int(n_cells_reg),
        'cell_count_difference': int(cell_count_diff),
        'cell_count_correlation': float(cell_count_ratio)
    }


def save_overlay_visualization(
    ref_img: np.ndarray,
    mask_ref: np.ndarray,
    mask_reg: np.ndarray,
    output_path: str,
    moving_name: str
) -> None:
    """Save visualization showing reference and registered segmentation overlays."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    img_min, img_max = ref_img.min(), ref_img.max()
    if img_max > img_min:
        ref_display = ((ref_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        ref_display = np.zeros_like(ref_img, dtype=np.uint8)

    axes[0].imshow(ref_display, cmap='gray')
    axes[0].contour(mask_ref > 0, colors='green', linewidths=0.5)
    axes[0].set_title('Reference Segmentation')
    axes[0].axis('off')

    axes[1].imshow(ref_display, cmap='gray')
    axes[1].contour(mask_reg > 0, colors='red', linewidths=0.5)
    axes[1].set_title('Registered Segmentation')
    axes[1].axis('off')

    binary_ref = mask_ref > 0
    binary_reg = mask_reg > 0

    rgb_overlay = np.stack([ref_display, ref_display, ref_display], axis=2)

    ref_only = np.logical_and(binary_ref, ~binary_reg)
    rgb_overlay[ref_only, 0] = 0
    rgb_overlay[ref_only, 1] = 255
    rgb_overlay[ref_only, 2] = 0

    reg_only = np.logical_and(binary_reg, ~binary_ref)
    rgb_overlay[reg_only, 0] = 255
    rgb_overlay[reg_only, 1] = 0
    rgb_overlay[reg_only, 2] = 0

    overlap = np.logical_and(binary_ref, binary_reg)
    rgb_overlay[overlap, 0] = 255
    rgb_overlay[overlap, 1] = 255
    rgb_overlay[overlap, 2] = 0

    axes[2].imshow(rgb_overlay)
    axes[2].set_title('Overlap (Green=Ref, Red=Reg, Yellow=Both)')
    axes[2].axis('off')

    plt.suptitle(f'Segmentation Comparison - {moving_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    log_progress(f"  Saved overlay visualization: {output_path}")


def process_registered_image(
    reference_path: str,
    registered_path: str,
    output_prefix: str,
    max_dim: int = 2048,
    min_nucleus_size: int = 100,
    max_nucleus_size: int = 5000
) -> None:
    """Process single registered image and compute overlap metrics against reference.

    Parameters
    ----------
    reference_path : str
        Path to reference image
    registered_path : str
        Path to registered image
    output_prefix : str
        Prefix for output files (e.g., 'sample1_DAPI')
    max_dim : int
        Maximum dimension for downsampling
    min_nucleus_size : int
        Minimum nucleus size in pixels
    max_nucleus_size : int
        Maximum nucleus size in pixels
    """
    log_progress("=" * 70)
    log_progress("SEGMENTATION OVERLAP ESTIMATION (DeepCell-Based)")
    log_progress("=" * 70)

    log_progress("\n[1/3] Processing reference image...")
    ref_img = load_dapi_channel(reference_path, max_dim=max_dim)
    mask_ref = segment_nuclei_deepcell(ref_img, min_nucleus_size, max_nucleus_size)

    log_progress(f"\n[2/3] Processing registered image...")
    log_progress(f"  Registered: {Path(registered_path).name}")
    reg_img = load_dapi_channel(registered_path, max_dim=max_dim)
    mask_reg = segment_nuclei_deepcell(reg_img, min_nucleus_size, max_nucleus_size)

    log_progress(f"\n[3/3] Computing overlap metrics...")
    overlap_metrics = compute_segmentation_metrics(mask_ref, mask_reg)

    # Save visualization
    viz_path = f"{output_prefix}_segmentation_overlay.png"
    save_overlay_visualization(ref_img, mask_ref, mask_reg, viz_path, output_prefix)

    # Compile results
    results = {
        "reference_image": Path(reference_path).name,
        "registered_image": Path(registered_path).name,
        "segmentation_params": {
            "method": "deepcell",
            "max_dim": max_dim,
            "min_nucleus_size": min_nucleus_size,
            "max_nucleus_size": max_nucleus_size
        },
        "overlap_metrics": overlap_metrics
    }

    # Save JSON results
    output_path = f"{output_prefix}_segmentation_overlap.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    log_progress(f"\n  Saved results: {output_path}")
    log_progress(f"    IoU: {overlap_metrics['iou']:.4f}, Dice: {overlap_metrics['dice']:.4f}")

    log_progress("\n" + "=" * 70)
    log_progress("✓ Segmentation overlap estimation complete!")
    log_progress("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimate segmentation overlap between reference and registered images using DeepCell',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--reference', required=True,
                       help='Path to reference image')
    parser.add_argument('--registered', required=True,
                       help='Path to registered image')
    parser.add_argument('--output-prefix', required=True,
                       help='Prefix for output files (e.g., sample1_DAPI)')
    parser.add_argument('--max-dim', type=int, default=2048,
                       help='Maximum image dimension for processing')
    parser.add_argument('--min-nucleus-size', type=int, default=100,
                       help='Minimum nucleus size in pixels')
    parser.add_argument('--max-nucleus-size', type=int, default=5000,
                       help='Maximum nucleus size in pixels')

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        process_registered_image(
            reference_path=args.reference,
            registered_path=args.registered,
            output_prefix=args.output_prefix,
            max_dim=args.max_dim,
            min_nucleus_size=args.min_nucleus_size,
            max_nucleus_size=args.max_nucleus_size
        )
        return 0

    except Exception as e:
        log_progress(f"ERROR: Segmentation overlap estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

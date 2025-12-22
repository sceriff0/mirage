#!/usr/bin/env python3
"""Estimate registration error using segmentation-based metrics with StarDist.

This script provides a robust, biologically meaningful measure of registration quality
by segmenting nuclei (DAPI channel) in both reference and registered images using
StarDist deep learning model, then computing overlap metrics:
- Intersection over Union (IoU/Jaccard)
- Dice Coefficient
- Cell count correlation
- Spatial overlap statistics

StarDist provides more accurate segmentation than threshold-based methods, especially
for crowded or touching nuclei. This approach is complementary to feature-based TRE
and provides dense, interpretable quality metrics that directly relate to biological
structure alignment.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import tifffile


def log_progress(message: str) -> None:
    """Print timestamped progress messages."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_dapi_channel(image_path: str, dapi_channel_index: int = 0) -> np.ndarray:
    """Load DAPI channel from image using memory-mapped I/O.

    Parameters
    ----------
    image_path : str
        Path to the image file
    dapi_channel_index : int
        Index of DAPI channel (default: 0)

    Returns
    -------
    np.ndarray
        DAPI channel as numpy array (H, W)
    """
    log_progress(f"Loading DAPI channel: {os.path.basename(image_path)}")

    # Use memory-mapped I/O to avoid loading entire multichannel image
    with tifffile.TiffFile(image_path) as tif:
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

        log_progress(f"  Image shape: {image_shape}")
        log_progress(f"  Image dtype: {image_dtype}")

        # Memory-map the file (doesn't load into RAM)
        image_memmap = tif.asarray(out='memmap')

        # Handle different image formats
        if image_memmap.ndim == 2:
            # Single channel image, assume it's DAPI
            log_progress("  Single channel image (assuming DAPI)")
            dapi_image = np.array(image_memmap, copy=True)
        elif image_memmap.ndim == 3:
            # Multichannel image (C, Y, X) format
            log_progress(f"  Multichannel image with {n_channels} channels")

            if dapi_channel_index >= n_channels:
                raise ValueError(
                    f"DAPI channel index {dapi_channel_index} out of range "
                    f"for image with {n_channels} channels"
                )

            # Extract ONLY the DAPI channel into RAM (not all channels)
            log_progress(f"  Extracting DAPI channel (index {dapi_channel_index})")
            dapi_image = np.array(image_memmap[dapi_channel_index, :, :], copy=True)
        else:
            raise ValueError(
                f"Unexpected image dimensions: {image_shape}. "
                f"Expected 2D (Y, X) or 3D (C, Y, X)"
            )

    log_progress(f"  DAPI channel shape: {dapi_image.shape}")
    log_progress(f"  DAPI dtype: {dapi_image.dtype}")
    log_progress(f"  DAPI value range: [{dapi_image.min()}, {dapi_image.max()}]")

    return dapi_image


def segment_nuclei_stardist(
    img: np.ndarray,
    model: StarDist2D,
    n_tiles: Tuple[int, int] = (16, 16),
    pmin: float = 1.0,
    pmax: float = 99.8
) -> np.ndarray:
    """Segment nuclei using StarDist deep learning model.

    This provides more accurate segmentation than thresholding methods,
    especially for crowded or touching nuclei.

    Parameters
    ----------
    img : np.ndarray
        Grayscale DAPI image (raw intensity values)
    model : StarDist2D
        Loaded StarDist model
    n_tiles : Tuple[int, int]
        Number of tiles for tiled processing (for memory efficiency)
    pmin : float
        Lower percentile for normalization
    pmax : float
        Upper percentile for normalization

    Returns
    -------
    np.ndarray
        Labeled segmentation mask (0=background, 1,2,3...=nuclei)
    """
    log_progress("  Segmenting nuclei using StarDist...")

    # Normalize using CSBDeep (same as in segment.py)
    normalized = normalize(img, pmin, pmax, axis=(0, 1))

    # Convert to float32 to save memory
    if normalized.dtype != np.float32:
        normalized = normalized.astype(np.float32)

    # Segment using StarDist
    labels, _ = model.predict_instances(
        normalized,
        n_tiles=n_tiles,
        show_tile_progress=False,
        verbose=False
    )

    n_nuclei = labels.max()
    log_progress(f"    Detected {n_nuclei} nuclei")

    return labels.astype(np.uint32)


def load_stardist_model(
    model_dir: str,
    model_name: str,
    use_gpu: bool = False
) -> StarDist2D:
    """Load pre-trained StarDist model.

    Parameters
    ----------
    model_dir : str
        Directory containing the model
    model_name : str
        Name of the model (e.g., '2D_versatile_fluo')
    use_gpu : bool
        Use GPU acceleration if available

    Returns
    -------
    StarDist2D
        Loaded StarDist model
    """
    log_progress(f"Loading StarDist model: {model_name}")
    log_progress(f"  Model directory: {model_dir}")
    log_progress(f"  GPU enabled: {use_gpu}")

    # Verify the model path exists
    model_path = Path(model_dir) / model_name
    config_file = model_path / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    model = StarDist2D(None, name=model_name, basedir=model_dir)

    if hasattr(model, 'config'):
        model.config.use_gpu = use_gpu

    log_progress("  ✓ Model loaded successfully")

    return model


def compute_segmentation_metrics(
    mask_ref: np.ndarray,
    mask_reg: np.ndarray
) -> Dict:
    """Compute overlap metrics between reference and registered segmentation masks.

    Parameters
    ----------
    mask_ref : np.ndarray
        Reference segmentation mask (labeled)
    mask_reg : np.ndarray
        Registered segmentation mask (labeled)

    Returns
    -------
    dict
        Dictionary containing IoU, Dice, and other overlap metrics
    """
    log_progress("  Computing overlap metrics...")

    # Binarize masks (any label > 0 = foreground)
    binary_ref = mask_ref > 0
    binary_reg = mask_reg > 0

    # Compute intersection and union
    intersection = np.logical_and(binary_ref, binary_reg).sum()
    union = np.logical_or(binary_ref, binary_reg).sum()

    # IoU (Jaccard Index)
    iou = intersection / union if union > 0 else 0.0

    # Dice Coefficient
    sum_masks = binary_ref.sum() + binary_reg.sum()
    dice = 2 * intersection / sum_masks if sum_masks > 0 else 0.0

    # Pixel-level metrics
    ref_pixels = binary_ref.sum()
    reg_pixels = binary_reg.sum()

    # False positive rate (over-segmentation in registered)
    false_positive_pixels = np.logical_and(binary_reg, ~binary_ref).sum()
    fp_rate = false_positive_pixels / ref_pixels if ref_pixels > 0 else 0.0

    # False negative rate (under-segmentation in registered)
    false_negative_pixels = np.logical_and(binary_ref, ~binary_reg).sum()
    fn_rate = false_negative_pixels / ref_pixels if ref_pixels > 0 else 0.0

    # Cell count metrics
    n_cells_ref = mask_ref.max()
    n_cells_reg = mask_reg.max()
    cell_count_diff = abs(n_cells_ref - n_cells_reg)
    cell_count_ratio = min(n_cells_ref, n_cells_reg) / max(n_cells_ref, n_cells_reg) if max(n_cells_ref, n_cells_reg) > 0 else 0.0

    # Coverage (what fraction of reference is covered by registered)
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


def compute_spatial_error_distribution(
    mask_ref: np.ndarray,
    mask_reg: np.ndarray,
    grid_size: int = 10
) -> Dict:
    """Compute spatial distribution of segmentation errors across the image.

    Divides image into grid and computes IoU per region to identify
    areas with poor registration.

    Parameters
    ----------
    mask_ref : np.ndarray
        Reference segmentation mask
    mask_reg : np.ndarray
        Registered segmentation mask
    grid_size : int
        Number of grid divisions per dimension

    Returns
    -------
    dict
        Statistics about spatial error distribution
    """
    h, w = mask_ref.shape
    grid_h = h // grid_size
    grid_w = w // grid_size

    ious = []

    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * grid_h
            y_end = (i + 1) * grid_h if i < grid_size - 1 else h
            x_start = j * grid_w
            x_end = (j + 1) * grid_w if j < grid_size - 1 else w

            ref_patch = mask_ref[y_start:y_end, x_start:x_end] > 0
            reg_patch = mask_reg[y_start:y_end, x_start:x_end] > 0

            intersection = np.logical_and(ref_patch, reg_patch).sum()
            union = np.logical_or(ref_patch, reg_patch).sum()

            if union > 0:
                patch_iou = intersection / union
                ious.append(patch_iou)

    if len(ious) > 0:
        return {
            'mean_iou': float(np.mean(ious)),
            'std_iou': float(np.std(ious)),
            'min_iou': float(np.min(ious)),
            'max_iou': float(np.max(ious)),
            'n_patches': len(ious)
        }
    else:
        return {
            'mean_iou': 0.0,
            'std_iou': 0.0,
            'min_iou': 0.0,
            'max_iou': 0.0,
            'n_patches': 0
        }


def save_overlay_visualization(
    ref_img: np.ndarray,
    mask_ref: np.ndarray,
    mask_reg: np.ndarray,
    output_path: str,
    moving_name: str
) -> None:
    """Save visualization showing reference (green) and registered (red) segmentation overlays.

    Parameters
    ----------
    ref_img : np.ndarray
        Reference DAPI image (grayscale)
    mask_ref : np.ndarray
        Reference segmentation mask
    mask_reg : np.ndarray
        Registered segmentation mask
    output_path : str
        Path to save visualization
    moving_name : str
        Name for plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Normalize image for display
    ref_display = (ref_img * 255).astype(np.uint8)

    # Panel 1: Reference segmentation (green)
    axes[0].imshow(ref_display, cmap='gray')
    axes[0].contour(mask_ref > 0, colors='green', linewidths=0.5)
    axes[0].set_title('Reference Segmentation')
    axes[0].axis('off')

    # Panel 2: Registered segmentation (red)
    axes[1].imshow(ref_display, cmap='gray')
    axes[1].contour(mask_reg > 0, colors='red', linewidths=0.5)
    axes[1].set_title('Registered Segmentation')
    axes[1].axis('off')

    # Panel 3: Overlap (green=reference, red=registered, yellow=overlap)
    binary_ref = mask_ref > 0
    binary_reg = mask_reg > 0

    rgb_overlay = np.stack([ref_display, ref_display, ref_display], axis=2)

    # Green = reference only
    ref_only = np.logical_and(binary_ref, ~binary_reg)
    rgb_overlay[ref_only, 0] = 0  # Remove red
    rgb_overlay[ref_only, 1] = 255  # Full green
    rgb_overlay[ref_only, 2] = 0  # Remove blue

    # Red = registered only
    reg_only = np.logical_and(binary_reg, ~binary_ref)
    rgb_overlay[reg_only, 0] = 255  # Full red
    rgb_overlay[reg_only, 1] = 0  # Remove green
    rgb_overlay[reg_only, 2] = 0  # Remove blue

    # Yellow = overlap
    overlap = np.logical_and(binary_ref, binary_reg)
    rgb_overlay[overlap, 0] = 255  # Full red
    rgb_overlay[overlap, 1] = 255  # Full green
    rgb_overlay[overlap, 2] = 0  # Remove blue

    axes[2].imshow(rgb_overlay)
    axes[2].set_title('Overlap (Green=Ref, Red=Reg, Yellow=Both)')
    axes[2].axis('off')

    plt.suptitle(f'Segmentation Comparison - {moving_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    log_progress(f"  Saved overlay visualization: {output_path}")


def estimate_registration_error_segmentation(
    reference_path: str,
    registered_path: str,
    output_dir: str,
    model_dir: str,
    model_name: str,
    dapi_channel_index: int = 0,
    use_gpu: bool = False,
    n_tiles: Tuple[int, int] = (16, 16),
    pmin: float = 1.0,
    pmax: float = 99.8
) -> Tuple[dict, str]:
    """Estimate registration error using segmentation-based metrics with StarDist.

    Parameters
    ----------
    reference_path : str
        Path to reference image
    registered_path : str
        Path to registered image
    output_dir : str
        Directory to save results
    model_dir : str
        Directory containing StarDist model
    model_name : str
        Name of StarDist model (e.g., '2D_versatile_fluo')
    dapi_channel_index : int
        Index of DAPI channel (default: 0)
    use_gpu : bool
        Use GPU acceleration if available
    n_tiles : Tuple[int, int]
        Number of tiles for processing (default: (16, 16))
    pmin : float
        Lower percentile for normalization (default: 1.0)
    pmax : float
        Upper percentile for normalization (default: 99.8)

    Returns
    -------
    results_dict : dict
        Dictionary containing segmentation metrics
    output_path : str
        Path to saved results JSON file
    """
    log_progress("=" * 70)
    log_progress("REGISTRATION ERROR ESTIMATION (Segmentation-Based with StarDist)")
    log_progress("=" * 70)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load StarDist model
    log_progress("\n[1/5] Loading StarDist model...")
    model = load_stardist_model(model_dir, model_name, use_gpu=use_gpu)

    # Load DAPI channels
    log_progress("\n[2/5] Loading DAPI channels...")
    ref_img = load_dapi_channel(reference_path, dapi_channel_index=dapi_channel_index)
    reg_img = load_dapi_channel(registered_path, dapi_channel_index=dapi_channel_index)

    # Segment nuclei
    log_progress("\n[3/5] Segmenting nuclei...")
    log_progress("  Reference image:")
    mask_ref = segment_nuclei_stardist(ref_img, model, n_tiles=n_tiles, pmin=pmin, pmax=pmax)

    log_progress("  Registered image:")
    mask_reg = segment_nuclei_stardist(reg_img, model, n_tiles=n_tiles, pmin=pmin, pmax=pmax)

    # Compute overlap metrics
    log_progress("\n[4/5] Computing overlap metrics...")
    overlap_metrics = compute_segmentation_metrics(mask_ref, mask_reg)

    # Compute spatial error distribution
    log_progress("  Computing spatial error distribution...")
    spatial_metrics = compute_spatial_error_distribution(mask_ref, mask_reg, grid_size=10)

    # Save overlay visualization
    log_progress("\n[5/5] Creating visualization...")
    moving_basename = Path(registered_path).stem
    viz_path = os.path.join(output_dir, f"{moving_basename}_segmentation_overlay.png")

    # Normalize reference image for visualization (0-1 range)
    ref_img_norm = ref_img.astype(np.float32)
    if ref_img_norm.max() > 1.0:
        ref_img_norm = ref_img_norm / ref_img_norm.max()

    save_overlay_visualization(ref_img_norm, mask_ref, mask_reg, viz_path, moving_basename)

    # Prepare results
    results = {
        "reference_image": os.path.basename(reference_path),
        "registered_image": os.path.basename(registered_path),
        "dapi_channel_index": dapi_channel_index,
        "segmentation_params": {
            "model_name": model_name,
            "model_dir": model_dir,
            "n_tiles": n_tiles,
            "pmin": pmin,
            "pmax": pmax,
            "use_gpu": use_gpu,
            "method": "stardist"
        },
        "overlap_metrics": overlap_metrics,
        "spatial_distribution": spatial_metrics
    }

    # Save results
    output_filename = f"{moving_basename}_segmentation_error.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    log_progress(f"\n✓ Results saved to: {output_path}")
    log_progress("=" * 70)

    return results, output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimate registration error using segmentation-based metrics with StarDist',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--reference', required=True,
                       help='Path to reference image')
    parser.add_argument('--registered', required=True,
                       help='Path to registered image')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--model-dir', required=True,
                       help='Directory containing StarDist model')
    parser.add_argument('--model-name', required=True,
                       help='StarDist model name (e.g., "2D_versatile_fluo")')

    # Optional arguments
    parser.add_argument('--dapi-channel', type=int, default=0,
                       help='Index of DAPI channel in multichannel image')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                       help='Use GPU acceleration (if available)')
    parser.add_argument('--n-tiles', type=int, nargs=2, default=[16, 16],
                       metavar=('Y', 'X'),
                       help='Number of tiles for processing (Y X)')
    parser.add_argument('--pmin', type=float, default=1.0,
                       help='Lower percentile for normalization')
    parser.add_argument('--pmax', type=float, default=99.8,
                       help='Upper percentile for normalization')

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        results, output_path = estimate_registration_error_segmentation(
            reference_path=args.reference,
            registered_path=args.registered,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            model_name=args.model_name,
            dapi_channel_index=args.dapi_channel,
            use_gpu=args.use_gpu,
            n_tiles=tuple(args.n_tiles),
            pmin=args.pmin,
            pmax=args.pmax
        )

        log_progress(f"\nSegmentation-based error estimation complete!")
        log_progress("=" * 70)
        log_progress(f"  IoU (Jaccard):        {results['overlap_metrics']['iou']:.4f}")
        log_progress(f"  Dice Coefficient:     {results['overlap_metrics']['dice']:.4f}")
        log_progress(f"  Coverage:             {results['overlap_metrics']['coverage']:.4f}")
        log_progress(f"  Cell Count (Ref):     {results['overlap_metrics']['n_cells_reference']}")
        log_progress(f"  Cell Count (Reg):     {results['overlap_metrics']['n_cells_registered']}")
        log_progress(f"  Cell Count Corr:      {results['overlap_metrics']['cell_count_correlation']:.4f}")
        log_progress(f"\n  Results saved: {output_path}")
        log_progress("=" * 70)

        return 0

    except Exception as e:
        log_progress(f"ERROR: Segmentation-based error estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

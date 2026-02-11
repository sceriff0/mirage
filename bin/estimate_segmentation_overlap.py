#!/usr/bin/env python3
"""Estimate segmentation overlap between reference and registered images using StarDist.

This script provides robust, biologically meaningful measures of registration quality
by segmenting nuclei (DAPI channel) in both reference and registered images using
StarDist deep learning model, then computing overlap metrics:
- Intersection over Union (IoU/Jaccard)
- Dice Coefficient
- Per-nucleus registration error (instance-level correspondence)
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# Import from shared library modules
from logger import log_progress

# Import StarDist and dependencies
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.measure import regionprops

__all__ = ["main"]


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
        # Use series if available (OME-TIFF), otherwise fall back to pages
        if tif.series:
            source = tif.series[0]
        else:
            # Non-OME TIFF or corrupted OME metadata - fall back to pages
            log_progress("  Warning: No OME series found, falling back to raw pages")
            if not tif.pages:
                raise ValueError(f"TIFF file appears corrupted: {image_path}")
            source = tif.pages[0]

        image_shape = source.shape
        image_dtype = source.dtype

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


def segment_nuclei_stardist(
    img: np.ndarray,
    model: StarDist2D,
    n_tiles: tuple = (24, 24),
    pmin: float = 1.0,
    pmax: float = 99.8
) -> np.ndarray:
    """Segment nuclei using StarDist deep learning model.

    Uses same normalization and parameters as segment.py for consistency.

    Parameters
    ----------
    img : np.ndarray
        DAPI channel image (2D)
    model : StarDist2D
        Pre-loaded StarDist model
    n_tiles : tuple
        Number of tiles for tiled processing
    pmin : float
        Lower percentile for normalization
    pmax : float
        Upper percentile for normalization

    Returns
    -------
    labels : np.ndarray
        Nuclei segmentation mask with unique cell labels
    """
    log_progress("  Segmenting nuclei using StarDist...")
    log_progress(f"    Normalization percentiles: [{pmin}, {pmax}]")
    log_progress(f"    Tiling: {n_tiles}")

    # Normalize using CSBDeep (same as segment.py)
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    normalized = normalize(img, pmin, pmax, axis=(0, 1))

    # Predict nuclei instances using StarDist
    labels, _ = model.predict_instances(
        normalized,
        n_tiles=n_tiles,
        show_tile_progress=False,
        verbose=False
    )

    n_nuclei = len(np.unique(labels)) - 1  # Subtract background
    log_progress(f"    Detected {n_nuclei} nuclei")

    return labels.astype(np.uint32)


def compute_per_nucleus_error(
    mask_ref: np.ndarray,
    mask_reg: np.ndarray,
    max_distance: float = 50.0
) -> Dict:
    """Compute per-nucleus registration error using Hungarian matching.

    This provides instance-level correspondence between reference and registered
    nuclei, computing the centroid distance for each matched pair.

    Parameters
    ----------
    mask_ref : np.ndarray
        Reference nuclei segmentation mask
    mask_reg : np.ndarray
        Registered nuclei segmentation mask
    max_distance : float
        Maximum distance (pixels) to consider nuclei as potentially matching

    Returns
    -------
    dict
        Per-nucleus error statistics including:
        - matched_pairs: Number of successfully matched nuclei
        - mean_centroid_distance: Mean distance between matched centroids
        - median_centroid_distance: Median distance
        - std_centroid_distance: Standard deviation
        - unmatched_reference: Number of reference nuclei without matches
        - unmatched_registered: Number of registered nuclei without matches
        - distances: List of all matched distances
    """
    log_progress("  Computing per-nucleus registration error...")

    # Extract region properties for each nucleus
    ref_props = regionprops(mask_ref)
    reg_props = regionprops(mask_reg)

    n_ref = len(ref_props)
    n_reg = len(reg_props)

    log_progress(f"    Reference nuclei: {n_ref}")
    log_progress(f"    Registered nuclei: {n_reg}")

    if n_ref == 0 or n_reg == 0:
        log_progress("    WARNING: No nuclei detected in one or both images")
        return {
            'matched_pairs': 0,
            'mean_centroid_distance': 0.0,
            'median_centroid_distance': 0.0,
            'std_centroid_distance': 0.0,
            'q25_centroid_distance': 0.0,
            'q75_centroid_distance': 0.0,
            'q90_centroid_distance': 0.0,
            'q95_centroid_distance': 0.0,
            'max_centroid_distance': 0.0,
            'unmatched_reference': n_ref,
            'unmatched_registered': n_reg,
            'distances': []
        }

    # Build cost matrix based on centroid distances
    ref_centroids = np.array([prop.centroid for prop in ref_props])
    reg_centroids = np.array([prop.centroid for prop in reg_props])

    # Compute pairwise distances
    cost_matrix = cdist(ref_centroids, reg_centroids, metric='euclidean')

    # Solve assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter matches by maximum distance threshold
    matched_distances = cost_matrix[row_ind, col_ind]
    valid_matches = matched_distances <= max_distance

    valid_row_ind = row_ind[valid_matches]
    valid_col_ind = col_ind[valid_matches]
    valid_distances = matched_distances[valid_matches]

    n_matched = len(valid_distances)
    n_unmatched_ref = n_ref - n_matched
    n_unmatched_reg = n_reg - n_matched

    log_progress(f"    Matched pairs: {n_matched}")
    log_progress(f"    Unmatched reference: {n_unmatched_ref}")
    log_progress(f"    Unmatched registered: {n_unmatched_reg}")

    if n_matched > 0:
        mean_dist = float(np.mean(valid_distances))
        median_dist = float(np.median(valid_distances))
        std_dist = float(np.std(valid_distances))
        q25_dist = float(np.percentile(valid_distances, 25))
        q75_dist = float(np.percentile(valid_distances, 75))
        q90_dist = float(np.percentile(valid_distances, 90))
        q95_dist = float(np.percentile(valid_distances, 95))
        max_dist = float(np.max(valid_distances))

        log_progress(f"    Mean centroid distance: {mean_dist:.2f} pixels")
        log_progress(f"    Median centroid distance: {median_dist:.2f} pixels")
        log_progress(f"    Q95 centroid distance: {q95_dist:.2f} pixels")
    else:
        mean_dist = 0.0
        median_dist = 0.0
        std_dist = 0.0
        q25_dist = 0.0
        q75_dist = 0.0
        q90_dist = 0.0
        q95_dist = 0.0
        max_dist = 0.0

    return {
        'matched_pairs': int(n_matched),
        'mean_centroid_distance': mean_dist,
        'median_centroid_distance': median_dist,
        'std_centroid_distance': std_dist,
        'q25_centroid_distance': q25_dist,
        'q75_centroid_distance': q75_dist,
        'q90_centroid_distance': q90_dist,
        'q95_centroid_distance': q95_dist,
        'max_centroid_distance': max_dist,
        'unmatched_reference': int(n_unmatched_ref),
        'unmatched_registered': int(n_unmatched_reg),
        'distances': valid_distances.tolist()
    }


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


def save_nucleus_distance_histogram(
    distances: list,
    output_path: str,
    output_prefix: str
) -> None:
    """Save histogram of per-nucleus centroid distances.

    Parameters
    ----------
    distances : list
        List of centroid distances for matched nuclei
    output_path : str
        Path to save histogram
    output_prefix : str
        Prefix for title
    """
    if len(distances) == 0:
        log_progress("  No matched nuclei to visualize")
        return

    distances_arr = np.array(distances)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(distances_arr, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Centroid Distance (pixels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Per-Nucleus Registration Error Distribution', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    mean_dist = np.mean(distances_arr)
    median_dist = np.median(distances_arr)
    q95_dist = np.percentile(distances_arr, 95)
    stats_text = (
        f'n = {len(distances_arr)} matched nuclei\n'
        f'Mean: {mean_dist:.2f}px\n'
        f'Median: {median_dist:.2f}px\n'
        f'Q95: {q95_dist:.2f}px'
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    log_progress(f"  Saved nucleus distance histogram: {output_path}")


def process_registered_image(
    reference_path: str,
    registered_path: str,
    output_prefix: str,
    model_dir: str,
    model_name: str = "2D_versatile_fluo",
    max_dim: int = 2048,
    n_tiles: tuple = (24, 24),
    pmin: float = 1.0,
    pmax: float = 99.8,
    max_nucleus_distance: float = 50.0
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
    model_dir : str
        Directory containing StarDist model
    model_name : str
        StarDist model name (default: "2D_versatile_fluo")
    max_dim : int
        Maximum dimension for downsampling
    n_tiles : tuple
        Number of tiles for StarDist processing
    pmin : float
        Lower percentile for normalization
    pmax : float
        Upper percentile for normalization
    max_nucleus_distance : float
        Maximum distance for nucleus matching (pixels)
    """
    log_progress("=" * 70)
    log_progress("SEGMENTATION OVERLAP ESTIMATION (StarDist-Based)")
    log_progress("=" * 70)

    # Load StarDist model
    log_progress("\n[1/5] Loading StarDist model...")
    log_progress(f"  Model: {model_name}")
    log_progress(f"  Model dir: {model_dir}")

    model_path = Path(model_dir) / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model = StarDist2D(None, name=model_name, basedir=model_dir)
    model.config.use_gpu = True
    log_progress("  ✓ Model loaded")

    # Process reference image
    log_progress("\n[2/5] Processing reference image...")
    log_progress(f"  Reference: {Path(reference_path).name}")
    ref_img = load_dapi_channel(reference_path, max_dim=max_dim)
    mask_ref = segment_nuclei_stardist(ref_img, model, n_tiles=n_tiles, pmin=pmin, pmax=pmax)

    # Process registered image
    log_progress(f"\n[3/5] Processing registered image...")
    log_progress(f"  Registered: {Path(registered_path).name}")
    reg_img = load_dapi_channel(registered_path, max_dim=max_dim)
    mask_reg = segment_nuclei_stardist(reg_img, model, n_tiles=n_tiles, pmin=pmin, pmax=pmax)

    # Compute overlap metrics
    log_progress(f"\n[4/5] Computing overlap metrics...")
    overlap_metrics = compute_segmentation_metrics(mask_ref, mask_reg)

    # Compute per-nucleus registration error
    per_nucleus_metrics = compute_per_nucleus_error(
        mask_ref, mask_reg, max_distance=max_nucleus_distance
    )

    # Save visualization
    log_progress(f"\n[5/5] Saving visualizations...")
    viz_path = f"{output_prefix}_segmentation_overlay.png"
    save_overlay_visualization(ref_img, mask_ref, mask_reg, viz_path, output_prefix)

    # Save per-nucleus distance histogram
    if len(per_nucleus_metrics['distances']) > 0:
        hist_path = f"{output_prefix}_nucleus_distance_histogram.png"
        save_nucleus_distance_histogram(
            per_nucleus_metrics['distances'], hist_path, output_prefix
        )

    # Compile results
    results = {
        "reference_image": Path(reference_path).name,
        "registered_image": Path(registered_path).name,
        "segmentation_params": {
            "method": "stardist",
            "model_name": model_name,
            "max_dim": max_dim,
            "n_tiles": n_tiles,
            "normalization_pmin": pmin,
            "normalization_pmax": pmax,
            "max_nucleus_distance": max_nucleus_distance
        },
        "overlap_metrics": overlap_metrics,
        "per_nucleus_metrics": per_nucleus_metrics
    }

    # Save JSON results
    output_path = f"{output_prefix}_segmentation_overlap.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    log_progress(f"\n  Saved results: {output_path}")
    log_progress(f"    Pixel-wise Dice: {overlap_metrics['dice']:.4f}")
    log_progress(f"    Matched nuclei: {per_nucleus_metrics['matched_pairs']}")
    log_progress(f"    Mean nucleus distance: {per_nucleus_metrics['mean_centroid_distance']:.2f} px")

    log_progress("\n" + "=" * 70)
    log_progress("✓ Segmentation overlap estimation complete!")
    log_progress("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimate segmentation overlap between reference and registered images using StarDist',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--reference', required=True,
                       help='Path to reference image')
    parser.add_argument('--registered', required=True,
                       help='Path to registered image')
    parser.add_argument('--output-prefix', required=True,
                       help='Prefix for output files (e.g., sample1_DAPI)')
    parser.add_argument('--model-dir', required=True,
                       help='Directory containing StarDist model')

    # Optional arguments
    parser.add_argument('--model-name', default='2D_versatile_fluo',
                       help='StarDist model name')
    parser.add_argument('--max-dim', type=int, default=2048,
                       help='Maximum image dimension for processing')
    parser.add_argument('--n-tiles', type=int, nargs=2, default=[24, 24],
                       metavar=('Y', 'X'),
                       help='Number of tiles for StarDist processing (Y X)')
    parser.add_argument('--pmin', type=float, default=1.0,
                       help='Lower percentile for normalization')
    parser.add_argument('--pmax', type=float, default=99.8,
                       help='Upper percentile for normalization')
    parser.add_argument('--max-nucleus-distance', type=float, default=50.0,
                       help='Maximum distance (pixels) for nucleus matching')

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        process_registered_image(
            reference_path=args.reference,
            registered_path=args.registered,
            output_prefix=args.output_prefix,
            model_dir=args.model_dir,
            model_name=args.model_name,
            max_dim=args.max_dim,
            n_tiles=tuple(args.n_tiles),
            pmin=args.pmin,
            pmax=args.pmax,
            max_nucleus_distance=args.max_nucleus_distance
        )
        return 0

    except Exception as e:
        log_progress(f"ERROR: Segmentation overlap estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

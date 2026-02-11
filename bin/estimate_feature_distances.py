#!/usr/bin/env python3
"""Estimate feature-based distances BEFORE and AFTER registration.

This script processes a single moving/registered image pair to compute feature-based
alignment quality metrics both before and after registration. It provides a clear view
of registration improvement for individual images.

Metrics computed:
- Feature match counts (before vs after)
- Match ratios (before vs after)
- Mean descriptor distances (before vs after)
- Target Registration Error (TRE) statistics via feature distances
- Improvement percentages

This provides complementary quality assessment to segmentation-based overlap metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# Import from shared library modules
from logger import get_logger
from image_utils import load_image_grayscale
from registration_utils import (
    build_feature_detector,
    build_feature_matcher,
    match_feature_points,
)

logger = get_logger(__name__)

__all__ = ["main"]


def log_progress(message: str) -> None:
    """Compatibility wrapper for existing progress output."""
    logger.info(message)

# Disable numba caching
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

def compute_feature_distances(
    ref_kp: np.ndarray,
    mov_kp: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """Compute distance statistics between matched keypoints.

    Parameters
    ----------
    ref_kp : np.ndarray
        Reference keypoints (N, 2)
    mov_kp : np.ndarray
        Moving/registered keypoints (N, 2)

    Returns
    -------
    distances : np.ndarray
        Euclidean distances for each keypoint pair
    statistics : dict
        Statistical summary of the distances
    """
    # Compute Euclidean distances
    distances = np.linalg.norm(ref_kp - mov_kp, axis=1)

    # Compute statistics
    stats = {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "q25": float(np.percentile(distances, 25)),
        "q75": float(np.percentile(distances, 75)),
        "q90": float(np.percentile(distances, 90)),
        "q95": float(np.percentile(distances, 95)),
        "q99": float(np.percentile(distances, 99)),
        "n_points": int(len(distances))
    }

    return distances, stats


def save_distance_histogram(
    before_distances: np.ndarray,
    after_distances: np.ndarray,
    output_path: str,
    output_prefix: str
) -> None:
    """Save comparison histogram of feature distances before and after registration.

    Parameters
    ----------
    before_distances : np.ndarray
        Distances before registration
    after_distances : np.ndarray
        Distances after registration
    output_path : str
        Path to save the histogram
    output_prefix : str
        Prefix for the title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Before registration
    ax1.hist(before_distances, bins=50, edgecolor='black', alpha=0.7, color='red')
    ax1.set_xlabel('Feature Distance (pixels)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Before Registration', fontsize=14)
    ax1.grid(True, alpha=0.3)

    mean_before = np.mean(before_distances)
    median_before = np.median(before_distances)
    stats_text = f'Mean: {mean_before:.2f}px\nMedian: {median_before:.2f}px'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

    # After registration
    ax2.hist(after_distances, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Feature Distance (pixels)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('After Registration', fontsize=14)
    ax2.grid(True, alpha=0.3)

    mean_after = np.mean(after_distances)
    median_after = np.median(after_distances)
    improvement = ((mean_before - mean_after) / mean_before * 100) if mean_before > 0 else 0
    stats_text = f'Mean: {mean_after:.2f}px\nMedian: {median_after:.2f}px\nImprovement: {improvement:.1f}%'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

    fig.suptitle(f'Feature Distance Distribution - {output_prefix}', fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    log_progress(f"  Saved histogram: {output_path}")


def process_image_pair(
    reference_path: str,
    moving_path: str,
    registered_path: str,
    output_prefix: str,
    detector_type: str = "superpoint",
    max_dim: Optional[int] = 2048,
    n_features: int = 5000
) -> None:
    """Process a single moving/registered image pair against a reference.

    Parameters
    ----------
    reference_path : str
        Path to reference image
    moving_path : str
        Path to moving image (before registration)
    registered_path : str
        Path to registered image (after registration)
    output_prefix : str
        Prefix for output files (e.g., 'sample1_DAPI')
    detector_type : str
        Feature detector type (superpoint, disk, dedode, brisk)
    max_dim : int, optional
        Maximum image dimension for downsampling
    n_features : int
        Number of features to keep
    """
    log_progress("=" * 70)
    log_progress("FEATURE DISTANCE ESTIMATION (Before vs After Registration)")
    log_progress("=" * 70)

    # Initialize detector and matcher
    log_progress(f"\n[1/5] Initializing feature detector: {detector_type}")
    detector = build_feature_detector(detector_type, logger=logger)
    matcher = build_feature_matcher(detector_type, logger=logger)

    # Load and process reference image
    log_progress(f"\n[2/5] Processing reference image...")
    log_progress(f"  Reference: {Path(reference_path).name}")
    log_progress(f"  Extracting DAPI channel (channel 0) for feature detection...")
    reference_img = load_image_grayscale(reference_path, max_dim=max_dim)

    log_progress("  Detecting reference features...")
    ref_kp, ref_desc = detector.detect_and_compute(reference_img)
    log_progress(f"    Detected {len(ref_kp)} keypoints")

    # Load moving image (BEFORE registration)
    log_progress(f"\n[3/5] Processing moving image (BEFORE registration)...")
    log_progress(f"  Moving: {Path(moving_path).name}")
    log_progress(f"  Extracting DAPI channel (channel 0) for feature detection...")
    moving_img = load_image_grayscale(moving_path, max_dim=max_dim)

    # Detect features in moving image
    log_progress("  Detecting features in moving image...")
    mov_kp, mov_desc = detector.detect_and_compute(moving_img)
    log_progress(f"    Detected {len(mov_kp)} keypoints")

    # Match BEFORE registration
    log_progress("  Matching features (BEFORE registration)...")
    _, filtered_match_info_before, _, _ = match_feature_points(
        matcher,
        reference_img,
        ref_desc,
        ref_kp,
        moving_img,
        mov_desc,
        mov_kp,
    )

    n_matches_before = filtered_match_info_before.n_matches if hasattr(filtered_match_info_before, 'n_matches') else len(filtered_match_info_before.matched_kp1_xy)
    mean_desc_distance_before = filtered_match_info_before.distance if hasattr(filtered_match_info_before, 'distance') else 0.0

    log_progress(f"    Matches: {n_matches_before}")

    # Compute pixel distances BEFORE
    before_distances, before_stats = compute_feature_distances(
        filtered_match_info_before.matched_kp1_xy,
        filtered_match_info_before.matched_kp2_xy
    )
    log_progress(f"    Mean distance: {before_stats['mean']:.2f} pixels")

    # Load registered image (AFTER registration)
    log_progress(f"\n[4/5] Processing registered image (AFTER registration)...")
    log_progress(f"  Registered: {Path(registered_path).name}")
    log_progress(f"  Extracting DAPI channel (channel 0) for feature detection...")
    registered_img = load_image_grayscale(registered_path, max_dim=max_dim)

    # Detect features in registered image
    log_progress("  Detecting features in registered image...")
    reg_kp, reg_desc = detector.detect_and_compute(registered_img)
    log_progress(f"    Detected {len(reg_kp)} keypoints")

    # Match AFTER registration
    log_progress("  Matching features (AFTER registration)...")
    _, filtered_match_info_after, _, _ = match_feature_points(
        matcher,
        reference_img,
        ref_desc,
        ref_kp,
        registered_img,
        reg_desc,
        reg_kp,
    )

    n_matches_after = filtered_match_info_after.n_matches if hasattr(filtered_match_info_after, 'n_matches') else len(filtered_match_info_after.matched_kp1_xy)
    mean_desc_distance_after = filtered_match_info_after.distance if hasattr(filtered_match_info_after, 'distance') else 0.0

    log_progress(f"    Matches: {n_matches_after}")

    # Compute pixel distances AFTER (TRE)
    after_distances, after_stats = compute_feature_distances(
        filtered_match_info_after.matched_kp1_xy,
        filtered_match_info_after.matched_kp2_xy
    )
    log_progress(f"    Mean TRE: {after_stats['mean']:.2f} pixels")

    # Compute improvement metrics
    log_progress(f"\n[5/5] Computing improvement metrics...")
    improvement = {
        "distance_reduction_pixels": float(before_stats['mean'] - after_stats['mean']),
        "distance_reduction_percent": float(((before_stats['mean'] - after_stats['mean']) / before_stats['mean'] * 100) if before_stats['mean'] > 0 else 0),
        "match_count_increase": int(n_matches_after - n_matches_before),
        "descriptor_distance_decrease": float(mean_desc_distance_before - mean_desc_distance_after)
    }
    log_progress(f"  Distance reduction: {improvement['distance_reduction_pixels']:.2f} pixels ({improvement['distance_reduction_percent']:.1f}%)")
    log_progress(f"  Match count change: {improvement['match_count_increase']:+d}")

    # Save histogram
    histogram_path = f"{output_prefix}_distance_histogram.png"
    save_distance_histogram(before_distances, after_distances, histogram_path, output_prefix)

    # Compile results
    results = {
        "reference_image": Path(reference_path).name,
        "moving_image": Path(moving_path).name,
        "registered_image": Path(registered_path).name,
        "detector_type": detector_type,
        "max_dim": max_dim,
        "n_features": n_features,
        "before_registration": {
            "n_keypoints": len(mov_kp),
            "n_matches": int(n_matches_before),
            "match_ratio": float(n_matches_before) / float(min(len(ref_kp), len(mov_kp))) if min(len(ref_kp), len(mov_kp)) > 0 else 0.0,
            "mean_descriptor_distance": float(mean_desc_distance_before),
            "feature_distances": before_stats
        },
        "after_registration": {
            "n_keypoints": len(reg_kp),
            "n_matches": int(n_matches_after),
            "match_ratio": float(n_matches_after) / float(min(len(ref_kp), len(reg_kp))) if min(len(ref_kp), len(reg_kp)) > 0 else 0.0,
            "mean_descriptor_distance": float(mean_desc_distance_after),
            "feature_distances": after_stats
        },
        "improvement": improvement
    }

    # Save JSON results
    output_path = f"{output_prefix}_feature_distances.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    log_progress(f"\n  Saved results: {output_path}")
    log_progress(f"    Mean TRE: {after_stats['mean']:.2f} pixels")
    log_progress(f"    Improvement: {improvement['distance_reduction_percent']:.1f}%")

    log_progress("\n" + "=" * 70)
    log_progress("✓ Feature distance estimation complete!")
    log_progress("=" * 70)




def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimate feature distances before and after registration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--reference', required=True,
                       help='Path to reference image')
    parser.add_argument('--moving', required=True,
                       help='Path to moving image (before registration)')
    parser.add_argument('--registered', required=True,
                       help='Path to registered image (after registration)')
    parser.add_argument('--output-prefix', required=True,
                       help='Prefix for output files (e.g., sample1_DAPI)')
    parser.add_argument('--detector', default='superpoint',
                       choices=['superpoint', 'disk', 'dedode', 'brisk'],
                       help='Feature detector type')
    parser.add_argument('--max-dim', type=int, default=2048,
                       help='Maximum image dimension for processing')
    parser.add_argument('--n-features', type=int, default=5000,
                       help='Number of features to keep')

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        process_image_pair(
            reference_path=args.reference,
            moving_path=args.moving,
            registered_path=args.registered,
            output_prefix=args.output_prefix,
            detector_type=args.detector,
            max_dim=args.max_dim,
            n_features=args.n_features
        )
        return 0

    except Exception as e:
        log_progress(f"ERROR: Feature distance estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

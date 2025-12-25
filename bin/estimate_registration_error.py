#!/usr/bin/env python3
"""Estimate registration error using pre-computed features.

This script re-detects and matches features between registered images and reference,
then compares them with pre-registration features to compute registration error metrics.

The script provides multiple error metrics:
1. Target Registration Error (TRE): Euclidean distance between matched keypoints
2. Keypoint displacement statistics (mean, median, std, max)
3. Feature matching improvement metrics
4. Spatial error distribution (quartiles)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvips
import matplotlib.pyplot as plt

# Disable numba caching
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

from valis import feature_detectors
from valis import feature_matcher


def log_progress(message: str) -> None:
    """Print timestamped progress messages."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_image_grayscale(image_path: str, max_dim: Optional[int] = None) -> np.ndarray:
    """Load image and convert to grayscale for feature detection.

    Parameters
    ----------
    image_path : str
        Path to the image file
    max_dim : int, optional
        Maximum dimension for downsampling

    Returns
    -------
    np.ndarray
        Grayscale image as numpy array (H, W)
    """
    vips_img = pyvips.Image.new_from_file(image_path)

    # Downsample if needed
    if max_dim is not None:
        current_max = max(vips_img.width, vips_img.height)
        if current_max > max_dim:
            scale = max_dim / current_max
            vips_img = vips_img.resize(scale)

    # Convert to grayscale if multi-channel
    if vips_img.bands > 1:
        vips_img = vips_img.extract_band(0)

    # Convert to numpy
    img_array = vips_img.numpy()

    # Ensure 8-bit
    if img_array.dtype != np.uint8:
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)

    return img_array


def get_feature_detector(detector_type: str = "superpoint"):
    """Get feature detector instance."""
    detector_type = detector_type.lower()

    if detector_type == "superpoint":
        return feature_detectors.SuperPointFD()
    elif detector_type == "disk":
        return feature_detectors.DiskFD()
    elif detector_type == "dedode":
        return feature_detectors.DeDoDeFD()
    elif detector_type == "brisk":
        return feature_detectors.BriskFD()
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def get_feature_matcher(detector_type: str = "superpoint"):
    """Get feature matcher instance."""
    detector_type = detector_type.lower()

    if detector_type == "superpoint":
        return feature_matcher.SuperGlueMatcher()
    elif detector_type in ["disk", "dedode"]:
        return feature_matcher.LightGlueMatcher()
    else:
        return feature_matcher.Matcher(
            match_filter_method='USAC_MAGSAC',
            ransac_thresh=7
        )


def compute_target_registration_error(
    ref_kp: np.ndarray,
    reg_kp: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """Compute Target Registration Error (TRE) between matched keypoints.

    TRE is the Euclidean distance between corresponding keypoints after registration.

    Parameters
    ----------
    ref_kp : np.ndarray
        Reference keypoints (N, 2)
    reg_kp : np.ndarray
        Registered (transformed) keypoints (N, 2)

    Returns
    -------
    distances : np.ndarray
        Euclidean distances for each keypoint pair
    statistics : dict
        Statistical summary of the errors
    """
    # Compute Euclidean distances
    distances = np.linalg.norm(ref_kp - reg_kp, axis=1)

    # Compute statistics
    stats = {
        "mean_error": float(np.mean(distances)),
        "median_error": float(np.median(distances)),
        "std_error": float(np.std(distances)),
        "min_error": float(np.min(distances)),
        "max_error": float(np.max(distances)),
        "q25_error": float(np.percentile(distances, 25)),
        "q75_error": float(np.percentile(distances, 75)),
        "q90_error": float(np.percentile(distances, 90)),
        "q95_error": float(np.percentile(distances, 95)),
        "q99_error": float(np.percentile(distances, 99)),
        "n_points": int(len(distances))
    }

    return distances, stats


def save_error_histogram(distances: np.ndarray, output_path: str, moving_name: str):
    """Save histogram of registration errors.

    Parameters
    ----------
    distances : np.ndarray
        Registration errors (TRE)
    output_path : str
        Path to save the histogram
    moving_name : str
        Name of the moving image for the title
    """
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Target Registration Error (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Registration Error Distribution - {moving_name}', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    mean_err = np.mean(distances)
    median_err = np.median(distances)
    std_err = np.std(distances)

    stats_text = f'Mean: {mean_err:.2f}px\nMedian: {median_err:.2f}px\nStd: {std_err:.2f}px'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log_progress(f"  Saved error histogram: {output_path}")


def estimate_registration_error(
    reference_path: str,
    registered_path: str,
    pre_features_path: str,
    output_dir: str,
    detector_type: str = "superpoint",
    max_dim: Optional[int] = 2048,
    n_features: int = 5000
) -> Tuple[dict, str]:
    """Estimate registration error using direct feature alignment measurement.

    This method uses a simplified, robust approach:
    1. Load pre-registration feature matches (baseline)
    2. Re-detect and match features after registration
    3. Compute direct residual distances (TRE)
    4. Compare pre vs post to quantify improvement

    No transformation matrix estimation is performed, avoiding circular dependencies
    and estimation errors. This provides a clean, independent validation metric.

    Parameters
    ----------
    reference_path : str
        Path to reference image
    registered_path : str
        Path to registered (transformed) moving image
    pre_features_path : str
        Path to pre-registration feature matching JSON
    output_dir : str
        Directory to save error estimation results
    detector_type : str
        Feature detector type
    max_dim : int, optional
        Maximum image dimension
    n_features : int
        Number of features to keep

    Returns
    -------
    results_dict : dict
        Dictionary containing error statistics
    output_path : str
        Path to saved results JSON file
    """
    log_progress("=" * 70)
    log_progress("REGISTRATION ERROR ESTIMATION (Feature-Based)")
    log_progress("=" * 70)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load pre-registration features
    log_progress("\n[1/4] Loading pre-registration feature data...")
    with open(pre_features_path, 'r') as f:
        pre_features = json.load(f)

    # Handle both old and new JSON format
    n_matches_pre = pre_features.get('n_matches_filtered', pre_features.get('n_matches', 0))
    log_progress(f"  Pre-registration matches: {n_matches_pre}")
    log_progress(f"  Pre-registration match ratio: {pre_features['match_ratio']:.2%}")

    # Load images
    log_progress("\n[2/4] Loading images...")
    ref_img = load_image_grayscale(reference_path, max_dim=max_dim)
    reg_img = load_image_grayscale(registered_path, max_dim=max_dim)

    # Initialize detector and matcher
    log_progress("\n[3/4] Initializing feature detector and matcher...")
    detector = get_feature_detector(detector_type)
    matcher = get_feature_matcher(detector_type)

    # Detect features in registered images using correct VALIS API
    log_progress("\n[4/4] Detecting and matching features after registration...")
    log_progress(f"  Reference image: {os.path.basename(reference_path)}")
    ref_kp, ref_desc = detector.detect_and_compute(ref_img, mask=None)
    log_progress(f"    Detected {len(ref_kp)} keypoints")

    log_progress(f"  Registered image: {os.path.basename(registered_path)}")
    reg_kp, reg_desc = detector.detect_and_compute(reg_img, mask=None)
    log_progress(f"    Detected {len(reg_kp)} keypoints")

    # Note: detect_and_compute already filters to MAX_FEATURES (20000) internally
    # Additional filtering is not needed as these are already numpy arrays

    # Match features using correct VALIS API
    log_progress("  Matching features...")

    # Check if this is SuperGlue or LightGlue matcher (needs images)
    if isinstance(matcher, feature_matcher.SuperGlueMatcher) or \
       (hasattr(feature_matcher, 'LightGlueMatcher') and isinstance(matcher, feature_matcher.LightGlueMatcher)):
        # SuperGlue and LightGlue need the images
        match_info12, filtered_match_info, match_info21, filtered_match_info21 = matcher.match_images(
            img1=ref_img,
            desc1=ref_desc,
            kp1_xy=ref_kp,
            img2=reg_img,
            desc2=reg_desc,
            kp2_xy=reg_kp
        )
    else:
        # Standard Matcher only needs descriptors and keypoints
        match_info12, filtered_match_info, match_info21, filtered_match_info21 = matcher.match_images(
            desc1=ref_desc,
            kp1_xy=ref_kp,
            desc2=reg_desc,
            kp2_xy=reg_kp
        )

    n_matches_post = filtered_match_info.n_matches if hasattr(filtered_match_info, 'n_matches') else len(filtered_match_info.matched_kp1_xy)
    mean_distance_post = filtered_match_info.distance if hasattr(filtered_match_info, 'distance') else 0.0

    log_progress(f"  Post-registration matches: {n_matches_post}")
    log_progress(f"  Mean descriptor distance: {mean_distance_post:.4f}")

    # Compute post-registration residual error (PRIMARY METRIC)
    log_progress("\nComputing Post-Registration Feature Alignment (TRE)...")

    post_reg_stats = {}
    tre_distances = None

    if hasattr(filtered_match_info, 'matched_kp1_xy') and hasattr(filtered_match_info, 'matched_kp2_xy'):
        matched_ref_kp = filtered_match_info.matched_kp1_xy
        matched_reg_kp = filtered_match_info.matched_kp2_xy

        # Direct residual distance = Target Registration Error
        tre_distances, post_reg_stats = compute_target_registration_error(matched_ref_kp, matched_reg_kp)

        log_progress(f"  Mean TRE: {post_reg_stats['mean_error']:.2f} pixels")
        log_progress(f"  Median TRE: {post_reg_stats['median_error']:.2f} pixels")
        log_progress(f"  Std TRE: {post_reg_stats['std_error']:.2f} pixels")
        log_progress(f"  95th percentile TRE: {post_reg_stats['q95_error']:.2f} pixels")

    # Compute pre-registration baseline distances
    log_progress("\nComputing Pre-Registration Baseline Distances...")

    pre_reg_stats = {}

    if 'matched_kp_reference' in pre_features and 'matched_kp_moving' in pre_features:
        # Load original matched keypoints
        ref_kp_original = np.array(pre_features['matched_kp_reference'])
        mov_kp_original = np.array(pre_features['matched_kp_moving'])

        # Compute original misalignment (before registration)
        original_distances = np.linalg.norm(ref_kp_original - mov_kp_original, axis=1)

        pre_reg_stats = {
            "mean_error": float(np.mean(original_distances)),
            "median_error": float(np.median(original_distances)),
            "std_error": float(np.std(original_distances)),
            "min_error": float(np.min(original_distances)),
            "max_error": float(np.max(original_distances)),
            "q25_error": float(np.percentile(original_distances, 25)),
            "q75_error": float(np.percentile(original_distances, 75)),
            "q90_error": float(np.percentile(original_distances, 90)),
            "q95_error": float(np.percentile(original_distances, 95)),
            "q99_error": float(np.percentile(original_distances, 99)),
            "n_points": int(len(original_distances))
        }

        log_progress(f"  Mean baseline distance: {pre_reg_stats['mean_error']:.2f} pixels")
        log_progress(f"  Median baseline distance: {pre_reg_stats['median_error']:.2f} pixels")

        # Compute improvement
        if post_reg_stats:
            improvement = pre_reg_stats['mean_error'] - post_reg_stats['mean_error']
            improvement_pct = (improvement / pre_reg_stats['mean_error']) * 100 if pre_reg_stats['mean_error'] > 0 else 0

            log_progress(f"\n  Registration Improvement:")
            log_progress(f"    Absolute: {improvement:.2f} pixels")
            log_progress(f"    Relative: {improvement_pct:.1f}%")

            # Save TRE histogram
            if tre_distances is not None:
                moving_basename = Path(registered_path).stem
                histogram_path = os.path.join(output_dir, f"{moving_basename}_tre_histogram.png")
                save_error_histogram(tre_distances, histogram_path, moving_basename)

    else:
        log_progress("  WARNING: Pre-registration matched keypoints not available")
        log_progress("           Cannot compute baseline for comparison")

    # Prepare comprehensive results
    moving_basename = os.path.basename(registered_path)
    results = {
        "reference_image": os.path.basename(reference_path),
        "registered_image": moving_basename,
        "detector_type": detector_type,
        "max_dim": max_dim,

        # Pre-registration statistics
        "pre_registration": {
            "n_matches": pre_features.get('n_matches_filtered', pre_features.get('n_matches', 0)),
            "match_ratio": pre_features['match_ratio'],
            "mean_descriptor_distance": pre_features.get('mean_descriptor_distance', 0.0)
        },

        # Post-registration statistics
        "post_registration": {
            "n_matches": int(n_matches_post),
            "match_ratio": float(n_matches_post) / float(min(len(ref_kp), len(reg_kp))) if min(len(ref_kp), len(reg_kp)) > 0 else 0.0,
            "mean_descriptor_distance": float(mean_distance_post)
        },

        # Baseline distances (before registration)
        "baseline_distances": pre_reg_stats,

        # Target Registration Error (post-registration residual alignment)
        "target_registration_error": post_reg_stats,

        # Improvement metrics
        "improvement": {
            "match_count_increase": int(n_matches_post) - pre_features.get('n_matches_filtered', pre_features.get('n_matches', 0)),
            "match_ratio_increase": (float(n_matches_post) / float(min(len(ref_kp), len(reg_kp))) if min(len(ref_kp), len(reg_kp)) > 0 else 0.0) - pre_features['match_ratio'],
            "descriptor_distance_decrease": pre_features.get('mean_descriptor_distance', 0.0) - float(mean_distance_post),
            "tre_improvement_pixels": pre_reg_stats.get('mean_error', 0.0) - post_reg_stats.get('mean_error', 0.0) if pre_reg_stats and post_reg_stats else None,
            "tre_improvement_percent": ((pre_reg_stats.get('mean_error', 0.0) - post_reg_stats.get('mean_error', 0.0)) / pre_reg_stats.get('mean_error', 1.0) * 100) if pre_reg_stats and post_reg_stats and pre_reg_stats.get('mean_error', 0.0) > 0 else None
        }
    }

    # Save results
    output_filename = f"{Path(moving_basename).stem}_registration_error.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    log_progress(f"\n✓ Results saved to: {output_path}")
    log_progress("=" * 70)

    return results, output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimate registration error using feature matching'
    )
    parser.add_argument('--reference', required=True, help='Path to reference image')
    parser.add_argument('--registered', required=True, help='Path to registered image')
    parser.add_argument('--pre-features', required=True, help='Path to pre-registration features JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--detector', default='superpoint',
                       choices=['superpoint', 'disk', 'dedode', 'brisk', 'vgg'],
                       help='Feature detector type (default: superpoint)')
    parser.add_argument('--max-dim', type=int, default=2048,
                       help='Maximum image dimension (default: 2048)')
    parser.add_argument('--n-features', type=int, default=5000,
                       help='Number of features to keep (default: 5000)')
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        results, output_path = estimate_registration_error(
            args.reference,
            args.registered,
            args.pre_features,
            args.output_dir,
            args.detector,
            args.max_dim,
            args.n_features
        )

        log_progress(f"\nError estimation complete!")
        log_progress("=" * 70)

        if "mean_error" in results.get("baseline_distances", {}):
            log_progress(f"  Baseline distance (before reg): {results['baseline_distances']['mean_error']:.2f} pixels")

        if "mean_error" in results.get("target_registration_error", {}):
            log_progress(f"  Mean TRE (after reg):           {results['target_registration_error']['mean_error']:.2f} pixels")
            log_progress(f"  Median TRE:                     {results['target_registration_error']['median_error']:.2f} pixels")

        if results['improvement'].get('tre_improvement_pixels') is not None:
            log_progress(f"  Improvement:                    {results['improvement']['tre_improvement_pixels']:.2f} pixels ({results['improvement']['tre_improvement_percent']:.1f}%)")

        log_progress(f"  Post-registration matches:      {results['post_registration']['n_matches']}")
        log_progress(f"\n  Results saved: {output_path}")
        log_progress("=" * 70)

        return 0

    except Exception as e:
        log_progress(f"ERROR: Error estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

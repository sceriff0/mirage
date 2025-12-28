#!/usr/bin/env python3
"""Compute and match features between moving image and reference using VALIS.

This script uses VALIS's feature detection and matching capabilities to:
1. Detect features in both reference and moving images
2. Match features between the two images
3. Save the matched features and metadata for downstream analysis

The script supports SuperPoint, DISK, and DeDoDe feature detectors with
appropriate matchers (SuperGlue, LightGlue).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvips
import tifffile

# Disable numba caching
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_CACHING'] = '1'

# Import from lib modules for DRY principle
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from logger import log_progress
from image_utils import load_image_grayscale

from valis import feature_detectors
from valis import feature_matcher

# Function definitions removed - now imported from lib modules:
# - log_progress() -> imported from lib.logger
# - load_image_grayscale() -> imported from lib.image_utils


def get_feature_detector(detector_type: str = "superpoint"):
    """Get feature detector instance.

    Parameters
    ----------
    detector_type : str
        Type of detector: 'superpoint', 'disk', 'dedode', or 'brisk'

    Returns
    -------
    Feature detector instance (FeatureDD subclass)
    """
    detector_type = detector_type.lower()

    if detector_type == "superpoint":
        log_progress("Initializing SuperPoint feature detector")
        return feature_detectors.SuperPointFD()
    elif detector_type == "disk":
        log_progress("Initializing DISK feature detector")
        return feature_detectors.DiskFD()
    elif detector_type == "dedode":
        log_progress("Initializing DeDoDe feature detector")
        return feature_detectors.DeDoDeFD()
    elif detector_type == "brisk":
        log_progress("Initializing BRISK feature detector")
        return feature_detectors.BriskFD()
    elif detector_type == "vgg":
        log_progress("Initializing VGG feature detector")
        return feature_detectors.VggFD()
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def get_feature_matcher(detector_type: str = "superpoint"):
    """Get feature matcher instance compatible with detector.

    Parameters
    ----------
    detector_type : str
        Type of detector (determines matcher)

    Returns
    -------
    Feature matcher instance (Matcher, SuperGlueMatcher, or LightGlueMatcher)
    """
    detector_type = detector_type.lower()

    if detector_type == "superpoint":
        log_progress("Initializing SuperGlue matcher")
        return feature_matcher.SuperGlueMatcher()
    elif detector_type in ["disk", "dedode"]:
        log_progress("Initializing LightGlue matcher")
        return feature_matcher.LightGlueMatcher()
    else:
        log_progress("Initializing standard Matcher with USAC_MAGSAC filter")
        return feature_matcher.Matcher(
            match_filter_method='USAC_MAGSAC',
            ransac_thresh=7
        )


def compute_and_match_features(
    reference_path: str,
    moving_path: str,
    output_dir: str,
    detector_type: str = "superpoint",
    max_dim: Optional[int] = 2048,
    n_features: int = 5000
) -> Tuple[dict, str]:
    """Compute and match features between reference and moving images.

    Parameters
    ----------
    reference_path : str
        Path to reference image
    moving_path : str
        Path to moving image
    output_dir : str
        Directory to save feature matching results
    detector_type : str
        Feature detector type
    max_dim : int, optional
        Maximum image dimension for feature detection
    n_features : int
        Number of features to keep

    Returns
    -------
    results_dict : dict
        Dictionary containing matching statistics
    output_path : str
        Path to saved results JSON file
    """
    log_progress("=" * 70)
    log_progress("FEATURE DETECTION AND MATCHING")
    log_progress("=" * 70)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load images
    log_progress("\n[1/4] Loading images...")
    ref_img = load_image_grayscale(reference_path, max_dim=max_dim)
    mov_img = load_image_grayscale(moving_path, max_dim=max_dim)

    # Initialize detector and matcher
    log_progress("\n[2/4] Initializing feature detector and matcher...")
    detector = get_feature_detector(detector_type)
    matcher = get_feature_matcher(detector_type)

    # Detect features using the correct VALIS API (detect_and_compute with underscores)
    # Returns: kp_pos_xy (numpy array of shape (N, 2)), desc (numpy array of shape (N, M))
    log_progress("\n[3/4] Detecting features...")
    log_progress(f"  Reference image: {os.path.basename(reference_path)}")
    ref_kp, ref_desc = detector.detect_and_compute(ref_img, mask=None)
    log_progress(f"    Detected {len(ref_kp)} keypoints")

    log_progress(f"  Moving image: {os.path.basename(moving_path)}")
    mov_kp, mov_desc = detector.detect_and_compute(mov_img, mask=None)
    log_progress(f"    Detected {len(mov_kp)} keypoints")

    # Note: detect_and_compute already filters to MAX_FEATURES (20000) internally
    # Additional filtering is not needed as these are already numpy arrays, not cv2.KeyPoint objects

    # Match features using VALIS match_images API
    # Standard Matcher.match_images signature: (desc1, kp1_xy, desc2, kp2_xy, additional_filtering_kwargs)
    # SuperGlueMatcher/LightGlueMatcher signature: (img1, desc1, kp1_xy, img2, desc2, kp2_xy, additional_filtering_kwargs)
    log_progress("\n[4/4] Matching features...")

    # Check if this is SuperGlue or LightGlue matcher (needs images)
    if isinstance(matcher, feature_matcher.SuperGlueMatcher) or \
       (hasattr(feature_matcher, 'LightGlueMatcher') and isinstance(matcher, feature_matcher.LightGlueMatcher)):
        # SuperGlue and LightGlue need the images
        match_info12, filtered_match_info12, match_info21, filtered_match_info21 = matcher.match_images(
            img1=ref_img,
            desc1=ref_desc,
            kp1_xy=ref_kp,
            img2=mov_img,
            desc2=mov_desc,
            kp2_xy=mov_kp
        )
    else:
        # Standard Matcher only needs descriptors and keypoints
        match_info12, filtered_match_info12, match_info21, filtered_match_info21 = matcher.match_images(
            desc1=ref_desc,
            kp1_xy=ref_kp,
            desc2=mov_desc,
            kp2_xy=mov_kp
        )

    # Use the filtered match info (after RANSAC/GMS filtering)
    match_info = filtered_match_info12

    # Extract statistics from MatchInfo object
    n_matches = match_info.n_matches if hasattr(match_info, 'n_matches') else len(match_info.matched_kp1_xy)
    mean_distance = match_info.distance if hasattr(match_info, 'distance') else 0.0

    log_progress(f"  Matches found: {n_matches}")
    log_progress(f"  Mean descriptor distance: {mean_distance:.4f}")

    # Prepare results
    moving_basename = os.path.basename(moving_path)
    results = {
        "reference_image": os.path.basename(reference_path),
        "moving_image": moving_basename,
        "detector_type": detector_type,
        "max_dim": max_dim,
        "n_features_requested": n_features,
        "n_keypoints_reference": len(ref_kp),
        "n_keypoints_moving": len(mov_kp),
        "n_matches": int(n_matches),
        "mean_descriptor_distance": float(mean_distance),
        "match_ratio": float(n_matches) / float(min(len(ref_kp), len(mov_kp))) if min(len(ref_kp), len(mov_kp)) > 0 else 0.0,
    }

    # Save matched keypoint coordinates from MatchInfo object
    if hasattr(match_info, 'matched_kp1_xy') and hasattr(match_info, 'matched_kp2_xy'):
        results["matched_kp_reference"] = match_info.matched_kp1_xy.tolist()
        results["matched_kp_moving"] = match_info.matched_kp2_xy.tolist()

    # Save results to JSON
    output_filename = f"{Path(moving_basename).stem}_features.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    log_progress(f"\n✓ Results saved to: {output_path}")
    log_progress("=" * 70)

    return results, output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute and match features between reference and moving images using VALIS'
    )
    parser.add_argument('--reference', required=True, help='Path to reference image')
    parser.add_argument('--moving', required=True, help='Path to moving image')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--detector', default='superpoint',
                       choices=['superpoint', 'disk', 'dedode', 'brisk', 'vgg'],
                       help='Feature detector type (default: superpoint)')
    parser.add_argument('--max-dim', type=int, default=2048,
                       help='Maximum image dimension for feature detection (default: 2048)')
    parser.add_argument('--n-features', type=int, default=5000,
                       help='Number of features to keep (default: 5000)')
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        results, output_path = compute_and_match_features(
            args.reference,
            args.moving,
            args.output_dir,
            args.detector,
            args.max_dim,
            args.n_features
        )

        log_progress(f"\nFeature matching complete!")
        log_progress(f"  Matches: {results['n_matches']}")
        log_progress(f"  Match ratio: {results['match_ratio']:.2%}")
        log_progress(f"  Results: {output_path}")

        return 0

    except Exception as e:
        log_progress(f"ERROR: Feature matching failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

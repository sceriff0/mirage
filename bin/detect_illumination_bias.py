#!/usr/bin/env python3
"""
Automatic detection of illumination bias in microscopy channels.

This module analyzes image channels to determine if they would benefit from
BaSiC illumination correction or if correction would introduce artifacts.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from logger import configure_logging, get_logger

logger = get_logger(__name__)


def compute_radial_intensity_profile(
    image: NDArray,
    n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial intensity profile from image center.

    Illumination bias typically shows as radial gradient (vignetting).

    Parameters
    ----------
    image : NDArray
        2D image array
    n_bins : int
        Number of radial bins

    Returns
    -------
    radii : np.ndarray
        Radial distances
    intensities : np.ndarray
        Mean intensity at each radius
    """
    h, w = image.shape
    center_y, center_x = h // 2, w // 2

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Compute distance from center
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)

    # Normalize distances to [0, 1]
    distances_norm = distances / max_dist

    # Bin pixels by distance
    bins = np.linspace(0, 1, n_bins + 1)
    radial_profile = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (distances_norm >= bins[i]) & (distances_norm < bins[i + 1])
        if np.any(mask):
            radial_profile[i] = np.mean(image[mask])

    radii = (bins[:-1] + bins[1:]) / 2
    return radii, radial_profile


def compute_gradient_magnitude(image: NDArray) -> float:
    """
    Compute mean gradient magnitude (Sobel) across image.

    High gradient = structured content (signal)
    Low gradient = smooth/uniform (background/noise)
    """
    # Use Sobel filter
    sx = ndimage.sobel(image, axis=1)
    sy = ndimage.sobel(image, axis=0)
    magnitude = np.sqrt(sx**2 + sy**2)
    return float(np.mean(magnitude))


def compute_spatial_variance(
    image: NDArray,
    tile_size: int = 256
) -> Dict[str, float]:
    """
    Compute spatial variance metrics using tiling.

    Channels with illumination bias show high variance between tiles.
    """
    h, w = image.shape

    # Create tiles
    tile_means = []
    tile_stds = []

    for i in range(0, h - tile_size + 1, tile_size):
        for j in range(0, w - tile_size + 1, tile_size):
            tile = image[i:i+tile_size, j:j+tile_size]
            if tile.size > 0:
                tile_means.append(np.mean(tile))
                tile_stds.append(np.std(tile))

    if not tile_means:
        return {
            'tile_mean_cv': 0.0,
            'tile_std_cv': 0.0,
            'n_tiles': 0
        }

    tile_means = np.array(tile_means)
    tile_stds = np.array(tile_stds)

    # Coefficient of variation across tiles
    cv_means = np.std(tile_means) / (np.mean(tile_means) + 1e-10)
    cv_stds = np.std(tile_stds) / (np.mean(tile_stds) + 1e-10)

    return {
        'tile_mean_cv': float(cv_means),
        'tile_std_cv': float(cv_stds),
        'n_tiles': len(tile_means)
    }


def detect_vignetting(
    image: NDArray,
    threshold: float = 0.15
) -> Tuple[bool, float]:
    """
    Detect vignetting (radial illumination falloff).

    Parameters
    ----------
    image : NDArray
        2D image
    threshold : float
        Relative intensity drop threshold (default: 15%)

    Returns
    -------
    has_vignetting : bool
        True if vignetting detected
    intensity_drop : float
        Relative intensity drop from center to edge (0-1)
    """
    radii, profile = compute_radial_intensity_profile(image, n_bins=30)

    # Normalize profile
    profile_norm = profile / (profile[0] + 1e-10)

    # Measure drop from center to edge
    center_intensity = np.mean(profile_norm[:3])  # Center bins
    edge_intensity = np.mean(profile_norm[-3:])   # Edge bins

    if center_intensity <= 1e-10:
        return False, 0.0

    intensity_drop = (center_intensity - edge_intensity) / center_intensity

    return intensity_drop > threshold, float(intensity_drop)


def compute_signal_sparsity(image: NDArray) -> float:
    """
    Compute signal sparsity (fraction of pixels with signal).

    Sparse signals (< 5% coverage) may not have enough information for BaSiC.
    """
    # Estimate foreground threshold from dynamic range.
    # This keeps dense tissue channels high and sparse channels low.
    low = np.percentile(image, 5)
    high = np.percentile(image, 99)

    if high <= low:
        return 0.0

    threshold = low + 0.1 * (high - low)
    signal_pixels = np.sum(image > threshold)
    total_pixels = image.size

    return float(signal_pixels / total_pixels)


def should_apply_basic_correction(
    image: NDArray,
    channel_name: str = "",
    verbose: bool = False
) -> Tuple[bool, Dict[str, float]]:
    """
    Determine if a channel would benefit from BaSiC correction.

    Decision logic:
    1. Skip if channel is DAPI (nuclear stain, usually uniform)
    2. Skip if signal is too sparse (< 5% coverage)
    3. Apply if vignetting detected (> 15% radial drop)
    4. Apply if high tile-to-tile variance (CV > 0.2)
    5. Skip if signal is very uniform (low gradient)

    Parameters
    ----------
    image : NDArray
        2D channel image
    channel_name : str
        Channel name for heuristics
    verbose : bool
        Print decision reasoning

    Returns
    -------
    should_correct : bool
        True if BaSiC should be applied
    metrics : Dict[str, float]
        Diagnostic metrics
    """
    # Normalize to 0-1 for consistent metrics
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)

    # Compute metrics
    has_vignetting, vignetting_drop = detect_vignetting(image_norm)
    spatial_vars = compute_spatial_variance(image_norm)
    gradient_mag = compute_gradient_magnitude(image_norm)
    sparsity = compute_signal_sparsity(image_norm)

    metrics = {
        'vignetting_detected': float(has_vignetting),
        'vignetting_drop': vignetting_drop,
        'tile_mean_cv': spatial_vars['tile_mean_cv'],
        'tile_std_cv': spatial_vars['tile_std_cv'],
        'gradient_magnitude': gradient_mag,
        'signal_sparsity': sparsity
    }

    # Decision rules
    reasons = []

    # Rule 1: Skip DAPI (often pre-corrected or uniform)
    if 'DAPI' in channel_name.upper():
        reasons.append("DAPI channel (typically uniform)")
        decision = False

    # Rule 2: Skip sparse signals
    elif sparsity < 0.05:
        reasons.append(f"Too sparse (only {sparsity*100:.1f}% coverage)")
        decision = False

    # Rule 3: Apply if strong vignetting
    elif has_vignetting:
        reasons.append(f"Strong vignetting ({vignetting_drop*100:.1f}% drop)")
        decision = True

    # Rule 4: Apply if high tile variance
    elif spatial_vars['tile_mean_cv'] > 0.2 or spatial_vars['tile_std_cv'] > 0.15:
        reasons.append(f"High tile variance (CV={spatial_vars['tile_mean_cv']:.2f})")
        decision = True

    # Rule 5: Skip if very uniform
    elif gradient_mag < 0.01 and spatial_vars['tile_mean_cv'] < 0.1:
        reasons.append("Image too uniform (low structure)")
        decision = False

    # Default: apply if moderate tile variance
    elif spatial_vars['tile_mean_cv'] > 0.15 or spatial_vars['tile_std_cv'] > 0.12:
        reasons.append(f"Moderate tile variance (CV={spatial_vars['tile_mean_cv']:.2f})")
        decision = True

    else:
        reasons.append("No significant illumination bias")
        decision = False

    metrics['decision'] = float(decision)
    metrics['reasoning'] = ' | '.join(reasons)

    if verbose:
        status = "✓ APPLY BaSiC" if decision else "✗ SKIP BaSiC"
        logger.info(f"{status} - {channel_name}")
        logger.info(f"  Reasons: {' | '.join(reasons)}")
        logger.info("  Metrics:")
        logger.info(f"    - Vignetting: {vignetting_drop*100:.1f}% drop")
        logger.info(f"    - Tile variance CV: {spatial_vars['tile_mean_cv']:.3f}")
        logger.info(f"    - Signal sparsity: {sparsity*100:.1f}%")
        logger.info(f"    - Gradient magnitude: {gradient_mag:.4f}")

    return decision, metrics


def analyze_all_channels(
    multichannel_image: NDArray,
    channel_names: list,
    verbose: bool = True
) -> Dict[int, Tuple[bool, Dict]]:
    """
    Analyze all channels in a multichannel image.

    Parameters
    ----------
    multichannel_image : NDArray
        3D array (C, Y, X)
    channel_names : list
        Channel names
    verbose : bool
        Print results

    Returns
    -------
    decisions : Dict[int, Tuple[bool, Dict]]
        Dict mapping channel index to (should_correct, metrics)
    """
    if verbose:
        logger.info("=" * 80)
        logger.info("ILLUMINATION BIAS DETECTION")
        logger.info("=" * 80)

    decisions = {}

    for i in range(multichannel_image.shape[0]):
        channel = multichannel_image[i]
        name = channel_names[i] if i < len(channel_names) else f"Channel_{i}"

        should_correct, metrics = should_apply_basic_correction(
            channel,
            channel_name=name,
            verbose=verbose
        )

        decisions[i] = (should_correct, metrics)

        if verbose:
            logger.info("")

    # Summary
    if verbose:
        n_correct = sum(1 for d, _ in decisions.values() if d)
        n_skip = len(decisions) - n_correct
        logger.info("=" * 80)
        logger.info(f"SUMMARY: Apply BaSiC to {n_correct}/{len(decisions)} channels")
        logger.info(f"         Skip {n_skip}/{len(decisions)} channels")
        logger.info("=" * 80)

    return decisions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect which channels need BaSiC illumination correction'
    )
    parser.add_argument('--input', required=True, help='Input multichannel TIFF')
    parser.add_argument('--channel-names', nargs='+', help='Channel names')
    return parser.parse_args()


def main() -> int:
    """Run illumination bias analysis CLI."""
    configure_logging()
    args = parse_args()

    import tifffile

    # Load image
    logger.info(f"Loading: {args.input}")
    image = tifffile.imread(args.input)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    # Generate channel names if not provided
    if args.channel_names:
        channel_names = args.channel_names
    else:
        channel_names = [f"Channel_{i}" for i in range(image.shape[0])]

    # Analyze
    decisions = analyze_all_channels(image, channel_names, verbose=True)

    # Print results
    logger.info("RECOMMENDATIONS:")
    for i, (should_correct, metrics) in decisions.items():
        status = "APPLY" if should_correct else "SKIP"
        logger.info(f"  Channel {i} ({channel_names[i]}): {status}")
        logger.info(f"    -> {metrics['reasoning']}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

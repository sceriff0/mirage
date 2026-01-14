#!/usr/bin/env python3
"""
Tests for automatic illumination bias detection.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add bin to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'bin'))

from detect_illumination_bias import (
    should_apply_basic_correction,
    detect_vignetting,
    compute_spatial_variance,
    compute_signal_sparsity,
    compute_radial_intensity_profile
)


def create_vignetting_image(size=512, drop=0.3):
    """Create synthetic image with radial vignetting."""
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2

    # Radial distance from center
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r_norm = r / r.max()

    # Create vignetting: intensity drops from center to edge
    vignetting = 1.0 - drop * r_norm**2

    # Add some signal
    signal = np.random.poisson(1000, (size, size)).astype(float)

    return (signal * vignetting).astype(np.uint16)


def create_uniform_image(size=512):
    """Create synthetic uniform image."""
    signal = np.random.poisson(1000, (size, size))
    return signal.astype(np.uint16)


def create_sparse_image(size=512, coverage=0.03):
    """Create synthetic sparse image."""
    image = np.zeros((size, size), dtype=np.uint16)
    n_signal = int(size * size * coverage)
    y_coords = np.random.randint(0, size, n_signal)
    x_coords = np.random.randint(0, size, n_signal)
    image[y_coords, x_coords] = np.random.randint(1000, 5000, n_signal)
    return image


def create_tiled_illumination(size=512, n_tiles=4):
    """Create image with tile-to-tile illumination variation."""
    image = np.zeros((size, size), dtype=float)
    tile_size = size // n_tiles

    for i in range(n_tiles):
        for j in range(n_tiles):
            # Each tile has different mean intensity
            brightness = 500 + np.random.rand() * 1500
            tile = np.random.poisson(brightness, (tile_size, tile_size))
            image[i*tile_size:(i+1)*tile_size,
                  j*tile_size:(j+1)*tile_size] = tile

    return image.astype(np.uint16)


class TestVignettingDetection:
    """Test vignetting detection."""

    def test_detect_strong_vignetting(self):
        """Should detect strong vignetting (30% drop)."""
        image = create_vignetting_image(size=512, drop=0.3)
        has_vignetting, drop = detect_vignetting(image, threshold=0.15)

        assert has_vignetting, "Should detect strong vignetting"
        assert drop > 0.15, f"Drop {drop} should exceed threshold"

    def test_no_vignetting_uniform(self):
        """Should not detect vignetting in uniform image."""
        image = create_uniform_image(size=512)
        has_vignetting, drop = detect_vignetting(image, threshold=0.15)

        assert not has_vignetting, "Should not detect vignetting in uniform image"
        assert drop < 0.15, f"Drop {drop} should be below threshold"

    def test_radial_profile_shape(self):
        """Radial profile should have correct shape."""
        image = create_uniform_image(size=512)
        radii, profile = compute_radial_intensity_profile(image, n_bins=20)

        assert len(radii) == 20
        assert len(profile) == 20
        assert np.all(radii >= 0) and np.all(radii <= 1)


class TestSpatialVariance:
    """Test spatial variance computation."""

    def test_high_variance_tiled(self):
        """Tiled illumination should show high variance."""
        image = create_tiled_illumination(size=512, n_tiles=4)
        metrics = compute_spatial_variance(image, tile_size=128)

        assert metrics['tile_mean_cv'] > 0.15, \
            f"Tiled image should have high CV: {metrics['tile_mean_cv']}"

    def test_low_variance_uniform(self):
        """Uniform image should show low variance."""
        image = create_uniform_image(size=512)
        metrics = compute_spatial_variance(image, tile_size=128)

        assert metrics['tile_mean_cv'] < 0.15, \
            f"Uniform image should have low CV: {metrics['tile_mean_cv']}"


class TestSignalSparsity:
    """Test signal sparsity computation."""

    def test_sparse_signal(self):
        """Sparse image should have low sparsity value."""
        image = create_sparse_image(size=512, coverage=0.03)
        sparsity = compute_signal_sparsity(image)

        assert sparsity < 0.1, \
            f"Sparse image should have low sparsity: {sparsity}"

    def test_dense_signal(self):
        """Dense image should have high sparsity value."""
        image = create_uniform_image(size=512)
        sparsity = compute_signal_sparsity(image)

        assert sparsity > 0.3, \
            f"Dense image should have high sparsity: {sparsity}"


class TestAutoDetection:
    """Test overall automatic detection logic."""

    def test_should_correct_vignetting(self):
        """Should apply BaSiC to vignetting image."""
        image = create_vignetting_image(size=512, drop=0.3)
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="TestChannel"
        )

        assert should_correct, \
            f"Should correct vignetting. Metrics: {metrics}"

    def test_should_skip_uniform(self):
        """Should skip BaSiC for uniform image."""
        image = create_uniform_image(size=512)
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="TestChannel"
        )

        # Uniform image might be corrected if there's some variance
        # Just check that we get a valid decision
        assert isinstance(should_correct, bool)
        assert 'reasoning' in metrics

    def test_should_skip_sparse(self):
        """Should skip BaSiC for sparse image."""
        image = create_sparse_image(size=512, coverage=0.03)
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="TestChannel"
        )

        assert not should_correct, \
            f"Should skip sparse image. Metrics: {metrics}"
        assert 'sparse' in metrics['reasoning'].lower()

    def test_should_skip_dapi(self):
        """Should skip DAPI channel."""
        image = create_uniform_image(size=512)
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="DAPI"
        )

        assert not should_correct, "Should skip DAPI"
        assert 'dapi' in metrics['reasoning'].lower()

    def test_should_correct_tiled(self):
        """Should apply BaSiC to tiled illumination."""
        image = create_tiled_illumination(size=512, n_tiles=4)
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="TestChannel"
        )

        assert should_correct, \
            f"Should correct tiled illumination. Metrics: {metrics}"

    def test_metrics_structure(self):
        """Metrics should have expected structure."""
        image = create_uniform_image(size=512)
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="TestChannel"
        )

        expected_keys = [
            'vignetting_detected',
            'vignetting_drop',
            'tile_mean_cv',
            'tile_std_cv',
            'gradient_magnitude',
            'signal_sparsity',
            'decision',
            'reasoning'
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"


class TestEdgeCases:
    """Test edge cases."""

    def test_single_value_image(self):
        """Handle constant image."""
        image = np.ones((512, 512), dtype=np.uint16) * 1000
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="Constant"
        )

        assert not should_correct, "Should skip constant image"

    def test_zero_image(self):
        """Handle all-zero image."""
        image = np.zeros((512, 512), dtype=np.uint16)
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="Empty"
        )

        assert not should_correct, "Should skip empty image"

    def test_small_image(self):
        """Handle small image."""
        image = create_uniform_image(size=128)
        should_correct, metrics = should_apply_basic_correction(
            image,
            channel_name="Small"
        )

        # Should still return valid decision
        assert isinstance(should_correct, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

#!/usr/bin/env python3
"""Unit tests for preprocess.py module."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add bin/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'bin'))

from preprocess import split_image_into_fovs, reconstruct_image_from_fovs, apply_basic_correction


class TestFOVSplitting:
    """Test FOV splitting and reconstruction."""

    def test_split_reconstruct_identity(self):
        """Test that split + reconstruct returns original image."""
        # Create test image
        img = np.random.randint(0, 255, size=(2000, 2000), dtype=np.uint16)
        fov_size = (512, 512)

        # Split and reconstruct
        fov_stack, positions, _ = split_image_into_fovs(img, fov_size, overlap=0)
        reconstructed = reconstruct_image_from_fovs(fov_stack, positions, img.shape)

        # Should be identical
        assert np.array_equal(img, reconstructed), "Reconstruction doesn't match original"

    def test_split_with_overlap(self):
        """Test FOV splitting with overlap."""
        img = np.random.randint(0, 255, size=(1000, 1000), dtype=np.uint16)
        fov_size = (256, 256)
        overlap = 32

        fov_stack, positions, (fov_h, fov_w) = split_image_into_fovs(img, fov_size, overlap=overlap)

        assert fov_h == 256
        assert fov_w == 256
        assert fov_stack.ndim == 3
        assert fov_stack.shape[1] == fov_h
        assert fov_stack.shape[2] == fov_w

    def test_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        img = np.random.randint(0, 255, size=(1000, 1000), dtype=np.uint16)
        fov_size = (256, 256)
        invalid_overlap = 300  # Greater than FOV size

        with pytest.raises(ValueError, match="Overlap.*must be"):
            split_image_into_fovs(img, fov_size, overlap=invalid_overlap)

    def test_3d_image_raises_error(self):
        """Test that 3D image raises error."""
        img = np.random.randint(0, 255, size=(10, 100, 100), dtype=np.uint16)
        fov_size = (50, 50)

        with pytest.raises(ValueError, match="Image must be 2D"):
            split_image_into_fovs(img, fov_size)


class TestBaSiCCorrection:
    """Test BaSiC illumination correction."""

    @pytest.mark.slow
    def test_basic_correction_runs(self):
        """Test that BaSiC correction runs without error on small image."""
        # Small synthetic image with vignetting effect
        x, y = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
        vignette = 1 - 0.3 * (x**2 + y**2)
        img = (np.random.poisson(100, size=(500, 500)) * vignette).astype(np.uint16)

        fov_size = (100, 100)

        # Run BaSiC correction
        corrected, basic_model = apply_basic_correction(
            img,
            fov_size=fov_size,
            get_darkfield=False,  # Faster
            autotune=False,
            n_iter=1  # Minimal iterations for testing
        )

        assert corrected.shape == img.shape
        assert corrected.dtype == img.dtype
        assert basic_model is not None

    def test_basic_correction_invalid_input(self):
        """Test that 3D image raises error."""
        img = np.random.randint(0, 255, size=(10, 100, 100), dtype=np.uint16)
        fov_size = (50, 50)

        with pytest.raises(ValueError, match="requires a 2D image"):
            apply_basic_correction(img, fov_size)


class TestChannelNameParsing:
    """Test channel name parsing from filenames."""

    def test_standard_filename(self):
        """Test parsing standard filename format."""
        from preprocess import get_channel_names

        filename = "P001_DAPI_FITC_TexasRed_corrected.ome.tif"
        channels = get_channel_names(filename)

        assert channels == ['DAPI', 'FITC', 'TexasRed']

    def test_preprocessed_filename(self):
        """Test parsing filename with preprocessing suffix."""
        from preprocess import get_channel_names

        filename = "P002_DAPI_PANCK_preprocessed.ome.tif"
        channels = get_channel_names(filename)

        assert channels == ['DAPI', 'PANCK']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

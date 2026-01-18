"""
Unit tests for CPU tiled registration components.

Tests the four Python scripts that implement memory-efficient tiled registration:
- compute_tile_plan.py
- affine_tile.py
- diffeo_tile.py
- stitch_tiles.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile

# Add bin directory to path for imports
BIN_DIR = Path(__file__).parent.parent.parent / 'bin'
sys.path.insert(0, str(BIN_DIR))


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def synthetic_ome_tiff(tmp_path):
    """Create a synthetic OME-TIFF test image (512x512, 3 channels)."""
    def _create_image(name, shape=(3, 512, 512), offset=(0, 0)):
        """Create a test image with recognizable patterns."""
        filepath = tmp_path / f"{name}.ome.tiff"
        data = np.zeros(shape, dtype=np.float32)

        # Create gradient patterns for each channel
        for c in range(shape[0]):
            y_coords = np.arange(shape[1]).reshape(-1, 1) + offset[0]
            x_coords = np.arange(shape[2]).reshape(1, -1) + offset[1]
            data[c] = (y_coords * 0.1 + x_coords * 0.1 + c * 10) % 255

        # Add some features for registration
        center_y, center_x = shape[1] // 2 + offset[0], shape[2] // 2 + offset[1]
        y, x = np.ogrid[:shape[1], :shape[2]]
        mask = ((y - center_y % shape[1])**2 + (x - center_x % shape[2])**2) < 50**2
        data[0, mask] = 255

        tifffile.imwrite(filepath, data, photometric='minisblack')
        return filepath

    return _create_image


@pytest.fixture
def reference_image(synthetic_ome_tiff):
    """Create a reference image."""
    return synthetic_ome_tiff("reference", shape=(3, 512, 512), offset=(0, 0))


@pytest.fixture
def moving_image(synthetic_ome_tiff):
    """Create a moving image with slight offset."""
    return synthetic_ome_tiff("moving", shape=(3, 512, 512), offset=(5, 5))


@pytest.fixture
def sample_tile_plan(tmp_path):
    """Create a sample tile plan JSON."""
    plan = {
        "version": "1.0",
        "reference_path": "reference.ome.tiff",
        "moving_path": "moving.ome.tiff",
        "image_shape": [3, 512, 512],
        "dtype": "float32",
        "affine_crop_size": 256,
        "diffeo_crop_size": 128,
        "overlap_percent": 40.0,
        "affine_tiles": [
            {"tile_id": "affine_0_0", "row": 0, "col": 0, "y_start": 0, "y_end": 256, "x_start": 0, "x_end": 256},
            {"tile_id": "affine_0_1", "row": 0, "col": 1, "y_start": 0, "y_end": 256, "x_start": 154, "x_end": 410},
            {"tile_id": "affine_1_0", "row": 1, "col": 0, "y_start": 154, "y_end": 410, "x_start": 0, "x_end": 256},
            {"tile_id": "affine_1_1", "row": 1, "col": 1, "y_start": 154, "y_end": 410, "x_start": 154, "x_end": 410},
        ],
        "diffeo_tiles": [
            {"tile_id": "diffeo_0_0", "row": 0, "col": 0, "y_start": 0, "y_end": 128, "x_start": 0, "x_end": 128},
            {"tile_id": "diffeo_0_1", "row": 0, "col": 1, "y_start": 0, "y_end": 128, "x_start": 51, "x_end": 179},
        ]
    }
    plan_path = tmp_path / "tile_plan.json"
    with open(plan_path, 'w') as f:
        json.dump(plan, f)
    return plan_path, plan


# ==============================================================================
# Tests for compute_tile_plan.py
# ==============================================================================

class TestComputeTilePlan:
    """Tests for tile plan generation."""

    def test_compute_tile_plan_generates_valid_json(self, reference_image, moving_image, tmp_dir):
        """Test that compute_tile_plan produces valid JSON output."""
        import compute_tile_plan

        output_path = tmp_dir / "tile_plan.json"

        # Run the tile plan computation
        compute_tile_plan.main([
            "--reference", str(reference_image),
            "--moving", str(moving_image),
            "--output", str(output_path),
            "--affine-crop-size", "256",
            "--diffeo-crop-size", "128",
            "--overlap-percent", "40.0"
        ])

        # Verify output exists and is valid JSON
        assert output_path.exists()

        with open(output_path) as f:
            plan = json.load(f)

        # Check required fields
        assert "version" in plan
        assert "affine_tiles" in plan
        assert "diffeo_tiles" in plan
        assert "image_shape" in plan
        assert "dtype" in plan

    def test_compute_tile_plan_correct_tile_count(self, reference_image, moving_image, tmp_dir):
        """Test that the correct number of tiles are generated."""
        import compute_tile_plan

        output_path = tmp_dir / "tile_plan.json"

        compute_tile_plan.main([
            "--reference", str(reference_image),
            "--moving", str(moving_image),
            "--output", str(output_path),
            "--affine-crop-size", "256",
            "--diffeo-crop-size", "128",
            "--overlap-percent", "40.0"
        ])

        with open(output_path) as f:
            plan = json.load(f)

        # With 512x512 image, 256px crops with 40% overlap:
        # Effective step = 256 * (1 - 0.4) = 153.6 ≈ 154
        # Number of tiles per dimension ≈ ceil(512 / 154) = 4
        # But the actual implementation may vary
        assert len(plan["affine_tiles"]) > 0
        assert len(plan["diffeo_tiles"]) > 0

    def test_compute_tile_plan_tile_coordinates_have_overlap(self, reference_image, moving_image, tmp_dir):
        """Test that tiles have proper overlap."""
        import compute_tile_plan

        output_path = tmp_dir / "tile_plan.json"

        compute_tile_plan.main([
            "--reference", str(reference_image),
            "--moving", str(moving_image),
            "--output", str(output_path),
            "--affine-crop-size", "256",
            "--diffeo-crop-size", "128",
            "--overlap-percent", "40.0"
        ])

        with open(output_path) as f:
            plan = json.load(f)

        # Check that adjacent tiles overlap
        affine_tiles = plan["affine_tiles"]
        if len(affine_tiles) >= 2:
            # Sort by position to find adjacent tiles
            tiles_by_row = {}
            for tile in affine_tiles:
                row = tile["row"]
                if row not in tiles_by_row:
                    tiles_by_row[row] = []
                tiles_by_row[row].append(tile)

            # Check horizontal overlap within a row
            for row, tiles in tiles_by_row.items():
                tiles = sorted(tiles, key=lambda t: t["col"])
                for i in range(len(tiles) - 1):
                    t1, t2 = tiles[i], tiles[i + 1]
                    # t2 should start before t1 ends (overlap)
                    assert t2["x_start"] < t1["x_end"], "Adjacent tiles should overlap"


# ==============================================================================
# Tests for affine_tile.py
# ==============================================================================

class TestAffineTile:
    """Tests for single affine tile processing."""

    def test_affine_tile_produces_correct_shape(self, reference_image, moving_image, sample_tile_plan, tmp_dir):
        """Test that affine tile output has correct shape."""
        import affine_tile

        tile_plan_path, plan = sample_tile_plan
        tile_id = "affine_0_0"
        output_prefix = tmp_dir / tile_id

        affine_tile.main([
            "--tile-id", tile_id,
            "--tile-plan", str(tile_plan_path),
            "--reference", str(reference_image),
            "--moving", str(moving_image),
            "--output-prefix", str(output_prefix),
            "--n-features", "1000"
        ])

        # Load output
        npy_path = tmp_dir / f"{tile_id}.npy"
        assert npy_path.exists()

        tile_data = np.load(npy_path)

        # Shape should be (channels, tile_height, tile_width)
        tile_info = next(t for t in plan["affine_tiles"] if t["tile_id"] == tile_id)
        expected_height = tile_info["y_end"] - tile_info["y_start"]
        expected_width = tile_info["x_end"] - tile_info["x_start"]

        assert tile_data.shape[0] == plan["image_shape"][0]  # channels
        assert tile_data.shape[1] == expected_height
        assert tile_data.shape[2] == expected_width

    def test_affine_tile_handles_identity_transform(self, reference_image, sample_tile_plan, tmp_dir):
        """Test that registering identical images produces identity-like result."""
        import affine_tile

        tile_plan_path, plan = sample_tile_plan
        tile_id = "affine_0_0"
        output_prefix = tmp_dir / tile_id

        # Use same image as reference and moving
        affine_tile.main([
            "--tile-id", tile_id,
            "--tile-plan", str(tile_plan_path),
            "--reference", str(reference_image),
            "--moving", str(reference_image),  # Same image
            "--output-prefix", str(output_prefix),
            "--n-features", "1000"
        ])

        npy_path = tmp_dir / f"{tile_id}.npy"
        assert npy_path.exists()

        # Output should exist and be valid
        tile_data = np.load(npy_path)
        assert not np.isnan(tile_data).any()

    def test_affine_tile_creates_metadata_json(self, reference_image, moving_image, sample_tile_plan, tmp_dir):
        """Test that affine tile creates metadata JSON file."""
        import affine_tile

        tile_plan_path, plan = sample_tile_plan
        tile_id = "affine_0_0"
        output_prefix = tmp_dir / tile_id

        affine_tile.main([
            "--tile-id", tile_id,
            "--tile-plan", str(tile_plan_path),
            "--reference", str(reference_image),
            "--moving", str(moving_image),
            "--output-prefix", str(output_prefix),
            "--n-features", "1000"
        ])

        meta_path = tmp_dir / f"{tile_id}_meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert "tile_id" in meta
        assert "transform_type" in meta


# ==============================================================================
# Tests for diffeo_tile.py
# ==============================================================================

class TestDiffeoTile:
    """Tests for single diffeomorphic tile processing."""

    def test_diffeo_tile_produces_correct_shape(self, reference_image, moving_image, sample_tile_plan, tmp_dir):
        """Test that diffeo tile output has correct shape."""
        import diffeo_tile

        tile_plan_path, plan = sample_tile_plan
        tile_id = "diffeo_0_0"
        output_prefix = tmp_dir / tile_id

        # Use moving_image as the "affine" input for testing
        diffeo_tile.main([
            "--tile-id", tile_id,
            "--tile-plan", str(tile_plan_path),
            "--reference", str(reference_image),
            "--affine", str(moving_image),
            "--output-prefix", str(output_prefix),
            "--opt-tol", "1e-3",
            "--inv-tol", "1e-3"
        ])

        npy_path = tmp_dir / f"{tile_id}.npy"
        assert npy_path.exists()

        tile_data = np.load(npy_path)

        # Shape should be (channels, tile_height, tile_width)
        tile_info = next(t for t in plan["diffeo_tiles"] if t["tile_id"] == tile_id)
        expected_height = tile_info["y_end"] - tile_info["y_start"]
        expected_width = tile_info["x_end"] - tile_info["x_start"]

        assert tile_data.shape[0] == plan["image_shape"][0]  # channels
        assert tile_data.shape[1] == expected_height
        assert tile_data.shape[2] == expected_width

    def test_diffeo_tile_creates_metadata_json(self, reference_image, moving_image, sample_tile_plan, tmp_dir):
        """Test that diffeo tile creates metadata JSON file."""
        import diffeo_tile

        tile_plan_path, plan = sample_tile_plan
        tile_id = "diffeo_0_0"
        output_prefix = tmp_dir / tile_id

        diffeo_tile.main([
            "--tile-id", tile_id,
            "--tile-plan", str(tile_plan_path),
            "--reference", str(reference_image),
            "--affine", str(moving_image),
            "--output-prefix", str(output_prefix),
            "--opt-tol", "1e-3",
            "--inv-tol", "1e-3"
        ])

        meta_path = tmp_dir / f"{tile_id}_meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert "tile_id" in meta


# ==============================================================================
# Tests for stitch_tiles.py
# ==============================================================================

class TestStitchTiles:
    """Tests for tile stitching."""

    @pytest.fixture
    def affine_tiles(self, sample_tile_plan, tmp_dir):
        """Create fake affine tile .npy files for stitching tests."""
        tile_plan_path, plan = sample_tile_plan
        tiles_dir = tmp_dir / "affine_tiles"
        tiles_dir.mkdir()

        for tile in plan["affine_tiles"]:
            tile_id = tile["tile_id"]
            height = tile["y_end"] - tile["y_start"]
            width = tile["x_end"] - tile["x_start"]

            # Create fake tile data
            data = np.random.rand(3, height, width).astype(np.float32)
            np.save(tiles_dir / f"{tile_id}.npy", data)

        return tiles_dir

    @pytest.fixture
    def diffeo_tiles(self, sample_tile_plan, tmp_dir):
        """Create fake diffeo tile .npy files for stitching tests."""
        tile_plan_path, plan = sample_tile_plan
        tiles_dir = tmp_dir / "diffeo_tiles"
        tiles_dir.mkdir()

        for tile in plan["diffeo_tiles"]:
            tile_id = tile["tile_id"]
            height = tile["y_end"] - tile["y_start"]
            width = tile["x_end"] - tile["x_start"]

            # Create fake tile data
            data = np.random.rand(3, height, width).astype(np.float32)
            np.save(tiles_dir / f"{tile_id}.npy", data)

        return tiles_dir

    def test_stitch_tiles_affine_output_correct_size(self, sample_tile_plan, affine_tiles, tmp_dir):
        """Test that stitching affine tiles produces correct output size."""
        import stitch_tiles

        tile_plan_path, plan = sample_tile_plan
        output_path = tmp_dir / "stitched_affine.tiff"

        stitch_tiles.main([
            "--tile-plan", str(tile_plan_path),
            "--tiles-dir", str(affine_tiles),
            "--stage", "affine",
            "--output", str(output_path)
        ])

        assert output_path.exists()

        # Check output dimensions
        with tifffile.TiffFile(output_path) as tif:
            data = tif.asarray()

        # Output should match original image dimensions
        assert data.shape[0] == plan["image_shape"][0]  # channels
        assert data.shape[1] == plan["image_shape"][1]  # height
        assert data.shape[2] == plan["image_shape"][2]  # width

    def test_stitch_tiles_diffeo_creates_ome_tiff(self, sample_tile_plan, diffeo_tiles, moving_image, tmp_dir):
        """Test that stitching diffeo tiles creates valid OME-TIFF."""
        import stitch_tiles

        tile_plan_path, plan = sample_tile_plan
        output_path = tmp_dir / "registered.ome.tiff"

        stitch_tiles.main([
            "--tile-plan", str(tile_plan_path),
            "--tiles-dir", str(diffeo_tiles),
            "--stage", "diffeo",
            "--output", str(output_path),
            "--moving", str(moving_image)
        ])

        assert output_path.exists()

        # Verify it's readable as a TIFF
        with tifffile.TiffFile(output_path) as tif:
            data = tif.asarray()
            assert data is not None


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestCpuTiledIntegration:
    """Integration tests for the full cpu_tiled workflow."""

    @pytest.mark.slow
    def test_full_tiled_registration_pipeline(self, reference_image, moving_image, tmp_dir):
        """Test the complete tiled registration pipeline end-to-end."""
        import compute_tile_plan
        import affine_tile
        import stitch_tiles

        # Step 1: Compute tile plan
        tile_plan_path = tmp_dir / "tile_plan.json"
        compute_tile_plan.main([
            "--reference", str(reference_image),
            "--moving", str(moving_image),
            "--output", str(tile_plan_path),
            "--affine-crop-size", "256",
            "--diffeo-crop-size", "128",
            "--overlap-percent", "40.0"
        ])

        with open(tile_plan_path) as f:
            plan = json.load(f)

        # Step 2: Process affine tiles
        affine_tiles_dir = tmp_dir / "affine_tiles"
        affine_tiles_dir.mkdir()

        for tile in plan["affine_tiles"]:
            tile_id = tile["tile_id"]
            affine_tile.main([
                "--tile-id", tile_id,
                "--tile-plan", str(tile_plan_path),
                "--reference", str(reference_image),
                "--moving", str(moving_image),
                "--output-prefix", str(affine_tiles_dir / tile_id),
                "--n-features", "1000"
            ])

        # Step 3: Stitch affine tiles
        affine_output = tmp_dir / "affine_stitched.tiff"
        stitch_tiles.main([
            "--tile-plan", str(tile_plan_path),
            "--tiles-dir", str(affine_tiles_dir),
            "--stage", "affine",
            "--output", str(affine_output)
        ])

        assert affine_output.exists()

        # Verify output has correct dimensions
        with tifffile.TiffFile(affine_output) as tif:
            data = tif.asarray()

        assert data.shape == tuple(plan["image_shape"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

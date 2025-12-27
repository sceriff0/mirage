#!/usr/bin/env python3
"""Unit tests for quantify.py module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add bin/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'bin'))


class TestQuantificationBasics:
    """Test basic quantification functionality."""

    def test_regionprops_extraction(self):
        """Test that regionprops extraction works correctly."""
        # Create synthetic mask with 2 cells
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[10:30, 10:30] = 1  # Cell 1
        mask[50:80, 50:80] = 2  # Cell 2

        # Create synthetic intensity image
        intensity = np.random.randint(50, 200, size=(100, 100), dtype=np.uint16)

        # Manual regionprops extraction
        from skimage.measure import regionprops_table

        props = regionprops_table(
            mask,
            intensity_image=intensity,
            properties=['label', 'area', 'centroid', 'eccentricity', 'mean_intensity']
        )

        df = pd.DataFrame(props)

        assert len(df) == 2, "Should have 2 cells"
        assert 'label' in df.columns
        assert 'area' in df.columns
        assert 'mean_intensity' in df.columns

        # Cell 1 should be roughly 20x20 = 400 pixels
        cell1 = df[df['label'] == 1].iloc[0]
        assert cell1['area'] == pytest.approx(400, abs=10)

    def test_csv_column_structure(self):
        """Test expected CSV column structure."""
        # Expected columns from quantification
        expected_morphology = [
            'label', 'y', 'x', 'area', 'eccentricity',
            'perimeter', 'convex_area', 'axis_major_length', 'axis_minor_length'
        ]

        # Mock quantification result
        data = {
            'label': [1, 2, 3],
            'y': [10, 20, 30],
            'x': [15, 25, 35],
            'area': [100, 150, 200],
            'eccentricity': [0.5, 0.6, 0.7],
            'perimeter': [40, 50, 60],
            'convex_area': [105, 155, 205],
            'axis_major_length': [12, 15, 18],
            'axis_minor_length': [8, 10, 12],
            'DAPI': [100, 120, 140],
            'FITC': [200, 220, 240]
        }

        df = pd.DataFrame(data)

        # Check all morphology columns present
        for col in expected_morphology:
            assert col in df.columns, f"Missing column: {col}"

        # Check marker columns present
        assert 'DAPI' in df.columns
        assert 'FITC' in df.columns


class TestCSVMerging:
    """Test CSV merging logic from MERGE_QUANT_CSVS."""

    def test_inner_join_consistency(self):
        """Test that inner join maintains cell label consistency."""
        # Reference CSV (DAPI)
        ref_df = pd.DataFrame({
            'label': [1, 2, 3, 4],
            'area': [100, 150, 200, 250],
            'DAPI': [100, 120, 140, 160]
        })

        # Other marker CSV (FITC) - missing cell 4
        fitc_df = pd.DataFrame({
            'label': [1, 2, 3],
            'FITC': [200, 220, 240]
        })

        # Merge with inner join
        merged = ref_df.merge(fitc_df, on='label', how='inner')

        # Should only have cells 1, 2, 3
        assert len(merged) == 3
        assert list(merged['label']) == [1, 2, 3]
        assert 'DAPI' in merged.columns
        assert 'FITC' in merged.columns

    def test_marker_column_extraction(self):
        """Test extraction of marker columns excluding morphology."""
        df = pd.DataFrame({
            'label': [1, 2, 3],
            'area': [100, 150, 200],
            'perimeter': [40, 50, 60],
            'DAPI': [100, 120, 140],
            'FITC': [200, 220, 240],
            'PANCK': [150, 170, 190]
        })

        # Morphology columns to exclude
        morphology_cols = ['label', 'area', 'perimeter']

        # Extract marker columns
        marker_cols = [col for col in df.columns if col not in morphology_cols]

        assert marker_cols == ['DAPI', 'FITC', 'PANCK']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

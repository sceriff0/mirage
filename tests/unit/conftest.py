#!/usr/bin/env python3
"""pytest configuration and fixtures for unit tests."""

import pytest
import numpy as np


@pytest.fixture
def synthetic_2d_image():
    """Create a small synthetic 2D image for testing."""
    return np.random.randint(0, 255, size=(512, 512), dtype=np.uint16)


@pytest.fixture
def synthetic_3d_image():
    """Create a small synthetic 3D multichannel image for testing."""
    return np.random.randint(0, 255, size=(3, 512, 512), dtype=np.uint16)


@pytest.fixture
def synthetic_mask():
    """Create a synthetic segmentation mask with 10 cells."""
    mask = np.zeros((512, 512), dtype=np.int32)

    # Create 10 random cells
    np.random.seed(42)
    for i in range(1, 11):
        y = np.random.randint(50, 450)
        x = np.random.randint(50, 450)
        size = np.random.randint(20, 40)
        mask[y:y+size, x:x+size] = i

    return mask


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )

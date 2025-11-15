#!/usr/bin/env python3
"""Generate small test fixtures for unit and integration tests.

Creates a few small TIFF and NPY files under tests/testdata/ so tests and a
small pipeline run can exercise I/O without large data.
"""
import os
import numpy as np
import tifffile

OUT_DIR = os.path.join(os.path.dirname(__file__), 'testdata')

os.makedirs(OUT_DIR, exist_ok=True)

# Small DAPI-like image (single channel)
dapi = np.zeros((32, 32), dtype=np.uint16)
dapi[8:24, 8:24] = 1000
tifffile.imwrite(os.path.join(OUT_DIR, 'sample_DAPI.tif'), dapi)

# Two marker channels
marker1 = np.zeros((32, 32), dtype=np.uint16)
marker1[10:14, 10:14] = 500
tifffile.imwrite(os.path.join(OUT_DIR, 'sample_MARKER1.tif'), marker1)

marker2 = np.zeros((32, 32), dtype=np.uint16)
marker2[12:20, 12:20] = 800
tifffile.imwrite(os.path.join(OUT_DIR, 'sample_MARKER2.tif'), marker2)

# A tiny segmentation mask (labels)
mask = np.zeros((32, 32), dtype=np.int32)
mask[9:15, 9:15] = 1
mask[16:22, 16:22] = 2
np.save(os.path.join(OUT_DIR, 'sample_mask.npy'), mask)

print(f"Generated testdata in {OUT_DIR}")

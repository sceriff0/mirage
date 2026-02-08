#!/usr/bin/env python3
"""Generate complete test fixtures for full pipeline testing.

Creates realistic test data including:
- Multi-channel OME-TIFF images (reference and moving)
- Segmentation masks
- CSV files for all pipeline entry points
- Both valid and invalid test cases for validation testing
"""
import os
import numpy as np
import tifffile
from pathlib import Path

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)

print("Generating comprehensive test data...")

# =============================================================================
# 1. Generate realistic multi-channel OME-TIFF images
# =============================================================================

def create_multichannel_image(filename, size=(128, 128), n_channels=3,
                              add_noise=True, shift=(0, 0)):
    """Create a realistic multi-channel OME-TIFF with synthetic cell-like structures."""
    channels = []

    for ch in range(n_channels):
        img = np.zeros(size, dtype=np.uint16)

        # Add some "cells" - bright spots with gaussian-like intensity
        n_cells = np.random.randint(10, 20)
        for _ in range(n_cells):
            # Random cell center (with optional shift for moving images)
            cy = np.random.randint(20, size[0]-20) + shift[0]
            cx = np.random.randint(20, size[1]-20) + shift[1]

            # Cell size
            radius = np.random.randint(3, 8)

            # Intensity varies by channel
            if ch == 0:  # DAPI - brightest, nuclear
                intensity = np.random.randint(8000, 15000)
            else:  # Markers - variable
                intensity = np.random.randint(2000, 10000)

            # Draw circle with gaussian falloff
            yy, xx = np.ogrid[:size[0], :size[1]]
            circle_mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2

            # Add gaussian falloff
            dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            gaussian = np.exp(-(dist**2) / (2 * (radius/2)**2))

            img = np.maximum(img, (circle_mask * gaussian * intensity).astype(np.uint16))

        # Add background noise
        if add_noise:
            noise = np.random.normal(100, 20, size).astype(np.int32)
            img = np.clip(img.astype(np.int32) + noise, 0, 65535).astype(np.uint16)

        channels.append(img)

    # Stack channels (C, Y, X)
    multichannel = np.stack(channels, axis=0)

    # Save as OME-TIFF with proper metadata
    tifffile.imwrite(
        filename,
        multichannel,
        photometric='minisblack',
        metadata={'axes': 'CYX', 'Channel': {'Name': ['DAPI', 'PANCK', 'SMA'][:n_channels]}}
    )
    print(f"  Created {filename} - shape: {multichannel.shape}")
    return multichannel

# Patient P001 - Reference and 2 moving images
print("\n1. Creating multi-channel OME-TIFF images...")
create_multichannel_image(OUT_DIR / 'P001_ref.ome.tiff', n_channels=3, shift=(0, 0))
create_multichannel_image(OUT_DIR / 'P001_mov1.ome.tiff', n_channels=3, shift=(5, 5))
create_multichannel_image(OUT_DIR / 'P001_mov2.ome.tiff', n_channels=3, shift=(-3, 4))

# Patient P002 - Single slide (reference only)
create_multichannel_image(OUT_DIR / 'P002_ref.ome.tiff', n_channels=3, shift=(0, 0))

# =============================================================================
# 2. Generate segmentation masks
# =============================================================================
print("\n2. Creating segmentation masks...")

def create_segmentation_mask(filename, size=(128, 128), n_cells=15):
    """Create a realistic segmentation mask with labeled cells."""
    mask = np.zeros(size, dtype=np.int32)

    label = 1
    attempts = 0
    max_attempts = n_cells * 10

    while label <= n_cells and attempts < max_attempts:
        attempts += 1

        # Random cell center
        cy = np.random.randint(10, size[0]-10)
        cx = np.random.randint(10, size[1]-10)
        radius = np.random.randint(4, 9)

        # Check for overlap
        yy, xx = np.ogrid[:size[0], :size[1]]
        circle_mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2

        if mask[circle_mask].sum() == 0:  # No overlap
            mask[circle_mask] = label
            label += 1

    np.save(filename, mask)
    print(f"  Created {filename} - {label-1} cells")
    return mask

create_segmentation_mask(OUT_DIR / 'P001_cell_mask.npy', n_cells=20)
create_segmentation_mask(OUT_DIR / 'P002_cell_mask.npy', n_cells=15)

# =============================================================================
# 3. Generate valid input CSVs for each pipeline entry point
# =============================================================================
print("\n3. Creating valid input CSVs...")

# Use absolute paths resolved from the output directory
# This ensures CSVs work regardless of where the pipeline is launched from
TESTDATA_ABS = str(OUT_DIR.resolve())

# 3a. Valid input for preprocessing step (ND2 conversion disabled)
with open(OUT_DIR / 'valid_preprocessing.csv', 'w') as f:
    f.write('patient_id,path_to_file,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,true,DAPI|PANCK|SMA\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_mov1.ome.tiff,false,DAPI|PANCK|SMA\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_mov2.ome.tiff,false,DAPI|PANCK|SMA\n')
    f.write(f'P002,{TESTDATA_ABS}/P002_ref.ome.tiff,true,DAPI|PANCK|SMA\n')
print(f"  Created valid_preprocessing.csv")

# 3b. Valid checkpoint CSV for registration step
with open(OUT_DIR / 'valid_checkpoint_registration.csv', 'w') as f:
    f.write('patient_id,preprocessed_image,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,true,DAPI|PANCK|SMA\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_mov1.ome.tiff,false,DAPI|PANCK|SMA\n')
print(f"  Created valid_checkpoint_registration.csv")

# 3c. Valid checkpoint CSV for postprocessing step
with open(OUT_DIR / 'valid_checkpoint_postprocessing.csv', 'w') as f:
    f.write('patient_id,registered_image,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,true,DAPI|PANCK|SMA\n')
print(f"  Created valid_checkpoint_postprocessing.csv")

# =============================================================================
# 4. Generate INVALID input CSVs for validation testing
# =============================================================================
print("\n4. Creating invalid input CSVs for validation tests...")

# 4a. Multiple references per patient
with open(OUT_DIR / 'invalid_multi_ref.csv', 'w') as f:
    f.write('patient_id,path_to_file,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,true,DAPI|PANCK|SMA\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_mov1.ome.tiff,true,DAPI|PANCK|SMA\n')  # SECOND REF!
    f.write(f'P001,{TESTDATA_ABS}/P001_mov2.ome.tiff,false,DAPI|PANCK|SMA\n')
print(f"  Created invalid_multi_ref.csv (multiple references)")

# 4b. No reference per patient
with open(OUT_DIR / 'invalid_no_ref.csv', 'w') as f:
    f.write('patient_id,path_to_file,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_mov1.ome.tiff,false,DAPI|PANCK|SMA\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_mov2.ome.tiff,false,DAPI|PANCK|SMA\n')
print(f"  Created invalid_no_ref.csv (no reference)")

# 4c. DAPI not in channel 0 (pre-converted OME-TIFF input)
with open(OUT_DIR / 'invalid_dapi_position.csv', 'w') as f:
    f.write('patient_id,path_to_file,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,true,PANCK|DAPI|SMA\n')  # DAPI NOT FIRST
print(f"  Created invalid_dapi_position.csv (DAPI not in position 0)")

# 4d. Missing DAPI channel
with open(OUT_DIR / 'invalid_no_dapi.csv', 'w') as f:
    f.write('patient_id,path_to_file,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,true,PANCK|SMA\n')  # NO DAPI
print(f"  Created invalid_no_dapi.csv (missing DAPI)")

# 4e. Invalid checkpoint - missing required column
with open(OUT_DIR / 'invalid_checkpoint_missing_col.csv', 'w') as f:
    f.write('patient_id,preprocessed_image,is_reference\n')  # Missing 'channels'
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,true\n')
print(f"  Created invalid_checkpoint_missing_col.csv (missing column)")

# 4f. Invalid checkpoint - malformed is_reference
with open(OUT_DIR / 'invalid_checkpoint_bad_ref.csv', 'w') as f:
    f.write('patient_id,preprocessed_image,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,yes,DAPI|PANCK|SMA\n')  # 'yes' not 'true'
print(f"  Created invalid_checkpoint_bad_ref.csv (invalid is_reference)")

# 4g. File does not exist
with open(OUT_DIR / 'invalid_file_not_found.csv', 'w') as f:
    f.write('patient_id,path_to_file,is_reference,channels\n')
    f.write(f'P001,/nonexistent/path/file.ome.tiff,true,DAPI|PANCK|SMA\n')
print(f"  Created invalid_file_not_found.csv (file not found)")

# =============================================================================
# 5. Update test.config input to use valid data
# =============================================================================
print("\n5. Creating test.config input CSV...")
with open(OUT_DIR / 'test_input.csv', 'w') as f:
    f.write('patient_id,path_to_file,is_reference,channels\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_ref.ome.tiff,true,DAPI|PANCK|SMA\n')
    f.write(f'P001,{TESTDATA_ABS}/P001_mov1.ome.tiff,false,DAPI|PANCK|SMA\n')
print(f"  Created test_input.csv for test profile")

print("\n" + "="*70)
print("✓ Test data generation complete!")
print("="*70)
print(f"\nGenerated files in {OUT_DIR}:")
print("\nValid data:")
print("  - P001_ref.ome.tiff, P001_mov1.ome.tiff, P001_mov2.ome.tiff")
print("  - P002_ref.ome.tiff")
print("  - P001_cell_mask.npy, P002_cell_mask.npy")
print("  - valid_preprocessing.csv")
print("  - valid_checkpoint_registration.csv")
print("  - valid_checkpoint_postprocessing.csv")
print("  - test_input.csv")
print("\nInvalid data (for validation testing):")
print("  - invalid_multi_ref.csv")
print("  - invalid_no_ref.csv")
print("  - invalid_dapi_position.csv")
print("  - invalid_no_dapi.csv")
print("  - invalid_checkpoint_missing_col.csv")
print("  - invalid_checkpoint_bad_ref.csv")
print("  - invalid_file_not_found.csv")
# =============================================================================
# 6. Generate additional test fixtures for module tests
# =============================================================================
print("\n6. Creating additional test fixtures for module tests...")

# 6a. Merged quantification CSV for phenotype tests
import json

with open(OUT_DIR / 'sample_merged_quant.csv', 'w') as f:
    f.write('label,centroid_x,centroid_y,area,perimeter,eccentricity,major_axis,minor_axis,solidity,DAPI,PANCK,SMA\n')
    np.random.seed(42)
    for i in range(1, 21):
        cx = np.random.uniform(10, 118)
        cy = np.random.uniform(10, 118)
        area = np.random.randint(150, 350)
        perimeter = np.random.uniform(45, 75)
        eccentricity = np.random.uniform(0.3, 0.6)
        major = np.random.uniform(12, 25)
        minor = np.random.uniform(8, 18)
        solidity = np.random.uniform(0.85, 0.98)
        dapi = np.random.randint(6000, 12000)
        panck = np.random.randint(1500, 8000)
        sma = np.random.randint(1000, 5000)
        f.write(f'{i},{cx:.1f},{cy:.1f},{area},{perimeter:.1f},{eccentricity:.2f},{major:.1f},{minor:.1f},{solidity:.2f},{dapi},{panck},{sma}\n')
print(f"  Created sample_merged_quant.csv (20 cells)")

# 6b. Max dimensions file for padding tests
with open(OUT_DIR / 'sample_max_dims.txt', 'w') as f:
    f.write('MAX_HEIGHT 256\n')
    f.write('MAX_WIDTH 256\n')
print(f"  Created sample_max_dims.txt")

# 6c. Individual dimensions file
with open(OUT_DIR / 'sample_dims.txt', 'w') as f:
    f.write('P001_ref.ome.tiff 128 128\n')
    f.write('P001_mov1.ome.tiff 128 128\n')
print(f"  Created sample_dims.txt")

# 6d. Sample features JSON
features = {
    "keypoints": [
        {"x": 25.5, "y": 30.2, "descriptor": [0.1] * 256},
        {"x": 50.1, "y": 45.8, "descriptor": [0.2] * 256},
        {"x": 80.3, "y": 70.5, "descriptor": [0.3] * 256},
        {"x": 100.0, "y": 95.2, "descriptor": [0.15] * 256},
        {"x": 60.5, "y": 110.8, "descriptor": [0.25] * 256}
    ],
    "detector": "superpoint",
    "n_features": 5
}
with open(OUT_DIR / 'sample_features.json', 'w') as f:
    json.dump(features, f, indent=2)
print(f"  Created sample_features.json (5 keypoints)")

# 6e. Phenotype mapping JSON
phenotype_mapping = {
    "phenotypes": {
        "0": "Unassigned",
        "1": "PANCK+",
        "2": "SMA+",
        "3": "PANCK+SMA+"
    },
    "colors": {
        "0": "#808080",
        "1": "#00FF00",
        "2": "#FF0000",
        "3": "#FFFF00"
    }
}
with open(OUT_DIR / 'sample_phenotype_mapping.json', 'w') as f:
    json.dump(phenotype_mapping, f, indent=2)
print(f"  Created sample_phenotype_mapping.json")

# 6f. Sample GeoJSON for phenotype output
geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[50, 30], [55, 30], [55, 35], [50, 35], [50, 30]]]
            },
            "properties": {
                "objectType": "cell",
                "classification": {"name": "PANCK+", "colorRGB": 65280},
                "measurements": [{"name": "DAPI", "value": 8500}]
            }
        }
    ]
}
with open(OUT_DIR / 'sample_phenotypes.geojson', 'w') as f:
    json.dump(geojson, f, indent=2)
print(f"  Created sample_phenotypes.geojson")

# 6g. Feature distance metrics JSON
metrics = {
    "pre_registration": {
        "mean_distance": 15.3,
        "median_distance": 12.8,
        "std_distance": 5.2,
        "n_matches": 45
    },
    "post_registration": {
        "mean_distance": 2.1,
        "median_distance": 1.8,
        "std_distance": 0.9,
        "n_matches": 45
    },
    "improvement": 86.3
}
with open(OUT_DIR / 'sample_feature_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Created sample_feature_metrics.json")

# 6h. Single channel TIF images (already exist but ensure proper format)
for ch_name in ['DAPI', 'PANCK', 'SMA']:
    img = np.random.randint(100, 10000, size=(128, 128), dtype=np.uint16)
    tifffile.imwrite(OUT_DIR / f'sample_{ch_name}.tif', img, photometric='minisblack')
print(f"  Created single-channel sample TIFs")

# 6i. Channels text file
with open(OUT_DIR / 'sample_channels.txt', 'w') as f:
    f.write('DAPI\n')
    f.write('PANCK\n')
    f.write('SMA\n')
print(f"  Created sample_channels.txt")

print("\n" + "="*70)
print("Test data generation complete!")
print("="*70)
print(f"\nGenerated files in {OUT_DIR}:")
print("\nValid data:")
print("  - P001_ref.ome.tiff, P001_mov1.ome.tiff, P001_mov2.ome.tiff")
print("  - P002_ref.ome.tiff")
print("  - P001_cell_mask.npy, P002_cell_mask.npy")
print("  - valid_preprocessing.csv")
print("  - valid_checkpoint_registration.csv")
print("  - valid_checkpoint_postprocessing.csv")
print("  - test_input.csv")
print("\nInvalid data (for validation testing):")
print("  - invalid_multi_ref.csv")
print("  - invalid_no_ref.csv")
print("  - invalid_dapi_position.csv")
print("  - invalid_no_dapi.csv")
print("  - invalid_checkpoint_missing_col.csv")
print("  - invalid_checkpoint_bad_ref.csv")
print("  - invalid_file_not_found.csv")
print("\nModule test fixtures:")
print("  - sample_merged_quant.csv")
print("  - sample_max_dims.txt")
print("  - sample_dims.txt")
print("  - sample_features.json")
print("  - sample_phenotype_mapping.json")
print("  - sample_phenotypes.geojson")
print("  - sample_feature_metrics.json")
print("  - sample_DAPI.tif, sample_PANCK.tif, sample_SMA.tif")
print("  - sample_channels.txt")
print("\nThese files can be used to test:")
print("  1. Full pipeline execution with -profile test")
print("  2. Individual process testing with nf-test")
print("  3. Input validation and error handling")
print("  4. Module-level unit tests")

# Pipeline Metadata and Reference Selection Improvements

## Summary of Changes

This document summarizes the improvements made to CSV handling, patient_id tracking, and reference image selection throughout the ATEIA pipeline.

## Issues Fixed

### 1. ✅ Registration Now Uses `is_reference` Metadata
**Previous Behavior:** Registration identified the reference image by matching filenames against `params.reg_reference_markers` (e.g., searching for files containing "DAPI" and "FITC").

**New Behavior:** Registration uses the `is_reference` boolean from the metadata to identify the reference image.

**Benefits:**
- More reliable and explicit reference selection
- No dependency on filename conventions
- Consistent with the input CSV design
- Works correctly when restarting from checkpoints

**Files Modified:**
- [subworkflows/local/registration.nf](subworkflows/local/registration.nf#L53-L83)

### 2. ✅ Metadata Preserved Through Registration
**Previous Behavior:** Metadata was stripped before registration, and the checkpoint CSV used a fallback to extract patient_id from filenames.

**New Behavior:**
- Metadata (`patient_id`, `is_reference`, `channels`) is preserved throughout the registration workflow
- Registered files are matched back to their original metadata
- Checkpoint CSV properly stores all metadata

**Benefits:**
- Full traceability of patient_id through the pipeline
- No reliance on filename conventions
- Accurate checkpoint CSVs

**Files Modified:**
- [subworkflows/local/registration.nf](subworkflows/local/registration.nf#L112-L136)
- [subworkflows/local/registration.nf](subworkflows/local/registration.nf#L257-L270) (checkpoint generation)

### 3. ✅ Postprocessing Uses `is_reference` Metadata
**Previous Behavior:** Postprocessing identified the reference image by matching filenames against `params.reg_reference_markers`.

**New Behavior:** Postprocessing uses the `is_reference` boolean from metadata.

**Benefits:**
- Consistent reference selection method across all pipeline steps
- Works correctly when restarting from checkpoints

**Files Modified:**
- [subworkflows/local/postprocess.nf](subworkflows/local/postprocess.nf#L42-L58)

### 4. ✅ Patient ID Added to Postprocessing Checkpoint
**Previous Behavior:** The postprocessing checkpoint CSV only contained file paths without patient_id or is_reference metadata.

**Format:**
```csv
phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
```

**New Behavior:** The checkpoint now includes patient_id and is_reference.

**Format:**
```csv
patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
P001,true,/results/phenotype.csv,/results/phenotype_mask.tif,...
```

**Benefits:**
- Full traceability in checkpoint CSVs
- Metadata available for downstream analysis
- Consistent with other checkpoint formats

**Files Modified:**
- [subworkflows/local/postprocess.nf](subworkflows/local/postprocess.nf#L81-L111)
- [main.nf](main.nf#L166-L180) (loading from checkpoint)

### 5. ✅ Metadata Flows Through All Pipeline Steps
**Previous Behavior:** Metadata was lost when transitioning between pipeline steps (registration → postprocessing).

**New Behavior:** All pipeline steps now pass `[meta, file]` tuples:
- Preprocessing → Registration: ✅ metadata preserved
- Registration → Postprocessing: ✅ metadata preserved
- Checkpoint loading: ✅ metadata reconstructed

**Files Modified:**
- [main.nf](main.nf#L100-L113) (registration step)
- [main.nf](main.nf#L115-L135) (postprocessing step)

### 6. ✅ Channel 3 Confirmed Optional
**Status:** Channel 3 (`channel_3` column) is correctly implemented as optional.

**Implementation:**
- CSV parsing checks if the column exists and is not empty: [main.nf](main.nf#L45-L49)
- Conversion script handles variable number of channels
- Documentation updated

## Checkpoint CSV Formats

### `preprocessed.csv`
```csv
patient_id,padded_image,is_reference,channels
P001,/results/P001/preprocessing/padded/P001_DAPI_FITC_padded.ome.tif,true,DAPI|FITC
P002,/results/P002/preprocessing/padded/P002_DAPI_FITC_padded.ome.tif,false,DAPI|FITC
```

### `registered.csv`
```csv
patient_id,registered_image,is_reference,channels
P001,/results/P001/registration/P001_DAPI_FITC_registered.ome.tif,true,DAPI|FITC
P002,/results/P002/registration/P002_DAPI_FITC_registered.ome.tif,false,DAPI|FITC
```

### `postprocessed.csv` (NEW FORMAT)
```csv
patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
P001,true,/results/phenotype.csv,/results/phenotype_mask.tif,/results/phenotype_mapping.csv,/results/merged.csv,/results/cell_mask.tif
```

## DAPI Position Flexibility (Bonus Improvement)

### Previous Requirement
DAPI had to be in `channel_1` position.

### New Behavior
DAPI can be in **any channel position** (channel_1, channel_2, or channel_3). The conversion process automatically moves DAPI to channel 0 in the output OME-TIFF.

**Examples:**
```csv
# All valid:
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/P001.nd2,true,DAPI,FITC,Texas Red      # DAPI in position 1
P002,/data/P002.nd2,true,FITC,DAPI,Texas Red      # DAPI in position 2
P003,/data/P003.nd2,true,FITC,Texas Red,DAPI      # DAPI in position 3
```

**Files Modified:**
- [main.nf](main.nf#L51-L55) (validation)
- [bin/convert_image.py](bin/convert_image.py) (conversion logic)
- [docs/INPUT_FORMAT.md](docs/INPUT_FORMAT.md#L25-L46) (documentation)

## Documentation Updates

Updated documentation files:
- [docs/INPUT_FORMAT.md](docs/INPUT_FORMAT.md) - Comprehensive updates to reflect all changes
  - Updated DAPI position requirements
  - Added postprocessed.csv format
  - Updated validation rules
  - Updated common errors section

## Testing Recommendations

1. **Test reference selection with `is_reference`:**
   - Verify reference image is correctly identified in registration
   - Verify reference image is correctly identified in postprocessing

2. **Test checkpoint restart:**
   - Start from `preprocessed.csv` → verify metadata preserved
   - Start from `registered.csv` → verify metadata preserved
   - Start from `postprocessed.csv` → verify patient_id and is_reference loaded correctly

3. **Test DAPI position flexibility:**
   - Test with DAPI in channel_1, channel_2, and channel_3
   - Verify output always has DAPI in channel 0

4. **Test optional channel_3:**
   - Test with 2-channel images (no channel_3)
   - Test with 3-channel images (with channel_3)

## Backward Compatibility Notes

### Breaking Changes
1. **Postprocessed checkpoint CSV format changed**
   - Old format: `phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask`
   - New format: `patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask`
   - **Impact:** Existing `postprocessed.csv` files from old pipeline runs will not work with new code when restarting from `--step results`

### Non-Breaking Changes
1. **Input CSV format:** No changes to input CSV format - fully backward compatible
2. **Preprocessed/registered checkpoints:** Format unchanged, fully compatible
3. **DAPI validation:** Now more flexible (accepts DAPI in any position vs. requiring channel_1)

## Migration Guide

If you have existing `postprocessed.csv` checkpoint files from previous pipeline runs:

1. **Option 1:** Re-run from an earlier checkpoint (recommended)
   ```bash
   nextflow run main.nf --input registered.csv --step postprocessing
   ```

2. **Option 2:** Manually update the CSV format
   Add `patient_id` and `is_reference` columns at the beginning of your existing `postprocessed.csv` file.

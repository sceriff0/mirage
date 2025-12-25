# Channel Handling and DAPI Positioning

## Overview

The ATEIA pipeline automatically places DAPI in **channel 0** of all output images, regardless of where DAPI appears in your input CSV or source images. This document explains how this works and what to expect.

## Key Principle

**DAPI can be specified in ANY position in your input CSV** (`channel_1`, `channel_2`, or `channel_3`), but will **ALWAYS end up in channel 0** in the output.

## How It Works

### 1. Input CSV (Flexible Order)

You can specify channels in any order, as long as DAPI is present:

```csv
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,data/P001.nd2,true,DAPI,FITC,Texas Red
P002,data/P002.nd2,false,FITC,DAPI,Texas Red
P003,data/P003.nd2,false,Texas Red,FITC,DAPI
```

All three examples above are **valid** - DAPI can be in position 1, 2, or 3.

### 2. Automatic Conversion

The `convert_image.py` script automatically:
1. Detects which position DAPI is in
2. Moves DAPI to channel 0
3. Preserves the relative order of other channels
4. Updates metadata accordingly

### 3. Output Files

All output OME-TIFF files will have DAPI in channel 0:

**Example transformations:**

| Input CSV Channels | Output OME-TIFF Channels | Checkpoint CSV |
|-------------------|-------------------------|----------------|
| `DAPI, FITC, Texas Red` | `[DAPI, FITC, Texas Red]` | `DAPI\|FITC\|Texas Red` |
| `FITC, DAPI, Texas Red` | `[DAPI, FITC, Texas Red]` | `DAPI\|FITC\|Texas Red` |
| `Texas Red, FITC, DAPI` | `[DAPI, Texas Red, FITC]` | `DAPI\|Texas Red\|FITC` |
| `FITC, Texas Red, DAPI` | `[DAPI, FITC, Texas Red]` | `DAPI\|FITC\|Texas Red` |

**Note:** When DAPI is moved, other channels maintain their original order relative to each other.

### 4. Checkpoint CSVs

All checkpoint CSVs store channels in the **output order** (DAPI first):

```csv
patient_id,padded_image,is_reference,channels
P001,/results/P001/padded.ome.tif,true,DAPI|FITC|Texas Red
P002,/results/P002/padded.ome.tif,false,DAPI|FITC|Texas Red
P003,/results/P003/padded.ome.tif,false,DAPI|Texas Red|FITC
```

## Why This Matters

### Consistent Segmentation
The segmentation module always uses **channel 0** for nuclei detection. By ensuring DAPI is always in channel 0, we guarantee:
- Nuclei are always segmented from the correct channel
- No manual configuration needed
- Works across different image sources

### Predictable Analysis
All downstream analysis can rely on channel 0 being DAPI:
- Quantification tools know where to find nuclei
- Visualization scripts can assume channel positions
- Analysis results are comparable across samples

## Complete Example

### Input CSV
```csv
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/P001.nd2,true,FITC,Texas Red,DAPI
```

### Processing Steps

1. **Conversion** (`convert_image.py`):
   - Reads input channels: `[FITC, Texas Red, DAPI]`
   - Detects DAPI at position 2 (index 2)
   - Moves DAPI to position 0
   - Output channels: `[DAPI, FITC, Texas Red]`
   - Writes `P001_channels.txt`: `DAPI,FITC,Texas Red`

2. **Preprocessing** subworkflow:
   - Reads `P001_channels.txt`
   - Updates metadata with output channel order
   - Passes through BaSiC correction and padding
   - Generates checkpoint CSV with `DAPI|FITC|Texas Red`

3. **All Output Files**:
   - `P001_DAPI_FITC_Texas Red.ome.tif` (converted)
   - `P001_DAPI_FITC_Texas Red_preprocessed.ome.tif`
   - `P001_DAPI_FITC_Texas Red_padded.ome.tif`
   - All have DAPI in channel 0

## Validation

The pipeline validates that DAPI is present but does NOT require it to be first:

✅ **Valid inputs:**
```csv
channel_1,channel_2,channel_3
DAPI,FITC,Texas Red     # DAPI first
FITC,DAPI,Texas Red     # DAPI second
Texas Red,FITC,DAPI     # DAPI third
```

❌ **Invalid input:**
```csv
channel_1,channel_2,channel_3
FITC,GFP,Texas Red      # No DAPI - ERROR!
```

## Image Format Handling

### ND2 Files
For ND2 files, the conversion also handles the legacy channel reversal:
1. Read ND2 file
2. Reverse channel order (legacy behavior)
3. Apply DAPI repositioning to output order
4. Save as OME-TIFF

### TIFF Files
For existing TIFF files:
1. Read TIFF file
2. Detect channel order
3. Apply DAPI repositioning
4. Save as OME-TIFF

### OME-TIFF Files
For existing OME-TIFF files:
1. Read OME metadata
2. Apply DAPI repositioning
3. Re-save with updated metadata

## Troubleshooting

### "DAPI channel not found"
**Error message:** `DAPI channel not found in: [FITC, GFP, Texas Red]`

**Cause:** No channel named "DAPI" in the CSV

**Solution:** Ensure one channel column contains exactly "DAPI" (case-insensitive)

### Channel count mismatch
**Warning:** `Channel count mismatch: image has 3, specified 2`

**Cause:** Number of channel columns in CSV doesn't match image channels

**Solution:** Add or remove channel columns to match your image

### Unexpected channel order in output
**Issue:** Output channels not in expected order

**Explanation:** This is normal! The pipeline preserves relative order of non-DAPI channels after moving DAPI to position 0.

**Example:**
- Input: `Texas Red, FITC, DAPI`
- Output: `DAPI, Texas Red, FITC` (not `DAPI, FITC, Texas Red`)

If you need a specific order, adjust your input CSV accordingly.

## Best Practices

1. **Be Consistent:** Use the same channel order across all samples in a cohort for easier comparison

2. **Document Your Order:** Note the original channel positions in your metadata

3. **Check Outputs:** After first conversion, verify channel order in output OME-TIFF matches expectations

4. **Use Checkpoint CSVs:** When restarting, checkpoint CSVs always have DAPI-first order

## Summary

- ✅ DAPI can be in **any position** in input CSV
- ✅ DAPI will **always be in channel 0** in outputs
- ✅ Other channels preserve their relative order
- ✅ Checkpoint CSVs store the output order (DAPI first)
- ✅ All image formats supported (ND2, TIFF, OME-TIFF)

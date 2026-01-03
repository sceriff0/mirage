# Input CSV Format Guide

The ATEIA pipeline uses a standardized CSV format for input that supports patient tracking, multiple image formats, and flexible channel configuration.

## CSV Schema

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | String | Unique identifier for the patient/sample | `P001`, `patient_123` |
| `path_to_file` | Path | Absolute or relative path to image file | `/data/images/sample1.nd2` |
| `is_reference` | Boolean | Whether this image is the reference for registration | `true`, `false` |
| `channel_1` | String | First channel name (DAPI required but can be in any position) | `DAPI`, `FITC` |
| `channel_2` | String | Second channel name | `FITC`, `GFP`, etc. |

### Optional Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `channel_3` | String | Third channel name (optional) | `Texas Red`, `Cy5` |

## Important Rules

### 1. **DAPI Required (Flexible Position)**
DAPI **MUST** be present in one of the channel columns, but it can be in **any position** (channel_1, channel_2, or channel_3). The conversion process will automatically move DAPI to channel 0.

```csv
✅ ALL CORRECT - DAPI can be in any position:
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/P001.nd2,true,DAPI,FITC,Texas Red
P002,/data/P002.nd2,true,FITC,DAPI,Texas Red
P003,/data/P003.nd2,true,FITC,Texas Red,DAPI

❌ INCORRECT (no DAPI):
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/P001.nd2,true,FITC,GFP,Texas Red
```

### 2. **Automatic Channel Reordering**
The conversion process automatically places DAPI in **channel 0** (array index 0) regardless of its position in the CSV. Other channels maintain their relative order.

**Examples:**
- CSV: `FITC, DAPI, Texas Red` → Output OME-TIFF: `[DAPI, FITC, Texas Red]`
- CSV: `FITC, Texas Red, DAPI` → Output OME-TIFF: `[DAPI, FITC, Texas Red]`
- CSV: `DAPI, FITC, Texas Red` → Output OME-TIFF: `[DAPI, FITC, Texas Red]` (no change needed)

### 3. **Supported Image Formats**
The pipeline automatically detects and converts from:
- **ND2** (`.nd2`) - Nikon microscopy format
- **TIFF** (`.tif`, `.tiff`) - Standard TIFF files
- **OME-TIFF** (`.ome.tif`, `.ome.tiff`) - Already in OME format

Format is inferred from the file extension.

### 4. **Reference Image Selection (Uses `is_reference` Metadata)**
Exactly **one** image should have `is_reference=true`. This image is used as the reference for:
- Image registration (all other images aligned to this) - **Uses `is_reference` metadata, not filename matching**
- Segmentation (nuclei segmented from DAPI channel of reference)

## Example Input CSV

### Minimal Example (2 channels)

```csv
patient_id,path_to_file,is_reference,channel_1,channel_2
P001,/data/images/P001_DAPI_FITC.nd2,true,DAPI,FITC
P002,/data/images/P002_DAPI_FITC.nd2,false,DAPI,FITC
P003,/data/images/P003_DAPI_FITC.nd2,false,DAPI,FITC
```

### Full Example (3 channels)

```csv
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/cohort1/P001.nd2,true,DAPI,FITC,Texas Red
P002,/data/cohort1/P002.nd2,false,DAPI,FITC,Texas Red
P003,/data/cohort2/P003.tif,false,DAPI,GFP,Cy5
```

### Mixed Formats Example

```csv
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/nd2/P001.nd2,true,DAPI,FITC,Texas Red
P002,/data/tiff/P002.tif,false,DAPI,FITC,Texas Red
P003,/data/ome/P003.ome.tif,false,DAPI,FITC,Texas Red
```

## Output File Naming

After conversion, files are named following this pattern:
```
{patient_id}_{channel_1}_{channel_2}_{channel_3}.ome.tif
```

**Examples:**
- Input: `P001`, channels: `DAPI, FITC`
- Output: `P001_DAPI_FITC.ome.tif`

- Input: `patient_123`, channels: `DAPI, GFP, Texas Red`
- Output: `patient_123_DAPI_GFP_Texas Red.ome.tif`

## Checkpoint CSV Formats

After preprocessing, the pipeline generates checkpoint CSVs that maintain patient metadata:

### `preprocessed.csv`

```csv
patient_id,padded_image,is_reference,channels
P001,/results/P001/preprocessing/padded/P001_DAPI_FITC_padded.ome.tif,true,DAPI|FITC
P002,/results/P002/preprocessing/padded/P002_DAPI_FITC_padded.ome.tif,false,DAPI|FITC
```

**Note:** Channels are pipe-separated (`|`) in checkpoint CSVs.

### `registered.csv`

```csv
patient_id,registered_image,is_reference,channels
P001,/results/P001/registration/P001_DAPI_FITC_registered.ome.tif,true,DAPI|FITC
P002,/results/P002/registration/P002_DAPI_FITC_registered.ome.tif,false,DAPI|FITC
```

### `postprocessed.csv`

```csv
patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
P001,true,/results/phenotype.csv,/results/phenotype_mask.tif,/results/phenotype_mapping.csv,/results/merged.csv,/results/cell_mask.tif
```

**Note:** The postprocessing checkpoint includes patient_id and is_reference metadata along with paths to all postprocessing output files (phenotype results, masks, quantification CSVs).

## Usage with Pipeline

### Starting from Raw Images

```bash
nextflow run main.nf \
  --input samples.csv \
  --outdir results \
  --step preprocessing
```

Where `samples.csv` follows the input format described above.

### Restarting from Checkpoint

```bash
nextflow run main.nf \
  --input results/csv/preprocessed.csv \
  --step registration \
  --outdir results \
  --registration_method cpu_multires
```

## Validation

The pipeline validates:
1. ✅ DAPI is present in at least one channel column
2. ✅ Patient IDs are present
3. ✅ Image files exist
4. ✅ Channel count matches image data
5. ✅ Exactly one reference image (at preprocessing step)

Validation errors will stop the pipeline with a descriptive message.

## Common Errors

### Error: "DAPI channel not found"
**Cause:** None of the channel columns contain DAPI

**Solution:** Ensure DAPI appears in one of the `channel_1`, `channel_2`, or `channel_3` columns

### Error: "Channel count mismatch"
**Cause:** Number of `channel_*` columns doesn't match image channels

**Solution:**
- For 2-channel images: only specify `channel_1` and `channel_2`
- For 3-channel images: specify `channel_1`, `channel_2`, and `channel_3`

### Error: "Channel 'X' not found in image"
**Cause:** Channel name in CSV doesn't match actual image channels

**Solution:** Verify channel names match your image metadata (case-sensitive)

## Best Practices

1. **Use Absolute Paths**: Absolute paths in `path_to_file` ensure portability

2. **Consistent Channel Names**: Use the same channel names across all samples in a cohort

3. **Reference Selection**: Choose the reference image with:
   - Best image quality
   - Clearest DAPI signal
   - Representative of the cohort

4. **Patient ID Format**: Use consistent, descriptive patient IDs without special characters

5. **Channel Naming**: Use standard names (DAPI, FITC, Texas Red, Cy5, GFP, etc.) for compatibility

## Advanced: Manual Channel Mapping

If your image has channels in an unusual order, the pipeline will automatically remap them according to your CSV specification. The conversion script:

1. Reads the original image
2. Identifies channel positions
3. Extracts channels in the order specified in the CSV
4. Places DAPI in position 0
5. Saves as standardized OME-TIFF with correct metadata

This ensures all downstream analysis receives consistently ordered channels.

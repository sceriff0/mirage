# Pipeline Restartability Guide

The ATEIA pipeline implements a **checkpoint-based restart system** that is independent of Nextflow's `-resume` functionality. This allows you to restart the pipeline from any major step, even after deleting the `work/` directory or moving to a different machine.

## How It Works

At each major pipeline step, the pipeline automatically generates **checkpoint CSV files** that contain absolute paths to all intermediate results. These CSVs are saved to `results/csv/` and can be used as input to restart from any step.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│preprocessing │────▶│ registration │────▶│postprocessing│────▶│   results    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  preprocessed.csv     registered.csv     postprocessed.csv
  + padded images      + registered imgs  + phenotype data
```

## Pipeline Steps

The pipeline defines four major steps as entry points:

| Step | Description | Starting Point |
|------|-------------|----------------|
| `preprocessing` | (Default) Start from raw ND2/TIFF files | Raw image files |
| `registration` | Start from preprocessed, padded images | `preprocessed.csv` |
| `postprocessing` | Start from registered images | `registered.csv` |
| `results` | Start from quantified/phenotyped data | `postprocessed.csv` |

## Checkpoint CSV Formats

### 1. `preprocessed.csv`
Generated after: Image conversion, BaSiC correction, and padding

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier |
| `padded_image` | Absolute path to padded OME-TIFF image |
| `is_reference` | Whether this is the reference image |
| `channels` | Pipe-separated channel names (DAPI first) |

**Example:**
```csv
patient_id,padded_image,is_reference,channels
P001,/path/to/results/P001/preprocessing/padded/P001_DAPI_FITC_padded.ome.tif,true,DAPI|FITC
P002,/path/to/results/P002/preprocessing/padded/P002_DAPI_FITC_padded.ome.tif,false,DAPI|FITC
```

### 2. `registered.csv`
Generated after: Image registration (VALIS, GPU, or CPU methods)

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier |
| `registered_image` | Absolute path to registered OME-TIFF image |
| `is_reference` | Whether this is the reference image |
| `channels` | Pipe-separated channel names |

**Example:**
```csv
patient_id,registered_image,is_reference,channels
P001,/path/to/results/P001/registration/P001_DAPI_FITC_registered.ome.tif,true,DAPI|FITC
P002,/path/to/results/P002/registration/P002_DAPI_FITC_registered.ome.tif,false,DAPI|FITC
```

### 3. `postprocessed.csv`
Generated after: Segmentation, quantification, and phenotyping

| Column | Description |
|--------|-------------|
| `phenotype_csv` | Absolute path to phenotype CSV file |
| `phenotype_mask` | Absolute path to phenotype mask image |
| `phenotype_mapping` | Absolute path to phenotype mapping JSON |
| `merged_csv` | Absolute path to merged quantification CSV |
| `cell_mask` | Absolute path to cell segmentation mask |

**Example:**
```csv
phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
/path/to/results/phenotype.csv,/path/to/results/phenotype_mask.tif,/path/to/results/phenotype_mapping.json,/path/to/results/merged.csv,/path/to/results/cell_mask.tif
```

## Usage Examples

### Example 1: Full Pipeline Run (Default)

Run the complete pipeline from a CSV listing your images:

First, create an input CSV (`samples.csv`):
```csv
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/images/P001.nd2,true,DAPI,FITC,Texas Red
P002,/data/images/P002.nd2,false,DAPI,FITC,Texas Red
P003,/data/images/P003.nd2,false,DAPI,FITC,Texas Red
```

Then run the pipeline:
```bash
nextflow run main.nf \
  --input samples.csv \
  --outdir results \
  --registration_method gpu
```

This will:
1. Convert images to standardized OME-TIFF (DAPI in channel 0)
2. Run all steps: preprocessing → registration → postprocessing → results
3. Generate checkpoint CSVs at each step: `preprocessed.csv`, `registered.csv`, `postprocessed.csv`

**Important:** See [INPUT_FORMAT.md](INPUT_FORMAT.md) for detailed CSV format specification

---

### Example 2: Restart from Registration (After Deleting work/)

Suppose you completed preprocessing but want to try a different registration method:

```bash
# Delete the work directory (optional)
rm -rf work/

# Restart from registration step using the preprocessing checkpoint
nextflow run main.nf \
  --input results/csv/preprocessed.csv \
  --step registration \
  --outdir results \
  --registration_method cpu_multires  # Try a different method!
```

This will:
- Skip preprocessing
- Load padded images from `preprocessed.csv`
- Run registration, postprocessing, and results

---

### Example 3: Restart from Postprocessing

You want to change segmentation or phenotyping parameters:

```bash
nextflow run main.nf \
  --input results/csv/registered.csv \
  --step postprocessing \
  --outdir results \
  --seg_model cellpose \
  --seg_gamma 0.7  # Adjust parameters
```

This will:
- Skip preprocessing and registration
- Load registered images from `registered.csv`
- Run postprocessing and results with new parameters

---

### Example 4: Restart from Results Only

You only want to regenerate pyramid TIFFs with different parameters:

```bash
nextflow run main.nf \
  --input results/csv/postprocessed.csv \
  --step results \
  --outdir results \
  --pyramid_resolutions 6  # More pyramid levels
  --tilex 1024             # Larger tile size
```

This will:
- Skip all analysis steps
- Load phenotyped data from `postprocessed.csv`
- Only run results conversion

---

## Key Features

### 1. **Work Directory Independence**
Checkpoint CSVs use **absolute paths** to published results, so you can:
- Delete the `work/` directory to save space
- Move to a different machine (if paths are accessible)
- Run multiple analyses from the same checkpoints

### 2. **Automatic CSV Generation**
Checkpoint CSVs are automatically created at the end of each major step - you don't need to do anything special.

### 3. **Parameter Flexibility**
When restarting from a checkpoint, you can change:
- Registration methods (`--registration_method`)
- Segmentation parameters (`--seg_*`)
- Quantification/phenotyping parameters (`--pheno_*`, `--quant_*`)
- Output formats (`--pyramid_*`, `--tilex`, `--tiley`)

### 4. **Storage Trade-off**
This approach requires more storage (intermediate files are published), but provides:
- Robust restartability
- No dependence on cache integrity
- Easy sharing of intermediate results

---

## Best Practices

1. **Keep Checkpoint CSVs**: Don't delete files in `results/csv/` - they enable restart capability

2. **Preserve Published Results**: Intermediate results in `results/preprocessing/`, `results/registration/`, etc. are needed for restarts

3. **Use Absolute Paths**: Checkpoint CSVs contain absolute paths, so ensure paths remain valid if moving between systems

4. **Clean Work Directory**: After successful runs, you can safely delete `work/` to save space - restarts will work from published results

5. **Experiment with Parameters**: Use restarts to efficiently test different parameters without re-running early steps

---

## Troubleshooting

### Error: "Missing columns for step X"
**Cause:** The input CSV doesn't have the required columns for the step.

**Solution:** Make sure you're using the correct checkpoint CSV:
- For `--step registration`, use `preprocessed.csv`
- For `--step postprocessing`, use `registered.csv`
- For `--step results`, use `postprocessed.csv`

### Error: "File not found" when loading from CSV
**Cause:** The paths in the checkpoint CSV are no longer valid.

**Solution:**
- Verify the files still exist at the specified paths
- If you moved results, update the paths in the CSV
- Re-run from an earlier step to regenerate missing files

### Checkpoint CSV not generated
**Cause:** Pipeline may have failed before checkpoint creation.

**Solution:**
- Check pipeline logs for errors
- Ensure the pipeline completed the step successfully
- Look for the CSV in `results/csv/`

---

## Implementation Notes

This restartability implementation follows the design principles from nf-core/sarek:
- Checkpoint CSVs are generated in Nextflow (not external scripts)
- Published results use predictable directory structures
- Step validation ensures correct inputs
- Metadata preservation through CSV columns

For more details on the underlying architecture, see the [restartability design document](../restartability.md).

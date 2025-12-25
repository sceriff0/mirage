# ATEIA Pipeline Documentation

This directory contains comprehensive documentation for the ATEIA Whole Slide Imaging (WSI) processing pipeline.

## Quick Start

1. **Prepare your input CSV** following the format in [INPUT_FORMAT.md](INPUT_FORMAT.md)
2. **Run the pipeline** with your configuration
3. **Use restart capability** as described in [RESTARTABILITY.md](RESTARTABILITY.md)

## Documentation Index

### [INPUT_FORMAT.md](INPUT_FORMAT.md)
**Essential reading for all users**

Describes the standardized CSV input format for the pipeline:
- Required and optional columns
- Patient ID tracking
- Channel ordering rules (DAPI must be first)
- Supported image formats (ND2, TIFF, OME-TIFF)
- Reference image selection
- Complete examples

**When to read:** Before running the pipeline for the first time

---

### [RESTARTABILITY.md](RESTARTABILITY.md)
**Learn how to restart the pipeline from any step**

Explains the checkpoint-based restart system:
- How checkpoint CSVs work
- Restarting from any major pipeline step
- Checkpoint CSV formats
- Work directory independence
- Usage examples and best practices

**When to read:**
- When you want to try different parameters without re-running early steps
- After pipeline failures or interruptions
- When moving analyses between systems

---

## Key Concepts

### Patient-Centric Processing
The pipeline tracks samples by `patient_id` throughout all steps. This enables:
- Organized output directories per patient
- Easy tracking of results
- Flexible restart from checkpoints while maintaining provenance

### DAPI-First Channel Ordering
All images are standardized with **DAPI in channel 0** (array index 0), regardless of the original image channel order. This ensures:
- Consistent segmentation (always uses channel 0)
- Predictable downstream analysis
- Compatibility across different image sources

### Checkpoint-Based Restartability
The pipeline automatically generates CSV checkpoints at each major step:
```
preprocessing → preprocessed.csv
registration → registered.csv
postprocessing → postprocessed.csv
```

These checkpoints enable:
- Restarting from any step
- Parameter experimentation
- Work directory cleanup (save space)
- Recovery from failures

## Common Workflows

### First-Time Full Run

```bash
# 1. Create input CSV (see INPUT_FORMAT.md)
cat > samples.csv <<EOF
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/P001.nd2,true,DAPI,FITC,Texas Red
P002,/data/P002.nd2,false,DAPI,FITC,Texas Red
EOF

# 2. Run full pipeline
nextflow run main.nf \
  --input samples.csv \
  --outdir results \
  --registration_method gpu
```

### Restart from Registration

```bash
# Try a different registration method
nextflow run main.nf \
  --input results/csv/preprocessed.csv \
  --step registration \
  --outdir results \
  --registration_method cpu_multires
```

### Restart from Segmentation

```bash
# Adjust segmentation parameters
nextflow run main.nf \
  --input results/csv/registered.csv \
  --step postprocessing \
  --outdir results \
  --seg_gamma 0.7
```

## Pipeline Steps

| Step | What It Does | Input | Output |
|------|--------------|-------|--------|
| **preprocessing** | Convert → BaSiC correction → Padding | User CSV | `preprocessed.csv` + padded images |
| **registration** | Align images to reference | `preprocessed.csv` | `registered.csv` + aligned images |
| **postprocessing** | Segment → Quantify → Phenotype | `registered.csv` | `postprocessed.csv` + analysis data |
| **results** | Generate pyramid TIFFs | `postprocessed.csv` | Final visualizations |

## File Organization

After running the pipeline, your output directory structure will be:

```
results/
├── csv/                          # Checkpoint CSVs for restart
│   ├── preprocessed.csv
│   ├── registered.csv
│   └── postprocessed.csv
├── P001/                         # Per-patient results
│   ├── gpu/                      # Per-method subdirectory
│   │   ├── converted/            # Standardized OME-TIFF
│   │   ├── preprocessing/        # Padded images
│   │   ├── registration/         # Registered images
│   │   └── analysis/             # Segmentation, quantification
│   └── ...
├── P002/
│   └── ...
└── ...
```

## Input Requirements

### Minimal (2 channels)
```csv
patient_id,path_to_file,is_reference,channel_1,channel_2
P001,/data/P001.nd2,true,DAPI,FITC
```

### Full (3 channels)
```csv
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,/data/P001.nd2,true,DAPI,FITC,Texas Red
```

**Critical Rules:**
- ✅ `channel_1` **MUST** be `DAPI`
- ✅ Exactly one image with `is_reference=true`
- ✅ Patient IDs must be unique
- ✅ File paths must be valid

## Troubleshooting

### "First channel must be DAPI"
**Fix:** Ensure `channel_1` column contains `DAPI` for all rows in your input CSV

### "Channel count mismatch"
**Fix:** Number of `channel_*` columns must match the actual channels in your images

### "No reference image found"
**Fix:** Set exactly one image to `is_reference=true` in your input CSV

### Pipeline stops during preprocessing
**Solution:** Check image file paths are correct and files are readable

### Want to change parameters mid-run
**Solution:** Use restart capability (see RESTARTABILITY.md) to continue from a checkpoint with new parameters

## Best Practices

1. **Start Small:** Test with 2-3 images before running large cohorts
2. **Use Absolute Paths:** In input CSV for portability
3. **Check Reference Image:** Choose highest quality image with clear DAPI signal
4. **Save Checkpoints:** Don't delete `results/csv/` directory
5. **Clean Work Directory:** After successful runs, delete `work/` to save space
6. **Document Parameters:** Keep notes on which parameters worked best

## Getting Help

- **Input format questions:** See [INPUT_FORMAT.md](INPUT_FORMAT.md)
- **Restart questions:** See [RESTARTABILITY.md](RESTARTABILITY.md)
- **Pipeline design:** See [../restartability.md](../restartability.md) for architectural details

## Advanced Topics

### Custom Channel Names
The pipeline supports any channel names, as long as DAPI is first:
```csv
patient_id,path_to_file,is_reference,channel_1,channel_2,channel_3
P001,data.nd2,true,DAPI,Custom-Ab-1,Custom-Ab-2
```

### Mixed Image Formats
You can mix ND2, TIFF, and OME-TIFF in the same run:
```csv
patient_id,path_to_file,is_reference,channel_1,channel_2
P001,data/P001.nd2,true,DAPI,FITC
P002,data/P002.tif,false,DAPI,FITC
P003,data/P003.ome.tif,false,DAPI,FITC
```

The conversion script automatically detects format from file extension.

### Checkpoint CSV Structure
Checkpoint CSVs maintain metadata using:
- `patient_id`: Track samples
- `is_reference`: Identify reference image
- `channels`: Pipe-separated list (e.g., `DAPI|FITC|Texas Red`)

This enables full pipeline restart with complete context.

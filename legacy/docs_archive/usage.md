# ATEIA Pipeline Usage Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Input Preparation](#input-preparation)
3. [Running the Pipeline](#running-the-pipeline)
4. [Checkpoint-Based Restart](#checkpoint-based-restart)
5. [Registration Methods](#registration-methods)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Run (Full Pipeline)

```bash
nextflow run main.nf \
  --input input.csv \
  --outdir ./results \
  --registration_method gpu \
  -profile slurm
```

### Test Run

```bash
nextflow run main.nf -profile test
```

### Show Help

```bash
nextflow run main.nf --help
```

---

## Input Preparation

### Input CSV Format

The pipeline accepts a CSV file with patient metadata. The required columns depend on the starting step:

#### For Preprocessing Step (Full Pipeline)

```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001_image.nd2,true,FITC|DAPI|Texas Red
P002,/data/P002_image.nd2,false,FITC|DAPI|Texas Red
```

**Required Columns:**
- `patient_id`: Unique patient identifier
- `path_to_file`: Absolute path to image file (ND2 or TIFF)
- `is_reference`: `true` or `false` - designates reference image for registration
- `channels`: Pipe-delimited channel names (DAPI can be in any position)

**Important Notes:**
- At least one image per patient must have `is_reference=true`
- DAPI channel must be present in all images
- Channel names should match across all images for a patient

#### For Registration Step (Resume from Preprocessing)

```csv
patient_id,preprocessed_image,is_reference,channels
P001,/results/P001/preprocess/P001_corrected.ome.tif,true,DAPI|FITC|Texas Red
```

See [docs/INPUT_FORMAT.md](INPUT_FORMAT.md) for complete specification.

---

## Running the Pipeline

### Execution Profiles

#### SLURM Cluster (Default)

```bash
nextflow run main.nf \
  --input input.csv \
  --outdir results \
  --slurm_partition gpu_partition \
  --slurm_account my_account \
  -profile slurm
```

#### Local Execution

```bash
nextflow run main.nf \
  --input input.csv \
  --outdir results \
  -profile local
```

#### Docker (Local Development)

```bash
nextflow run main.nf \
  --input input.csv \
  --outdir results \
  -profile docker
```

### Key Parameters

#### Registration Method

```bash
--registration_method valis  # VALIS batch registration (high memory)
--registration_method valis_pairs
--registration_method gpu    # GPU pairwise registration (recommended)
--registration_method cpu    # CPU pairwise registration (slower)
--registration_method cpu_tiled
```

#### GPU Configuration

```bash
--seg_gpu true              # Use GPU for segmentation
--gpu_type nvidia_h200:1    # Specify GPU type for SLURM
```

#### Resource Limits

```bash
--max_memory 700.GB
--max_cpus 64
--max_time 240.h
```

---

## Checkpoint-Based Restart

The pipeline generates checkpoint CSVs at each major step, enabling restart from any point without re-running completed steps.

### Step 1: Preprocessing

**Run:**
```bash
nextflow run main.nf \
  --input raw_images.csv \
  --step preprocessing \
  --outdir results
```

**Output:** `results/csv/preprocessed.csv`

### Step 2: Registration

**Run:**
```bash
nextflow run main.nf \
  --input results/csv/preprocessed.csv \
  --step registration \
  --registration_method gpu \
  --outdir results
```

**Output:** `results/csv/registered.csv`

### Step 3: Postprocessing

**Run:**
```bash
nextflow run main.nf \
  --input results/csv/registered.csv \
  --step postprocessing \
  --outdir results
```

**Output:** `results/csv/postprocessed.csv`

### Step 4: Copy Results

**Run:**
```bash
nextflow run main.nf \
  --step copy_results \
  --savedir /final/archive \
  --outdir results
```

See [docs/RESTARTABILITY.md](RESTARTABILITY.md) for details.

---

## Registration Methods

### VALIS (Batch Registration)

Best for: Multiple images per patient, SuperPoint feature detection preferred

```bash
--registration_method valis \
--reg_max_image_dim 4000 \
--reg_micro_reg_fraction 0.125 \
--skip_micro_registration true
```

**Memory Requirements:** 256-512 GB per patient
**Processing Time:** 2-6 hours per patient (4-8 images)

### GPU Registration (Recommended)

Best for: Large images, faster processing, lower memory

```bash
--registration_method gpu \
--gpu_reg_affine_crop_size 10000 \
--gpu_reg_diffeo_crop_size 2000 \
--gpu_reg_overlap_percent 40.0
```

**Memory Requirements:** 128-256 GB per image
**Processing Time:** 1-3 hours per image
**GPU Required:** 1x NVIDIA H200 or A100 (40GB+ VRAM)

### CPU Registration

Best for: No GPU available, smaller images

```bash
--registration_method cpu \
--cpu_reg_affine_crop_size 10000 \
--cpu_reg_diffeo_crop_size 2000
```

**Memory Requirements:** 64-128 GB per image
**Processing Time:** 3-8 hours per image

See [docs/PARAMETER_MAPPING.md](PARAMETER_MAPPING.md) for parameter details.

---

## Common Use Cases

### Use Case 1: Full Pipeline with GPU

```bash
nextflow run main.nf \
  --input samples.csv \
  --outdir results \
  --registration_method gpu \
  --seg_gpu true \
  --slurm_partition gpu_queue \
  -profile slurm
```

### Use Case 2: Resume Failed Registration

If registration failed for some patients:

1. Check `results/csv/preprocessed.csv`
2. Remove successfully registered patients
3. Re-run:

```bash
nextflow run main.nf \
  --input preprocessed_failed.csv \
  --step registration \
  --registration_method gpu \
  --outdir results \
  -profile slurm
```

### Use Case 3: Re-quantify with Different Parameters

Skip preprocessing and registration:

```bash
nextflow run main.nf \
  --input results/csv/registered.csv \
  --step postprocessing \
  --phenotype_config assets/phenotype_config.json \
  --outdir results_new_phenotypes
```

### Use Case 4: QC Only

Generate registration QC without full processing:

```bash
nextflow run main.nf \
  --input samples.csv \
  --step preprocessing \
  --skip_registration_qc false \
  --qc_scale_factor 0.25 \
  --outdir qc_results
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**VALIS Registration:**
- Reduce `reg_max_image_dim`: `6000` → `4000`
- Process fewer images per batch

**GPU Registration:**
- Reduce `gpu_reg_affine_crop_size`: `10000` → `8000`
- Reduce `gpu_reg_diffeo_crop_size`: `2000` → `1500`
- Increase `gpu_reg_overlap_percent` slightly (better blending, more memory)

**Segmentation:**
- Set `seg_whole_image false`
- Reduce `seg_n_tiles_x` and `seg_n_tiles_y`

### Registration Quality Issues

**Low alignment accuracy:**
- Increase `gpu_reg_n_features`: `5000` → `7000`
- Use multi-resolution CPU method for difficult cases
- Check QC images in `results/registration_qc/`

**Tile artifacts:**
- Increase `gpu_reg_overlap_percent`: `40.0` → `50.0`
- Reduce crop size for smoother blending

### Missing Reference Image

Error: "No reference image found for patient X"

**Solution:**
- Ensure at least one image per patient has `is_reference=true`
- Or enable auto-reference: `--allow_auto_reference true`

### Container/Singularity Issues

**Singularity bind errors:**
```bash
# Add bind paths in profile
singularity.runOptions = '--bind /data --bind /models'
```

**Permission denied in container:**
```bash
docker.runOptions = '-u $(id -u):$(id -g)'
```

### Nextflow Resume Issues

If `-resume` doesn't work:
- Clear work directory: `rm -rf work/`
- Use checkpoint CSVs instead (more reliable)
- Check `.nextflow.log` for cache issues

---

## Additional Resources

- [Input Format Specification](INPUT_FORMAT.md)
- [Output Structure](output.md)
- [Parameter Reference](PARAMETER_MAPPING.md)
- [Registration Error Estimation](REGISTRATION_ERROR_ESTIMATION.md)
- [SLURM Configuration](SLURM_GUIDE.md)

<div align="center">
<img src="https://github.com/sceriff0/ateia/blob/main/assets/logo.png?raw=true" alt="alt text" width="600" height="600">
</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Nextflow-≥23.04.0-green?style=for-the-badge&logo=nextflow" alt="Nextflow"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-SLURM-orange?style=for-the-badge" alt="SLURM"/></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-CUDA-76B900?style=for-the-badge&logo=nvidia" alt="CUDA"/></a>
</p>

<p align="center">
  <b>A Nextflow pipeline for whole slide image preprocessing, registration, segmentation and quantification</b>
</p>

---

## 📋 Table of Contents

- [Pipeline Overview](#-pipeline-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#️-configuration)
- [Registration Methods](#-registration-methods)
- [Segmentation](#-segmentation)
- [Adding Custom Registration Methods](#-adding-custom-registration-methods)
- [TODOs](#-todos)

---

## 🔬 Pipeline Overview

![alt text](https://github.com/sceriff0/ateia/blob/main/assets/overview.png)
---

## 📦 Installation

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Nextflow | ≥23.04.0 | `curl -s https://get.nextflow.io \| bash` |
| Singularity | ≥3.8.0 | Or Docker for local development |
| Java | ≥11 | Required by Nextflow |

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ateia.git
cd ateia

# 2. Install Nextflow (if not already installed)
curl -s https://get.nextflow.io | bash
chmod +x nextflow
sudo mv nextflow /usr/local/bin/

# 3. Test installation
nextflow run main.nf --help
```

### Container Images

All dependencies are containerized. Images are automatically pulled on first run:

| Container | Purpose |
|-----------|---------|
| `bolt3x/attend_image_analysis:preprocess` | BaSiC correction, image conversion |
| `cdgatenbee/valis-wsi:1.0.0` | VALIS registration |
| `bolt3x/attend_image_analysis:debug_diffeo` | GPU/CPU diffeomorphic registration |
| `bolt3x/attend_image_analysis:segmentation_gpu` | StarDist segmentation |
| `bolt3x/attend_image_analysis:quantification_gpu` | Cell quantification |

---

## 🚀 Quick Start

### 1. Prepare Input CSV

Create a CSV file with your image metadata:

```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001_panel1.nd2,true,DAPI|CD3|CD8|CD4
P001,/data/P001_panel2.nd2,false,DAPI|PANCK|SMA|VIMENTIN
P002,/data/P002_panel1.nd2,true,DAPI|CD3|CD8|CD4
P002,/data/P002_panel2.nd2,false,DAPI|PANCK|SMA|VIMENTIN
```

> **Note:** Each patient should have exactly one `is_reference=true` image. DAPI will always be moved to the first channel during preprocessing.

### 2. Run with Defaults

```bash
nextflow run main.nf \
    --input samples.csv \
    --outdir ./results \
    --registration_method valis
```

### 3. SLURM Cluster Submission

Create a submission script (`submit.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=ateia
#SBATCH --output=ateia_%j.log
#SBATCH --error=ateia_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Load required modules (adjust for your cluster)
module load singularity/3.8.0
module load java/11

# Run pipeline
nextflow run main.nf \
    --input /path/to/samples.csv \
    --outdir /scratch/results \
    --registration_method gpu \
    -profile slurm \
```

Submit:
```bash
sbatch submit.sh
```

### 4. Resume from Checkpoint

The pipeline generates checkpoint CSVs at each major step. To resume from registration:

```bash
nextflow run main.nf \
    --input results/csv/preprocessed.csv \
    --step registration \
    --outdir ./results \
```

---

## ⚙️ Configuration

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | *required* | Input CSV or glob pattern |
| `--outdir` | `./results` | Output directory |
| `--step` | `preprocessing` | Entry point: `preprocessing`, `registration`, `postprocessing` |
| `--registration_method` | `valis` | Registration method: `valis`, `valis_pairs`, `gpu`, `cpu` |

### Preprocessing Parameters

```groovy
params {
    // BaSiC illumination correction
    preproc_tile_size    = 1950      // FOV size for BaSiC tiling
    preproc_skip_dapi    = true      // Skip correction for DAPI channel
    preproc_autotune     = false     // Auto-tune BaSiC parameters
    preproc_n_iter       = 3         // Number of iterations
    preproc_pool_workers = 4         // Parallel workers
}
```

### Registration Parameters

```groovy
params {
    // VALIS (batch registration)
    reg_max_processed_dim  = 512     // Resolution for rigid registration
    reg_max_non_rigid_dim  = 2048    // Resolution for non-rigid registration  
    reg_micro_reg_fraction = 0.125   // Micro-registration size fraction
    reg_num_features       = 5000    // SuperPoint features to detect
    skip_micro_registration = false  // Skip micro-rigid step (faster)
    
    // GPU registration (affine + diffeomorphic)
    gpu_reg_affine_crop_size = 10000 // Crop size for affine stage
    gpu_reg_diffeo_crop_size = 2000  // Crop size for diffeomorphic stage
    gpu_reg_overlap_percent  = 40.0  // Overlap between crops (%)
    gpu_reg_opt_tol          = 1e-6  // Optimization tolerance
    
    // CPU registration (same parameters, multi-threaded)
    cpu_reg_affine_crop_size = 10000
    cpu_reg_diffeo_crop_size = 2000
    cpu_reg_overlap_percent  = 40.0
    
    // Padding (for GPU/CPU methods)
    padding = false                   // Pad images to uniform size
}
```

### Segmentation Parameters

```groovy
params {
    seg_gpu             = true       // Use GPU for segmentation
    seg_n_tiles_y       = 16         // Tiling for large images
    seg_n_tiles_x       = 16
    seg_expand_distance = 10         // Nuclei → cell expansion (px)
    seg_pmin            = 1.0        // Normalization percentile (low)
    seg_pmax            = 99.8       // Normalization percentile (high)
    
    // StarDist model
    segmentation_model_dir = "/path/to/models/"
    segmentation_model     = "stardist_custom_model"
}
```

### Phenotyping Parameters

```groovy
params {
    // Marker panel for phenotyping (empty = use all)
    pheno_markers = ['CD163', 'CD14', 'CD45', 'CD3', 'CD8', 'CD4', 
                     'FOXP3', 'PANCK', 'VIMENTIN', 'SMA']
    
    // Expression cutoffs (z-score thresholds)
    pheno_cutoffs = [0.7, 0.9, 0.4, 0.2, 0.4, 0.9, 
                     1.3, 0.2, 0.2, 0.2]
    
    pheno_quality_percentile = 95    // Quality control filtering
    pheno_noise_percentile   = 5     // Noise removal threshold
}
```

### Resource Limits

```groovy
params {
    max_memory = '512.GB'
    max_cpus   = 64
    max_time   = '240.h'
}
```

---

## 🎯 Registration Methods

ATEIA provides four registration methods, each with distinct trade-offs:

### 1. VALIS (Recommended)

**Virtual Alignment of pathoLogy Image Series**

VALIS is the gold standard for multi-modal histology registration. It uses:
- **SuperPoint/SuperGlue** for deep learning-based feature matching
- **Micro-rigid registration** for local refinement
- **Optical flow** for non-rigid deformation

```bash
nextflow run main.nf --registration_method valis
```

**Pros:**
- Handles large intensity differences between stains
- Robust to tissue deformation
- Builds optimal transformation graph across all images

**Cons:**
- Requires all images loaded simultaneously (higher memory)
- Slower for many images per patient

### 2. GPU Diffeomorphic

Two-stage registration using CUDA-accelerated cuDIPY:

1. **Affine stage**: ORB feature matching with RANSAC
2. **Diffeomorphic stage**: Symmetric diffeomorphic registration (SyN)

```bash
nextflow run main.nf \
    --registration_method gpu \
    --gpu_type nvidia_h200:1
```

**Pros:**
- Faster than CPU
- Handles very large images via crop-based processing

**Cons:**
- Requires NVIDIA GPU with CUDA
- May have tile boundary artifacts (mitigated by overlap)

### 3. CPU Diffeomorphic

Multi-threaded CPU implementation using DIPY:

```bash
nextflow run main.nf --registration_method cpu
```

**Pros:**
- No GPU required
- Same algorithm as GPU method
- Good for batch processing on CPU clusters
- Handles very large images via crop-based processing
  
**Cons:**
- Slower than GPU
- May have tile boundary artifacts (mitigated by overlap)

### 4. VALIS Pairs

Pairwise VALIS registration (reference vs. each moving image separately):

```bash
nextflow run main.nf --registration_method valis_pairs
```

**Pros:**
- Lower memory than batch VALIS
- Retains most of VALIS quality

**Cons:**
- No multi-image optimization 

---

## 🧠 Segmentation

ATEIA uses **StarDist** for nuclear segmentation—a deep learning approach that predicts star-convex polygons for each nucleus.

### Custom Model Training

We fine-tuned StarDist on our specific tissue types. See our training notebook:

🔗 **[StarDist Fine-tuning Notebook](https://github.com/your-org/ateia-models/blob/main/notebooks/stardist_finetuning.ipynb)**

### Using Your Own Model

```groovy
params {
    segmentation_model_dir = "/path/to/your/models/"
    segmentation_model     = "my_custom_stardist"
}
```

Model directory should contain:
```
my_custom_stardist/
├── config.json
├── thresholds.json
└── weights_best.h5
```

---

## 🔧 Adding Custom Registration Methods

The pipeline uses an **adapter pattern** for registration methods. Each method implements a common interface while handling method-specific logic internally.

### Data Flow in `registration.nf`

```groovy
// Input: ch_preprocessed = [[meta, file], [meta, file], ...]

// STEP 1: Group by patient
ch_grouped = ch_preprocessed
    .map { meta, file -> [meta.patient_id, meta, file] }
    .groupTuple(by: 0)
    .map { patient_id, metas, files ->
        def items = [metas, files].transpose()
        def ref = items.find { it[0].is_reference }
        [patient_id, ref, items]
    }

// STEP 2: Route to adapter based on method
switch(method) {
    case 'valis': VALIS_ADAPTER(ch_grouped); break
    case 'gpu':   GPU_ADAPTER(ch_grouped); break
    case 'cpu':   CPU_ADAPTER(ch_grouped); break
}

// STEP 3: All adapters emit: [meta, registered_file]
```

### Creating a New Adapter

1. **Create the adapter file** (`subworkflows/local/adapters/my_adapter.nf`):

```groovy
nextflow.enable.dsl = 2

include { MY_REGISTER } from '../../../modules/local/my_register'

workflow MY_ADAPTER {
    take:
    ch_grouped  // [patient_id, reference_item, all_items]

    main:
    // Convert to your method's required format
    ch_to_register = ...

    // Run your registration
    MY_REGISTER(ch_to_register)

    ch_registered = MY_REGISTER.out.registered

    // remember to add reference if your registration method only saves the registered image (e.g. when working in pairs)
    // also if you always save reference, only add it once to final output
    
    emit:
    registered = ch_registered // Must be [meta, file] format
}
```

2. **Create the module** (`modules/local/my_register.nf`):

```groovy
process MY_REGISTER {
    tag "${meta.patient_id}"
    container "your/container:tag"

    // meta is always present to preserve it
    input:
    tuple val(meta), ...

    output:
    tuple val(meta), ...
    path "versions.yml", emit: versions

    script:
    """
    my_registration_script.py \
        ...
    """
}
```

3. **Register in `registration.nf`**:

```groovy
include { MY_ADAPTER } from './adapters/my_adapter'

// In the switch statement:
case 'my_method':
    MY_ADAPTER(ch_grouped)
    ch_registered = MY_ADAPTER.out.registered
    break
```

4. **Add to config validation** (`lib/ParamUtils.groovy`):

```groovy
def validateRegistrationMethod(method) {
    def valid = ['valis', 'valis_pairs', 'gpu', 'cpu', 'my_method']
    if (!valid.contains(method)) {
        error "Invalid method: ${method}. Valid: ${valid.join(', ')}"
    }
}
```

### Adapter Interface Contract

Every adapter **must**:
- Accept `ch_grouped`: `[patient_id, [ref_meta, ref_file], [[meta, file], ...]]`
- Emit `registered`: `[[meta, file], [meta, file], ...]`
- Include reference images in previous output (unchanged)
- Preserve `meta` fields (`patient_id`, `is_reference`, `channels`)

---

## 📝 TODOs

<!-- Add your TODOs here -->

### High Priority
- [ ] 

### Medium Priority
- [ ] 

### Low Priority
- [ ] 

### Documentation
- [ ] 

---

## 📚 Citation

If you use ATEIA in your research, please cite:

```bibtex
@software{ateia2026,
  author = {DIMA}
}
```

---

## 📄 License

This project is licensed under the 

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a Pull Request.

---

# Parameter Mapping: argparse → Nextflow Config

This document maps all Python script parameters to Nextflow config parameters.

## Preprocessing (preprocess.py)

| Python Argument | Nextflow Param | Type | Default | Description |
|----------------|----------------|------|---------|-------------|
| `--image` | N/A | str | required | Input image path (from process) |
| `--channels` | N/A | list | required | Channel file paths (from process) |
| `--patient_id` | N/A | str | required | Patient identifier (from process) |
| `--output_dir` | N/A | str | required | Output directory (from process) |
| `--fov_size` | `params.preproc_tile_size` | int | 512 | FOV/tile size for BaSiC |
| `--skip_dapi` | `params.preproc_skip_dapi` | bool | true | Skip DAPI correction |
| `--autotune` | `params.preproc_autotune` | bool | false | Autotune BaSiC params |
| `--n_iter` | `params.preproc_n_iter` | int | 3 | Autotuning iterations |
| `--log_file` | `params.log_file` | str | "" | Log file path |

## Registration (register.py)

| Python Argument | Nextflow Param | Type | Default | Description |
|----------------|----------------|------|---------|-------------|
| `--input-files` | N/A | list | required | Input files (from process) |
| `--out` | N/A | str | required | Output merged file (from process) |
| `--qc-dir` | N/A | str | optional | QC directory (from process) |

## Segmentation (segment.py)

| Python Argument | Nextflow Param | Type | Default | Description |
|----------------|----------------|------|---------|-------------|
| `--input` | N/A | str | required | Input DAPI image (from process) |
| `--out` | N/A | str | required | Output mask path (from process) |
| `--use-gpu` | `params.seg_gpu` | bool | true | Use GPU acceleration |
| `--model` | `params.seg_model` | str | 'cellpose' | Model name/type |
| `--output_dir` | N/A | str | './segmentation' | Output directory |
| `--model_dir` | `params.segmentation_model_dir` | str | "/models/" | Model directory |
| `--whole_image` | `params.seg_whole_image` | bool | true | Process whole image |
| `--crop_size` | `params.seg_crop_size` | int | 8000 | Crop size |
| `--overlap` | `params.segmentation_overlap` | int | 2500 | Crop overlap |
| `--gamma` | `params.seg_gamma` | float | 0.6 | Gamma correction |
| `--log_file` | `params.log_file` | str | "" | Log file path |

## Quantification (quantify.py)

| Python Argument | Nextflow Param | Type | Default | Description |
|----------------|----------------|------|---------|-------------|
| `--mode` | N/A | str | 'cpu' | Processing mode (cpu/gpu) |
| `--patient_id` | N/A | str | optional | Patient ID (from process) |
| `--mask_file` | N/A | str | required | Segmentation mask (from process) |
| `--indir` | N/A | str | required | Channel images directory (from process) |
| `--outdir` | N/A | str | required | Output directory (from process) |
| `--output_file` | N/A | str | optional | Exact output CSV path |
| `--min_area` | `params.quant_min_area` | int | 0 | Minimum cell area (pixels) |
| `--log_file` | `params.log_file` | str | "" | Log file path |

## Phenotyping (phenotype.py)

| Python Argument | Nextflow Param | Type | Default | Description |
|----------------|----------------|------|---------|-------------|
| `--cell_data` | N/A | str | required | Cell quantification CSV (from process) |
| `--segmentation_mask` | N/A | str | required | Segmentation mask (from process) |
| `-o, --output_dir` | N/A | str | required | Output directory (from process) |
| `--markers` | `params.pheno_markers` | list | DEFAULT_MARKERS | Marker names for phenotyping |
| `--cutoffs` | `params.pheno_cutoffs` | list | DEFAULT_CUTOFFS | Expression cutoffs per marker |
| `--quality_percentile` | `params.pheno_quality_percentile` | float | 1.0 | Quality filter percentile |
| `--noise_percentile` | `params.pheno_noise_percentile` | float | 0.01 | Noise removal percentile |
| `--log_file` | `params.log_file` | str | "" | Log file path |

## Usage in Nextflow Modules

### Example: Segmentation Module

```nextflow
process SEGMENT {
    // ...
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 scripts/segment.py \\
        --input ${merged_file} \\
        --out segmentation/${merged_file.simpleName}_segmentation.tif \\
        --use-gpu ${params.seg_gpu} \\
        --model ${params.seg_model} \\
        --model_dir ${params.segmentation_model_dir} \\
        --crop_size ${params.seg_crop_size} \\
        --overlap ${params.segmentation_overlap} \\
        --gamma ${params.seg_gamma} \\
        ${args}
    """
}
```

### Example: Phenotyping Module

```nextflow
process PHENOTYPE {
    // ...
    
    script:
    def args = task.ext.args ?: ''
    def markers_arg = params.pheno_markers ? "--markers ${params.pheno_markers.join(' ')}" : ''
    def cutoffs_arg = params.pheno_cutoffs ? "--cutoffs ${params.pheno_cutoffs.join(' ')}" : ''
    """
    python3 scripts/phenotype.py \\
        --cell_data ${quant_csv} \\
        --segmentation_mask ${seg_mask} \\
        -o pheno \\
        ${markers_arg} \\
        ${cutoffs_arg} \\
        --quality_percentile ${params.pheno_quality_percentile} \\
        --noise_percentile ${params.pheno_noise_percentile} \\
        ${args}
    """
}
```

## Adding New Parameters

To add a new parameter:

1. Add argparse argument in Python script
2. Add corresponding `params.*` in `nextflow.config`
3. Update process script block to pass parameter
4. Document in this file
5. Update `conf/modules.config` if needed for process-specific overrides


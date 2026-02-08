# Parameter Mapping: Nextflow → Python CLI

This document lists the active parameter mapping from Nextflow `params.*` to
Python script flags in the current pipeline.

## `preprocess.py`

| Nextflow param | Python flag |
|---|---|
| `params.preproc_tile_size` | `--fov_size` |
| `params.preproc_skip_dapi` | `--skip_dapi` |
| `params.preproc_autotune` | `--autotune` |
| `params.preproc_n_iter` | `--n_iter` |
| `params.preproc_pool_workers` | `--pool_workers` |
| `params.preproc_overlap` | `--fov_overlap` |
| `params.preproc_no_darkfield` | `--no_darkfield` |

## `register.py` (VALIS)

| Nextflow param | Python flag |
|---|---|
| `params.reg_reference_markers` | `--reference-markers` |
| `params.memory_mode` | `--memory-mode` |
| `params.reg_micro_reg_fraction` | `--micro-reg-fraction` |
| `params.reg_max_image_dim` | `--max-image-dim` |
| `params.skip_micro_registration` | `--skip-micro-registration` |
| `params.reg_parallel_warping` | `--parallel-warping` |
| `params.reg_n_workers` | `--n-workers` |
| `params.reg_use_tiled_registration` | `--use-tiled-registration` |
| `params.reg_tile_size` | `--tile-size` |

## `register_gpu.py`

| Nextflow param | Python flag |
|---|---|
| `params.gpu_reg_affine_crop_size` | `--affine-crop-size` |
| `params.gpu_reg_diffeo_crop_size` | `--diffeo-crop-size` |
| `params.gpu_reg_overlap_percent` | `--overlap-percent` |
| `params.gpu_reg_n_features` | `--n-features` |
| `params.gpu_reg_n_workers` | `--n-workers` |
| `params.gpu_reg_opt_tol` | `--opt-tol` |
| `params.gpu_reg_inv_tol` | `--inv-tol` |

## `register_cpu.py`

| Nextflow param | Python flag |
|---|---|
| `params.cpu_reg_affine_crop_size` | `--affine-crop-size` |
| `params.cpu_reg_diffeo_crop_size` | `--diffeo-crop-size` |
| `params.cpu_reg_overlap_percent` | `--overlap-percent` |
| `params.cpu_reg_n_features` | `--n-features` |
| `params.cpu_reg_opt_tol` | `--opt-tol` |
| `params.cpu_reg_inv_tol` | `--inv-tol` |

## `segment.py`

| Nextflow param | Python flag |
|---|---|
| `params.segmentation_model_dir` | `--model-dir` |
| `params.segmentation_model` | `--model-name` |
| `params.seg_n_tiles_y`, `params.seg_n_tiles_x` | `--n-tiles` |
| `params.seg_expand_distance` | `--expand-distance` |
| `params.seg_pmin` | `--pmin` |
| `params.seg_pmax` | `--pmax` |
| `params.seg_gpu` | `--use-gpu` |

## `quantify.py`

| Nextflow param | Python flag |
|---|---|
| `params.quant_min_area` | `--min_area` |

## `phenotype.py`

| Nextflow param | Python flag |
|---|---|
| `params.phenotype_config` | `--config` |
| `params.pixel_size` | `--pixel_size` |
| `params.pheno_quality_percentile` | `--quality_percentile` |
| `params.pheno_noise_percentile` | `--noise_percentile` |

## `merge_channels_pyramid.py`

| Nextflow param | Python flag |
|---|---|
| `params.pixel_size` | `--physical-size-x`, `--physical-size-y` |
| `params.pyramid_resolutions` | `--pyramid-resolutions` |
| `params.pyramid_scale` | `--pyramid-scale` |
| `params.tilex` | `--tile-size` |
| `params.compression` | `--compression` |

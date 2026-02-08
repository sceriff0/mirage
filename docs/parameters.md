# Parameters

This page documents the active canonical parameter surface.

## Canonical Sources

- Defaults: `nextflow.config` (`params { ... }`)
- Validation/enums: `lib/ParamUtils.groovy`
- Schema: `nextflow_schema.json`

## High-Impact Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | `''` | Required for `preprocessing`, `registration`, `postprocessing` |
| `outdir` | `./results` | Main output root |
| `savedir` | `null` | Optional archive destination |
| `copy_delete_source` | `false` | Deletes source only after verified copy |
| `step` | `preprocessing` | `preprocessing`, `registration`, `postprocessing`, `copy_results` |
| `registration_method` | `valis` | `valis`, `valis_pairs`, `gpu`, `cpu`, `cpu_tiled` |
| `debug_channels` | `false` | Enables debug channel `.view` output |
| `dry_run` | `false` | Validation-only mode |

## Preprocessing

- `preproc_tile_size`
- `preproc_skip_dapi`
- `preproc_autotune`
- `preproc_n_iter`
- `preproc_pool_workers`
- `preproc_overlap`
- `preproc_no_darkfield`
- `preprocess_qc_scale_factor`

## Registration

- Common:
  - `allow_auto_reference`
  - `padding`
  - `pad_mode`
  - `skip_registration_qc`
  - `qc_scale_factor`
  - `enable_feature_error`
  - `feature_detector`
  - `feature_max_dim`
  - `feature_n_features`
- VALIS:
  - `reg_reference_markers`
  - `memory_mode`
  - `reg_micro_reg_fraction`
  - `reg_max_image_dim`
  - `skip_micro_registration`
  - `reg_parallel_warping`
  - `reg_n_workers`
  - `reg_use_tiled_registration`
  - `reg_tile_size`
- GPU pairwise:
  - `gpu_reg_affine_crop_size`
  - `gpu_reg_diffeo_crop_size`
  - `gpu_reg_overlap_percent`
  - `gpu_reg_n_features`
  - `gpu_reg_n_workers`
  - `gpu_reg_opt_tol`
  - `gpu_reg_inv_tol`
- CPU pairwise/tiled:
  - `cpu_reg_affine_crop_size`
  - `cpu_reg_diffeo_crop_size`
  - `cpu_reg_overlap_percent`
  - `cpu_reg_n_features`
  - `cpu_reg_opt_tol`
  - `cpu_reg_inv_tol`

## Postprocessing

- Segmentation:
  - `seg_gpu`
  - `seg_pmin`
  - `seg_pmax`
  - `seg_n_tiles_y`
  - `seg_n_tiles_x`
  - `seg_expand_distance`
  - `segmentation_model_dir`
  - `segmentation_model`
- Quantification:
  - `quant_min_area`
- Phenotyping:
  - `phenotype_config`
  - `pheno_quality_percentile`
  - `pheno_noise_percentile`
- Pyramid export:
  - `pixel_size`
  - `tilex`
  - `tiley`
  - `pyramid_resolutions`
  - `pyramid_scale`
  - `compression`

## Infrastructure and Limits

- Cluster/runtime:
  - `slurm_partition`
  - `slurm_account`
  - `slurm_qos`
  - `gpu_type`
  - `publish_dir_mode`
- Limits:
  - `max_memory`
  - `max_cpus`
  - `max_time`
- Trace:
  - `enable_trace`
  - `trace_dir`


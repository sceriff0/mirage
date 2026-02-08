# Registration Methods

Use `--registration_method` to select one of:

- `valis`
- `valis_pairs`
- `gpu`
- `cpu`
- `cpu_tiled`

## Method Summary

| Method | Typical Use | Tradeoff |
|---|---|---|
| `valis` | Multi-image registration using VALIS graph workflow | Robust but higher memory footprint |
| `valis_pairs` | Pairwise VALIS-style registration | Simpler pairing behavior |
| `gpu` | Pairwise affine + diffeomorphic registration on GPU | Fastest when GPUs are available |
| `cpu` | Pairwise affine + diffeomorphic registration on CPU | No GPU needed, slower |
| `cpu_tiled` | Memory-constrained CPU pairwise tiled workflow | Lower per-task memory, more orchestration |

## Key Parameters by Family

### Common

- `allow_auto_reference`
- `padding`
- `pad_mode`
- `skip_registration_qc`

### VALIS

- `reg_reference_markers`
- `reg_max_image_dim`
- `reg_micro_reg_fraction`
- `skip_micro_registration`
- `reg_parallel_warping`

### GPU Pairwise

- `gpu_reg_affine_crop_size`
- `gpu_reg_diffeo_crop_size`
- `gpu_reg_overlap_percent`
- `gpu_reg_n_features`
- `gpu_reg_n_workers`

### CPU Pairwise / Tiled

- `cpu_reg_affine_crop_size`
- `cpu_reg_diffeo_crop_size`
- `cpu_reg_overlap_percent`
- `cpu_reg_n_features`


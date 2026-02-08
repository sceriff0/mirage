# Running on SLURM

## Standard SLURM Profile

```bash
nextflow run main.nf \
  -profile slurm \
  --input input.csv \
  --outdir results \
  --registration_method gpu \
  --slurm_partition gpu \
  --slurm_account <account>
```

## GPU Controls

- Segmentation GPU toggle: `--seg_gpu true|false`
- Cluster GPU type hint: `--gpu_type nvidia_h200:1`

## Resource Limits

These top-level limits cap process-specific resources:

- `--max_memory`
- `--max_cpus`
- `--max_time`

Process-level defaults and retries are defined in `conf/modules.config`.

## Resume

```bash
nextflow run main.nf -profile slurm -resume \
  --input results/csv/preprocessed.csv \
  --step registration \
  --registration_method gpu \
  --outdir results
```


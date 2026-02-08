# Getting Started

## Minimal Prerequisites

- Nextflow (DSL2-capable)
- Container runtime (`singularity` or `docker`)
- CSV input with required columns for your selected `--step`

## Quick Run: Full Pipeline

```bash
nextflow run main.nf \
  --input input.csv \
  --outdir results \
  --step preprocessing \
  --registration_method gpu \
  -profile slurm
```

## Quick Run: Registration from Checkpoint

```bash
nextflow run main.nf \
  --input results/csv/preprocessed.csv \
  --step registration \
  --registration_method cpu_tiled \
  --outdir results \
  -profile slurm
```

## Quick Run: Copy Results to Archive

```bash
nextflow run main.nf \
  --step copy_results \
  --outdir results \
  --savedir /path/to/archive \
  --copy_delete_source false \
  -profile slurm
```

## Dry Validation Only

```bash
nextflow run main.nf \
  --input input.csv \
  --step preprocessing \
  --dry_run true
```

## First Checks After Launch

1. Verify `results/csv/` checkpoint CSVs are created.
2. Confirm your chosen registration method outputs in `results/<patient_id>/registered/`.
3. Confirm postprocessing outputs in `results/<patient_id>/`.


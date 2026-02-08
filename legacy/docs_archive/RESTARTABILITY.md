# Pipeline Restartability Guide

The pipeline supports checkpoint-based restart using CSV outputs in `results/csv/`.

## Entry Steps

| Step | Input required | Input format |
|---|---|---|
| `preprocessing` | yes | raw-input CSV (`path_to_file`) |
| `registration` | yes | `preprocessed.csv` (`preprocessed_image`) |
| `postprocessing` | yes | `registered.csv` (`registered_image`) |
| `copy_results` | no | uses `--outdir` and `--savedir` only |

## Checkpoint Files

### `preprocessed.csv`

Columns:

- `patient_id`
- `preprocessed_image`
- `is_reference`
- `channels`

### `registered.csv`

Columns:

- `patient_id`
- `registered_image`
- `is_reference`
- `channels`

### `postprocessed.csv`

Columns:

- `patient_id`
- `phenotype_csv`
- `phenotype_geojson`
- `phenotype_mapping`
- `merged_csv`
- `cell_mask`
- `pyramid`

## Common Restart Commands

Run full pipeline from raw input:

```bash
nextflow run main.nf \
  --input samples.csv \
  --step preprocessing \
  --outdir results
```

Restart from registration:

```bash
nextflow run main.nf \
  --input results/csv/preprocessed.csv \
  --step registration \
  --registration_method gpu \
  --outdir results
```

Restart from postprocessing:

```bash
nextflow run main.nf \
  --input results/csv/registered.csv \
  --step postprocessing \
  --outdir results
```

Copy already generated results to an archive location:

```bash
nextflow run main.nf \
  --step copy_results \
  --outdir results \
  --savedir /final/archive
```

By default the source is retained. To delete source after successful copy and checksum verification, opt in with:

```bash
--copy_delete_source true
```

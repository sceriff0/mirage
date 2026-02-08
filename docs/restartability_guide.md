# Restartability and Copy Results

ATEIA is checkpoint-driven. Each major stage writes a CSV that can be used as input to the next stage.

## Valid Step Entrypoints

- `preprocessing`
- `registration`
- `postprocessing`
- `copy_results`

## Restart Patterns

## Full pipeline from raw/preprocessing input

```bash
nextflow run main.nf \
  --input input.csv \
  --step preprocessing \
  --registration_method gpu \
  --outdir results
```

## Resume at registration

```bash
nextflow run main.nf \
  --input results/csv/preprocessed.csv \
  --step registration \
  --registration_method cpu_tiled \
  --outdir results
```

## Resume at postprocessing

```bash
nextflow run main.nf \
  --input results/csv/registered.csv \
  --step postprocessing \
  --outdir results
```

## Copy/archive only

```bash
nextflow run main.nf \
  --step copy_results \
  --outdir results \
  --savedir /archive/ateia_run
```

## Safety Behavior for `copy_results`

- `savedir` must be provided.
- `savedir` cannot equal `outdir`.
- `savedir` cannot be nested under `outdir`.
- Source directory must exist for `copy_results`.
- Source deletion is opt-in (`copy_delete_source=true`) and performed only after verification.


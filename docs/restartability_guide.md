# Restartability

MIRAGE is checkpoint-driven. Each major stage writes a CSV that can be used as input to the next stage.

## Valid Step Entrypoints

- `preprocessing`
- `registration`
- `postprocessing`

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

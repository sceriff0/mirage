# Outputs

## Output Root

All outputs are written under `--outdir` (default `./results`).

## Typical Structure

```text
results/
  csv/
    preprocessed.csv
    registered.csv
    postprocessed.csv
  <patient_id>/
    preprocessed/
    registered/
    qc/
      preprocess/
      registration/
    segmentation/
    quantification/
    phenotyping/
    pyramid/
```

## Checkpoint CSVs

### `preprocessed.csv`

Used as `--input` when `--step registration`.

Columns:

- `patient_id`
- `preprocessed_image`
- `is_reference`
- `channels`

### `registered.csv`

Used as `--input` when `--step postprocessing`.

Columns:

- `patient_id`
- `registered_image`
- `is_reference`
- `channels`

### `postprocessed.csv`

Manifest of postprocessing outputs for each patient.

## Trace Outputs

If `enable_trace=true`, generated files include:

- `${trace_dir}/trace.txt`
- `${trace_dir}/report.html`
- `${trace_dir}/timeline.html`


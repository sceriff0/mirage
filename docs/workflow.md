# Workflow

## Pipeline Stages

## 1. Preprocessing

- Convert/normalize source images
- Correct illumination artifacts
- Split channels
- Emit checkpoint: `results/csv/preprocessed.csv`

## 2. Registration

- Group by patient
- Select reference image
- Register moving images to reference using selected method
- Emit checkpoint: `results/csv/registered.csv`

## 3. Postprocessing

- Segmentation
- Quantification
- Phenotyping
- Merge channels and produce pyramidal OME-TIFF
- Emit checkpoint: `results/csv/postprocessed.csv`

## 4. Copy Results

- Archive/copy full output tree from `outdir` to `savedir`
- Optional source deletion only if `--copy_delete_source true`
- Path preflight checks prevent unsafe copy/delete configurations

## Workflow Control

`main.nf` orchestrates step-specific entrypoints:

- `preprocessing`: runs all downstream stages
- `registration`: starts from preprocessed checkpoint
- `postprocessing`: starts from registered checkpoint
- `copy_results`: copy/archive only

## Tracing

If `enable_trace=true`, Nextflow trace/report/timeline files are written to `trace_dir`.


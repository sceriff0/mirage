# MIRAGE Documentation

MIRAGE is a Nextflow DSL2 pipeline for multiplex microscopy processing:

1. Preprocessing (image conversion and illumination correction)
2. Registration (multiple methods, including tiled CPU)
3. Postprocessing (segmentation, quantification, phenotyping, pyramid export)
4. Result copy/archival (`copy_results`)

This site is the canonical documentation surface for running and maintaining the pipeline.

## Quick Links

- Start here: [Getting Started](getting_started.md)
- Input CSV formats: [Input Format](input_spec.md)
- Step and channel flow: [Workflow](workflow.md)
- Parameters and defaults: [Parameters](parameters.md)
- Restart and archival: [Restartability](restartability_guide.md)

## Canonical Runtime Surface

- `--step`: `preprocessing`, `registration`, `postprocessing`, `copy_results`
- `--registration_method`: `valis`, `valis_pairs`, `gpu`, `cpu`, `cpu_tiled`
- `copy_results` does not require `--input`; it requires `--outdir` and `--savedir`

## Source of Truth

- Pipeline entrypoint: `main.nf`
- Defaults: `nextflow.config`
- Schema: `nextflow_schema.json`

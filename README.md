# ateia — WSI processing pipeline

This repository implements a Nextflow DSL2 pipeline for whole-slide image (WSI) processing. It follows nf-core-style conventions: modular DSL2 modules under `modules/`, workflows under `workflows/`, and small Python helper scripts in `scripts/`.

Quick start

1. Install Nextflow and required Python environment.
2. Run locally:

```bash
nextflow run main.nf -c nextflow.config --input './data/*.nd2'
```

Run on SLURM with GPU profile:

```bash
nextflow run main.nf -profile slurm,gpu -c nextflow.config --input '/path/to/data/*.nd2'
```

Repository layout

- `main.nf` — root pipeline that composes the WSI workflow.
- `workflows/wsi_processing/main.nf` — pipeline orchestration.
- `modules/*/main.nf` — per-stage DSL2 modules (preprocessing, registration, segmentation, quantification, phenotyping).
- `scripts/*.py` — Python helpers called by processes.

Notes

- This repo was refactored to follow nf-core-like structure and to make registration collect all preprocessed files into a single merged WSI in `merged/merged_all.ome.tif`.
- Before running on a fresh system, create a Python environment and install dependencies from `requirements.txt`.

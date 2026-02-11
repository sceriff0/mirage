# MIRAGE

MIRAGE is a Nextflow DSL2 pipeline for whole-slide and multiplex microscopy workflows:

- preprocessing
- registration
- segmentation
- quantification
- phenotyping
- pyramidal OME-TIFF export

## Quick Start

```bash
nextflow run main.nf \
  --input input.csv \
  --outdir results \
  --step preprocessing \
  --registration_method gpu \
  -profile slurm
```

## Documentation

- ReadTheDocs/MkDocs source is under `docs/`
- MkDocs config: `mkdocs.yml`
- RTD config: `.readthedocs.yaml`

Build locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Canonical Runtime Enums

- `--step`: `preprocessing`, `registration`, `postprocessing`, `copy_results`
- `--registration_method`: `valis`, `valis_pairs`, `gpu`, `cpu`, `cpu_tiled`


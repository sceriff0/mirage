# Installation

## Requirements

- Java 11+
- Nextflow
- One container backend:
  - Singularity/Apptainer (recommended on HPC)
  - Docker (local/dev)

## Clone and Enter Repository

```bash
git clone <your-repo-url> mirage
cd mirage
```

## Verify Nextflow

```bash
nextflow -version
```

## ReadTheDocs/MkDocs Dependencies

Local docs build dependencies are in `docs/requirements.txt`:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Execution Profiles

Defined in `nextflow.config`:

- `standard` (SLURM default)
- `slurm`
- `test`
- `local`
- `docker`
- `singularity`

## Container Images

Containers are defined under `params.container` in `nextflow.config` and process-specific settings in `conf/modules.config`.


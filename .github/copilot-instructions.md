# Copilot / AI agent instructions — ateia

Purpose: Provide concise, actionable context so an AI coding agent can be productive immediately in this repository.

Big picture
- This repository is a Nextflow DSL2 pipeline (root `main.nf`) that includes the workflow defined at `workflows/wsi_processing/main.nf`:
  - Example: `include { WSI_PROCESSING } from './workflows/wsi_processing/main.nf'` and `WSI_PROCESSING(params.input)` in `main.nf`.
- Processing stages are organized as modules under `modules/` (one stage per subfolder, each with a `main.nf`). Common modules: `preprocessing`, `segmentation`, `quantification`, `registration`, `phenotyping`.
- Small Python helper scripts live in `scripts/` (e.g. `preprocess.py`, `quantify.py`, `segment.py`) and are invoked by Nextflow processes.

Key files to inspect
- `main.nf` — root pipeline that ties everything together.
- `nextflow.config` — default params and profiles. Notable entries:
  - `params` block defines pipeline defaults (e.g. `input`, `preproc_tile_size`, `seg_model`, `seg_gpu`).
  - `profiles` defines `slurm` and `gpu` profiles used for cluster runs.
- `workflows/wsi_processing/main.nf` — core workflow implementation and orchestration.
- `modules/*/main.nf` — per-stage implementations; check `process` definitions, labels, resource directives (cpus/mem/time).
- `conf/` — cluster/institutional configs live here (provide site-specific SLURM config as needed).
- `scripts/*.py` — small Python entrypoints used by processes; inspect for expected CLI args and side-effects.
- `tests/run_tests.sh` — minimal smoke test script (run with `bash tests/run_tests.sh`).

Developer workflows (how to run)
- Quick local run (uses defaults from `nextflow.config`):
  - nextflow run main.nf -c nextflow.config --input './data/*.nd2'
- Run on Slurm with GPU label (combine profiles):
  - nextflow run main.nf -profile slurm,gpu -c nextflow.config --input '/path/to/data/*.nd2'
- Run the lightweight smoke test:
  - bash tests/run_tests.sh
- Inspect a module's processes to change resources or add logging by editing `modules/<stage>/main.nf`.

Project-specific conventions & patterns
- DSL2 module-per-folder convention: each stage exposes a workflow (or process) from `modules/<name>/main.nf` and is composed in `workflows/.../main.nf` or the root `main.nf`.
- Parameters are centralized in `nextflow.config` under `params`. Prefer adding defaults there and passing runtime overrides via CLI (`--param value`).
- Profiles are used to switch executors and labels (`slurm`, `gpu`). Use `-profile` to select cluster behavior.
- Python scripts in `scripts/` are single-purpose; when modifying them, keep CLI interface stable (they are likely called from Nextflow `shell`/`script` blocks).
- Tests are currently a minimal smoke test script; there is no project-wide CI config discovered. Treat `tests/run_tests.sh` as the canonical local test entrypoint.

Integration points / things to watch
- Nextflow processes communicate via channels; changes to process outputs (filenames, channel contents) require updating downstream process inputs.
- Cluster config lives in `conf/` and `nextflow.config` profiles; coordinate changes with ops (Slurm queues, GPU labels).
- There is no discovered `requirements.txt` or `pyproject.toml` in repo root — verify Python dependencies before running scripts on new environments.

When making changes
- To add a new pipeline stage: create `modules/<stage>/main.nf` exposing a workflow (e.g. `workflow NEW_STAGE { ... }`), then include and compose it in `workflows/wsi_processing/main.nf` or `main.nf`.
- To change defaults: update `nextflow.config` `params` block.
- To add a cluster profile: add under `profiles` in `nextflow.config` and add cluster settings in `conf/` as needed.

Examples drawn from repo
- Root include: `include { WSI_PROCESSING } from './workflows/wsi_processing/main.nf'` (see `main.nf`).
- Params example: `seg_model = 'cellpose'` and `seg_gpu = true` (see `nextflow.config`).

If anything here is incomplete or unclear, tell me which part you want expanded (for example: exact process names in `modules/segmentation`, or a list of Python CLI args) and I will update this file.

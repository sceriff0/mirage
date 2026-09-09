# Installation

This page gets MIRAGE running on your machine — laptop, workstation, or HPC login node. The whole setup is: install **Nextflow**, pick **one container backend**, clone the repo, and verify. Budget about 10–15 minutes, most of it spent pulling container images on the first run.

!!! tip "In a hurry?"
    Already have Nextflow and Docker? Skip to [Clone the repository](#clone-the-repository), then head straight to the [Walkthrough](usage.md).

## Requirements

| Requirement | Version / detail | Notes |
|---|---|---|
| **Nextflow** | `>=25.04.0` | Uses the `nf-schema@2.5.1` plugin; older versions will not run. |
| **Java** | 11 or newer | Required by Nextflow. Check with `java -version`. |
| **Container backend** | one of Singularity/Apptainer, Docker, or Conda | Singularity recommended on HPC; Docker for local/dev. |
| **Free disk** | ~10 GB | For container images plus intermediate outputs. |
| **Network access on first run** | outbound access to the Nextflow plugin registry | Nextflow fetches `nf-schema` from the registry the first time the pipeline runs. On a network-isolated compute node, pre-provision the plugin first — see [Offline / air-gapped execution](usage.md#offline-air-gapped-execution). |

!!! warning "Nextflow version matters"
    MIRAGE requires Nextflow **25.04.0 or later**. If `nextflow -version` reports something older, update it (next section) before doing anything else.

## Install Nextflow

If you don't have Nextflow yet:

```bash
curl -s https://get.nextflow.io | bash
# move the launcher onto your PATH
sudo mv nextflow /usr/local/bin/
```

If you already have it but it's too old, self-update in place:

```bash
nextflow self-update
```

!!! info "Java first"
    Nextflow needs Java 11+. On most clusters this is a module (`module load java`); on a Mac, `brew install openjdk@17` works well.

## Make a site config { #size-your-run }

`max_cpus` and `max_memory` are **required and have no default** — the pipeline
refuses to launch without them, because a default here would be a guess about
your machine:

```text
ERROR ~ Validation of pipeline parameters failed!

The following invalid input values have been detected:

* Missing required parameter(s): max_cpus, max_memory
```

Copy the template once and layer it on every run with `-c`:

```bash
cp conf/site.config.template site.config
```

Then edit `max_cpus`, `max_memory` and `max_time` to match the machine or the
partition, plus `slurm_partition` / `slurm_account` / `slurm_qos` on a cluster.
`site.config` at the repository root is gitignored, so cluster paths and account
names never reach a commit. Every command on this site ends in `-c site.config`.

!!! tip "The bundled test profiles need no site config"
    `-profile test` and `-profile test_full` pin their own small ceilings, so the
    demo runs with nothing else set.

!!! note "Every command on this site is verified to actually launch"
    `tests/test_documented_commands_are_runnable.py` checks that every documented
    `nextflow run` command carries `--outdir` and the sizing pair (`-c
    site.config`, a pinning profile, or a preset that carries both) — a STATIC
    token check, run in the Python test suite. `tests/documented_commands_launch.sh`
    goes further: it substitutes each command's placeholders for this repo's own
    fixtures and actually runs `nextflow -stub -params-file params/dry_run.json`
    against every one, so an unrecognised flag or a schema rejection fails CI too,
    not just a missing `--outdir`. Both run in `_test-suite.yml` — the static
    check inside the Python suite, the launches in their own
    `documented-commands` job.

## Choose a container backend

Every MIRAGE process runs inside a container with a pinned version tag — you don't install the scientific tools (VALIS, StarDist, Bio-Formats, …) yourself. Pick the backend that matches where you're running.

=== "Docker"

    Best for **local development and laptops**. Make sure the Docker daemon is running, then add `docker` to your profile:

    ```bash
    nextflow run . --input samplesheet.csv --outdir results -profile docker -c site.config
    ```

    !!! tip
        On Docker Desktop, give the VM enough memory (8 GB+) in **Settings → Resources** so segmentation and registration don't get OOM-killed.

=== "Singularity / Apptainer"

    **Recommended on HPC**, where Docker is usually unavailable or disallowed. Singularity runs rootless and plays well with shared filesystems:

    ```bash
    nextflow run . --input samplesheet.csv --outdir results -profile singularity -c site.config
    ```

    Apptainer (the renamed successor to Singularity) is a drop-in replacement and uses the same `singularity` profile.

    !!! tip "Cache images once"
        Point `NXF_SINGULARITY_CACHEDIR` at a shared, persistent path so images are pulled once and reused across runs and users:

        ```bash
        export NXF_SINGULARITY_CACHEDIR=/path/to/shared/singularity_cache
        ```

=== "Conda"

    A containerless option using Conda-managed environments. Slower to set up and less reproducible than containers, but useful where neither Docker nor Singularity is available:

    ```bash
    nextflow run . --input samplesheet.csv --outdir results -profile conda -c site.config
    ```

!!! note "Combine profiles"
    Container profiles combine with execution profiles via commas: `-profile slurm,singularity` on a cluster, `-profile test,docker` for the bundled demo, `-profile local,docker` for a capped local run.

## Clone the repository

```bash
git clone https://github.com/sceriff0/mirage.git
cd mirage
```

All commands on this site assume you run them from the repo root (`nextflow run .`).

## Verify

Confirm Nextflow is present and new enough:

```bash
nextflow -version
```

You should see version **25.04.0 or later**. A quick stub run validates the whole wiring without pulling heavy images or running real tools:

```bash
nextflow run . -profile test,docker -stub --outdir results_stub
```

If that completes without `FAILED` processes, your installation is sound. (You'll need the test data generated first for a *real* run — see the [Walkthrough](usage.md).)

## Pre-pulling container images (optional)

The first real run downloads each tool's image, which can take several minutes. Pull them ahead of time to make that first run snappy. MIRAGE runs **13 images**: 11 first-party (`modules/local/*.nf`), plus 2 externals — one upstream-maintained and not vendored (`REGISTER`'s VALIS image, `modules/local/register.nf:22`), and one vendored nf-core module whose image `conf/modules.config`'s `withName: 'BASICPY'` block repins to a digest (`conf/modules.config:362`; the vendored `modules/nf-core/basicpy/main.nf:5` itself still names the image by tag):

| Image | Used for |
|---|---|
| `bolt3x/mirage-convert:1.0.0` | `CONVERT_IMAGE` (format conversion) |
| `bolt3x/mirage-preprocess:1.0.0` | `SPLIT_CHANNELS`, `PREFLIGHT_SCALE`, `TILE_FOR_BASIC` / `APPLY_PROFILES` (illumination correction around `BASICPY`), `GENERATE_PREPROCESS_QC`, `GENERATE_QC_REPORT`, `AGGREGATE_SIZE_LOGS` |
| `bolt3x/mirage-quantify:1.0.0` | quantification, property extraction, GeoJSON export, postprocessing QC |
| `bolt3x/mirage-stardist:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method stardist` |
| `bolt3x/mirage-instanseg:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method instantseg` (default) |
| `bolt3x/mirage-cellsam:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method cellsam` |
| `bolt3x/mirage-merge:1.0.0` | `MERGE_AND_PYRAMID` |
| `bolt3x/mirage-regqc:1.0.0` | `GENERATE_REGISTRATION_QC` |
| `bolt3x/mirage-tiled:1.0.0` | the `tiled` (STARE) registration backend, and `WARP_SEG_QC`'s tiled path |
| `bolt3x/mirage-spatialdata:1.0.0` | `EXPORT_SPATIALDATA` |
| `bolt3x/mirage-segeval:1.0.0` | `SEG_QUALITY_EVAL`, `MERGE_SEG_EVAL` (opt-in) |
| `cdgatenbee/valis-wsi@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6` (upstream, not vendored) | `REGISTER`, and `WARP_SEG_QC`'s VALIS path |
| `docker.io/labsyspharm/basicpy-docker-mcmicro@sha256:355b14e2ec80b7b152272f333afd47234f007d0d37633b3ec948e87ec2c8e9b4` (vendored nf-core module's own image, repinned) | `BASICPY` — real by default, since illumination correction is not `--skip_preprocessing` |

=== "Docker"

    ```bash
    docker pull cdgatenbee/valis-wsi@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6
    docker pull docker.io/labsyspharm/basicpy-docker-mcmicro@sha256:355b14e2ec80b7b152272f333afd47234f007d0d37633b3ec948e87ec2c8e9b4
    docker pull bolt3x/mirage-<component>:1.0.0   # e.g. convert, preprocess, quantify, tiled...
    ```

=== "Singularity / Apptainer"

    ```bash
    singularity pull docker://cdgatenbee/valis-wsi@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6
    singularity pull docker://docker.io/labsyspharm/basicpy-docker-mcmicro@sha256:355b14e2ec80b7b152272f333afd47234f007d0d37633b3ec948e87ec2c8e9b4
    singularity pull docker://bolt3x/mirage-<component>:1.0.0
    ```

!!! note "Where tags live, and what they pin"
    Every MIRAGE-owned process names its image in `modules/local/*.nf`'s
    `container` directive, or — for the per-backend images — in
    `lib/SegBackends.groovy` / `lib/WarpBackends.groovy`; `conf/modules.config`
    owns those processes' *resources*, not their images. The one exception is
    the single vendored nf-core module, `BASICPY`: its own `modules/nf-core/basicpy/main.nf:5`
    pins `docker.io/labsyspharm/basicpy-docker-mcmicro:1.2.0-patch5` by tag, and
    `conf/modules.config`'s `withName: 'BASICPY'` block overrides it to the same
    image pinned by content digest (`container = '...@sha256:355b14e2...'`,
    `conf/modules.config:362`) — done there, rather than edited into the vendored
    file, precisely so the vendored module stays byte-identical to upstream
    (`tests/test_basicpy_module_is_vendored_unmodified.py`). MIRAGE never uses
    `:latest`. Every `FROM` inside
    `containers/*/Dockerfile` is **digest-pinned** (`FROM <base>@sha256:...`), so
    an image rebuilt a year from now still starts from the same base bytes.

    The eleven first-party images themselves are pinned by **version tag**
    (`bolt3x/mirage-<component>:<manifest.version>`), because a digest cannot
    exist before `release.yml` has pushed them. The two externals
    (`cdgatenbee/valis-wsi`, `docker.io/labsyspharm/basicpy-docker-mcmicro`) are
    referenced by content digest directly. Every task's `versions.yml` records
    `container:` as `task.container` resolved it — a tag for a first-party image
    today, a digest for an external one — so a result can already be traced to
    the exact bytes an external tool ran; the first-party images' *published*
    digests land in [`containers/README.md`](https://github.com/sceriff0/mirage/blob/main/containers/README.md)'s
    image-mapping table once a release has pushed them.

## GPU notes

Segmentation can run on a GPU for a substantial speedup, or fall back to CPU.

- **Enable GPU**: `seg_gpu = true`. Your container runtime must expose the GPU — Singularity needs `--nv`, and on SLURM you must request a GPU (e.g. `--gres=gpu:1`). See the [SLURM guide](usage.md#running-on-hpc) for cluster specifics.
- **CPU fallback**: `seg_gpu = false`. Slower but works everywhere — this is what the `test` profile uses so the demo runs on any laptop.

!!! warning
    If you request `seg_gpu = true` without exposing a GPU to the container, segmentation will fail or silently fall back depending on the backend. When in doubt on a new machine, start with `seg_gpu = false`.

## Building the docs locally (optional)

Contributing to this site? The docs are a MkDocs Material site. Build and live-preview them:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open <http://127.0.0.1:8000> — the site rebuilds on every save.

## Execution profiles overview

Profiles are defined in `nextflow.config` and combined with commas. Pick one **execution** profile and one **container** profile.

| Profile | Kind | What it does |
|---|---|---|
| `test` | data + caps | Bundled synthetic dataset, small resource caps, segmentation on CPU. |
| `test_full` | data + caps | Larger synthetic dataset with more realistic parameters. |
| `instantseg_test` | data + caps | Test profile exercising the InstanSeg segmentation backend. |
| `cellsam_test` | data + caps | Test profile exercising the CellSAM segmentation backend. |
| `docker` | container | Run processes in Docker containers (local/dev). |
| `singularity` | container | Run processes in Singularity/Apptainer containers (recommended on HPC). |
| `conda` | container | Run with Conda-managed environments (no containers). |
| `slurm` | executor | Submit each process as a SLURM job. |
| `local` | executor | Local executor with conservative caps (4 CPU / 16 GB). |

!!! note "Your own site profile"
    There is no shipped site profile. Copy `conf/site.config.template` to
    `site.config` and layer it with `-c site.config` — see
    [Make a site config](installation.md#size-your-run). A named profile is
    possible too (`profiles { mysite { includeConfig 'conf/mysite.config' } }`),
    but note that a profile body is evaluated while `nextflow.config` is parsed,
    whereas a `-c` file is layered afterwards and wins on params.

!!! example "Common combinations"
    - Laptop demo: `-profile test,docker`
    - Local capped run on your data: `-profile local,docker`
    - HPC production run: `-profile slurm,singularity`

## Next steps

Head to **[Usage](usage.md)** — the four-stage model, the samplesheet, the
end-to-end run on synthetic data, resuming from checkpoints, and what lands in
`--outdir`. To tune the run, see **[Parameters](parameters.md)**.

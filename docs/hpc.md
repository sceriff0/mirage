# Running on a cluster

<p class="standfirst">On a cluster MIRAGE submits every process as its own SLURM job and runs it in
an Apptainer/Singularity container. You set, once, how large one job may be; everything else is a
profile on the command line.</p>

## The command

```bash
nextflow run . -profile slurm,singularity \
  --input samplesheet.csv --outdir results --start preprocessing \
  -c site.config
```

Start it inside `tmux` or `screen`, or as a job of its own, so it survives your logging
out. The Nextflow process itself is the *head*: it stays running for the whole analysis
and submits the jobs.

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
names never reach a commit. The same file sizes a large workstation: on a laptop,
`-profile local` sets small ceilings (4 CPUs, 16 GB) instead.

A minimal `site.config`:

```groovy
params {
    slurm_partition = 'compute'
    slurm_account   = 'myproject'
    max_memory      = '500.GB'
    max_cpus        = 64
    max_time        = '120.h'
}
```

!!! tip "The bundled test profiles need no site config"
    `-profile test` and `-profile test_full` pin their own small ceilings, so the
    demo runs with nothing else set.

!!! warning "Six settings cannot go in a `-c site.config`"
    `concurrency`, `max_forks`, `queue_size`, `cleanup_work`, `enable_trace` and
    `trace_dir` each drive a setting fixed while `nextflow.config` is parsed, before
    any `-c` file is merged. The pipeline refuses such a run at launch instead of
    silently ignoring the pin. Pass them with `-params-file` or on the command line
    (booleans only in a `-params-file`).

## Cluster settings

- **Scheduler settings** — set `slurm_partition`, `slurm_account`, `slurm_qos` in
  `site.config` (or as `--slurm_partition` etc.).
- **Resource caps** — `max_memory` and `max_cpus` have **no default** (required at
  launch); `max_time` defaults to `240.h`. All three clamp every per-process request;
  memory and time scale with `task.attempt`, so retries automatically ask for more (up
  to the cap). See the [cluster parameters](parameters.md#cluster-resources) and
  [Resources](resources.md).
- **How many jobs at once** — `--concurrency` (default 5) moves both the per-process and
  the whole-pipeline limit; see [Resources → Execution & concurrency](resources.md#execution-concurrency).
- **GPU jobs** — see [GPU or no GPU](gpu.md#on-a-cluster).
- **Cache images once** — point `NXF_SINGULARITY_CACHEDIR` (and `SINGULARITY_CACHEDIR`)
  at a shared, writable, persistent path, so images are pulled once and reused across
  runs and users:

    ```bash
    export NXF_SINGULARITY_CACHEDIR=/path/to/shared/singularity_cache
    ```

- **`DEEPCELL_ACCESS_TOKEN` on Singularity** (CellSAM only) — `singularity.envWhitelist`
  forwards it to the container by *reference*, so it must be present in the
  environment on the compute node itself, and a SLURM site launching with
  `--export=NONE` will silently stop delivering it.

## Execution profiles

Profiles are defined in `nextflow.config` and combine with commas: pick one
**execution** profile and one **container** profile.

| Profile | Kind | What it does |
|---|---|---|
| `docker` | container | Run processes in Docker (laptop, workstation). Passes `--platform linux/amd64`. |
| `singularity` | container | Run processes in Singularity/Apptainer (clusters). Apptainer uses this same profile. |
| `conda` | container | Enables Conda. No MIRAGE process declares a Conda environment, so this profile does not provide the tools; use `docker` or `singularity`. |
| `local` | executor | Local executor with conservative caps (4 CPU / 16 GB / 72 h). |
| `slurm` | executor | Submit each process as a SLURM job. |
| `test` / `test_full` | data + caps | Bundled synthetic datasets, small resource caps, CPU segmentation. |
| `instantseg_test` / `cellsam_test` | data + caps | Test profiles exercising those segmentation backends. |

```bash
# Laptop demo
nextflow run . -profile test,docker --outdir results
# Laptop or workstation, your own slides, no GPU
nextflow run . -profile local,docker -params-file params/no_gpu.json --input samplesheet.csv --outdir results
# Cluster
nextflow run . -profile slurm,singularity --input samplesheet.csv --outdir results -c site.config
```

!!! note "Your own site profile"
    There is no shipped site profile. Copy `conf/site.config.template` to
    `site.config` and layer it with `-c site.config`, as above. A named profile is
    possible too (`profiles { mysite { includeConfig 'conf/mysite.config' } }`),
    but a profile body is evaluated while `nextflow.config` is parsed, whereas a
    `-c` file is layered afterwards and wins on params.

## Pre-pulling container images { #pre-pulling-container-images }

The first real run downloads each tool's image, which can take several minutes. Pull
them ahead of time to make that first run quicker. MIRAGE runs **13 images**: 12
first-party `bolt3x/mirage-*` images, each built from `containers/<name>/` (the VALIS one
on top of the digest-pinned upstream `cdgatenbee/valis-wsi:1.2.0`), and one external, the
vendored nf-core `BASICPY` module's image, which `conf/modules.config`'s
`withName: 'BASICPY'` block repins to a digest:

| Image | Used for |
|---|---|
| `bolt3x/mirage-convert:1.0.0` | `CONVERT_IMAGE` (format conversion) |
| `bolt3x/mirage-preprocess:1.0.0` | `SPLIT_CHANNELS`, `PREFLIGHT_SCALE`, `TILE_FOR_BASIC` / `APPLY_PROFILES` (illumination correction around `BASICPY`), `GENERATE_PREPROCESS_QC`, `GENERATE_QC_REPORT`, `AGGREGATE_SIZE_LOGS` |
| `bolt3x/mirage-quantify:1.0.0` | quantification, property extraction, GeoJSON export, postprocessing QC |
| `bolt3x/mirage-stardist:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method stardist` (default) |
| `bolt3x/mirage-instanseg:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method instantseg` |
| `bolt3x/mirage-cellsam:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method cellsam` |
| `bolt3x/mirage-merge:1.0.0` | `MERGE_AND_PYRAMID`, `EXTRACT_MASK_SERIES` |
| `bolt3x/mirage-regqc:1.0.0` | `GENERATE_REGISTRATION_QC` |
| `bolt3x/mirage-stare:1.0.0` | the `tiled` (STARE) registration backend (default), and `WARP_SEG_QC`'s tiled path |
| `bolt3x/mirage-spatialdata:1.0.0` | `EXPORT_SPATIALDATA` |
| `bolt3x/mirage-segeval:1.0.0` | `SEG_QUALITY_EVAL`, `MERGE_SEG_EVAL` (opt-in) |
| `bolt3x/mirage-valis:1.0.0` (upstream `cdgatenbee/valis-wsi:1.2.0`, digest-pinned, with only the non-root fixes) | `REGISTER`, and `WARP_SEG_QC`'s VALIS path |
| `docker.io/labsyspharm/basicpy-docker-mcmicro@sha256:355b14e2ec80b7b152272f333afd47234f007d0d37633b3ec948e87ec2c8e9b4` (vendored nf-core module's own image, repinned) | `BASICPY` — pulled only when `skip_preprocessing` is set `false` (illumination correction is off by default) |

=== "Singularity / Apptainer"

    ```bash
    singularity pull docker://docker.io/labsyspharm/basicpy-docker-mcmicro@sha256:355b14e2ec80b7b152272f333afd47234f007d0d37633b3ec948e87ec2c8e9b4
    singularity pull docker://bolt3x/mirage-<component>:1.0.0
    ```

=== "Docker"

    ```bash
    docker pull docker.io/labsyspharm/basicpy-docker-mcmicro@sha256:355b14e2ec80b7b152272f333afd47234f007d0d37633b3ec948e87ec2c8e9b4
    docker pull bolt3x/mirage-<component>:1.0.0   # e.g. convert, preprocess, quantify, stare, valis...
    ```

!!! note "Where tags live, and what they pin"
    Every MIRAGE-owned process names its image in `modules/local/*.nf`'s
    `container` directive, or — for the per-backend images — in
    `lib/SegBackends.groovy` / `lib/WarpBackends.groovy`; `conf/modules.config`
    owns those processes' *resources*, not their images. The one exception is
    the single vendored nf-core module, `BASICPY`: its own `modules/nf-core/basicpy/main.nf`
    pins `docker.io/labsyspharm/basicpy-docker-mcmicro:1.2.0-patch5` by tag, and
    `conf/modules.config`'s `withName: 'BASICPY'` block overrides it to the same
    image pinned by content digest, so the vendored
    module stays byte-identical to upstream
    (`tests/test_basicpy_module_is_vendored_unmodified.py`). MIRAGE never uses
    `:latest`. Every `FROM` inside `containers/*/Dockerfile` is **digest-pinned**
    (`FROM <base>@sha256:...`), so an image rebuilt later still starts from the
    same base bytes.

    The first-party images themselves are pinned by **version tag**
    (`bolt3x/mirage-<component>:<manifest.version>`), because a digest cannot
    exist before `release.yml` has pushed them. The one external pulled at run
    time (`docker.io/labsyspharm/basicpy-docker-mcmicro`) is referenced by content
    digest directly, and so is upstream's `cdgatenbee/valis-wsi:1.2.0` as the
    base `containers/valis` builds FROM. Every task's `versions.yml` records
    `container:` as `task.container` resolved it — a tag for a first-party image,
    a digest for an external one. The first-party images' *published* digests are
    recorded in [`containers/README.md`](https://github.com/sceriff0/mirage/blob/main/containers/README.md)'s
    image-mapping table once a release has pushed them.

## Offline / air-gapped execution { #offline-air-gapped-execution }

Nextflow downloads its `nf-schema` plugin from the plugin registry the first time
the pipeline runs. On a compute node with no outbound network that fails at launch:

```text
ERROR ~ Plugin with id nf-schema not found in any repository
```

Pre-provision the plugin. Either route below works — pick one.

**1. A shared plugins directory** (recommended for a cluster)

On a host that *does* have network access, and with the same Nextflow version
the cluster runs:

```bash
export NXF_PLUGINS_DIR=/shared/nextflow/plugins   # readable from the compute nodes
nextflow plugin install nf-schema@2.5.1
ls "$NXF_PLUGINS_DIR"      # -> nf-schema-2.5.1/
```

`rsync` that directory across if it is not already on shared storage, then
export the same variable wherever the pipeline is launched (`~/.bashrc`, the
SLURM job script, or a site profile). Nextflow resolves plugins from
`NXF_PLUGINS_DIR` and never contacts the registry when the pinned version is
already present.

**2. The default cache, primed from the login node**

With no `NXF_PLUGINS_DIR` set, plugins live in `$NXF_HOME/plugins` (default
`~/.nextflow/plugins`). If the login node has network access and `$HOME` is
shared with the compute nodes, priming it once is enough:

```bash
nextflow plugin install nf-schema@2.5.1     # on the login node
```

Then set `NXF_OFFLINE=true` on the cluster so Nextflow skips every remote check
(plugin registry, version check) instead of waiting for it to time out:

```bash
export NXF_OFFLINE=true
nextflow run . -profile slurm,singularity -c site.config --input samplesheet.csv --outdir results
```

The plugin version is pinned exactly, never as a range, so the copy you
provision is the copy that runs. Re-provisioning is only needed when the pin in
`nextflow.config` changes.

The container images and the models need the network too: pre-pull the images
([above](#pre-pulling-container-images)), pre-download the StarDist model into
`segmentation_model_dir`, and for CellSAM set `cellsam_model_path` — see
[Choosing methods](choosing_methods.md#segmentation-stardist-instanseg-or-cellsam).

## When something goes wrong

Singularity cache errors, `mksquashfs` kills during image pulls and out-of-memory exits
are covered in [Troubleshooting](troubleshooting.md).

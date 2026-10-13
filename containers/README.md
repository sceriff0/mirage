# Mirage Container Images

This directory vendors the Dockerfiles for every image the Mirage pipeline runs.

**The build context is the REPOSITORY ROOT, not `containers/<name>/`.**
`.github/workflows/containers.yml` passes `context: .` with
`file: containers/<name>/Dockerfile`, because every image takes its Python pins from
[`requirements/`](../requirements) — `constraints.txt` is the one version authority shared
by all seventeen images AND by the CI workflows, and a per-directory context cannot reach
outside itself. A local build must therefore be run from the repository root:

```sh
docker build -f containers/segeval/Dockerfile -t bolt3x/mirage-segeval:1.0.0 .
```

Three images take no constraints file and say so in their own header:
`containers/spatialdata` (the python:3.12 / `zarr>=3` island), `containers/convert`
(bioio caps `tifffile` and forces `numpy` 2.x) and `containers/palom` (PALOM caps
`scikit-image` and `tifffile`; a registration-benchmark image, see below). The divergences are recorded in
`tests/test_container_harmonisation.py`'s `PINNED_EXCEPTIONS`, and
`tests/test_ci_stack_pinned.py` derives its exception list from that table rather than
from a hand-written name list.

**How a container's requirements are checked.** `tests/test_container_harmonisation.py`'s
`test_container_installs_what_its_scripts_import` walks each image's own `bin/` scripts
(and the local modules they import) with `ast`, at MODULE SCOPE only — an `import` nested
inside a `def` executes only if something calls it, so it is a runtime dependency, not a
module one, and the walker does not report it. A `class` body is excluded too, for
symmetry rather than the same reason: it DOES execute at import time, but no script in
`bin/` imports anything inside a class body, so the exclusion costs nothing. A container
whose scripts reach a package ONLY through such a lazy import declares it explicitly in that file's
`REQUIRED_RUNTIME_IMPORTS` table (`{container: {import_name: reason}}`), read alongside the
walker's own module-scope findings before the installed-package check runs. Each entry is
paired with `test_required_runtime_imports_are_actually_reached`, which walks the container's
scripts unrestricted (`ast.walk`) to prove the lazy import still exists somewhere reachable —
an entry whose import has since been removed fails there rather than silently letting the
container drop a dependency nothing in it needs any more.

Images are built and published to **Docker Hub**, one public repository per image,
tagged with the pipeline version:

```
bolt3x/mirage-<component>:<manifest.version>
```

The build-context directory, the Docker Hub repository and the `container` directive in
`modules/local/*.nf` all use the same component name, so there is nothing to translate
between them. `tests/test_container_image_naming.py` asserts that.

Publishing is automated by [`.github/workflows/containers.yml`](../.github/workflows/containers.yml)
and the release flow in [`.github/workflows/release.yml`](../.github/workflows/release.yml).
CI authenticates with the `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repository
secrets. Docker Hub images are **public**, so no pull credentials are needed on
the HPC/cluster side (unlike GHCR, whose default-private packages caused
`403 Forbidden` on `singularity pull`).

**`containers/images.json` is the build list.** The GitHub Actions matrix and the
workflow's `only` filter both read it, so adding a directory here without adding a
row there means the image is never built or pushed — the run goes green and the tag
simply does not exist on Docker Hub. That is how `:tiled` shipped unpublished.
`tests/test_container_build_matrix.py` fails on the mismatch in both directions,
and additionally on any `mirage-<dir>:<tag>` a module pulls that the matrix does
not publish.

To rebuild ONE image without republishing the other ten:

```sh
gh workflow run containers.yml --ref <branch> -f version=<tag> -f only=segeval
```

> The tag is the pipeline version from `manifest.version`, and the modules pin it
> directly. `:latest` is never used.
>
> This replaces an earlier layout that used a single repository with one *content-descriptive*
> tag per image (`:preprocess`, `:convert_bioformats_2`, …). That put component names and real
> versions (`v2.3`, `v2.4`) in the same namespace, and — because the build workflow resolved a
> `version` input it then never applied to the tag — every publish overwrote one mutable name,
> leaving no earlier image to roll back to.
>
> **Reproducibility.** Every `FROM` in this directory, and both external images the pipeline
> pulls (`cdgatenbee/valis-wsi`, `docker.io/labsyspharm/basicpy-docker-mcmicro`), are pinned
> by content digest (`@sha256:…`) as of 2026-09-02 — ruling R6, guarded by
> `tests/test_base_images_are_digest_pinned.py`. The first-party images themselves are pinned
> by tag, because a digest cannot exist before `release.yml` has pushed them; their published
> digests are recorded in this table by the release step (`release.yml` → plan 14) once the
> images are pushed; until then the tag is the identifier.

## Image mapping

| Build context (`containers/<name>/`) | Docker Hub image:tag | Pipeline process(es) that use it | Base image and stack |
| --- | --- | --- | --- |
| `convert` | `bolt3x/mirage-convert:1.0.0` | `CONVERT_IMAGE` | `eclipse-temurin:21-jre-jammy` + Python 3.10 + `bioio` 3.5.0 with **six** reader plugins (`bioio-ome-tiff`, `-tifffile`, `-nd2`, `-czi`, `-lif`, `-bioformats`), and the Bio-Formats jars + JVM cache **baked under `/root`** so a read-only-`$HOME` / air-gapped cluster never fetches them. `BIOFORMATS_VERSION` pins `ome:formats-gpl:8.1.1`. Takes no `constraints.txt` (documented exception: `bioio` caps `tifffile` and forces `numpy` 2.x). |
| `preprocess` | `bolt3x/mirage-preprocess:1.0.0` | `APPLY_PROFILES`, `GENERATE_PREPROCESS_QC`, `GENERATE_QC_REPORT`, `SPLIT_CHANNELS`, `TILE_FOR_BASIC`, `PREFLIGHT_SCALE`, `AGGREGATE_SIZE_LOGS` (7 modules) | `ubuntu:22.04` + Python 3.11 + numpy/scipy/scikit-image/tifffile/zarr. **No BaSiCPy** — illumination correction runs in the vendored nf-core `BASICPY` module's own image. |
| `quantify` | `bolt3x/mirage-quantify:1.0.0` | `QUANTIFY`, `EXTRACT_CELL_PROPERTIES`, `EXTRACT_NUCLEI_PROPERTIES`, `EXPORT_GEOJSON`, `GENERATE_POSTPROCESSING_QC`, `SEG_QC_GEOJSON`, `MERGE_QUANT_CSVS`, `NUCLEAR_RETENTION`, `CELL_QC` (9 modules, one `container` directive each) | `nvidia/cuda:12.2.2-devel-ubuntu22.04` + numpy/pandas/scipy/scikit-image/matplotlib/tifffile |
| `stardist` | `bolt3x/mirage-stardist:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method stardist` | `tensorflow/tensorflow:2.15.0-gpu-jupyter` + StarDist 0.9.1 + csbdeep 0.8.2, with `gputools` 0.3.1 / `edt` 3.1.2 as StarDist's optional OpenCL and EDT accelerators. TensorFlow comes from the base, not from pip. |
| `cellsam` | `bolt3x/mirage-cellsam:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method cellsam` | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` + `cellSAM` pinned to a git commit. torch 2.6.0 is the first release carrying the CVE-2025-32434 (`torch.load`) fix, and cellSAM downloads weights, so that path is reachable. |
| `instanseg` | `bolt3x/mirage-instanseg:1.0.0` | `SEGMENT` / `SEG_QC_SEGMENT` when `--seg_method instantseg` *(default)* | `pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime` + `instanseg-torch` 0.1.1 |
| `merge` | `bolt3x/mirage-merge:1.0.0` | `MERGE_AND_PYRAMID` (1 module; the dev branch's add_cycle path adds `EXTRACT_MASK_SERIES`) | `cdgatenbee/valis-wsi:1.0.0` — the **same upstream VALIS image** `REGISTER` runs in, which is what lets this image reuse its libvips build — plus numpy/tifffile/imagecodecs/opencv-python/zarr for the pyramid write. |
| `regqc` | `bolt3x/mirage-regqc:1.0.0` | `GENERATE_REGISTRATION_QC` | `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04` + numpy/scipy/scikit-image/tifffile/zarr and the **non-headless** `opencv-python` (`ffmpeg`/`libsm6`/`libxext6` are its runtime libraries). Nothing else: the TensorFlow/StarDist stack, Miniconda and `bftools` were removed 2026-09-02, having been imported by nothing. |
| `stare` | `bolt3x/mirage-stare:1.0.0` | `TILED_COARSE`, `TILED_REG_TILE`, `TILED_SOLVE`, `TILED_STITCH`, and `WARP_SEG_QC`'s tiled backend (STARE: `registration_method='tiled'`) | `python:3.11-slim` (same digest the retired `tiled` image used) + `procps` + numpy/scipy/scikit-image/tifffile/zarr/numcodecs/imagecodecs (`requirements/stare.txt`) and the `stare` package (a pinned release of [sceriff0/stare](https://github.com/sceriff0/stare), `requirements/stare.txt`). **Slim:** no torch, no kornia, no libGL, no learned weights — `smoke.sh` asserts torch is NOT importable. No JVM, no BioFormats, no libvips, no CUDA runtime. Replaces `bolt3x/mirage-tiled:1.0.0` (build context `containers/tiled`, removed), which carried torch/kornia and the DISK+LightGlue checkpoints for the COARSE front-end retired on 2026-09-27. |
| `spatialdata` | `bolt3x/mirage-spatialdata:1.0.0` | `EXPORT_SPATIALDATA` (+ the out-of-band `bin/join_flowpath.py` cohort join) | `python:3.12-slim` + spatialdata/anndata/geopandas/shapely/pyarrow/**zarr 3** — CPU only, no JVM, no GPU. Takes no `constraints.txt` (documented `python:3.12` / `zarr>=3` island); `requirements/spatialdata.txt` is exact-pinned. |
| `segeval` | `bolt3x/mirage-segeval:1.0.0` | `SEG_QUALITY_EVAL`, `MERGE_SEG_EVAL` (opt-in, `-params-file params/seg_quality_eval.json`) | `python:3.11-slim` + numpy/scipy/pandas/scikit-image/scikit-learn/matplotlib/tifffile/xmltodict, for the vendored CellSegmentationEvaluator 1.5.19 subset under `bin/utils/cse/`. **No `aicsimageio`** — it is on the forbidden list and cannot coexist with the harmonised `tifffile`. |
| `valis120` | `bolt3x/mirage-valis120:1.0.0` | none — the registration benchmark's `valis120` and `valis120_hires` rows (`benchmarks/regbench/methods/valis.py`) | **upstream's own image** `cdgatenbee/valis-wsi:1.2.0`, by digest: Ubuntu 24.04, Python 3.12.3 in a virtual environment under `/root/.local`, `valis-wsi` 1.2.0, torch 2.7.1 (the CUDA build; the benchmark gives these rows no GPU), torchvision 0.22.1, Kornia 0.8.1, pyvips 3.0.0 on a source-built libvips 8.16.0, OpenJDK 11, Bio-Formats 7.0.0 (read from Docker Hub's registry API, not from a running container; the Dockerfile's header lists what was read). **Nothing of that stack is reinstalled.** Added on top (`requirements/valis120.txt`, `-c constraints.txt` and a freeze of the base as a second constraint): `zarr` 2 / `numcodecs` / `imagecodecs` / `pytest`, and `stare-registration` with `--no-deps` from the one line of `requirements/stare.txt`. The one version of upstream's stack that changes is `tifffile` 2025.6.11 → 2025.5.10, which STARE's reader needs and VALIS does not import. **Made usable by a user who is not root** (`place_assets.py`, no network): `/root` opened for reading (upstream's Python lives there, mode 0700), the Bio-Formats jar linked beside the installed package, `TORCH_HOME=/opt/torch` linked to upstream's baked DISK + LightGlue weights (sha256-checked), `valis/data` writable; `smoke.sh` imports VALIS and starts its JVM as uid 65534 with the network closed. **Run it with `--writable-tmpfs` under Apptainer/Singularity**: VALIS 1.2.0 rewrites `valis/data/bf_formats.txt` inside the package on every JVM start. |
| `deeperhistreg` | `bolt3x/mirage-deeperhistreg:1.0.0` | none — the registration benchmark's eight `deeperhistreg_*` rows, GPU and CPU (`benchmarks/regbench/methods/deeperhistreg.py`) | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (the digest `cellsam` uses; torch and torchvision come from the base) + `procps` + the PyPI release `deeperhistreg` 1.0.1 and the whole stack its repository's `requirements.txt` lists, because the package declares no dependencies (`requirements/deeperhistreg.txt`: `-c constraints.txt`, `-r stare.txt`, every resolved distribution pinned; `opencv-contrib-python-headless`, `openslide-bin` and `pyvips-binary` stand in for system libraries). `smoke.sh` asserts the torch imported is still the base's CUDA build. |
| `palom` | `bolt3x/mirage-palom:1.0.0` | none — the registration benchmark's `palom` and `palom_b1024` rows (`benchmarks/regbench/methods/palom.py`) | `python:3.10-slim` + `procps` + the PyPI release `palom` 2026.4.1 — CPU only, no JVM, no GPU. **Takes no `constraints.txt`**: PALOM requires `scikit-image<0.20` and `tifffile<=2024.7.2`, so `requirements/palom.txt` is a complete pin file (62 distributions) and the four divergences are `PINNED_EXCEPTIONS`. `stare-registration` is installed with `--no-deps` from the one line of `requirements/stare.txt` (its `scikit-image>=0.24` cannot be met here; the benchmark uses only `stare.slide_io` and `stare.ome`). |
| `stare-regbench` | `bolt3x/mirage-stare-regbench:1.0.0` | none — the registration benchmark's `stare` run and its `tissue` and `score` steps (`benchmarks/regbench/methods/stare.py`, `tissue.py`, `score.py`) | `python:3.11-slim` (the digest `stare` uses) + `procps` + everything `requirements/stare.txt` installs, by `-r` (the same pinned `stare` release and numeric stack as the pipeline's image, `-c constraints.txt`), plus `psutil` (so STARE's memory is read by the same instrument as every other method's), `pytest`, and `opencv-python-headless` 4.10.0.82 for the tissue-level block: `tissue` and `score` run here, so one OpenCV stands behind a whole result. Every resolved distribution pinned (`requirements/stare-regbench.txt`). `containers/stare` and `requirements/stare.txt` are untouched. Still no torch (`smoke.sh` asserts it). |
| `valis100` | `bolt3x/mirage-valis100:1.0.0` | none — the registration benchmark's `valis_mirage` row (`benchmarks/regbench/methods/valis_mirage.py`) | **upstream's own image** `cdgatenbee/valis-wsi:1.0.0`, by the digest `REGISTER` runs + `procps`. **Nothing of that stack is reinstalled.** Added on top (`requirements/valis100.txt`, constrained by a freeze of the base; takes no `constraints.txt`): `zarr` 2.17.2 (not the harmonised 2.18.3, which needs numpy>=1.24 where upstream holds 1.23.5) / `numcodecs` / `imagecodecs` / `pytest`, and `stare-registration` with `--no-deps`, which the harness's slide reader needs and upstream's image lacks (the benchmark's preflight refused the bare image on 2026-10-13). `smoke.sh` fails the build if a line of upstream's freeze changed. |
| `ashlar120` | `bolt3x/mirage-ashlar120:1.0.0` | none — the registration benchmark's `ashlar`, `ashlar_clean`, `affine_ashlar` and `affine_ashlar_clean` rows (`benchmarks/regbench/methods/ashlar.py`) | **the official image** `labsyspharm/ashlar:1.20.0`, by digest + `procps`. **Nothing of that stack is reinstalled.** Added on top (`requirements/ashlar120.txt`, constrained by a freeze of the base): `imagecodecs` / `pytest`, and `stare-registration` with `--no-deps`, which tifffile needs for the cohort's compressed tiles and the official image lacks (same preflight). `smoke.sh` fails the build if a line of the official freeze changed. |
| VALIS (not vendored) | `cdgatenbee/valis-wsi:1.0.0` (upstream) | `REGISTER`, and `WARP_SEG_QC`'s VALIS backend | upstream maintained image — **not rebuilt or published by us** (see the note below). Referenced by content digest, not by tag. |

> **Six of the seventeen are not pipeline images.** `valis120`, `valis100`, `ashlar120`,
> `deeperhistreg`, `palom` and `stare-regbench` run no `modules/local` process: they belong
> to the registration benchmark (`benchmarks/regbench`, run with the repository
> bind-mounted). The first five each hold one competitor, as its authors publish it:
> `deeperhistreg` and `palom` are the PyPI releases on the stack their metadata asks for,
> and `valis120`, `valis100` and `ashlar120` are built FROM the images their authors
> publish, with nothing of their stack reinstalled (`valis100` is the digest the pipeline's
> `REGISTER` runs; it and `ashlar120` add only the codecs the benchmark's slide reader
> needs). `stare-regbench` is the benchmark's own STARE image (the pipeline's `stare` plus
> `psutil`, `pytest` and OpenCV), where its scorer also runs. All six are built,
> smoke-tested and published by the same workflow, and obey the same rules (digest-pinned
> base, pins in `requirements/<name>.txt`, `procps`, a `smoke.sh` run at build time). Two
> guards that derive an image's needs from the pipeline scripts it runs do not apply to
> them; `tests/test_container_harmonisation.py`'s `BENCHMARK_IMAGES` is that list and says
> why.

> **zarr major versions differ on purpose.** `spatialdata` pins `zarr>=3.0.0`
> (required by `spatialdata>=0.8.0`, which writes NGFF `"0.5-dev-spatialdata"`),
> while `stare` pins `zarr==2.18.3` for `tifffile`'s `aszarr` region reads. Keeping
> them in separate images is what lets both constraints hold without either being
> downgraded.

> The historical typo'd context directory `istantseg` was renamed to `instanseg` as part
> of the one-repo-per-image rename, so the build-context directory, the Docker Hub
> repository and the `container` directive all now agree on the spelling. The published
> image is `bolt3x/mirage-instanseg:1.0.0` on **Docker Hub** (not GHCR — see "Publishing"
> above), even though `params.seg_method` is `instantseg`.

### VALIS — uses the upstream image (not vendored)

`REGISTER` references the maintained upstream image [`cdgatenbee/valis-wsi:1.0.0`](https://hub.docker.com/r/cdgatenbee/valis-wsi)
directly. We do **not** vendor or republish it: its from-source libvips build is
heavy (and the original vendored Dockerfile failed to build in CI — `meson` was
missing). The upstream image is `linux/amd64` and already battle-tested, so there
is little value in re-hosting it. It is therefore excluded from
`containers.yml`. If you ever want a self-hosted copy, add a `containers/valis/`
Dockerfile that installs `meson`/`ninja` before the libvips build and re-add
`valis` to the build matrix.

All published images are built for **`linux/amd64`** (the pipeline's HPC target).

## Runtime requirements every image must satisfy

**`procps` (i.e. `ps`) is mandatory in every image**, regardless of what the process
actually runs. `params.enable_trace` defaults to `true` (`nextflow.config`), so Nextflow
injects its task-metrics wrapper into *every* task, and that wrapper begins with a hard
guard (`nextflow/executor/command-trace.txt`):

```bash
command -v ps &>/dev/null || { >&2 echo "Command 'ps' required by nextflow to collect task metrics cannot be found"; exit 1; }
```

It runs **before** the process's `script:` block, so a missing `ps` fails the task with
**exit status 1, empty stdout and no traceback** — the failure looks like the tool
crashed silently, not like a missing system package. Debian/Ubuntu bases do **not** ship
`procps`: `python:*-slim` and `ubuntu:22.04` both lack it. Rather than reason about which
bases happen to have it — a claim nothing checked, and one this file previously got wrong
— **all seventeen install it explicitly**, and every `containers/<name>/smoke.sh` ends with a
`ps -e` assertion that runs at build time and again through `containers.yml`'s `docker
run`. `apt-get install procps` is a no-op where the base already provides it. Guarded by
`tests/test_container_smoke_tests.py::test_every_smoke_script_proves_ps_exists`.

`AGGREGATE_SIZE_LOGS` is why this is not theoretical: it ran in bare `ubuntu:22.04` until
2026-09-02, so with the shipped `enable_trace` default it failed with exit status 1 and
empty stdout on every real run. It now uses `bolt3x/mirage-preprocess`.

When adding a new build context, either install `procps` or verify the base has it, and
add a build-time assertion so a regression fails the **build** rather than a cluster run:

```dockerfile
RUN ps -e -o pid= -o ppid= > /dev/null && echo "procps OK"
```


## Contexts that are NOT vendored

Only the seventeen directories above are built and published. Two images the pipeline pulls are
not built here at all:

| Image | Why it is not vendored |
| --- | --- |
| `cdgatenbee/valis-wsi` | Upstream-maintained; its from-source libvips build is heavy and an earlier vendored attempt failed in CI on a missing `meson`. `containers/merge` builds **from** it, so the two can never disagree about libvips. |
| `docker.io/labsyspharm/basicpy-docker-mcmicro` | The vendored nf-core `BASICPY` module's own image (`modules/nf-core/basicpy/main.nf`). Re-hosting it would mean maintaining a fork of the vendored image too. |

Both are referenced by content digest; `tests/test_container_image_naming.py`'s
`EXTERNAL_ALLOWED` is the list of externals the pipeline is permitted to pull at all.

## `container` directives

The pipeline modules (`.nf` files) reference the Docker Hub tags in the mapping
table above directly (e.g. `container 'bolt3x/mirage-preprocess:1.0.0'`).
The image NAME is content-descriptive (one repository per image); the TAG is a
fixed, immutable SemVer version tied to `manifest.version` — pin it explicitly,
never `:latest`, so runs stay reproducible.

`modules/local/segment.nf` is the one dynamic case: `SEGMENT` picks its container at runtime
from `params.seg_method`, via the table in `lib/SegBackends.groovy` (`cellsam` →
`mirage-cellsam`, `instantseg` → `mirage-instanseg`, otherwise the StarDist default
`mirage-stardist`).

`REGISTER` uses the upstream `cdgatenbee/valis-wsi:1.0.0` directly; VALIS is deliberately not
re-hosted, because its from-source libvips build is heavy and upstream maintains it.

# MIRAGE

[![CI](https://github.com/sceriff0/mirage/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sceriff0/mirage/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/mirage-pipeline/badge/?version=stable)](https://mirage-pipeline.readthedocs.io/)
[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A525.04.0-23aa62.svg)](https://www.nextflow.io/)
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed?logo=docker)](https://www.docker.com/)
[![run with singularity](https://img.shields.io/badge/run%20with-singularity-1d355c.svg)](https://sylabs.io/docs/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<!-- DOI: filled in by plan 14 once the Zenodo deposition exists. The badge is a
     CONCEPT DOI (it resolves to the newest version, so it never needs updating):
     [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.<CONCEPT_ID>.svg)](https://doi.org/10.5281/zenodo.<CONCEPT_ID>)
     Zenodo shows both the concept DOI and the version DOI on the deposition page;
     take the one labelled "Cite all versions". -->

## Introduction

MIRAGE is a Nextflow DSL2 pipeline for whole slide image (WSI) processing. It supports multiplex fluorescence imaging workflows end-to-end: illumination correction, multi-modal registration, cell segmentation, single-cell marker quantification, and QuPath-compatible GeoJSON export for interactive gating via FlowPath. The pipeline is designed for HPC SLURM environments but also runs locally via Docker or Singularity.

## Pipeline Summary

1. **Image format conversion** — every input is read through a dispatch table
   (`bin/utils/ome_io.py`) and staged as OME-TIFF with the nuclear marker moved to
   channel 0. See [Supported formats](#supported-formats).
2. **Illumination correction** (`TILE_FOR_BASIC` → `BASICPY` → `APPLY_PROFILES`) — per-channel
   flatfield correction via the BaSiC algorithm, through nf-core's
   [BASICPY](https://github.com/peng-lab/BaSiCPy) module at upstream defaults (no darkfield is
   estimated or removed); produces a corrected OME-TIFF per image.
3. **Multi-modal image registration** — two interchangeable backends, chosen with
   `--registration_method`:
   - **`valis`** (default) — graph-based whole-stack alignment via
     [VALIS](https://github.com/MathOnco/valis); rigid + non-rigid, with optional staged
     micro-registration.
   - **`tiled`** (STARE) — JVM-free, internally tiled and fully parallel; coarse DISK/LightGlue
     anchor, per-tile refinement, global solve, tiled warp. Runs on a workstation at
     `--reg_tiled_mode low`.
4. **Cell segmentation** (`SEGMENT`) — three interchangeable backends via `--seg_method`:
   **InstanSeg** (default; channel-invariant, runs on an unconfigured clone),
   **StarDist** (needs a trained model directory), **CellSAM** (needs a weight download).
   Outputs nuclear and whole-cell instance masks per patient.
5. **Single-cell marker quantification** (`QUANTIFY`) — per-cell intensity statistics across
   every registered channel, with optional Nucleus / Cytoplasm / Cell compartments; outputs CSV
   tables.
6. **QuPath GeoJSON export** (`EXPORT_GEOJSON`) — every cell with raw marker intensities and
   morphological measurements, in QuPath-native GeoJSON, for interactive gating via FlowPath.
7. **Pyramidal OME-TIFF export** — all registered channels assembled into a tiled,
   multi-resolution OME-TIFF for QuPath / napari / OMERO.
8. **SpatialData export** — an additive scverse-native `.zarr` store carrying the same masks,
   polygons, measurements and QC. Written by default; `skip_spatialdata_export` turns it off.
9. **Quality control reporting** — before/after registration overlays, segmentation and
   intensity QC, an aggregated HTML report, and a computational-resource report.

## Supported formats

Every reader is installed in the `mirage-convert` image; the dispatch happens in
`bin/utils/ome_io.py::detect_reader`, and an extension no reader claims raises
`UnsupportedFormatError` at conversion time rather than producing a corrupt OME-TIFF.

| Extension | Reader |
|---|---|
| `.ome.tif`, `.ome.tiff`, `.tif`, `.tiff` | `bioio` |
| `.nd2`, `.czi`, `.lif` | `bioio` |
| `.svs`, `.qptiff`, `.vsi`, `.scn`, `.mrxs`, `.bif`, `.ims` | `bioio-bioformats` (Bio-Formats, jars baked into the image) |
| `.ndpi`, `.ndpis` | `tifffile` |
| `.h5`, `.hdf5` | `hdf5` |

Everything above is exercised on synthetic fixtures in CI on every push. The five
vendor formats that cannot be synthesised (`.czi`, `.nd2`, `.lif`, real scanner
`.ndpi` bytes, `.svs`) need a cluster run against real vendor files, and as of
this release none has been recorded yet — the current state, tracked format by
format, is in
[`docs/validation/format_validation.md`](docs/validation/format_validation.md).

## Quick Start

### Prerequisites

1. **Java 11 or newer** — Nextflow's runtime. Check with `java -version`.
2. **Nextflow `>=25.04.0`** — [install](https://www.nextflow.io/docs/latest/getstarted.html).
   Verified on 25.04.7 and 26.04.6.
3. **The `nf-schema@2.5.1` plugin** — Nextflow fetches it from the plugin registry on the first
   run. On a network-isolated compute node it must be pre-provisioned; see
   [Offline / air-gapped execution](docs/usage.md#offline-air-gapped-execution).
4. **[Singularity/Apptainer](https://apptainer.org/)** (recommended on HPC) or
   **[Docker](https://docs.docker.com/get-docker/)** (local/dev).
5. Clone the repository:

   ```bash
   git clone https://github.com/sceriff0/mirage.git
   cd mirage
   ```

### The samplesheet

One row per image; rows are grouped by `patient_id` and each patient is processed
independently. A `--start preprocessing` samplesheet needs exactly these four columns
(`lib/ParamUtils.groovy`'s `STEPS` is the source of truth), and each patient needs exactly one
`is_reference=true` row:

```csv
patient_id,path_to_file,is_reference,channels
P001,/data/raw/P001_panel1.nd2,true,DAPI|CD3|CD8|CD4
P001,/data/raw/P001_panel2.nd2,false,DAPI|PANCK|SMA|VIMENTIN
P002,/data/raw/P002_panel1.czi,true,DAPI|CD3|CD8|CD4
P002,/data/raw/P002_panel2.czi,false,DAPI|FOXP3|KI67|CD20
```

`channels` is pipe-separated, in acquisition order, and must include one of
`params.nuclear_markers` (default `DAPI`, `CELLTOX`). Later stages take a different image
column — the full table is in
[Usage → The samplesheet](docs/usage.md#the-samplesheet).

### Size your run first

`max_cpus` and `max_memory` have no default and are required — a run that omits
them is refused at launch. Make a `site.config` once and layer it on every
command with `-c`:

```bash
cp conf/site.config.template site.config
# edit max_cpus / max_memory (and the SLURM fields, if any) to match your machine
```

`site.config` is gitignored, so your paths never reach a commit.

### Full pipeline from raw images

Use `--start` to choose the entry point and `--stop` to terminate early at a given step. Both flags accept `preprocessing`, `registration`, `segmentation`, or `postprocessing`. If `--stop` is omitted, the pipeline runs through to the end.

```bash
nextflow run . \
  --input samplesheet.csv \
  --outdir results \
  --start preprocessing \
  --stop registration \
  --registration_method valis \
  -profile slurm,singularity \
  -c site.config \
  -params-file params/full_pipeline.json
```

### Resume from registration checkpoint

```bash
nextflow run . \
  --input results/csv/preprocessed.csv \
  --start registration \
  --registration_method valis \
  --outdir results \
  -profile slurm,singularity \
  -c site.config \
  -resume
```

### Resume from segmentation checkpoint

```bash
nextflow run . \
  --input results/csv/registered.csv \
  --start segmentation \
  --outdir results \
  -profile slurm,singularity \
  -c site.config \
  -resume
```

### Resume from postprocessing checkpoint

```bash
nextflow run . \
  --input results/csv/segmented.csv \
  --start postprocessing \
  --outdir results \
  -profile slurm,singularity \
  -c site.config \
  -resume
```

### Dry-run (validation only)

```bash
nextflow run . \
  --input samplesheet.csv \
  --outdir results \
  --start preprocessing \
  -c site.config \
  -params-file params/dry_run.json
```

## Documentation

Full documentation is hosted at **<https://mirage-pipeline.readthedocs.io/>**.

- [Installation](docs/installation.md) — dependencies, containers, and setup
- [Usage](docs/usage.md) — samplesheet format, parameters, profiles
- [Outputs](docs/usage.md#outputs) — output directory layout and file descriptions
- [Quick start walkthrough](docs/usage.md#quick-start-synthetic-data) — end-to-end run on the bundled test data
- [Troubleshooting](docs/usage.md#troubleshooting-faq) — common failures and remediation
- [Parameters](docs/parameters.md) — full parameter reference
- [Registration QC](docs/registration_qc.md) — interpreting registration quality metrics
- [Citation](docs/citation.md) — how to cite MIRAGE and its dependencies
- [Supported formats](docs/validation/format_validation.md) — which readers were validated, and on what
- [Resources](docs/resources.md) — per-process requests, retry policy, containers
- [Developing](docs/developing.md) — running the suite, the guards, and how to add a process

To build the documentation locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Credits

MIRAGE was developed by Valerio Fassi and Yinxiu Zhan at the Istituto Europeo di Oncologia. Questions and bug reports: the [GitHub issue tracker](https://github.com/sceriff0/mirage/issues).

## Citations

If you use MIRAGE in your research, please cite the relevant tools listed in [CITATIONS.md](CITATIONS.md).

Core dependencies include:

- **VALIS** — Gatenbee et al. (2023), *Nature Communications* — [https://doi.org/10.1038/s41467-023-40218-9](https://doi.org/10.1038/s41467-023-40218-9)
- **StarDist** — Schmidt et al. (2018), *MICCAI* — [https://doi.org/10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30)
- **BaSiCPy** — Peng et al. (2017), *Nature Communications* — [https://doi.org/10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)
- **Nextflow** — Di Tommaso et al. (2017), *Nature Biotechnology* — [https://doi.org/10.1038/nbt.3820](https://doi.org/10.1038/nbt.3820)

## License

MIRAGE is released under the MIT License. See [LICENSE](LICENSE) for details.

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

**From separately acquired staining rounds of one tissue section to a single-cell table.**

MIRAGE is a Nextflow pipeline for multiplexed immunofluorescence on whole slides. When a section
is stained, imaged and stripped round after round on an ordinary microscope, every round is a
separate image, shifted, rotated and locally deformed against the others. MIRAGE aligns the rounds
at the scale of a nucleus, finds every cell, measures every marker in it, and writes files that
open directly in QuPath and in scverse tools.

Full documentation: **<https://mirage-pipeline.readthedocs.io/>**

## What it does

| Step | What happens | Default |
|---|---|---|
| **1. Preprocessing** | Reads each round (ND2, CZI, LIF, NDPI, SVS, TIFF, HDF5 and more) and writes it as OME-TIFF with the nuclear stain first. Illumination correction with BaSiC is optional. | illumination correction off |
| **2. Registration** | Aligns every round of a patient onto one reference round, on the nuclear stain. | **STARE** |
| **3. Segmentation** | Finds nuclei and whole cells on the registered image. | **StarDist** |
| **4. Postprocessing** | Measures every marker per cell in nucleus, cytoplasm and whole cell, and exports. | median per compartment |

**Registration.** [STARE](https://github.com/sceriff0/stare) (`--registration_method tiled`, the
default) measures local displacements tile by tile and fits them into one continuous displacement
field, so the memory of a task is set by a fixed window and not by the size of the slide. It needs
no GPU and a nuclear stain shared by every round.
[VALIS](https://github.com/MathOnco/valis) 1.2.0 at its defaults (`--registration_method valis`)
is kept as a second mode.

**Segmentation.** StarDist with MIRAGE's trained nuclear model (downloaded once from the
Hugging Face Hub at a pinned commit) is the default; InstanSeg and CellSAM are selected with
`--seg_method`.

**Phenotyping** is not part of MIRAGE. The exported cells carry marker intensities, and
[FlowPath](https://github.com/sceriff0/flowpath), a QuPath extension, gates them into cell
populations.

## Quick start

You need Java 17 or newer, [Nextflow](https://www.nextflow.io/) 25.04.0 or newer, and Docker (or
Apptainer/Singularity on a cluster). Every analysis tool comes in a container. New to all of
this? Follow [Run your own slides](docs/quickstart.md),
which starts from an empty computer.

**1. Get MIRAGE**

```bash
git clone https://github.com/sceriff0/mirage.git
cd mirage
```

**2. Describe your slides** in `samplesheet.csv`, one row per image file:

```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001_round1.nd2,true,DAPI|PANCK|SMA
P001,/data/P001_round2.nd2,false,DAPI|CD45|CD163
P001,/data/P001_round3.nd2,false,DAPI|CD8|CD4
```

One row per patient is the reference (`is_reference=true`); the others are aligned onto it.
`channels` lists the channel names in acquisition order, separated by `|`, and every row must
contain the nuclear stain (`DAPI`).

**3. Run**

On a laptop or workstation with an Intel or AMD processor (4 CPUs, 16 GB, segmentation on the
CPU). The images are built for x86-64 only; on an Apple-silicon Mac segmentation does not run:

```bash
nextflow run . -profile local,docker \
  -params-file params/no_gpu.json \
  --input samplesheet.csv --outdir results
```

On a SLURM cluster, after setting `max_cpus` and `max_memory` in your own `site.config`
(`cp conf/site.config.template site.config`):

```bash
nextflow run . -profile slurm,singularity -c site.config \
  --input samplesheet.csv --outdir results
```

If a run stops, repeat the command with `-resume`.

## What you get

For each patient, under `results/<patient_id>/`:

| File | What it is |
|---|---|
| `pyramid/pyramid.ome.tiff` | All rounds aligned into one multi-resolution image, for QuPath, napari or OMERO |
| `geojson/export/cells.geojson` | Every cell as a QuPath object with its measurements |
| `quantification/merged_quant.csv` | One row per cell: marker intensities and morphology |
| `spatialdata/<patient_id>.zarr` | The same cells as a SpatialData store for scverse tools |
| `qc/` | Before/after registration overlays and segmentation checks |

`results/qc/` holds an HTML report for the whole run, and `results/csv/` the checkpoint files
that let a later run start from registration, segmentation or postprocessing.

## Good to know

- **Memory.** A STARE registration task asks for 2 to 4 GB whatever the slide size. Segmentation
  and quantification scale with the slide, so whole-slide cohorts need a workstation or a cluster.
- **Sizing is required.** `max_cpus` and `max_memory` have no default. `-profile local` sets
  them for a small machine; anywhere else they come from `site.config`.
- **True/false settings** go in a `-params-file`, never on the command line.
- **Formats.** Which readers were validated, and on what, is recorded in
  [`docs/validation/format_validation.md`](docs/validation/format_validation.md).

## Documentation

- [Run your own slides](docs/quickstart.md) — from an empty computer to a running pipeline
- [Installation](docs/installation.md) — requirements, containers, sizing a run
- [Running MIRAGE](docs/usage.md) — the samplesheet, stages, checkpoints, clusters, troubleshooting
- [Inputs & outputs](docs/outputs.md) — every file a run writes
- [Parameters](docs/parameters.md) — every setting and its default
- [The pipeline](docs/pipeline.md) — every process in order
- [The paper](docs/paper.md) — what the manuscript reports and where to find it here
- [Developing](docs/developing.md) — tests, guards, adding a process

## Citing

If you use MIRAGE, cite the pipeline and the tools your run used; see
[`docs/citation.md`](docs/citation.md), [`CITATION.cff`](CITATION.cff) and
[`CITATIONS.md`](CITATIONS.md).

## Credits

MIRAGE, STARE and FlowPath are developed at the European Institute of Oncology (IEO), Milan, by
Valerio Fassi and Yinxiu Zhan with the Imaging and Screening Unit. Questions and bug reports: the
[issue tracker](https://github.com/sceriff0/mirage/issues).

## License

MIT. See [LICENSE](LICENSE).

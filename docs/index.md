---
hide:
  - navigation
  - toc
---

<div class="mirage-hero" markdown>

<div class="kick">Whole-slide multiplexed immunofluorescence</div>

# MIRAGE

From separately acquired staining rounds of one tissue section to a single-cell table: MIRAGE
aligns the rounds at the scale of a nucleus, finds every cell, measures every marker, and writes
files that open in QuPath and scverse.

<div class="mirage-badges" markdown>
[Run your own slides :octicons-arrow-right-24:](quickstart.md){ .md-button .md-button--primary }
[What a run produces](outputs.md){ .md-button }
[The paper](paper.md){ .md-button }
[:fontawesome-brands-github: Source](https://github.com/sceriff0/mirage){ .md-button }
</div>

</div>

## Where to start

<div class="paths" markdown>

<div class="path" markdown>
<div class="who">I have slides and little coding experience</div>

Start from an empty computer: install three things, fill in one table, paste one command.

[Run your own slides](quickstart.md)
</div>

<div class="path" markdown>
<div class="who">I run pipelines on a cluster</div>

Requirements, containers, sizing, checkpoints, SLURM and offline nodes.

[Installation](installation.md) · [Running MIRAGE](usage.md) · [Parameters](parameters.md)
</div>

<div class="path" markdown>
<div class="who">I want to know how it works</div>

Every process in order, the registration method, the resource model, and what the paper measured.

[The pipeline](pipeline.md) · [The paper](paper.md) · [Resources](resources.md)
</div>

</div>

## What a run does

<div class="stages" markdown>

<div class="stage" markdown>
<div class="n">1</div>
**Preprocessing**

Reads each round (ND2, CZI, LIF, NDPI, SVS, TIFF, HDF5 and more) and writes it as OME-TIFF with
the nuclear stain first. Illumination correction with BaSiC is optional and off by default.
</div>

<div class="stage" markdown>
<div class="n">2</div>
**Registration**

Aligns every round of a patient onto one reference round. **STARE**, the default, measures
displacements tile by tile and fits one slide-wide field; VALIS is the second mode.
</div>

<div class="stage" markdown>
<div class="n">3</div>
**Segmentation**

Finds nuclei and whole cells on the registered image. **StarDist** with MIRAGE's trained model
is the default; InstanSeg and CellSAM are one flag away.
</div>

<div class="stage" markdown>
<div class="n">4</div>
**Postprocessing**

Measures every marker per cell in nucleus, cytoplasm and whole cell, and exports a pyramidal
OME-TIFF, QuPath GeoJSON and a SpatialData store.
</div>

</div>

The four stages run in this order and each can be restarted on its own. MIRAGE stops at
quantified cells: cell types are called afterwards, in QuPath, with
[FlowPath](https://flowpath.readthedocs.io/).

## The command

```bash
nextflow run . -profile local,docker \
  -params-file params/no_gpu.json \
  --input samplesheet.csv --outdir results
```

That is a complete run at the defaults on a laptop or workstation.
[Run your own slides](quickstart.md) explains each part and gives the cluster form.

## Why STARE is the default

Registration of whole slides usually means holding a whole slide in memory. STARE does not: each
task reads a fixed window, so its memory does not grow with the slide.

<div class="stats" markdown>

<div class="stat" markdown>
<div class="v">0.84</div>
matched-pair nuclear Dice with STARE, against 0.80 for VALIS, over 64 round pairs
</div>

<div class="stat" markdown>
<div class="v">0.7 GB</div>
peak memory of the largest STARE registration job per case, against 31.6 GB for VALIS
</div>

<div class="stat" markdown>
<div class="v">no GPU</div>
and no STARE task above 2.4 GB, at about 2.8 times the CPU time of VALIS
</div>

</div>

These are the measurements of the paper on eight cases; [The paper](paper.md) gives the full table
and what each number means. STARE needs a nuclear stain shared by every round, which is why VALIS
stays available.

## Find your way

<div class="grid cards" markdown>

-   :material-play-circle-outline:{ .lg .middle } **Run your own slides**

    ---

    Install, describe your slides, run, open the results.

    [:octicons-arrow-right-24: Get started](quickstart.md)

-   :material-console:{ .lg .middle } **Running MIRAGE**

    ---

    The samplesheet, the four stages, checkpoints, clusters, troubleshooting.

    [:octicons-arrow-right-24: Guide](usage.md)

-   :material-file-tree:{ .lg .middle } **Inputs & outputs**

    ---

    Every file a run reads and writes, and how the measurement columns are named.

    [:octicons-arrow-right-24: Outputs](outputs.md)

-   :material-check-decagram-outline:{ .lg .middle } **Checking registration**

    ---

    The overlays and the per-round scores a run reports about its own alignment.

    [:octicons-arrow-right-24: Registration QC](registration_qc.md)

-   :material-tune:{ .lg .middle } **Parameters**

    ---

    Every setting, grouped by stage, with its default.

    [:octicons-arrow-right-24: Reference](parameters.md)

-   :material-sitemap:{ .lg .middle } **The pipeline**

    ---

    Every process in execution order, with six schematics of the hard parts.

    [:octicons-arrow-right-24: How it works](pipeline.md)

</div>

!!! note "When the docs and the code disagree"
    The code wins:
    [`nextflow.config`](https://github.com/sceriff0/mirage/blob/main/nextflow.config) holds every
    default. Please [open an issue](https://github.com/sceriff0/mirage/issues) if you find a
    mismatch.

<!-- Provenance of the three figures above: manuscript main text, Results and Supplementary
     Table 2, read in full on 2026-11-13 [FULL]. -->

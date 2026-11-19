# Choosing methods

<p class="standfirst">The defaults are the configuration of the paper: STARE registration, StarDist
segmentation, illumination correction off. This page says what each alternative changes and when
to pick it. Every setting it names is listed with its default on the Parameters page.</p>

## At a glance

| Step | Default | Alternatives | Switch with |
|---|---|---|---|
| Illumination correction | **off** | BaSiC flatfield correction | `skip_preprocessing` (boolean, in a `-params-file`) |
| Registration | **STARE** (`tiled`) | VALIS 1.2.0 | `--registration_method valis` |
| Segmentation | **StarDist** | InstanSeg, CellSAM | `--seg_method instantseg` / `cellsam` |

## Registration: STARE or VALIS

Registration aligns every round of a patient onto its reference round, on the nuclear
stain.

**STARE** (`--registration_method tiled`, the default) measures local displacements tile
by tile and fits them into one continuous displacement field. Each task reads a fixed
window, so its memory does not grow with the slide. It is JVM-free, runs fully in
parallel and needs no GPU.

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

These are the measurements of the paper on eight cases; [The paper](paper.md) gives the
full table and what each number means.

<!-- Provenance of the three figures above: manuscript main text, Results and Supplementary
     Table 2, read in full on 2026-11-13 [FULL]. -->

**VALIS** (`--registration_method valis`) runs valis-wsi 1.2.0 at its defaults, with no
registration settings of its own. It is kept because **STARE needs a nuclear stain shared
by every round**: choose VALIS when your rounds do not all carry one. VALIS's memory is
not bounded by a parameter: its first attempt asks 64 GB (sized from the paper cohort's
peaks) and each retry doubles that — see [Resources](resources.md). `reg_valis_mode hires` switches it to VALIS's own
high-resolution example; see [Parameters → VALIS](parameters.md#valis).

| | STARE (`tiled`) | VALIS (`valis`) |
|---|---|---|
| Needs a nuclear stain in every round | yes | no |
| Memory per task | bounded by a fixed window (2–4 GB requested at the defaults) | not bounded by a parameter (64 GB requested on the first attempt) |
| GPU | never | never |
| Works with `--mode add_cycle` | no | yes |
| Tuning knobs | [Tiled / STARE](parameters.md#tiled-stare-registration_methodtiled) | none (`reg_valis_mode` only) |

## Segmentation: StarDist, InstanSeg or CellSAM

One process, `SEGMENT`, dispatches to three genuinely different segmenters via
`--seg_method`. They are **not** drop-in equivalents: they read different things,
build the whole-cell mask differently, and have different prerequisites.

| | `instantseg` | `stardist` **(default)** | `cellsam` |
|---|---|---|---|
| **Reads** | The full multichannel image — **channel-invariant** | One channel, hardcoded to **index 0** | One nuclear channel, **resolved by marker name** |
| **Whole-cell mask** | Predicted natively, alongside nuclei | Expanded from nuclei by `seg_expand_distance` | Expanded from nuclei by `seg_expand_distance` |
| **Precondition** | A writable model cache | Channel 0 **must** be a nuclear marker — otherwise the task exits 1 | A nuclear channel must exist — a negative index exits 1 |
| **Weights** | Apache-2.0, auto-downloaded from BioImage.IO | Trained model [`DSUnit/stardist`](https://huggingface.co/DSUnit/stardist), auto-downloaded at a pinned commit | **Gated** — needs `DEEPCELL_ACCESS_TOKEN` |
| **Runs on a fresh clone?** | **Yes** | **Yes** (needs network on first use, or a pre-downloaded copy in `segmentation_model_dir`) | No — set `cellsam_model_path` or export a token |
| **Container image** | `bolt3x/mirage-instanseg:1.0.0` | `bolt3x/mirage-stardist:1.0.0` | `bolt3x/mirage-cellsam:1.0.0` |
| **Entry point** | `segment_instantseg.py` | `segment.py` | `segment_cellsam.py` |

**Why StarDist is the default.** It runs the trained nuclear model used for the paper's
cohort, published at [huggingface.co/DSUnit/stardist](https://huggingface.co/DSUnit/stardist)
and pinned to one commit, so every run loads the same weights. `segment.py` downloads its
three files (`config.json`, `thresholds.json`, `weights_best.h5`) on first use; on a
cluster without network access, pre-download them into
`<segmentation_model_dir>/DSUnit__stardist__68b94406104b/` and they are used offline. A
branch name instead of a commit is refused.

**When to pick InstanSeg.** It is the only backend that predicts the whole-cell mask
directly rather than dilating nuclei by a fixed radius — so **`seg_expand_distance` has no
effect when InstanSeg is selected**. It needs no trained model of yours; tune
`seg_instantseg_batch_size` and `seg_instantseg_tile_size` to the GPU's memory.

**When to pick CellSAM.** It finds the nuclear channel by name rather than by position.
Its weights are gated: export `DEEPCELL_ACCESS_TOKEN` before launching, or pre-download
the weights and set `cellsam_model_path` (the only route on compute nodes without
internet).

All three write both a `*_cell_mask.tif` and a `*_nuclei_mask.tif`, so everything
downstream is backend-agnostic. An unrecognised `--seg_method` is rejected by name at
launch rather than silently running a different segmenter. The backend-specific
parameters are in [Parameters → Segmentation](parameters.md#segmentation).

## Illumination correction: on or off

Illumination correction is **off by default** (`skip_preprocessing = true`). Conversion
to OME-TIFF always runs — everything downstream assumes that layout — so turning the
correction off removes only the BaSiC step.

To turn it on, set `"skip_preprocessing": false` in a `-params-file`
(`params/full_pipeline.json` does this). It is a boolean, so it cannot go on the command
line. When on:

- BaSiC runs at nf-core BASICPY's upstream defaults: only the multiplicative flatfield is
  estimated and divided out; no darkfield is estimated.
- The nuclear/fiducial channels are left **uncorrected** (`preproc_skip_nuclear = true`).
  They drive both registration and segmentation, so correcting them would change what
  both consume.
- The run pulls one more container image (BASICPY's), and the corrected slides are
  published under `<patient_id>/preprocessed/` instead of `converted/`.

How the correction is split into three processes, and why, is in
[Illumination correction](basic_illumination.md).

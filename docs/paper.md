# The paper

<p class="standfirst">MIRAGE, the STARE registration method it runs by default, and the FlowPath
gating extension are described together in one manuscript. This page states what that manuscript
reports and where each part lives in the software, so a reader of either can find the other.</p>

**STARE enables whole-slide multiplexed immunofluorescence on conventional microscopes by
bounded-memory non-rigid registration.** V. Fassi, C. Borriero, A. Dondi, A. Chiodin,
M. Compagnoni, G. Frigè, S. Chiocca, P. Bossi, L. Mazzarella, M. Barberis, S. Rodighiero and
Y. Zhan. Manuscript in preparation.

## What it describes

| Part | What it is | Where |
|---|---|---|
| **The protocol** | Iterative staining, imaging and stripping on a wide-field microscope: two antibodies plus DAPI per round, nine rounds, 18 markers on one section | the manuscript |
| **STARE** | Tile-based non-rigid registration of separately acquired rounds, with the memory of a task set by a fixed window | [github.com/sceriff0/stare](https://github.com/sceriff0/stare), run here as `--registration_method tiled` |
| **MIRAGE** | The pipeline from raw images to single-cell tables | this repository |
| **FlowPath** | A QuPath extension that gates the exported cells down a declared population hierarchy | [github.com/sceriff0/flowpath](https://github.com/sceriff0/flowpath), [documentation](https://flowpath.readthedocs.io/) |

## The configuration of the paper is the default

A run with no settings changed uses the configuration the cohort was analysed with:

| Setting | Value | Parameter |
|---|---|---|
| Registration | STARE (`stare-registration` 1.3.0) at its defaults: 2,048 px tile core, 64 px stride | `registration_method = tiled` |
| Second registration mode | VALIS (`valis-wsi` 1.2.0) at its defaults | `registration_method = valis` |
| Segmentation | StarDist with the trained model at `huggingface.co/DSUnit/stardist`, pinned to one commit; cells are nuclei grown by 10 px | `seg_method = stardist` |
| Illumination correction | not applied | `skip_preprocessing = true` |
| Quantification | median per marker in nucleus, cytoplasm and whole cell | `quantify_compartments = true` |

One setting of the cohort is not a default: the pixel size. MIRAGE reads it from each file
(`pixel_size = auto`); the cohort's was 0.325 µm.

## What was measured

All five methods registered the same converted rounds of eight cases, and were scored on the same
64 round pairs by how well the same nuclei line up: the Dice coefficient of matched nuclei
(expanded by 10 px), and the distance left between their centroids.

| Method | Matched-pair nuclear Dice, median (IQR) | Residual centroid distance, µm, median (IQR) | Peak memory per case, GB, median (range) | CPU time per case, h, median (range) |
|---|---|---|---|---|
| Unregistered | 0.51 (0.49–0.51) | 3.65 (3.43–3.75) | – | – |
| **STARE** 1.3.0, defaults | **0.84** (0.82–0.86) | **0.63** (0.49–0.73) | **0.7** (0.68–1.2) | 3.3 (0.6–7.5) |
| VALIS 1.2.0, default `register()` | 0.80 (0.76–0.83) | 0.96 (0.83–1.28) | 31.6 (17.2–51.4) | 1.4 (0.4–1.8) |
| PALOM 2026.4.1 | 0.80 (0.78–0.82) | 1.06 (0.85–1.20) | 5.4 (1.8–7.7) | 0.4 (0.1–0.5) |
| DeeperHistReg 1.0.1, high-resolution preset, CPU | 0.77 (0.71–0.79) | 1.33 (1.20–1.84) | 13.5 (5.0–20.2) | 36.4 (6.2–58.1) |
| ASHLAR 1.20.0, as shipped | 0.76 (0.69–0.80) | 1.30 (1.08–1.94) | 6.6 (1.9–9.5) | 0.4 (0.1–0.6) |

How to read it:

- **Accuracy.** STARE was higher in Dice and lower in centroid distance than every other method
  in all eight cases.
- **Memory.** The STARE figure is its largest registration job, with the image writer excluded as
  for every method; its resampling job, which writes the image, peaked at up to 2.4 GB. No STARE
  task needed a GPU.
- **The price.** STARE used about 2.8 times the CPU time of VALIS. Its tile tasks are independent,
  so on a cluster that time is spent in parallel.
- **Scope.** The memory bound is a property of registration. Segmentation and quantification
  need memory in proportion to the slide; see [Resources](resources.md).
- **When to use VALIS.** STARE aligns on a nuclear stain present in every round. Without one, use
  `--registration_method valis`.

## Where to find things

| In the manuscript | In the software |
|---|---|
| Fig. 2a, the four STARE stages; Supplementary Note 3.2 | [The pipeline](pipeline.md), [Schematic 2 · registration](figures/registration-schematic.html){ target=_blank }, [Schematic 5 · coarse alignment](figures/coarse-schematic.html){ target=_blank } |
| Supplementary Table 1, STARE parameters | [Parameters → Tiled / STARE](parameters.md#tiled-stare-registration_methodtiled) |
| Fig. 4a, the four subworkflows | [The pipeline](pipeline.md), [Schematic 1 · pipeline](figures/pipeline-schematic.html){ target=_blank } |
| Supplementary Note 2, resource requests and retries | [Resources](resources.md) |
| Methods, image conversion and illumination correction | [Illumination correction](basic_illumination.md), [Format validation](validation/format_validation.md) |
| Methods, segmentation; Supplementary Note 4 | [Parameters → Segmentation](parameters.md#segmentation) |
| Methods, quantification and export; Supplementary Table 4 | [Inputs & outputs](outputs.md), [SpatialData & FlowPath](spatialdata.md) |
| The scores a run reports about its own registration | [Checking registration](registration_qc.md) |
| Registration benchmark, Fig. 3 and Supplementary Table 2 | the `benchmarks/regbench/` harness on the [`benchmarking` branch](https://github.com/sceriff0/mirage/tree/benchmarking) |

The six schematics on this site are explanatory figures of the software. They are not the
figures of the manuscript and are numbered separately.

## Versions

| Component | Version |
|---|---|
| MIRAGE | pipeline manifest 1.0.0, Nextflow 25.04.0 or newer |
| STARE | `stare-registration` 1.3.0, image `bolt3x/mirage-stare:1.0.0` |
| VALIS | `valis-wsi` 1.2.0, image `bolt3x/mirage-valis:1.0.0` |
| StarDist model | `huggingface.co/DSUnit/stardist`, commit `68b94406104bb069a58b64f4c890375c7f2f15df` |
| FlowPath | for QuPath 0.7.0 |

To cite the software, see [Citation](citation.md).

<!-- Provenance: every number and statement on this page was transcribed from the manuscript
     draft dated 2026-11-06 (main text: abstract, Results, Methods, Code availability;
     supplement: Notes 2, 3, 6, 7 and Supplementary Tables 1, 2, 4), read in full on
     2026-11-13 [FULL]. Re-check this page against the manuscript whenever the manuscript
     changes. -->

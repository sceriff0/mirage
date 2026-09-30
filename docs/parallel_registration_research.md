# Parallel registration — research dossier

Raw material for designing a **novel, fully-parallel WSI registration method** targeting
laptops / low-memory machines, inspired by ASHLAR's spanning-tree stitching and VALIS's
serial-along-an-ordering rigid registration, and constrained to remain drop-in compatible with
mirage's `reg_qc = 2` QC contract.

This document **gathers facts only** — it does not propose the new algorithm. Every non-obvious
claim is cited: `path:line` for this codebase, URLs for external sources. Read the "open
questions" at the end for the handoff to the designer.

Primary external sources:
- ASHLAR paper: Muhlich et al., *Bioinformatics* 38(19):4613–4621 (2022), DOI
  [10.1093/bioinformatics/btac544](https://academic.oup.com/bioinformatics/article/38/19/4613/6668278);
  source: [github.com/labsyspharm/ashlar](https://github.com/labsyspharm/ashlar).
- VALIS paper: Gatenbee et al., *Nature Communications* 14:4502 (2023), DOI
  [10.1038/s41467-023-40218-9](https://www.nature.com/articles/s41467-023-40218-9);
  source: [github.com/MathOnco/valis](https://github.com/MathOnco/valis).

---

## 1. How mirage registration works today

### 1.1 The DAG (subworkflow `REGISTRATION`)

Entry point `subworkflows/local/registration.nf`. Steps, in order:

1. **Images enter as-is** — no padding step. Both backends align inputs of differing sizes natively
   (VALIS resolves them into a shared space; the tiled/STARE backend warps each moving slide into the
   reference's shape), so no common-canvas step is needed.
2. **Group by patient + identify reference** (`registration.nf:106-143`). Slides are grouped with
   a streaming `groupTuple(size: images_count)` keyed on `patient_id`; the reference is the item
   whose `meta.is_reference == true`. Missing reference → hard error unless
   `params.allow_auto_reference` (then first slide is promoted, `registration.nf:117-140`).
3. **Single-slide short-circuit** (`registration.nf:150-155`): a patient with one slide bypasses
   VALIS entirely (VALIS crashes on a lone image) — the reference *is* the registered output.
4. **Registration via the VALIS adapter** — `VALIS_ADAPTER(ch_grouped_multi)`
   (`registration.nf:182-188`). The adapter (`subworkflows/local/adapters/valis_adapter.nf`)
   reshapes patient-grouped data into one `REGISTER` invocation per patient (all slides at once)
   and matches registered outputs back to metas by **OME channel signature** read from a
   `channels_manifest.json` (`lib/RegisteredMatch.groovy`) — no filename parsing.
5. **QC generation** (see §2), **checkpoint CSV** (`registered.csv`, in
   `registration.nf`).

**Key structural fact for the redesign:** registration today is a **single monolithic
per-patient process** (`REGISTER`). All slides of a patient go into one container, one JVM, one
`valis.registration.Valis` object. There is **no tile-level or pair-level parallelism inside a
patient** — the only parallelism is *across* patients (Nextflow runs one `REGISTER` task per
patient concurrently). The distributed/tiled low-memory path was **archived 2026-07-24** (git tag
`archive/tiled-valis-2026-07-24`) and is no longer wired in (`registration.nf:166-171`).

### 1.2 The `REGISTER` process and resource labels

`modules/local/register.nf`:
- `label 'process_high'`, container `cdgatenbee/valis-wsi:1.0.0` (`register.nf:15-19`).
- `process_high` = **8 CPUs, `200.GB + 100.GB * task.attempt`, 12 h** (`conf/modules.config:48-52`).
  So attempt 1 already asks **300 GB RAM**; attempt 2 → 400 GB, attempt 3 → 500 GB. This is a
  cluster-only footprint — **it cannot run on a laptop today**. That gap is the entire motivation
  for the new method.
- JVM heap is sized separately and scales with the retry ramp: `min(reg_jvm_heap_gb ?? (32 + 16*attempt),
  task.memory - 4)` GB (`register.nf:56-57`).
- Inputs: `tuple(meta, patient_id, path(reference, stageAs:'ref/*'), path(preproc_files,
  stageAs:'input_?/*'), val(all_metas))` (`register.nf:24`). Reference is staged both separately
  (for `--reference`) and inside the input set.
- Outputs (`register.nf:26-35`): `registered_slides/*_registered.ome.tiff` + `channels_manifest.json`;
  `versions.yml`; `*.size.csv`; **`registrar.pickle`** (optional, for seg-QC); **`reg_stage_checkpoint/`**
  (optional, only at `reg_qc >= 2`).

Other registration-subworkflow processes and their labels:

| Process | Label | Resources (`conf/modules.config`) | Role |
|---|---|---|---|
| `REGISTER` | `process_high` | 8 CPU / 200+100·att GB / 12 h | VALIS rigid+non-rigid+micro, all slides/patient |
| `SEG_QC_GEOJSON` | `process_high` + `gpu` container | 8 CPU / 200+100·att GB | StarDist DAPI seg on native slide → cell GeoJSON (reg_qc=2) |
| `WARP_SEG_QC` | `process_medium` | 4 CPU / 100+100·att GB / 4 h | warp polygons through stages, score overlap (reg_qc=2) |
| `GENERATE_REGISTRATION_QC` | `process_high` | 8 CPU / 200+100·att GB | RGB before/after DAPI overlay (reg_qc≥1) |

There is no `process_high_memory` label — `conf/modules.config` defines only `process_single`,
`process_low`, `process_medium`, and `process_high` (8 CPU / 200+100·att GB / 12 h, the ceiling
used by the registration processes above).

### 1.3 What VALIS features mirage actually uses

Invoked from `bin/register.py` via `valis.registration.Valis` (`register.py:70, 442`). The
registrar kwargs are the single source of truth in `bin/utils/valis_config.py`
(`build_registrar_kwargs`, `valis_config.py:44-74`):

- **Rigid + non-rigid + micro**, all three stages. `registrar.register()` does rigid + non-rigid
  (`register.py:452`); `registrar.register_micro(...)` does the micro-registration residual
  (`register.py:705-710`), which is **caught-and-continued** — a micro failure never fails the run
  (`register.py:716-719`).
- **Reference selection**: mirage always passes an explicit `--reference <filename>`
  (`register.py:367-376`, `align_to_reference=True`, `crop='reference'` in `valis_config.py:58-60`).
  So mirage **overrides VALIS's automatic reference/ordering** — the reference is chosen upstream
  from the CSV `is_reference` flag, not by VALIS graph centrality.
- **Feature detector / matcher** are pinned by memory preset (`valis_config.py:17-41`): all presets
  use `feature_detectors.SuperPointFD` + `feature_matcher.SuperGlueMatcher()`, `num_features=5000`.
  `high`: processed dim 2048 / non-rigid 4096; `medium`: 1024 / 4096; `low`: 256 / 1024 + tiling.
  (Note: upstream VALIS `main` has since moved its *defaults* to `VggFD` + `LightGlueMatcher` —
  mirage's pinned `valis-wsi:1.0.0` container predates that and mirage forces SuperPoint/SuperGlue
  regardless.)
- Non-rigid registrar: `OpticalFlowWarper`; VALIS auto-switches to `NonRigidTileRegistrar` when its
  own estimate exceeds ~10 GB (`register.py:414-423`). `affine_optimizer_cls=None` (SimpleElastix
  not available, `register.py:426-431`). `micro_rigid_registrar_cls = MicroRigidRegistrar`
  (`valis_config.py:70-72`).
- **Warping** is sequential per slide via `slide_obj.warp_and_save_slide(..., non_rigid=use_non_rigid,
  crop=True)` (`register.py:844-890`).

### 1.4 JVM / memory workarounds already in the code (pain points to inherit or avoid)

- BioFormats needs a JVM; **once killed it cannot restart in the same process** — VALIS's own error
  handlers call `kill_jvm()`, which then blocks warping. `bin/register.py` explicitly checks
  `jpype.isJVMStarted()` before warping and fails loudly (`register.py:779-812`).
- JVM heap heuristic `total_input_size*3 + 8`, capped at 75% system RAM (`register.py:135-143`,
  `valis_config.py:104-111`). Memory note *mirage-jvm-heap-undersized-references* records this
  heuristic OOMs on high-LZW multi-channel references — the fix scales heap with the retry ramp
  (`register.nf:56-57`).
- `scyjava<1.11` derives the jgo/Maven cache from `Path.home()` and ignores env knobs; on a
  read-only `$HOME` the JVM crashes with EROFS. Worked around by `point_jvm_cache_off_readonly_home()`
  before JVM start (`valis_config.py:92-99`). matplotlib/XDG caches likewise redirected to `/tmp`
  (`register.py:64-67`).
- `NonRigidTileRegistrar` has a `fwd_dxdy`-is-`pyvips.Image` bug that silently leaves `fwd_dxdy=None`;
  mirage repairs it from `bk_dxdy` inverse after `register()` (`register.py:456-478`).

These are all **BioFormats/JVM/pyvips-heavyweight** problems. A laptop-friendly method that avoids
the JVM (pure Python/numpy/skimage I/O) sidesteps most of this class of bug — but then loses
VALIS's pyramidal OME-TIFF reader and must solve large-image I/O itself.

### 1.5 The meta-map channel contract

CSV → meta parsing in `CsvUtils.parseMetadata` (`lib/CsvUtils.groovy:237-250`). A meta map carries:
- `patient_id` (grouping key)
- `is_reference` — strict boolean, parsed by `parseIsReference`, only `true`/`false` accepted
  (`CsvUtils.groovy:230-235`). **Exactly one** reference per patient is enforced
  (`validateInputSemantics`, `CsvUtils.groovy:334-340`); zero allowed only with
  `allow_auto_reference`.
- `channels` — non-empty `|`-split list; **a nuclear marker must be present** (at any position),
  located by name not index. Which names qualify comes from `params.nuclear_markers`
  (`nextflow.config`, default `['DAPI', 'CELLTOX']`) via `MarkerUtils.hasNuclear`, not a
  hardcoded `DAPI` (`lib/CsvUtils.groovy:218`).
- `id` — per-image unique id `patient_id + source-stem` (`subworkflows/local/input_check.nf:50-54`).
- `images_count` / `channels_count` — pre-computed from the CSV
  (`CsvUtils.countImagesPerPatient` / `countChannelsPerPatient`, `CsvUtils.groovy:54-152`) and
  injected into meta so `groupTuple(by:, size:)` can **stream** (emit as soon as a patient's slides
  arrive, without buffering the whole run) — `subworkflows/local/input_check.nf:58-86`.

**All channels carry `[meta, file]` tuples.** Reference vs moving is distinguished **only** by
`meta.is_reference`; there is no separate `ref`/`mov`/`is_ref` field. The registration subworkflow
branches on it repeatedly (`registration.nf:117, 202-205, 235-238, 280, 291`). Any replacement
process must consume `[meta, files]` in and emit `[meta, registered_file]` out, preserving
`patient_id`, `is_reference`, `channels`, and the OME channel names in the output OME-XML (the
adapter matches outputs back by channel signature, `lib/RegisteredMatch.groovy`).

---

## 2. What `reg_qc = 2` requires (the QC contract the new method must reproduce)

`reg_qc` levels (`lib/ParamUtils.groovy:60, 82`; default **2** — `nextflow.config:66`):
`0` = none, `1` = DAPI overlay image, `2` = `1` **plus** staged segmentation-overlap QC.
`skip_registration_qc=true` forces 0. The single definition is `ParamUtils.regQcLevel`.

### 2.1 Level ≥ 1 — DAPI RGB overlay
`GENERATE_REGISTRATION_QC` (`modules/local/generate_registration_qc.nf`) runs
`bin/generate_registration_qc.py` for every non-reference slide vs its patient reference. Outputs
per slide: `{basename}_QC_RGB_fullres.tif`, `{basename}_QC_RGB.tif`, `{basename}_QC_RGB.png`
(registered→red, reference→green, aligned=yellow). Downsample factor `params.qc_scale_factor`
(default 0.25). This only needs the **registered pixels** — any method that produces registered
OME-TIFFs satisfies level 1 for free.

### 2.2 Level = 2 — staged, fixed-correspondence segmentation-overlap QC
This is the hard contract. Full design rationale in `docs/registration_qc.md`. Two processes:

- **`SEG_QC_GEOJSON`** (`modules/local/seg_qc_geojson.nf`) — segments each slide's DAPI on its
  **native (pre-registration)** image with StarDist and writes a geometry-only cell GeoJSON. The
  stem is VALIS's `valtils.get_name` convention so it equals the registrar `slide_dict` key.
- **`WARP_SEG_QC`** (`modules/local/warp_seg_qc.nf`, `bin/warp_seg_qc.py`) — loads the **VALIS
  registrar pickle** and warps the reference + moving native GeoJSONs through each stage, scores
  per-pair overlap.

**Critical dependency — this is what makes reg_qc=2 VALIS-specific today.** The QC scores four
transform states: `native`, `rigid`, `non_rigid`, `micro` (`docs/registration_qc.md:12-19`). VALIS
composes non-rigid and micro **destructively** into one displacement field, so the intermediate
`non_rigid` state exists only for an instant inside `register.py`. `REGISTER` therefore snapshots
each slide's forward displacement field **after `register()`, before `register_micro()`** into
`reg_stage_checkpoint/` (`register.py:648-680`; rationale `docs/registration_qc.md:60-79`). Without
that checkpoint the QC still runs but reports only `native/rigid/micro`, sets
`stages_separable: false`, and records a `note` (`docs/registration_qc.md:74-79`).

Correspondence is established **once** at the `rigid` anchor stage — **mutual nearest centroid
within one median nuclear radius** — then held fixed, so every inter-stage delta is pure geometry
(`docs/registration_qc.md:10-40`). Tunable via `--match-radius-factor` / `--match-radius-px`.

**The output artifact (this is the contract a replacement must emit):**
`<outdir>/<patient>/qc/registration/<patient>_<slide>_seg_qc.json` with, per stage over the fixed
pairs (`bin/warp_seg_qc.py:89-95, 270-347`; example `docs/registration_qc.md:82-109`):
- `iou_mean`, `iou_p10/p50/p90`, `iou_max`, `frac_iou_ge_0.5`
- `displacement_px_p50/p90/max` (+ `displacement_um_*` when pixel size known) — the headline number
- `dice_matched` (Dice restricted to matched pairs)
- top level: `stages_separable`, `stage_order`, `matching{method:"mutual_nn_centroid",
  anchor_stage:"rigid", radius_px, median_cell_radius_px, n_pairs, pair_fraction,
  pair_fraction_ref, pair_fraction_moving}`, `delta_vs_anchor{stage: {...}}`, `counts{features_ref,
  features_moving}`, `n_pairs_scored`.
- `delta_vs_anchor` = each stage minus `rigid`; **positive displacement delta ⇒ that stage made
  alignment worse** — the failure this design exists to surface.

**Design-critical consequence:** the new method must expose enough transform structure to
reproduce these staged deltas. A monolithic single-transform method would only ever report
`native` vs one final stage (`stages_separable: false`). To keep `reg_qc=2` meaningful, the new
method should either (a) expose named intermediate transform stages that WARP_SEG_QC can warp
polygons through, or (b) ship an equivalent per-pair displacement/IoU scorer that consumes the new
method's transform representation instead of a VALIS pickle. Note WARP_SEG_QC currently **hard-codes
the VALIS container and loads a VALIS `registrar.pickle`** (`warp_seg_qc.nf:27-30`,
`warp_seg_qc.py` header) — this coupling is the single biggest QC-compatibility obstacle.

---

## 3. ASHLAR's graph / minimum-spanning-tree approach (primary source: `ashlar/reg.py`)

ASHLAR separates two problems: **intra-cycle mosaic stitching** (`EdgeAligner`) and **inter-cycle
registration** (`LayerAligner`). Graph library throughout: **NetworkX** (`import networkx as nx`).

### 3.1 Pairwise primitive — phase correlation (`ashlar/utils.py`, `register()` ≈ lines 35–57)
1. **Whiten** both overlap patches — Laplacian-of-Gaussian high-pass (`whiten()`,
   `scipy.ndimage.gaussian_laplace`), suppresses low-frequency illumination differences.
2. **Window** with a 2D Hann window (`window()`) to kill FFT edge artifacts.
3. **`skimage.registration.phase_cross_correlation(..., upsample_factor)`** for sub-pixel
   translation.
4. **Four-quadrant disambiguation** of the FFT periodicity aliasing — test all four candidate
   shifts, pick the one maximizing direct correlation.
5. Returns `(shift, error)` where `error = -log(correlation / total_amplitude)`, else `inf`.
   `nccw()` computes the same error metric without re-aligning.

This is a **CPU-only, small-memory, embarrassingly-parallel** kernel — it operates on overlap
windows, never whole slides.

### 3.2 EdgeAligner — intra-cycle mosaic (`ashlar/reg.py`)
- **Neighbor graph** (`neighbors_graph`, ≈520–540): `scipy.spatial.distance.pdist(positions,
  'cityblock')`; tiles closer than `tile_size.max()+1` become edges; `nx.from_edgelist` →
  `nx.Graph`. Only *overlapping* tile pairs are registered — not all pairs.
- **Per-edge registration** (`register_pair(t1,t2)` → `_register()`, ≈632–670): calls
  `utils.register` on the overlap, applies a padding/direction correction. **Embarrassingly
  parallel across edges.**
- **Error threshold via a permutation / null distribution** (`compute_threshold()`, ≈577–630):
  registers ~1000 **known-non-overlapping** random tile-strip pairs to build a null error
  distribution, takes the α-percentile (default α=0.01) as `max_error`. Edges above threshold are
  distrusted. This is ASHLAR's statistically-principled "is this shift real or noise?" test —
  **directly reusable and itself parallel.**
- **Global position solve** (**serial reduction**):
  - `build_spanning_tree()` (≈671–713): builds a graph weighted by per-edge error and extracts a
    **minimum spanning tree** (via `nx.single_source_dijkstra_path` / shortest-path from component
    centers) — the tree keeps only the most-confident edges and avoids accumulating error through
    weak links.
  - `calculate_positions()`: propagates shifts along the spanning tree from a root tile, accumulating
    absolute tile positions (error propagation along the tree).
  - `fit_model()`: fits `sklearn.linear_model.LinearRegression` on the largest connected component to
    regularize positions and place disconnected components.

### 3.3 LayerAligner — inter-cycle (cycle N → reference cycle) (`ashlar/reg.py`)
- **Coarse global offset** (`coarse_align()`, ≈747–759): `thumbnail.calculate_cycle_offset()`
  estimates one whole-cycle shift from downsampled thumbnails.
- **Per-tile fine registration** (`register_all()` / `register(t)`, ≈761–774): each reference-cycle
  tile is matched to the overlapping region of the moving cycle via `utils.register`. **Parallel per
  tile.**
- **Constrain / outlier rejection** (`constrain_positions()`, ≈776–819): drops zero-shift
  auto-correlation artifacts and shifts exceeding `max_shift_pixels`; fills discarded tiles from the
  reference cycle's fitted linear model.
- **Mosaic assembly** (`Mosaic.assemble_channel()`, ≈828–902): `utils.paste` each raw tile at its
  solved position with `pastefunc_blend`. Tile reads/pastes are independent (loop is sequential but
  trivially parallelizable / streamable).

**ASHLAR's decomposition maps almost 1:1 onto the mirage cyclic-IF problem** (multi-cycle
registration to a fixed reference), and its per-edge/per-tile phase correlation + MST solve is the
canonical "parallel pairwise, serial global" pattern.

---

## 4. VALIS's ordering / tree approach (primary source: `valis/serial_rigid.py`, `registration.py`)

**Important correction to the common assumption:** VALIS does **not** use NetworkX or a graph
minimum-spanning-tree for image ordering. It uses **hierarchical clustering with optimal leaf
ordering** to lay the images out in a 1-D chain, then registers serially along that chain.

Class **`SerialRigidRegistrar`** (`serial_rigid.py`, ≈427–1273). Orchestrated by top-level
`register_images()` (≈1275–1496). Pipeline:

1. **Feature detection** on every image — `generate_img_obj_list()` (≈470–511). Default detector in
   pinned mirage build: SuperPoint (`SuperPointFD`); current upstream default `VggFD`.
2. **All-pairs feature matching** — `match_imgs()` (≈519–545), parallelized across pairs with
   `pqdm`. (If images are pre-sorted, `match_sorted_imgs()` matches only adjacent pairs, ≈487–517.)
   **This all-pairs matching is the embarrassingly-parallel core**, analogous to ASHLAR's per-edge
   registration.
3. **Similarity → distance matrix** — `build_metric_matrix(metric="n_matches")` (≈911–967).
   Default similarity is the **number of matched features** between a pair
   (`DEFAULT_SIMILARITY_METRIC = "n_matches"`, `registration.py` ≈line 105); can also use a
   `scipy.spatial.distance.cdist` metric on feature descriptors.
4. **Ordering** — `sort()` (≈969–987) → `order_Dmat(D)` (≈161–189): `fastcluster.linkage(...,
   'single')` + `scipy.cluster.hierarchy.optimal_leaf_ordering()`. Produces a 1-D ordering of
   images by similarity (adjacent = most similar).
5. **Reference + iteration order** — `get_iter_order()` (≈989–1011):
   `ref_img_idx = warp_tools.get_ref_img_idx(img_f_list, ref_img_name)` (user name lookup, else a
   default), then `iter_order = warp_tools.get_alignment_indices(size, ref_img_idx)` — a serial
   alignment chain **radiating from the reference** through the sorted order. `iter_order` is a list
   of `(from_idx, to_idx)` pairs (`registration.py` docstring ≈1268-1270).
6. **Serial rigid transforms** — `align_to_prev()` (≈1088–1130) estimates each image's rigid
   transform relative to its neighbor in the chain, composing toward the reference. Optional global
   optimization step in `register_images()`.

**Then, in `registration.py`** (`Valis.register()`), rigid is followed by **non-rigid** (default
`OpticalFlowWarper` / tiled variant) and optional **micro-registration** (`register_micro()`).
mirage bypasses steps 3–5's automatic reference choice by always passing `reference_img_f` +
`align_to_reference=True` (§1.3), so in mirage VALIS effectively registers each moving slide **to
the fixed reference** rather than freely along the similarity chain.

**Contrast with ASHLAR:** VALIS orders *whole images* by feature-match count and registers along a
serial chain (clustering-based); ASHLAR orders *tiles* by physical adjacency and solves positions on
a NetworkX **minimum spanning tree** weighted by phase-correlation error. Both share the same
skeleton: **all-pairs (or all-neighbor) pairwise registration first, then a global ordering/solve.**

---

## 5. The serial-vs-parallel decomposition (the crux of "fully parallel")

Both algorithms factor into the **same two-phase shape**, and the split is the design lever:

| Phase | ASHLAR | VALIS | Cost profile | Parallelism |
|---|---|---|---|---|
| **A. Pairwise registration** | `register_pair` / `_register` per overlapping tile edge; per-tile in `LayerAligner` | `match_imgs` all-pairs feature matching | O(#pairs) independent tasks, each on a **small window / one image**, low RAM | **Embarrassingly parallel** — the whole win |
| **A′. Noise threshold** | `compute_threshold` permutation null (~1000 random non-overlap pairs) | (implicit in match filtering / RANSAC) | independent samples | **Parallel** |
| **B. Global position/order solve** | `build_spanning_tree` (MST) + `calculate_positions` + `fit_model` | `build_metric_matrix` + `sort` (hier. clustering) + `get_iter_order` + serial `align_to_prev` | O(N) tiles/images, tiny data (shifts + errors only) | **Serial reduction** — but cheap and low-memory |
| **C. Warp + write** | `Mosaic.assemble_channel` / `paste` per tile | `warp_and_save_slide` per slide | O(pixels), high I/O | **Parallel per tile/slide/channel**, streamable |

Insights for the designer:
- **Phase A is where the memory and compute cost lives, and it is fully parallel.** Splitting it
  into Nextflow tasks (or laptop worker processes) over image pairs / tile pairs turns the current
  monolithic 300 GB `REGISTER` into many small low-RAM tasks. A pair of downsampled overlap windows
  is megabytes, not gigabytes.
- **Phase B is a serial reduction but operates on tiny data** — a list of pairwise shifts + errors,
  never pixels. It is a cheap fan-in (like mirage's existing `MAX_DIM`). MST (ASHLAR) and
  clustering+chain (VALIS) are two concrete recipes; either could run in a single small process.
- **Phase C is parallel per output** and can stream (mirage already warps slides one-at-a-time and
  could fan them out).
- The current mirage DAG only parallelizes **across patients**; the untapped axis is **within a
  patient, across image/tile pairs** — exactly ASHLAR's model.
- **reg_qc=2 wrinkle:** the staged QC (§2) needs *named intermediate transforms* to score. A
  parallel pairwise + global-solve method naturally yields at least `rigid`-equivalent (global
  affine) and, if a non-rigid refinement is added, a `non_rigid`-equivalent stage — mapping onto the
  QC's `rigid`/`non_rigid`/`micro` stage order. The transform representation should be designed so a
  polygon can be warped through each named stage (replacing the VALIS-pickle dependency).

---

## 6. Lightweight / CPU-friendly building blocks for laptops

Cite-able primitives suitable for the parallel pairwise phase on low-memory machines:

- **Phase correlation** — `skimage.registration.phase_cross_correlation(reference, moving,
  upsample_factor=…)` returns `(shift, error, phasediff)`, sub-pixel via matrix-multiply DFT
  upsampling; O(n log n) FFT, memory ≈ a few × the overlap window.
  [skimage docs](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation).
  This is exactly ASHLAR's kernel (§3.1). Rigid-translation only; combine with the whiten+Hann
  pre-filter for illumination robustness.
- **Feature-based (rotation/scale/affine)** — OpenCV `ORB` (fast, binary, license-free) or `SIFT`,
  matched with `cv2.BFMatcher` / FLANN, then `cv2.estimateAffinePartial2D` / `findHomography` with
  RANSAC. ORB is the laptop-friendly default (no patent, low memory).
  [OpenCV feature docs](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html).
  This is the "VALIS-style" branch (feature matches also give the `n_matches` similarity for
  ordering).
- **Pyramidal / coarse-to-fine** — register downsampled thumbnails first for a coarse global offset
  (ASHLAR's `thumbnail.calculate_cycle_offset`, §3.3), then refine at higher resolution on small
  windows. Keeps peak memory at the coarse level; the fine level only ever loads overlap tiles.
  `skimage.transform.pyramid_gaussian` builds the pyramid.
- **Non-rigid (optional, if a `non_rigid` QC stage is wanted)** — optical flow
  (`skimage.registration.optical_flow_tvl1` / Farneback in OpenCV) on downsampled images produces a
  displacement field; this is the CPU analog of VALIS's `OpticalFlowWarper`. Memory scales with the
  (downsampled) field resolution, matching VALIS's `max_non_rigid_registration_dim_px` knob.
- **I/O without a JVM** — `tifffile` (already a mirage dependency, `register.py:39`) reads/writes
  pyramidal OME-TIFF and supports memory-mapped / per-page tile reads, avoiding the BioFormats JVM
  entirely and the whole class of JVM bugs in §1.4. Large slides can be processed tile-by-tile via
  `tifffile`'s page/tile access.

---

## 7. Open questions for the designer

1. **Tile grid vs whole-slide pairs.** mirage's inputs are already-stitched per-cycle OME-TIFFs
   (one image per slide/cycle), not raw tile grids like ASHLAR's microscope output. So the
   *intra-cycle* `EdgeAligner` MST does not directly apply — the parallel axis is **inter-cycle**
   (LayerAligner-style) plus optional **intra-image tiling** for memory. Should the new method tile
   each slide internally to parallelize a single pairwise registration, or parallelize only across
   the (usually few) cycles per patient? How many cycles/slides per patient are typical?
2. **reg_qc=2 transform contract.** Will the new method expose named `rigid`/`non_rigid`/`micro`-
   equivalent stages and a polygon-warp API, so `WARP_SEG_QC` can be rewritten to consume the new
   transform instead of a VALIS pickle? Or will `reg_qc=2` degrade to `stages_separable:false`
   (native vs final only) for the new method? The current `WARP_SEG_QC` hard-codes the VALIS
   container and pickle format (`warp_seg_qc.nf:27-30`).
3. **Reference handling.** mirage fixes the reference from CSV `is_reference` and overrides VALIS's
   automatic ordering. Does the new method still need any similarity ordering (VALIS-style) among
   moving slides, or does registering every moving slide directly to the fixed reference (star
   topology) suffice? Star topology is maximally parallel but can accumulate error when a moving
   slide overlaps the reference poorly.
4. **Global solve necessity.** With a fixed reference and star topology, is a global MST/least-
   squares solve (Phase B) even needed, or only when chaining through intermediate slides?
5. **Non-rigid on a laptop.** Is optical-flow non-rigid affordable at the memory budget, or should
   the laptop method be rigid/affine-only and mark `non_rigid`/`micro` as not-run in the QC (which
   the checkpoint mechanism already supports gracefully, §2.2)?
6. **Nextflow granularity.** Should Phase A become one process per image-pair (max parallelism, more
   scheduling overhead) or one process per moving slide (coarser)? How does this interact with the
   streaming `groupTuple(size: images_count)` pattern and the `[meta, file]` contract (§1.5)?
7. **Determinism / QC comparability.** VALIS results are deterministic (SuperPoint/SuperGlue
   inference). Phase-correlation + RANSAC introduces RNG (ASHLAR's permutation threshold, RANSAC).
   Does the QC comparison across methods need seed-pinning to be reproducible?
8. **Container / dependency story.** A JVM-free method (tifffile + skimage + opencv) is far lighter
   than `cdgatenbee/valis-wsi:1.0.0`, but then `SEG_QC_GEOJSON` (StarDist GPU) and any residual
   VALIS-based QC still pull heavy containers. Should the laptop path ship its own slim container and
   a CPU StarDist fallback?

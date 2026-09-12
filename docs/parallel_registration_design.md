# STARE — a fully-parallel, tiled registration method for mirage

> **Superseded in part.** This document records the design as it was implemented on
> `feat/tiled-registration`, when COARSE's anchor was a classical corner detector. For
> v1.0.0 that front-end was replaced by the learned DISK + LightGlue matcher and the three
> classical alternatives were deleted outright, which changes COARSE's memory model from
> nearly-flat to linear in thumbnail AREA.
>
> **What that invalidated, and has been corrected in place:** §5's memory table and
> paragraph, §5's COARSE row and primitive-split sentence, §1's constraint 2, and every
> "≤8 GB / laptop" claim that rested on the old front-end. STARE is **not** laptop-sized at
> the shipped `high` tier — COARSE alone asks ~48 GB. Anything in this file still phrased in
> the past tense about a corner detector (§5's OOM anecdote, the §"Implementation status"
> phase list, §11's sparse-tissue note) is a record of what the method USED to be and is
> left as written.
>
> **What is unchanged and still authoritative:** the M0 → per-tile → solve → stitch
> decomposition, the halo contract, the streaming reads, and everything downstream of
> COARSE. `bin/tiled_coarse.py` cites this file for the thumbnail rationale, which still
> holds — and holds harder now that the thumbnail bound is the memory knob.
>
> **Added 2026-09-12 — the SOLVE stage is no longer empty, and the method is a package.**
> §6b below describes the `robust` solver (`stare.solve`: neighbour-consistency rejection,
> in-fill, Tikhonov smoothing, invertibility check) that replaced "lay the translations on the
> grid and zero-fill", selectable against the byte-identical `legacy` path by
> `reg_tiled_solver`. §3's "no global solve" and §9.1's novelty claim are corrected in place.
> The four stages now live in `packages/stare/` (`pip install -e packages/stare`, CLI `stare`);
> `bin/tiled_*.py` are shims over it.

**Status:** implemented on branch `feat/tiled-registration` (Phases 1–2 + Nextflow wiring, 56
Python tests, JVM-free stub run green). Remaining: reg_qc=2 seg-QC Nextflow dispatch, the slim
container, real-data accuracy validation, and the optional per-tile Nextflow fan-out (§5). See the
status box after §10.
**Working name:** **STARE** — *STar-Anchored Registration with Error (TRE)*.
**Method id:** `registration_method = 'tiled'`.
**Companion:** `docs/parallel_registration_research.md` — primary-source notes on ASHLAR & VALIS.

> Naming note: an earlier draft of this file called the method "PARSEC" and built it around a
> spanning **tree**. That was a misread of the request — the ask was for VALIS-style **TRE**
> (*Target Registration Error*), not a *tree*. There are **no trees** in this design; a fixed
> reference removes the ordering problem that ASHLAR/VALIS need trees for. See §3.

---

## 1. Goal & constraints

A second `registration_method` alongside `valis` that is:

1. **Fully parallel at the Nextflow level** — every expensive unit is an independent process
   (per slide *and per tile*), so a cluster runs them all at once and a smaller machine runs a
   few at a time. No monolithic per-patient task. (This is a statement about task GRANULARITY,
   not about fitting on a laptop — see constraint 2, corrected.)
2. **Bounded per-process memory, with one costly step.** *(Corrected for v1.0.0 — this
   constraint originally read "the default choice for laptops / low-end machines — every
   process fits in ≤8 GB", on measured peaks of COARSE 0.91 GB, REG_TILE 1.31 GB,
   SOLVE <1.31 GB, STITCH 1.35 GB on a 16384² 2-channel slide.)* Everything downstream of the
   anchor still holds that bound and is region-streamed: REG_TILE, SOLVE and STITCH are all
   well under 8 GB regardless of slide size. **COARSE is not**: its matcher is now DISK, a
   U-Net whose activations scale with thumbnail area, so it asks **~48 GB at the shipped
   `high` tier** (`reg_tiled_coarse_max_dim` 2048) and ~5 GB at `low` (512). STARE is
   therefore a *cluster* backend at its default tier; `--reg_tiled_mode low` is what makes it
   workstation-viable. (A single-task `TILED_REGISTER` alternative existed behind a flag; it
   could not hold even the old bound — both whole slides plus an all-channel float32 copy and
   the full warped output live at once — so the flag and the process were removed rather than
   left as an unbounded opt-out.) No JVM, no BioFormats, no whole-slide-in-RAM step. The
   tiling and stitching processes are themselves low-memory and stream.
3. **Native TRE** — emits a VALIS-style Target Registration Error per slide *and* a spatial TRE
   heatmap, as a free byproduct of registration.
4. **reg_qc=2 compatible** — same staged segmentation-overlap QC (`native → rigid → refined`),
   with *zero changes to the QC scorer*.
5. **Non-negative output** — warped/stitched pixels are never negative (protects quantification).
6. **A drop-in adapter** — same subworkflow channel contract as `VALIS_ADAPTER`, selected by
   `params.registration_method`.

ASHLAR is the reference point (phase-correlation + tiled mosaic). STARE reuses ASHLAR's tile +
phase-correlation machinery but, because mirage supplies a fixed reference, replaces ASHLAR's
global MST solve with **reference-anchoring** — every tile lands in absolute coordinates on its
own. That is the novel core (§9).

---

## 2. Why VALIS is the wrong tool on a laptop (evidence from this repo)

| Property of current `REGISTER` | Where | Why it hurts low-end machines |
|---|---|---|
| One monolithic fan-in task per patient, **`process_high` = 200 + 100·attempt GB** (attempt 1 asks **~300 GB**), 8 CPU, 12 h | `register.nf:15` + `conf/modules.config:24-27` | Categorically cluster-only; the only parallelism is *across patients*. |
| JVM heap 32 GB base, +16 GB per retry | `register.nf:57` | BioFormats needs a huge JVM; a laptop has 8–16 GB total. |
| Loads *all* slides at once to build the transform graph | `valis_adapter.nf:8`, `register.py:442` | Peak RAM ≈ all slides in one address space. |
| Sequential per-slide warp | `register.py:843` | The embarrassingly-parallel part is serialized. |
| Can emit **negative pixels**, patched downstream | `register.py:893` (clipped by `split_multichannel.py`) | Overshoot corrupts quantification unless clipped after the fact. |

STARE removes every row: per-tile ≤8 GB tasks, no JVM (pure NumPy/OpenCV/scikit-image/tifffile),
tiles fan out, and non-negativity is guaranteed at the source (§7).

---

## 3. Core concept: reference-anchored parallelism, no trees

ASHLAR and VALIS build trees/orderings because **they must discover a reference and a
registration order from the data** — ASHLAR chains tiles via a minimum spanning tree
(`build_spanning_tree`), VALIS orders whole images via a hierarchical-clustering dendrogram
(`serial_rigid.py`: `order_Dmat` = `fastcluster.linkage` + `optimal_leaf_ordering`).

Mirage hands you the reference explicitly (`is_reference` from the CSV, `registration.nf:117`),
and cyclic-IF re-stains the **same physical section** across cycles. Two consequences:

- **The reference is the global coordinate frame.** Any slide — or any *tile* — that registers to
  the corresponding reference region gets **absolute coordinates for free**. No inter-image or
  inter-tile reconciliation, hence **no tree, no global solve.**
  *(Corrected 2026-09-12: no tree and no ordering problem, but there IS a small solve. ASHLAR's
  spanning tree was never only a positioning device — it is also its cross-tile consistency
  check, and dropping it dropped that check. §6b restores neighbour consistency on the control
  grid without a tree: a per-tile measurement that disagrees with its neighbours is rejected and
  bridged from them. The solve runs on kilobytes and never touches the fan-out.)*
- **Topology is a star:** every moving slide → the reference, independently and in parallel.

The whole design is: make that star *tiled* (for the ≤8 GB memory bound) and *TRE-instrumented*
(for quality), while keeping every unit independent.

---

## 4. The enabling insight: reg_qc is already method-agnostic

The reg_qc=2 scorer in `bin/warp_seg_qc.py` **never imports VALIS**. Its core — `run()` (`:203`),
`score_stage()` (`:154`), `plan_stages()` (`:113`) — works entirely through an *injected* callable
`warp(slide_name, xy, stage) -> warped_xy`, and `write_report()` already accepts an injected
`warp` + `stages` and skips the VALIS loader when they are provided (`warp_seg_qc.py:377-398`).

So STARE does not reimplement QC. It supplies:

1. A per-slide **transform manifest** — `M₀` (global rigid) + a **control-point grid** for the
   mesh field (KB). Replaces the VALIS `registrar.pickle`.
2. A **warper module** `bin/utils/tiled_stage_warp.py` exposing
   `make_warper(manifest) -> warp(name, xy, stage)` where
   `warp(name, xy, 'rigid') = M₀·xy` and `warp(name, xy, 'refined') = M₀·xy + F(xy)`, with `F`
   the smooth mesh field (§6). Pure NumPy — **no JVM**.
3. The stage plan `[native, rigid, refined]`.

Everything else (`warp_seg_qc.py`, `seg_qc_geojson.py`, `utils/cell_pairs.py`) is reused
byte-for-byte. There is no `micro` stage and no destructive composition, so the VALIS
`stage_checkpoint` machinery (`register.py:660`) is simply not needed — `ch_stage_checkpoint` is
`Channel.empty()`.

**Per-slide variable separability (honest wrinkle):** because refinement is TRE-gated (§5), a
slide (or region) that stayed rigid reports `[native, rigid]`; a refined one reports
`[native, rigid, refined]`. `plan_stages` already handles variable separability — this is more
honest than faking an identity `refined` stage.

Exact reg_qc=2 artifact STARE must keep producing (per moving slide):
`<outdir>/<patient>/qc/registration/<patient>_<slide>_seg_qc.json` with keys `iou_mean`,
`iou_p10/p50/p90`, `frac_iou_ge_0.5`, `displacement_px_p50/p90/max` (+ `_um`), `dice_matched`,
`delta_vs_anchor`, `stages_separable`, `matching{…}`, `counts{…}`. Reusing the scorer gives this
shape for free.

---

## 5. Architecture — Nextflow-parallel throughout, ≤8 GB everywhere but COARSE

```
per patient  (patients already parallel)
 └─ per moving slide  (STAR: each slide → the fixed reference, independent)
     COARSE    thumbnail match (DISK + LightGlue) → global rigid M₀    ~48 GB @ high
               # learned features absorb inter-cycle ROTATION/scale; per slide.
               # NOT cheap: peak is linear in thumbnail AREA -- see the table below
     TILE      stream tiles from the tiled OME-TIFF (region reads, halo)      ~1 GB
               # low-mem split; no whole-slide load
     REG_TILE  per tile ∥: moving DAPI tile + reference region (placed by M₀) ≤8 GB
               phase-correlation residual  # near-TRANSLATION after M₀ → ASHLAR's kernel fits
               → one control-point displacement dᵢ at tile centre cᵢ
               → per-tile TRE;  TRE-gated: only high-TRE tiles get a local non-rigid nudge
     WARP_TILE per tile ∥: sample smooth field F = interp({cᵢ→dᵢ}); BILINEAR resample ≤8 GB
               # all channels; non-negative by construction (§7)
     STITCH    write pyramidal OME-TIFF tile-by-tile; feather-blend halos      ≤8 GB
               # low-mem merge; never whole slide in RAM
 └─ emit: registered slide + intrinsic TRE (per-slide table + spatial heatmap)
```

**Primitive split (falls out of the architecture):** COARSE uses feature matching — learned
keypoints and a learned matcher (DISK + LightGlue) as of v1.0.0 — because inter-cycle
repositioning can carry rotation and small scale; after M₀ the per-tile residual is
near-pure-translation, so REG_TILE uses **phase-correlation** (ASHLAR's whitened, Hann-windowed
`phase_cross_correlation`) — cheapest possible, no keypoints needed.

**Memory, per step — and the knob that controls each.** *(This heading used to read "memory
sanity-check (8 GB is generous)"; COARSE broke that premise, see below.)* This table
covers *every* step, which the original version of this paragraph did not: it analysed REG_TILE,
WARP_TILE and STITCH and took COARSE on trust from the word "thumbnail" above. COARSE was in fact
implemented with a full-resolution ORB over an eagerly decoded slide, and OOM-killed at 32 GB on a
26k² input before it was made to match this design (see CHANGELOG, Unreleased → Fixed).

| step | peak driver | ~peak | knob | what you pay for cheapening it |
|---|---|---|---|---|
| COARSE | DISK + LightGlue over the anchor thumbnail | ~48 GB at the shipped tier | `--reg_tiled_coarse_max_dim` (2048) | M0 residual scales with the decimation factor; it must stay well inside `--reg_tiled_halo` |
| REG_TILE | one DAPI tile + halo, both slides | ~50 MB | `--reg_tiled_tile` (2048), `--reg_tiled_halo` (256) | smaller tiles → finer mesh but more tasks; smaller halo → less tolerance for M0 error |
| SOLVE | control points only (kB) | ~10 MB | — | — |
| STITCH | one output write-tile, all channels | ~100 MB | `--reg_tiled_out_tile` (1024) | smaller tiles → more write calls, no accuracy cost |

COARSE's memory is the term worth internalising, and it changed for v1.0.0. The anchor's matcher
is now DISK, a U-Net, which allocates activations over the WHOLE plane it is handed: measured on
the pinned stack (torch 2.3.1 / kornia 0.7.3, CPU) at **3.03 GB for a 512 px thumbnail and 8.78 GB
for 1024 px**, a clean linear-in-megapixels fit of **`GB ≈ 1.1 + 7.3 · Mpx`**. That is ~20x the
classical detector this design was written against, and *linear in area* rather than nearly flat —
so `reg_tiled_coarse_max_dim` is no longer a mild accuracy/cost dial but the thing that decides
whether the step runs at all. 4096 px extrapolates to ~123 GB, which is why the tier column tops
out at 2048, and why `TILED_COARSE`'s memory request is derived from this same bound rather than
being a flat constant. A native-resolution gigapixel plane is thousands of GB, which is why the
anchor is estimated on a thumbnail and `reg_tiled_coarse_max_dim` — not tile size — is the knob
that bounds COARSE. Everything else is genuinely region-streamed: `tiled_io.open_lazy` region reads for
REG_TILE and STITCH, byte-budgeted row bands for COARSE's decimated read.

Tile size *is* the mesh-grid resolution knob: smaller tiles → finer non-rigid but more tasks;
8192² tiles are safe if you want fewer tasks.

---

## 6. TRE + seam continuity (the two quality mechanisms)

**TRE — intrinsic, VALIS `error_df` semantics.** Each REG_TILE scores the residual of the
phase-correlation match *after* applying its transform — a Target Registration Error per tile,
aggregated to a per-slide table (mean/percentiles, in px and µm) and a **spatial heatmap** (one
value per tile — strictly richer than VALIS's single `error_df` number).

**Seam continuity — smooth mesh/grid warp.** Independent per-tile fields would tear cells at tile
boundaries (the problem ASHLAR's MST solves). Instead each tile contributes **one displacement
control-point** `dᵢ` at its centre `cᵢ`; the actual warp is a *single continuous field*
`F(x) = interp({cᵢ → dᵢ})` (bilinear over the control grid, or thin-plate spline). Seam-free by
construction. WARP_TILE and the reg_qc=2 warper sample the **same** `F`, so QC measures exactly
what shipped. Manifest = `M₀` + control grid (KB). TRE-gated tiles that don't refine just
contribute `dᵢ = 0` — the interpolation stays smooth.

---

## 6b. The SOLVE stage (added 2026-09-12)

Until this date SOLVE contained no algorithm: after the three gates (confidence, range, TRE)
it laid the accepted per-tile translations on the grid, median-filtered them over accepted
cells only, and left every rejected or unmeasured cell at `[0, 0]` — a *step* in the field
wherever a tile was dropped, and no defence against a **confident but wrong** tile (a partly
blank crop that correlated against the wrong structure passes every single-tile score; see
the "KNOWN, UNCLOSED EXPOSURE" note that used to sit in `bin/tiled_solve.py`). Every comparable
method — approximating TPS, elastix FFD, RegWSI's diffusive solve, PIV — goes *reject →
regularise → densify*. `stare.solve` now does the same, on the control grid, numpy/scipy only:

| step | what | parameter | source |
|---|---|---|---|
| gates | confidence (`error ≤ max_error`), range (`|d| < max_disp`), TRE (`tre ≥ gate_tre`) | `reg_tiled_max_error`, `reg_tiled_max_disp`, `reg_tiled_gate_tre` | unchanged |
| neighbour consistency | normalised median test: reject a cell whose displacement differs from the median of its accepted 8-neighbours by > 2.0 median-absolute-deviations (+ 0.1 px noise floor); cells with < 3 accepted neighbours are not judged | `nmt_threshold = 2.0`, `nmt_epsilon = 0.1` | Westerweel & Scarano 2005, *Exp. Fluids* 39 |
| in-fill | a rejected or unmeasured cell takes the inverse-distance-weighted mean of accepted cells within 2 grid steps; beyond reach it stays 0, so the field decays into background instead of extrapolating | `infill_radius = 2` | — |
| smoothing | Tikhonov: `argmin_u Σ wᵢ|uᵢ − dᵢ|² + λ Σ|∇u|²`, `wᵢ = 1 − errorᵢ` on measured cells, 0.25 on in-filled ones, sparse solve | `λ = 1.0` (grid units) | Rohr et al. approximating TPS; RegWSI's diffusive term |
| invertibility | STITCH inverts `F` by fixed-point iteration, which converges when the field's Lipschitz constant is < 1 (Chen et al. 2008). The Jacobian of `u` on the grid is reported (max operator norm, min `det(I + J)`); if the norm reaches 0.9 the field is scaled to it and the manifest says so | `max_lipschitz = 0.9` | Chen et al. 2008; Kuang et al. 2019 |

`reg_tiled_solver = 'robust'` (default) selects this; `'legacy'` reproduces the pre-2026-09-12
mesh byte-for-byte (`packages/stare/tests/test_solve.py` pins it against a verbatim copy of
the old stage). The solver's name and diagnostics are recorded in the manifest and in
`*_tre.json` under `solve`. Because the mesh — and therefore every downstream accuracy number —
changes with the solver, the arm benchmark carries it as an axis rather than silently moving the
default: `docs/benchmarks_real.md`, "Re-running a subset after a code change".

---

## 7. Non-negative output (guaranteed, not patched)

Downstream quantification (per-cell mean/median marker intensity) is corrupted by negative
pixels. STARE never generates them:

- **WARP_TILE uses bilinear resampling only.** Bilinear is a convex combination of the four
  neighbouring source pixels, so the result lies in `[min, max]` of non-negative inputs → never
  negative. **Bicubic/Lanczos are forbidden** here: their overshoot (ringing) manufactures
  negatives. This matches the repo's own guidance (`register.py:1046`).
- **STITCH feather-blend is convex** (weights ≥0, sum to 1) → preserves non-negativity across
  halos.
- **Belt-and-suspenders:** after resample+blend, `clamp(0, dtype_max)` and preserve the source
  dtype (uint16), catching any floating-point rounding to −ε.
- The mesh field carries *signed displacements* (coordinates) — unrelated to pixel values, and
  correct.

Unlike the VALIS path, which relies on `split_multichannel.py` to clip negatives *after* warping
(`register.py:893`), STARE's output is non-negative before it is ever written.

---

## 8. Nextflow wiring (drop-in adapter)

The hook exists: `params.registration_method` is defined (`nextflow.config:61`) and validated
via its `nextflow_schema.json` enum (nf-schema, not `ParamUtils`), but `registration.nf:182` hardcodes
`VALIS_ADAPTER`. Add the value `'tiled'` to the validator and branch:

```groovy
// subworkflows/local/registration.nf, STEP 3
if (params.registration_method == 'tiled') {
    TILED_ADAPTER(ch_grouped_multi)                 // new: subworkflows/local/adapters/tiled_adapter.nf
    ch_registered       = TILED_ADAPTER.out.registered
    ch_registrar_pickle = TILED_ADAPTER.out.manifest   // transform manifest (M₀ + control grid), not a pickle
    ch_stage_checkpoint = Channel.empty()              // STARE needs none (§4)
    …
} else {
    VALIS_ADAPTER(ch_grouped_multi)
    …
}
```

`TILED_ADAPTER` emits the **identical channel contract** (`registered [meta,file]`,
`registrar [pid, manifest]`, `stage_checkpoint` empty, `size_logs`, `versions`, `summary`) so
nothing downstream changes. Internally it is the fan-out of §5.

New modules: `modules/local/tiled_coarse.nf`, `tiled_tile.nf`, `tiled_reg_tile.nf`,
`tiled_warp_tile.nf`, `tiled_stitch.nf`. New bin scripts: `tiled_coarse.py`, `tiled_tile.py`,
`tiled_reg_tile.py`, `tiled_warp_tile.py`, `tiled_stitch.py`, plus
`bin/utils/tiled_stage_warp.py` (the QC seam) and `bin/utils/mesh_field.py` (shared field
interpolation — used by WARP_TILE *and* the warper, single source of truth).

reg_qc=2 dispatch: add `--method tiled` to `warp_seg_qc.py` so it injects
`tiled_stage_warp.make_warper(manifest)` + `stages=[native,rigid,refined]` instead of the VALIS
loader (`warp_seg_qc.py:105,377`). The scorer is untouched.

Container: a slim `python + opencv + scikit-image + tifffile + numpy + scipy` image — **no JVM,
no libvips-from-source**.

Resource labels: STARE tasks genuinely need only a few GB, but the standard labels are
cluster-sized (`process_low`=32 GB, `process_medium`=200 GB in `conf/modules.config`). Ship
dedicated lean `withName:'TILED_*'` overrides (2–8 GB) or pair with a memory-capped profile like
`conf/test.config` (pins `process_high`=6 GB). Under such a profile a 4-core/16 GB laptop runs
2–4 tiles concurrently and the pipeline *completes* instead of OOM-killing.

> Exec-bit rule (CLAUDE.md): every name-invoked `bin/tiled_*.py` must be
> `git update-index --chmod=+x` → mode `100755`, or it fails exit 126 on the cluster. Import-only
> `bin/utils/*.py` stay `100644`.

---

## 9. What is genuinely new here

1. ~~**Reference-anchored tiled registration — tiling without a global solve.**~~ **Retracted
   2026-08-25 (research fleet) and corrected 2026-09-12.** ASHLAR's *cycle-registration* phase
   has anchored later cycles' tiles to a fixed reference since 2021, so reference anchoring is
   prior art; and its spanning tree is also its cross-tile consistency check, which STARE had
   dropped rather than replaced. What survives as a differentiator is the combination in §9.2–4
   plus a **non-rigid**, WSI-to-WSI solve that restores neighbour consistency on a grid without a
   tree (§6b) — an engineering contribution, to be claimed as such.
2. **Registration-as-a-DAG-of-≤8 GB-processes.** The archived tiled path tiled only the *warp*
   (monolithic VALIS `REG_PREP`); STARE tiles the *registration estimation* itself, JVM-free.
3. **Intrinsic per-tile TRE → a spatial error heatmap** that doubles as the refinement gate —
   quality metric and control signal are the same object.
4. **Non-negativity by construction** (convex resample + convex blend), not post-hoc clipping.

---

## 10. Implementation plan

1. **QC seam first (lowest risk).** Write `bin/utils/tiled_stage_warp.py` + `mesh_field.py` and a
   fake manifest; unit-test that `warp_seg_qc.run()` scores a `[native,rigid,refined]` plan
   through it (the scorer is already injectable — `warp_seg_qc.py:377`). Proves reg_qc=2 works
   before any registration exists.
2. **Rigid core.** `tiled_coarse.py` (M₀) + `tiled_tile.py` + `tiled_reg_tile.py` (phase-corr
   residual, TRE, no non-rigid yet) + `tiled_warp_tile.py` (bilinear) + `tiled_stitch.py`. Wire
   `TILED_ADAPTER` and the `registration.nf` branch. Validate on the test profile
   (`nextflow run . -profile test,docker -stub`) against known VALIS output. Assert non-negativity.
3. **Mesh non-rigid + reg_qc=2 end-to-end.** Add TRE-gated control-point refinement, the smooth
   field, and the `refined` stage; dispatch `warp_seg_qc.py --method tiled`.
4. **TRE outputs.** Per-slide table + spatial heatmap.
5. **nf-test** modules, exec-bit the bin scripts, CI stub coverage, `low`/`high` presets.

## Risks / open questions

- **Accuracy ceiling.** A coarse mesh non-rigid is weaker than VALIS optical-flow micro-
  registration. STARE is deliberately the *fast/low-mem* option; VALIS stays the *high-accuracy*
  option. The intrinsic TRE + reg_qc=2 quantify the gap so a user knows when to escalate.
- **Mesh resolution ↔ parallelism tradeoff.** Finer control grid = better non-rigid but more
  tiles/tasks. Tile size is the knob; document the tradeoff, don't hide it.
- **Archived path lessons.** `archive/tiled-valis-2026-07-24` (`REG_PREP→REG_WARP→REG_ASSEMBLE`,
  patched VALIS container) was removed 2026-07-24 — appears to be scope/publication cleanup, not a
  viability failure. STARE differs fundamentally (JVM-free, per-*tile registration*, mesh-warp
  continuity). Confirm with the author whether any removal reason must be designed around.
- **Sparse-fluorescence COARSE.** *(Written against the classical detector: ORB on DAPI needed
  enough keypoints.)* DISK is a learned detector and finds keypoints on far sparser tissue, but
  the concern is not retired — phase-correlation on the thumbnail remains the M₀ fallback if a
  slide is sparse enough that even DISK under-matches.
- **Output frame.** Anchor to the reference slide's native grid (identity for the reference), so
  registered pixel coordinates match what postprocessing/segmentation expects.
- **OME channel manifest.** STITCH must still emit `channels_manifest.json` (filename → OME
  channel names) so `TILED_ADAPTER` matches registered files back to meta by channel signature
  (`lib/RegisteredMatch.groovy`). Reuse `create_channels_manifest.py`.

## Implementation status (branch `feat/tiled-registration`)

**Done & verified**
- **Phase 1 — reg_qc=2 seam:** `bin/utils/mesh_field.py` (smooth field + non-negative bilinear
  resampler), `bin/utils/tiled_stage_warp.py` (`make_warper`). Unit-proven that the existing
  `warp_seg_qc` scorer runs a `[native, rigid, refined]` plan through the tiled warper unchanged.
- **Phase 2 — rigid core:** `tile_grid`, `coarse_align` (ORB+RANSAC M0 + residual TRE),
  `tile_residual` (whitened/Hann phase-corr), `tiled_warp` (inverse-map bilinear, non-negative),
  `tiled_manifest` (TRE-gated control grid), `tiled_pipeline.register_slide` (end-to-end). An
  end-to-end test realigns a synthetically warped slide (corr > 0.9), a pure shift needs no mesh,
  a non-rigid warp is captured by the mesh.
- **CLI:** `bin/tiled_register.py` (100755) — real OME-TIFF I/O, smoke-tested.
- **Nextflow wiring:** `modules/local/tiled_register.nf`, `subworkflows/local/adapters/tiled_adapter.nf`,
  the `registration.nf` method branch, `validateRegistrationMethod += 'tiled'`, `reg_tiled_*`
  params, and a lean 8 GB resource block. **Stub run green** end-to-end, JVM-free.
  (The single-task `TILED_REGISTER` described here was later removed — see CHANGELOG.)
- **reg_qc=2 seg-QC dispatch (done):** `warp_seg_qc.py --method tiled` builds the warper from the
  STARE manifest (no JVM), and `WARP_SEG_QC_TILED` + the valis/tiled dispatch branch in
  `subworkflows/local/seg_qc.nf` (called from `registration.nf`) feed it one manifest per moving
  slide. **Stub run green at reg_qc=2** — emits the `native/rigid/refined`
  `_seg_qc.json`. Unit-tested through the real CLI `main()`.
  (Historical: this status entry describes the state as of `feat/tiled-registration`.
  `WARP_SEG_QC_TILED` was later merged into `WARP_SEG_QC`, dispatching on the `method`
  input via `lib/WarpBackends.groovy` — Task 3 of arch-group-c-config, 2026-08-07 — so
  the process name above no longer exists; the dispatch behavior it describes still
  does. See `containers/README.md` and `containers/tiled/Dockerfile` for the
  already-updated references.)
- **Slim container (done):** `containers/tiled/` (`python:3.11-slim` + numpy/scipy/scikit-image/
  tifffile, no JVM/libvips/GPU). **Built and verified locally** — the CLIs import and run
  in-container; **~438 MB** vs the multi-GB VALIS image. Added to the `containers.yml` matrix.
- **Intrinsic TRE (done — VALIS-analogous, emitted by both paths):** `_tre.json` carries
  `coarse_tre_px` (rigid feature-fit residual, like VALIS's rigid error), a per-tile `rigid_tre_px`
  **spatial heatmap** VALIS doesn't have, and — in the default monolithic path — `residual_after_px`,
  the per-tile residual *after* the mesh (STARE's post-registration final-accuracy number, the
  analogue of VALIS's non-rigid error; test: it beats the rigid residual). Built by the shared
  `bin/utils/tre_report.py`; the fan-out (`TILED_SOLVE`) now emits the rigid spatial heatmap too
  (previously dropped). The fan-out's final-accuracy residual comes from the reg_benchmark harness.
- **Final QC-report integration (done):** the `_tre.json` already flowed into the report's
  `registration_tre/` input; `generate_qc_report.py` now renders it as a "Registration Accuracy
  (STARE Tiled TRE)" subsection: a per-slide caption carrying the headline numbers
  (coarse / rigid p50-p90 / post-refinement final p50-p90 / refined / accepted-of-total
  tiles), a **per-stage error-distribution plot** for the rigid and post-refinement
  stages built from the accepted tiles, and the **per-tile SVG heatmap** of the spatial
  TRE. It sits alongside the VALIS rTRE table and the per-stage seg-QC displacement
  plots. Unit-tested.
- **Accuracy harness (done):** `bin/utils/reg_benchmark.py` — a ground-truth-free residual-TRE +
  correlation metric that runs on any method's output, so VALIS vs tiled is a direct
  number-to-number comparison on the same slide. Validated on synthetic ground truth (STARE drops
  the residual TRE below 2 px; a pure 11.66 px shift is fully removed). The CLI that wrapped it,
  `bin/registration_benchmark.py`, lives on the `benchmarking` branch with the sweep that drives
  it; on this branch the library is exercised by `tests/test_reg_benchmark.py` and
  `tests/test_tiled_fanout.py` and is allowlisted in `tests/test_no_dead_bin_modules.py`.
- **Per-tile Nextflow fan-out (done, and now the only shape):** the adapter runs
  `TILED_COARSE → TILED_REG_TILE (one task per tile) → TILED_SOLVE → TILED_STITCH` — the
  little-process-per-tile design, all JVM-free. `warp_image` gained an `out_origin` so each tile
  task warps only its window (and the stitch warps in row strips). The fan-out chain is proven to
  compose into a correct registration end-to-end (synthetic ground truth), and the DAG is **stub
  green at reg_qc 1 and 2**; the default (`false`) monolithic path is unchanged.
- **Streaming gigapixel stitch (done):** `TILED_STITCH` no longer materialises the moving slide or
  the output. It reads only each output tile's source pixels (`source_region` inverse-map + a lazy
  `zarr` region read via `tifffile` `aszarr`), warps that tile (`warp_image` `out_origin`+`src_origin`),
  and writes it straight to a tiled OME-TIFF. Peak memory is one source crop + one output tile per
  channel. Proven bit-identical to the whole-image warp (±1 rounding); container gains `zarr`
  (amd64 image built & verified, 482 MB).
- **~67 Python tests passing; ruff clean.**

**Convention settled during implementation:** the mesh lives in the **reference frame** (sampled
at the rigid position `M0·xy`), which the tiled implementation produces naturally and which gives
a decoupled warp inverse. §5/§6 describe this.

**Remaining**
- **Run the accuracy harness on real WSIs vs VALIS** (operational — the metric and tooling are in
  place; this is executing them on real slides + a VALIS run, which needs the cluster).
- **nf-test** module + integration coverage (the Python cores and the stub DAG are covered; native
  nf-test cases for the new processes would round it out).
```

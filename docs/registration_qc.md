# Staged registration QC (`reg_qc = 2`)

At `reg_qc = 2` the pipeline answers a question the nuclear-channel overlay cannot: **which registration
stage actually improved the alignment, and by how much?**

It does that by segmenting each slide's native image, establishing cell-to-cell
correspondence **once**, and then re-measuring those same cell pairs after each stage of the
VALIS transform.

The chain is three processes — `SEG_QC_SEGMENT` → `SEG_QC_GEOJSON` → `WARP_SEG_QC`.
`SEG_QC_SEGMENT` is `SEGMENT` under an alias, so the QC segments with **the run's own
segmenter** (`--seg_method`, default `instantseg`) rather than with a second, fixed one.

!!! abstract "See it as a figure"
    This ladder is panel **f** of **Supplementary Figure S2** —
    [registration](figures/registration-schematic.html){ target=_blank } — and panel **e** of
    **Supplementary Figure S3** —
    [quality control](figures/qc-schematic.html){ target=_blank }, which places it in the
    pipeline-wide QC architecture.

    **Supplementary Figure S6** — [accuracy measures](figures/accuracy-schematic.html){ target=_blank } —
    places this chain against the pipeline's two other accuracy numbers and says which of
    them can be read as independent evidence.

## The `reg_qc = 1` overlay is a before/after pair

Below the staged metrics sits the cheaper answer, on at every level from
`reg_qc = 1` up: one image per moving slide showing what registration
corrected.

`GENERATE_REGISTRATION_QC` takes the tuple `[meta, registered, native, reference]` —
the registered moving slide, its **native** (pre-registration) counterpart, and
the patient's reference — and renders **two composites side by side**, separated
by a blue band:

| Panel | Green | Red |
|---|---|---|
| **Before** (left) | reference | the native moving slide, un-registered |
| **After** (right) | reference | the registered moving slide |

Perfect alignment is yellow (red + green); misalignment shows red/green
fringing. **Fringing that shrinks from the left panel to the right one is the
correction registration applied.** The single "after" composite this used to
render showed what registration *produced* and gave a reader nothing to compare
it against — an unregistered pair and a perfectly registered one both look
plausible on their own.

**Both panels are drawn on the reference canvas, origin-aligned, reconciling
differing dimensions by pad-or-crop — never by rescaling.** Rescaling the native
image onto the reference's shape would absorb the scale component of the
misalignment into the resampling, so the "before" panel would understate the
pre-registration error and make registration look better than it was. Padding
and cropping preserve every pixel's original position in the reference frame,
which is what a reader of the figure is entitled to assume they are seeing.
(`bin/utils/qc.py:compose_on_reference_canvas`.)

**Channel selection is fail-fast.** The nuclear/fiducial channel in every panel is
resolved from `nuclear_markers` (`metadata.pick_nuclear_index`); if no configured
marker matches a slide's OME channel names, `create_registration_qc` raises rather
than silently falling back to channel 0.

**The CLI keeps `--native` optional; the pipeline never exercises that path.**
`bin/generate_registration_qc.py --native` is `nargs="+", default=None` — a
caller who omits it gets the old single "after" panel
(`bin/generate_registration_qc.py:151-158`, `bin/utils/qc.py`'s `native_nuc`
guarded on `if native_path is not None`, ~line 599) — that flexibility exists
because the same script also has callers outside this process. But
`GENERATE_REGISTRATION_QC` (`modules/local/generate_registration_qc.nf:38,64`)
declares `native_image` as a **required** path in its input tuple and always
passes `--native ${native_image}`, so a pipeline run never takes the
single-panel branch. If the file that path names does not resolve,
`create_registration_qc` raises `FileNotFoundError`
(`bin/utils/qc.py:479-480`) rather than silently falling back to one panel.

The published names are unchanged —
`<outdir>/<patient>/qc/registration/<slide>_QC_RGB.png`, `_QC_RGB.tif` and
`_QC_RGB_fullres.tif` — so `GENERATE_QC_REPORT` and anything reading the output
tree sees one artifact per moving slide exactly as before. It is simply twice as
wide.

The native image costs nothing extra to obtain: it is the stream that entered
registration (`REGISTER_PATIENT.out.images_multi` on the linear path,
`PREPROCESSING.out.preprocessed` under `--mode add_cycle`), joined back in on
`meta.id`. It does cost memory — the process now holds three full-resolution
planes instead of two — which is why its request is tiered on the combined size
of all three inputs (see [Resources](resources.md#registration-qc)).

Under `--mode add_cycle` the pair reads the same way, with one asymmetry worth
knowing: the reference is the **frozen prior** reference read out of
`--prior_outdir`, so the "before" panel measures the new cycle against a frame
established in an earlier run. That is the drift the mode exists to detect.

## Why the correspondence is fixed, and fixed *there*

VALIS applies four transform states in this order:

| Stage | What it is | Where it comes from |
|---|---|---|
| `native` | no transform | the segmentation as written |
| `rigid` | the rigid transform, after `MicroRigidRegistrar` refined it | the registrar pickle, `non_rigid=False` |
| `non_rigid` | rigid + the wave-1 non-rigid displacement field | the REGISTER **stage checkpoint** |
| `micro` | rigid + non-rigid + the micro-registration residual | the registrar pickle, `non_rigid=True` |

`rigid` is the anchor. Two reasons:

1. **Before registration the cells are too far apart to pair.** Two rounds of a cyclic-IF
   experiment sit a whole tissue-offset apart in their native frames; any matcher run there is
   pairing noise, not cells.
2. **`rigid` is the transform every later stage builds on.** `MicroRigidRegistrar` runs *inside*
   `Valis.register()`, before the non-rigid stage, so the rigid transform in the final pickle is
   already the one the non-rigid and micro fields were fitted on top of. A pairing valid at
   `rigid` is therefore valid at every later stage by construction.

Holding the pairing is the point. If the match were re-derived per stage, a score could move
because *different cells got paired* — or because a cell that was occluded in one stage's raster
reappeared in another's. Neither is a registration effect. With the pairing fixed, every change
between stages is geometry.

Pairing rule: **optimal one-to-one assignment** (`scipy.optimize.linear_sum_assignment`)
minimising total centroid distance, within **1.5 median nuclear radii**, solved exactly per
connected component of the candidate graph — components on rigid-aligned tissue are normally
1-4 cells, so this is exact, not an approximation, and never materializes a whole-slide cost
matrix. A component whose size exceeds `--max-component-cells` falls back to mutual-NN inside
that component only. The older **mutual nearest centroid** rule (each cell pairs only if it is
the other's nearest neighbour, which stops a dense cluster collapsing onto one popular target)
remains selectable via `params.seg_qc_pairing` / `--pairing mutual_nn`. The radius, in both
cases, stops a cell with no true partner pairing with whatever happens to be nearest, and is
tunable (`params.seg_qc_match_radius_factor` / `--match-radius-factor`, or an absolute
`--match-radius-px`).

## Metrics

Per stage, over the fixed pairs:

- **Per-pair IoU** — `iou_mean`, `iou_p10/p50/p90`, `iou_max`, and `frac_iou_ge_0.5`.
  Each pair is rasterized in its own bounding-box window (supersampled 2× by default), so peak
  memory is one nucleus rather than one slide.
- **Centroid residual displacement** — `displacement_px_p50/p90/max`, plus `displacement_um_*`
  when the slide metadata carries a physical pixel size. This is usually the number to quote: it
  has no resolution floor the way a thresholded IoU does.
- **`dice_matched`** — areal Dice restricted to the matched pairs. Named to make clear it is not
  a whole-slide foreground Dice; unmatched cells contribute nothing.

`delta_vs_anchor` reports each stage minus `rigid`. **A positive displacement delta means that
stage made the alignment worse** — which is exactly the failure this design exists to surface,
since micro-registration is caught-and-continued in `bin/register.py` and would otherwise fail
silently.

## The REGISTER stage checkpoint

VALIS composes its stages destructively:

- `MicroRigidRegistrar` overwrites `slide.M`.
- `register_micro()` does `fwd_dxdy = fwd_dxdy + micro_residual` and writes it back onto the
  same attribute.

So a finished registrar pickle holds **one** field that is rigid + non-rigid + micro, and no
reader can recover the intermediate. REGISTER therefore snapshots each slide's forward
displacement field at the only moment it exists — after `register()` returns, before
`register_micro()` is called — into `reg_stage_checkpoint/`. Fields are at non-rigid
registration resolution (a few thousand pixels a side), not slide resolution: tens of MB.

Writing it can never fail a registration; it is QC input, and REGISTER's own success does not
depend on it. The QC process that *reads* it is gating, though — `GENERATE_REGISTRATION_QC`
carries `retry-then-fail`, so if that process itself dies the run fails. When the checkpoint is
merely missing, the QC still runs and reports `native`, `rigid` and `micro`, sets
`stages_separable: false`, and records a `note` saying why `non_rigid` is absent. When
micro-registration did not run (skipped, or it raised and was caught), the checkpoint records
that and the QC omits the `micro` stage rather than reporting a byte-for-byte duplicate of
`non_rigid`.

## Output

`<outdir>/<patient>/qc/registration/<patient>_<slide>_seg_qc.json`:

```json
{
  "patient_id": "P001",
  "moving": "P001_cycle2",
  "reference": "P001_cycle1",
  "stages_separable": true,
  "stage_order": ["native", "rigid", "non_rigid", "micro"],
  "matching": {
    "method": "lsa_centroid",
    "anchor_stage": "rigid",
    "radius_px": 9.3,
    "median_cell_radius_px": 6.2,
    "n_components": 181940,
    "largest_component_cells": 4,
    "n_fallback_components": 0,
    "n_fallback_cells": 0,
    "n_pairs": 184203,
    "pair_fraction": 0.91
  },
  "stages": {
    "rigid":     { "n_pairs": 184203, "iou_mean": 0.42, "displacement_px_p50": 4.1, "displacement_um_p50": 1.33 },
    "non_rigid": { "n_pairs": 184203, "iou_mean": 0.71, "displacement_px_p50": 1.6, "displacement_um_p50": 0.52 },
    "micro":     { "n_pairs": 184203, "iou_mean": 0.78, "displacement_px_p50": 1.1, "displacement_um_p50": 0.36 }
  },
  "delta_vs_anchor": {
    "micro": { "iou_mean": 0.36, "displacement_px_p50": -3.0 }
  },
  "counts": { "features_ref": 201884, "features_moving": 198110 }
}
```

Read it as: rigid alone left a median residual of 4.1 px (1.33 µm); non-rigid took that to
1.6 µm-worth; micro-registration bought a further 0.5 px. `pair_fraction` below ~0.5 means the
pairing itself is thin — check the rigid stage before trusting the later numbers.

## Reading the numbers critically

- **`pair_fraction`** is the first thing to check. A low value means rigid registration left the
  slides too far apart for a nuclear-radius match, so every later stage is being measured on a
  biased subset of cells that happened to land close.
- **`n_pairs_scored` vs `n_pairs`** — the difference is pairs whose bounding box exceeded
  `--max-pair-window-px` (a warp artifact or merged blob). A large gap means distorted geometry,
  not a bad alignment.
- **IoU is bounded by segmentation agreement, not just registration.** Two independent
  segmentation runs on the same nucleus in different rounds do not produce identical outlines, so `iou_mean`
  has a ceiling below 1 even under perfect registration. The *delta* between stages is the
  registration signal; the absolute level is not.
- **Vertices are not clipped to the aligned frame by default**, unlike `Slide.warp_geojson`.
  Clipping flattens boundary-straddling cells onto the crop edge in both slides, inflating their
  IoU for reasons unrelated to registration. Pass `--clip-to-frame` to reproduce VALIS exactly.

## In the HTML QC report

`GENERATE_QC_REPORT` renders these JSONs twice over, both as plots:

- **Registration QC → Warp-Segmentation QC** — one error-distribution plot per
  registration stage, over the stages' `displacement_um_p50` (or
  `displacement_px_p50` when no pixel size was available), **one point per slide**.
  A directory holding both calibrated (µm) and uncalibrated (px) slides for the
  same stage is never mixed into one histogram: it is split into up to two plots,
  one per unit actually present, each titled with its own unit — the STARE
  (tiled) backend is the common source of px-only slides, not an edge case. The
  JSON carries summary statistics, not per-cell values, so the distribution is
  across slides; a single-slide run renders a single bar and says `n=1`. The
  caption also reports the matched-cell count (`matching.n_pairs`) across slides,
  min and median, so a p50 measured over 12 matched cells reads differently from
  one measured over 40 000 — the histogram shape alone cannot tell the two apart.
- **Feature-TRE vs Cell-Displacement Reconciliation** — a log-log scatter of the
  registration method's own intrinsic TRE (x) against these cell displacements (y),
  one point per slide-stage, with the 3× divergence band drawn as two diagonals.
  A point outside the band is a slide-stage where the registrar's own keypoints and
  the independently segmented nuclei disagree about whether the slide is aligned;
  that disagreement is the failure neither measure catches alone.

The per-slide records themselves are not re-tabulated in the page — they are copied
verbatim into `qc/mirage_qc_data_<timestamp>/seg_qc/`, which is where a per-slide
number belongs.

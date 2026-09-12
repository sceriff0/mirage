# ANHIR landmark harness

Scores the pipeline's registration backends (`valis`, `tiled` = STARE) on the public
[ANHIR challenge](https://anhir.grand-challenge.org/) with the challenge's own metrics,
alongside the challenge's bUnwarpJ baseline and the unregistered pose, and packages the
held-out cases for upload to grand-challenge.org. Its two output tables are the
ihc_method hand-off (`benchmarks/pull_to_ihc_method.sh --anhir`).

## What the challenge measures

From <https://anhir.grand-challenge.org/Performance_Metrics/>, implemented in
`landmarks.py` and `metrics.py`:

| quantity | definition |
|---|---|
| TRE | Euclidean distance between a warped source landmark and its target landmark |
| rTRE | TRE / image diagonal (`sqrt(w² + h²)`), dimensionless |
| per case | median, mean and max rTRE over the landmarks |
| robustness | fraction of landmarks whose rTRE *after* registration is below the initial rTRE |
| ranking | methods ranked per case by median rTRE; the primary score is the mean rank over cases |
| robust subset | cases with robustness > 0.5 |

Warped landmarks are **source landmarks moved into the target image's frame**, in pixels
of the image at the scale registered. Both images of a case share one scale, so the
identity transform is the initial pose.

## Data

`challenge/` is gitignored. The download (`challenge/anhir/`) holds:

- `dataset_medium.csv` — the cover table: 481 cases, 230 `training` and 251 `evaluation`.
- `dataset_medium.z01 … z05` + `dataset_medium.zip` — the **images only**, a split archive
  (294 JPEGs, 12.8 GB). No landmark file is inside it.
- `BmUnwarpJ/<case_id>/` — the challenge's bUnwarpJ baseline: `warped_source_landmarks.txt`
  (ImageJ point format) and `TIME.txt` (milliseconds) for 237 of the cases.

**The landmark archive is a separate download** from the challenge's Data page. Unpack it
to `challenge/anhir/landmarks/` so that `challenge/anhir/landmarks/COAD_01/scale-25pc/HE.csv`
exists. Only training cases have target landmarks; evaluation cases are scored server-side,
and without the landmark archive nothing is scored locally (the harness still builds the
submission package).

## Run

```bash
# 1. rejoin the split archive and extract the images (needs Info-ZIP `zip`; ~13 GB more disk)
python -m benchmarks.anhir.prepare join --data-root challenge/anhir

# 2. OME-TIFFs + samplesheet + pairs manifest (start with the training cases)
python -m benchmarks.anhir.prepare convert --data-root challenge/anhir --work anhir_work --status training

# 3. register with each backend (one pipeline run per method; needs the site config on a cluster)
ANHIR_SITE_CONFIG=site.config ANHIR_PROFILE=slurm,singularity \
  benchmarks/anhir/run_pairs.sh anhir_work anhir_results tiled
ANHIR_SITE_CONFIG=site.config ANHIR_PROFILE=slurm,singularity \
  benchmarks/anhir/run_pairs.sh anhir_work anhir_results valis

# 4. warp the source landmarks through each published transform
python -m benchmarks.anhir.warp --method tiled --pairs anhir_work/pairs_manifest.csv \
    --outdir anhir_results/tiled --trace anhir_results/tiled/trace.txt --out anhir_results/tiled_warped
python -m benchmarks.anhir.warp --method initial  --pairs anhir_work/pairs_manifest.csv --out anhir_results/initial_warped
python -m benchmarks.anhir.warp --method bunwarpj --pairs anhir_work/pairs_manifest.csv \
    --baseline-root challenge/anhir/BmUnwarpJ --out anhir_results/bunwarpj_warped

# 5. score + package
python -m benchmarks.anhir.evaluate --dataset challenge/anhir/dataset_medium.csv \
    --landmarks-root challenge/anhir/landmarks \
    --warped tiled=anhir_results/tiled_warped --warped valis=anhir_results/valis_warped \
    --warped initial=anhir_results/initial_warped --warped bunwarpj=anhir_results/bunwarpj_warped \
    --out anhir_results/tables --submit tiled --submit valis

# 6. hand off to ihc_method
benchmarks/pull_to_ihc_method.sh <arm_results_root> ../ihc_method --anhir anhir_results/tables
```

**The VALIS leg of step 4 must run inside the pipeline's VALIS container**
(`modules/local/register.nf`'s `container`): unpickling a registrar needs the `valis`
package and a BioFormats JVM. Mount the repo and the results and run the same command
there; the `tiled` leg is pure NumPy and runs anywhere.

## How the pipeline is driven

- Every case is one **patient** (`anhir<case_id>`) with the target slide as the reference
  and the source slide as the moving slide, so one run registers every case.
- The JPEGs are converted **once per distinct image** to a two-channel uint8 OME-TIFF:
  `DAPI` = inverted luminance (nuclei bright, which is what both backends anchor on) and
  `<STEM>` = plain luminance. Two channels because the pipeline claims each channel name once
  per patient and a moving slide whose only channel is nuclear would have nothing left.
  Per-case files are symlinks named `<patient>_<stem>.ome.tif` so no two samplesheet rows
  share a basename.
- `--start registration --stop registration`: the OME-TIFFs stand in as the preprocessed
  slides (BaSiC illumination correction is for fluorescence), and nothing after
  registration runs.
- The transform each case is scored through is what the run **publishes**:
  `registered/manifest/*_manifest.json` (STARE, warped with the pipeline's own
  `tiled_stage_warp.make_warper`, `refined` stage) or
  `registered/transform/*_registrar.pickle` (VALIS, `Slide.warp_xy` with
  `crop="reference"` so coordinates land in the target image's own frame). Keep
  `cleanup_level` at `none`.
- Execution time is summed from `-with-trace` over `REGISTER` / `TILED_COARSE` /
  `TILED_REG_TILE` / `TILED_SOLVE` per patient; the challenge normalises times against
  a `computer-performances.json` you generate on the machine that ran the registration
  (BIRL's `bm_experiments/bm_comp_perform.py`).

## Outputs

`anhir_cases.csv` — one row per (case, method): `case_id, tissue, scale, status,
source_image, target_image, method, n_landmarks, scored, rtre_median, rtre_mean, rtre_max,
tre_median_px, robustness, rank_median_rtre, time_min, imputed_initial`. `scored` is false
for evaluation cases (metrics NaN). `imputed_initial` is true when the method produced no
warped landmarks for a training case: the challenge counts a missing registration as the
initial pose, ranked last, so that row carries the unregistered error rather than vanishing.

`anhir_aggregates.csv` — one row per (method, subset): `n_cases, avg_median_rtre,
med_median_rtre, avg_mean_rtre, avg_max_rtre, avg_robustness, med_robustness,
avg_rank_median_rtre, avg_time_min`, with subsets `all`, `training`, `evaluation`,
`robust` and `tissue:<name>`.

`anhir_missing.csv` — cases a method produced no warped landmarks for. A method's average
rank is over the cases it did run, so read this before comparing two methods.

`submission/<method>.zip` — `registration-results.csv` (the cover table with
`Warped source landmarks` and `Execution time [minutes]` filled in) plus `landmarks/`.

## What "performs well" means here

The leaderboard's top tier is dense: the winning entries sit at an average-of-median rTRE
around 0.002–0.003 with robustness ≥ 0.98, and VALIS's published ANHIR number is in that
band. Reviewers recognise the server-scored evaluation set, not a training-only table,
so submit. See `.planning/research/2026-09-12-stare-paper-positioning/REPORT.md` for the
competitor numbers and what a standalone STARE paper would additionally need.

## Relation to the deleted harness

An earlier landmark harness (removed at 61e26ec; `benchmarks/README.md` section B is its
record) drove `bin/register.py` and a single-task STARE entry point that no longer exists.
This harness drives the **pipeline** instead, so the STARE leg goes through the same
four-stage fan-out and the same published manifest as a production run.
`benchmarks/tests/test_no_competition_framing.py` still forbids any reference to that
deleted harness by path or name; ANHIR itself is live again.

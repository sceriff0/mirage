# benchmarks/figures — the paper's data figures, from the results

Every data panel is made by a script here, from the regbench results (`REGBENCH_OUT`) and
one pipeline run. Schematics are drawn by hand. The panels are composed in Affinity.
Each script also writes every panel as its own PNG, plus a `_source_data.csv` holding
the plotted values and every choice made.

**Everything, start to end** (from the regbench root, `REGBENCH_INPUT` = the samplesheet):

```bash
bash <checkout>/benchmarks/figures/run_all.sh launch   # the benchmark: one sbatch chain
bash <checkout>/benchmarks/figures/run_all.sh status   # done when nothing is queued
bash <checkout>/benchmarks/figures/run_all.sh paper    # paper tables + every figure
```

Run `paper` before `keep_results.sh`, which frees the written images the crop and overlay
figures read. The figures alone:

```bash
pip install -r requirements/ci.txt        # matplotlib, pandas, scikit-image, STARE (pinned)
cd <regbench root>                         # where regbench.env sets REGBENCH_CASES / _OUT
bash <checkout>/benchmarks/figures/make_figures.sh             # all of them
bash <checkout>/benchmarks/figures/make_figures.sh cohort crops # some of them
# figures: cost cohort crops crops_all overlay segmentation by_case_round seg_backends
```

Every choice (variants, cases, zoom sizes, display, seed) sits in one block at the top of
`make_figures.sh`. The paper's figures are the ones made from that file as committed. Any
one choice can be overridden for a single run from the environment
(`VALIS=valis120_micro bash ...`).

These scripts do NOT run in the scorer's image, which has no matplotlib or pandas. They
import `benchmarks.regbench` from this checkout, so every score is computed by the
harness's own code.

**The manuscript's panels, as submitted.** Each of these is the script that drew the
submitted panel, copied unchanged; run on the tables below, each writes a
`*_source_data.csv` byte-identical to the one submitted (checked 2026-11-06):

| Manuscript | Script | Reads |
|---|---|---|
| Fig. 2c | `fig2c_peak_rss.py <REGBENCH_OUT>/tables/synthetic <out>` (`--no-hires` for the variant without the high-resolution example) | `stare_stage_peaks.csv`, `synthetic_cost.csv` of the synthetic run |
| Fig. 3b–e | `fig3bcde_benchmark.py <REGBENCH_OUT>/tables <out>` | `cells.csv`, `resources.csv` of the cohort run |
| Extended Data Fig. 6 | `ed_fig6_by_case_round.py <REGBENCH_OUT>/tables <out>` (writes `Figure_S6.*`, the name it was drafted under) | `cells.csv` of the cohort run |
| every accuracy number of the text, Supplementary Table 2 and Supplementary Note 6 | `python -m benchmarks.regbench.paper.accuracy_stats <REGBENCH_OUT>/tables` | `cells.csv`, `cells_nuclei.csv`, `ashlar_discard.csv` |

Both metrics are tested on every one of the 64 protocol round pairs, against the same five
(VALIS, PALOM, ASHLAR, DeeperHistReg and the unregistered condition), Holm over the five
(ruling 2026-11-05: no readable-displacement filter).

**The results are not in this repository** (the input slides cannot be shared; Data
availability). `<REGBENCH_OUT>/tables` is what `regbench score` writes. The manuscripts' cohort
tables are the 78-pair scoring the paper's runs produced (local copy
`new_results/partial-20261103-1350/tables`, ignored by git, despite the folder's name the
complete one); the synthetic tables are `new_results/out-20261104/tables/synthetic`.

**Older renderings and the other panels.** The scripts below came first; where one draws a
panel that has a script above, the one above is what was submitted. Names follow the
manuscript (Extended Data Figs. 4–7 were drafted as Supplementary Figs. S4–S7, Supplementary
Note 6 as Section S8, Supplementary Table 2 as Table S2).

| Script | Reproduces | Reads |
|---|---|---|
| `fig_synthetic_cost.py` | Not a paper figure: peak RSS and CPU time vs input size per patient on the COHORT, VALIS vs STARE (`make_figures.sh cost`; despite its name it reads `COST_DATASET`, the cohort by default) | `tables/summary_resources.csv`, `tables/resources.csv` |
| `fig_rss_vs_size.py` | An earlier Fig. 2c (log y axis; submitted: `fig2c_peak_rss.py`): peak RSS vs slide size on the synthetic matrix, STARE per task, VALIS at its defaults (`--valis`, default `valis120`, solid) and at the high-resolution example of its documentation (`--valis-hires`, default `valis120_hires`, dotted), the cohort's slides overlaid, the node line dashed (`--node-gb 200` for the paper). Not in `make_figures.sh` (it reads the synthetic run's tables too). `regbench plots fig2c_cost_vs_input` draws the same panel from `paper/` | `<synthetic>/tables/resources.csv`, `tables/resources.csv` |
| `fig_stare_seam.py` | Fig. 2b: one round pair after STARE, an inset across a tile boundary and single-cell insets. Not in `make_figures.sh` (the case and tile corner are chosen by hand) | cases, the STARE result's written image |
| `regbench plots figS_synth_cost_scaling` (`benchmarks/regbench/paper/cost_plots.py`, not a script here) | Extended Data Fig. 4: wall-clock time on the synthetic matrix, log-log, the fitted exponent per run (its wall panel). Its x axis is the moving image in gigapixels (area), not px per side | `paper/` of the synthetic set |
| `fig_cohort_scores.py` | An earlier Fig. 3b–e (78 pairs, centroid distance against four; submitted: `fig3bcde_benchmark.py`): Dice, centroid distance, peak RSS, CPU time per method; paired Wilcoxon vs STARE, Holm-corrected | `tables/cells.csv`, `tables/summary_resources.csv` |
| `fig_crop_grid.py` | Fig. 3a: one patient's rounds, the same crop unregistered and after each method, Dice per crop | cases, results' written images, `warped.npz` |
| `fig_crop_grid.py` (`make_figures.sh crops_all`) | Extended Data Fig. 5: the same, rows before / VALIS / STARE, one column per moving round | as Fig. 3a |
| `fig_overlay.py` | Fig. 1c (its "before" file: the overview, the three fields boxed) and Fig. 4b (before and after STARE): a case's DAPI, three zooms: translation, rotation, deformation | cases, one result's `warped.npz` and images |
| `fig_by_case_round.py` (`by_case_round`) | An earlier Extended Data Fig. 6 (submitted: `ed_fig6_by_case_round.py`): Dice (a, b) and residual centroid distance (c, d, log) per method, grouped by case and by moving round (its markers); `by_case_round_medians.csv` holds each group's medians and best method, to check the legend's "8 of 8 cases" | `tables/cells.csv`, the cases' `case.json` |
| `fig_seg_backends.py` (`seg_backends`) | Extended Data Fig. 7: one region segmented by each backend (rows), whole-cell outlines yellow and nuclear outlines cyan over DAPI, 50 µm bars; b, the pairwise Dice of the whole-cell masks (the polygons filled) | the slide, one `geojson/<patient>_cells.geojson` per backend run |
| `fig_segmentation.py` | Fig. 4c: a slide's DAPI with the pipeline's StarDist outlines in a zoomed inset | `pyramid/<patient>.ome.tif`, `geojson/<patient>_cells.geojson` |

## How each choice is made (all deterministic, all pinnable)

- **Box-plot jitter:** `--seed` (default `0`), the only randomness in the package.
- **Fig. 3a patient** (`CROP_PATIENT`): among the patients with at least `CROP_COLUMNS`
  rounds, the one whose rounds lie closest to the cohort median Dice of every row's method.
  The distance is the sum of |Dice − median| / IQR, averaged over the patient's rounds, and
  ties go to the smallest id. So the patient is typical, not the best. The ranking is
  written to `crops_<patient>_patient_choice.csv`.
- **Fig. 3a crop** (`CROP_AT`): the `CROP_UM` square holding the most reference nuclei.
  Ties go to the top-left. The crop's Dice is the harness's `dice_matched`, using the pairs
  matched on the whole slide whose reference nucleus lies inside the crop. The whole-case
  value is printed beside it as a check against `cells.csv`.
- **Fig. 4b case** (`OVERLAY_CASE`): the case closest to the median Dice of
  `OVERLAY_MEDIAN_OF`, by the same rule; the choice is in `_case_choice.csv`.
- **Fig. 4b zooms** (`OVERLAY_ZONES`): the displacement field of `OVERLAY_ZONES_FROM` is
  fitted with a rigid transform (Kabsch) per window. Translation = |mean shift|,
  rotation = |angle| × half the window, deformation = RMS residual. The zoom for each kind
  is the window where that term most exceeds the other two. Only tissue-rich windows count
  (at least the median cell count), and the three windows do not overlap. Every candidate
  window is listed in `_candidates.csv`.
- **Fig. 4c zoom** (`SEG_ZOOM_AT`): the window holding the most cells. Ties go to the
  top-left.

## Display: state this in the legends

The overviews are linearly stretched between the 0.5 and 99.8 tissue percentiles. Each
zoom is processed per channel: a linear stretch between the 1st and 99.8th percentiles
inside the zoom, CLAHE (clip limit 0.01), an unsharp mask (radius 1 px, amount 0.6), then
3× cubic upsampling. Suggested wording: *"Zoomed panels: per-channel contrast stretch
(1st–99.8th percentile), adaptive histogram equalisation and unsharp masking, for display
only; all scores were computed on the unprocessed images."* Overlay colours: reference
cyan, moving magenta; where the two agree the pixel is white or grey.

## Not here

The phenotyping and differential-expression figures (Fig. 6) are not here; they need
the FlowPath and DESeq2 inputs. Schematics are not here either.

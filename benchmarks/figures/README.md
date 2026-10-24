# benchmarks/figures — the paper's data figures, from the results

Every data panel is made by a script here, from the regbench results (`REGBENCH_OUT`) and
one pipeline run. Schematics are drawn by hand. The panels are composed in Affinity.
Each script also writes every panel as its own PNG, plus a `_source_data.csv` holding
the plotted values and every choice made.

```bash
pip install -r requirements/ci.txt        # matplotlib, pandas, scikit-image, STARE (pinned)
cd <regbench root>                         # where regbench.env sets REGBENCH_CASES / _OUT
bash <checkout>/benchmarks/figures/make_figures.sh             # all of them
bash <checkout>/benchmarks/figures/make_figures.sh cohort crops # some of them
```

Every choice (variants, cases, zoom sizes, display, seed) sits in one block at the top of
`make_figures.sh`. The paper's figures are the ones made from that file as committed. Any
one choice can be overridden for a single run from the environment
(`VALIS=valis120_micro bash ...`).

These scripts do NOT run in the scorer's image, which has no matplotlib or pandas. They
import `benchmarks.regbench` from this checkout, so every score is computed by the
harness's own code.

| Script | Reproduces | Reads |
|---|---|---|
| `fig_synthetic_cost.py` | Peak RSS and CPU time vs input size on the synthetic set, VALIS vs STARE | `tables/summary_resources.csv`, `tables/resources.csv` |
| `fig_cohort_scores.py` | Fig. 3b–e: Dice, centroid distance, peak RSS, CPU time per method; paired Wilcoxon vs STARE, Holm-corrected | `tables/cells.csv`, `tables/summary_resources.csv` |
| `fig_crop_grid.py` | Fig. 3a: one patient's rounds, the same crop unregistered and after each method, Dice per crop | cases, results' written images, `warped.npz` |
| `fig_overlay.py` | Fig. 4b: a case's DAPI before (and after) registration, three zooms: translation, rotation, deformation | cases, one result's `warped.npz` and images |
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

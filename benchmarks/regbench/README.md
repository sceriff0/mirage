# regbench: STARE against VALIS and DeeperHistReg

One benchmark, five datasets, the same scoring for every method:

| dataset | what it is | scored by |
|---|---|---|
| `synthetic` | nuclear slides warped by a **known** map (rigid, smooth, mixed, noise, lost nuclei, sparse tissue, dim rounds, a smaller moving slide) | landmark TRE against the truth **and** cell Dice / displacement |
| `semisynth` | windows of a **real** nuclear image warped by the same known maps | the same, on real texture |
| `anhir` | the 230 ANHIR training pairs (brightfield, mixed stains); the 251 hidden pairs for a server submission | the challenge's landmark metrics (rTRE, robustness, rank) |
| `hyreco` | HyReCo: 54 sections stained, scanned, re-stained and scanned again, with manual landmarks | landmark TRE in µm; the only real same-section data with independent truth |
| `multiplex` | your own multiplex-IF rounds, from a manifest CSV | matched Dice and centroid displacement of nuclei, as mirage's `reg_qc=2`: self-consistency, not accuracy |

**Two tiers, never pooled.** `summary.md` opens with the **headline** tier, the cases with a
published basis, and reports everything else as **secondary**:

| tier | what is in it | why |
|---|---|---|
| headline | `hyreco` | real same-section pairs with manual landmarks |
| headline | `semisynth`, families `grid` and `multiscale` | real texture; deformation parameters from published generators and measured residuals (see "Real images under a known map") |
| headline | `multiplex` (cell Dice and displacement) | your data; self-consistency, read against `dice_null` |
| secondary | `synthetic` (Gaussian-blob slides, all families, the scale suite) | image model and most amplitudes are this benchmark's own choices; kept for controlled tests and for cost against size |
| secondary | `semisynth`, families `wave`, `bumps`, `seams` | parameters without a source; stress tests |
| secondary | `anhir` training pairs | public landmarks, so not blind; the blind score is the challenge server's |

Cell Dice is also reported on the headline `semisynth` cases, where the `truth` row gives its
ceiling. Make those cases from your own slides and the ceiling applies to your tissue.

and, for every run, **what it cost**: wall time, CPU time and peak memory of the whole process
tree, GPU memory, and the scheduler's own accounting as a cross-check.

It lives on the `benchmarking` branch and is a harness, not part of the `stare` package.
`benchmarks/anhir/` (the earlier STARE-only ANHIR harness) is kept and reused: this package
imports its cover-table, landmark and proxy code.

## Methods

| `--method` | result names | what runs |
|---|---|---|
| `stare` | `stare_rigid`, `stare` | this checkout's `stare register` (COARSE anchor alone; anchor + mesh) |
| `valis` | `valis_rigid`, `valis`, `valis_micro` | VALIS 1.2.0 from PyPI (`valis-wsi`), upstream defaults, fixed reference; then `register_micro()` |
| `deeperhistreg` | `deeperhistreg` | DeeperHistReg 1.0.1, `default_initial_nonrigid` |
| `initial` | `initial` | no registration; always scored, never needs running |

**About "the ANHIR winner".** The top of the live ANHIR leaderboard is closed: its first 15
entries link neither code nor a paper (leaderboard, anhir.grand-challenge.org
`[PARTIAL: top 31 of 74 rows read by a research agent on 2026-10-07; the rest not read]`).
The best-ranked entry with installable code is "DeeperHistReg - NR 1024" at median rTRE
0.00172, tied with "VALIS 1.0.0rc7" at 0.00172 `[same source, same caveat]`. DeeperHistReg is
the successor of the AGH entry that won the 2019 challenge, so it is what this benchmark runs
as the ANHIR reference method, but it is a tie with VALIS, not a win. If you meant another
method, adding it is one file (see *Adding a method*).

Two configurations of one method can sit side by side with `--label`, e.g. VALIS with its
micro-rigid stage (off upstream, on in mirage), or a vendored VALIS:

```bash
python -m regbench run ... --method valis --label valis_mr --opt micro_rigid=1
python -m regbench run ... --method valis --label valis_mirage --opt valis_src=/path/holding/a/valis/package
```

`valis_src` wants a folder that contains a package directory named `valis`; mirage's vendored
copy is `valis_lib/`, so point it at a folder with a `valis -> .../mirage/valis_lib` symlink,
inside an environment that has that version's dependencies (mirage's VALIS container).

## The contract

Each method only has to provide a **point map from moving-image pixels to reference-image
pixels**. That is enough to score everything:

- ANHIR landmarks and synthetic probes are points;
- nuclei are segmented on each slide's **native** image, and only the moving slide's polygon
  vertices are warped. No method is scored on pixels it interpolated itself.

```
<cases>/<dataset>/<case>/case.json     the two images, pixel size, group, shapes
<cases>/<dataset>/<case>/points.npz    points to warp, and the truth
<out>/<variant>/<dataset>/<case>/warped.npz   the same points in the reference frame
<out>/<variant>/<dataset>/<case>/run.json     ok, error, resources, versions, host
<out>/tables/                                  everything the scorer writes
```

So each method runs in its own environment and the scorer needs none of them.
`case.json` stores absolute paths: prepare and run on the same filesystem.

## Quick start (laptop, no data needed)

```bash
git switch benchmarking
export PYTHONPATH=src:benchmarks
python -m regbench prepare synthetic --suite test --cases /tmp/rb/cases   # 4 cases, 1024 px
for i in 0 1 2 3; do python -m regbench run --cases /tmp/rb/cases --out /tmp/rb/out \
    --dataset synthetic --method stare --index $i --workers 4; done
python -m regbench score --cases /tmp/rb/cases --out /tmp/rb/out
```

`score` prints `tables/summary.md`. Suites: `test` (4 cases, seconds), `ci` (4 cases, 1536 px),
`full` (25 cases at 4096 px: rotations to 180°, amplitudes to 60 px, noise, dropout, sparse
tissue, dim rounds) and `scale` (below). Use one suite per `--cases` folder.

## The scaling test: 1024 px to 65536 px, one command

```bash
git clone -b benchmarking https://github.com/sceriff0/stare && cd stare
export REGBENCH_CONDA_STARE=stare                       # see Environments
export REGBENCH_SIF_VALIS=$HOME/valis.sif
export REGBENCH_VENV_DEEPERHISTREG=$HOME/envs/dhr
export REGBENCH_SBATCH_DEEPERHISTREG="--gres=gpu:1 --partition=gpu"   # optional
benchmarks/regbench/slurm/submit_scale.sh -- --partition=cpu --constraint=<one-node-type>
```

That is everything: no data to download. It generates seven synthetic pairs, 1024, 2048,
4096, 8192, 16384, 32768 and 65536 px a side (1 Mpx to 4.3 Gpx per slide), all with the
**same** deformation (1° rotation, a (30, −20) px shift, a 10 px smooth warp of 1500 px
wavelength), so the only thing that changes down the suite is the slide. Each size is one job
that writes the pair, then one job per method that waits for it, then a scorer.

| size (px) | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|---|---|
| memory | 8G | 8G | 16G | 32G | 64G | 128G | 256G |
| time limit | 0:30 | 0:30 | 1:00 | 2:00 | 4:00 | 12:00 | 24:00 |

Every method gets the same allocation at a given size (`REGBENCH_CPUS`, default 8, each).
`--sizes "1024 2048 4096"` runs a subset, `--methods "stare valis"` picks methods,
`--mem-scale 2` / `--time-scale 2` multiply the table. Results go to
`$SCRATCH/regbench_scale/results-<commit>/tables/`: `summary.md` has a **Scaling** section and
`scaling.csv` one row per (size, method) with TRE, Dice, wall, CPU, peak memory and seconds
per megapixel.

A method that runs out of memory or time at some size is a **result**: its `run.json` says
"killed before finishing", `slurm_accounting.psv` has the scheduler's verdict (`OUT_OF_MEMORY`,
`TIMEOUT`), and the table shows it as failed at that size. For a preview while jobs are still
running, run `python -m regbench score --cases ... --out ...` at any time (the submit script
prints the exact command).

The slides are written tile by tile from the seed, so generating the 65536 px pair needs
about 0.5 GB, not the slide in memory (measured: 0.34–0.45 GB at 1024–4096 px). The reference
keeps all its cells; the moving slide's cells are sampled to 200 000 for the cell metrics.
Measured here for 1024 / 2048 / 4096 px with STARE: 0.14 / 0.13 / 0.12 px median TRE. The
sizes from 8192 px up have not been run anywhere yet.

## Environments

| method | install | notes |
|---|---|---|
| STARE, `prepare`, `score` | `pip install -e ".[test]" psutil` | NumPy / SciPy / scikit-image / tifffile only |
| VALIS | `pip install valis-wsi==1.2.0 "pyvips[binary]" psutil`, or the image `cdgatenbee/valis-wsi:1.2.0` | Python 3.9 or 3.10. The pip route is what was run here; the image tag is from VALIS's docs and was not pulled |
| DeeperHistReg | `pip install torch torchvision` (your CUDA or CPU index), then `pip install -r benchmarks/regbench/requirements-deeperhistreg.txt` | its PyPI package declares no dependencies, hence the file. Runs on CPU; a GPU is strongly advised for real slides |

`psutil` is optional on Linux (the monitor reads `/proc` without it).

## On the cluster

```bash
git clone -b benchmarking https://github.com/sceriff0/stare && cd stare
export REGBENCH_REPO=$PWD
export REGBENCH_CASES=$SCRATCH/regbench/cases
export REGBENCH_OUT=$SCRATCH/regbench/results           # default: ~/regbench_results/stare-<commit>

# one environment per method: REGBENCH_SIF_<M>, REGBENCH_CONDA_<M> or REGBENCH_VENV_<M>
export REGBENCH_CONDA_STARE=stare
export REGBENCH_SIF_VALIS=$HOME/valis.sif               # apptainer pull valis.sif docker://cdgatenbee/valis-wsi:1.2.0
export REGBENCH_VENV_DEEPERHISTREG=$HOME/envs/dhr
export REGBENCH_SBATCH_DEEPERHISTREG="--gres=gpu:1 --partition=gpu"

# 1. prepare (once; idempotent)
S=benchmarks/regbench/slurm
sbatch -c 8  --mem=32G -t 2:00:00 $S/prepare.sbatch synthetic --suite full
sbatch -c 16 --mem=32G -t 4:00:00 $S/prepare.sbatch anhir --data-root /path/to/anhir
sbatch -c 16 --mem=96G -t 8:00:00 $S/prepare.sbatch multiplex --manifest rounds.csv --diameter 24

sbatch -c 8  --mem=32G -t 1:00:00 $S/prepare.sbatch synthetic --suite dev
sbatch -c 8  --mem=32G -t 2:00:00 $S/prepare.sbatch semisynth --image /data/slide.ome.tif --channel 0 --diameter 24
sbatch -c 4  --mem=32G -t 24:00:00 $S/prepare.sbatch hyreco --data-root /data/HyReCo-Additional

# 2. choose the competitors' options on the dev_* cases (before any test result exists)
$S/submit_calibrate.sh -- --partition=cpu --constraint=<one-node-type>

# 3. when the lock job is done: run every method on every dataset, both arms, then score
$S/submit.sh -- --partition=cpu --constraint=<one-node-type>
```

### Two arms per competitor

| arm | results named | options |
|---|---|---|
| default | `valis*`, `deeperhistreg` | each package's defaults |
| recommended | `valis_rec*`, `deeperhistreg_rec` | what `regbench calibrate` locked, else the package's documented higher-accuracy setting (VALIS micro-registration at 25 % of the slide's long side; DeeperHistReg `default_initial_nonrigid_high_resolution`) |

STARE has one arm, its released defaults: nothing about it is chosen on data the benchmark
holds. Calibration (`regbench/calibrate.py`) gives each competitor the same budget, six
candidates fixed in the source, run on `dev_*` cases whose seeds and image windows no scored
case shares; the winner is the lowest mean per-case median rTRE. Both arms are always
reported, so the effect of the choice is visible. `submit.sh --arm default` skips step 2.
The scorer leaves `dev_*` cases out of every table.

### Real images under a known map (`semisynth`)

`prepare semisynth --image X` takes the `--windows` (8) windows of `--n` (4096) px with most
tissue, plus one development window, and pairs each with five deformation families. Two are
headline:

- `grid`: a 9 × 9 grid of random control-point moves, SD drawn from 3-7 px (Casamitjana et
  al., arXiv:2104.14873 `[PARTIAL: synthetic-dataset sections only]`).
- `multiscale`: random fields at 1/8, 1/16 and 1/32 of the window, the resolutions SynthMorph
  sums (arXiv:2004.10282 `[PARTIAL: method text and ablation; Table I not extracted]`), scaled
  to an RMS displacement of `--residual-um` (2 µm). That is the non-rigid residual measured
  between two scans of one section: median 1.6 µm (Lotz et al., arXiv:2106.13150 `[PARTIAL:
  abstract, intro, sec. 4.2, Tables 2-3]`) to 1.9 µm (Muhlich et al. 2022, PMC9525007
  `[PARTIAL: main text and methods through an extractor]`). The equal split of energy between
  the three scales is this benchmark's choice: nobody has measured that spectrum. The slide
  must carry a pixel size, or give `--pixel-size-um`.

`wave`, `bumps` and `seams` are secondary. The reference is the untouched window; the moving
slide is the same image through the known map, with fresh noise at `--noise` times the
image's own background noise. Run it once per slide; give a slide from each tissue or
scanner you care about. With `--mov-image Y`, a different round of the same section that is
already registered to `X`, the moving pixels are real and independent; the truth then
includes that upstream registration's residual, which is a floor under every error.

### HyReCo

Download from IEEE DataPort (doi:10.21227/pzj5-bs61; free login) and unpack so that one
folder holds `HE/<case>.tif|.csv` and `PHH3/<case>.tif|.csv`. Landmarks are in millimetres
and are converted with the TIFF's resolution tags or `--pixel-size-um`; a case whose
landmarks fall outside its image is refused. The layout and units are from the dataset page
`[PARTIAL: DataPort page read through a summariser; no file was opened]`, and **this adapter
has not been run on the real files**, only on synthetic stand-ins: check the first prepared
case before submitting the array. `--ref-dir` / `--mov-dir` point it at any two stain folders.

### ANHIR's hidden pairs

```bash
python -m regbench prepare anhir --data-root /path/to/anhir --status evaluation --cases $REGBENCH_CASES
# ... run the methods ...
python -m regbench anhir-submission --cases $REGBENCH_CASES --out $REGBENCH_OUT \
    --data-root /path/to/anhir --variant stare --dest submission_stare
```

writes `registration-results.csv` and one warped-landmark CSV per case. The column names
follow the BIRL convention from memory (`[MEMORY]`; the challenge's submission page did not
load on 2026-10-08): check them, and whether the server still accepts uploads, before
relying on this. Training-pair numbers are not a blind test: their landmarks are public.

`submit.sh` submits one array per (dataset, method) and a scorer that waits for all of them.
Every method gets the **same** allocation (`REGBENCH_CPUS`=8, `REGBENCH_MEM`=64G,
`REGBENCH_TIME`=04:00:00). Whole slides want more of each: set them before submitting.
`REGBENCH_OPTS_<M>` passes method options, `REGBENCH_LABEL_<M>` renames a configuration.

ANHIR's layout and downloads (the landmark archive needs a challenge login) are described in
`benchmarks/anhir/README.md`; run its `join` step first.

### Your multiplex data

`rounds.csv`, one row per (reference round, moving round):

```csv
case_id,reference,moving,group,ref_channel,mov_channel,pixel_size_um,ref_cells,mov_cells
P01_r2,/data/P01/round1.ome.tif,/data/P01/round2.ome.tif,P01,0,0,,,
P01_r3,/data/P01/round1.ome.tif,/data/P01/round3.ome.tif,P01,0,0,,,
```

Only the first three columns are required. Give the slides **before** registration.
`ref_channel` / `mov_channel` is the nuclear channel's index. `pixel_size_um` defaults to the
OME header. `ref_cells` / `mov_cells` are optional segmentations of the *native* images (cell
GeoJSON or integer label TIFF): pass the masks your pipeline ships to score its cells.
Without them a built-in threshold + watershed segmenter runs tile by tile; tell it the nuclear
diameter in pixels (`--diameter`). It is a plain classical segmenter, good enough to pair
nuclei, not a replacement for StarDist / Cellpose masks. `--max-cells N` scores a random
sample of moving cells. A reference shared by several rounds is extracted and segmented once.

## What is measured

**Landmarks** — the ANHIR definitions (anhir.grand-challenge.org/Performance_Metrics/
`[FULL per the research agent's read on 2026-10-07; not re-read by the author of this file]`):
TRE is the distance between a warped landmark and its target, rTRE = TRE / reference image
diagonal; per case the median, mean and max rTRE; robustness = fraction of landmarks brought
closer than the unregistered pose; rank = the method's rank on that case by median rTRE. A
case a method failed on, or was never run on, counts at the initial pose (`imputed_initial`).
`common` is the subset every method registered. The challenge's own "average rank" ranks
against the other leaderboard entries, so ranks here are only among the methods run here.

**Cells** — mirage's full-transform record (`bin/warp_seg_qc.py::score_full_transform`,
`bin/utils/cell_pairs.py`, read in the mirage checkout), restated in `regbench/cells.py`:
cells are paired one-to-one by minimum total centroid distance within 1.5 median nuclear
radii; each pair is rasterised in its own window for an IoU;
`dice_matched = 2 Σ intersection / Σ (area_ref + area_mov)` over the pairs; displacement is
the distance between paired centroids (px, and µm when the pixel size is known). **Read Dice
together with `pair_fraction`**: a registration that is off by more than the match radius
pairs few cells, or pairs neighbours by chance. One simplification against mirage: a cell is
one outer ring (holes dropped, a MultiPolygon keeps its largest part).

On real slides this is **self-consistency**: it shows the two segmentations overlap after
warping, on the channel the methods registered, with no independent truth. Two references
say how to read it:

- `dice_null`, `pair_fraction_null`: the same score with the warped cells shifted four
  nuclear radii, i.e. what chance pairing gives in that tissue. A Dice near it means nothing.
- the `truth` row (synthetic and semisynth): the moving cells under the true map. Its Dice
  is the ceiling for that pair, below 1 because of noise, resampling and lost cells.

**Regularity** — every method also warps a regular lattice of moving points (at most 512 a
side). `fold_pct` is the share of it where the Jacobian determinant of the map is not
positive, i.e. where the map folds, and `sd_log_jac` the spread of its logarithm: the two
numbers registration papers report beside accuracy. A method can buy overlap with an
implausible map; these show it. `frac_not_improved` is the share of cases whose median error
is not below the unregistered pose, failed runs included, and `med_p90_tre_um` the median
over cases of the 90th-percentile error in µm.

**Uncertainty** — every headline aggregate has a 95 % bootstrap interval over cases
(`*_lo`, `*_hi`), and `tables/pairwise.csv` compares every two methods case by case with a
two-sided Wilcoxon signed-rank test, Holm-corrected within a dataset and metric. With fewer
than six paired cases no p-value is given.

**Deformations** — besides the two-wave field, `multiscale` (three random fields of
correlation length 2000 / 500 / 125 px), `grid` (9 × 9 random control points, SD 3-7 px),
`bumps` (local Gaussian bumps of width 25, 100 and 400 px) and `seams` (a shift per 1024 px
tile, SD 1.5-6 px, discontinuous). Bump peaks are 0.2-0.6 of their width, at most 50 px, so
the map does not fold; the 25 px bumps are finer than STARE's 64 px lattice on purpose.

**Resources** — `regbench/resources.py`. A sampler thread walks the process tree five times
a second, because a method's work is not one process (STARE's tile pool, VALIS's JVM and
native threads, torch):

| column | meaning |
|---|---|
| `wall_min` | elapsed, from the call into the method until the variant exists. Includes reading images and any conversion the method needs; excludes warping the benchmark's points (the challenge's rule too) |
| `cpu_min`, `cores_used`, `cpu_efficiency` | user + system CPU over the whole tree, exited children included; the mean cores busy; that over the cores it was told to use |
| `peak_rss_gb` | largest simultaneous sum of resident memory over the tree |
| `cgroup_peak_gb` | the job cgroup's own peak, where the kernel exposes it: the number SLURM enforces `--mem` against, so the one to size jobs by. Covers the whole task |
| `gpu_peak_gb` | torch's peak allocated CUDA memory |
| `wall_s_per_mpx`, `cpu_s_per_mpx` | per megapixel of the two slides together, so slides of different size compare |
| `tables/slurm_accounting.psv` | `sacct` for every run job (Elapsed, TotalCPU, MaxRSS, ...): the scheduler's independent account |

What makes the comparison fair, and what does not:

- Every method gets the same CPUs and memory, and the thread pools are set to match how each
  parallelises (`workers` processes × 1 BLAS thread for STARE; `workers` threads for the rest).
- Timings compare methods only on **one node type**: pass `--constraint`. `cpu_model` and
  `host` are recorded per run so a mixed run is visible.
- `stare_rigid` and `stare` come from one registration, as do the three VALIS variants: a
  variant's cost is the cost of *reaching* it. STARE's rigid stage is not billed separately.
- Summed RSS counts pages shared between processes once per process, so it is an upper
  bound; `cgroup_peak_gb` is exact. With neither `psutil` nor `/proc` (macOS without psutil)
  only a lower bound exists and `rss_method` says `rusage`.
- A GPU method's `cpu_min` is not its whole cost. Compare it on wall time and report the GPU.
- Failed runs are excluded from the resource medians and counted in `n_failed`.

## Tests

```bash
PYTHONPATH=src:benchmarks pytest benchmarks -q
```

runs STARE for real on synthetic slides and checks the scorer against known answers and
against mirage's own code: the truth's direction (moving nuclei pushed through the generator's map land on reference
nuclei), sub-pixel STARE error, imputation of a failed method, resources of a child process,
tile-wise segmentation without duplicates, the ANHIR and multiplex layouts.
`tests/test_external_methods.py` runs VALIS and DeeperHistReg and is skipped where they are
not installed. To run it in a method's environment without STARE's I/O stack:

```bash
python -m regbench prepare synthetic --suite test --cases /tmp/tc      # STARE environment
REGBENCH_TEST_CASES=/tmp/tc PYTHONPATH=src:benchmarks <method-env>/bin/python -m pytest \
    benchmarks/regbench/tests/test_external_methods.py
```

`.github/workflows/benchmark.yml` does all of this on every push to `benchmarking`: the
harness tests, then VALIS and DeeperHistReg installed from PyPI and run on the `test` suite,
with the comparison table as the job summary.

## Exact implementations

Each method is the published package, called through its own entry point, with its own
defaults. Nothing is re-implemented, and every run records what was called in
`run.json` → `info.implementation` (package, version, source path, entry point, every setting
that differs from the defaults, the point-warp function, and what the benchmark did around
it):

| method | registration | point warp | differs from the package's defaults |
|---|---|---|---|
| STARE | `stare.cli.register`, this checkout | `stare.stage_warp.make_warper` | nothing; STITCH is skipped (points are scored) |
| VALIS | `registration.Valis(...).register()`, then `.register_micro()` | `Slide.warp_xy_from_to` | `reference_img_f` + `align_to_reference=True` (a fixed reference) |
| DeeperHistReg | `deeperhistreg.run_registration` with `configs.default_initial_nonrigid()` | the package's `dhr_utils.warping.warp_landmarks` | loader `pil`, resample ratios 1.0 (inputs are already ≤ 8192 px), `save_final_images=False`, the device |

What the benchmark adds around them is listed under *Adapter notes*; the one that changes
what a method computes is DeeperHistReg's swapped roles. DeeperHistReg's ANHIR leaderboard
configuration ("NR 1024") is not published in its repository as far as was read, so its
numbers here are the package defaults, not a reproduction of that entry.

The scoring is held to the same standard: `tests/test_mirage_parity.py` runs mirage's own
`bin/utils/cell_pairs.py` on the same polygons and requires identical pairs, IoU, Dice and
displacement (it passes; skipped where no mirage checkout is found, set `MIRAGE_REPO`).

## Things the benchmark already found

From the synthetic suites on 2026-10-07/08 (macOS arm64, CPU). Check them before quoting.

1. **STARE 1.2.1's SOLVE can flatten the mesh to nothing (`s = 1e6`), leaving the rigid
   anchor.** `s` is chosen by block cross-validation before any outlier rejection, and two
   things push that choice to a flat field:
   - *Windows the moving slide does not cover.* REG-TILE emits full-weight vectors for
     lattice windows with no moving pixels under them (measured residuals 7–35 px against
     ~3 px elsewhere). No `s` predicts them, a flat field predicts them least badly (CV loss
     17.1 at `s = 1e6` against 23.6 at `1e-4`), and that choice applies to the whole slide.
     With ~10–20 % of the lattice uncovered: 4 of 6 seeds at 1024 px, 2 of 6 at 2048 px.
     Smaller uncovered fractions were not measured.
   - *Small lattices.* Held-out patches on the border must be extrapolated, flexible fits
     extrapolate worse than flat ones, and on a 15 × 15 lattice 44 % of nodes are border:
     2 of 6 seeds at 1024 px with full-size slides, 0 of 14 at 2048 px and above.

   A collapsed run scored 2.5–3.4 px median TRE and Dice 0.74–0.78, against 0.3 px and 0.93
   when the mesh worked. Dice alone does not reveal it, so the tables carry
   `stare_smoothing_s` and `stare_flat_field` per case. Both triggers are pinned by strict
   `xfail`s in `tests/test_pipeline.py`. Whether it happens on real slides is unmeasured; ANHIR
   pairs, which rarely share a shape, are where the first trigger should show.
2. **VALIS 1.2.0 `register_micro()` made the result worse on every synthetic case tried.**
   After it, the moving slide's forward and backward fields were exactly twice the fields
   from `register()` (correlation 1.0 between the "residual" and the existing field), and the
   median TRE went from 0.16 px to 5.1 px on a 4096 px case. That reads as the micro pass
   re-estimating the whole displacement and adding it to itself, but only the symptom was
   measured, not the cause in VALIS's code, and only on synthetic nuclei with the calls in
   `methods/valis.py`. So `valis_micro` is reported as measured; treat `valis` as VALIS's
   result until this is understood. mirage's vendored VALIS is 1.0.0 with a different
   `register_micro`, and was not run here (its dependencies were not installed).

## Adapter notes

- **DeeperHistReg's roles are swapped on purpose.** Its field is a backward map on its
  target grid (`warped(x) = source(x + u(x))`; `dhr_utils/utils.py::np_df_to_pyvips_df`,
  `dhr_utils/warping.py::warp_landmarks` `[PARTIAL: those functions, run.py,
  full_resolution.py, apply_deformation.py, pair_full_loader.py and the default config
  read from the repo's main branch; registration internals not read]`). The benchmark hands
  its reference over as DeeperHistReg's source and the moving slide as its target, so the
  field is the moving → reference map with no inversion. The method therefore registers
  reference onto moving, the opposite of the others; for a symmetric cost that should not
  matter, but it is not the direction its ANHIR entry was run in.
- DeeperHistReg gets 8-bit RGB images of at most 8192 px on the long side (`--opt max_dim`).
  Fluorescence is inverted to dark-on-white, the appearance its preprocessing assumes.
- VALIS gets the nuclear OME-TIFF for fluorescence and the original RGB image for brightfield.
  Points go through `Slide.warp_xy_from_to` into the reference slide's own pixel frame.
- STARE gets one nuclear channel; for ANHIR that is the brightfield proxy (`--proxy lum|hema`),
  which is out of domain for it, as `benchmarks/anhir/README.md` explains.

## Adding a method

Create `regbench/methods/<name>.py` with `VARIANTS = ("<name>",)` and

```python
def register(case, work, opts):
    ...                                   # import the method's software here
    yield "<name>", lambda xy: ..., {"version": ...}   # xy: moving px -> reference px
```

and add the name to `METHODS` in `regbench/__init__.py`. Yield a variant per stage, as soon
as it exists.

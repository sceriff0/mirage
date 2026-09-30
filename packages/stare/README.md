# STARE

**S**calable **T**ile-parallel **A**lignment by **R**obust **E**stimation. Tile-parallel,
JVM-free, non-rigid registration of whole-slide images, for cyclic immunofluorescence and
any other same-section re-imaging where a nuclear channel is shared between rounds.

**3.0.0: the name is STARE again.** Versions 2.x shipped this method as `drape-registration`
(import `drape`, CLI `drape`); 3.0.0 is `stare-registration` (import `stare`, CLI `stare`)
with the same code. There is no `drape` alias: replace `import drape` with `import stare`
and `drape <stage>` with `stare <stage>`. The 0.1.x `stare-registration` was the earlier
method (TRE gate, legacy/robust solvers), retired in 2.0.0. The mirage parameters keep
their names (`registration_method='tiled'`, `reg_tiled_*`) and so do the `TILED_*` processes.

```bash
pip install git+https://github.com/sceriff0/stare   # standalone repository
pip install -e packages/stare                       # or from a mirage checkout

stare register --reference ref.ome.tif --moving mov.ome.tif \
    --out mov_registered.ome.tif --manifest mov_manifest.json --workers 8
```

A slim container with STARE preinstalled is published as `bolt3x/mirage-stare:1.0.0`. It has
no JVM, no GPU and no torch.

The four stages are also individual subcommands (`stare coarse`, `stare reg-tile`,
`stare solve`, `stare stitch`), so a workflow engine can fan the tile stage out across
nodes. That is how the [mirage](https://github.com/sceriff0/mirage) Nextflow pipeline runs it.

**Where development happens.** STARE is developed inside mirage as `packages/stare/`, which
is the single source of truth:
- mirage's `bin/tiled_*.py` scripts are shims over it;
- `tests/test_stare_package_parity.py` in mirage asserts that the pipeline and
  `stare register` produce the same manifest and the same pixels.

The standalone repository [sceriff0/stare](https://github.com/sceriff0/stare) is generated
from that directory with `git subtree split -P packages/stare`, so the two cannot diverge.

## How it works, in brief

1. **COARSE.** A global rigid anchor from an exhaustive rotation sweep with globally
   normalised cross-correlation on tissue-masked thumbnails. It falls back to ORB + RANSAC,
   and refuses rather than guesses.
2. **REG-TILE.** Per tile, displacement vectors on a slide-global lattice of 50 %-overlapping
   windows, following the particle-image-velocimetry recipe:
   - high-pass + Hann-windowed cross-correlation;
   - two passes with window deformation;
   - a 3-point Gaussian sub-pixel fit;
   - peak-ratio and foreground validity.
3. **SOLVE.** Remove a robust affine, then apply Garcia's robust DCT-PLS smoother. The
   smoothing is chosen by blocked cross-validation, with an h-block buffer when the measured
   residual autocorrelation calls for it. Per-vector uncertainties are calibrated on held-out
   residuals.
4. **STITCH.** An interpolating cubic B-spline field, inverted by fixed-point iteration to a
   tolerance, applied by streaming bilinear pull-back resampling.

Every design choice traces to open-access literature read in full, to a derivation, or to a
stated measurement. The literature is PIVlab and OpenPIV, Garcia 2010 and 2011,
De Brabanter et al. 2011, Unser 1999, Behrmann et al. 2019, ASHLAR and SOFIMA.

## Fan-out contract

Each stage is a function with a file contract, so any engine that can run a command
per row can run STARE; the mirage pipeline is one such engine (one Nextflow task per
stage invocation, through the `bin/tiled_*.py` shims).

| stage | reads | writes |
|---|---|---|
| `stare coarse --reference R --moving M --max-dim N --out-m0 M0.json --out-tiles tiles.csv` | the nuclear channel of both slides, decimated | `M0.json` (the global anchor, reference dims, the coarse residual) and `tiles.csv`, one row per tile (`ix iy cx cy x0 y0 x1 y1 rx0 ry0 rx1 ry1`) |
| `stare reg-tile --reference R --moving M --m0 M0.json --plan tiles.csv --row N --out X_ctrl.json` (or the same tile as explicit `--ix --iy --cx --cy --rx0 --ry0 --rx1 --ry1`) | one reference tile and the moving crop its inverse map draws from | one control JSON: the tile's window vectors on the slide-global lattice (`--stride`, window 2 × stride), plus their median displacement, TRE, correlation error and the foreground fractions |
| `stare solve --m0 M0.json --controls 'X_*_ctrl.json' --moving-name M --out-manifest M_manifest.json [--out-tre M_tre.json] [--max-disp D]` | every control JSON the glob matches (each must carry `vectors`; a pre-v2 one-point-per-tile JSON is refused) | the transform manifest (`M0` + mesh + `solver`, always `dctpls`) and the TRE report (with the solve's own report under `"solve"`) |
| `stare stitch --moving M --manifest M_manifest.json --out M_registered.ome.tif --pixel-size P` | the moving slide, tile by tile | the registered OME-TIFF |

`--plan/--row` and the explicit geometry produce the identical control JSON; the row form
exists so a SLURM array job, or any engine that only has an integer index, can address a
tile.

`stare register` maps the same `reg_tile` function over the same rows of the same
`tiles.csv` with a local `multiprocessing` pool (`--workers`), between the same `coarse`
and the same `solve` + `stitch`. Nothing about the math changes with the executor, which is
why its manifest and its pixels equal the pipeline's — `tests/test_stare_package_parity.py`
in mirage asserts exactly that.

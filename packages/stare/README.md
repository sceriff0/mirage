# STARE

Tile-parallel, JVM-free, non-rigid registration of whole-slide images, for cyclic
immunofluorescence and any other same-section re-imaging where a nuclear channel is
shared between rounds.

```bash
pip install -e packages/stare            # from the mirage checkout
pip install -e "packages/stare[anchor-disk]"   # + the learned coarse anchor (torch, kornia)

stare register --reference ref.ome.tif --moving mov.ome.tif \
    --out mov_registered.ome.tif --manifest mov_manifest.json --workers 8
```

The four stages are also individual subcommands (`stare coarse`, `stare reg-tile`,
`stare solve`, `stare stitch`) so a workflow engine can fan the tile stage out across
nodes, which is how the mirage pipeline runs it.

This package lives inside the mirage repository as `packages/stare/` and is the single
source of truth for the method: mirage's `bin/tiled_*.py` are shims over it, and
`tests/test_stare_package_parity.py` asserts the two paths produce the same manifest and
the same pixels. It is extracted to its own repository with
`git subtree split -P packages/stare` when it ships on its own.

## Fan-out contract

Each stage is a function with a file contract, so any engine that can run a command
per row can run STARE; the mirage pipeline is one such engine (one Nextflow task per
stage invocation, through the `bin/tiled_*.py` shims).

| stage | reads | writes |
|---|---|---|
| `stare coarse --reference R --moving M --max-dim N --out-m0 M0.json --out-tiles tiles.csv` | the nuclear channel of both slides, decimated | `M0.json` (the global anchor, reference dims, the coarse residual) and `tiles.csv`, one row per tile (`ix iy cx cy x0 y0 x1 y1 rx0 ry0 rx1 ry1`) |
| `stare reg-tile --reference R --moving M --m0 M0.json --plan tiles.csv --row N --out X_ctrl.json` (or the same tile as explicit `--ix --iy --cx --cy --rx0 --ry0 --rx1 --ry1`) | one reference tile and the moving crop its inverse map draws from | one control JSON: the tile's displacement, TRE, correlation error and foreground fractions |
| `stare solve --m0 M0.json --controls 'X_*_ctrl.json' --moving-name M --out-manifest M_manifest.json [--out-tre M_tre.json] [--solver legacy\|robust]` | every control JSON the glob matches | the transform manifest (`M0` + mesh + `solver`) and the TRE report (with the solve's own report under `"solve"`) |
| `stare stitch --moving M --manifest M_manifest.json --out M_registered.ome.tif --pixel-size P` | the moving slide, tile by tile | the registered OME-TIFF |

`--plan/--row` and the explicit geometry produce the identical control JSON; the row form
exists so a SLURM array job, or any engine that only has an integer index, can address a
tile.

`stare register` maps the same `reg_tile` function over the same rows of the same
`tiles.csv` with a local `multiprocessing` pool (`--workers`), between the same `coarse`
and the same `solve` + `stitch`. Nothing about the math changes with the executor, which is
why its manifest and its pixels equal the pipeline's — `tests/test_stare_package_parity.py`
in mirage asserts exactly that, with `--solver legacy` on both sides.

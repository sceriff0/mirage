"""STARE: Scalable Tile-parallel Alignment by Robust Estimation.

Tile-parallel, JVM-free, non-rigid registration of cyclic-IF whole-slide images.

Four stages, each a function and a CLI subcommand (``stare <stage>``), plus
``stare register`` which runs all four in one process with a local worker pool:

    coarse    one global affine ``M0`` from a decimated nuclear-channel thumbnail,
              and the tile plan
    reg-tile  a grid of window displacement vectors per tile (one per ``stride``
              px), by correlation after the rigid warp -- the fan-out
    solve     the mesh: the vectors on one slide-global lattice, robust affine +
              robust DCT-PLS, fold certificate (``stare.solve``)
    stitch    the streaming inverse-map warp of the moving slide

The mirage pipeline's ``bin/tiled_*.py`` scripts are thin shims over these
functions, so a pipeline run and ``stare register`` produce the same manifest.
"""

__version__ = "3.0.0"

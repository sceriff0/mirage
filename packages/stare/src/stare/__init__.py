"""STARE: tile-parallel, JVM-free, non-rigid registration of whole-slide images.

Four stages, each a function and a CLI subcommand (``stare <stage>``), plus
``stare register`` which runs all four in one process with a local worker pool:

    coarse    one global affine ``M0`` from a decimated nuclear-channel thumbnail,
              and the tile plan
    reg-tile  one residual displacement per tile, by phase correlation after the
              rigid warp -- the fan-out
    solve     the control-grid mesh: gates, neighbour-consistency rejection,
              in-fill, regularised smoothing, invertibility check (``stare.solve``)
    stitch    the streaming inverse-map warp of the moving slide

The mirage pipeline's ``bin/tiled_*.py`` scripts are thin shims over these
functions, so a pipeline run and ``stare register`` produce the same manifest.
"""

__version__ = "0.1.0"

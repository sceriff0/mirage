"""Shim: ``tile_residual`` now lives in the ``drape`` package as ``drape.tile_residual``.

DRAPE's source of truth is ``packages/drape`` (``pip install -e packages/drape``).
This file exists so the pipeline's flat import convention
(``from tile_residual import ...`` after a ``sys.path.insert(0, .../bin/utils)``) keeps
resolving. It replaces itself in ``sys.modules`` with the package module rather
than re-exporting names, so ``import tile_residual`` yields the SAME module object the
stages use -- a test that monkeypatches an attribute on it patches the real one.
"""

import sys

from drape import tile_residual as _impl

sys.modules[__name__] = _impl

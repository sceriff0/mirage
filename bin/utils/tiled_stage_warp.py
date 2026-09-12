"""Shim: ``tiled_stage_warp`` now lives in the ``stare`` package as ``stare.stage_warp``.

STARE's source of truth is ``packages/stare`` (``pip install -e packages/stare``).
This file exists so the pipeline's flat import convention
(``from tiled_stage_warp import ...`` after a ``sys.path.insert(0, .../bin/utils)``) keeps
resolving. It replaces itself in ``sys.modules`` with the package module rather
than re-exporting names, so ``import tiled_stage_warp`` yields the SAME module object the
stages use -- a test that monkeypatches an attribute on it patches the real one.
"""

import sys

from stare import stage_warp as _impl

sys.modules[__name__] = _impl

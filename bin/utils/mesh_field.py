"""Shim: ``mesh_field`` now lives in the ``stare`` package as ``stare.mesh_field``.

STARE's source of truth is ``packages/stare`` (``pip install -e packages/stare``).
This file exists so the pipeline's flat import convention
(``from mesh_field import ...`` after a ``sys.path.insert(0, .../bin/utils)``) keeps
resolving. It replaces itself in ``sys.modules`` with the package module rather
than re-exporting names, so ``import mesh_field`` yields the SAME module object the
stages use -- a test that monkeypatches an attribute on it patches the real one.
"""

import sys

from stare import mesh_field as _impl

sys.modules[__name__] = _impl

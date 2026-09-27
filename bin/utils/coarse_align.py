"""Shim: ``coarse_align`` now lives in the ``drape`` package as ``drape.coarse_align``.

DRAPE's source of truth is ``packages/drape`` (``pip install -e packages/drape``).
This file exists so the pipeline's flat import convention
(``from coarse_align import ...`` after a ``sys.path.insert(0, .../bin/utils)``) keeps
resolving. It replaces itself in ``sys.modules`` with the package module rather
than re-exporting names, so ``import coarse_align`` yields the SAME module object the
stages use -- a test that monkeypatches an attribute on it patches the real one.
"""

import sys

from drape import coarse_align as _impl

sys.modules[__name__] = _impl

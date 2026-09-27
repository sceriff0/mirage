"""Shim: ``tre_report`` now lives in the ``drape`` package as ``drape.tre_report``.

DRAPE's source of truth is ``packages/drape`` (``pip install -e packages/drape``).
This file exists so the pipeline's flat import convention
(``from tre_report import ...`` after a ``sys.path.insert(0, .../bin/utils)``) keeps
resolving. It replaces itself in ``sys.modules`` with the package module rather
than re-exporting names, so ``import tre_report`` yields the SAME module object the
stages use -- a test that monkeypatches an attribute on it patches the real one.
"""

import sys

from drape import tre_report as _impl

sys.modules[__name__] = _impl

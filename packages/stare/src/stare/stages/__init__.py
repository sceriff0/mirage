"""The four STARE stages, one module each, each exposing ``main(argv) -> int``.

Importing this package sets the cache-directory environment the stages ran with
as pipeline scripts, BEFORE any of them imports numpy/scikit-image: a cluster
whose ``$HOME`` is a read-only mount turns a numba or matplotlib cache write
into a crash far from its cause. ``setdefault`` for the paths, so an operator's
own value wins; caching itself is disabled outright, which is what the pipeline
scripts always did.
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

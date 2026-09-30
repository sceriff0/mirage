#!/usr/bin/env python3
"""Shim: the STARE ``stitch`` stage now lives in ``stare.stages.stitch``.

STARE's source of truth is ``packages/stare`` (``pip install -e packages/stare``);
this file is what the Nextflow module invokes by name, so it keeps its shebang and
its executable bit. When imported (the tests do ``import tiled_stitch``) it replaces
itself in ``sys.modules`` with the stage module, so a monkeypatch on
``tiled_stitch.<name>`` reaches the function the stage actually calls.
"""

import sys

from stare.stages import stitch as _impl
from stare.stages.stitch import main

if __name__ != "__main__":
    sys.modules[__name__] = _impl

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shim: the STARE ``coarse`` stage now lives in ``stare.stages.coarse``.

STARE's source of truth is ``packages/stare`` (``pip install -e packages/stare``);
this file is what the Nextflow module invokes by name, so it keeps its shebang and
its executable bit. When imported (the tests do ``import tiled_coarse``) it replaces
itself in ``sys.modules`` with the stage module, so a monkeypatch on
``tiled_coarse.<name>`` reaches the function the stage actually calls.
"""

import sys

from stare.stages import coarse as _impl
from stare.stages.coarse import main

if __name__ != "__main__":
    sys.modules[__name__] = _impl

if __name__ == "__main__":
    raise SystemExit(main())

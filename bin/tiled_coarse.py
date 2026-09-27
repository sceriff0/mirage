#!/usr/bin/env python3
"""Shim: the DRAPE ``coarse`` stage now lives in ``drape.stages.coarse``.

DRAPE's source of truth is ``packages/drape`` (``pip install -e packages/drape``);
this file is what the Nextflow module invokes by name, so it keeps its shebang and
its executable bit. When imported (the tests do ``import tiled_coarse``) it replaces
itself in ``sys.modules`` with the stage module, so a monkeypatch on
``tiled_coarse.<name>`` reaches the function the stage actually calls.
"""

import sys

from drape.stages import coarse as _impl
from drape.stages.coarse import main

if __name__ != "__main__":
    sys.modules[__name__] = _impl

if __name__ == "__main__":
    raise SystemExit(main())

"""Private method plug-ins: optional methods kept OUTSIDE the repository.

A method that cannot be published (unreleased code, no licence) is benchmarked through
a plug-in: a folder ``benchmarks/regbench/private/`` that git ignores, holding a
``plugin.py``. It is read only when ``REGBENCH_PRIVATE=1``. With the variable unset, or
the folder absent, nothing is imported and the run list, the counts, the tables and every
test are exactly what they are without it. ``REGBENCH_PRIVATE_DIR`` names another folder
(the tests use a temporary one).

What ``plugin.py`` may define, every name optional:

``METHODS``            ``{method name: module path, or a module-like object}``: a
                       regbench method (``regbench.methods``)
``RUNS``               a list of dicts with the fields of ``variants.Run``
``FAMILY``             ``{result-name prefix: family}``
``SEAM_OWNERS``        result names whose seam mask the scorer applies to every variant
``SELFCHECK_SIZES``    ``{method or label: an entry of selfcheck.SIZES}``
``SELFCHECK_OPTS``     ``{run name: {option: value}}`` added to the run's options for
                       its selfcheck only
``SELFCHECK_GATED_BY`` ``{run name: the run whose selfcheck checks it}``: reported,
                       never a gate (the key keeps its older name)
``COST_COLUMNS``       ``{column: key of run.json's info}``: extra columns of
                       ``resources.csv``

An image is named like any other: ``REGBENCH_SIF_<image>`` for the ``image`` of a run.
The launcher passes the whole environment to its jobs, so the variables above reach
them.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

ENV, ENV_DIR = "REGBENCH_PRIVATE", "REGBENCH_PRIVATE_DIR"
DEFAULT_DIR = Path(__file__).resolve().parent / "private"
_MODULE = "regbench_private_plugin"
_loaded = {}


def enabled():
    return os.environ.get(ENV) == "1"


def folder():
    return Path(os.environ.get(ENV_DIR) or DEFAULT_DIR)


def load():
    """The plug-in module, or None when the switch is off or no ``plugin.py`` is there."""
    if not enabled():
        return None
    path = folder() / "plugin.py"
    if not path.is_file():
        return None
    key = str(path.resolve())
    if key not in _loaded:
        if path.resolve() == (DEFAULT_DIR / "plugin.py").resolve():
            module = importlib.import_module(f"{__package__}.private.plugin")
        else:
            spec = importlib.util.spec_from_file_location(_MODULE, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE] = module
            spec.loader.exec_module(module)
        _loaded[key] = module
    return _loaded[key]


def get(name, default=None):
    """One declaration of the plug-in, or ``default`` without a plug-in."""
    module = load()
    return default if module is None else getattr(module, name, default)


def method_names():
    return tuple(get("METHODS", {}) or {})


def method(name):
    """The module of a plug-in method, or None when ``name`` is not one."""
    found = (get("METHODS", {}) or {}).get(name)
    if isinstance(found, str):
        return importlib.import_module(found)
    return found

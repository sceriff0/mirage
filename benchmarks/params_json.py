"""Moved to ``benchmarks.regbench.params_json``. This shim keeps
``python3 -m benchmarks.params_json`` working for ``run_arms.sh`` / ``run_sweep.sh``
until they are removed with the arms and the sweep."""

from benchmarks.regbench.params_json import *  # noqa: F401,F403
from benchmarks.regbench.params_json import main

if __name__ == "__main__":
    raise SystemExit(main())

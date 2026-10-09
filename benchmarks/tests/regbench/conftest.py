"""Shared fixtures: the small synthetic ``test`` suite, prepared once per session.

Preparing cases needs STARE's dependencies (it writes and segments OME-TIFFs). A method's own
environment need not have them: set ``REGBENCH_TEST_CASES`` to a folder prepared elsewhere
(``python -m benchmarks.regbench prepare synthetic --suite test --cases DIR``) and the fixtures use it.
"""

import os
from pathlib import Path

import pytest

from benchmarks.regbench.cases import load_case
from benchmarks.regbench.datasets import synthetic

SPECS = synthetic.SUITES["test"]


@pytest.fixture(scope="session")
def cases_root(tmp_path_factory):
    if os.environ.get("REGBENCH_TEST_CASES"):
        return Path(os.environ["REGBENCH_TEST_CASES"])
    root = tmp_path_factory.mktemp("cases")
    for name, spec in SPECS.items():
        synthetic.prepare_one(root, name, spec)
    return root


@pytest.fixture(scope="session")
def synthetic_cases(cases_root):
    return {name: load_case(cases_root / "synthetic" / name) for name in SPECS}

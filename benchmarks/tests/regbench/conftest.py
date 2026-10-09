"""Shared fixtures: the small synthetic ``test`` suite, prepared once per session.

Preparing cases needs STARE's dependencies (it writes and segments OME-TIFFs). A method's own
environment need not have them: set ``REGBENCH_TEST_CASES`` to a folder prepared elsewhere
(``python -m benchmarks.regbench prepare synthetic --suite test --cases DIR``) and the fixtures use it.
"""

import argparse
import os
from pathlib import Path

import pytest

from benchmarks.regbench.cases import load_case
from benchmarks.regbench.datasets import synthetic
from benchmarks.regbench.imageio import write_nuclear

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


@pytest.fixture(scope="session")
def real_like(tmp_path_factory):
    """A slide with tissue in part of it only, standing in for a real nuclear channel."""
    spec = {**synthetic._BASE, "family": "t", "n": 1536, "seed": 8, "keep": 0.5}
    ref, _, _ = synthetic.make_pair(spec)
    path = tmp_path_factory.mktemp("real") / "slide.ome.tif"
    write_nuclear(path, ref, 0.5)
    return path


def semisynth_args(image, cases, **kw):
    """The namespace ``prepare semisynth`` hands the dataset, for a 512 px window."""
    base = dict(
        image=str(image),
        channel=0,
        mov_image=None,
        mov_channel=0,
        name="slide",
        n=512,
        windows=2,
        diameter=synthetic.NUCLEUS_DIAMETER,
        noise=1.0,
        pixel_size_um=0.5,
        residual_um=2.0,
        truth=False,
        cases=str(cases),
        index=None,
        workers=1,
        force=False,
    )
    return argparse.Namespace(**{**base, **kw})

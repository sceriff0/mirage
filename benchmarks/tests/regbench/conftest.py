"""Shared fixtures: the test tree's toy cases (``toy_cases.py``), prepared once per session.

Preparing cases needs STARE's dependencies (it writes and segments OME-TIFFs). A method's own
environment need not have them: set ``REGBENCH_TEST_CASES`` to a folder prepared elsewhere
(``python -m benchmarks.tests.regbench.toy_cases DIR``) and the fixtures use it.
"""

import argparse
import os
from pathlib import Path

import pytest

from benchmarks.regbench import scene
from benchmarks.regbench.cases import load_case
from benchmarks.regbench.imageio import write_nuclear

from . import toy_cases as toy

SPECS = toy.SPECS


@pytest.fixture(scope="module")
def local_stages():
    """For a module that registers with STARE in its own process: the run itself starts
    the stage processes (``REGBENCH_STAGES=local``, ``regbench.methods.stare``). Nothing
    states it by default, as nothing does for a run; the module asks for this with
    ``pytestmark = pytest.mark.usefixtures("local_stages")``."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("REGBENCH_STAGES", "local")
        yield


@pytest.fixture(scope="session")
def cases_root(tmp_path_factory):
    if os.environ.get("REGBENCH_TEST_CASES"):
        return Path(os.environ["REGBENCH_TEST_CASES"])
    root = tmp_path_factory.mktemp("cases")
    for name, spec in SPECS.items():
        toy.prepare_toy(root, name, spec)
    return root


@pytest.fixture(scope="session")
def toy_cases(cases_root):
    return {name: load_case(cases_root / toy.DATASET / name) for name in SPECS}


@pytest.fixture(scope="session")
def real_like(tmp_path_factory):
    """A slide with tissue in part of it only, standing in for a real nuclear channel."""
    spec = {**scene._BASE, "family": "t", "n": 1536, "seed": 8, "keep": 0.5}
    ref, _, _ = scene.make_pair(spec)
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
        diameter=scene.NUCLEUS_DIAMETER,
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

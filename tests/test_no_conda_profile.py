"""No Conda profile while no process ships a Conda environment.

A `conda` profile with no `conda` directive on any process runs every tool from the
host PATH and fails mid-run. Without the profile, `-profile conda` fails at launch
with Nextflow's unknown-profile error.
"""

import re

from tests.nfmodel import REPO_ROOT, strip_comments


def _processes_declaring_conda():
    hits = []
    for nf in sorted((REPO_ROOT / "modules").rglob("*.nf")):
        if re.search(r"^\s*conda\s", strip_comments(nf.read_text()), re.M):
            hits.append(nf.relative_to(REPO_ROOT))
    return hits


def test_no_conda_profile_without_conda_environments():
    cfg = strip_comments((REPO_ROOT / "nextflow.config").read_text())
    has_profile = re.search(r"^\s*(conda|mamba)\s*\{", cfg, re.M)
    vendored = {"modules/nf-core/basicpy/main.nf"}  # refuses Conda by design
    envs = [p for p in _processes_declaring_conda() if str(p) not in vendored]
    assert not has_profile or envs, (
        "nextflow.config defines a conda/mamba profile, but no first-party process "
        "declares a conda environment: the profile would run tools from the host PATH"
    )

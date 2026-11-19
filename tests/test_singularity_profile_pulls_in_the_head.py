"""Singularity images are pulled by the Nextflow head, as in nf-core's `singularity` profile.

`ociAutoPull = true` makes every task run `singularity exec docker://<ref>`, so each task
converts its own image into Apptainer's cache, which is not safe for concurrent writers.
Tasks starting together on a cold cache can die with "unexpected end of JSON input".

Left off (Nextflow's default, and the nf-core pipeline template's `singularity` profile),
the head pulls each image once into singularity.cacheDir under a temporary name and
renames it into place, so tasks only ever read a finished file.

Scans the comment-stripped view: nextflow.config's comment explaining the setting must not
trip the guard, and a commented-out setting is not a live one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.nfmodel import strip_comments

REPO = Path(__file__).resolve().parent.parent

# Matches both the dotted form (`singularity.ociAutoPull = true`) and the scoped form
# (`singularity { ociAutoPull = true }`), for singularity and apptainer alike.
OCI_AUTO_PULL_ON_RE = re.compile(r"\bociAutoPull\s*=\s*true\b")


def _shipped_configs() -> list[Path]:
    return [REPO / "nextflow.config", *sorted((REPO / "conf").glob("*.config"))]


@pytest.mark.parametrize(
    ("snippet", "live"),
    [
        ("singularity {\n    singularity.ociAutoPull  = true\n}\n", True),
        ("singularity {\n    ociAutoPull = true\n}\n", True),
        ("apptainer.ociAutoPull = true\n", True),
        ("// singularity.ociAutoPull = true\n", False),
        ("singularity.ociAutoPull = false\n", False),
    ],
)
def test_guard_sees_a_live_setting_and_ignores_a_comment(snippet, live):
    assert bool(OCI_AUTO_PULL_ON_RE.search(strip_comments(snippet))) is live


def test_no_config_makes_tasks_convert_their_own_images():
    offenders = [
        path.relative_to(REPO).as_posix()
        for path in _shipped_configs()
        if OCI_AUTO_PULL_ON_RE.search(strip_comments(path.read_text()))
    ]
    assert not offenders, (
        f"{offenders} set ociAutoPull = true. Every task then converts its own image into "
        "Apptainer's shared cache, which is not safe for concurrent writers ('conveyor "
        "failed to get: unexpected end of JSON input', head_neck 2026-09-15). Leave it "
        "off so the Nextflow head pulls each image once, as nf-core's singularity profile does."
    )

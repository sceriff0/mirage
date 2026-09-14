"""The `singularity` profile must convert images in the task, not in the Nextflow head.

With `ociAutoPull` off, Nextflow pulls every missing image from inside the head process
(`singularity pull --name <ref>.img.pulling.<ts> docker://<ref>`, run in cacheDir), and
that command takes no extra arguments. `singularity pull` converts the layers with
mksquashfs, whose cache defaults to 25% of the node's PHYSICAL memory rather than the
job's cgroup limit, so on a fresh cache a large image outgrows the head allocation:

    FATAL: While making image from oci registry: ... while creating squashfs:
           .../bin/mksquashfs command failed: signal: killed

(head_neck run, mirage-stardist, 2026-09-14). `singularity.pullTimeout` cannot help.

`singularity.ociAutoPull = true` hands the pull to each task's `singularity exec
docker://<ref>`, on the compute node, inside that task's allocation. Measured
2026-09-14 on Nextflow 25.04.7 and 26.04.6 with a recording fake `singularity`: zero
head-side pulls, every task execs the `docker://` ref, and an image already present in
singularity.cacheDir is NOT consulted (Apptainer's own cache is used instead).

Read on the comment-stripped view, so a commented-out setting (or this guard's own
explanation quoted in a comment) cannot satisfy it.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.nfmodel import strip_comments
from tests.test_no_secret_interpolation import _extract_singularity_profile_block

REPO = Path(__file__).resolve().parent.parent

OCI_AUTO_PULL_RE = re.compile(r"^\s*singularity\.ociAutoPull\s*=\s*true\s*$", re.M)


def _singularity_profile() -> str:
    text = strip_comments((REPO / "nextflow.config").read_text())
    return _extract_singularity_profile_block(text)


def test_singularity_profile_pulls_images_in_the_task_not_the_head():
    block = _singularity_profile()
    assert OCI_AUTO_PULL_RE.search(block), (
        "nextflow.config's `singularity` profile no longer sets "
        "`singularity.ociAutoPull = true`. Without it Nextflow converts every image "
        "inside the head job, where mksquashfs is unbounded and is OOM-killed on a "
        "fresh cache (mirage-stardist, 2026-09-14)."
    )


def test_no_config_turns_oci_auto_pull_back_off():
    offenders = []
    for path in sorted((REPO / "conf").glob("*.config")) + [REPO / "nextflow.config"]:
        code = strip_comments(path.read_text())
        if re.search(r"ociAutoPull\s*=\s*false", code):
            offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, (
        f"{offenders} set ociAutoPull = false, which moves image conversion back "
        "into the Nextflow head job"
    )

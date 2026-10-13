"""Ruling R6: every base image and every external image is pinned by content digest.

A version TAG is mutable. `python:3.11-slim` is republished weekly, `nvidia/cuda:12.2.2-*`
has been rebuilt against new CUDA patch releases, and even `ubuntu:22.04` moves on every
point release. Rebuilding an image from a mutable tag therefore produces a DIFFERENT image
than the one that generated the published results, with nothing in the repository recording
which one that was. containers/README.md has admitted this in a "Remaining reproducibility
caveat" paragraph since the one-repo-per-image rename; this file is what closes it.

TWO FORMS, AND THE DIFFERENCE IS THE CONTAINER RUNTIME.

  * A Dockerfile `FROM` keeps its tag AND carries the digest --
    `FROM image:tag@sha256:<64 hex>`. BuildKit documents that form and the tag is what
    makes the line readable to a human.
  * A Nextflow `container` directive carries the digest and NO tag --
    `image@sha256:<64 hex>`. Docker accepts either, but Apptainer/Singularity documents
    an image reference as NAME[:TAG|@DIGEST] -- tag OR digest -- and has a history of
    bugs with the combined form. The cluster pulls through Singularity, so these take
    the form both runtimes agree on. The tag lives in the comment beside them.

FIRST-PARTY IMAGES ARE DELIBERATELY NOT COVERED. bolt3x/mirage-* digests do not exist
until release.yml has pushed them, so pinning one in modules/local/*.nf is a
chicken-and-egg. Their record is the published-digest column of containers/README.md.
"""

import re
from pathlib import Path

import pytest

from tests.nfmodel import strip_comments

REPO = Path(__file__).resolve().parent.parent
CONTAINERS = REPO / "containers"

_DIGEST = r"sha256:[0-9a-f]{64}"

# `FROM <name>[:<tag>]@sha256:<64 hex>` -- AS-stage names and --platform flags tolerated.
_FROM_PINNED = re.compile(
    rf"^FROM\s+(?:--\S+\s+)*\S+@{_DIGEST}(?:\s+AS\s+\S+)?\s*$", re.M
)
_FROM_ANY = re.compile(r"^FROM\s+", re.M)

# The image references the pipeline pulls at RUNTIME that are not first-party.
# Each is (file, the exact pinned string that must appear).
EXTERNAL_SITES = {
    "modules/local/register.nf": (
        "cdgatenbee/valis-wsi@sha256:"
        "eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6"
    ),
    "lib/WarpBackends.groovy": (
        "cdgatenbee/valis-wsi@sha256:"
        "eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6"
    ),
    "conf/modules.config": (
        "docker.io/labsyspharm/basicpy-docker-mcmicro@sha256:"
        "355b14e2ec80b7b152272f333afd47234f007d0d37633b3ec948e87ec2c8e9b4"
    ),
}


def _dockerfiles():
    found = sorted(CONTAINERS.glob("*/Dockerfile"))
    # Seventeen since 2026-10-13: the eleven pipeline images plus the registration
    # benchmark's six (valis120, deeperhistreg, palom, stare-regbench, valis100, ashlar120).
    assert len(found) == 17, (
        f"expected seventeen build contexts, found {len(found)}: {[p.parent.name for p in found]}. "
        "The glob, not the repository, is what changed -- and a glob that misses the "
        "interesting files passes vacuously."
    )
    return found


@pytest.mark.parametrize("path", _dockerfiles(), ids=lambda p: p.parent.name)
def test_every_from_line_carries_a_digest(path):
    # Comment-stripped: several of these Dockerfiles quote an unpinned reference in their
    # header prose ("Build: docker build -t bolt3x/mirage-convert:1.0.0 ..."), and a raw
    # scan would count it as a FROM line -- or, worse, let prose satisfy the pinned form.
    code = "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    froms = _FROM_ANY.findall(code)
    assert froms, f"{path} has no FROM line at all; the comment stripper ate the file"
    pinned = _FROM_PINNED.findall(code)
    assert len(pinned) == len(froms), (
        f"{path.relative_to(REPO)} has {len(froms)} FROM line(s) but {len(pinned)} carry a "
        f"@sha256: digest. A version tag is mutable, so a rebuild can produce a different "
        f"image than the one that generated the published results. Resolve the digest with\n"
        f"    docker buildx imagetools inspect <ref> --format '{{{{json .Manifest.Digest}}}}'\n"
        f"and write `FROM <image>:<tag>@sha256:<digest>` -- keep the tag, it is what makes "
        f"the line readable."
    )


@pytest.mark.parametrize("rel", sorted(EXTERNAL_SITES))
def test_every_external_runtime_image_is_digest_pinned(rel):
    """Read off the string-preserving view: the reference IS a string literal.

    `strip_comments` blanks comments and keeps string contents verbatim, which is the
    right view here -- `strip_comments_and_strings` would blank exactly the text under
    assertion, and the raw text would let a commented-out reference satisfy the check.
    """
    text = strip_comments((REPO / rel).read_text())
    want = EXTERNAL_SITES[rel]
    assert want in text, (
        f"{rel} does not pin its external image by digest. Expected the exact string\n"
        f"    {want}\n"
        "Note the form: digest and NO tag. Apptainer documents an image reference as "
        "NAME[:TAG|@DIGEST] and has known bugs with the combined tag+digest form, and the "
        "cluster pulls through Singularity. Record the tag in the comment beside it."
    )
    stale = re.search(r"cdgatenbee/valis-wsi:[0-9]|basicpy-docker-mcmicro:[0-9]", text)
    assert not stale, (
        f"{rel} still carries an unpinned tag reference {stale.group(0)!r} alongside the "
        "digest-pinned one. Two references to the same image is how they drift."
    )


def test_first_party_images_are_deliberately_not_digest_pinned():
    """The rule has a boundary, and an untested boundary is an assumption.

    A bolt3x/mirage-* digest cannot exist before release.yml pushes the image, so a
    digest there would be unresolvable. If this ever starts failing, someone has pinned
    one and the pipeline can no longer be built from a clean tree.
    """
    offenders = []
    for path in sorted((REPO / "modules").rglob("*.nf")) + sorted(
        (REPO / "lib").glob("*.groovy")
    ):
        for line in strip_comments(path.read_text()).splitlines():
            if re.search(rf"bolt3x/mirage-[a-z0-9-]+(?::[^\s'\"]+)?@{_DIGEST}", line):
                offenders.append(f"{path.relative_to(REPO)}: {line.strip()}")
    assert not offenders, (
        "a first-party image is digest-pinned: "
        + "; ".join(offenders)
        + ". Those digests "
        "do not exist until release.yml has pushed the images, so the reference is "
        "unresolvable from a clean tree. Record published digests in containers/README.md "
        "instead."
    )

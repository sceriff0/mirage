"""Container image names are a published interface, and this file is where their shape is fixed.

Every first-party image is ``bolt3x/mirage-<component>:<version>``: one Docker Hub repository
per image, the layout nf-core and biocontainers use. The component is the build-context
directory name, and the version in the tag is tied to ``manifest.version`` so an image set and
the pipeline that pulls it cannot drift apart. The retired single-repository layout
(``bolt3x/attend_image_analysis``) must not be referenced.
"""

import json
import re
from pathlib import Path

import pytest

from tests.nfmodel import strip_comments as _strip_comments

REPO = Path(__file__).resolve().parent.parent

NAMESPACE = "bolt3x"
PREFIX = "mirage-"

# Third-party images the pipeline pulls as-is. VALIS is upstream-maintained and deliberately not
# re-hosted (its from-source libvips build is heavy). basicpy-docker-mcmicro is the vendored
# nf-core BASICPY module's own container (mcmicro-maintained); BASICPY is a vendored module
# (modules/nf-core/basicpy/), not a bolt3x/mirage-<component> build, so re-hosting it would mean
# maintaining a fork of the vendored image too. (No bare `ubuntu:22.04`: it ships no procps,
# which Nextflow's task-metrics wrapper needs.)
EXTERNAL_ALLOWED = {
    "docker.io/labsyspharm/basicpy-docker-mcmicro@sha256:355b14e2ec80b7b152272f333afd47234f007d0d37633b3ec948e87ec2c8e9b4",
    # The vendored module (modules/nf-core/basicpy/main.nf) stays byte-for-byte upstream and
    # so still carries this tagged reference; the R6 digest pin lives in conf/modules.config's
    # `withName: 'BASICPY'` override instead. See modules/nf-core/basicpy/MIRAGE-NOTES.md.
    "docker.io/labsyspharm/basicpy-docker-mcmicro:1.2.0-patch5",
}

# The retired single-repository name. Nothing may reference it again.
LEGACY_REPO = "attend_image_analysis"

# The tag may be interpolated (`bolt3x/mirage-segeval:${params.segeval_tag}`) and an external
# reference may carry a digest (`cdgatenbee/valis-wsi@sha256:<64 hex>`, see
# tests/test_base_images_are_digest_pinned.py). A tag or a digest is required, or the pattern
# would match any quoted path on a line containing the word "container".
_IMAGE_REF = re.compile(
    r"""['"]("""
    r"""[A-Za-z0-9][A-Za-z0-9._/-]*"""
    r"""(?::[A-Za-z0-9$][A-Za-z0-9._${}-]*(?:@sha256:[0-9a-f]{64})?"""
    r"""|@sha256:[0-9a-f]{64})"""
    r""")['"]"""
)
# A first-party image is pinned by TAG, never by digest: the digest does not exist until
# the image has been built and pushed, and the tag is what release.yml publishes. The
# published digests are recorded in containers/README.md's mapping table instead.
_FIRST_PARTY = re.compile(
    rf"^{NAMESPACE}/{PREFIX}([a-z0-9]+(?:-[a-z0-9]+)*):(\d+\.\d+\.\d+)$"
)


def _manifest_version():
    text = (REPO / "nextflow.config").read_text()
    block = re.search(r"manifest\s*\{(.*?)\n\}", text, re.S).group(1)
    return re.search(r"version\s*=\s*'([^']+)'", block).group(1)


def _searched_files():
    return (
        sorted((REPO / "modules" / "local").glob("*.nf"))
        + sorted((REPO / "modules" / "nf-core").rglob("*.nf"))
        + sorted((REPO / "lib").glob("*.groovy"))
        + sorted((REPO / "conf").glob("*.config"))
    )


def _image_refs():
    """(file, ref) for every quoted image reference in the pipeline's own sources.

    Read through `tests.nfmodel.strip_comments`, which blanks comments but keeps
    string contents (the image ref is a string literal), so a commented-out
    `// container = '...'` line is not read as a live reference.
    """
    out = []
    for path in _searched_files():
        for line in _strip_comments(path.read_text()).splitlines():
            if "container" not in line and "Image" not in line:
                continue
            for ref in _IMAGE_REF.findall(line):
                if "/" in ref or ref in EXTERNAL_ALLOWED:
                    out.append((path.relative_to(REPO).as_posix(), ref))
    return out


def _container_dirs():
    root = REPO / "containers"
    return sorted(p.name for p in root.iterdir() if (p / "Dockerfile").is_file())


def test_no_source_references_the_legacy_single_repository():
    """The one-repo-many-tags layout is retired; a stray reference would pull a stale image.

    Scans every file under ``containers/`` as well as the pipeline sources, so a Dockerfile's
    build/push comments are covered too.
    """
    offenders = []
    containers_files = sorted(
        p for p in (REPO / "containers").rglob("*") if p.is_file()
    )
    for path in (
        _searched_files()
        + containers_files
        + [REPO / "CLAUDE.md", REPO / ".github/workflows/containers.yml"]
    ):
        if not path.is_file():
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if LEGACY_REPO in line:
                offenders.append(f"{path.relative_to(REPO)}:{i}")
    assert not offenders, (
        f"{LEGACY_REPO} is the retired single repository but is still referenced at {offenders}."
    )


def test_every_image_reference_is_either_first_party_or_an_allowed_external():
    bad = [
        (f, r)
        for f, r in _image_refs()
        if r not in EXTERNAL_ALLOWED and not _FIRST_PARTY.match(r)
    ]
    assert not bad, (
        f"image references that are neither {NAMESPACE}/{PREFIX}<component>:<x.y.z> nor an "
        f"allowed external image: {bad}. Allowed externals are {sorted(EXTERNAL_ALLOWED)}."
    )


def test_first_party_images_are_pinned_to_the_manifest_version():
    """An image set and the pipeline that pulls it must not drift apart silently."""
    want = _manifest_version()
    wrong = [
        (f, r)
        for f, r in _image_refs()
        if (m := _FIRST_PARTY.match(r)) and m.group(2) != want
    ]
    assert not wrong, (
        f"these images are not pinned to manifest.version ({want}): {wrong}"
    )


def test_every_referenced_component_has_a_build_context():
    missing = []
    dirs = set(_container_dirs())
    for f, r in _image_refs():
        m = _FIRST_PARTY.match(r)
        if m and m.group(1) not in dirs:
            missing.append((f, r, m.group(1)))
    assert not missing, (
        f"referenced images with no containers/<component>/Dockerfile to build them: {missing}. "
        f"Existing build contexts: {sorted(dirs)}"
    )


def test_the_build_workflow_covers_every_build_context_exactly_once():
    """The matrix must cover every build context on disk, exactly once.

    Reads `containers/images.json`, the single source of the workflow's matrix and its
    `only` filter (tests/test_container_build_matrix.py forbids an inline copy).
    """
    entries = json.loads((REPO / "containers/images.json").read_text())
    listed = [e["dir"] for e in entries]
    assert sorted(listed) == sorted(set(listed)), f"duplicate matrix entries: {listed}"
    assert set(listed) == set(_container_dirs()), (
        f"build matrix {sorted(set(listed))} does not match the build contexts on disk "
        f"{_container_dirs()}. An unlisted context is never built or published."
    )
    workflow = (REPO / ".github/workflows/containers.yml").read_text()
    assert "fromJson(needs.resolve.outputs.matrix)" in workflow, (
        "containers.yml must take its matrix from the resolve job's images.json-derived "
        "output; an inline list is a second copy that drifts"
    )


def test_the_build_workflow_applies_the_version_to_the_tag():
    """The specific bug: a required `version` input that never reached the published tag.

    Without this, every publish overwrites one mutable name and there is no prior image to roll
    back to.
    """
    text = (REPO / ".github/workflows/containers.yml").read_text()
    tags_line = re.search(r"^\s*tags:\s*(.+)$", text, re.M)
    assert tags_line, "containers.yml has no tags: line"
    assert "version" in tags_line.group(1), (
        f"the published tag {tags_line.group(1).strip()!r} does not include the resolved "
        f"version, so every push overwrites the same mutable tag and nothing is rollback-able."
    )


@pytest.mark.parametrize("name", _container_dirs())
def test_build_context_directories_use_the_component_name(name):
    """The directory, the image and the pipeline reference must all say the same word."""
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name), (
        f"containers/{name} is not a lowercase hyphenated component name, so it cannot map "
        f"cleanly onto a Docker Hub repository."
    )


def test_the_build_workflow_strips_a_leading_v_from_the_version():
    """GitHub release tags are v1.0.0; manifest.version -- and so the image reference -- is 1.0.0.

    Without normalisation the release path publishes mirage-<component>:v1.0.0 while every
    process pulls mirage-<component>:1.0.0, so a release would build ten images that nothing
    can pull and fail only once a run reaches the first process.
    """
    text = (REPO / ".github/workflows/containers.yml").read_text()
    assert "${VERSION#v}" in text, (
        "containers.yml does not strip a leading 'v' from the resolved version, so a GitHub "
        "release tagged v1.0.0 publishes images the pipeline cannot pull."
    )


def test_the_build_workflow_preflights_the_registry_credentials():
    """Missing credentials must fail once, in `setup`, with a message that says what is missing.

    Without a preflight the run fans out to one job per image and each dies separately on
    docker/login-action's "Username and password required", which names neither the secret nor
    the fix. A build-only run never reaches the login step, so it says nothing about publishing.
    """
    text = (REPO / ".github/workflows/containers.yml").read_text()
    assert "DOCKERHUB_USERNAME" in text and "DOCKERHUB_TOKEN" in text
    preflight = re.search(
        r"name: Preflight.*?(?=\n      - name:|\n  [a-z])", text, re.S
    )
    assert preflight, (
        "containers.yml has no preflight step, so a publish with missing secrets fans out "
        "into one opaque failure per image instead of failing once with a readable reason."
    )
    body = preflight.group(0)
    assert "secrets.DOCKERHUB_TOKEN" in body and "exit 1" in body, (
        "the preflight step does not actually check the token and fail: " + body[:300]
    )

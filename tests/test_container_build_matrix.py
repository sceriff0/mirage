"""containers/images.json is the ONE list of images to build. Keep it honest.

A build context under containers/ that is missing from the matrix is never
published, and the first run that pulls its tag fails with "manifest unknown".

The list is also read by the workflow's `only` filter, so a name here that has no
directory produces a dispatch option that always fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.nfmodel import processes, strip_comments

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_JSON = REPO_ROOT / "containers" / "images.json"
CONTAINERS = REPO_ROOT / "containers"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "containers.yml"
LIB_DIR = REPO_ROOT / "lib"

# VALIS is not built here: the pipeline pulls the upstream cdgatenbee/valis-wsi image.
NOT_BUILT_HERE: dict[str, str] = {}


def _entries() -> list[dict]:
    return json.loads(IMAGES_JSON.read_text())


def _build_context_dirs() -> set[str]:
    return {
        d.name
        for d in CONTAINERS.iterdir()
        if d.is_dir() and (d / "Dockerfile").exists()
    }


def test_every_build_context_is_in_the_matrix():
    listed = {e["dir"] for e in _entries()}
    missing = sorted(_build_context_dirs() - listed - set(NOT_BUILT_HERE))
    assert not missing, (
        f"containers/ has build context(s) absent from images.json: {missing}. "
        "An unlisted image is never built or pushed -- the run stays green and the "
        "tag simply does not exist on Docker Hub. Add it, or add it to "
        "NOT_BUILT_HERE with the reason."
    )


def test_every_matrix_entry_has_a_dockerfile():
    missing = sorted(
        e["dir"]
        for e in _entries()
        if not (CONTAINERS / e["dir"] / "Dockerfile").exists()
    )
    assert not missing, (
        f"images.json names dirs with no Dockerfile: {missing}. The matrix job "
        "would fail, and the `only` filter would offer a name that cannot build."
    )


def test_tags_are_unique_and_descriptive():
    """One Docker Hub repo, one descriptive tag per image. A duplicate tag means
    two images silently overwrite each other on push."""
    tags = [e["tag"] for e in _entries()]
    dupes = sorted({t for t in tags if tags.count(t) > 1})
    assert not dupes, f"duplicate tags in images.json: {dupes}"
    assert "latest" not in tags, "never :latest -- see containers/README.md"


def test_workflow_reads_the_json_rather_than_repeating_it():
    """The matrix must come from images.json, not a hand-written `include:` list."""
    wf = WORKFLOW.read_text()
    assert "fromJson(needs.resolve.outputs.matrix)" in wf, (
        "containers.yml no longer builds its matrix from containers/images.json"
    )
    assert "containers/images.json" in wf, (
        "containers.yml does not read containers/images.json"
    )
    # No inline per-image list left behind.
    assert not re.search(r"^\s*- \{ dir: \w+", wf, re.M), (
        "containers.yml still carries an inline image list; images.json is the "
        "single source and a second copy will drift"
    )


# A `container` directive, in either of the two forms this repo actually uses:
# the Nextflow process directive (`container 'x'`, no colon) and a Groovy map
# entry (`container : 'x'`, as in lib/SegBackends.groovy and lib/WarpBackends.groovy
# -- see _referenced_tags()). `\s*:?\s*` covers both without a separate branch.
# "containerOptions" is not matched: the character right after "container" must be
# whitespace, ':', or a quote, and 'O' is none of those.
_CONTAINER_DIRECTIVE_RE = re.compile(r"container\s*:?\s*[\"']([^\"']+)[\"']")

# First-party images are bolt3x/mirage-<component>:<version>. Only the component is
# captured: it is the part that must agree with containers/images.json.
_MIRAGE_COMPONENT_RE = re.compile(r"^bolt3x/mirage-([a-z0-9]+(?:-[a-z0-9]+)*):")


def _referenced_tags() -> set:
    """First-party mirage component names every process actually pulls.

    Scans modules/local/*.nf and lib/**/*.groovy: SEGMENT and WARP_SEG_QC resolve
    their container through SegBackends/WarpBackends, so those image names appear
    only in lib/. Read through `strip_comments` (strings kept), because a container
    reference lives inside a string literal.
    """
    referenced: set = set()

    def _scan(text: str) -> None:
        for m in _CONTAINER_DIRECTIVE_RE.finditer(strip_comments(text)):
            cm = _MIRAGE_COMPONENT_RE.match(m.group(1))
            if cm:
                referenced.add(cm.group(1))

    for proc in processes().values():
        _scan(proc.raw_body)
    # rglob, so a backend table added under a lib/ subdirectory is still scanned.
    for groovy in sorted(LIB_DIR.rglob("*.groovy")):
        _scan(groovy.read_text())
    return referenced


def test_the_reference_scan_is_not_vacuous():
    """An empty `referenced` set would make every assertion over it pass vacuously."""
    referenced = _referenced_tags()
    assert referenced, (
        "no container tags were extracted from modules/local/ or lib/ -- the "
        "extractor is stale, not the repo"
    )


def test_modules_pull_only_tags_the_matrix_publishes():
    """Every bolt3x/mirage-<component> a module pulls must be a component this
    workflow actually builds -- otherwise the pipeline references an image
    nothing publishes."""
    # `dir` is the field in the image name (bolt3x/mirage-<dir>); `tag` currently
    # equals it but is a separate field.
    published = {e["dir"] for e in _entries()}
    referenced = _referenced_tags()
    unknown = sorted(referenced - published)
    assert not unknown, (
        f"modules/lib pull component(s) this workflow never builds: {unknown}. "
        f"Either add a build context + images.json entry, or fix the reference. "
        f"Published: {sorted(published)}"
    )

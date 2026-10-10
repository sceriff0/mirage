"""containers/README.md's image table must agree with the Dockerfiles it describes.

Three of its eleven base-image cells were wrong at once on 2026-09-02, and each was wrong
in a way a reader would act on:

  * `cellsam` was documented as pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime. The real
    base is 2.6.0-cuda12.4-cudnn9-runtime, and the Dockerfile's own header explains why --
    torch 2.6.0 is the first release carrying the CVE-2025-32434 fix, which matters because
    cellSAM downloads model weights and torch.load is on that path. The README documented
    the vulnerable version.
  * `instanseg` was documented as 2.5.1-cuda11.8, and its own Dockerfile LABEL said
    "PyTorch 2.5.1 + CUDA 12.4". Three numbers, two of them wrong, in two places.
  * `merge` was documented with a PyTorch base. It is FROM cdgatenbee/valis-wsi:1.0.0 --
    the same upstream VALIS image REGISTER runs in, which is the whole reason it can share
    that image's libvips build.

BASE IMAGES ARE WHAT THIS FILE CHECKS. They are the one column that can be verified
mechanically, and the one that misled three ways at once. Process lists are corrected by
hand in the same commit and are not machine-checked here: docs/resources.md already carries
a per-image process table, and duplicating that comparison would put two guards on one fact.

THE COMPARISON IS TAG-AND-NAME, DIGEST STRIPPED. Every FROM carries an @sha256: digest
(ruling R6, tests/test_base_images_are_digest_pinned.py). Requiring the README to repeat a
64-character digest in a table cell would make it unreadable AND put the digest in two
places; the published-digest column at the bottom of the README is where digests live.

COMMENT-STRIPPED ON PURPOSE (Verification reality item 7). _from_ref strips full-line `#`
comments before matching FROM, and _mapping_rows only reads lines that actually start with
`|` inside the "## Image mapping" section -- a comment mentioning a stale base image cannot
satisfy either check the way a trailing-comment loophole satisfied other guards in this repo.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "containers" / "README.md"
CONTAINERS = REPO / "containers"

_FROM = re.compile(
    r"(?m)^FROM\s+(?:--\S+\s+)*(\S+?)(?:@sha256:[0-9a-f]{64})?\s*(?:AS\s+\S+)?$"
)
# A markdown table row, split on unescaped pipes. Backticked contents are what we compare.
_BACKTICKED = re.compile(r"`([^`]+)`")


def _from_ref(container):
    """`name:tag` from containers/<container>/Dockerfile, digest stripped."""
    text = (CONTAINERS / container / "Dockerfile").read_text()
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    matches = _FROM.findall(code)
    assert len(matches) == 1, (
        f"containers/{container}/Dockerfile has {len(matches)} FROM line(s) after comment "
        f"stripping; this guard assumes exactly one. Found: {matches}"
    )
    return matches[0]


def _container_dirs():
    found = sorted(p.name for p in CONTAINERS.iterdir() if (p / "Dockerfile").is_file())
    # Fifteen since 2026-10-11: the eleven pipeline images plus the registration
    # benchmark's four (valis120, deeperhistreg, palom, stare-regbench).
    assert len(found) == 15, f"expected fifteen build contexts, found {found}"
    return found


def _mapping_rows():
    """{build-context name: full row text} for the "Image mapping" table.

    Scoped to that table by its heading, not to every table in the file: the README also
    carries a legacy-contexts table whose first column is a directory name too, and
    matching on shape alone would compare rows describing images that do not exist.
    """
    text = README.read_text()
    start = text.index("## Image mapping")
    end = text.index("## Runtime requirements every image must satisfy")
    rows = {}
    for line in text[start:end].splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        names = _BACKTICKED.findall(cells[0])
        if len(names) == 1 and (CONTAINERS / names[0] / "Dockerfile").is_file():
            rows[names[0]] = line
    return rows


def test_the_table_is_found_and_covers_every_build_context():
    """A heading rename would make every case below vacuous."""
    rows = _mapping_rows()
    missing = sorted(set(_container_dirs()) - set(rows))
    assert not missing, (
        f"containers/README.md's Image mapping table has no row for {missing}. An image "
        f"with no row is an image nobody reading this directory knows exists."
    )
    extra = sorted(set(rows) - set(_container_dirs()))
    assert not extra, f"the table has rows for non-existent build contexts: {extra}"


@pytest.mark.parametrize("container", _container_dirs())
def test_the_documented_base_image_matches_the_dockerfile(container):
    row = _mapping_rows()[container]
    want = _from_ref(container)
    assert want in row, (
        f"containers/README.md documents containers/{container} with a base image that is "
        f"not its FROM. The Dockerfile says `{want}`; that string does not appear in the "
        f"row:\n  {row.strip()}\n"
        f"Three of these eleven cells were wrong at once before this guard existed, and "
        f"one of them documented a torch version predating the CVE fix the real base was "
        f"chosen for."
    )


def test_the_readme_does_not_claim_the_retired_descriptive_tag_scheme():
    """Two blockquotes made OPPOSITE claims about what the tag means.

    One said "a content-descriptive tag (e.g. preprocess, convert_bioformats_2, tiled) --
    not a release version"; the next said "The tag is the pipeline version from
    manifest.version". The second is true. The first describes the retired
    one-repository-many-tags layout, and leaving it made the directory's own documentation
    the reason someone might publish a mutable name again.
    """
    text = README.read_text()
    assert "convert_bioformats_2`) — not a release version" not in text, (
        "containers/README.md still carries the retired 'content-descriptive tag, not a "
        "release version' blockquote, which contradicts the blockquote directly beneath "
        "it. The tag is manifest.version; tests/test_container_image_naming.py asserts it."
    )


def test_the_readme_says_every_image_installs_procps():
    """It used to name three. After 2026-09-02 every image does (eleven then, fifteen
    with the benchmark's four), and every smoke.sh asserts it."""
    text = README.read_text()
    assert "all fifteen install it explicitly" in text, (
        "containers/README.md's procps section still names a subset of images. Every image "
        "installs procps now and every containers/<dir>/smoke.sh asserts `ps -e` -- see "
        "tests/test_container_smoke_tests.py::test_every_smoke_script_proves_ps_exists."
    )

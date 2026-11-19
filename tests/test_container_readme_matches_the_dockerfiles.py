"""containers/README.md's image table must agree with the Dockerfiles it describes.

Base images are what this file checks: the one column that can be verified mechanically
(a wrong one can, e.g., document a torch version without a CVE fix). Process lists are not
checked here; docs/resources.md carries the per-image process table.

The comparison is tag and name, digest stripped: every FROM carries an @sha256: digest
(tests/test_base_images_are_digest_pinned.py), and the README keeps digests in its own
column.

_from_ref strips full-line `#` comments before matching FROM, and _mapping_rows reads only
lines starting with `|` inside the "## Image mapping" section, so a comment naming a stale
base image cannot satisfy either check.
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
    # Seventeen since 2026-11-05: the twelve pipeline images (containers/valis joined) plus the
    # registration benchmark's five (valis120, deeperhistreg, palom, stare-regbench, ashlar120;
    # valis100 left with the valis_mirage row).
    assert len(found) == 17, f"expected seventeen build contexts, found {found}"
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
    """The README must say the tag is the pipeline version (manifest.version), not a
    content-descriptive tag, which describes the retired single-repository layout.
    """
    text = README.read_text()
    assert "convert_bioformats_2`) — not a release version" not in text, (
        "containers/README.md still carries the retired 'content-descriptive tag, not a "
        "release version' blockquote, which contradicts the blockquote directly beneath "
        "it. The tag is manifest.version; tests/test_container_image_naming.py asserts it."
    )


def test_the_readme_says_every_image_installs_procps():
    """The README says every image installs procps, which every smoke.sh asserts."""
    text = README.read_text()
    assert "all seventeen install it explicitly" in text, (
        "containers/README.md's procps section still names a subset of images. Every image "
        "installs procps now and every containers/<dir>/smoke.sh asserts `ps -e` -- see "
        "tests/test_container_smoke_tests.py::test_every_smoke_script_proves_ps_exists."
    )

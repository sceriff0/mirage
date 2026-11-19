"""CHANGELOG.md must be shaped like a released project's changelog.

Four properties:

* every `## [X.Y.Z]` heading appears once -- two sections for one version means
  one of them is wrong and nothing says which;
* the version `nextflow.config`'s manifest declares has a section, and that
  section is DATED (a released version is not "Unreleased");
* an `## [Unreleased]` section exists, so the next change has somewhere to go
  without re-opening a released section;
* every version heading has a matching link definition at the foot, so the
  headings on GitHub actually link somewhere.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "CHANGELOG.md"

_HEADING = re.compile(r"^##\s+\[([^\]]+)\](?:\s+-\s+(\d{4}-\d{2}-\d{2}))?\s*$", re.M)
_LINK_DEF = re.compile(r"^\[([^\]]+)\]:\s+\S+\s*$", re.M)


def _manifest_version():
    text = (REPO / "nextflow.config").read_text()
    match = re.search(r"^\s*version\s*=\s*'([^']+)'", text, re.M)
    assert match, "could not read manifest.version out of nextflow.config"
    return match.group(1)


def _headings():
    found = _HEADING.findall(CHANGELOG.read_text())
    assert len(found) >= 2, (
        f"only {len(found)} version headings parsed out of CHANGELOG.md -- the "
        "heading shape changed and every check below would pass vacuously"
    )
    return found


def test_no_version_has_two_sections():
    names = [name for name, _ in _headings()]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, (
        "CHANGELOG.md has more than one section for "
        + ", ".join(duplicates)
        + ". A reader looking up what is in a version finds whichever comes first, "
        "and nothing says which one is right."
    )


def test_the_manifest_version_has_a_dated_section():
    version = _manifest_version()
    dated = {name: date for name, date in _headings()}
    assert version in dated, (
        f"nextflow.config's manifest.version is {version} but CHANGELOG.md has no "
        f"[{version}] section."
    )
    assert dated[version], (
        f"CHANGELOG.md's [{version}] section carries no release date. A released "
        "version is dated; only [Unreleased] is not."
    )


def test_an_unreleased_section_exists():
    names = [name for name, _ in _headings()]
    assert "Unreleased" in names, (
        "CHANGELOG.md has no [Unreleased] section, so the next change has nowhere "
        "to go but into a released section."
    )


def test_every_heading_has_a_link_definition():
    defined = set(_LINK_DEF.findall(CHANGELOG.read_text()))
    missing = sorted({name for name, _ in _headings()} - defined)
    assert not missing, (
        "CHANGELOG.md headings with no link definition at the foot of the file: "
        + ", ".join(missing)
        + ". On GitHub these render as bracketed text that links nowhere."
    )

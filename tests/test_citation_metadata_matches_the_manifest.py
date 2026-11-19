"""CITATION.cff must describe the version that actually ships.

CFF is what GitHub renders in the "Cite this repository" panel and what reference
managers import, so a stale field there is a wrong citation in somebody's paper.

`version` is checked against `nextflow.config`'s manifest rather than against a
hardcoded string, so the two cannot drift; `release.yml`'s verify-config already
holds the manifest to the git tag, which closes the chain tag -> manifest -> CFF.
"""

import datetime as dt
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFF = REPO / "CITATION.cff"


def _field(name):
    match = re.search(rf"^{name}:\s*\"?([^\"\n]+)\"?\s*$", CFF.read_text(), re.M)
    assert match, f"CITATION.cff has no top-level `{name}:` field"
    return match.group(1).strip()


def _manifest_version():
    text = (REPO / "nextflow.config").read_text()
    match = re.search(r"^\s*version\s*=\s*'([^']+)'", text, re.M)
    assert match, "could not read manifest.version out of nextflow.config"
    return match.group(1)


def test_the_cff_version_is_the_manifest_version():
    assert _field("version") == _manifest_version(), (
        f"CITATION.cff says version {_field('version')} but nextflow.config's "
        f"manifest says {_manifest_version()}. GitHub's citation panel would hand "
        "out the wrong version."
    )


def test_the_release_date_is_a_real_date():
    value = _field("date-released")
    try:
        dt.date.fromisoformat(value)
    except ValueError:  # pragma: no cover - the assertion below is the message
        raise AssertionError(
            f"CITATION.cff's date-released is {value!r}, which is not an ISO date"
        ) from None


def test_no_placeholder_markers_survive():
    offenders = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(CFF.read_text().splitlines(), start=1)
        if "TODO(" in line
    ]
    assert not offenders, (
        "CITATION.cff still carries placeholder markers, which GitHub renders "
        "verbatim in the citation panel:\n  " + "\n  ".join(offenders)
    )

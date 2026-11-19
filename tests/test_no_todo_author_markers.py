"""No tracked file may carry the author placeholder.

A placeholder in LICENSE is a licence with no grantor, and one in CITATION.cff
propagates into citation managers and the Zenodo record.

Files come from `git ls-files`, the shipped set, rather than a glob that would walk
virtualenvs, test data and caches and need an exclusion list.

The MARKER constant is built from two fragments at import time, so this tracked
file never contains it and needs no self-exemption.

`TODO(release)` is not matched: CITATION.cff carries one for the DOI, open until the
tagged release is archived. test_the_release_marker_is_not_matched pins that.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Assembled at import time so this module's own bytes never contain it.
MARKER = "TODO" + "(author)"


def _tracked_files():
    """Every file git tracks, as absolute paths. The shipped set, by definition."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / name for name in out.split("\0") if name]


def _offenders():
    """(relative path, 1-based line number, line) for every occurrence of MARKER.

    Read as BYTES. `git ls-files` includes PNGs and a favicon; decoding them as
    text would either raise or need a suffix allowlist, and a suffix allowlist is
    a place for a real file to hide. A byte search over a binary is harmless.
    """
    needle = MARKER.encode()
    hits = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        data = path.read_bytes()
        if needle not in data:
            continue
        for lineno, line in enumerate(data.split(b"\n"), start=1):
            if needle in line:
                hits.append(
                    (
                        str(path.relative_to(ROOT)),
                        lineno,
                        line.decode("utf-8", errors="replace").strip(),
                    )
                )
    return hits


def test_no_tracked_file_carries_the_author_placeholder():
    offenders = _offenders()
    assert not offenders, (
        f"{len(offenders)} tracked file location(s) still carry the author "
        "placeholder:\n  "
        + "\n  ".join(f"{p}:{n}: {line}" for p, n, line in offenders)
        + "\n\nThese are the identity of the released artefact: a LICENSE with a "
        "placeholder grantor, a CITATION.cff that propagates into every citation "
        "manager and into the archived release record, a manifest.author read by "
        "`nextflow info`. Supply the real values; there is nothing in the git "
        "history that is a safe substitute for asking."
    )


def test_the_scan_reads_the_whole_shipped_tree():
    """A 'must not find' scan passes over an empty set, so the enumerator must
    really enumerate."""
    tracked = _tracked_files()
    assert len(tracked) >= 300, (
        f"`git ls-files` returned only {len(tracked)} path(s) from {ROOT}. The "
        "repository ships several hundred; a number this small means the subprocess "
        "ran outside the work tree and the scan read almost nothing."
    )
    names = {p.name for p in tracked}
    for required in (
        "LICENSE",
        "CITATION.cff",
        "README.md",
        "mkdocs.yml",
        "nextflow.config",
    ):
        assert required in names, (
            f"{required} is not tracked, so this guard is not watching the file it "
            "was written for. Either it was renamed or the enumerator is wrong."
        )


def test_the_detector_actually_detects(tmp_path):
    """The other way a negative guard passes having proved nothing: the needle
    never matches the form it is looking for. Proved on a synthetic file."""
    probe = tmp_path / "probe.md"
    probe.write_text(f"Written by {MARKER}.\n")
    assert MARKER.encode() in probe.read_bytes()


def test_the_release_marker_is_not_matched():
    """`TODO(release)` is a different, legitimately-open marker: CITATION.cff carries
    one for the DOI, which cannot be filled until the tagged release is archived.
    Widening this guard to `TODO(` would make the release phase's own work fail."""
    assert MARKER.encode() not in b"# TODO" + b"(release): add `doi:` here"

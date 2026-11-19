"""No reference to the deleted COARSE front-ends outside history and the allow-list.

SIFT, the log-polar Fourier method and the `reg_tiled_frontend` dispatch knob are forbidden.
ORB is not: it is the fallback of the live COARSE anchor (stare/coarse_align.py).

Scope is `git ls-files` (the tracked tree), not Path.rglob, which ignores .gitignore and
would read untracked `.nf-test/` work dirs and virtualenvs.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Not \b: `\b` treats `_` as a word character and would miss `_sift_features` or
# `_frontend_fourier_mellin`. The lookaround excludes only letters and digits, so `_`
# is a boundary while words such as sifting or shift still do not match.
PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(sift|fourier[_ -]?mellin|reg_tiled_frontend)(?![A-Za-z0-9])",
    re.I,
)

EXCLUDE_PREFIXES = ("docs/_archive/", "tests/testdata/")
ALLOW_FILES = {
    # History. Keeps its mentions by design.
    "CHANGELOG.md",
    # This guard itself: its docstring and PATTERN name the terms it forbids.
    "tests/test_no_legacy_frontends.py",
    # The companion guard: test_the_deleted_frontends_are_really_gone names the deleted
    # symbols in a hasattr sweep. This guard covers stray text, that one live attributes.
    "tests/test_coarse_frontend.py",
    # Forbids these terms in docs/figures/*.html, so its RETIRED dict names them.
    "tests/test_figures_have_no_retired_names.py",
    # A survey of prior art: SIFT and the log-polar Fourier method are the algorithms
    # it surveys.
    "docs/parallel_registration_research.md",
}


def _candidates():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    assert len(out) > 100, (
        f"git ls-files returned only {len(out)} paths -- scope is wrong"
    )
    keep = [
        r
        for r in out
        if r not in ALLOW_FILES
        and not r.startswith(EXCLUDE_PREFIXES)
        and Path(r).suffix not in {".pyc", ".png", ".tif", ".tiff"}
    ]
    # Catches a filter that swallows most of the tree (a bad suffix set, a too-broad
    # EXCLUDE_PREFIXES or ALLOW_FILES), which would make a clean result vacuous.
    assert len(keep) > 0.8 * len(out), (
        f"the filter kept only {len(keep)} of {len(out)} tracked files -- it is excluding "
        "most of the repo, so a clean result would prove nothing"
    )
    return keep


def test_no_legacy_frontend_references():
    hits = []
    for rel in _candidates():
        p = REPO / rel
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:120]}")
    assert not hits, "reference(s) to a deleted COARSE front-end remain:\n" + "\n".join(
        hits
    )


def test_the_allowlist_has_no_dead_entries():
    """An exemption that exempts nothing is worse than no exemption.

    It reads as a real constraint and hides that the thing it was written for is gone.
    A file that stops existing is the same defect, so both are reported.
    """
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.split()
    )
    dead = []
    for rel in sorted(ALLOW_FILES):
        path = REPO / rel
        if rel not in tracked or not path.is_file():
            dead.append(f"{rel}: not a tracked file any more")
            continue
        if not any(
            PATTERN.search(line)
            for line in path.read_text(errors="ignore").splitlines()
        ):
            dead.append(
                f"{rel}: contains no match for this guard's own pattern, so exempting "
                "it exempts nothing"
            )
    assert not dead, (
        "ALLOW_FILES entr(y/ies) exempt nothing. Delete them -- a stale exemption "
        "reads as a decision somebody made and quietly widens the guard's blind "
        "spot:\n  " + "\n  ".join(dead)
    )

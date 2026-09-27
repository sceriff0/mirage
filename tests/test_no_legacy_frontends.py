"""No reference to the deleted COARSE front-ends outside history and the allow-list.

ORB IS NO LONGER ON THE LIST (2026-09-27). It came back, deliberately, as the FALLBACK of the
new COARSE anchor (drape/coarse_align.py: NCC rotation sweep first, scikit-image ORB + RANSAC
only when the sweep is ambiguous), so the word now names a live component and forbidding it
repo-wide would forbid documenting the method. SIFT, the log-polar Fourier method and the
`reg_tiled_frontend` dispatch knob stay deleted and stay forbidden.

Spec 2026-08-28 Phase 2b. Scope is `git ls-files` -- the TRACKED tree -- deliberately,
not Path.rglob: rglob ignores .gitignore and matched 107 untracked files (.nf-test work
dirs, virtualenvs, .planning/, docs/superpowers/) including this guard's own plan.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# NOT \b. `\b` treats `_` as a word character, so it MISSES exactly the identifiers
# this guard exists to keep out: `_frontend_orb`, `_orb_features`, `normalize_for_orb`,
# `_ORB_PCT`, `_sift_features` and `_frontend_fourier_mellin` all slipped through the
# original \b form -- verified against the compiled pattern, not reasoned about. The
# lookaround form below excludes only letters and digits, so `_` becomes a boundary.
#
# The pre-flight scan's counterexamples were re-run against THIS form: absorb, orbit,
# absorbance, sifting, shift and drift still produce no match, so the fix costs nothing.
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
    # The COMPANION guard. test_the_deleted_frontends_are_really_gone names all eight
    # deleted symbols (`_frontend_orb`, `normalize_for_orb`, ...) in a hasattr sweep --
    # naming them is the whole point of it. It was passing the ORIGINAL \b pattern only
    # because \b treats `_` as a word character; tightening the pattern surfaced it at
    # once. This guard covers stray TEXT, that one covers live ATTRIBUTES; the exemption
    # is what keeps them from cancelling each other out.
    "tests/test_coarse_frontend.py",
    # Same class of reason again: this is the negative-rule guard that forbids ORB,
    # SIFT, Fourier-Mellin and reg_tiled_frontend from appearing in docs/figures/*.html
    # (plan 11 Task 2). Its RETIRED dict and module docstring name all four terms
    # verbatim as the very things it forbids, and its drill in the same task's brief
    # temporarily writes "ORB" into a figure to prove the guard fires -- so this file
    # legitimately contains the tokens this guard exists to keep out of prose.
    "tests/test_figures_have_no_retired_names.py",
    # THREE ENTRIES WERE DELETED FROM HERE ON 2026-09-01 (CI redesign, Phase 7):
    # `.github/workflows/ci.yml`, `.github/workflows/release.yml` and
    # `tests/test_ci_stack_pinned.py`. Their stated reason was that they quote the
    # literal runtime error string "RuntimeError: ORB found no features." for a CI
    # pin. Running PATTERN (this file's own compiled regex, not a reading of the
    # prose) against each of the three returned ZERO matches: the string had been
    # gone for some time and the exemptions were exempting nothing while reading as
    # though a real constraint lived there.
    #
    # test_the_allowlist_has_no_dead_entries below is the fix for the class, not just
    # the instance -- an entry that stops matching now fails, so this cannot happen
    # again without someone deleting a test to allow it.
    # The research doc is a genuine survey of prior art -- SIFT and the log-polar Fourier
    # method are names of the algorithms it surveys, and renaming them would be a lie. (The
    # design doc and the reg_benchmark ORB-oracle files left this list on 2026-09-27: with ORB
    # dropped from PATTERN they matched nothing.)
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
    # NOT `assert keep`. With `len(out) > 100` asserted above and ALLOW_FILES holding a
    # dozen entries, `keep` cannot empty without the count tripping first -- that form was
    # decoration, not a second check. This one can actually fire: it catches a filter that
    # silently swallows most of the tree (a bad suffix set, an EXCLUDE_PREFIXES typo that
    # matches everything, an ALLOW_FILES that grew into a blanket), which is the realistic
    # way this guard would go quietly vacuous while still reporting zero hits.
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

    It reads as a constraint someone weighed, so the next reader routes around it --
    and it hides the fact that the thing it was written for is gone. Three of the
    entries above were in exactly that state when this check was added (2026-09-01):
    the ORB error string they claimed to protect had left `.github/` entirely, and
    the guard's own pattern returned zero matches for all three.

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

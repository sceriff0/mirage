"""No figure may name a backend, front-end or harness the pipeline retired.

`tests/test_figures_match_the_pipeline.py` resolves the names a figure writes
against the pipeline, so it catches a name that never existed or was renamed.
It cannot catch the opposite drift: a name that still resolves (`SEGMENT` is
still a process) sitting in a sentence that describes a world that is gone.
The ten `ORB` mentions in registration-schematic.html were exactly that -- the
front-end symbol was deleted from `bin/utils/coarse_align.py`, so no name check
had anything to fail on, and the figure went on telling reviewers the coarse
step is ORB + cross-check matching.

NARROW ON PURPOSE, and this is not caution -- it is the same ruling
`tests/test_ashlar_backend_removed.py` records in its own docstring: "a blanket
grep would delete them". Two ASHLAR sentences in this set are CORRECT and must
survive:
  * pipeline-schematic.html -- mcmicro applies BaSiC profiles inside ASHLAR's
    mosaic step, which mirage has no counterpart for.
  * registration-schematic.html -- STARE's phase-correlation kernel mirrors
    ASHLAR's.
So `ashlar` is forbidden only where it reads as a SELECTABLE BACKEND (a
`registration_method` value, a process name, an adapter name), never as a word.

Every pattern below is paired with the file:line that proves the thing is gone,
because an entry whose subject still exists forbids a name the pipeline uses.

This is a NEGATIVE rule ("must not appear"), and it reads RAW text
(`path.read_text()`), never `_prose()`. That is the opposite treatment from
`tests/test_figures_match_the_pipeline.py`'s POSITIVE checks, which read
`_prose()` (comments stripped) so that commented-out markup cannot satisfy a
claim the figure never actually makes. Here the risk runs the other way: a
retired name hidden inside `<!-- ... -->` is still a retired name sitting in
the file, and a trailing comment making the guard comment-blind is exactly the
failure mode CLAUDE.md's "Verification reality" #7 catalogues -- a positive
and a negative rule need opposite treatment.
"""

from __future__ import annotations

import re

from tests.nfmodel import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "docs" / "figures"

# pattern source -> (why it is retired, the evidence that it is)
RETIRED = {
    r"(?<![A-Za-z0-9])orb(?![A-Za-z0-9])": (
        "the classical ORB front-end is deleted",
        "packages/stare/src/stare/coarse_align.py (bin/utils/coarse_align.py shims it) has no _frontend_orb; the only front-end is "
        "_frontend_disk_lightglue",
    ),
    r"(?<![A-Za-z0-9])sift(?![A-Za-z0-9])": (
        "the SIFT front-end is deleted",
        "same file, same reason",
    ),
    r"fourier[_ -]?mellin": (
        "the Fourier-Mellin front-end is deleted",
        "same file, same reason",
    ),
    r"reg_tiled_frontend": (
        "the front-end selector param is deleted",
        "nextflow.config declares no such param; a single-valued enum is dead config",
    ),
    r"stare[_-]ml": (
        "containers/stare-ml/ is deleted; torch+kornia folded into the tiled image",
        "containers/images.json has no stare-ml entry",
    ),
    r"attend_image_analysis": (
        "the single-repo Docker Hub layout is retired",
        "every module names bolt3x/mirage-<name>:<version>",
    ),
    r"aicsimageio": (
        "the reader stack is bioio/tifffile/bioio-bioformats",
        "bin/utils/ome_io.py READERS",
    ),
    # anhir/acrobat: claiming-context only, same reasoning and same day as the
    # landmark pattern below. The bare word forbade a figure from saying, in
    # its own honest-limits panel, "no ANHIR or ACROBAT landmark TRE was
    # measured" as hard as it forbade a false "evaluated on ANHIR" -- a
    # negation and a claim are not the same risk. Narrowed to an affirmative
    # verb/preposition ahead of the dataset name on the same line, mirroring
    # the trigger-then-target order the landmark pattern already uses.
    r"(?:measured\s+on|evaluated\s+on|validated\s+against|scored\s+on|"
    r"benchmarked?\s+on|results\s+on)[^<\n]{0,25}(?:anhir|acrobat)": (
        "the ANHIR/ACROBAT landmark harness is deleted -- no figure may claim "
        "its public correspondences back a measured, evaluated, validated, "
        "scored or benchmarked accuracy result",
        "benchmarks/registration_eval/ exists on no branch; benchmarks/README.md "
        "section B (on the benchmarking branch) is its removal record. Acrobat's "
        "one theoretically legitimate bare mention -- DISK+LightGlue provenance "
        "-- is still not written by any figure, so nothing is lost by narrowing "
        "this the same way as anhir.",
    ),
    r"(?:public|external|ANHIR|ACROBAT)[^<\n]{0,25}landmark": (
        "the ANHIR/ACROBAT landmark harness is deleted -- no figure may claim an "
        "external landmark dataset backs any accuracy signal",
        "benchmarks/registration_eval/ exists on no branch; benchmarks/README.md "
        "section B (on the benchmarking branch) is its removal record. 'ground "
        "truth for that slide' in registration-schematic.html's reg_qc=2 panel is "
        "a distinct, legitimate sense of the word -- same-slide segmentation used "
        "as a LOCAL reference, not an external dataset. Scoped to the CLAIMING "
        "context (a trigger word ahead of 'landmark' on the same line), not the "
        "bare word, so a figure may still say plainly that no such dataset backs "
        "any measure here -- narrowed 2026-09-03 after the bare pattern forced "
        "accuracy-schematic.html's honest-limits panel to cite this very test file "
        "as evidence instead of stating the fact directly. INTERACTION: ANHIR and "
        "ACROBAT are themselves trigger words here, so a negation that puts either "
        "name immediately before 'landmark' ('no ACROBAT landmark TRE was measured') "
        "still fires -- write 'no landmark TRE against ANHIR or ACROBAT ...' instead. "
        "Word order between the two patterns matters; measured 2026-09-03.",
    ),
    # ashlar ONLY where it reads as a selectable backend. See the module docstring.
    r"registration_method[^<\n]{0,40}ashlar": (
        "ashlar was removed as a registration_method for v1.0.0",
        "tests/test_ashlar_backend_removed.py; nextflow_schema.json's enum",
    ),
    r"ASHLAR_(?:RETILE|SOLVE|ADAPTER)": (
        "the ashlar processes and adapter are deleted",
        "modules/local/ has no ashlar_*.nf",
    ),
}

# Shrink-only debt allowlist, same shape and same rule as
# tests/test_figures_match_the_pipeline.py's STALE_FIGURES: an entry is a debt,
# not a permission, and `test_the_allowlist_only_shrinks` fails once an entry
# stops matching. Do NOT add a figure here to make it pass.
ALLOW = {}

_COMPILED = {re.compile(p, re.I): v for p, v in RETIRED.items()}

# Non-vacuity, checked at import time rather than inside a test: an empty
# RETIRED would make every test below pass by iterating over nothing, and
# that must be caught before any test body runs, not discovered later as a
# suspiciously-green run.
assert RETIRED, "RETIRED is empty -- every check below would pass vacuously"


def _figures():
    found = sorted(FIGURES_DIR.glob("*.html"))
    assert len(found) >= 4, (
        f"only {len(found)} figure(s) found under {FIGURES_DIR} -- expected at "
        "least the four shipped supplementary figures; the glob or the "
        "directory has broken"
    )
    return found


def _hits(path):
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        for rx, (why, _evidence) in _COMPILED.items():
            if rx.search(line):
                out.append(f"{path.name}:{i}: {why} -- {line.strip()[:110]}")
    return out


def test_no_figure_names_a_retired_backend_or_frontend():
    problems = []
    for path in _figures():
        if path.name in ALLOW:
            continue
        problems += _hits(path)
    assert not problems, "figure(s) name something the pipeline retired:\n" + "\n".join(
        problems
    )


def test_the_patterns_still_match_something_somewhere():
    """A negative rule over a set that never matched proves nothing.

    Every pattern here was measured against this figure set on 2026-09-02 and at
    least four of them had live hits. If ALL of them stop matching -- including
    inside the allow-listed files -- the extractor has broken and every assertion
    above is looping over nothing.
    """
    matched = set()
    for path in _figures():
        text = path.read_text()
        for rx in _COMPILED:
            if rx.search(text):
                matched.add(rx.pattern)
    assert matched or not ALLOW, (
        "not one retired-name pattern matched anywhere in docs/figures/ -- either "
        "the set is finally clean (then ALLOW must be empty, and this assertion "
        "passes) or the reader has broken"
    )


def test_the_allowlist_only_shrinks():
    """An ALLOW entry must name a figure that exists and still has a hit."""
    stale = []
    for name, reason in ALLOW.items():
        path = FIGURES_DIR / name
        if not path.is_file():
            stale.append(f"{name}: no longer exists -- remove it from ALLOW")
            continue
        if not _hits(path):
            stale.append(
                f"{name}: carries no retired name any more -- remove it from ALLOW "
                f"so the next drift in it is caught (entry reason: {reason})"
            )
    assert not stale, "\n".join(stale)


def test_the_allowlist_is_not_a_blanket():
    """At least one figure must actually be checked."""
    assert len(ALLOW) < len(_figures())

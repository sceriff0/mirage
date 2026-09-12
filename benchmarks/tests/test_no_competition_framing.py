"""No LIVE reference to a ground-truth harness that exists on no branch.

Two harnesses occupied the ground-truth slot and both were deleted:
`benchmarks/registration_eval/` (ANHIR/ACROBAT expert landmarks, 61e26ec) and
`benchmarks/stare_bench/` (a synthetic displacement field, cacc850). What
survived was not prose. `make_figures.run()` REQUIRED a `reg_eval_csv`
argument, and the marker file its opt-out wrote told the operator to go run
`benchmarks/registration_eval/run_registration.sh` and `aggregate_eval.py` --
two scripts that exist nowhere. A dangling instruction in an output file is
worse than a missing feature: it reads as a step somebody forgot to take.

SCOPE IS `git ls-files`, the TRACKED tree, deliberately -- not Path.rglob, which
ignores .gitignore and would sweep in .venv, .nf-test work dirs and
docs/superpowers/ (this guard's own plan among them). Same choice, same reason,
as tests/test_no_legacy_frontends.py. The corollary: a NEW file is invisible to
this guard until it is `git add`-ed, so run it after staging, not before.

NARROWED 2026-09-12, because the premise moved. `benchmarks/anhir/` is a live
ANHIR landmark harness again -- it drives the PIPELINE (`--start registration
--stop registration`) and warps landmarks through the transform each backend
publishes, so it does not depend on the deleted single-task scripts. The bare
`anhir` and `ANHIR/ACROBAT` clauses that used to sit in PATTERN would have
forbidden the harness's own name; they are gone. What this guard still forbids
is what is still absent: a PATH into either deleted harness, a bare
(word-bounded) reference to either by name (`registration_eval`, `stare_bench`),
and `registration_eval`'s own tools (`aggregate_eval`, `prepare_pairs.py`,
`run_registration.sh`) -- a prose sentence naming one of these without the
`benchmarks/` prefix used to slip past a path-only check, which is exactly how a
LIVE instruction survived in benchmarks/configs/sweep.yaml. `acrobat` was never
forbidden as a bare word (`bin/utils/coarse_align.py` cites it as the provenance
of the DISK+LightGlue front-end that ships) and still is not.

The claiming-context rule for FIGURES ("evaluated on ANHIR") lives in
tests/test_figures_have_no_retired_names.py and is unchanged: no figure may claim
an ANHIR result until a run has produced one.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# A path into a deleted harness, a bare reference to either deleted harness, or
# one of registration_eval's own scripts (aggregate_eval.py, prepare_pairs.py,
# run_registration.sh). The dataset names themselves are NOT here: see the
# module docstring's 2026-09-12 note.
PATTERN = re.compile(
    r"benchmarks/registration_eval"
    r"|benchmarks/stare_bench"
    r"|\bregistration_eval\b"
    r"|\bstare_bench\b"
    r"|\baggregate_eval\b"
    r"|\bprepare_pairs\.py"
    r"|\brun_registration\.sh",
    re.I,
)

EXCLUDE_PREFIXES = ("docs/_archive/", "tests/testdata/")

ALLOW_FILES = {
    # The removal record itself. Section B is the whole point of it, and it now
    # carries a "nothing here is runnable" banner naming the two deletion SHAs.
    "benchmarks/README.md",
    # This guard. Naming the forbidden strings is what it is for.
    "benchmarks/tests/test_no_competition_framing.py",
    # An honest-limits NEGATION ("no landmark TRE against ANHIR or ACROBAT was
    # measured"), not a claim. dev's tests/test_figures_have_no_retired_names.py
    # deliberately narrowed its own anhir/acrobat rule to CLAIMING context (see
    # its comment ~:76-106: "a negation and a claim are not the same risk") --
    # the claiming-context risk in figures is owned by that guard, not this one.
    "docs/figures/accuracy-schematic.html",
    # Names the forbidden strings as its own patterns, like this guard.
    "tests/test_figures_have_no_retired_names.py",
    # Four DATED plan/spec records, bannered as superseded on 2026-09-02 (release
    # plan 13 Task 8) and left otherwise untouched. They are gitignored
    # (.gitignore:104) and excluded from the site (mkdocs.yml exclude_docs), so
    # they are published nowhere; rewriting a June/July plan to describe a
    # September tree would make it a lie about its own date. Same treatment,
    # same reason, as docs/parallel_registration_design.md in
    # tests/test_no_legacy_frontends.py. (The fifth,
    # 2026-06-17-registration-accuracy-eval.md, was deleted by dev's plan 02.)
    "docs/superpowers/plans/2026-06-17-benchmark-docs.md",
    "docs/superpowers/plans/2026-06-17-benchmarking-foundation.md",
    "docs/superpowers/specs/2026-06-16-benchmarking-framework-design.md",
    "docs/superpowers/specs/2026-07-24-benchmark-paper-data-design.md",
}


def _tracked():
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
    assert len(keep) > 0.8 * len(out), (
        f"the filter kept only {len(keep)} of {len(out)} tracked files -- it is "
        "excluding most of the repo, so a clean result would prove nothing"
    )
    return keep


def test_no_live_reference_to_a_deleted_ground_truth_harness():
    hits = []
    for rel in _tracked():
        p = REPO / rel
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:120]}")
    assert not hits, (
        "reference(s) to a ground-truth harness that exists on no branch:\n"
        + "\n".join(hits)
    )


def test_the_deleted_harnesses_really_are_deleted():
    """The premise. If either directory came back, this guard forbids a path that
    is live again and every assertion above is wrong rather than merely stale."""
    for d in ("benchmarks/registration_eval", "benchmarks/stare_bench"):
        assert not (REPO / d).exists(), (
            f"{d} exists again -- this guard's whole premise is gone. Delete it, "
            "or narrow PATTERN to whatever is still absent."
        )


def test_the_allowlist_has_no_dead_entries():
    """An exemption that exempts nothing reads as a constraint somebody weighed."""
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.split()
    )
    problems = []
    for rel in sorted(ALLOW_FILES):
        if rel not in tracked:
            problems.append(f"{rel}: not a tracked file -- remove it from ALLOW_FILES")
            continue
        text = (REPO / rel).read_text(errors="ignore")
        if not PATTERN.search(text):
            problems.append(
                f"{rel}: no longer matches the pattern -- remove it from ALLOW_FILES"
            )
    assert not problems, "\n".join(problems)

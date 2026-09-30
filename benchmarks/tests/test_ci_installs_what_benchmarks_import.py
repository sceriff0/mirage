"""Every third-party module benchmarks/ imports must be installed by benchmarks.yml.

WHY THIS EXISTS. On 2026-08-26 and 2026-08-27 the stare-bench job failed in ~25 s with
`Interrupted: 4 errors during collection` and exit code 2. Nothing was wrong with any
test: benchmarks/analysis/lib/regress.py imports sklearn and test_plotting.py imports
matplotlib, and the job's `pip install` line named neither. pytest died during COLLECTION,
so ZERO tests ran -- and the run looked like an ordinary red build rather than a job that
never started. It stayed that way for five days.

This is the CI-side twin of tests/test_fixtures_have_a_producer.py: a test can only run
where the thing it needs exists. That guard covers fixtures; this one covers imports.

ONE LEG now: benchmarks/ itself, TOP-LEVEL imports only. An import inside a function is a
deliberate optional dependency (bin/utils/coarse_align.py's torch/kornia block is the
pattern) and is allowed to be absent; a top-level one kills collection.

There used to be a SECOND leg -- bin/tiled_{coarse,reg_tile,solve}.py, which the
deleted synthetic ground-truth rung's run script shelled out to, scanned at EVERY
import level because those scripts were EXECUTED BARE IN CI. It went with that
rung; see the long note above _installed_distributions for why re-pointing it
at benchmarks/run_ashlar_arm.sh is wrong,
and for the zarr failure that makes the every-level scan worth restoring if a benchmarks/
test ever executes a bin/ script in CI again.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "benchmarks.yml"
JOB = "benchmarks"
PKG = REPO / "benchmarks"
REQUIREMENTS = REPO / "requirements"

# THE RESOLVER IS SHARED, NOT PRIVATE. `tests/ci_actions.py` is the repository's one
# reader of workflow/composite-action structure; seven guards here once carried seven
# private parses of the same files and each had a different blind spot. This module used
# to carry an eighth -- a line-oriented `pip install` scanner -- which stopped seeing
# anything the moment the CI redesign moved every install behind
# `.github/actions/setup-python-env`. It now asks ci_actions which requirements/ files
# the `benchmarks` JOB installs, and expands them here.
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
ci_actions = importlib.import_module("ci_actions")

# import name -> the distribution that provides it, where they differ.
DIST_OF = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
}

# Modules that are NOT third-party: stdlib, and the repo's own top-level packages.
_FIRST_PARTY_PKGS = {"benchmarks", "bin", "lib", "tests", "utils"}


def _is_first_party(mod: str) -> bool:
    """True for the repo's own packages AND for anything importable out of bin/utils.

    The second half is not a special case. benchmarks/ashlar/solve.py sys.path-inserts the
    pipeline's bin/utils and then `from tiled_manifest import ...` -- reusing the
    pipeline's real manifest code so ashlar's manifest cannot drift from STARE's, which is
    the whole basis of the comparison. Resolving those against the filesystem,
    rather than listing them, means a new shared module does not have to be remembered
    here to avoid being reported as a missing PyPI distribution that does not exist.
    """
    if mod in _FIRST_PARTY_PKGS:
        return True
    for d in (REPO / "bin", REPO / "bin" / "utils"):
        if (d / f"{mod}.py").exists() or (d / mod / "__init__.py").exists():
            return True
    return False


def _expand(req: Path, seen: set[Path] | None = None) -> set[str]:
    """Distribution names a requirements file installs, following its `-r` chain.

    `-c` lines are NOT followed. A constraints file bounds a version; it installs
    nothing, so counting `-c constraints.txt` would report every harmonised package as
    installed by every file that merely constrains itself -- which is how a guard reports
    coverage it does not have.
    """
    seen = seen if seen is not None else set()
    req = req.resolve()
    if req in seen or not req.is_file():
        return set()
    seen.add(req)
    names: set[str] = set()
    for raw in req.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            names |= _expand(req.parent / line.split(None, 1)[1].strip(), seen)
            continue
        if line.startswith("-"):  # -c, --index-url, --extra-index-url, ...
            continue
        names.add(re.split(r"[=<>!~\[;\s]", line)[0].strip().lower())
    return {n for n in names if n}


def _requirement_files() -> list[str]:
    """The requirements/ files the `benchmarks` job installs, per ci_actions."""
    jobs = ci_actions.load_jobs(WORKFLOW)
    assert JOB in jobs, (
        f"{WORKFLOW.name} has no `{JOB}` job any more, so this guard resolves the "
        f"install of nothing. Jobs found: {sorted(jobs)}"
    )
    return ci_actions.requirement_files_installed(jobs[JOB])


def _installed_distributions() -> set[str]:
    """Every distribution the benchmarks job installs, via requirements/."""
    names: set[str] = set()
    for rel in _requirement_files():
        names |= _expand(REPO / rel)
    return names


def _top_level_imports() -> dict[str, set[Path]]:
    """module -> files that import it at MODULE level (not inside a function/class)."""
    found: dict[str, set[Path]] = {}
    for py in PKG.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:  # not this guard's job to police
            continue
        for node in tree.body:  # tree.body == top level only
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import -> first-party
                    continue
                mods = [(node.module or "").split(".")[0]]
            else:
                continue
            for m in mods:
                if m and not _is_first_party(m) and m not in sys.stdlib_module_names:
                    found.setdefault(m, set()).add(py.relative_to(REPO))
    return found


def _bin_module_path(name: str) -> Path | None:
    for d in (REPO / "bin", REPO / "bin" / "utils"):
        cand = d / f"{name}.py"
        if cand.exists():
            return cand
    return None


# THE SUBPROCESS LEG IS GONE, AND MUST NOT BE RE-ADDED IN THIS SHAPE.
#
# It scanned bin/tiled_{coarse,reg_tile,solve}.py -- the pipeline entry points the
# deleted synthetic ground-truth rung's run script (since deleted) shelled out to --
# at EVERY import level, because those scripts were EXECUTED BARE IN CI and
# bin/utils/tiled_io.py imports zarr inside a function. That premise died with the
# rung's removal: nothing under benchmarks/ runs a pipeline entry point in CI any more.
#
# The one remaining subprocess caller is benchmarks/run_ashlar_arm.sh (warp_seg_qc.py),
# and it runs ON THE CLUSTER, INSIDE CONTAINERS -- QC_EXEC selects the pipeline's tiled
# image, which is where warp_seg_qc.py's dependencies are supposed to live. Re-pointing
# the scanner at it (tried, and reverted) makes this job demand `pip install valis` and
# `scyjava`, because warp_seg_qc.py's VALIS branch is statically visible even though the
# tiled branch never takes it. That is a CI job asserting a contract it does not have.
#
# If a benchmarks/ test ever executes a bin/ script bare again, bring the every-level scan
# back with it -- the zarr failure it caught is real and invisible to a top-level scan.


def test_benchmarks_yml_installs_every_module_benchmarks_imports():
    installed = _installed_distributions()
    missing = []
    for mod, files in sorted(_top_level_imports().items()):
        dist = DIST_OF.get(mod, mod).lower()
        if dist not in installed:
            where = ", ".join(sorted(str(f) for f in files)[:3])
            missing.append(f"{mod} (dist {dist}) -- imported by {where}")
    assert not missing, (
        "benchmarks.yml does not install module(s) benchmarks/ imports at top level:\n  "
        + "\n  ".join(missing)
        + "\nWithout them pytest dies during COLLECTION (exit 2, zero tests run), which "
        "reads as an ordinary red build. Add them to the job's pip install line, pinned "
        "to the version ci.yml uses, or move the import inside the function that needs it."
    )


def test_the_scanner_actually_finds_imports():
    """A scanner that silently returns nothing would make the guard above vacuous."""
    mods = _top_level_imports()
    assert len(mods) >= 5, sorted(mods)
    assert "numpy" in mods, sorted(mods)


def test_the_install_resolver_actually_finds_packages():
    """Likewise for the workflow side: a resolver that returns nothing makes the
    guard above pass by finding no requirement to check against.

    THIS IS NOT A HYPOTHETICAL. Its ancestor scanned `benchmarks.yml` for a literal
    `pip install <names>` line, and the CI redesign moved every install in the
    repository behind `.github/actions/setup-python-env` -- at which point the old
    scanner returned an EMPTY SET and would have reported every import as missing.
    The rewrite resolves through `tests/ci_actions.py` instead, so the same move
    cannot silently blind it again.

    The three names exercise the resolver rather than merely being present:

      ruff           comes from a DIFFERENT requirements file (lint.txt) than the
                     other two, so a resolver that reads only the first `requirements:`
                     line loses it
      scikit-learn   reached only through requirements/ci.txt's `-r segeval.txt`, so
                     this proves the `-r` chain is followed one hop deep; its
                     distribution name also differs from its import name (`sklearn`),
                     so DIST_OF is proved applied rather than the import name matched raw
      numpy          named in segeval.txt AND stare.txt, i.e. reached twice -- the cycle
                     guard in _expand must not drop it
    """
    files = _requirement_files()
    assert files, (
        "ci_actions resolved NO requirements file for the `benchmarks` job, so "
        "_installed_distributions() is empty and the guard above checks nothing."
    )
    installed = _installed_distributions()
    assert {"numpy", "scikit-learn", "ruff"} <= installed, sorted(installed)


def test_a_constraints_file_does_not_count_as_an_install():
    """`-c` bounds a version; it installs nothing.

    requirements/constraints.txt names opencv-python, which NO file here `-r`-installs
    (CI installs the headless distribution). If `_expand` followed `-c`, it would come
    back as installed and this guard would credit the job with a package it does not
    have -- the same "declared counts as covered" mistake the sweep coverage guard was
    fixed for.
    """
    assert (REQUIREMENTS / "constraints.txt").is_file()
    assert "opencv-python" in _expand(REQUIREMENTS / "constraints.txt")
    assert "opencv-python" not in _installed_distributions()

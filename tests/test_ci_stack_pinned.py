#!/usr/bin/env python3
"""Guard: there is exactly ONE copy of every dependency version, and it lives in requirements/.

`requirements/constraints.txt` is the version authority; the Dockerfiles COPY it and pass
it to pip with `-c`, and `requirements/ci.txt` includes the files the images build from.
This asserts:

  1. No workflow or composite action names a package version, and every install goes
     through `-r requirements/<file>` (or `-c requirements/constraints.txt`). A closed rule:
     no package list to forget. It covers `pip install`, `pip3 install`,
     `python -m pip install` and `uv pip install`.
  2. Every `containers/*/Dockerfile` that pip-installs a constrained package passes the
     constraints file to that invocation, unless it is an exception in
     `test_container_harmonisation.PINNED_EXCEPTIONS`.
  3. `requirements/constraints.txt` is `test_container_harmonisation.HARMONISED` (read, not
     restated).

Workflows (`.github/workflows/*.yml`), composite actions (`.github/actions/*/action.yml`) and
containers are discovered by glob. What each job installs is resolved per job through
`tests/ci_actions.py`.
"""

import ast
import importlib
import re
import shlex
import sys
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
CONTAINERS_DIR = ROOT / "containers"
REQUIREMENTS_DIR = ROOT / "requirements"
CONSTRAINTS = REQUIREMENTS_DIR / "constraints.txt"

# The harmonisation module owns the Dockerfile parser and the per-image exceptions; import
# them rather than re-parse. sys.path is set so this does not depend on pytest's rootdir.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
harmonisation = importlib.import_module("test_container_harmonisation")
ci_actions = importlib.import_module("ci_actions")


def workflow_files() -> list[Path]:
    files = sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOWS_DIR.relative_to(ROOT)}"
    return files


def scanned_files() -> list[Path]:
    """Every file a CI `pip install` can live in: workflows AND composite actions.

    Every CI install happens in `.github/actions/setup-python-env`, so scanning only the
    workflows would leave assertion 1 passing over nothing.
    """
    files = ci_actions.scanned_files()
    assert files, "no workflow or composite-action files found under .github/"
    return files


def container_dirs() -> list[str]:
    dirs = sorted(
        p.name for p in CONTAINERS_DIR.iterdir() if (p / "Dockerfile").is_file()
    )
    assert dirs, f"no Dockerfiles found under {CONTAINERS_DIR.relative_to(ROOT)}"
    return dirs


# --------------------------------------------------------------------------------------
# Parsing a pip invocation out of a shell body
# --------------------------------------------------------------------------------------
# `uv pip install` is accepted: the rule is about where the version lives, not which binary
# resolves it.
_PIP_INSTALL_RE = re.compile(
    r"(?:uv\s+)?(?:python[0-9.]*\s+-m\s+)?pip[0-9]*\s+install\b(?P<args>[^\n]*)"
)

# Shell operators that end the argument list, so `pip install -r x.txt && echo numpy==1.0`
# does not read the echo's version. Applied to shlex tokens, so a quoted
# `"scipy >= 0, < 99"` is not cut at the `<`.
_SHELL_OPERATORS = frozenset(
    {"&&", "||", ";", "|", "&", ">", ">>", "<", "<<", "2>", "2>&1"}
)

# Bootstrap tooling. May be installed without `-r`/`-c`, but still not with a version.
_BOOTSTRAP_ONLY = frozenset({"pip", "setuptools", "wheel"})


def pip_invocations(text: str) -> list[str]:
    """Every pip/uv-pip install argument string in a shell body, comments excluded.

    Continuation backslashes are joined first, so a multi-line install is one invocation.
    Only whole-line `#` comments are dropped: this feeds negative rules, so a trailing
    comment quoting a pin makes the guard noisy rather than blind.
    """
    joined = re.sub(r"\\\s*\n", " ", text)
    live = "\n".join(ln for ln in joined.splitlines() if not ln.strip().startswith("#"))
    return [m.group("args") for m in _PIP_INSTALL_RE.finditer(live)]


def split_args(args: str) -> list[str]:
    """Shell-split one pip argument string.

    `shlex`, not `str.split`, so a quoted `\'numpy == 1.0.0\'` stays one requirement token.
    An unparseable body falls back to a naive split;
    `test_every_workflow_pip_invocation_is_shell_parseable` fails on it by name.
    """
    try:
        toks = shlex.split(args, comments=True)
    except ValueError:
        toks = args.split()
    # Truncate at the first shell operator: everything after it is a different command.
    for i, tok in enumerate(toks):
        if tok in _SHELL_OPERATORS:
            return toks[:i]
    return toks


def requirement_tokens(args: str) -> list[str]:
    """The package tokens of one pip invocation: options, their values and paths removed."""
    tokens = []
    for tok in split_args(args):
        tok = tok.strip()
        if not tok:
            continue
        # Options, option values that are URLs or paths, shell variables, and VCS installs.
        if tok.startswith(("-", "$", "http://", "https://", "git+")) or "/" in tok:
            continue
        tokens.append(tok)
    return tokens


def pinned_tokens(args: str) -> list[str]:
    """Tokens of one invocation that name a package AND a version specifier."""
    pinned = []
    for tok in requirement_tokens(args):
        try:
            req = Requirement(tok)
        except InvalidRequirement:
            continue
        if str(req.specifier):
            pinned.append(tok)
    return pinned


_FILE_OPTS = ("-r", "--requirement", "-c", "--constraint")


def option_targets(args: str, opts) -> list[str]:
    """Values of `-r`/`-c`-style options in one invocation, in either `-r X` or `-r=X` form."""
    toks = split_args(args)
    out = []
    for i, tok in enumerate(toks):
        if tok in opts:
            if i + 1 < len(toks):
                out.append(toks[i + 1])
            continue
        for o in opts:
            if tok.startswith(o + "="):
                out.append(tok[len(o) + 1 :])
    return out


def goes_through_a_requirements_file(args: str) -> bool:
    return bool(option_targets(args, _FILE_OPTS))


# --------------------------------------------------------------------------------------
# Non-vacuity: the detector must actually detect
# --------------------------------------------------------------------------------------
# The checks below are negative assertions over a discovered set; these tests fail if the set
# is empty or the detector never matches.


@pytest.mark.parametrize(
    "body,expected",
    [
        ("pip install numpy==1.0.0", ["numpy==1.0.0"]),
        ("pip3 install 'dask[array]==2026.7.1'", ["dask[array]==2026.7.1"]),
        ("python -m pip install anndata<0.12", ["anndata<0.12"]),
        ("uv pip install --system tifffile==2025.5.10", ["tifffile==2025.5.10"]),
        (
            "pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu",
            ["torch==2.3.1"],
        ),
        # A quoted requirement containing spaces (why split_args uses shlex).
        ("pip install 'numpy == 1.0.0'", ["numpy == 1.0.0"]),
        ('pip install "scipy >= 0, < 99"', ["scipy >= 0, < 99"]),
        # Must NOT fire on any of these.
        ("pip install -r requirements/ci.txt", []),
        ("uv pip install --system -r requirements/ci.txt", []),
        ("python -m pip install --upgrade pip setuptools wheel", []),
        ("pip install -c /tmp/requirements/constraints.txt numpy", []),
        ("  # pip install numpy==1.0.0  -- a comment, not an install", []),
        ("pip install -r requirements/ci.txt && echo numpy==9.9.9", []),
    ],
)
def test_the_version_detector_actually_detects(body, expected):
    """The forms this repo writes, plus three near-misses.

    The near-misses: a `#` comment quoting a removed pin, a version after a shell operator,
    and an option value (`--index-url https://...`) a naive splitter reads as a package.
    """
    found = [tok for args in pip_invocations(body) for tok in pinned_tokens(args)]
    assert found == expected


def test_the_scanned_files_actually_install_something():
    """An empty invocation set would make assertion 1 pass over nothing.

    A refactor that moves the installs somewhere this scanner cannot see would leave
    assertion 1 passing while nothing is checked.
    """
    counts = {
        p.relative_to(ROOT).as_posix(): len(pip_invocations(p.read_text()))
        for p in scanned_files()
    }
    total = sum(counts.values())
    assert total >= 1, (
        "no file under .github/workflows/ or .github/actions/ contains a pip invocation "
        f"({counts}). Either the Python installs left CI, or they were reworded past "
        "_PIP_INSTALL_RE -- update the pattern rather than leaving the checks below to "
        "pass over an empty set."
    )


# The requirements/ files CI installs, as a named set rather than a count:
#   _test-suite.yml::python-tests    ci.txt, torch-cpu.txt
#   _test-suite.yml::nextflow-stub, nf-test-stub, security-tests   testdata.txt
#   _test-suite.yml::ruff            lint.txt
#   _test-suite.yml::format-tests    format-tests.txt
#   nightly.yml::nf-test-real, nf-test-integration   testdata.txt
# Do not write the nf-core tools config's filename in this file: tests/test_nfmodel.py
# flags any module containing the Nextflow file extension as a Nextflow-source reader.
EXPECTED_CI_REQUIREMENTS = {
    "requirements/ci.txt",
    "requirements/torch-cpu.txt",
    "requirements/testdata.txt",
    "requirements/lint.txt",
    "requirements/format-tests.txt",
}


def test_ci_still_installs_the_files_it_is_supposed_to():
    """The real non-vacuity floor, and why the invocation count above can be 1.

    `.github/actions/setup-python-env` runs `pip install -r "$req"` in a loop, so what
    matters is which requirements/ files each job installs, resolved per job by
    `tests/ci_actions.py` through the `requirements:` input. Per job, not a union: a
    sibling job's install does nothing for another runner.
    """
    per_job = {}
    for wf in workflow_files():
        for name, job in ci_actions.load_jobs(wf).items():
            installed = ci_actions.requirement_files_installed(job)
            if installed:
                per_job[f"{wf.relative_to(ROOT).as_posix()}::{name}"] = installed

    distinct = {req for files in per_job.values() for req in files}

    # Two checks: a low floor catches a resolver that stopped resolving, and an exact set
    # catches a change to which files CI installs, naming the difference both ways.
    assert len(distinct) >= 3, (
        f"CI resolves to only {len(distinct)} distinct requirements/ file(s) across "
        "every job. That is far below anything a consolidation would produce, so the "
        "installs have gone somewhere tests/ci_actions.py cannot follow -- fix the "
        f"resolver, not this number: {per_job}"
    )
    if distinct != EXPECTED_CI_REQUIREMENTS:
        gone = sorted(EXPECTED_CI_REQUIREMENTS - distinct)
        new_files = sorted(distinct - EXPECTED_CI_REQUIREMENTS)
        raise AssertionError(
            "the set of requirements/ files CI installs has changed.\n"
            f"  no longer installed: {gone or 'none'}\n"
            f"  newly installed:     {new_files or 'none'}\n"
            "If a file was CONSOLIDATED into another or a job was deleted, that is a "
            "normal change: update EXPECTED_CI_REQUIREMENTS in the same commit and "
            "this test is done. If a file went missing with no such change, a job "
            "stopped installing something it needs -- and a missing pin is how this "
            "suite once ran with no cv2 and 29 tests silently deselected.\n"
            f"Per job: {per_job}"
        )
    for req in sorted(distinct):
        assert (REQUIREMENTS_DIR / Path(req).name).is_file(), (
            f"a CI job installs {req}, which does not exist. Either the file was deleted "
            "or the call site names it wrongly -- and a `pip install -r` of a missing "
            "file fails the job, so this is a red build waiting to happen."
        )


def test_no_workflow_step_that_installs_a_requirements_file_is_conditional():
    """`tests/ci_actions.py` evaluates an `if:` inside a composite action and ignores one
    on a workflow step -- it has no way to resolve `always()` or a matrix expression.

    That asymmetry is safe only while no workflow step gates an install.
    """
    offenders = []
    for wf in workflow_files():
        for name, job in ci_actions.load_jobs(wf).items():
            for step in ci_actions.steps_of(job):
                if "if" not in step:
                    continue
                if ci_actions.requirements_from_step(step):
                    offenders.append(
                        f"{wf.relative_to(ROOT).as_posix()}::{name}: step "
                        f"{step.get('name', step.get('uses', '?'))!r} installs a "
                        f"requirements file behind `if: {step['if']}`"
                    )
    assert not offenders, (
        "a workflow step installs a requirements/ file conditionally. "
        "tests/ci_actions.requirement_files_installed does not evaluate workflow-level "
        "`if:` expressions, so it would report that install as unconditional:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------------------
# Assertion 1 -- no workflow names a version; every install reads a file
# --------------------------------------------------------------------------------------


def test_no_workflow_pins_a_package_version():
    """A closed rule: no package list, so no package can be forgotten out of it."""
    offenders = []
    for path in scanned_files():
        for args in pip_invocations(path.read_text()):
            for tok in pinned_tokens(args):
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()}: `pip install ... {tok}`"
                )
    assert not offenders, (
        "workflow or composite-action YAML names a package version. Every version string belongs in "
        "requirements/ -- constraints.txt for anything a container image also installs, "
        "or the per-job file otherwise -- so the pin exists once and Dependabot can see "
        "it. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_every_workflow_install_goes_through_a_requirements_file():
    """The positive half of the same rule.

    Assertion 1 alone permits `pip install numpy`, which resolves whatever PyPI serves that
    day. An install must name a file, or install only pip's own bootstrap tooling.
    """
    offenders = []
    for path in scanned_files():
        for args in pip_invocations(path.read_text()):
            if goes_through_a_requirements_file(args):
                continue
            named = {
                Requirement(t).name.lower()
                for t in requirement_tokens(args)
                if _parses(t)
            }
            extra = sorted(named - _BOOTSTRAP_ONLY)
            if extra:
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()}: `pip install {' '.join(extra)}` "
                    "with no -r/-c"
                )
    assert not offenders, (
        "workflow or composite-action YAML installs packages without reading a requirements/ file, so those "
        "packages are unpinned and invisible to Dependabot:\n  "
        + "\n  ".join(offenders)
    )


def _parses(token: str) -> bool:
    try:
        Requirement(token)
    except InvalidRequirement:
        return False
    return True


# --------------------------------------------------------------------------------------
# PRESENCE, PER JOB -- a different property from "no version is named here"
# --------------------------------------------------------------------------------------
# Assertions 1 and 3 are about where a version lives; these check that the job running the
# suite installs what it needs. The unit is the job, since a sibling job's install does
# nothing for another runner. Which steps a job runs, including through composite actions,
# is resolved by `tests/ci_actions.py` only.
# --------------------------------------------------------------------------------------

# `pytest ... tests/`: the directory form, i.e. a job that runs the whole suite. The
# lookahead keeps `pytest tests/test_<one file>.py` from counting.
_RUNS_THE_PYTEST_SUITE_RE = re.compile(r"pytest\b[^\n]*\btests/(?=\s|$)")


def jobs_running_the_pytest_suite() -> list[tuple[Path, str, dict]]:
    """(workflow, job id, the job) for every JOB that runs the whole suite.

    Uses `job_run_scripts`, the command view with comments stripped, so a comment naming
    `pytest tests/` cannot manufacture a hit.
    """
    hits: list[tuple[Path, str, dict]] = []
    files_with_a_hit: set[Path] = set()
    for wf in workflow_files():
        for job_id, job in ci_actions.load_jobs(wf).items():
            if not isinstance(job, dict):
                continue
            if _RUNS_THE_PYTEST_SUITE_RE.search(ci_actions.job_run_scripts(job)):
                hits.append((wf, job_id, job))
                files_with_a_hit.add(wf)
    assert hits, (
        f"no JOB under {WORKFLOWS_DIR.relative_to(ROOT)} runs `pytest ... tests/`; the "
        "per-job checks below would pass over an empty set. Either the suite stopped "
        "running in CI, or the structural walk lost the step -- fix the walk, do not "
        "leave the checks passing over nothing."
    )
    # Non-vacuity: the text scan and the structural walk must agree on which files run the
    # suite, or the YAML walk lost a step and the per-job checks stop covering that file.
    text_hits = {
        wf
        for wf in workflow_files()
        if _RUNS_THE_PYTEST_SUITE_RE.search(
            ci_actions.strip_comment_lines(wf.read_text())
        )
    }
    lost = sorted(str(wf.relative_to(ROOT)) for wf in text_hits - files_with_a_hit)
    assert not lost, (
        f"these workflows contain a `pytest ... tests/` invocation in their raw text but "
        f"no JOB of theirs does: {lost}. The structural walk lost it, so the per-job "
        "checks below no longer cover those files."
    )
    return hits


def _requirements_installed_by(job: dict) -> list[Path]:
    """Every `requirements/<file>.txt` a JOB installs, as resolved by ci_actions.

    `ci_actions.requirement_files_installed` follows a `run: pip install -r ...`, the
    `with: requirements:` input of `setup-python-env`, and a file named literally inside
    another composite action -- and it RAISES `UnresolvableCondition` on an action-step
    `if:` it cannot evaluate rather than guessing an answer.
    """
    found: list[Path] = []
    for name in ci_actions.requirement_files_installed(job):
        path = REQUIREMENTS_DIR / Path(name).name
        if path.is_file() and path not in found:
            found.append(path)
    return found


def _declared_packages(path: Path, seen: set[Path] | None = None) -> set[str]:
    """Distribution names a requirements file declares, FOLLOWING its `-r` includes."""
    seen = set() if seen is None else seen
    if path in seen:
        return set()
    seen.add(path)
    names: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r", "--requirement")):
            target = (
                line.split(None, 1)[1].strip()
                if " " in line
                else line.split("=", 1)[-1]
            )
            nested = REQUIREMENTS_DIR / Path(target).name
            if nested.is_file():
                names |= _declared_packages(nested, seen)
            continue
        if line.startswith("-"):
            continue
        parsed = harmonisation._parse_req(line)
        if parsed:
            names.add(parsed[0])
    return names


def test_every_job_running_the_suite_installs_the_ci_requirements_file():
    """Per job: a job running `pytest tests/` installs `requirements/ci.txt`.

    Without it the suite does not fail, it deselects: every module-scope
    `pytest.importorskip` becomes a silent skip.
    """
    missing = [
        f"{wf.relative_to(ROOT).as_posix()}: job `{job_id}`"
        for wf, job_id, job in jobs_running_the_pytest_suite()
        if REQUIREMENTS_DIR / "ci.txt" not in _requirements_installed_by(job)
    ]
    assert not missing, (
        "job(s) run `pytest ... tests/` without `pip install -r requirements/ci.txt`. Each "
        "of these jobs runs on its own runner, so a sibling job's -- or a sibling "
        "workflow's -- install line does nothing for it:\n  " + "\n  ".join(missing)
    )


# Import name -> distribution name where they differ; anything else is assumed to match, and
# a mismatch fails the check below by name.
_IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "cv2": "opencv-python-headless",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
}


def module_scope_importorskips() -> dict[str, list[str]]:
    """{distribution: [test files]} for every MODULE-SCOPE pytest.importorskip under tests/.

    Parsed with `ast`, so only module-scope calls count: those silently remove the whole file
    when the package is absent, so they are the packages CI must install.
    """
    out: dict[str, list[str]] = {}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken test file fails elsewhere
            continue
        for node in tree.body:  # MODULE scope only: tree.body, never ast.walk
            for call in (
                ast.walk(node) if isinstance(node, (ast.Assign, ast.Expr)) else ()
            ):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "importorskip"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    name = call.args[0].value
                    dist = _IMPORT_TO_DIST.get(name, name)
                    out.setdefault(dist, []).append(path.name)
    return out


def test_the_module_scope_importorskip_scan_is_not_empty():
    """Non-vacuity for the check below, which is a negative assertion over a derived set."""
    found = module_scope_importorskips()
    assert len(found) >= 8, (
        "found almost no module-scope pytest.importorskip under tests/ "
        f"({sorted(found)}). Either the suite stopped guarding its imports, or the AST walk "
        "no longer finds them -- fix the walk rather than leaving the check below to pass "
        "over an empty set."
    )
    # Packages whose absence would silently deselect tests.
    for dist in ("pyyaml", "opencv-python-headless"):
        assert dist in found, (
            f"{dist} is no longer reached by the module-scope importorskip scan. It was "
            "found there when this check was written; if the suite genuinely stopped using "
            "it, remove it from this list deliberately."
        )


def test_every_job_running_the_suite_installs_what_the_suite_importorskips():
    """PER-JOB. `-r requirements/ci.txt` is necessary; this is what makes it sufficient.

    A job could install ci.txt and still miss torch, which lives in its own file for the
    CPU index. The required set is derived from the suite's module-scope importorskips, so a
    new one is covered as soon as it is written.
    """
    required = module_scope_importorskips()
    missing = []
    for wf, job_id, job in jobs_running_the_pytest_suite():
        declared: set[str] = set()
        for req in _requirements_installed_by(job):
            declared |= _declared_packages(req)
        for dist, files in sorted(required.items()):
            if dist not in declared:
                missing.append(
                    f"{wf.relative_to(ROOT).as_posix()}: job `{job_id}`: no requirements "
                    f"file it installs declares {dist} -- "
                    f"{len(files)} test file(s) importorskip it at MODULE scope "
                    f"({', '.join(sorted(set(files))[:3])}...), so they are deselected "
                    "with nothing in the output to show for it"
                )
    assert not missing, (
        "job(s) run the pytest suite without installing something the suite guards with a "
        "module-scope importorskip. A skip is not a pass:\n  " + "\n  ".join(missing)
    )


def test_every_workflow_pip_invocation_is_shell_parseable():
    """`split_args` falls back to a naive split when shlex cannot parse; fail on that.

    Otherwise an unbalanced quote would send the line through a splitter that cannot see
    `'numpy == 1.0.0'`.
    """
    bad = []
    for wf in workflow_files():
        for args in pip_invocations(wf.read_text()):
            try:
                shlex.split(args)
            except ValueError as exc:
                bad.append(
                    f"{wf.relative_to(ROOT).as_posix()}: `pip install{args}` -- {exc}"
                )
    assert not bad, (
        "pip invocation(s) are not shell-parseable, so split_args falls back to a naive "
        "split and a quoted pin becomes invisible to assertion 1:\n  "
        + "\n  ".join(bad)
    )


# --------------------------------------------------------------------------------------
# Assertion 2 -- every image that installs a constrained package reads the constraints file
# --------------------------------------------------------------------------------------
# The exception list is derived from `PINNED_EXCEPTIONS`: an image there cannot take the
# constraints file, because pip reports ResolutionImpossible on a divergent direct pin.
EXEMPT = frozenset({harmonisation.PY312_ISLAND}) | {
    container for container, _pkg in harmonisation.PINNED_EXCEPTIONS
}


def constrained_packages() -> set[str]:
    return set(harmonisation.HARMONISED)


def _dockerfile_text(container: str) -> str:
    return (CONTAINERS_DIR / container / "Dockerfile").read_text()


def _constrained_installs(container: str) -> set[str]:
    """Constrained packages the image installs, via the harmonisation module's own parser."""
    text = _dockerfile_text(container)
    tokens = harmonisation._dockerfile_pip_tokens(text)
    for req in harmonisation._requirements_files_installed(text):
        tokens += harmonisation._requirements_tokens(req)
    names = set()
    for tok in tokens:
        parsed = harmonisation._parse_req(tok)
        if parsed:
            names.add(parsed[0])
    return names & constrained_packages()


def _uncovered_invocations(container: str) -> list[str]:
    """pip invocations naming a constrained package that no constraints file reaches.

    An invocation is covered either by `-c <...>constraints.txt` on the command line, or by
    installing a `requirements/<name>.txt` that itself opens with `-c constraints.txt` (pip
    resolves that path relative to the file naming it, which is why the Dockerfiles COPY
    both into one directory).
    """
    text = _dockerfile_text(container)
    uncovered = []
    for args in pip_invocations(text):
        names = set()
        for tok in requirement_tokens(args):
            parsed = harmonisation._parse_req(tok)
            if parsed:
                names.add(parsed[0])
        if not names & constrained_packages():
            continue
        # Anchored on `requirements/constraints.txt`, so `-c /tmp/my-old-constraints.txt`
        # does not count. The Dockerfiles COPY to /tmp/requirements/constraints.txt.
        if any(
            re.fullmatch(r"(?:.*/)?requirements/constraints\.txt", target)
            for target in option_targets(args, ("-c", "--constraint"))
        ):
            continue
        via_file = any(
            re.search(r"(?m)^\s*-c\s+constraints\.txt\s*$", req.read_text())
            for req in harmonisation._requirements_files_installed(args)
        )
        if via_file:
            continue
        uncovered.append(f"pip install{args}")
    return uncovered


@pytest.mark.parametrize("container", container_dirs())
def test_image_installing_a_constrained_package_reads_the_constraints_file(container):
    installs = sorted(_constrained_installs(container))
    if not installs:
        pytest.skip(f"containers/{container} installs no constrained package")
    if container in EXEMPT:
        pytest.skip(
            f"containers/{container} is a documented exception "
            f"(test_container_harmonisation.PINNED_EXCEPTIONS / PY312_ISLAND)"
        )
    text = _dockerfile_text(container)
    assert re.search(r"COPY[^\n]*requirements/constraints\.txt", text), (
        f"containers/{container}/Dockerfile installs {installs}, which requirements/"
        f"constraints.txt pins, but never COPYs that file -- so the image's pins are a "
        f"second copy of the version authority again. Add "
        f"`COPY requirements/constraints.txt /tmp/requirements/constraints.txt` and pass "
        f"`-c /tmp/requirements/constraints.txt` to pip. (The build context is the "
        f"repository root; see .github/workflows/containers.yml.)"
    )
    uncovered = _uncovered_invocations(container)
    assert not uncovered, (
        f"containers/{container}/Dockerfile COPYs requirements/constraints.txt but does "
        f"not pass it to every pip invocation that names a constrained package, so those "
        f"pins are unconstrained:\n  " + "\n  ".join(uncovered)
    )


@pytest.mark.parametrize("container", sorted(EXEMPT))
def test_every_exempt_image_is_still_diverging(container):
    """A stale exemption is worse than none: it reads as a reviewed decision.

    An image is exempt only while it genuinely cannot take the constraints file.
    """
    installs = _constrained_installs(container)
    assert installs, (
        f"containers/{container} is exempt from the constraints file but installs no "
        f"constrained package at all, so the exemption buys nothing. Remove it from "
        f"test_container_harmonisation.PINNED_EXCEPTIONS."
    )
    text = _dockerfile_text(container)
    assert not re.search(r"COPY[^\n]*requirements/constraints\.txt", text), (
        f"containers/{container} is listed as an exception yet now COPYs requirements/"
        f"constraints.txt. If the divergence is gone, delete its PINNED_EXCEPTIONS "
        f"entries; if it is not, the build will fail with ResolutionImpossible."
    )


# --------------------------------------------------------------------------------------
# Assertion 3 -- the constraints file IS the harmonised table
# --------------------------------------------------------------------------------------


def test_the_constraints_file_is_the_harmonised_table():
    """`test_container_harmonisation.HARMONISED` must be READ from the file, not restated.

    A hardcoded table would check the images against numbers nothing installs.
    """
    fresh = harmonisation._read_constraints(CONSTRAINTS)
    assert harmonisation.HARMONISED == fresh, (
        "tests/test_container_harmonisation.py's HARMONISED table no longer equals "
        "requirements/constraints.txt. It must READ that file -- the same file the "
        "Dockerfiles COPY and pip enforces with `-c` -- not restate it:\n"
        f"  HARMONISED  = {harmonisation.HARMONISED}\n"
        f"  constraints = {fresh}"
    )


# Distribution names that package the same upstream release. The images install
# `opencv-python`; CI installs `opencv-python-headless` (no display libs). pip resolves on the
# distribution name, so both are pinned and this table keeps their versions equal. The second
# name is installed by no image, so it is exempt from the "shared by more than one image" rule.
EQUIVALENT_DISTRIBUTIONS = (("opencv-python", "opencv-python-headless"),)


def test_equivalent_distributions_are_pinned_to_the_same_release():
    """The tie the constraints file cannot express on its own."""
    pins = harmonisation.HARMONISED
    wrong = []
    for a, b in EQUIVALENT_DISTRIBUTIONS:
        for name in (a, b):
            assert name in pins, (
                f"requirements/constraints.txt no longer pins {name!r}, so the pair "
                f"({a}, {b}) is untied again and CI can test against a different build "
                f"than production ships. Pin it, or drop the pair from "
                f"EQUIVALENT_DISTRIBUTIONS with a reason."
            )
        if pins[a] != pins[b]:
            wrong.append(f"{a}=={pins[a]} vs {b}=={pins[b]}")
    assert not wrong, (
        "requirements/constraints.txt pins two packagings of the SAME upstream release at "
        "different versions. They must move together, or CI exercises a different build "
        "than the image ships: " + "; ".join(wrong)
    )


def test_every_constrained_package_is_installed_by_more_than_one_image():
    """The constraints file is for SHARED packages; a solo pin belongs in that image's file.

    Otherwise constraints.txt grows into a second global requirements file.
    """
    counts = {pkg: 0 for pkg in constrained_packages()}
    for container in container_dirs():
        for pkg in _constrained_installs(container):
            counts[pkg] += 1
    # The second name of an equivalent pair is installed by no image; its constraint holds
    # CI to the first name's release.
    exempt = {b for _a, b in EQUIVALENT_DISTRIBUTIONS}
    lonely = sorted(p for p, n in counts.items() if n < 2 and p not in exempt)
    assert not lonely, (
        f"requirements/constraints.txt constrains {lonely}, which fewer than two container "
        "images install. Move the pin into the one image's own requirements file, or into "
        "requirements/ci.txt if only CI needs it -- see the 'deliberately not constrained "
        "here' block at the bottom of constraints.txt for the four packages that are "
        "recorded rather than constrained, and why."
    )

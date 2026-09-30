#!/usr/bin/env python3
"""Resolve a GitHub Actions workflow through the LOCAL COMPOSITE ACTIONS it uses.

WHY THIS MODULE EXISTS. Until 2026-09-01 every step CI ran lived inside
`.github/workflows/*.yml`, so a guard test that globbed that directory saw
everything. Phase 2 of the CI redesign moved ~70 duplicated step instances into
four composite actions under `.github/actions/**`. A guard whose glob stops at
`.github/workflows/` therefore goes BLIND the moment a step moves — it keeps
passing, over a file that no longer contains the thing it was written to check.
That is this repository's single most repeated failure mode (see CLAUDE.md,
"Verification reality"), so the resolution lives in ONE place rather than being
re-derived by each guard with a different blind spot.

WHAT "RESOLVE" MEANS HERE. A workflow's effective step list is its own steps plus
the steps of every `uses: ./.github/actions/<name>` it references, transitively.
This module gives guards three views of that:

  * `resolved_text(path)`   — the workflow's text concatenated with the text of
    every local action it reaches. For scans that only ask "does this string
    appear anywhere in what this workflow runs".
  * `job_resolved_run_text(job)` — the same, narrowed to ONE JOB: the job's own
    `run:` bodies plus the full text of the local actions that job uses. This is
    what keeps a per-job claim per-job.
  * `requirement_files_installed(job)` — the ordered list of `requirements/*.txt`
    files a job ends up installing, whether the `pip install -r` is written in
    the job, passed as the `requirements:` input of `setup-python-env`, or named
    literally inside an action the job calls. PER JOB, deliberately: a
    whole-workflow union let a sibling job's install line satisfy a check on
    behalf of the job that actually needed the package, which is how CI's
    blocking gate could have run with no cv2 (29 tests module-scope
    `importorskip` away in silence) and no PyYAML (which silently deletes both of
    tests/test_ci_job_hygiene.py's own guards).

THE SECOND KIND OF INDIRECTION (2026-09-01, CI redesign Phase 5): a job whose
whole body is `uses: ./.github/workflows/<x>.yml` — a REUSABLE WORKFLOW call.
`.github/workflows/_test-suite.yml` holds the five blocking suites, and both
`ci.yml` and `release.yml` reach them that way.

That indirection is deliberately NOT folded into the three views above, and the
reason is the same one that makes `requirement_files_installed` per-job:

  * `job_resolved_run_text(job)` must stay the ONE JOB's commands. A caller job
    is not five jobs; folding the called workflow in would make `ci.yml`'s
    one-line `test-suite` job look like an aggregator (it would contain
    `_test-suite.yml`'s `needs.<job>.result` reads), and would let a sibling LEG
    of the suite satisfy a per-job claim about another leg — the union problem,
    one level up.
  * `requirement_files_installed(job)` likewise. A caller job installs nothing.

The called workflow's jobs are not lost by that: every guard here discovers
workflows BY GLOB over `.github/workflows/`, so `_test-suite.yml`'s jobs are
scanned directly, on their own, as the jobs they are. What the glob cannot answer
is "does the thing this GATE requires include check X", because that crosses a
file boundary — so it gets its own, explicitly-named view:

  * `called_local_workflows(path)` — `{job_id: Path}` for each job in `path`
    whose body calls a local reusable workflow.
  * `workflow_run_scripts(path)` — `run_scripts(path)` plus that of every local
    workflow its jobs call, transitively. For scans that ask "does everything
    this workflow sets in motion RUN X".

RAW VIEW vs COMMAND VIEW — pick by the question, not by convenience.
`resolved_text` / `job_resolved_run_text` return the files' literal bytes,
comments included. They are correct only when the thing under assertion IS the
literal text (a `uses:` line, a cache key, an action reference). Every claim of
the form "CI runs this command" must go through the comment-stripped views —
`run_scripts`, `workflow_run_scripts`, `job_run_scripts` — because a comment
naming a command satisfies the raw view while the command does not execute. That
error has been made four times on this repository, on four unrelated guards.

NOT A GENERAL YAML MODEL. It answers exactly the questions this repo's CI guards
ask. Anything more would be a second parser with its own blind spots.

Consumers: tests/test_ci_stack_pinned.py, tests/test_ci_job_hygiene.py,
tests/test_release_workflow_graph.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
ACTIONS_DIR = ROOT / ".github" / "actions"

# `uses: ./.github/actions/<name>` — the only local-action form this repo writes.
# Matched against raw text rather than parsed structure so it works on an action
# file and a workflow file alike, and so a `uses:` inside a comment block is
# harmless (it would only ever widen a scan).
# The leading character may not be a dot: prose in this repo writes
# `uses: ./.github/actions/...` as an ellipsis, and a name of "..." matched the
# looser form and was reported as a missing action.
LOCAL_ACTION_USES_RE = re.compile(r"\./\.github/actions/([A-Za-z0-9_-][A-Za-z0-9._-]*)")

# `uses: ./.github/workflows/<file>.yml` as a JOB body — a reusable-workflow
# call. Same shape and the same leading-dot exclusion as above; the `.yml`/`.yaml`
# suffix is required, which is what distinguishes it from prose naming the
# directory.
LOCAL_WORKFLOW_USES_RE = re.compile(
    r"\./\.github/workflows/([A-Za-z0-9_-][A-Za-z0-9._-]*\.ya?ml)"
)

# `-r requirements/<file>.txt` / `--requirement=requirements/<file>.txt`, and the
# bare `requirements/<file>.txt` form a `with: requirements:` block writes.
REQUIREMENTS_FILE_RE = re.compile(r"requirements/[A-Za-z0-9._-]+\.txt")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def workflow_files() -> list[Path]:
    """Every workflow under `.github/workflows/`, DISCOVERED BY GLOB."""
    files = sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOWS_DIR.relative_to(ROOT)}"
    return files


def action_files() -> list[Path]:
    """Every local composite action under `.github/actions/*/action.yml`.

    Not asserted non-empty: a repository with no composite actions is legal, and
    the guards that care assert their own non-vacuity against what they resolve.
    """
    if not ACTIONS_DIR.is_dir():
        return []
    return sorted(ACTIONS_DIR.glob("*/action.yml")) + sorted(
        ACTIONS_DIR.glob("*/action.yaml")
    )


def action_file(name: str) -> Path | None:
    """The action.yml for `./.github/actions/<name>`, or None if it is missing."""
    for suffix in ("action.yml", "action.yaml"):
        candidate = ACTIONS_DIR / name / suffix
        if candidate.is_file():
            return candidate
    return None


def scanned_files() -> list[Path]:
    """Workflows AND local composite actions — the full set CI steps can live in."""
    return workflow_files() + action_files()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def local_actions_used(text: str) -> list[str]:
    """Local action names referenced by a blob of YAML, in first-seen order."""
    seen: list[str] = []
    for name in LOCAL_ACTION_USES_RE.findall(text):
        if name not in seen:
            seen.append(name)
    return seen


def resolved_action_files(text: str) -> list[Path]:
    """Every local action file reachable from `text`, transitively.

    A name with no action.yml is skipped rather than raising: that is a real
    defect, but it is `test_ci_job_hygiene.test_every_local_action_referenced_exists`'s
    to report, and raising here would turn one clear failure into many confusing
    ones.
    """
    out: list[Path] = []
    pending = local_actions_used(text)
    seen: set[str] = set()
    while pending:
        name = pending.pop(0)
        if name in seen:
            continue
        seen.add(name)
        path = action_file(name)
        if path is None:
            continue
        out.append(path)
        pending.extend(local_actions_used(path.read_text()))
    return out


def resolved_paths(path: Path) -> list[Path]:
    """`path` plus every local action file it reaches, transitively."""
    return [path, *resolved_action_files(path.read_text())]


def strip_comment_lines(text: str) -> str:
    """Drop whole-line `#` comments.

    THIS IS LOAD-BEARING, NOT TIDINESS. Every view below is substring-matched by
    guards looking for a COMMAND, and this repo's YAML explains its commands at
    length -- `.github/actions/setup-nextflow/action.yml` names
    `nextflow plugin install` in prose precisely because that command is one of
    test_release_workflow_graph.BLOCKING_CHECKS' needles. Measured: replacing the
    real command with `echo DELETED-PROVISIONING` while leaving the comment
    standing left the release-parity guard GREEN (14 passed). A comment must
    never be able to satisfy a claim about what CI runs.

    Same rule, same reason, as `test_ci_stack_pinned.pip_invocations`, which has
    stripped `#` lines since it was written: a comment explaining why a pin was
    removed necessarily quotes the pin.
    """
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _run_bodies(steps) -> list[str]:
    return [str(s["run"]) for s in steps if isinstance(s, dict) and "run" in s]


def _action_run_bodies(path: Path) -> list[str]:
    action = yaml.safe_load(path.read_text()) or {}
    return _run_bodies((action.get("runs") or {}).get("steps") or [])


def resolved_text(path: Path) -> str:
    """The text of `path` plus every local action it reaches, COMMENTS STRIPPED.

    Still contains YAML prose (`description:` blocks), so it is the right view for
    "does this file mention X" and the WRONG view for "does CI run the command X".
    Use `resolved_run_text` for the latter.
    """
    return strip_comment_lines("\n".join(p.read_text() for p in resolved_paths(path)))


def resolved_run_text(path: Path) -> str:
    """Every `run:` body a WORKFLOW executes, its local actions included.

    Comments stripped, and `description:`/`name:` prose never present at all --
    this is a view of COMMANDS. It is what a guard asserting "CI runs X" must read.
    """
    bodies: list[str] = []
    for _name, job in load_jobs(path).items():
        bodies += _run_bodies(job.get("steps") or [])
    for action in resolved_action_files(path.read_text()):
        bodies += _action_run_bodies(action)
    return strip_comment_lines("\n".join(bodies))


# ---------------------------------------------------------------------------
# Reusable-workflow calls
# ---------------------------------------------------------------------------
def workflow_file(name: str) -> Path | None:
    """The workflow file for `./.github/workflows/<name>`, or None if missing."""
    candidate = WORKFLOWS_DIR / name
    return candidate if candidate.is_file() else None


def called_local_workflows(path: Path) -> dict[str, Path]:
    """`{job_id: called workflow}` for each job in `path` that calls one.

    Only a job whose BODY is `uses: ./.github/workflows/<x>.yml` counts — that is
    what a reusable-workflow call is. A `uses:` on a STEP is a composite action
    and belongs to `local_actions_used`; GitHub does not allow a workflow there.
    """
    out: dict[str, Path] = {}
    for job_id, job in load_jobs(path).items():
        if not isinstance(job, dict):
            continue
        for name in LOCAL_WORKFLOW_USES_RE.findall(str(job.get("uses", ""))):
            called = workflow_file(name)
            if called is not None:
                out[job_id] = called
    return out


def called_workflow_closure(path: Path) -> list[Path]:
    """Every local workflow reachable from `path` through job-level `uses:`.

    Transitive and cycle-safe. `path` itself is not included.
    """
    out: list[Path] = []
    pending = list(called_local_workflows(path).values())
    seen = {path}
    while pending:
        nxt = pending.pop(0)
        if nxt in seen:
            continue
        seen.add(nxt)
        out.append(nxt)
        pending.extend(called_local_workflows(nxt).values())
    return out


# ---------------------------------------------------------------------------
# Comment-blind command view
# ---------------------------------------------------------------------------
# TWICE ON 2026-09-01 A GUARD ON THIS REPOSITORY PASSED BECAUSE A COMMENT
# SATISFIED IT: a comment naming `nextflow plugin install` satisfied a
# release-parity needle, and a commit's header prose naming an owner satisfied an
# owner check. A scan that asks "is this command RUN" must therefore read `run:`
# bodies only, and must drop the shell comments inside them too — a `#` line in a
# script explaining why a step was removed necessarily quotes the thing removed.


def _run_bodies_of_file(path: Path) -> list[str]:
    """Every `run:` script in one file: workflow jobs' steps and an action's steps."""
    data = yaml.safe_load(path.read_text()) or {}
    out: list[str] = []
    for job in (data.get("jobs") or {}).values():
        if isinstance(job, dict):
            out += [str(s["run"]) for s in steps_of(job) if "run" in s]
    for step in (data.get("runs") or {}).get("steps") or []:
        if isinstance(step, dict) and "run" in step:
            out.append(str(step["run"]))
    return out


def _strip_shell_comments(script: str) -> str:
    """Drop `#` comments from a shell script — whole-line AND trailing.

    QUOTE-AWARE, because a naive "cut at the first `#`" is wrong on this
    repository's own scripts: `VERSION="${VERSION#v}"` (build-images.yml) and
    `echo "version_no_v=${VERSION#v}"` (release.yml) both carry a `#` inside
    double quotes that is parameter expansion, not a comment. Cutting there would
    silently truncate two real commands.

    The rule implemented is the shell's own: a `#` starts a comment when it is
    outside quotes and is either at the start of the line or preceded by
    whitespace. `${VERSION#v}` fails the second half (it follows `N`), and both
    occurrences fail the first half as well (they are inside `"`).

    The trailing case is closed rather than documented-around because "a comment
    satisfied a guard" has fired three separate times on this repository. The one
    real trailing comment in the tree is `nf-core lint --dir . || true  # nf-core
    lint is advisory, not blocking`, which is exactly the shape someone would use
    to leave a command's name behind after deleting it.
    """
    out = []
    for line in script.splitlines():
        kept = strip_line_comment(line)
        if kept.strip():
            out.append(kept.rstrip())
    return "\n".join(out)


def strip_line_comment(line: str) -> str:
    """One line with its `#` comment removed, quote-aware. `""` if it was all comment.

    The single definition of "where does the comment start" in this module. Exposed
    because guards that must report a LINE NUMBER cannot use
    `_strip_shell_comments` (it drops lines, so the numbering shifts) and would
    otherwise re-derive the rule with `line.split("#", 1)[0]` — the naive cut that
    truncates `VERSION="${VERSION#v}"`.
    """
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i == 0 or line[i - 1].isspace():
                return line[:i]
    return line


def strip_shell_comments(script: str) -> str:
    """`script` with whole-line AND trailing `#` comments removed, quote-aware.

    `_strip_shell_comments` above drops whole-LINE comments, which is the right
    scope for `run_scripts`'s "does CI run this command" question and NOT enough
    for "has this command been commented out": `true  # bash tests/cleanup_work.sh`
    leaves the needle on a line that is otherwise a real command, and a guard
    matching the text reports that CI still runs something it has stopped running.
    Measured on this repository on 2026-09-01, twice.

    Quote-aware, so a `#` inside a string literal (a colour, a URL fragment, a
    `sed 's/#.*//'`) is kept. It is a heuristic and not a shell parser -- it knows
    nothing about here-documents or backslash-escaped quotes -- so use it for
    "is this command still invoked" questions and not to execute anything.
    """
    out = []
    for line in script.splitlines():
        kept: list[str] = []
        in_single = in_double = False
        for char in line:
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            elif char == "#" and not in_single and not in_double:
                break
            kept.append(char)
        out.append("".join(kept))
    return "\n".join(out)


def run_scripts(path: Path) -> str:
    """Every `run:` body in `path` and in every local action it uses, comment-free.

    Both whole-line and TRAILING `#` comments are dropped from each script (see
    `_strip_shell_comments`). What is left is what the runner executes, which is
    the only thing a "does CI run this command" guard may be allowed to match.
    """
    scripts: list[str] = []
    for p in resolved_paths(path):
        scripts += _run_bodies_of_file(p)
    return "\n".join(_strip_shell_comments(s) for s in scripts)


def workflow_run_scripts(path: Path) -> str:
    """`run_scripts(path)` plus that of every local workflow its jobs call."""
    parts = [run_scripts(path)]
    parts += [run_scripts(p) for p in called_workflow_closure(path)]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Job-level views
# ---------------------------------------------------------------------------
def load_jobs(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    jobs = data.get("jobs")
    assert jobs, f"no `jobs:` block found in {path.relative_to(ROOT)}"
    return jobs


def steps_of(job: dict) -> list[dict]:
    return [s for s in (job.get("steps") or []) if isinstance(s, dict)]


def job_local_actions(job: dict) -> list[Path]:
    """Local action files a single job reaches through its `uses:` steps."""
    out: list[Path] = []
    for step in steps_of(job):
        for path in resolved_action_files(str(step.get("uses", ""))):
            if path not in out:
                out.append(path)
    return out


def job_run_text(job: dict) -> str:
    """The job's own `run:` bodies, in order, comments stripped."""
    return strip_comment_lines("\n".join(_run_bodies(steps_of(job))))


def job_resolved_run_text(job: dict) -> str:
    """The job's `run:` bodies PLUS the FULL TEXT of the local actions it uses.

    RAW. Comments included, and the action files are included whole rather than
    just their `run:` bodies -- an action's `uses:` steps (nf-core/setup-nextflow,
    actions/setup-python) are part of what the job runs, and a guard asking "does
    this job install Nextflow" has to be able to see a `uses:` line.

    THAT IS THE ONLY QUESTION THIS VIEW ANSWERS CORRECTLY. It is the wrong view
    for "does this job RUN command X", because a comment naming X satisfies it.
    That mistake has now been made four times on this repository, on four
    unrelated guards; the fourth was inside the guard written to close the third.
    The measured case: `.github/actions/setup-nextflow/action.yml` documents
    `nextflow plugin install` in prose BECAUSE that command is one of
    test_release_workflow_graph.BLOCKING_CHECKS' needles, so replacing the command
    with `echo DELETED-PROVISIONING` and leaving the comment standing left the
    parity check green.
    Use `job_run_scripts` for any command-runs claim, and `job_local_actions(job)`
    to ask which ACTIONS a job uses.
    """
    parts = [job_run_text(job)]
    parts += [p.read_text() for p in job_local_actions(job)]
    return "\n".join(parts)


def job_run_scripts(job: dict) -> str:
    """The job's `run:` bodies plus those of the local actions it uses, COMMENT-FREE.

    The per-job counterpart of `run_scripts`, and the view every "does this job
    run X" guard must use. Comments — whole-line and trailing — are stripped by
    `_strip_shell_comments`, so a command that has been COMMENTED OUT reads as
    absent, which is what it is at runtime.

    Commenting out is the likelier failure than deleting: it is what someone does
    to a flaky check under time pressure, and it leaves the literal string in the
    file for a raw-text guard to find.
    """
    scripts = [str(s["run"]) for s in steps_of(job) if "run" in s]
    for path in job_local_actions(job):
        scripts += [
            str(s["run"])
            for s in ((yaml.safe_load(path.read_text()) or {}).get("runs") or {}).get(
                "steps"
            )
            or []
            if isinstance(s, dict) and "run" in s
        ]
    return "\n".join(_strip_shell_comments(s) for s in scripts)


# ---------------------------------------------------------------------------
# What a job actually pip-installs
# ---------------------------------------------------------------------------
class UnresolvableCondition(Exception):
    """An `if:` inside a composite action this module cannot evaluate.

    Raised rather than guessed. Guessing "the step runs" would let a guard report
    that a job installs a package it does not, which is the exact direction of
    error this whole module exists to prevent; guessing "it does not" would hide
    a real install. Widen `_eval_action_if` instead.
    """


# The only `if:` form this repo's composite actions use: a comparison of one
# input against a string literal. `${{ }}` is optional — GitHub evaluates a step
# `if:` as an expression with or without the braces.
_ACTION_IF_RE = re.compile(
    r"^\s*(?:\$\{\{)?\s*inputs\.([A-Za-z0-9_-]+)\s*(==|!=)\s*'([^']*)'\s*(?:\}\})?\s*$"
)


def _action_inputs(action: dict, supplied: dict) -> dict[str, str]:
    """The input values a composite action sees: its defaults, overridden by `with:`."""
    values = {
        name: str(spec.get("default", ""))
        for name, spec in (action.get("inputs") or {}).items()
        if isinstance(spec, dict)
    }
    values.update({k: str(v) for k, v in supplied.items()})
    return values


def _eval_action_if(condition: str, inputs: dict[str, str]) -> bool:
    match = _ACTION_IF_RE.match(str(condition))
    if not match:
        raise UnresolvableCondition(
            f"cannot evaluate composite-action step condition {condition!r}. "
            "tests/ci_actions.py deliberately refuses to guess: a wrong guess here "
            "makes a guard report that a job installs something it does not. "
            "Teach _eval_action_if the new form."
        )
    name, op, literal = match.groups()
    if name not in inputs:
        raise UnresolvableCondition(
            f"condition {condition!r} reads an input {name!r} the action does not declare"
        )
    equal = inputs[name] == literal
    return equal if op == "==" else not equal


def requirements_from_step(
    step: dict, inputs: dict[str, str] | None = None
) -> list[str]:
    """requirements/*.txt files ONE step installs, in order.

    Three forms, all of which exist in this repo:

      1. a `run:` body containing `pip install -r requirements/<f>.txt`;
      2. `uses: ./.github/actions/setup-python-env` with a literal, newline
         separated `requirements:` input — the form every call site is REQUIRED to
         use (the action declares the input with no default precisely so that a
         file can never be installed without a job naming it);
      3. `uses: ./.github/actions/<other>` where the file is named literally
         inside that action (generate-testdata names requirements/testdata.txt).

    Form 3 is why this recurses through the action's own steps — and why it
    evaluates the `if:` on those steps against the inputs the caller passed.
    ci.yml's `python-tests` calls generate-testdata with `setup-python: 'false'`,
    so the requirements/testdata.txt install inside that action does NOT happen
    there, and reporting it would be a lie about the one job whose installs
    matter most.

    `inputs` is None for a step written in a workflow job (there is no input
    context) and a mapping for a step inside a composite action.
    """
    files: list[str] = []

    body = str(step.get("run", ""))
    if body:
        for m in re.finditer(
            r"(?:-r|--requirement)[=\s]+['\"]?(requirements/[A-Za-z0-9._-]+\.txt)", body
        ):
            files.append(m.group(1))

    uses = str(step.get("uses", ""))
    names = local_actions_used(uses)
    if not names:
        return files

    with_block = {k: v for k, v in (step.get("with") or {}).items()}
    declared = str(with_block.get("requirements", ""))
    if declared:
        files += REQUIREMENTS_FILE_RE.findall(declared)

    for name in names:
        path = action_file(name)
        if path is None:
            continue
        action = yaml.safe_load(path.read_text()) or {}
        nested_inputs = _action_inputs(action, with_block)
        nested = [
            s
            for s in ((action.get("runs") or {}).get("steps") or [])
            if isinstance(s, dict)
        ]
        for nested_step in nested:
            if "if" in nested_step and not _eval_action_if(
                nested_step["if"], nested_inputs
            ):
                continue
            files += requirements_from_step(nested_step, nested_inputs)

    return files


def requirement_files_installed(job: dict) -> list[str]:
    """Every requirements/*.txt file a JOB ends up installing, in order.

    Duplicates are preserved: order is what makes an install-order claim
    checkable, and de-duplicating would hide a file installed twice.
    """
    files: list[str] = []
    for step in steps_of(job):
        files += requirements_from_step(step)
    return files

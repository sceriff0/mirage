#!/usr/bin/env python3
"""Resolve a GitHub Actions workflow through the LOCAL COMPOSITE ACTIONS it uses.

CI steps live both in `.github/workflows/*.yml` and in composite actions under
`.github/actions/**`, so a guard that only globs the workflows directory misses
any step that sits in an action. The resolution lives here, in one place.

A workflow's effective step list is its own steps plus the steps of every
`uses: ./.github/actions/<name>` it references, transitively. Views:

  * `resolved_text(path)`: the workflow's text plus the text of every local
    action it reaches.
  * `job_resolved_run_text(job)`: the same, narrowed to one job.
  * `requirement_files_installed(job)`: the ordered `requirements/*.txt` files a
    job installs, whether written in the job, passed to `setup-python-env`, or
    named inside an action. Per job, so a sibling job's install cannot satisfy a
    check on another job.

A job whose body is `uses: ./.github/workflows/<x>.yml` (a reusable workflow
call) is not folded into those views: a caller job is not the called jobs, and
folding them in would let one leg of the suite satisfy a claim about another.
The called workflow's jobs are still scanned directly, because guards glob
`.github/workflows/`. Crossing that boundary has its own views:

  * `called_local_workflows(path)`: `{job_id: Path}` for each job in `path`
    that calls a local reusable workflow.
  * `workflow_run_scripts(path)`: `run_scripts(path)` plus that of every local
    workflow its jobs call, transitively.

Raw vs command view: `resolved_text` / `job_resolved_run_text` return literal
bytes, comments included, and are right only when the literal text is the
subject (a `uses:` line, a cache key). Any "CI runs this command" claim must use
the comment-stripped views (`run_scripts`, `workflow_run_scripts`,
`job_run_scripts`), because a comment naming a command satisfies the raw view.

Not a general YAML model: it answers only the questions this repo's CI guards ask.

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

# `uses: ./.github/actions/<name>`, matched on raw text so it works on action and
# workflow files alike (a `uses:` in a comment only widens a scan).
# The name may not start with a dot, so prose writing `actions/...` is not matched.
LOCAL_ACTION_USES_RE = re.compile(r"\./\.github/actions/([A-Za-z0-9_-][A-Za-z0-9._-]*)")

# `uses: ./.github/workflows/<file>.yml` as a JOB body — a reusable-workflow
# call. Same leading-dot exclusion as above; the `.yml`/`.yaml` suffix is
# required so prose naming the directory does not match.
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

    A name with no action.yml is skipped rather than raising;
    `test_ci_job_hygiene.test_every_local_action_referenced_exists` reports it.
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

    The views below are substring-matched by guards looking for a command, and
    this repo's YAML names its commands in comments. A comment must never satisfy
    a claim about what CI runs.
    """
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _run_bodies(steps) -> list[str]:
    return [str(s["run"]) for s in steps if isinstance(s, dict) and "run" in s]


def _action_run_bodies(path: Path) -> list[str]:
    action = yaml.safe_load(path.read_text()) or {}
    return _run_bodies((action.get("runs") or {}).get("steps") or [])


def resolved_text(path: Path) -> str:
    """The text of `path` plus every local action it reaches, comments stripped.

    Still contains YAML prose (`description:` blocks), so it suits "does this file
    mention X" but not "does CI run the command X".
    Use `resolved_run_text` for the latter.
    """
    return strip_comment_lines("\n".join(p.read_text() for p in resolved_paths(path)))


def resolved_run_text(path: Path) -> str:
    """Every `run:` body a workflow executes, its local actions included.

    Comments stripped and no `description:`/`name:` prose: a view of commands.
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

    Only a job whose body is `uses: ./.github/workflows/<x>.yml` counts. A `uses:`
    on a step is a composite action; GitHub does not allow a workflow there.
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
# A scan asking "is this command run" reads `run:` bodies only, with their shell
# comments dropped: a comment explaining a removed step quotes the thing removed.


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
    """Drop `#` comments from a shell script, whole-line and trailing.

    Quote-aware, using the shell's rule: a `#` starts a comment when it is outside
    quotes and at the start of the line or after whitespace. A naive cut would
    truncate `VERSION="${VERSION#v}"`, which is parameter expansion.
    """
    out = []
    for line in script.splitlines():
        kept = strip_line_comment(line)
        if kept.strip():
            out.append(kept.rstrip())
    return "\n".join(out)


def strip_line_comment(line: str) -> str:
    """One line with its `#` comment removed, quote-aware. `""` if it was all comment.

    The one definition of where a comment starts. Public for guards that report a
    line number and so cannot use `_strip_shell_comments`, which drops lines.
    Use it instead of `line.split("#", 1)[0]`, which truncates `${VERSION#v}`.
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

    Keeps line numbering (no lines are dropped). A trailing comment such as
    `true  # bash tests/cleanup_work.sh` must not read as the command still running.

    Quote-aware, so a `#` inside a string literal is kept. A heuristic, not a
    shell parser: it knows nothing about here-documents or escaped quotes.
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

    Whole-line and trailing `#` comments are dropped (see `_strip_shell_comments`),
    so what is left is what the runner executes.
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
    """The job's `run:` bodies plus the full text of the local actions it uses.

    Raw: action files are included whole, comments too, so a guard can see an
    action's `uses:` steps (e.g. nf-core/setup-nextflow). Wrong for "does this job
    run command X", since a comment naming X satisfies it: use `job_run_scripts`
    for that, and `job_local_actions(job)` for which actions a job uses.
    """
    parts = [job_run_text(job)]
    parts += [p.read_text() for p in job_local_actions(job)]
    return "\n".join(parts)


def job_run_scripts(job: dict) -> str:
    """The job's `run:` bodies plus those of the local actions it uses, comment-free.

    The per-job counterpart of `run_scripts`, for "does this job run X" guards. A
    commented-out command reads as absent, as it is at runtime.
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

    Raised rather than guessed: either guess would misreport what a job installs.
    Widen `_eval_action_if` instead.
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
         separated `requirements:` input (the input has no default, so a job must
         name every file);
      3. `uses: ./.github/actions/<other>` where the file is named literally
         inside that action (generate-testdata names requirements/testdata.txt).

    Form 3 is why this recurses through the action's steps and evaluates their
    `if:` against the caller's inputs: with `setup-python: 'false'`,
    generate-testdata does not install requirements/testdata.txt.

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

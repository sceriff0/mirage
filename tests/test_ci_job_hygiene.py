#!/usr/bin/env python3
"""Guard: workflow job hygiene — every job bounds its own runtime, and no job
sets up Docker buildx it never uses.

Without `timeout-minutes` a hung job burns GitHub's 6-hour default. A job that
sets up buildx but runs container-free (`-profile test`) only slows itself down.

Scope is every workflow under `.github/workflows/`, discovered by glob (do not
reintroduce a filename list), plus `.github/actions/**`: a job's steps include
those of every `uses: ./.github/actions/<name>` it references, resolved through
`tests/ci_actions.py`. Two checks own the actions directory: every referenced
action exists, and every action is referenced.

`timeout-minutes` is checked on workflows only: GitHub rejects it on a composite
action's steps, so those are bounded by the calling job's timeout.

PyYAML parses `jobs:` structurally; it is installed in CI, so the importorskip
does not silently skip there.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
ACTIONS_DIR = ROOT / ".github" / "actions"

# The shared workflow/composite-action resolver, imported rather than re-derived.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
ci_actions = importlib.import_module("ci_actions")


def workflow_files() -> list[Path]:
    """Every workflow under `.github/workflows/`, DISCOVERED BY GLOB."""
    files = sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOWS_DIR.relative_to(ROOT)}"
    return files


def _load_jobs(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    jobs = data.get("jobs")
    assert jobs, f"no `jobs:` block found in {path.relative_to(ROOT)}"
    return jobs


def _resolved_steps(job: dict) -> list[dict]:
    """A job's own steps PLUS the steps of every local composite action it uses.

    This is what "the steps this job runs" means since the Phase-2 move. Reading
    `job["steps"]` alone answers a question about YAML layout, not about CI.
    """
    steps = list(ci_actions.steps_of(job))
    for path in ci_actions.job_local_actions(job):
        action = yaml.safe_load(path.read_text()) or {}
        steps += [
            s
            for s in ((action.get("runs") or {}).get("steps") or [])
            if isinstance(s, dict)
        ]
    return steps


def test_every_job_declares_a_timeout():
    """A job with no `timeout-minutes` inherits GitHub Actions' 360-minute
    (6-hour) default — a hung `nextflow run` or `nf-test` invocation burns the
    whole runner budget before anything kills it. Every job must bound its own
    runtime explicitly.

    Reusable-workflow calls (a job whose body is `uses: ./.github/workflows/x.yml`)
    are the one exception, and not by exemption: GitHub Actions *rejects*
    `timeout-minutes` on such a job. The bound is owned by the jobs of the called
    workflow, and since this guard globs every workflow file, those jobs are
    themselves in scope here.
    """
    missing = []
    for wf in workflow_files():
        for name, job in _load_jobs(wf).items():
            if "uses" in job:
                # Reusable-workflow call: timeout lives in the called workflow,
                # which this glob already covers.
                continue
            if "timeout-minutes" not in job:
                missing.append(f"{wf.relative_to(ROOT)}: {name}")
    assert not missing, (
        "job(s) declare no `timeout-minutes` and would inherit GitHub Actions' "
        "360-minute default if they ever hang:\n" + "\n".join(sorted(missing)) + "\n"
        "Add an explicit timeout to each."
    )


def test_buildx_job_must_actually_use_buildx():
    """A job that sets up `docker/setup-buildx-action` must actually use it.

    Two things count as using it, and they are the only two things buildx is for
    in this repo:

    * running something with a `docker` profile in one of the job's own run
      steps (`-profile ...docker` / `-with-docker`), i.e. a pipeline run that
      needs a container engine; or
    * a step that builds or pushes an image with a Docker action
      (`docker/build-push-action`, `docker/login-action`), i.e. the job's whole
      purpose IS the image build.

    The second clause covers `containers.yml`'s matrix job, which builds images
    and never runs a pipeline. A job matching neither clause has set up a
    container engine it does not use.
    """
    docker_profile_re = re.compile(r"-profile\s+[^\s]*docker|-with-docker")
    docker_build_actions = ("docker/build-push-action", "docker/login-action")

    offenders = []
    checked = 0
    for wf in workflow_files():
        for name, job in _load_jobs(wf).items():
            # RESOLVED steps, not `job["steps"]`: a buildx setup, a docker-profile
            # run, or an image build can all live in a composite action now.
            steps = _resolved_steps(job)
            uses_buildx = any(
                "docker/setup-buildx-action" in str(step.get("uses", ""))
                for step in steps
            )
            if not uses_buildx:
                continue
            checked += 1
            references_docker = any(
                docker_profile_re.search(str(step.get("run", "")))
                for step in steps
                if "run" in step
            )
            builds_an_image = any(
                any(a in str(step.get("uses", "")) for a in docker_build_actions)
                for step in steps
            )
            if not references_docker and not builds_an_image:
                offenders.append(f"{wf.relative_to(ROOT)}: {name}")

    # Non-vacuity: a "must not find" scan over an empty set proves nothing, and
    # this repo does have buildx jobs. If this ever hits zero, the discovery
    # predicate broke, not the repository.
    assert checked >= 1, (
        "no job in any workflow sets up docker/setup-buildx-action, so this scan "
        "examined nothing. Either buildx left CI entirely, or the step was "
        "reworded past the `docker/setup-buildx-action` match."
    )
    assert not offenders, (
        "job(s) set up docker/setup-buildx-action but neither run anything with "
        "a docker profile (`-profile ...docker` / `-with-docker`) nor build an "
        "image with a Docker action:\n" + "\n".join(sorted(offenders)) + "\n"
        "The buildx setup step is dead weight and should be removed."
    )


# ---------------------------------------------------------------------------
# The composite actions directory itself
# ---------------------------------------------------------------------------
def test_every_local_action_referenced_exists():
    """A typo'd `uses: ./.github/actions/<name>` fails at RUNTIME, on the runner.

    GitHub reports it as "Can't find 'action.yml' ... under
    '/home/runner/work/.../.github/actions/<name>'", after the checkout, in
    whichever job referenced it -- so a rename that misses one call site is a red
    build for that job only, discovered on push. It is a static fact; check it
    statically.

    SCOPE is every workflow AND every action (an action can call an action:
    setup-nf-test calls setup-nextflow, generate-testdata calls setup-python-env).
    """
    missing = []
    referenced = 0
    for path in ci_actions.scanned_files():
        for name in ci_actions.local_actions_used(path.read_text()):
            referenced += 1
            if ci_actions.action_file(name) is None:
                missing.append(
                    f"{path.relative_to(ROOT)} references ./.github/actions/{name}, "
                    "which has no action.yml"
                )
    assert referenced >= 1, (
        "nothing under .github/ references a local composite action, so this scan "
        "examined nothing. If the composite actions were deliberately removed, "
        "delete this test with them; otherwise the `uses:` form was reworded past "
        "ci_actions.LOCAL_ACTION_USES_RE."
    )
    assert not missing, "\n".join(sorted(missing))


def test_every_job_checks_out_before_using_a_local_action():
    """`uses: ./.github/actions/<name>` reads the action out of the WORKSPACE.

    Before `actions/checkout` runs there is no workspace, and GitHub fails the job
    with "Can't find 'action.yml' ... under
    '/home/runner/work/<repo>/<repo>/.github/actions/<name>'" -- a message that
    reads like the action is missing rather than like the checkout is late. It is
    a static ordering fact, so it is checked statically.

    A reusable-workflow call (`uses: ./.github/workflows/x.yml` as a JOB body) is
    not affected: GitHub resolves that from the repository, not the workspace.
    Such a job has no `steps:` and is skipped here on its own.
    """
    offenders = []
    checked = 0
    for wf in workflow_files():
        for name, job in _load_jobs(wf).items():
            first_checkout = None
            for index, step in enumerate(ci_actions.steps_of(job)):
                uses = str(step.get("uses", ""))
                if uses.startswith("./.github/actions/"):
                    checked += 1
                    if first_checkout is None or first_checkout > index:
                        offenders.append(
                            f"{wf.relative_to(ROOT)}: {name} uses {uses} at step "
                            f"{index} with no earlier actions/checkout"
                        )
                elif "actions/checkout" in uses and first_checkout is None:
                    first_checkout = index
    assert checked >= 1, (
        "no job uses a local composite action, so this scan examined nothing."
    )
    assert not offenders, "\n".join(sorted(offenders))


def test_every_local_action_is_used():
    """An action nothing calls is dead YAML that reads as live infrastructure.

    Same rule this repo already applies to `bin/utils/*.py`
    (tests/test_no_dead_bin_modules.py): a module nothing imports is deleted, not
    kept "just in case".
    """
    used = set()
    for path in ci_actions.scanned_files():
        used.update(ci_actions.local_actions_used(path.read_text()))
    orphans = sorted(
        p.parent.name for p in ci_actions.action_files() if p.parent.name not in used
    )
    assert not orphans, (
        "composite action(s) under .github/actions/ that nothing references: "
        f"{orphans}. Wire them up or delete them."
    )


def test_every_composite_action_run_step_declares_a_shell():
    """`shell:` is REQUIRED on a composite action's `run:` step and optional on a
    workflow's, which is the single easiest way to write an action that parses,
    reads correctly, and fails on the runner with "Required property is missing:
    shell".

    actionlint catches this too, in CI; this check also runs in the local Python
    suite.
    """
    offenders = []
    steps_seen = 0
    for path in ci_actions.action_files():
        action = yaml.safe_load(path.read_text()) or {}
        for step in (action.get("runs") or {}).get("steps") or []:
            if not isinstance(step, dict) or "run" not in step:
                continue
            steps_seen += 1
            if not step.get("shell"):
                offenders.append(
                    f"{path.relative_to(ROOT)}: step "
                    f"{step.get('name', '<unnamed>')!r} has `run:` but no `shell:`"
                )
    if not ci_actions.action_files():
        pytest.skip("no composite actions in this tree")
    assert steps_seen >= 1, (
        "the composite actions contain no `run:` step at all, so this scan "
        "examined nothing."
    )
    assert not offenders, "\n".join(sorted(offenders))

#!/usr/bin/env python3
"""Guard: one commit must flow through the whole release.

`.github/workflows/release.yml` used to build a release from THREE different
commits, and every job in it was green while doing so:

  * `test` (the gate) did a BARE `actions/checkout@v4`, i.e. `github.ref` — the
    branch the workflow_dispatch was fired from;
  * `bump-and-tag` checked out `ref: main`, bumped `manifest.version` there,
    committed and tagged THAT, regardless of the dispatch ref; and
  * `artifacts`, `publish-images` and `verify-config` all carried
    `needs: [validate, test]` — NOT `needs: bump-and-tag` — and did a bare
    checkout of their own.

Because `needs` is what orders jobs, `artifacts` and `publish-images` ran IN
PARALLEL with the version bump, from the UN-BUMPED tree. `mirage-v1.0.0.tar.gz`
would have shipped a `nextflow.config` still reading the previous version, and
the ten container images — whose tags are exactly `manifest.version`, which
`modules/local/*.nf` pull by name — would have been built from that same stale
tree. Nothing in the run would have looked wrong.

So the invariants below are about the SHAPE of the job graph, which is the only
place this defect is visible:

  1. every `actions/checkout` in release.yml names an explicit `ref`, drawn from
     a small allowed vocabulary — a bare checkout is the bug;
  2. every gate job checks out the SAME pinned SHA that `validate` resolved;
  3. the jobs that produce release artifacts depend on `bump-and-tag` and build
     from the CREATED TAG;
  4. those same jobs tolerate `bump-and-tag` being SKIPPED, because it is
     skipped on the `release: published` path and a job whose dependency is
     skipped is itself skipped — so without `always()` this fix would silently
     delete the artifact and image jobs from that trigger entirely; and
  5. the gate actually runs CI's blocking checks, asserted by the commands
     themselves rather than by job name.

Every "must not find anything" assertion here is paired with one proving a real
graph was parsed, per this repo's convention for zero-answer checks.

Plain pytest, paths derived from `__file__`, per the other guard tests in this
directory (test_ci_stack_pinned.py, test_ci_job_hygiene.py).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent

# The shared workflow/composite-action resolver. Since the CI redesign's Phase 2
# (2026-09-01) a job's steps are its own steps PLUS the steps of every
# `uses: ./.github/actions/<name>` it references, and Invariant 5 below reads
# COMMANDS out of those steps -- `nextflow plugin install` lives in
# .github/actions/setup-nextflow now, not in either workflow. Reading only
# `CI.read_text()` and the gate jobs' own `run:` bodies would have made that
# parametrised case fail on a correct tree, and worse, would have made the whole
# invariant unable to see any check that moves into an action later.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
ci_actions = importlib.import_module("ci_actions")
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"
CI = WORKFLOWS / "ci.yml"
CONTAINERS = WORKFLOWS / "containers.yml"

# The reusable image-build workflow, exactly as release.yml must name it.
CONTAINERS_USES = "./.github/workflows/containers.yml"

# The shared blocking test suite is DISCOVERED, never named. It is defined as
# "the local reusable workflow that BOTH gates transitively require", which is
# the property Phase 5 exists to create; hardcoding `_test-suite.yml` here would
# make a rename of that file look like a passing test over a broken graph, and
# would make "the two gates run the same thing" unfalsifiable by construction.
# See shared_gate_suite() below.

# `validate` resolves these; everything downstream reads them. Expressed as the
# expression text the workflow must contain, because that is what actually has
# to match for the checkout to land on the right commit.
PINNED_SHA = "needs.validate.outputs.sha"
RELEASE_TAG = "needs.validate.outputs.version"
GATE_REF = "steps.get_version.outputs.gate_ref"


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict), (
        f"{path.relative_to(ROOT)} did not parse to a mapping"
    )
    return data


def jobs_of(path: Path) -> dict:
    jobs = load(path).get("jobs")
    assert jobs, f"no `jobs:` block in {path.relative_to(ROOT)}"
    return jobs


def needs_of(job: dict) -> list[str]:
    """`needs:` normalised to a list — GitHub accepts a bare string too."""
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def run_text(job: dict) -> str:
    """WHAT A JOB RUNS: its `run:` bodies plus those of the composite actions it
    uses, with comments stripped.

    RESOLVED. A gate job that calls `./.github/actions/setup-nextflow` really does
    run `nextflow plugin install`; a scan that reads only the job's own `run:`
    blocks would report that the release gate stopped provisioning plugins the day
    the step was de-duplicated.

    AND COMMENT-BLIND. Every caller of this helper asks a "does this job run X"
    question -- discovery of a gate aggregator, of an artifact producer, of an
    advisory suite; the `rev-parse HEAD` check; the inline version check; the
    `!= "success"` comparison. On the raw view a COMMENT naming X answers all of
    them, and a commented-out command is the likelier way any of these actually
    gets disabled: it is what someone does to a flaky check under time pressure,
    and it leaves the literal string behind for a raw-text scan to find.

    Measured: with `job_resolved_run_text` here, commenting out the four-line
    `needs.ruff.result` block in the shared suite's aggregator -- leaving the
    string `needs.ruff.result` sitting in a comment --
    `test_every_gate_aggregator_reads_every_result_it_needs` PASSED, over a gate
    that no longer checks that leg at runtime. That is the same bug that guard was
    written to close, one level in.

    If a future check here genuinely asks about the files' literal bytes (a
    `uses:` line, a cache key), call `ci_actions.job_resolved_run_text` at that
    site and say why -- do not widen this helper back.
    """
    return ci_actions.job_run_scripts(job)


def checkout_steps(job: dict) -> list[dict]:
    return [
        step
        for step in (job.get("steps") or [])
        if "actions/checkout" in str(step.get("uses", ""))
    ]


def workflow_files() -> list[Path]:
    """Every workflow, DISCOVERED BY GLOB, never a filename list.

    The hardcoded form of this scope is exactly how the `skipped`-as-a-pass check
    below came to be checking one of the two files that carry a result aggregator:
    it read `release.yml`'s `test` job and nothing else, so re-adding
    `&& [ "${{ needs.ruff.result }}" != "skipped" ]` to ci.yml's `all-tests` left
    the whole suite green (62 passed).
    """
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOWS.relative_to(ROOT)}"
    return files


# A `run:` step reading `needs.<job>.result` is what a gate aggregator IS. Both
# ci.yml's `all-tests` and release.yml's `test` are nothing but a list of these.
RESULT_READ = re.compile(r"needs\.[A-Za-z0-9_.-]+\.result")


def result_aggregator_jobs(path: Path) -> dict:
    """Jobs that decide pass/fail by comparing `needs.<job>.result` in a `run:`
    step — DISCOVERED, never named, so a renamed or a THIRD aggregator is covered
    the day it is added."""
    return {
        job_id: job
        for job_id, job in jobs_of(path).items()
        if isinstance(job, dict) and RESULT_READ.search(run_text(job))
    }


def release_jobs() -> dict:
    return jobs_of(RELEASE)


def artifact_producing_jobs() -> dict:
    """Jobs whose OUTPUT is a release artifact — DISCOVERED, never named.

    Two kinds, and they are the two things a release ships:
      * the job that calls the reusable image build (`uses:` containers.yml); and
      * any job that tars up `mirage-<version>.tar.gz`.

    Naming them instead would let a third artifact job be added tomorrow with a
    bare checkout and no `bump-and-tag` dependency, and this file would not
    notice — which is precisely how `artifacts` and `publish-images` came to be
    wired to the wrong commit in the first place.
    """
    found = {}
    for job_id, job in release_jobs().items():
        if not isinstance(job, dict):
            continue
        if CONTAINERS_USES in str(job.get("uses", "")):
            found[job_id] = job
            continue
        text = run_text(job)
        if "tar " in text and "mirage-" in text:
            found[job_id] = job
    return found


def gate_jobs() -> dict:
    """The jobs the `test` aggregator depends on — read off its `needs:`."""
    jobs = release_jobs()
    assert "test" in jobs, (
        "release.yml has no `test` job. Everything downstream is gated on it by name; "
        "if it was renamed, this whole file is checking a graph that no longer exists."
    )
    ids = needs_of(jobs["test"])
    return {job_id: jobs[job_id] for job_id in ids if job_id in jobs}


# ---------------------------------------------------------------------------
# The shared blocking suite, DISCOVERED from the two gates
# ---------------------------------------------------------------------------
def _gate_ancestors(jobs: dict, roots: list[str]) -> set[str]:
    """Jobs a gate transitively `needs:` — UPSTREAM ONLY.

    Distinct from `_needs_dag_connected` below, which walks both directions on
    purpose. Here the question is "what does the gate REQUIRE", and release.yml's
    downstream publish chain (`publish-images`, which calls containers.yml) is
    emphatically not part of that. Walking both directions would have made
    `containers.yml` come back as a workflow the release gate requires.
    """
    reached = set(roots)
    changed = True
    while changed:
        changed = False
        for job_id in list(reached):
            job = jobs.get(job_id)
            if not isinstance(job, dict):
                continue
            for dep in needs_of(job):
                if dep not in reached:
                    reached.add(dep)
                    changed = True
    return reached - set(roots)


def gate_required_workflows(path: Path) -> set[Path]:
    """Local reusable workflows the gate in `path` transitively requires."""
    jobs = jobs_of(path)
    aggregators = sorted(result_aggregator_jobs(path))
    required = _gate_ancestors(jobs, aggregators)
    return {
        called
        for job_id, called in ci_actions.called_local_workflows(path).items()
        if job_id in required
    }


def shared_gate_suite() -> Path:
    """The one reusable workflow BOTH gates require. Discovered, not named.

    This IS Phase 5's invariant, expressed as a lookup rather than as prose: if
    the two gates stop reaching the same file, every needle test below fails at
    this call instead of quietly checking one workflow twice.
    """
    ci_required = gate_required_workflows(CI)
    release_required = gate_required_workflows(RELEASE)
    shared = ci_required & release_required
    assert len(shared) == 1, (
        "ci.yml's gate and release.yml's gate do not require exactly one reusable "
        f"workflow in common (ci: {sorted(p.name for p in ci_required)}, release: "
        f"{sorted(p.name for p in release_required)}, shared: "
        f"{sorted(p.name for p in shared)}).\n"
        "Since 2026-09-01 the blocking suites live in ONE reusable workflow that both "
        "gates call, which is what makes CI/release parity structural instead of a "
        "claim in two headers that had already diverged once. Restore the shared "
        "`uses:`, or -- if the suites really were split again -- restore the parity "
        "comparison this file deleted when they were merged."
    )
    return next(iter(shared))


# ---------------------------------------------------------------------------
# Non-vacuity: the graph parsed, and it is the graph this file is about.
# ---------------------------------------------------------------------------


def test_the_release_graph_parsed_and_has_the_jobs_this_file_reasons_about():
    """Every assertion below is a lookup into this graph. If the parse yields an
    empty or unrecognisable one, they all pass having checked nothing."""
    jobs = release_jobs()
    for required in ("validate", "test", "bump-and-tag", "artifacts", "publish-images"):
        assert required in jobs, (
            f"release.yml has no `{required}` job. Either it was renamed — in which case "
            "the assertions in this file are silently checking a different graph — or the "
            "release flow lost a stage."
        )
    assert len(jobs) >= 8, (
        f"release.yml declares only {len(jobs)} job(s): {sorted(jobs)}. The gate alone is "
        "five; a graph this small means the parse or the workflow lost most of it."
    )


def test_validate_declares_the_outputs_the_rest_of_the_graph_reads():
    """Every downstream `ref:` is one of validate's outputs. An undeclared output
    renders as an EMPTY STRING in an expression — so `ref: ''` would silently mean
    "actions/checkout's default ref", i.e. exactly the bare checkout this file
    forbids, while every textual assertion below still passed."""
    outputs = release_jobs()["validate"].get("outputs") or {}
    for name in ("version", "version_no_v", "gate_ref", "sha"):
        assert name in outputs, (
            f"release.yml's `validate` job declares no `{name}` output, so every "
            f"`needs.validate.outputs.{name}` downstream evaluates to the empty string "
            "and the checkout falls back to the default ref."
        )


# ---------------------------------------------------------------------------
# Invariant 1 + 2: no bare checkout; the gate tests one pinned commit.
# ---------------------------------------------------------------------------


def test_no_job_in_release_yml_does_a_bare_checkout():
    """A bare `actions/checkout` takes `github.ref`, which on a workflow_dispatch is
    whatever branch the operator happened to fire from and on a `release: published`
    is the tag. Mixing the two is how the gate, the tag and the tarball came from
    three different commits. Every checkout here must say which commit it means."""
    # Two different kinds of allowed value, and they need two different tests.
    # An EXPRESSION is matched as a substring because it is embedded in `${{ ... }}`.
    # A LITERAL branch name must match EXACTLY: `"main" in ref` also accepts
    # `refs/heads/maintenance`, `mainline`, `not-main` and anything else with those
    # four characters in it, so the substring form let a checkout of an entirely
    # different branch through while reporting that it had checked the ref.
    expressions = (PINNED_SHA, RELEASE_TAG, GATE_REF)
    literals = ("main",)
    allowed = expressions + literals

    seen = 0
    offenders = []
    for job_id, job in release_jobs().items():
        if not isinstance(job, dict):
            continue
        for step in checkout_steps(job):
            seen += 1
            ref = str((step.get("with") or {}).get("ref", "")).strip()
            if not ref:
                offenders.append(
                    f"{job_id}: `{step.get('name', '<unnamed>')}` has no `ref:`"
                )
            elif ref not in literals and not any(token in ref for token in expressions):
                offenders.append(
                    f"{job_id}: `{step.get('name', '<unnamed>')}` checks out {ref!r}, which "
                    f"is none of {allowed} (literal branch names must match exactly)"
                )

    # Non-vacuity: an empty `offenders` also means "found no checkout steps at all",
    # which would be a parse failure wearing a pass.
    assert seen >= 5, (
        f"only {seen} actions/checkout step(s) found across release.yml's jobs; the "
        "release flow has more than that, so this scan did not read the file it thinks "
        "it read."
    )
    assert not offenders, (
        "release.yml job(s) check out an unspecified or unexpected ref:\n"
        + "\n".join(offenders)
    )


def test_every_gate_job_checks_out_the_one_pinned_commit():
    """The gate runs on several runners. If each resolves `main` (or the tag) for
    itself, a push landing mid-run splits the gate across two trees and nothing
    reports it. `validate` resolves the ref ONCE to a SHA and the whole gate runs
    on that SHA.

    TWO SHAPES since 2026-09-01, and the second is the one that matters. A gate
    job either checks the repository out itself -- then its `actions/checkout`
    must name the pinned SHA -- or it CALLS a reusable workflow, in which case it
    has no steps at all and the SHA has to be handed over as an input, because a
    called workflow does its own checkout against the CALLER's `github.ref`. That
    is the same defect containers.yml's `workflow_call` block documents, one
    step to the left: here it would make the gate TEST a commit other than the one
    `bump-and-tag` tags.

    The floor is no longer "five jobs": the five hand-copied gate jobs became one
    `uses:` of the shared suite. What replaces that non-vacuity is
    `test_the_shared_test_suite_honours_the_ref_it_is_given`, which asserts the
    called workflow really has the jobs and really reads the input.
    """
    gate = gate_jobs()
    assert gate, (
        "release.yml's `test` job needs nothing this file can resolve to a job. Either "
        "the gate lost its `needs:` or the job ids no longer exist, and every "
        "assertion below is over an empty set."
    )
    wrong = []
    saw_pinned = 0
    for job_id, job in gate.items():
        called = str(job.get("uses", ""))
        if called:
            passed = str((job.get("with") or {}).get("ref", ""))
            if PINNED_SHA not in passed:
                wrong.append(
                    f"{job_id}: calls {called} with ref={passed!r}; a reusable workflow "
                    f"checks out the CALLER's github.ref unless given "
                    f"`ref: ${{{{ {PINNED_SHA} }}}}`"
                )
            else:
                saw_pinned += 1
            continue
        steps = checkout_steps(job)
        assert steps, f"gate job `{job_id}` never checks out the repository at all"
        for step in steps:
            ref = str((step.get("with") or {}).get("ref", ""))
            if PINNED_SHA not in ref:
                wrong.append(f"{job_id}: checks out {ref!r}, not {PINNED_SHA}")
            else:
                saw_pinned += 1
    assert not wrong, (
        "gate job(s) do not run on the commit `validate` pinned, so the gate can test "
        "a different tree from the one that gets tagged:\n" + "\n".join(wrong)
    )
    assert saw_pinned >= 1, (
        f"no gate job names {PINNED_SHA} at all ({sorted(gate)}). An empty `wrong` list "
        "here would only mean the scan found no checkout and no reusable call."
    )


def test_bump_and_tag_is_gated_on_the_test_aggregator():
    """The whole point of the gate. Without `test` in its `needs:`, a tag lands on a
    tree nothing verified."""
    needs = needs_of(release_jobs()["bump-and-tag"])
    assert "test" in needs, (
        f"release.yml's `bump-and-tag` needs {needs}, which does not include `test`. "
        "It would tag an unverified commit."
    )


def test_bump_and_tag_refuses_to_tag_a_commit_the_gate_did_not_test():
    """`bump-and-tag` checks out `main` by branch name, not by the pinned SHA, because
    it has to PUSH to that branch. That reopens the gap on a smaller scale if main
    moves mid-run, so it must compare HEAD against validate's pinned SHA and fail."""
    text = run_text(release_jobs()["bump-and-tag"])
    assert PINNED_SHA in text and "rev-parse HEAD" in text, (
        "release.yml's `bump-and-tag` does not compare the checked-out HEAD against "
        f"`{PINNED_SHA}`. If main moves while the gate is running it would tag a commit "
        "no gate job tested."
    )


# ---------------------------------------------------------------------------
# Invariant 3 + 4: artifacts come from the created tag, after the bump.
# ---------------------------------------------------------------------------


def test_artifact_jobs_depend_on_bump_and_tag():
    """THE blocker. `needs:` is the only thing that orders jobs; without
    `bump-and-tag` in it, the tarball and the images are built in parallel with the
    version bump, from the un-bumped tree."""
    producers = artifact_producing_jobs()
    # Non-vacuity: the discovery is by shape (a `uses:` of containers.yml, or a
    # `tar` of mirage-*.tar.gz). If it finds nothing, every assertion below is over
    # an empty set.
    assert len(producers) >= 2, (
        f"discovered only {sorted(producers)} as release-artifact producers. Expected at "
        "least the image build and the archive job — the discovery predicates "
        "(a `uses:` of containers.yml, or a `tar` of a mirage-* archive) no longer "
        "match what release.yml does."
    )
    missing = [
        job_id
        for job_id, job in producers.items()
        if "bump-and-tag" not in needs_of(job)
    ]
    assert not missing, (
        f"release-artifact job(s) {sorted(missing)} do not `need: bump-and-tag`, so they "
        "run in PARALLEL with the version bump and build from the un-bumped tree. That is "
        "the defect that would have shipped a mirage-v1.0.0.tar.gz containing a "
        "nextflow.config reading the previous version."
    )


def test_artifact_jobs_build_from_the_created_tag():
    """Depending on `bump-and-tag` only fixes the ORDER. The jobs must also check out
    the tag it created, or they still build from whatever `github.ref` happens to be."""
    producers = artifact_producing_jobs()
    assert producers, "no artifact-producing jobs discovered"

    wrong = []
    for job_id, job in producers.items():
        if CONTAINERS_USES in str(job.get("uses", "")):
            # A reusable workflow does its own checkout and sees the CALLER's
            # github.ref, so the tag has to be handed to it as an input.
            passed = str((job.get("with") or {}).get("ref", ""))
            if RELEASE_TAG not in passed:
                wrong.append(
                    f"{job_id}: calls {CONTAINERS_USES} with ref={passed!r}; a reusable "
                    f"workflow checks out the CALLER's github.ref unless given "
                    f"`ref: ${{{{ {RELEASE_TAG} }}}}`"
                )
            continue
        steps = checkout_steps(job)
        assert steps, f"artifact job `{job_id}` never checks out the repository"
        for step in steps:
            ref = str((step.get("with") or {}).get("ref", ""))
            if RELEASE_TAG not in ref:
                wrong.append(f"{job_id}: checks out {ref!r}, not the released tag")
    assert not wrong, (
        "release-artifact job(s) do not build from the created tag:\n"
        + "\n".join(wrong)
    )


def test_containers_workflow_honours_the_ref_it_is_given():
    """The other half of the same invariant, in the called workflow. release.yml can
    pass `ref:` all it likes; if containers.yml's own `actions/checkout` steps ignore
    it, the images are still built from the caller's ref."""
    data = load(CONTAINERS)
    triggers = data.get("on") or data.get(True) or {}
    call_inputs = ((triggers.get("workflow_call") or {}).get("inputs")) or {}
    assert "ref" in call_inputs, (
        "containers.yml's `workflow_call` declares no `ref` input, so release.yml "
        "cannot tell it which commit to build the images from."
    )

    seen = 0
    ignoring = []
    for job_id, job in jobs_of(CONTAINERS).items():
        if not isinstance(job, dict):
            continue
        for step in checkout_steps(job):
            seen += 1
            if "inputs.ref" not in str((step.get("with") or {}).get("ref", "")):
                ignoring.append(f"{job_id}: `{step.get('name', '<unnamed>')}`")
    assert seen >= 2, (
        f"found {seen} checkout step(s) in containers.yml; both the matrix-resolving "
        "`setup` job and the `build` job check out the repo, so this scan missed one."
    )
    assert not ignoring, (
        "containers.yml checkout step(s) ignore the `ref` input, so a release builds "
        "images from the caller's ref rather than the tag:\n" + "\n".join(ignoring)
    )


def test_jobs_needing_bump_and_tag_tolerate_it_being_skipped():
    """The trap this fix creates if left half-done.

    `bump-and-tag` carries `if: github.event_name == 'workflow_dispatch'`, so on the
    `release: published` path it is SKIPPED — and GitHub skips any job whose dependency
    was skipped. Adding `needs: bump-and-tag` without `always()` therefore DELETES the
    archive and the image publish from the release-published trigger entirely, silently,
    and the release still reports green.
    """
    offenders = []
    checked = 0
    for job_id, job in release_jobs().items():
        if not isinstance(job, dict) or "bump-and-tag" not in needs_of(job):
            continue
        checked += 1
        cond = str(job.get("if", ""))
        if "always()" not in cond:
            offenders.append(
                f"{job_id}: needs `bump-and-tag` but its `if:` has no always(), so it is "
                "SKIPPED on the release:published path where bump-and-tag is skipped"
            )
        elif "skipped" not in cond:
            offenders.append(
                f"{job_id}: has always() but never allows bump-and-tag to be 'skipped', "
                "so the release:published path is still excluded"
            )
        elif "needs.test.result == 'success'" not in cond:
            offenders.append(
                f"{job_id}: uses always() without re-asserting `needs.test.result == "
                "'success'`, so a FAILED gate no longer blocks it -- always() bypasses "
                "the needs-failure rule too"
            )
    assert checked >= 2, (
        f"only {checked} job(s) in release.yml depend on `bump-and-tag`; "
        "test_artifact_jobs_depend_on_bump_and_tag should have caught that first."
    )
    assert not offenders, "\n".join(offenders)


def test_no_release_artifact_is_produced_from_a_tree_that_disagrees_with_the_tag():
    """`verify-config` failing must stop the release SHIPPING, not merely go red.

    It compares `nextflow.config`'s `manifest.version` against the released tag,
    and it runs only on the `release` trigger (on a dispatch, `bump-and-tag`
    writes that version, so there is nothing to disagree with). `artifacts`
    protects itself with an equivalent step of its own. `publish-images` had
    neither: on a `release: published` of a tag whose tree still said 0.9.0,
    `verify-config` went red, `artifacts` went red, and ten images tagged 1.0.0
    were built from the 0.9.0 tree and PUSHED.

    So the invariant is per-producer and satisfiable two ways — depend on
    `verify-config`, or carry the check inline. A job that `uses:` a reusable
    workflow has no steps of its own, so for it only the first way exists.
    """
    producers = artifact_producing_jobs()
    assert len(producers) >= 2, (
        f"discovered only {sorted(producers)} as release-artifact producers; the "
        "discovery predicates no longer match what release.yml does."
    )
    assert "verify-config" in release_jobs(), (
        "release.yml has no `verify-config` job, so the `needs:` half of this "
        "invariant names a job that does not exist."
    )

    offenders = []
    for job_id, job in producers.items():
        if "verify-config" in needs_of(job):
            continue
        text = run_text(job)
        if "nextflow.config" in text and "version_no_v" in text:
            continue
        offenders.append(
            f"{job_id}: neither `needs: verify-config` nor an inline check of "
            "nextflow.config's version against the released tag"
        )
    assert not offenders, (
        "release-artifact job(s) can ship from a tree whose nextflow.config "
        "disagrees with the tag:\n" + "\n".join(offenders)
    )


def test_jobs_needing_verify_config_tolerate_it_being_skipped():
    """The same trap `bump-and-tag` sets, from the other trigger.

    `verify-config` carries `if: github.event_name == 'release'`, so on a
    workflow_dispatch it is SKIPPED — and GitHub skips any job whose dependency
    was skipped. Adding `needs: verify-config` without allowing 'skipped'
    therefore DELETES the image publish from the dispatch path entirely, silently,
    and the release still reports green.
    """
    checked = 0
    offenders = []
    for job_id, job in release_jobs().items():
        if not isinstance(job, dict) or "verify-config" not in needs_of(job):
            continue
        checked += 1
        cond = str(job.get("if", "")).replace('"', "'")
        if "always()" not in cond:
            offenders.append(
                f"{job_id}: needs `verify-config` but its `if:` has no always()"
            )
        elif "needs.verify-config.result == 'skipped'" not in cond:
            offenders.append(
                f"{job_id}: needs `verify-config` but never allows it to be 'skipped', so "
                "the workflow_dispatch path loses this job entirely"
            )
        elif "needs.verify-config.result == 'success'" not in cond:
            offenders.append(
                f"{job_id}: never requires `verify-config` to have SUCCEEDED, so a red "
                "config check no longer blocks it -- always() bypasses the needs-failure "
                "rule too"
            )
    assert checked >= 1, (
        "no job in release.yml depends on `verify-config`; "
        "test_no_release_artifact_is_produced_from_a_tree_that_disagrees_with_the_tag "
        "should have caught that first."
    )
    assert not offenders, "\n".join(offenders)


# ---------------------------------------------------------------------------
# Invariant 5: both gates run the same blocking checks, and it is still a real
# list of checks rather than an empty agreement.
# ---------------------------------------------------------------------------
# REWRITTEN 2026-09-01 (CI redesign Phase 5), AND THE HALF THAT WAS DELETED IS
# NAMED HERE so the next reader can judge the trade rather than trust it.
#
# What this used to assert, per needle: (a) the command appears in ci.yml, and
# (b) the command appears in the `run:` bodies of release.yml's gate jobs. Half
# (b) is gone. It compared two hand-maintained copies of the same five jobs, and
# there is now ONE copy: both gates reach the blocking suites through
# `uses: ./.github/workflows/<the shared suite>`, so "the release gate runs what
# CI blocks on" is a fact about `uses:` and `needs:`, checked once by
# test_the_two_gates_require_the_same_reusable_test_suite, instead of a
# string-by-string comparison of two files that can drift.
#
# What is NOT redundant, and is therefore kept verbatim: the list itself. Parity
# between two gates says nothing about whether either runs anything -- two gates
# agreeing on an empty suite agree perfectly. So every needle below is now
# asserted against the SHARED SUITE, and separately against each caller's
# transitive text, which is what would fail if a caller dropped the `uses:`.
#
# One entry per blocking check, keyed on the COMMAND rather than the job name.
BLOCKING_CHECKS = [
    ("python unit suite", "pytest -v tests/"),
    # ("DISK front-end non-skip proof", "tests/test_coarse_frontend.py") was here until
    # 2026-09-27: the DISK + LightGlue COARSE front-end it proved was not skipped is gone,
    # the anchor imports no torch, and the whole-session MIRAGE_STRICT_SKIPS floor
    # (tests/conftest.py) already fails the suite step on any unexpected skip.
    ("nf-test stub suite", "nf-test test --tag stub"),
    ("ruff lint", "ruff check ."),
    ("ruff format (blocking since release 1.0 phase 02)", "ruff format --check ."),
    ("lib/ probe (the only unit-test surface lib/*.groovy has)", "tests/lib_probe.nf"),
    (
        "profiles parse without the gitignored site config",
        "tests/check_profiles_parse.sh",
    ),
    ("param surface consistency", "tests/check_param_consistency.py"),
    ("pipeline validation script", "tests/run_validation_tests.sh"),
    ("work-directory cleanup", "tests/cleanup_work.sh"),
    # Added 2026-09-02 (plan 01). Same class as cleanup_work.sh: a shell check that
    # measures a session-teardown / session-to-session OUTCOME, so it can be neither
    # a pytest nor an nf-test, and it therefore needs its own step or it runs
    # nowhere. It ran nowhere -- it was named in the developer notes as "the
    # behavioural resume check" and appeared in no workflow file.
    ("resume determinism (behavioural)", "tests/resume_check.sh"),
    # Added 2026-09-06. Same class again: whether a cleanup_level pin REACHES the
    # publish gates is an evaluation-order outcome that `nextflow config` reports
    # wrongly (it prints the pinned value while the gates are already frozen), so
    # only a run can observe it.
    ("cleanup_level pin reaches the gates (behavioural)", "tests/cleanup_level_pin.sh"),
    ("shipped-defaults stub run", "-profile shipped_defaults_test"),
    ("DEEPCELL token-leak guard", "tests/modules/segment_deepcell_token.nf.test"),
    ("token-guard collection assertion", "--dry-run"),
    ("Nextflow plugin provisioning", "nextflow plugin install"),
    # Added 2026-09-01 with the composite actions. actionlint is BLOCKING and it
    # lives in the shared suite's `ruff` job, which every gate requires. It was
    # deliberately NOT put in ci.yml's old advisory `lint` job -- that job ran
    # `nf-core lint ... || true` and could never fail, and Phase 7 deleted it
    # outright. Both needles resolve through .github/actions/** like the rest of
    # this table.
    #
    # The needle is the COMMAND, not `rhysd/actionlint`: Phase 2's fix round
    # replaced the anonymous Docker Hub pull with a pinned, checksum-verified
    # release tarball inside `.github/actions/actionlint`, so the image name no
    # longer appears in any `run:` body and matching it would assert the old
    # implementation rather than the check.
    ("actionlint (workflow + composite-action syntax)", "actionlint"),
    # Added by phase 07. The format suite runs the CONVERT_IMAGE reader stack
    # against synthesised pyramidal/BigTIFF/RGB/8-bit/float/HDF5/NDPI fixtures.
    # The needle is the DIRECTORY form, which is also what keeps this suite out
    # of python-tests' collection: `pytest tests/` (the blocking python suite's
    # needle, above) would match a `tests/` run, not this one.
    (
        "format suite (convert image's reader stack)",
        "pytest -v tests/integration/formats",
    ),
    # Added 2026-09-02 (plan 12). The RTD site is a release artefact: a broken
    # internal link or a page that fell out of the nav ships as a 404 on
    # mirage-pipeline.readthedocs.io, and nothing else in this suite reads
    # mkdocs.yml. `--strict` is the whole point of the needle -- `mkdocs build`
    # without it exits 0 on every warning this check exists to catch.
    ("mkdocs strict site build", "mkdocs build --strict"),
    # Added 2026-09-04. The execution counterpart to the documented-command
    # static guard: every `nextflow run` a doc page shows is actually launched
    # under --dry_run. It ran as the last step of nextflow-stub from plan 12 and
    # now has its own job, so a deleted step would otherwise have no symptom
    # beyond a faster build.
    (
        "documented commands launch (execution leg)",
        "tests/documented_commands_launch.sh",
    ),
]


@pytest.mark.parametrize(
    "label,needle", BLOCKING_CHECKS, ids=[c[0] for c in BLOCKING_CHECKS]
)
def test_the_shared_test_suite_runs_every_blocking_check(label, needle):
    """Each blocking check must still be RUN, in the suite both gates require.

    The 13 in this list are the ones release.yml did not run while claiming it
    did: the whole nf-test stub suite, ruff, the DEEPCELL token-leak guard and its
    collection assertion, tests/lib_probe.nf, check_profiles_parse.sh,
    check_param_consistency.py, cleanup_work.sh and the shipped-defaults stub run.
    Merging the two copies fixed WHERE they run; it does not stop one of them being
    deleted, and a deleted check has no symptom.

    Resolved through the composite actions, so a check that moves into
    `.github/actions/**` (as `nextflow plugin install` already has) still resolves.

    COMMENT-BLIND, and that is not a detail. The previous version of this test
    matched against the whole file text, so a COMMENT naming a command satisfied
    it -- which happened, on this repository, on 2026-09-01: a comment naming
    `nextflow plugin install` passed the needle for a workflow that had stopped
    running it. `ci_actions.run_scripts` reads `run:` bodies only and drops the
    shell comments inside them. Proved by putting each needle in a comment and
    watching this fail.
    """
    suite = shared_gate_suite()
    assert needle in ci_actions.run_scripts(suite), (
        f"{needle!r} ({label}) is not RUN anywhere in {suite.name}, the suite both "
        "gates require. It was a BLOCKING check in ci.yml; if it was deliberately "
        "removed, remove it from BLOCKING_CHECKS in the same commit, and say why. "
        "(A comment mentioning it does not count -- this reads `run:` bodies with "
        "their comments stripped.)"
    )
    # And each caller must actually reach it. This is what would fail if ci.yml or
    # release.yml dropped the `uses:` — the case the deleted string-by-string
    # comparison used to cover, at one assertion instead of two files' worth.
    for caller in (CI, RELEASE):
        assert needle in ci_actions.workflow_run_scripts(caller), (
            f"{caller.name} does not reach {needle!r} ({label}) through anything it "
            f"calls. {suite.name} runs it, so {caller.name} has stopped calling the "
            "shared suite."
        )


def test_the_two_gates_require_the_same_reusable_test_suite():
    """PHASE 5's INVARIANT, and the reason the parity comparison above shrank.

    `ci.yml`'s `all-tests` and `release.yml`'s `test` must both transitively
    require the SAME local reusable workflow. While that holds, "the release gate
    runs what CI blocks on" is not a claim about two files' contents — it is the
    same YAML, once.

    It is asserted UPSTREAM-ONLY. release.yml's publish chain also calls a
    reusable workflow (containers.yml), downstream of the gate; counting that
    would make the release gate look like it requires the image build.
    """
    suite = shared_gate_suite()  # asserts there is exactly one, with the reason

    # Non-vacuity: the shared file must be a real suite, not an empty stub that
    # two gates trivially agree about.
    suite_jobs = jobs_of(suite)
    assert len(suite_jobs) >= 5, (
        f"{suite.name} declares only {sorted(suite_jobs)}. Both gates agreeing on a "
        "suite that runs almost nothing is a perfect agreement about nothing."
    )
    triggers = load(suite).get("on") or load(suite).get(True) or {}
    assert "workflow_call" in triggers, (
        f"{suite.name} has no `workflow_call:` trigger, so neither gate can actually "
        "call it and this graph does not run."
    )


def test_the_shared_test_suite_gate_covers_every_job_in_it():
    """A leg missing from the suite's own aggregator is invisible to both gates.

    A caller reads ONE result: the called workflow's conclusion. A workflow whose
    jobs were all SKIPPED concludes `success`, so the enumeration of legs has to
    exist somewhere, and it lives in the suite's aggregator. A job left out of
    that `needs:` list can stop running without either gate noticing — the exact
    "a needed job that stops running must block the merge" failure both callers'
    aggregators were rewritten to close.
    """
    suite = shared_gate_suite()
    jobs = jobs_of(suite)
    aggregators = sorted(result_aggregator_jobs(suite))
    assert len(aggregators) == 1, (
        f"{suite.name} carries {aggregators} gate aggregator(s); it needs exactly one "
        "job that reads every leg's `needs.<job>.result`, or the callers' single "
        "result check is the only thing standing between a skipped leg and a green "
        "merge."
    )
    gate_id = aggregators[0]
    covered = set(needs_of(jobs[gate_id]))
    # The SAME allowlist test_every_job_is_connected_to_its_workflows_gate uses, not a
    # second one: an exemption must be a single fact, so that deleting it re-arms both
    # checks at once and test_the_gate_disconnected_allowlist_has_no_dead_entries can
    # force the deletion the moment the job is wired in.
    exempt = {
        job_id
        for (file_name, job_id) in GATE_DISCONNECTED_JOBS
        if file_name == suite.name
    }
    uncovered = sorted(set(jobs) - covered - {gate_id} - exempt)
    assert not uncovered, (
        f"{suite.name}: job(s) {uncovered} are not in `{gate_id}`'s `needs:`, so a skip "
        "or a removal there is invisible to both gates. Add them, and see "
        "test_every_gate_aggregator_reads_every_result_it_needs for the other half: "
        "being in `needs:` is not the same as being CHECKED. If a job is deliberately "
        "outside the gate, put it in GATE_DISCONNECTED_JOBS with a reason -- that is "
        "the ONE allowlist, shared with "
        "test_every_job_is_connected_to_its_workflows_gate."
    )


def test_every_gate_aggregator_reads_every_result_it_needs():
    """`needs:` is the plumbing. The SCRIPT is the check, and they can disagree.

    A job listed in an aggregator's `needs:` but never read in its `run:` body is
    a leg nothing looks at. The aggregator carries `if: always()`, so it runs
    whatever its dependencies did; if the script never compares that leg's
    `result`, the aggregator exits 0, the workflow concludes `success`, and every
    caller reads `success`. That is exactly the all-skipped hazard the aggregators
    exist to close, one level down -- per LEG instead of per suite.

    Measured, on a tree where it was possible: deleting the four-line
    `needs.ruff.result` block from the shared suite's aggregator while leaving
    `ruff` in its `needs:` list left this whole file GREEN (43 passed). Add a
    `paths:` filter to `ruff` after that and a skipped lint merges.

    REPO-WIDE, over every aggregator this file can discover -- not only the one
    that was found broken. `ci.yml`'s `all-tests` and `release.yml`'s `test` have
    the same shape (each needs one job and reads one result), and the shape is
    what is being checked, not one instance of it.
    """
    offenders = []
    checked = 0
    for path in workflow_files():
        for gate_id, gate in result_aggregator_jobs(path).items():
            deps = needs_of(gate)
            # An aggregator that needs nothing cannot be checking anything, and
            # would otherwise contribute zero to the floor below and pass.
            assert deps, (
                f"{path.name}: gate aggregator `{gate_id}` declares no `needs:` at all, "
                "so it reports on nothing while looking like a gate."
            )
            text = run_text(gate)
            for dep in deps:
                checked += 1
                if f"needs.{dep}.result" not in text:
                    offenders.append(
                        f"{path.name}: `{gate_id}` needs `{dep}` but never reads "
                        f"`needs.{dep}.result` in its script, so that job's result "
                        "cannot fail the gate"
                    )
    # Non-vacuity: this is a "must not find" scan over a DISCOVERED set. The
    # current count is 7 (ci.yml `all-tests` 1, release.yml `test` 1, the shared
    # suite's aggregator 5); the floor is below that so a matrix or a legitimate
    # graph change does not trip it, but a collapse to nothing does.
    assert checked >= 5, (
        f"only {checked} aggregator dependency(ies) examined across "
        f"{[p.name for p in workflow_files()]}. The shared test suite's aggregator "
        "alone needs five jobs, so this scan did not read what it thinks it read."
    )
    assert not offenders, (
        "gate aggregator(s) list a job in `needs:` and never check its result:\n  "
        + "\n  ".join(offenders)
        + "\n\nAn `if: always()` aggregator runs regardless of what its dependencies "
        "did, so an unchecked leg can fail or skip and the gate still exits 0."
    )


def test_the_shared_test_suite_honours_the_ref_it_is_given():
    """The other half of the pinned-commit invariant, inside the called workflow.

    A caller can pass `ref:` all it likes; if the called workflow's own
    `actions/checkout` steps ignore it, every job there checks out the CALLER's
    `github.ref` instead. On the release path that means the gate tests a
    different commit from the one `bump-and-tag` tags — containers.yml's
    `workflow_call` block records the same defect one step to the right, where it
    built ten container images from an un-bumped tree.
    """
    suite = shared_gate_suite()
    call_inputs = (
        (
            (load(suite).get("on") or load(suite).get(True) or {}).get("workflow_call")
            or {}
        ).get("inputs")
    ) or {}
    assert "ref" in call_inputs, (
        f"{suite.name}'s `workflow_call` declares no `ref` input, so neither caller "
        "can tell it which commit to test."
    )
    assert call_inputs["ref"].get("required") is True, (
        f"{suite.name}'s `ref` input is not `required: true`. An optional ref means a "
        "caller that forgets it silently tests the caller's github.ref, which on the "
        "release path is not the commit being tagged."
    )
    assert "default" not in call_inputs["ref"], (
        f"{suite.name}'s `ref` input declares a default, which is the same hole as "
        "making it optional."
    )

    seen = 0
    ignoring = []
    for job_id, job in jobs_of(suite).items():
        if not isinstance(job, dict):
            continue
        for step in checkout_steps(job):
            seen += 1
            if "inputs.ref" not in str((step.get("with") or {}).get("ref", "")):
                ignoring.append(f"{job_id}: `{step.get('name', '<unnamed>')}`")
    assert seen >= 5, (
        f"found {seen} checkout step(s) in {suite.name}; every one of its jobs checks "
        "the repository out, so this scan missed some."
    )
    assert not ignoring, (
        f"{suite.name} checkout step(s) ignore the `ref` input, so the gate runs on the "
        "caller's ref rather than the commit it was handed:\n" + "\n".join(ignoring)
    )


def test_every_reusable_workflow_call_in_release_yml_names_its_ref():
    """A `uses:` of a local workflow IS a checkout, and takes the same rule.

    test_no_job_in_release_yml_does_a_bare_checkout scans `actions/checkout`
    steps, which a reusable-workflow call has none of — so without this, the one
    shape that produced this workflow's worst bug (ten images built from the wrong
    tree) would be the one shape exempt from the rule.
    """
    allowed = (PINNED_SHA, RELEASE_TAG, GATE_REF)
    calls = ci_actions.called_local_workflows(RELEASE)
    assert len(calls) >= 2, (
        f"release.yml calls {sorted(calls)} local reusable workflow(s). It calls the "
        "shared test suite and the image build, so a smaller number means the "
        "discovery no longer matches a `uses: ./.github/workflows/...` job body."
    )
    offenders = []
    for job_id, called in sorted(calls.items()):
        passed = str(
            (jobs_of(RELEASE)[job_id].get("with") or {}).get("ref", "")
        ).strip()
        if not any(token in passed for token in allowed):
            offenders.append(
                f"{job_id}: calls {called.name} with ref={passed or '<none>'!r}, which is "
                f"none of {allowed}"
            )
    assert not offenders, (
        "release.yml reusable-workflow call(s) do not name the commit they are for. A "
        "called workflow checks out the CALLER's github.ref, so an unset `ref:` input "
        "silently runs against a different tree:\n" + "\n".join(offenders)
    )


def test_both_workflows_carry_a_gate_aggregator_this_file_can_find():
    """Non-vacuity for the two scans below, which are "must not find anything"
    checks over a DISCOVERED set. ci.yml (`all-tests`) and release.yml (`test`)
    each carry exactly the one aggregator; if the discovery predicate stops
    matching, the `skipped` scan passes having read nothing."""
    found = {
        path.name: sorted(result_aggregator_jobs(path)) for path in workflow_files()
    }
    for name in ("ci.yml", "release.yml"):
        assert found.get(name), (
            f"no job in {name} reads `needs.<job>.result` in a `run:` step, so this "
            "file cannot see its gate aggregator. Either the gate was rewritten in a "
            "form this predicate does not match, or the file lost its gate."
        )


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_no_gate_aggregator_treats_skipped_as_a_pass(path):
    """`skipped` is not `success`, in EITHER workflow.

    ci.yml's `all-tests` and release.yml's `test` are the same gate written twice,
    and both used to read `!= "success" && != "skipped"` — harmless only while
    nothing in the needs list can skip. The moment one of those jobs gains a
    `paths:`/`if:` filter the gate goes decorative for it and reports green having
    verified nothing.

    This scan is over EVERY workflow, not the two named files: the previous
    version read release.yml's `test` job only, so the reviewer put the
    `skipped` clause straight back into ci.yml and the whole suite stayed green.
    """
    for job_id, job in result_aggregator_jobs(path).items():
        text = run_text(job)
        assert '!= "success"' in text, (
            f"{path.name}: job `{job_id}` reads `needs.<job>.result` but never compares "
            'it against "success"; it is not checking anything.'
        )
        assert "skipped" not in text, (
            f"{path.name}: job `{job_id}` allows a `skipped` needed job to count as a "
            "pass. A needed job that stops running must block, not be waved through."
        )


# `needs.<dep>.result` followed by a comparison. The gate writes
#     if [ "${{ needs.ruff.result }}" != "success" ]; then
# so between the `}}` and the operator there is a closing quote and whitespace; an
# occurrence with nothing comparable after it (an `echo` reporting the value) yields
# no match, which is exactly the distinction this is for.
_RESULT_COMPARISON_TEMPLATE = (
    r"needs\.{dep}\.result\s*\}}\}}\s*[\"\']?\s*(!=|==)\s*[\"\']?([A-Za-z_]+)"
)


def _result_comparisons(text: str, dep: str) -> set:
    """`{(operator, literal)}` for every comparison applied to `needs.<dep>.result`.

    A dependency can be MENTIONED without being CHECKED -- every branch in these
    aggregators echoes the value it just rejected -- so "the substring appears" is
    not the question. The operator is not either: `== "failure"` reads like a check
    and passes `cancelled`, `skipped` and every future conclusion GitHub adds.
    """
    pattern = _RESULT_COMPARISON_TEMPLATE.format(dep=re.escape(dep))
    return set(re.findall(pattern, text))


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_every_gate_aggregator_reads_every_job_it_needs(path):
    """A job in the gate's `needs:` that the gate never CHECKS is not gated.

    `needs:` alone gives you two of the three things you want -- ordering, and a
    skip when the dependency fails outright -- and not the third: with
    `if: always()`, which every aggregator here carries so that it can report
    rather than vanish, a FAILED dependency no longer stops the job. The result
    check in the script is the whole gate at that point, so a job added to `needs:`
    without a matching `!= "success"` branch is required in appearance and
    advisory in fact.

    THIS IS THE `alls-green` PROPERTY, and adopting `alls-green` was considered and
    declined to get it. That action derives the check from `toJSON(needs)`, which
    would make the mistake unmakeable -- but it also removes the `run:` step that
    reads `needs.<job>.result`, and THAT is how `result_aggregator_jobs` discovers
    a gate at all. Every gate assertion in this file is built on that discovery
    (the `skipped` scan, the connectivity scan, `shared_gate_suite`), so swapping
    the shell check for a third-party action would blind them while looking like an
    improvement. Asserting the property directly costs one test and no dependency.
    """
    for job_id, job in result_aggregator_jobs(path).items():
        text = run_text(job)
        problems = []
        for dep in needs_of(job):
            comparisons = _result_comparisons(text, dep)
            if not comparisons:
                problems.append(
                    f"`{dep}`: never compared at all"
                    + (
                        " (the name appears, but only in a message)"
                        if f"needs.{dep}.result" in text
                        else ""
                    )
                )
            elif ("!=", "success") not in comparisons:
                problems.append(
                    f"`{dep}`: compared as {sorted(comparisons)}, never as "
                    '`!= "success"`'
                )
        assert not problems, (
            f"{path.name}: job `{job_id}`'s `needs:` and its result checks disagree:\n  "
            + "\n  ".join(problems)
            + "\n\nWith `if: always()` on the aggregator -- which every one here carries "
            "so that it reports rather than vanishes -- a FAILED dependency no longer "
            "stops the job, so the comparison in the script IS the gate. A dependency "
            "with no comparison is required in appearance and advisory in fact; one "
            'compared as `== "failure"` instead of `!= "success"` lets `cancelled` '
            "and `skipped` straight through, which is the same hole one operator to "
            'the left. Add a `!= "success"` branch for each, or take it out of '
            "`needs:` and put the job in a workflow nothing is gated on."
        )


def test_no_workflow_job_sets_continue_on_error():
    """`continue-on-error: true` makes a job or step report a green tick whatever
    it did, so the run summary stops saying whether the suite got better or worse.

    ci.yml's `nf-test-real` carried it as a SECOND advisory mechanism on top of
    simply being absent from `all-tests`' `needs:` — 129 real cases showing green
    unconditionally. It was removed; nothing forbade it coming back. That job now
    lives in nightly.yml (Phase 4), which this scan reaches because the scope is a
    glob and never a filename list — and it matters MORE there, not less: a
    nightly suite that is expected to be red is exactly where someone reaches for
    `continue-on-error` to quieten it, and quietening it is the one thing that
    would make the move pointless.

    STEP LEVEL IS FORBIDDEN TOO, AND THAT WAS RE-DECIDED IN PHASE 7 rather than
    inherited. The CI redesign's `|| true` register argued for narrowing this to
    JOB level, on the ground that job-level `continue-on-error` makes
    `needs.<job>.result` read `success` (it lies to the gate) while step-level does
    not (it is "a different thing and is safe"). Half of that is right: step-level
    is strictly narrower. It is still a green tick over a step that failed, which is
    the same defect one size down -- and there is no step in this repository that
    wants one. Narrowing a guard for a use nobody has is how a guard becomes
    decorative; the `allowed` dict below is the escape hatch, and it costs a written
    reason rather than a silent licence.

    If a legitimate use ever appears, allow-list it BY JOB ID here with the reason
    written down, rather than deleting the scan.
    """
    allowed: dict[tuple[str, str], str] = {}  # (workflow file name, job id) -> reason

    scanned = 0
    offenders = []
    for path in workflow_files():
        for job_id, job in jobs_of(path).items():
            if not isinstance(job, dict):
                continue
            scanned += 1
            where = f"{path.name}:{job_id}"
            if job.get("continue-on-error") and (path.name, job_id) not in allowed:
                offenders.append(f"{where}: job-level `continue-on-error`")
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                if step.get("continue-on-error") and (path.name, job_id) not in allowed:
                    offenders.append(
                        f"{where}: step `{step.get('name', '<unnamed>')}` sets "
                        "`continue-on-error`"
                    )

    # Non-vacuity: an empty `offenders` must mean "read the jobs and found none",
    # not "read no jobs".
    assert scanned >= 15, (
        f"scanned only {scanned} job(s) across {[p.name for p in workflow_files()]}; the "
        "workflows here declare far more than that, so this scan did not read what "
        "it thinks it read."
    )
    assert not offenders, (
        "`continue-on-error` makes a failing job report green. Make it advisory by "
        "LEAVING IT OUT of the gate's `needs:`, which reports its true colour:\n"
        + "\n".join(offenders)
    )


def test_no_workflow_branches_on_the_workflow_call_event_name():
    """A condition that can never be true, in the one place it costs a release.

    Inside a REUSABLE workflow the event payload is the CALLER's — GitHub's own
    words: "the event payload in the called workflow is the same event payload
    from the calling workflow". So `github.event_name` is never `workflow_call`;
    on a published release reaching containers.yml through release.yml's
    `publish-images`, it is `release`.

    containers.yml's version resolver branched on exactly that, could not match,
    fell through to its `push`-trigger default of `PUSH="false"`, and gave the
    operator a fully green "Publish Images" job with ZERO images on Docker Hub —
    while `modules/local/*.nf` pull `bolt3x/mirage-<component>:<manifest.version>`
    by name, so every process then fails to pull.

    A called workflow tells its two entry points apart by its `inputs`, which are
    populated for `workflow_dispatch` and `workflow_call` alike and absent on a
    plain `push`.
    """
    hits = []
    saw_event_name = 0
    for path in workflow_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            # Quote-aware, shared with every other comment-blind scan in this
            # file. `line.split("#", 1)[0]` was the previous form and truncates
            # `VERSION="${VERSION#v}"` mid-expansion.
            code = ci_actions.strip_line_comment(line)
            if "github.event_name" not in code:
                continue
            saw_event_name += 1
            if "workflow_call" in code:
                hits.append(f"{path.name}:{lineno}: {line.strip()}")

    # Non-vacuity: the scan has to have found real `github.event_name` uses, or a
    # clean result only means the files were not read.
    assert saw_event_name >= 4, (
        f"found only {saw_event_name} non-comment use(s) of `github.event_name` across "
        "the workflows; several jobs gate on it, so this scan missed them."
    )
    assert not hits, (
        "workflow(s) branch on `github.event_name` being 'workflow_call', which it "
        "never is — inside a called workflow the event name is the CALLER's:\n"
        + "\n".join(hits)
    )


# ---------------------------------------------------------------------------
# Invariant 6: a green workflow run means a green gate.
# ---------------------------------------------------------------------------
# Added 2026-09-01 with the CI redesign's Phase 4.
#
# A GitHub Actions run's conclusion is the WORST conclusion of any job in it.
# Gate membership does not enter into that: a job absent from `all-tests`'
# `needs:` is advisory to the MERGE and not to the RUN, so it still paints the
# commit, the badge and the PR checks list red. ci.yml carried `nf-test-real` and
# `nf-test-integration` on exactly those terms, and on every push to main or dev
# reported `failure` while its gate was green (verified on 84cc53f). People learn
# to ignore that red, which is the failure mode the whole redesign is against.
#
# The fix is location, not `continue-on-error`: an advisory suite lives in a
# workflow nothing is gated on -- .github/workflows/nightly.yml. The two checks
# below hold both halves of that.

# Jobs deliberately outside their workflow's gate DAG. BY (file, job id), with the
# reason, because an unexplained one is exactly the defect.
# EMPTY AS OF 2026-09-02 (CI redesign, Phase 7; and again after the `docs` fix
# round below), and that is the point rather than an oversight. The single Phase 7
# entry was ("ci.yml", "lint") -- the nf-core structural lint, whose one
# substantive step ended in `|| true` and therefore could never fail. Phase 7
# deleted the JOB: no released nf-core/tools version can lint this repository
# (2.14.1 through 4.1.0 all die on `manifest.name` having no `/`; see .nf-core.yml's
# header for the measurement), so there was nothing to un-swallow.
#
# A SECOND entry, ("_test-suite.yml", "docs"), existed briefly on 2026-09-02: the
# `docs` job was landed in `_test-suite.yml` outside `suite-gate`'s `needs:`, on the
# premise that being outside the gate's `needs:` made it non-blocking. CI run
# 33641966166 disproved that: `_test-suite.yml` is a `workflow_call` reusable
# workflow, and a called workflow's own CONCLUSION is the worst conclusion of ANY
# job inside it, gate membership included or not -- so the red `docs` job made
# `_test-suite.yml` itself conclude `failure`, and ci.yml's `all-tests` reads that
# one result regardless of `suite-gate`. The fix was relocation, not an exemption:
# `docs` now lives in `.github/workflows/nightly.yml`, which carries no gate
# aggregator at all, so there is no gate for it to be disconnected FROM and this
# allowlist has nothing to say about it. See nightly.yml's own job comment for the
# full account and the plan-12 return path.
#
# Every job in every gated workflow is now wired to its gate. Leave this dict empty
# if you can: an entry here is a job that cannot fail the MERGE and can still fail
# the RUN, which is how a green gate comes to report a red commit.
GATE_DISCONNECTED_JOBS: dict[tuple[str, str], str] = {}

# The advisory nf-test suites, keyed on the COMMAND, never on a job or file name.
# Naming the job would let a rename hide a deletion, and naming the file would go
# blind the day the workflow is split again.
ADVISORY_SUITES = {
    "nf-test real suite": "--tag real",
    "nf-test integration suite": "--tag integration",
    # "docs strict site build" lived here from 2026-09-02 until plan 12: the
    # `docs` job was relocated to nightly.yml because a called reusable
    # workflow's own conclusion is the worst conclusion of ANY job inside it
    # regardless of gate membership (CI run 33641966166). Plan 12 (2026-09-02)
    # moved it back into `_test-suite.yml` as a blocking, gate-enumerated job now
    # that the build is green -- see BLOCKING_CHECKS and
    # GATE_MEMBERSHIP["_test-suite.yml"].
}


def _needs_dag_connected(jobs: dict, roots: list[str]) -> set[str]:
    """Every job reachable from `roots` through `needs:`, in EITHER direction.

    Upstream (a job the gate needs) and downstream (a job that needs the gate, as
    release.yml's whole publish chain does) both count as connected: what matters
    is whether the job's result can reach the gate's verdict at all, not which way
    the arrow points.
    """
    connected = set(roots)
    changed = True
    while changed:
        changed = False
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            declared = set(needs_of(job))
            if job_id in connected and declared - connected:
                connected |= declared
                changed = True
            elif job_id not in connected and declared & connected:
                connected.add(job_id)
                changed = True
    return connected


def test_every_job_is_connected_to_its_workflows_gate():
    """In a workflow that HAS a gate, every job must be wired to it.

    A job with no `needs:` and nothing needing it is invisible to the gate and
    fully visible in the run status -- the worst of both. That was `nf-test-real`.

    Workflows with no gate aggregator (containers.yml, nightly.yml) are out of
    scope BY CONSTRUCTION, not by exemption: there is no gate for a job to be
    disconnected from, and nightly.yml being one of them is the entire point of
    it existing. They are counted, not skipped -- a skip is not a pass in this
    suite, and a per-file `pytest.skip` here would put two permanent entries in
    tests/expected_skips.txt for a condition that is not an environment fact.
    """
    offenders = []
    gated = []
    for path in workflow_files():
        aggregators = sorted(result_aggregator_jobs(path))
        if not aggregators:
            continue
        gated.append(path.name)
        jobs = jobs_of(path)
        connected = _needs_dag_connected(jobs, aggregators)
        offenders += [
            f"{path.name}: `{job_id}` is in neither direction of the {aggregators} DAG"
            for job_id in sorted(jobs)
            if job_id not in connected
            and (path.name, job_id) not in GATE_DISCONNECTED_JOBS
        ]
    # Non-vacuity: this is a "must not find" scan, and it examines only gated
    # workflows -- so a discovery predicate that stops matching turns it green.
    assert len(gated) >= 2, (
        f"only {gated} carries a gate aggregator. ci.yml (`all-tests`) and "
        "release.yml (`test`) both do; a smaller set means result_aggregator_jobs "
        "stopped recognising one of them and this scan read nothing."
    )
    assert not offenders, (
        "job(s) sit outside their workflow's gate DAG. Such a job cannot fail the "
        "MERGE and can still fail the RUN, so a green gate reports a red commit:\n  "
        + "\n  ".join(offenders)
        + "\n\nEither add it to the gate's `needs:`, or move it to a workflow that "
        "has no gate (nightly.yml), or add it to GATE_DISCONNECTED_JOBS with a "
        "reason that survives being read."
    )


def test_the_gate_disconnected_allowlist_has_no_dead_entries():
    """An exemption for a job that no longer exists, or that is now wired in, is a
    licence nobody is using and the next reader will trust."""
    stale = []
    for (file_name, job_id), reason in GATE_DISCONNECTED_JOBS.items():
        path = WORKFLOWS / file_name
        if not path.is_file():
            stale.append(f"{file_name}: no such workflow")
            continue
        jobs = jobs_of(path)
        if job_id not in jobs:
            stale.append(f"{file_name}: no job `{job_id}`")
            continue
        aggregators = sorted(result_aggregator_jobs(path))
        if aggregators and job_id in _needs_dag_connected(jobs, aggregators):
            stale.append(
                f"{file_name}: `{job_id}` IS connected to {aggregators} now -- "
                "delete the exemption"
            )
        assert reason.strip(), f"{file_name}:{job_id} has an empty reason"
    assert not stale, "\n".join(stale)


@pytest.mark.parametrize(
    "label,needle", sorted(ADVISORY_SUITES.items()), ids=sorted(ADVISORY_SUITES)
)
def test_each_advisory_suite_runs_in_a_workflow_nothing_is_gated_on(label, needle):
    """Two failures at once, and they pull in opposite directions.

    Moving a suite out of the blocking workflow is one keystroke away from
    DELETING it -- and a deleted advisory suite has no symptom at all, because its
    absence looks exactly like its silence. So: the command must still be run by
    some job somewhere.

    And it must be run somewhere that carries no gate aggregator, because a
    workflow whose run status people read as the merge verdict cannot host a suite
    that is expected to be red. Discovery is by COMMAND, so a renamed job or a
    third split still resolves.
    """
    homes = []
    for path in workflow_files():
        for job_id, job in jobs_of(path).items():
            if isinstance(job, dict) and needle in run_text(job):
                homes.append((path, job_id))
    assert homes, (
        f"no job in any workflow runs `nf-test ... {needle}` ({label}). The suite "
        "was moved to nightly.yml to keep ci.yml honest, not deleted -- and a "
        "deleted advisory suite is silent by definition. Restore it, or if it was "
        "removed on purpose, remove it from ADVISORY_SUITES in the same commit."
    )
    offenders = [
        f"{path.name}::{job_id}"
        for path, job_id in homes
        if result_aggregator_jobs(path)
    ]
    assert not offenders, (
        f"`nf-test ... {needle}` ({label}) runs in a workflow that carries a gate "
        f"aggregator: {offenders}. A run's conclusion is the worst conclusion of "
        "any job in it, so an advisory suite there reddens a workflow whose status "
        "is read as the merge verdict. That is the state Phase 4 removed."
    )


def test_the_integration_suite_is_still_manual_only():
    """`--tag integration` has NEVER run unattended, and the move must not change
    that.

    In ci.yml the job carried `if: github.event_name == 'workflow_dispatch'`, so
    it ran only when a human fired the workflow by hand. nightly.yml adds a
    `schedule:` trigger, and a job moved there with its `if:` dropped silently
    becomes a two-hour suite that runs every night against Docker Hub -- a change
    of behaviour that looks like a pure relocation in the diff. Asserted, keyed on
    the COMMAND rather than on the job's name or its file.

    The real suite deliberately has no such condition: running once a day on the
    default branch is exactly what it was moved here to do.
    """
    needle = ADVISORY_SUITES["nf-test integration suite"]
    checked = 0
    offenders = []
    for path in workflow_files():
        for job_id, job in jobs_of(path).items():
            if not isinstance(job, dict) or needle not in run_text(job):
                continue
            checked += 1
            cond = str(job.get("if", "")).replace('"', "'")
            if "workflow_dispatch" not in cond:
                offenders.append(
                    f"{path.name}::{job_id} runs `{needle}` under `if: {cond or '<none>'}`"
                )
    assert checked >= 1, (
        f"no job runs `nf-test ... {needle}`; "
        "test_each_advisory_suite_runs_in_a_workflow_nothing_is_gated_on should have "
        "caught that first."
    )
    assert not offenders, (
        "the nf-test integration suite is no longer restricted to a manual "
        "`workflow_dispatch`:\n  " + "\n  ".join(offenders) + "\n"
        "It has always been manual-only. If that is meant to change, change it "
        "deliberately and update this guard in the same commit."
    )


# ---------------------------------------------------------------------------
# Invariant 7: the MEMBERSHIP of each gate is a written-down fact.
# ---------------------------------------------------------------------------
# Added 2026-09-01, fix round 1 of Phase 4, from a reviewer's break.
#
# The reviewer moved the `nf-test-stub` job wholesale out of ci.yml into
# nightly.yml and dropped it from `all-tests`' `needs:`. **116 CI guards passed,
# 0 failed.** Phase 4 had just made that move idiomatic -- "advisory suites live
# in nightly.yml" reads as an invitation -- so the hole and the well-lit path to
# it arrived together.
#
# Two separate reasons nothing caught it, and closing either alone is not enough:
#
#   * `test_every_job_is_connected_to_its_workflows_gate` cannot see a job once it
#     is no longer in a workflow that HAS a gate. Its scope is gated workflows;
#     the job left that scope.
#   * `test_the_release_gate_runs_every_check_ci_blocks_on[nf-test stub suite]`
#     kept passing its ci.yml non-vacuity check, because `security-tests`'
#     collection assertion contains the substring `nf-test test --tag stub` too.
#     A COMMAND needle cannot distinguish "the suite runs" from "some other job
#     mentions the suite".
#
# So the missing assertion is about MEMBERSHIP, not about commands: which job ids
# the gate depends on, stated once, here.

# Each gated workflow's aggregator `needs:` set, written down. Keyed by file
# because gate membership IS a per-file fact -- but every gated workflow must
# appear (asserted below), so a new one cannot arrive undeclared.
GATE_MEMBERSHIP = {
    # Phase 5 moved the five blocking legs into ONE reusable workflow, so the
    # membership fact moved with them. ci.yml's `all-tests` and release.yml's
    # `test` now each require a single job -- the `uses:` call -- and the
    # enumeration lives where the legs live. Dropping a leg is still a diff
    # someone has to read HERE, which is the whole property; it is now written
    # once instead of twice, which is the reason the copies were merged.
    # `containers` is ci.yml's only blocking leg that is NOT in the shared suite:
    # Phase 6 put the PR image build in the gate deliberately (the caller job has
    # no `if:` and never skips, so an empty matrix one level down cannot wave the
    # gate through). release.yml builds images on the publish path instead, which
    # is downstream of its gate, so it has no counterpart here.
    "ci.yml": {"test-suite", "containers"},
    "release.yml": {"test-suite"},
    "_test-suite.yml": {
        "python-tests",
        "nextflow-stub",
        # Split out of nextflow-stub on 2026-09-04: 31 serial `nextflow run`
        # launches were ~400 s of that job and made it the gate's long pole.
        "documented-commands",
        "nf-test-stub",
        "security-tests",
        "format-tests",
        "ruff",
        "docs",
    },
}


def test_each_gate_needs_exactly_the_jobs_written_down_here():
    """Dropping a job from a gate's `needs:` must be a diff someone reads.

    A blocking check can be removed from CI in one line -- delete it from
    `needs:` -- and every other guard in this repository keeps passing, because
    they all reason about what the workflows CONTAIN rather than about what the
    gate REQUIRES. Reproduced by a reviewer, who removed `nf-test-stub` and got
    116 green CI guards.

    Every gated workflow must appear in GATE_MEMBERSHIP, so a third gate cannot
    be added without a line here.
    """
    gated = {}
    for path in workflow_files():
        aggregators = sorted(result_aggregator_jobs(path))
        if aggregators:
            gated[path.name] = (path, aggregators)

    assert set(gated) == set(GATE_MEMBERSHIP), (
        f"the set of workflows carrying a gate aggregator is {sorted(gated)}, but "
        f"GATE_MEMBERSHIP declares {sorted(GATE_MEMBERSHIP)}. A gate this file does "
        "not know about is a gate nothing checks the membership of."
    )

    problems = []
    for name, (path, aggregators) in gated.items():
        assert len(aggregators) == 1, (
            f"{name} has {aggregators}; this check assumes one aggregator per "
            "workflow and must be taught the second one deliberately."
        )
        jobs = jobs_of(path)
        actual = set(needs_of(jobs[aggregators[0]]))
        expected = GATE_MEMBERSHIP[name]
        if actual != expected:
            problems.append(
                f"{name}: `{aggregators[0]}` needs {sorted(actual)}, expected "
                f"{sorted(expected)} (missing {sorted(expected - actual)}, "
                f"unexpected {sorted(actual - expected)})"
            )
        for job_id in expected:
            if job_id not in jobs:
                problems.append(
                    f"{name}: GATE_MEMBERSHIP names `{job_id}`, which does not exist"
                )
    assert not problems, (
        "gate membership changed:\n  " + "\n  ".join(problems) + "\n\n"
        "If a check really should stop blocking, say so HERE in the same commit. "
        "Removing it from `needs:` alone leaves every other guard green."
    )


def test_every_nf_test_suite_is_either_blocking_or_declared_advisory():
    """`GATE_MEMBERSHIP` and `ADVISORY_SUITES` are COMPLEMENTS over the nf-test
    suites. A suite in neither is a new fact someone has to write down.

    This is the general form of the reviewer's break. A job that runs
    `nf-test test` is either:

      * wired into its own workflow's gate -- it blocks something; or
      * running one of the commands declared in ADVISORY_SUITES -- it is
        deliberately non-blocking, and nightly.yml is where it lives.

    A job in neither has stopped blocking without anyone declaring it advisory,
    which is precisely what "move it to nightly.yml" does by accident. Discovery
    is by the `nf-test test` invocation, so it holds for a suite this file has
    never heard of.
    """
    advisory_needles = tuple(ADVISORY_SUITES.values())
    checked = 0
    offenders = []
    for path in workflow_files():
        aggregators = sorted(result_aggregator_jobs(path))
        jobs = jobs_of(path)
        gate_needs = set()
        for agg in aggregators:
            gate_needs |= set(needs_of(jobs[agg]))
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            text = run_text(job)
            if "nf-test test" not in text:
                continue
            checked += 1
            if job_id in gate_needs:
                continue
            if any(needle in text for needle in advisory_needles):
                continue
            offenders.append(
                f"{path.name}::{job_id} runs `nf-test test` but is in neither "
                f"{path.name}'s gate `needs:` nor ADVISORY_SUITES"
            )
    assert checked >= 4, (
        f"only {checked} job(s) run `nf-test test` across the workflows; ci.yml, "
        "release.yml and nightly.yml between them run considerably more, so this "
        "scan did not read what it thinks it read."
    )
    assert not offenders, (
        "nf-test suite(s) neither block nor are declared advisory:\n  "
        + "\n  ".join(offenders)
        + "\n\nA suite that has quietly stopped blocking is the failure mode Phase 4 "
        "made easy: moving a job to nightly.yml looks like tidying. Put it back in "
        "the gate's `needs:` (and in GATE_MEMBERSHIP), or declare it in "
        "ADVISORY_SUITES with the reason it is allowed to be red."
    )

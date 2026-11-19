#!/usr/bin/env python3
"""Guard: one commit must flow through the whole release.

A release built from mixed commits stays green: a bare `actions/checkout` takes
`github.ref`, and jobs not ordered after `bump-and-tag` build from the un-bumped
tree, so the tarball and the container images (tagged `manifest.version`) would
ship stale. The invariants are about the shape of the job graph:

  1. every `actions/checkout` in release.yml names an explicit `ref` from a small
     allowed vocabulary;
  2. every gate job checks out the same pinned SHA that `validate` resolved;
  3. the jobs that produce release artifacts depend on `bump-and-tag` and build
     from the created tag;
  4. those jobs tolerate `bump-and-tag` being skipped (it is, on the
     `release: published` path), via `always()`; and
  5. the gate runs CI's blocking checks, asserted by the commands themselves.

Every "must not find anything" assertion is paired with one proving a real graph
was parsed.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent

# A job's steps are its own plus those of every `uses: ./.github/actions/<name>` it
# references; Invariant 5 reads commands out of both (e.g. `nextflow plugin install`
# lives in .github/actions/setup-nextflow).
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
ci_actions = importlib.import_module("ci_actions")
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"
CI = WORKFLOWS / "ci.yml"
CONTAINERS = WORKFLOWS / "containers.yml"

# The reusable image-build workflow, exactly as release.yml must name it.
CONTAINERS_USES = "./.github/workflows/containers.yml"

# The shared blocking test suite is discovered, not named: "the local reusable
# workflow both gates transitively require". See shared_gate_suite().

# `validate` resolves these; everything downstream reads them. Stored as the
# expression text the workflow must contain.
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
    """What a job runs: its `run:` bodies plus those of the composite actions it
    uses, with comments stripped.

    Resolved, so a check moved into `./.github/actions/setup-nextflow` still counts.
    Comment-blind, because every caller asks "does this job run X", and a
    commented-out command leaves its literal string behind.

    A check that genuinely asks about literal bytes (a `uses:` line, a cache key)
    should call `ci_actions.job_resolved_run_text` at that site instead.
    """
    return ci_actions.job_run_scripts(job)


def checkout_steps(job: dict) -> list[dict]:
    return [
        step
        for step in (job.get("steps") or [])
        if "actions/checkout" in str(step.get("uses", ""))
    ]


def workflow_files() -> list[Path]:
    """Every workflow, discovered by glob, never a filename list, so a check cannot
    silently cover only one of the files that carry a result aggregator.
    """
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOWS.relative_to(ROOT)}"
    return files


# A `run:` step reading `needs.<job>.result` is what a gate aggregator IS. Both
# ci.yml's `all-tests` and release.yml's `test` are nothing but a list of these.
RESULT_READ = re.compile(r"needs\.[A-Za-z0-9_.-]+\.result")


def result_aggregator_jobs(path: Path) -> dict:
    """Jobs that decide pass/fail by comparing `needs.<job>.result` in a `run:`
    step. Discovered, so a renamed or a third aggregator is covered."""
    return {
        job_id: job
        for job_id, job in jobs_of(path).items()
        if isinstance(job, dict) and RESULT_READ.search(run_text(job))
    }


def release_jobs() -> dict:
    return jobs_of(RELEASE)


def artifact_producing_jobs() -> dict:
    """Jobs whose output is a release artifact, discovered rather than named.

    Two kinds, the two things a release ships:
      * the job that calls the reusable image build (`uses:` containers.yml); and
      * any job that tars up `mirage-<version>.tar.gz`.

    Discovery means a new artifact job cannot escape the checks below.
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
    """Jobs a gate transitively `needs:`, upstream only.

    Unlike `_needs_dag_connected`, which walks both directions. Walking downstream
    would count release.yml's publish chain (containers.yml) as required by the gate.
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
    """The one reusable workflow both gates require. Discovered, not named.

    If the two gates stop reaching the same file, every needle test below fails
    here instead of checking one workflow twice.
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
    renders as an empty string, so `ref: ''` would silently be a bare checkout."""
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
    """A bare `actions/checkout` takes `github.ref`: the dispatching branch on a
    workflow_dispatch, the tag on `release: published`. Every checkout here must say
    which commit it means."""
    # An expression is matched as a substring (it sits inside `${{ ... }}`); a literal
    # branch name must match exactly, or `"main" in ref` accepts `maintenance`.
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

    # Non-vacuity: an empty `offenders` could also mean no checkout steps were found.
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
    itself, a push landing mid-run splits the gate across two trees. `validate`
    resolves the ref once to a SHA and the whole gate runs on that SHA.

    A gate job either checks out itself (its `actions/checkout` must name the
    pinned SHA) or calls a reusable workflow, which has no steps and must be handed
    the SHA as an input, because a called workflow checks out the caller's
    `github.ref`. Non-vacuity for the called side is
    `test_the_shared_test_suite_honours_the_ref_it_is_given`.
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
    """`needs:` is the only thing that orders jobs; without `bump-and-tag` in it, the
    tarball and the images are built in parallel with the version bump, from the
    un-bumped tree."""
    producers = artifact_producing_jobs()
    # Non-vacuity: discovery is by shape (a `uses:` of containers.yml, or a `tar` of
    # mirage-*.tar.gz); an empty set would make every assertion below vacuous.
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
    """Depending on `bump-and-tag` only fixes the order. The jobs must also check out
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
    """Artifact jobs survive `bump-and-tag` being skipped.

    `bump-and-tag` carries `if: github.event_name == 'workflow_dispatch'`, so it is
    skipped on `release: published`, and GitHub skips any job whose dependency was
    skipped. Without `always()` the archive and image publish would silently vanish
    from that trigger while the release reports green.
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
    """`verify-config` failing must stop the release shipping, not merely go red.

    It compares `nextflow.config`'s `manifest.version` against the released tag and
    runs only on the `release` trigger. Without this, images could be pushed with a
    tag the tree does not carry.

    Satisfiable two ways per producer: depend on `verify-config`, or carry the check
    inline. A job that `uses:` a reusable workflow has no steps, so only the first.
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
    """The same trap as `bump-and-tag`, from the other trigger.

    `verify-config` carries `if: github.event_name == 'release'`, so it is skipped on
    a workflow_dispatch. `needs: verify-config` without allowing 'skipped' would
    silently drop the image publish from the dispatch path.
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
# Both gates reach the blocking suites through one `uses:` of the shared suite, so
# "the release gate runs what CI blocks on" is checked once, by
# test_the_two_gates_require_the_same_reusable_test_suite. Parity alone says
# nothing about whether either runs anything, so every needle below is asserted
# against the shared suite and against each caller's transitive text.
#
# One entry per blocking check, keyed on the COMMAND rather than the job name.
BLOCKING_CHECKS = [
    ("python unit suite", "pytest -v tests/"),
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
    # Like cleanup_work.sh, a shell check of a session-to-session outcome: neither a
    # pytest nor an nf-test, so it needs its own step.
    ("resume determinism (behavioural)", "tests/resume_check.sh"),
    # Only a run can observe whether a cleanup_level pin reaches the publish gates;
    # `nextflow config` prints the pinned value even when the gates are frozen.
    ("cleanup_level pin reaches the gates (behavioural)", "tests/cleanup_level_pin.sh"),
    ("shipped-defaults stub run", "-profile shipped_defaults_test"),
    ("DEEPCELL token-leak guard", "tests/modules/segment_deepcell_token.nf.test"),
    ("token-guard collection assertion", "--dry-run"),
    ("Nextflow plugin provisioning", "nextflow plugin install"),
    # actionlint runs in the shared suite's `ruff` job, via .github/actions/actionlint
    # (a pinned, checksum-verified release tarball). The needle is the command, not
    # the image name.
    ("actionlint (workflow + composite-action syntax)", "actionlint"),
    # The format suite runs the CONVERT_IMAGE reader stack against synthesised
    # fixtures. The needle is the directory form; `pytest tests/` above would also
    # match a `tests/` run.
    (
        "format suite (convert image's reader stack)",
        "pytest -v tests/integration/formats",
    ),
    # The RTD site is a release artefact. Without `--strict`, `mkdocs build` exits 0
    # on broken links and pages missing from the nav.
    ("mkdocs strict site build", "mkdocs build --strict"),
    # The execution counterpart to the documented-command static guard: every
    # `nextflow run` a doc page shows is launched under --dry_run.
    (
        "documented commands launch (execution leg)",
        "tests/documented_commands_launch.sh",
    ),
]


@pytest.mark.parametrize(
    "label,needle", BLOCKING_CHECKS, ids=[c[0] for c in BLOCKING_CHECKS]
)
def test_the_shared_test_suite_runs_every_blocking_check(label, needle):
    """Each blocking check must still be run, in the suite both gates require.

    A deleted check has no symptom, so each is listed. Resolved through the
    composite actions, so a check that moves into `.github/actions/**` still
    resolves. Comment-blind: `ci_actions.run_scripts` reads `run:` bodies only and
    drops shell comments, so a comment naming a command does not satisfy a needle.
    """
    suite = shared_gate_suite()
    assert needle in ci_actions.run_scripts(suite), (
        f"{needle!r} ({label}) is not RUN anywhere in {suite.name}, the suite both "
        "gates require. It was a BLOCKING check in ci.yml; if it was deliberately "
        "removed, remove it from BLOCKING_CHECKS in the same commit, and say why. "
        "(A comment mentioning it does not count -- this reads `run:` bodies with "
        "their comments stripped.)"
    )
    # And each caller must reach it: this fails if ci.yml or release.yml drops the
    # `uses:`.
    for caller in (CI, RELEASE):
        assert needle in ci_actions.workflow_run_scripts(caller), (
            f"{caller.name} does not reach {needle!r} ({label}) through anything it "
            f"calls. {suite.name} runs it, so {caller.name} has stopped calling the "
            "shared suite."
        )


def test_the_two_gates_require_the_same_reusable_test_suite():
    """Both gates require the same local reusable workflow.

    `ci.yml`'s `all-tests` and `release.yml`'s `test` must both transitively require
    it, so "the release gate runs what CI blocks on" is the same YAML, once.

    Asserted upstream only: release.yml's publish chain also calls a reusable
    workflow (containers.yml), downstream of the gate.
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

    A caller reads one result, the called workflow's conclusion, and a workflow
    whose jobs all skipped concludes `success`. So the legs are enumerated in the
    suite's aggregator, and a job left out of its `needs:` could stop running
    unnoticed.
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
    # The same allowlist test_every_job_is_connected_to_its_workflows_gate uses, so an
    # exemption is a single fact.
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
    """`needs:` is the plumbing; the script is the check, and they can disagree.

    The aggregator carries `if: always()`, so if its script never compares a leg's
    `result`, it exits 0 and every caller reads `success`. Checked over every
    aggregator this file discovers.
    """
    offenders = []
    checked = 0
    for path in workflow_files():
        for gate_id, gate in result_aggregator_jobs(path).items():
            deps = needs_of(gate)
            # An aggregator that needs nothing checks nothing and would pass the floor.
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
    # Non-vacuity: a "must not find" scan over a discovered set. The floor sits
    # below the current count so a legitimate graph change does not trip it.
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

    If the called workflow's own `actions/checkout` steps ignore the `ref:` input,
    every job there checks out the caller's `github.ref`, and the gate tests a
    different commit from the one `bump-and-tag` tags.
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
    """A `uses:` of a local workflow is a checkout, and takes the same rule.

    test_no_job_in_release_yml_does_a_bare_checkout scans `actions/checkout` steps,
    which a reusable-workflow call has none of.
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
    """Non-vacuity for the two "must not find" scans below: ci.yml (`all-tests`) and
    release.yml (`test`) each carry exactly one aggregator."""
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
    """`skipped` is not `success`, in either workflow.

    A gate reading `!= "success" && != "skipped"` goes decorative the moment a
    needed job gains a `paths:`/`if:` filter. Scanned over every workflow.
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

    A dependency can be mentioned without being checked (each branch echoes the
    value it rejected). The operator matters too: `== "failure"` passes `cancelled`,
    `skipped` and any future conclusion.
    """
    pattern = _RESULT_COMPARISON_TEMPLATE.format(dep=re.escape(dep))
    return set(re.findall(pattern, text))


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_every_gate_aggregator_reads_every_job_it_needs(path):
    """A job in the gate's `needs:` that the gate never checks is not gated.

    With `if: always()`, which every aggregator here carries, a failed dependency no
    longer stops the job; the script's result check is the whole gate. A job added
    to `needs:` without a `!= "success"` branch is required in appearance only.

    `alls-green` would provide this property, but it removes the `run:` step that
    reads `needs.<job>.result`, which is how `result_aggregator_jobs` discovers a
    gate for every other assertion in this file.
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

    Forbidden at job and step level, in every workflow including nightly.yml, where
    an expected-red suite tempts it most. A legitimate use goes in the `allowed`
    dict below, by job id, with a written reason.
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
    """`github.event_name == 'workflow_call'` can never be true.

    Inside a reusable workflow the event payload is the caller's, so on a published
    release reaching containers.yml through `publish-images` it is `release`. A
    branch on `workflow_call` falls through to the `push` default and publishes no
    images while the job stays green.

    A called workflow tells its entry points apart by its `inputs`, populated for
    `workflow_dispatch` and `workflow_call` and absent on a plain `push`.
    """
    hits = []
    saw_event_name = 0
    for path in workflow_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            # Quote-aware; `line.split("#", 1)[0]` would truncate `${VERSION#v}`.
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
# A run's conclusion is the worst conclusion of any job in it, whatever the gate's
# `needs:` says, so an advisory job that fails paints the commit red while the gate
# is green. Advisory suites therefore live in a workflow nothing is gated on
# (.github/workflows/nightly.yml). The two checks below hold both halves.

# Jobs outside their workflow's gate DAG, by (file, job id), with the reason. Keep it
# empty if possible: an entry is a job that cannot fail the merge but can still fail
# the run. A failing job inside a called workflow also fails that workflow's
# conclusion, so relocation to nightly.yml is the fix, not an exemption.
GATE_DISCONNECTED_JOBS: dict[tuple[str, str], str] = {}

# The advisory nf-test suites, keyed on the command, never on a job or file name,
# so a rename or a split cannot hide a deletion.
ADVISORY_SUITES = {
    "nf-test real suite": "--tag real",
    "nf-test integration suite": "--tag integration",
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
    """In a workflow that has a gate, every job must be wired to it.

    A job with no `needs:` and nothing needing it is invisible to the gate and
    visible in the run status. Workflows with no gate aggregator (containers.yml,
    nightly.yml) are out of scope by construction; they are counted, not skipped.
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
    # Non-vacuity: a "must not find" scan over gated workflows only, so a discovery
    # predicate that stops matching would turn it green.
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
    """Every exemption names a job that exists and is still disconnected."""
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
    """An advisory suite is still run, and only where no gate aggregator is.

    A deleted advisory suite has no symptom, so its command must still be run by
    some job. And a workflow whose status people read as the merge verdict cannot
    host a suite expected to be red. Discovery is by command.
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
    """`--tag integration` runs only on a manual dispatch.

    nightly.yml has a `schedule:` trigger, so a job without the
    `workflow_dispatch` condition would become a two-hour nightly suite against
    Docker Hub. Keyed on the command. The real suite has no such condition: it is
    meant to run daily on the default branch.
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
# Command needles cannot catch a job moved out of a gated workflow:
# test_every_job_is_connected_to_its_workflows_gate only sees gated workflows, and
# another job may mention the same command (e.g. `nf-test test --tag stub` in
# security-tests' collection assertion). So gate membership is stated here.

# Each gated workflow's aggregator `needs:` set, keyed by file. Every gated workflow
# must appear (asserted below).
GATE_MEMBERSHIP = {
    # The blocking legs live in the shared suite, so ci.yml and release.yml each
    # require the one `uses:` job. `containers` is ci.yml's only blocking leg outside
    # it: the PR image build is gated (the caller job never skips). release.yml
    # builds images downstream of its gate, so it has no counterpart.
    "ci.yml": {"test-suite", "containers"},
    "release.yml": {"test-suite"},
    "_test-suite.yml": {
        "python-tests",
        "nextflow-stub",
        # Its own job: the serial `nextflow run` launches were the gate's long pole.
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

    Other guards reason about what workflows contain, not what the gate requires,
    so a one-line `needs:` deletion passes them all. Every gated workflow must
    appear in GATE_MEMBERSHIP.
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
    """`GATE_MEMBERSHIP` and `ADVISORY_SUITES` are complements over the nf-test suites.

    A job that runs `nf-test test` is either wired into its workflow's gate, or runs
    a command declared in ADVISORY_SUITES. A job in neither has stopped blocking
    without being declared advisory. Discovery is by the `nf-test test` invocation.
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

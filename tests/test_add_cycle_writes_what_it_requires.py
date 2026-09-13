"""An add_cycle run must leave behind everything the NEXT add_cycle demands.

`lib/ParamUtils.groovy`'s validateAddCycle refuses at launch unless every step in
`Layout.ADD_CYCLE_CHECKPOINTS` has a manifest under `--prior_outdir`. So the set
of manifests add_cycle READS is exactly the set it must WRITE, or cyclic-IF
cannot chain past two cycles.

It could not. Reproduced 2026-08-25 under -stub:

    cycle 2 (prior = the fixture)     exit 0,  csv/: preprocessed registered
    cycle 3 (prior = cycle 2's outdir) exit 1,
        mode='add_cycle': required checkpoint 'csv/postprocessed.csv' not found
        under --prior_outdir '...'. Was the prior run completed through
        postprocessing?

`subworkflows/local/registered_checkpoint.nf` had already been split out to fix
exactly this for the registered half; the postprocessed half was still owned by
`postprocess.nf`, which add_cycle never runs.

WHY A STATIC GUARD RATHER THAN A CHAINED nf-test. Proving it end to end takes a
run whose --prior_outdir is a previous run's --outdir: two pipeline invocations
in one case. nf-test's assertion context cannot see lib/ classes at all
(tests/layout.nf.test:5 records this), and no test in this repo uses a `setup`
block to stage a second run, so that harness would be new machinery for one
case. What actually regresses is cheaper to state: a writer disappearing from
add_cycle's reachable include graph. That is what this checks, and it reads the
required set from Layout rather than restating it -- so a THIRD entry in
ADD_CYCLE_CHECKPOINTS is covered the day it is added.

tests/checkpoint_manifest.nf.test's mode=add_cycle case asserts the resulting
files exist and that every path in them resolves; the two together are the
evidence.
"""

import re
from pathlib import Path

from tests.nfmodel import strip_comments

ROOT = Path(__file__).resolve().parents[1]
SUBWORKFLOWS = ROOT / "subworkflows" / "local"

_INCLUDE = re.compile(r"include\s*\{[^}]*\}\s*from\s*'([^']+)'")


def _layout_constant(name):
    """Resolve a `static final String NAME = 'value'` out of lib/Layout.groovy."""
    src = strip_comments((ROOT / "lib" / "Layout.groovy").read_text())
    m = re.search(rf"static\s+final\s+String\s+{name}\s*=\s*'([^']+)'", src)
    assert m, f"Layout.{name} is gone"
    return m.group(1)


def required_steps():
    """The steps validateAddCycle demands under --prior_outdir, read from
    Layout.ADD_CYCLE_CHECKPOINTS rather than restated here."""
    src = strip_comments((ROOT / "lib" / "Layout.groovy").read_text())
    m = re.search(r"ADD_CYCLE_CHECKPOINTS\s*=\s*\n?\s*\[([^\]]*)\]", src)
    assert m, "Layout.ADD_CYCLE_CHECKPOINTS is gone"
    names = [n.strip() for n in m.group(1).split(",") if n.strip()]
    assert names, "Layout.ADD_CYCLE_CHECKPOINTS is empty"
    return {n: _layout_constant(n) for n in names}


def _reachable_from(entry: Path):
    """Every .nf file reachable from `entry` through `include ... from '...'`.

    Transitive on purpose: add_cycle.nf does not include the registered writer
    directly -- it includes REGISTER_PATIENT, which includes it. A one-level
    scan would report a false failure for the half that already works.
    """
    seen, stack = set(), [entry.resolve()]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        for rel in _INCLUDE.findall(strip_comments(f.read_text())):
            target = (f.parent / rel).resolve()
            if not target.suffix:
                target = target.with_suffix(".nf")
            stack.append(target)
    return seen


# `CHECKPOINT_WRITER(Layout.<CONST>, ...)` -- the call that commits a file to writing a
# given manifest. Before subworkflows/local/checkpoint_writer.nf existed this was found
# by looking for a `collectFile(` and a `Layout.checkpointCsvName(Layout.<CONST>)` in the
# SAME file, because every writer carried both. Now exactly one file carries the
# collectFile and it takes the step as a parameter, so what identifies a writer is the
# call site that names the step.
_CKPT_CALL = re.compile(r"CHECKPOINT_WRITER\s*\(\s*Layout\.(\w+)")


def _writers():
    """constant name -> the .nf files that commission a manifest for it."""
    out = {}
    for f in sorted(ROOT.rglob("*.nf")):
        src = strip_comments(f.read_text())
        for const in _CKPT_CALL.findall(src):
            out.setdefault(const, set()).add(f.resolve())
    return out


def test_add_cycle_writes_every_checkpoint_a_follow_on_cycle_requires():
    steps = required_steps()
    reachable = _reachable_from(SUBWORKFLOWS / "add_cycle.nf")
    writers = _writers()

    missing = []
    for const, filename in steps.items():
        producing = writers.get(const, set()) & reachable
        if not producing:
            missing.append(
                f"Layout.{const} (csv/{filename}.csv): no writer reachable from "
                f"add_cycle.nf. validateAddCycle requires it under "
                f"--prior_outdir, so the NEXT cycle refuses at launch."
            )
    assert not missing, "\n".join(missing) + (
        "\n\nMake the writer mode-independent and call it from add_cycle.nf -- "
        "see subworkflows/local/register_patient.nf and postprocessed_checkpoint.nf, "
        "which call CHECKPOINT_WRITER for exactly this reason."
    )


def test_the_linear_path_writes_them_too():
    """The other direction. A writer moved OUT of the linear path would make the
    first cycle unchainable instead of the second, and the check above cannot
    see that."""
    steps = required_steps()
    reachable = _reachable_from(ROOT / "workflows" / "mirage.nf")
    writers = _writers()
    missing = [
        f"Layout.{const}: no writer reachable from workflows/mirage.nf"
        for const in steps
        if not (writers.get(const, set()) & reachable)
    ]
    assert not missing, "\n".join(missing)


def test_the_scan_is_not_vacuous():
    """Three ways this file could pass while checking nothing: no required
    steps, no include graph, or no writers found at all."""
    steps = required_steps()
    assert len(steps) >= 2, f"ADD_CYCLE_CHECKPOINTS shrank to {sorted(steps)}"
    reachable = _reachable_from(SUBWORKFLOWS / "add_cycle.nf")
    assert len(reachable) >= 5, (
        f"only {len(reachable)} file(s) reachable from add_cycle.nf -- the "
        f"include regex is stale, not the pipeline"
    )
    writers = _writers()
    assert writers, (
        "no checkpoint writers found anywhere -- the collectFile scan is stale"
    )
    # And the transitive walk really is transitive: add_cycle.nf does not include
    # checkpoint_writer.nf directly -- it reaches it through REGISTER_PATIENT and
    # through POSTPROCESSED_CHECKPOINT, both of which it does include. If this stops
    # holding, the walk has silently become one-level and would report a false pass.
    # (The anchor used to be registered_checkpoint.nf, which was deleted once the write
    # moved into checkpoint_writer.nf and it had nothing left to own.)
    direct = {
        (SUBWORKFLOWS / "add_cycle.nf")
        .parent.joinpath(rel)
        .resolve()
        .with_suffix(".nf")
        for rel in _INCLUDE.findall(
            strip_comments((SUBWORKFLOWS / "add_cycle.nf").read_text())
        )
    }
    ckpt_writer = (SUBWORKFLOWS / "checkpoint_writer.nf").resolve()
    assert ckpt_writer not in direct and ckpt_writer in reachable, (
        "checkpoint_writer.nf is now a DIRECT include of add_cycle.nf, so the "
        "transitive walk is no longer exercised by this repo's own layout"
    )

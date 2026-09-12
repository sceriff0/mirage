"""Which plan rows a code change touches -- the impact model, in code.

A pipeline change rarely touches every arm. A change confined to STARE's SOLVE
stage (``registration_method=tiled``) leaves the nine VALIS arms, the shared
preprocessing run, the segmentation arms and the ashlar baseline byte-identical;
re-running them repeats days of cluster time to reproduce numbers already on
disk. But re-running *too little* is worse, because it is invisible: a QC cross
that resumes a re-run base arm would keep scoring the OLD registration from the
OLD session, and the table would mix the two silently.

So the impact is a TABLE plus a CLOSURE, and both live here rather than in prose:

* ``COMPONENTS`` maps a changed component to a predicate over a plan row. A row
  matching any requested component is a SEED.
* ``affected_rows`` takes the transitive closure over the plan's dependency
  columns (``DEPENDENCY_COLUMNS``): a row that ``resume_run``s an affected arm is
  affected (the cross re-scores the re-run base), a row whose ``from_arm`` is
  affected is affected (it resumes that arm's checkpoint), and an external row
  whose ``ext_from_arm`` is affected is affected (it scores against that arm's
  published nuclei). Unaffected rows are not.

Both plan builders (build_arm_plan.py, build_run_plan.py) import this and apply
it AFTER full expansion, so a subset plan's rows are the full plan's rows -- same
run_id, same arm, same params -- and results land in the same directories. The
sweep plan has no dependency columns, so its closure is the seed set.

The vocabulary is deliberately small. Add a component here, with its predicate,
rather than hand-filtering a CSV: a hand filter is exactly the "too little"
failure above with no test behind it.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

# A plan row is a flat dict of strings/numbers, as build_arm_plan / build_run_plan emit.
Row = dict
Predicate = Callable[[Row], bool]

# The columns through which one row depends on another. Each names the ARM (and,
# on every arm-plan row, arm == run_id) whose output the row consumes.
DEPENDENCY_COLUMNS = ("resume_run", "from_arm", "ext_from_arm")


def _s(row: Row, key: str) -> str:
    v = row.get(key)
    return "" if v is None else str(v)


def _reg_method_is(method: str) -> Predicate:
    return lambda r: _s(r, "registration_method") == method


def _runs_qc_chain(r: Row) -> bool:
    """Rows that execute the reg_qc scorer: every row at reg_qc >= 1, plus the
    external (ashlar) rows, which are scored by that same scorer
    (bin/warp_seg_qc.py --method tiled) outside Nextflow."""
    if _s(r, "arm_kind") == "external":
        return True
    q = _s(r, "reg_qc")
    try:
        return float(q) >= 1
    except ValueError:
        return False


def _seg_backend_is(method: str) -> Predicate:
    """Rows whose SEGMENT backend is `method` -- whether as the segmentation
    step itself (segmentation/compute arms) or as the QC segmenter
    (SEG_QC_SEGMENT is SEGMENT under an alias, so params.seg_method selects it
    on every registration-step row too). External rows record the segmenter
    whose nuclei they inherit in ext_seg_method."""
    return lambda r: method in (_s(r, "seg_method"), _s(r, "ext_seg_method"))


def _preprocess(r: Row) -> bool:
    """Rows that RUN the preprocessing step: the shared preprocess arm, and any
    row with no --start gate (the compute profile; every sweep run)."""
    return _s(r, "start") in ("", "preprocessing")


def _ashlar(r: Row) -> bool:
    """The external ASHLAR arms (arm_kind=external, run by run_ashlar_arm.sh).

    They are NOT reached by `tiled`/`stare`: they carry no registration_method
    and their ext_from_arm is the VALIS reference whose nuclei they score on. A
    change to STARE's SOLVE stage leaves them byte-identical. `--changed ashlar`
    (or `--only 'ashlar.*'`) selects exactly them and nothing else, since no row
    depends on an external arm -- which is how their resource rows are collected
    once without re-running anything else."""
    return _s(r, "arm_kind") == "external" or _s(r, "ext_tool") == "ashlar"


# component name -> predicate over a plan row. `stare` is an alias of `tiled`
# because the docs use both names for the same backend.
def _solve(r: Row) -> bool:
    """Rows that run STARE's `robust` SOLVE stage (stare.solve). The `legacy` path
    is pinned byte-identical to the pre-package code, so a change to the solver
    reaches only the rows that selected the new one: the solver cross arms."""
    return _reg_method_is("tiled")(r) and _s(r, "reg_tiled_solver") not in (
        "",
        "legacy",
    )


COMPONENTS: dict[str, Predicate] = {
    "tiled": _reg_method_is("tiled"),
    "stare": _reg_method_is("tiled"),
    "solve": _solve,
    "valis": _reg_method_is("valis"),
    "ashlar": _ashlar,
    "qc": _runs_qc_chain,
    "preprocess": _preprocess,
}

# Parametrised components: `seg:<method>`.
_PARAMETRISED: dict[str, Callable[[str], Predicate]] = {
    "seg": _seg_backend_is,
}


def component_predicate(name: str) -> Predicate:
    """Resolve a `--changed` value to its predicate, or raise naming the vocabulary."""
    if name in COMPONENTS:
        return COMPONENTS[name]
    if ":" in name:
        family, _, arg = name.partition(":")
        if family in _PARAMETRISED and arg:
            return _PARAMETRISED[family](arg)
    raise ValueError(
        f"unknown changed component {name!r}; known: "
        f"{sorted(COMPONENTS)} and {[f'{k}:<value>' for k in _PARAMETRISED]}"
    )


def _row_name(r: Row) -> str:
    return _s(r, "arm") or _s(r, "run_id")


def seeds(plan: list[Row], changed: Iterable[str] = (), only: str | None = None):
    """The rows a change touches DIRECTLY: any component predicate matches, or
    the `only` regex matches the row's arm or run_id (re.search)."""
    preds = [component_predicate(c) for c in changed]
    rx = re.compile(only) if only else None
    out = []
    for r in plan:
        hit = any(p(r) for p in preds)
        if rx is not None and (rx.search(_s(r, "arm")) or rx.search(_s(r, "run_id"))):
            hit = True
        if hit:
            out.append(r)
    return out


def closure(plan: list[Row], seed_rows: Iterable[Row]) -> list[Row]:
    """Seeds plus every row that depends, transitively, on an affected row.

    Dependency is by NAME through DEPENDENCY_COLUMNS: a row is affected when any
    of those columns names an affected row's arm (or run_id). Returned in plan
    order, as the same dict objects the plan holds -- nothing is copied or
    rewritten, which is what makes the subset row-identical."""
    affected_names = {_row_name(r) for r in seed_rows}
    affected_ids = {id(r) for r in seed_rows}
    changed = True
    while changed:
        changed = False
        for r in plan:
            if id(r) in affected_ids:
                continue
            deps = {_s(r, c) for c in DEPENDENCY_COLUMNS} - {""}
            if deps & affected_names:
                affected_ids.add(id(r))
                affected_names.add(_row_name(r))
                changed = True
    return [r for r in plan if id(r) in affected_ids]


def affected_rows(
    plan: list[Row], changed: Iterable[str] = (), only: str | None = None
) -> list[Row]:
    """The subset of `plan` a change to `changed` components (and/or an `only`
    regex) requires re-running: seeds, then the transitive closure."""
    return closure(plan, seeds(plan, changed, only))


def dependants(plan: list[Row], name: str) -> list[Row]:
    """Rows that depend DIRECTLY on the arm/run named `name` (its QC crosses,
    the segmentation arms resuming its checkpoint, an external arm scoring
    against its nuclei)."""
    return [r for r in plan if name in {_s(r, c) for c in DEPENDENCY_COLUMNS}]

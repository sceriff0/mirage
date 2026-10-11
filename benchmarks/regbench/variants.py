"""The list of runs: every configuration the cohort is registered with, and nothing else.

One entry of ``RUNS`` is one ``python -m benchmarks.regbench run`` configuration: a method,
the label its results are stored under, its options, the container image it runs in,
whether it needs a GPU, and the run that must have finished first. The launcher reads the
list (``python -m benchmarks.regbench runs``); the scorer is told which result names to
expect (``expected_variants()``) and which are the primary rows (``PRIMARY``).

Totals: 18 runs, 35 result names, 4 runs on a GPU, 7 runs per patient (the rest per pair).

**Paused runs.** A run carries a flag, ``Run.active``. A paused run (``active=False`` on
its entry) stays implemented and tested, and its names still resolve, but it is not
launched, not expected by ``score``, not counted and not printed by ``runs`` (``runs
--all`` prints it, with ``0`` in column 13). Taking the flag off again is the whole
re-activation. Nothing is paused today. ``check()`` refuses a list in which a paused run
writes a primary row, gates an active run's selfcheck, or is what an active run waits
for; it runs at import and before every list this module hands out.

``runs`` prints one line per active run, tab-separated, in the order of ``RUNS``, with
these columns in this order (``RUNS_COLUMNS``). No field is ever empty: ``-`` stands for
"none", because a shell ``read`` with a tab separator merges consecutive tabs. Columns are
only ever added at the end, so a position never changes.

1. ``name``            the label, or the method when the run has no label. Result
                       directories, job names and ``REGBENCH_ALLOW_UNCHECKED`` use it.
2. ``method``          goes back as ``--method``.
3. ``label``           goes back as ``--label``; ``-`` when the run has none.
4. ``opts``            ``key=value`` items joined by commas, each going back as one
                       ``--opt key=value``; ``-`` when there are none. No key or value
                       holds a comma, a tab, a space or an ``=``.
5. ``image``           the image, as the suffix of ``REGBENCH_SIF_<image>``.
6. ``gpu``             ``1`` when the run needs a GPU, else ``0``.
7. ``scope``           ``pair`` (one array task per case) or ``group`` (one per patient).
8. ``after``           the ``name`` of the run whose results this one reads (the anchored
                       ASHLAR rows read STARE's rigid matrix); ``-`` when none.
9. ``gated_by``        the ``name`` of the run whose selfcheck gates this run: its own,
                       except for the rows of ``GATED_BY``.
10. ``selfcheck_mask`` what ``selfcheck --mask`` must be given for this method
                       (``overlap`` for ASHLAR, ``seams`` for PALOM); ``-`` when none.
11. ``primary``        the run's primary result name when it has one, else ``-``.
12. ``variants``       the result names the run writes, joined by commas.
13. ``active``         ``1``, or ``0`` for a paused run (seen only with ``runs --all``).
14. ``gate_variants``  the result names whose selfcheck status gates the run, joined by
                       commas (``gate_variants()``): all of the run's names, except for
                       DeeperHistReg, whose gate is the row of the written image
                       (``<label>_asdrawn``) alone.
15. ``realcheck``      the result name whose realcheck gates the run, or ``-``
                       (``realcheck_gate()``): the launcher sends the run's first unit
                       ahead with ``run --realcheck`` and starts the rest only when
                       that row's ``realcheck.json`` is not flagged.

**Primary rows** (ruling R3.12). VALIS 1.2.0 and DeeperHistReg each run at upstream's
default setting AND at upstream's high-resolution one; the high-resolution row is the
primary of both, by the user's ruling. ``PRIMARY_VALIS120`` and ``PRIMARY_DEEPERHISTREG``
are the one line to change for each.

**The images.** ``image`` is the suffix of the variable that holds the image's path on the
cluster: ``REGBENCH_SIF_<image>``. For an image this repository builds, the suffix is its
directory under ``containers/`` in capitals with ``-`` written ``_`` (``BUILT_HERE``):
``REGBENCH_SIF_STARE_REGBENCH``, ``REGBENCH_SIF_VALIS120``, ``REGBENCH_SIF_DEEPERHISTREG``,
``REGBENCH_SIF_PALOM``; the two pulled from upstream are ``REGBENCH_SIF_VALIS_MIRAGE`` and
``REGBENCH_SIF_ASHLAR`` (``PULLED``). ``SCORER_IMAGE`` names the one ``tissue`` and
``score`` run in.
"""

from __future__ import annotations

from typing import NamedTuple

from . import methods

# benchmarks.regbench.methods.deeperhistreg.HR_CONFIG, written out so that reading the
# list starts no adapter; the tests hold the two equal.
HR_CONFIG = "default_initial_nonrigid_high_resolution"


class Run(NamedTuple):
    method: str  # a name in benchmarks.regbench.METHODS
    label: str | None  # --label; None stores the results under the method's own names
    opts: dict  # --opt key=value, as the strings the command line carries
    image: str  # REGBENCH_SIF_<image>
    gpu: bool
    after: str | None  # the name of a run listed earlier that must have finished
    active: bool = (
        True  # False pauses the run: kept and tested, not launched or expected
    )


RUNS = (
    Run("stare", None, {}, "STARE_REGBENCH", False, None),
    Run("valis_mirage", None, {}, "VALIS_MIRAGE", False, None),
    Run("valis", "valis120", {}, "VALIS120", False, None),
    Run("valis", "valis120_hires", {"hires": "1"}, "VALIS120", False, None),
    Run(
        "deeperhistreg",
        "deeperhistreg_gpu",
        {"device": "cuda:0"},
        "DEEPERHISTREG",
        True,
        None,
    ),
    Run(
        "deeperhistreg",
        "deeperhistreg_cpu",
        {"device": "cpu"},
        "DEEPERHISTREG",
        False,
        None,
    ),
    Run(
        "deeperhistreg",
        "deeperhistreg_hr_gpu",
        {"device": "cuda:0", "config": HR_CONFIG},
        "DEEPERHISTREG",
        True,
        None,
    ),
    Run(
        "deeperhistreg",
        "deeperhistreg_hr_cpu",
        {"device": "cpu", "config": HR_CONFIG},
        "DEEPERHISTREG",
        False,
        None,
    ),
    Run(
        "deeperhistreg",
        "deeperhistreg_fluo_gpu",
        {"device": "cuda:0", "fluo": "1"},
        "DEEPERHISTREG",
        True,
        None,
    ),
    Run(
        "deeperhistreg",
        "deeperhistreg_fluo_cpu",
        {"device": "cpu", "fluo": "1"},
        "DEEPERHISTREG",
        False,
        None,
    ),
    Run(
        "deeperhistreg",
        "deeperhistreg_hr_fluo_gpu",
        {"device": "cuda:0", "config": HR_CONFIG, "fluo": "1"},
        "DEEPERHISTREG",
        True,
        None,
    ),
    Run(
        "deeperhistreg",
        "deeperhistreg_hr_fluo_cpu",
        {"device": "cpu", "config": HR_CONFIG, "fluo": "1"},
        "DEEPERHISTREG",
        False,
        None,
    ),
    Run("ashlar", None, {"affine": "none", "noise": "ref"}, "ASHLAR", False, None),
    Run(
        "ashlar",
        "ashlar_clean",
        {"affine": "none", "noise": "none"},
        "ASHLAR",
        False,
        None,
    ),
    Run(
        "ashlar",
        "affine_ashlar",
        {"affine": "stare", "noise": "ref"},
        "ASHLAR",
        False,
        "stare",
    ),
    Run(
        "ashlar",
        "affine_ashlar_clean",
        {"affine": "stare", "noise": "none"},
        "ASHLAR",
        False,
        "stare",
    ),
    Run("palom", None, {}, "PALOM", False, None),
    Run("palom", "palom_b1024", {"block": "1024"}, "PALOM", False, None),
)

# Ruling R3.12 (2026-10-11): the high-resolution row is primary for both methods that
# have one. Each is one line to switch back to upstream's default setting:
# "valis120" (register at 512 / 2048 px) and "deeperhistreg_gpu_asdrawn" (the 4096 px
# set). Both settings run either way. DeeperHistReg's is the row of the image written to
# disk in both cases.
PRIMARY_VALIS120 = "valis120_hires"
PRIMARY_DEEPERHISTREG = "deeperhistreg_hr_gpu_asdrawn"
PRIMARY = (
    "stare",
    "valis_mirage",
    PRIMARY_VALIS120,
    PRIMARY_DEEPERHISTREG,
    "ashlar",
    "palom",
)

# Result-name prefix -> the method a row is shown under. A family's name is a label for
# the method, NOT the name of its primary row: `valis120` and `valis120_hires` share the
# family `valis120`, whose primary is `valis120_hires` (`primary_of`). `palom_b1024` is in
# palom's family (non-primary). `ashlar_clean` stays in ASHLAR's family; the two anchored
# rows share a family of their own, which has no primary row and is never offered as
# ASHLAR's best.
FAMILY = {
    "stare": "stare",
    "valis_mirage": "valis_mirage",
    "valis120": "valis120",
    "deeperhistreg": "deeperhistreg",
    "ashlar": "ashlar",
    "affine_ashlar": "affine_ashlar",
    "palom": "palom",
}

# Which run's selfcheck gates a run, when it is not its own. `palom_b1024` is not here:
# its block grid differs from `palom`'s, so it stands behind its own selfcheck.
GATED_BY = {"ashlar_clean": "ashlar", "affine_ashlar_clean": "affine_ashlar"}

# Method -> the suffixes of the result names whose selfcheck status gates its runs, when
# that is not all of them. DeeperHistReg (ruling R3.12 b): only the row of the image its
# writer puts on disk. Its estimated-field row is checked and reported all the same; when
# it fails, that row is dropped from the results and the method still runs.
GATE_SUFFIXES = {"deeperhistreg": ("_asdrawn",)}

# The methods whose realcheck is a GATE (Task 5 and Task 7 of the port's plan): the two
# VALIS adapters. Their point path is derived by the benchmark, not handed over by the
# package, so it is also checked on a real case before the cohort is spent on it. The
# gate reads the run's PRIMARY row; a run of these methods without one has none.
REALCHECK_METHODS = ("valis_mirage", "valis")

# Image -> the build context under containers/ (images.json) for the ones this repository
# builds, and the upstream reference for the ones it pulls.
BUILT_HERE = {
    # the benchmark's OWN STARE image (the pipeline's containers/stare plus psutil,
    # pytest and OpenCV), not the pipeline's bolt3x/mirage-stare
    "STARE_REGBENCH": "stare-regbench",
    "VALIS120": "valis120",
    "DEEPERHISTREG": "deeperhistreg",
    "PALOM": "palom",
}
PULLED = {
    "VALIS_MIRAGE": "cdgatenbee/valis-wsi 1.0.0 (the digest of modules/local/register.nf)",
    "ASHLAR": "labsyspharm/ashlar:1.20.0",
}

# Where `tissue` and `score` run: the one image with the pinned OpenCV the tissue-level
# block is computed with, so a whole result rests on one OpenCV version. It also has
# everything `score` imports (numpy, scipy, tifffile, zarr, stare.slide_io).
SCORER_IMAGE = "STARE_REGBENCH"

RUNS_COLUMNS = (
    "name",
    "method",
    "label",
    "opts",
    "image",
    "gpu",
    "scope",
    "after",
    "gated_by",
    "selfcheck_mask",
    "primary",
    "variants",
    "active",
    "gate_variants",
    "realcheck",
)
NONE = "-"


def run_name(run):
    """What a run's results are stored under: its label, or the method's name."""
    return run.label or run.method


def variants_of(run):
    """The result names one run writes, by the method's own naming rule."""
    return methods.variant_names(run.method, run.label)


def active_runs():
    """The runs that are launched and expected: ``RUNS`` without the paused ones."""
    return tuple(r for r in RUNS if r.active)


def paused_names():
    """The names of the paused runs, in the order of ``RUNS`` (none today)."""
    return [run_name(r) for r in RUNS if not r.active]


def run_of(name):
    """The run stored under ``name``, paused or not; KeyError for a name of no run."""
    for run in RUNS:
        if run_name(run) == name:
            return run
    raise KeyError(f"no run named {name!r}")


def check():
    """Refuse a list whose paused runs are still needed by what runs.

    A paused run writes no results. So it cannot write a primary row (the summary would
    open on a row that does not exist), it cannot be the run whose selfcheck gates an
    active run (that run would start unchecked or never start), and it cannot be what an
    active run waits for (the anchored ASHLAR rows read STARE's matrix).
    """
    active = {run_name(r) for r in active_runs()}
    written = {v: run_name(r) for r in RUNS for v in variants_of(r)}
    for variant in PRIMARY:
        if variant not in written:
            raise ValueError(f"primary row {variant!r} is written by no run")
        if written[variant] not in active:
            raise ValueError(
                f"the run {written[variant]!r} is paused and writes the primary row "
                f"{variant!r}: name another primary row for its family first"
            )
    for run in active_runs():
        name = run_name(run)
        if gated_by(name) not in active:
            raise ValueError(
                f"the run {gated_by(name)!r} is paused and its selfcheck gates the "
                f"active run {name!r}: pause that one too, or change GATED_BY"
            )
        if run.after is not None and run.after not in active:
            raise ValueError(
                f"the active run {name!r} waits for {run.after!r}, which is paused"
            )


def expected_variants():
    """Every result name the cohort must have, in the order of ``RUNS``: the names of
    the active runs (35 with nothing paused)."""
    check()
    return [v for run in active_runs() for v in variants_of(run)]


def scope(method):
    """``pair`` (run case by case) or ``group`` (one process per patient)."""
    return methods.scope_of(method)


def counts():
    """``(runs, runs on a GPU, runs per patient)``, of the active runs."""
    runs = active_runs()
    return (
        len(runs),
        sum(1 for r in runs if r.gpu),
        sum(1 for r in runs if scope(r.method) == "group"),
    )


def family(variant):
    """The method a result name is shown under: the longest key of ``FAMILY`` that is
    the name itself or a prefix of it ending at an underscore. An off-seam row
    (``<variant>:non_<owner>``) has its variant's family. None for a name of no listed
    method (``initial``, ``truth``)."""
    name = str(variant).split(":", 1)[0]
    best = None
    for prefix in FAMILY:
        if name == prefix or name.startswith(prefix + "_"):
            if best is None or len(prefix) > len(best):
                best = prefix
    return FAMILY[best] if best is not None else None


def members(family_name, paused=False):
    """The active run names of one family, in the order of ``RUNS``; with ``paused``,
    the paused ones too."""
    runs = RUNS if paused else active_runs()
    return [run_name(r) for r in runs if family(run_name(r)) == family_name]


def primary_of(family_name):
    """The primary result name of a family, or None (the anchored ASHLAR rows). It is
    looked up in ``PRIMARY``, never taken to be the family's own name: the family
    ``valis120`` has the primary ``valis120_hires``."""
    for variant in PRIMARY:
        if family(variant) == family_name:
            return variant
    return None


def gated_by(name):
    """The name of the run whose selfcheck gates the run ``name``."""
    return GATED_BY.get(name, name)


def gate_variants(name):
    """The result names whose selfcheck status gates the run ``name``: all the names it
    writes, unless its method is in ``GATE_SUFFIXES``. For a run gated by another run's
    selfcheck (``gated_by``), ask for THAT run's names."""
    run = run_of(name)
    names = tuple(variants_of(run))
    suffixes = GATE_SUFFIXES.get(run.method)
    if not suffixes:
        return names
    gate = tuple(v for v in names if v.endswith(tuple(suffixes)))
    if not gate:
        raise ValueError(f"{name}: none of {names} ends in one of {suffixes}")
    return gate


def realcheck_gate(name):
    """The result name whose ``realcheck.json`` gates the run ``name``, or None: the
    run's primary row, for a method in ``REALCHECK_METHODS``. Not the exit code of ``run
    --realcheck``, which is also 3 when a non-primary row of the run is flagged."""
    run = run_of(name)
    if run.method not in REALCHECK_METHODS:
        return None
    primary = [v for v in variants_of(run) if v in PRIMARY]
    return primary[0] if primary else None


def _statuses(name, record, names):
    """The status of each name in a selfcheck record. ``selfcheck.json`` has
    ``variants: {name: status}``; the dict per variant inside ``pairs`` is read too."""
    from . import selfcheck

    found = record.get("variants") or {}
    out = []
    for v in names:
        entry = found.get(v, selfcheck.NOT_CHECKED)
        out.append(
            entry.get("status", selfcheck.NOT_CHECKED)
            if isinstance(entry, dict)
            else entry
        )
    return out, selfcheck


def gate_status(name, record):
    """The status that decides whether the run ``name`` may start, from the record a
    selfcheck wrote (``selfcheck.json``): the worst over ``gate_variants(name)``, a name
    the record lacks counting as ``not_checked``. It is the record's own ``status``
    except where a method's gate reads only some of its rows."""
    statuses, selfcheck = _statuses(name, record, gate_variants(name))
    return selfcheck.worst(statuses)


def dropped_variants(name, record):
    """The result names of the run ``name`` that are left out of the results because
    their own selfcheck failed while the gate held: the non-gating rows whose status
    blocks. Empty when the gate itself blocks (then the run does not start at all)."""
    statuses, selfcheck = _statuses(name, record, gate_variants(name))
    if selfcheck.worst(statuses) in selfcheck.BLOCKING:
        return []
    others = [v for v in variants_of(run_of(name)) if v not in gate_variants(name)]
    theirs, _ = _statuses(name, record, others)
    return [v for v, s in zip(others, theirs) if s in selfcheck.BLOCKING]


def droppable(variant):
    """True for a result name of the run list that gates no run: the only rows a failed
    selfcheck may leave out of the results (a gating row's failure blocks the method)."""
    for run in RUNS:
        if variant in variants_of(run):
            return variant not in gate_variants(run_name(run))
    raise KeyError(f"{variant!r} is no result name of the run list")


def selfcheck_mask(method):
    """What ``selfcheck --mask`` must be given for a method, or None. Read from the
    adapter's ``MASKS``, so the list and the adapter cannot name different masks."""
    masks = getattr(methods.load(method), "MASKS", None) or ()
    return masks[0] if masks else None


def opt_args(run):
    """A run's options as the ``key=value`` strings ``--opt`` takes."""
    return [f"{k}={v}" for k, v in run.opts.items()]


def runs_table(paused=False):
    """One row of strings per active run, in the columns of ``RUNS_COLUMNS``; with
    ``paused``, the paused runs too, in their place in ``RUNS``."""
    check()
    table = []
    for run in RUNS if paused else active_runs():
        names = variants_of(run)
        primary = [v for v in names if v in PRIMARY]
        table.append(
            (
                run_name(run),
                run.method,
                run.label or NONE,
                ",".join(opt_args(run)) or NONE,
                run.image,
                "1" if run.gpu else "0",
                scope(run.method),
                run.after or NONE,
                gated_by(run_name(run)),
                selfcheck_mask(run.method) or NONE,
                primary[0] if primary else NONE,
                ",".join(names),
                "1" if run.active else "0",
                ",".join(gate_variants(run_name(run))),
                realcheck_gate(run_name(run)) or NONE,
            )
        )
    return table


check()

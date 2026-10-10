"""The list of runs: every configuration the cohort is registered with, and nothing else.

One entry of ``RUNS`` is one ``python -m benchmarks.regbench run`` configuration: a method,
the label its results are stored under, its options, the container image it runs in,
whether it needs a GPU, and the run that must have finished first. The launcher reads the
list (``python -m benchmarks.regbench runs``); the scorer is told which result names to
expect (``expected_variants()``) and which are the primary rows (``PRIMARY``).

Totals: 18 runs, 35 result names, 4 runs on a GPU, 7 runs per patient (the rest per pair).

``runs`` prints one line per run, tab-separated, in the order of ``RUNS``, with these
columns in this order (``RUNS_COLUMNS``). No field is ever empty: ``-`` stands for "none",
because a shell ``read`` with a tab separator merges consecutive tabs.

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

The DeeperHistReg primary is an open decision of the plan (the shipped 4096 px set or the
8192 px one); ``PRIMARY_DEEPERHISTREG`` is the one line to change.
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


RUNS = (
    Run("stare", None, {}, "STARE", False, None),
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

# OPEN DECISION (the plan, Task 9): the 4096 px shipped set (below) or the 8192 px one,
# "deeperhistreg_hr_gpu_asdrawn". Either way it is the row of the image written to disk.
PRIMARY_DEEPERHISTREG = "deeperhistreg_gpu_asdrawn"
PRIMARY = (
    "stare",
    "valis_mirage",
    "valis120",
    PRIMARY_DEEPERHISTREG,
    "ashlar",
    "palom",
)

# Result-name prefix -> the method a row is shown under. `valis120_hires` is in
# valis120's family and `palom_b1024` in palom's (both non-primary). `ashlar_clean` stays
# in ASHLAR's family; the two anchored rows share a family of their own, which has no
# primary row and is never offered as ASHLAR's best.
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

# Image -> the build context under containers/ (images.json) for the ones this repository
# builds, and the upstream reference for the ones it pulls.
BUILT_HERE = {
    "STARE": "stare",
    "VALIS120": "valis120",
    "DEEPERHISTREG": "deeperhistreg",
    "PALOM": "palom",
}
PULLED = {
    "VALIS_MIRAGE": "cdgatenbee/valis-wsi 1.0.0 (the digest of modules/local/register.nf)",
    "ASHLAR": "labsyspharm/ashlar:1.20.0",
}

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
)
NONE = "-"


def run_name(run):
    """What a run's results are stored under: its label, or the method's name."""
    return run.label or run.method


def variants_of(run):
    """The result names one run writes, by the method's own naming rule."""
    return methods.variant_names(run.method, run.label)


def expected_variants():
    """Every result name the cohort must have, in the order of ``RUNS`` (35 names)."""
    return [v for run in RUNS for v in variants_of(run)]


def scope(method):
    """``pair`` (run case by case) or ``group`` (one process per patient)."""
    return methods.scope_of(method)


def counts():
    """``(runs, runs on a GPU, runs per patient)``."""
    return (
        len(RUNS),
        sum(1 for r in RUNS if r.gpu),
        sum(1 for r in RUNS if scope(r.method) == "group"),
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


def members(family_name):
    """The run names of one family, in the order of ``RUNS``."""
    return [run_name(r) for r in RUNS if family(run_name(r)) == family_name]


def primary_of(family_name):
    """The primary result name of a family, or None (the anchored ASHLAR rows)."""
    for variant in PRIMARY:
        if family(variant) == family_name:
            return variant
    return None


def gated_by(name):
    """The name of the run whose selfcheck gates the run ``name``."""
    return GATED_BY.get(name, name)


def selfcheck_mask(method):
    """What ``selfcheck --mask`` must be given for a method, or None. Read from the
    adapter's ``MASKS``, so the list and the adapter cannot name different masks."""
    masks = getattr(methods.load(method), "MASKS", None) or ()
    return masks[0] if masks else None


def opt_args(run):
    """A run's options as the ``key=value`` strings ``--opt`` takes."""
    return [f"{k}={v}" for k, v in run.opts.items()]


def runs_table():
    """One row of strings per run, in the columns of ``RUNS_COLUMNS``."""
    table = []
    for run in RUNS:
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
            )
        )
    return table

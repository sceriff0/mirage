"""The list of runs: every configuration the cohort is registered with, and nothing else.

One entry of ``RUNS`` is one ``python -m benchmarks.regbench run`` configuration: a method,
the label its results are stored under, its options, the container image it runs in,
whether it needs a GPU, and the run that must have finished first. The launcher reads the
list (``python -m benchmarks.regbench runs``); the scorer is told which result names to
expect (``expected_variants()``) and which are the primary rows (``PRIMARY``).

Totals are ``counts()`` and ``expected_variants()``; nothing here is a second copy to keep in
step. The STARE and ASHLAR runs are GENERATED from tables (``regimes.REGIMES`` x
``STARE_STRIDES``; ``methods.ashlar.ASHLAR_CELLS``), so a total moves with a table. A run
takes a dataset argument (``runs_for(dataset)``). A run carries ``Run.regimes``, the
regimes (``regimes.REGIMES``) whose datasets take it: every regime by default, the
generated STARE cells the regimes whose levels make them, and STARE's two tier runs the
cohort's alone (``regimes.COHORT_REGIME``); ``runs_for(dataset)`` keeps a run only on a
dataset of one of its regimes (``regimes.DATASET_REGIME``). The STARE runs of a
brightfield regime are in ``RUNS`` beside the cohort's and never in the cohort's list (as
at commit 5a46f778).

**A public dataset** (``regbench.PUBLIC``) has a list of its own, never pooled with the
cohort's: the runs of its regime, less every run left out of it with a reason
(``excluded_for(dataset)``): a method without an input rule for the dataset's modality
(``methods.input_rule``, whose words the adapter refuses such a case in), or a run the
dataset does not take although its method has one (``NOT_ON``). Ported from commit
5a46f778 (``_not_on``, ``excluded_for``, ``NOT_ON``) without its tuning side and its
preprocessing row. ``runs --dataset <name>`` prints the list in the same columns;
``expected_variants(dataset)`` and ``primary_for(dataset)`` are what its scorer is told.

**Paused runs.** A run carries a flag, ``Run.active``. A paused run (``active=False`` on
its entry) stays implemented and tested, and its names still resolve, but it is not
launched, not expected by ``score``, not counted and not printed by ``runs`` (``runs
--all`` prints it, with ``0`` in column 13). Taking the flag off again is the whole
re-activation. No run is paused today. ``check()`` refuses a list in which a paused run
writes a primary row, makes the selfcheck an active run is checked by, or is what an
active run waits for; it runs at import and before every list this module hands out.

``runs`` prints one line per active run, tab-separated, in the order of ``RUNS``, with
these columns in this order (``RUNS_COLUMNS``). No field is ever empty: ``-`` stands for
"none", because a shell ``read`` with a tab separator merges consecutive tabs. Columns are
only ever added at the end, so a position never changes.

1. ``name``            the label, or the method when the run has no label. Result
                       directories and job names use it.
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
9. ``checked_by``      the ``name`` of the run whose selfcheck checks this run: its own,
                       except for the rows of ``CHECKED_BY``. The selfcheck is
                       REPORTED, never a gate: no job of a run waits for it.
10. ``selfcheck_mask`` what ``selfcheck --mask`` must be given for this method
                       (``overlap`` for ASHLAR, ``seams`` for PALOM); ``-`` when none.
11. ``primary``        the run's primary result name when it has one, else ``-``.
12. ``variants``       the result names the run writes, joined by commas.
13. ``active``         ``1``, or ``0`` for a paused run (seen only with ``runs --all``).
14. ``checked_rows``   the result names the run's reported selfcheck status is read
                       from, joined by commas (``checked_rows()``): all of the run's
                       names. Every row stays in every table whatever its selfcheck
                       says.
15. ``realcheck``      the result name whose realcheck is reported for the run, or ``-``
                       (``realcheck_row()``): the run's first unit runs with ``run
                       --realcheck``, beside the rest of the run, which never waits
                       for it.
16. ``stages``         for a run whose registration is several scheduler jobs per round
                       (``stages_of()``): their names in order, joined by commas
                       (``coarse,tile,solve,stitch``); ``-`` for a run that is one job per unit;
                       ``local`` for a run whose stages run inside its one job of 4 CPUs
                       (``LOCAL_4CPU``: ``stare_local4``, a cost row of ``stare``).

The STARE runs of a set are its regime's levels x the three strides
(``regimes.DATASET_REGIME``; a baseline cell, STARE unchanged, is one run in every regime),
and for the cohort two tier runs, ``stare_medium`` and ``stare_low``
(``STARE_TIERS``: stride 64, no preprocessing); ``runs --dataset multiplex`` prints the list.
On the landmark set (``LANDMARK_REGIMES``: HyReCo's ``bf_same``) STARE runs the FULL
cross, tier x stride x level (45 runs: ``bf_same``'s five levels;
``stare_<tier>[_s<stride>]_<level>``, ``stare_tier_cell``), and every other method its
defaults only (50 runs; rulings 2026-10-19). The manuscript's HyReCo STARE row (Fig. 3f)
is ``stare_haem_none``: haematoxylin, high tier, stride 64, no intensity preprocessing.
The pipeline's VALIS runs at its three tiers x micro-registration depth 0, 1, 2
(``_valis_mirage_runs``): ``valis_mirage`` is (``high``, 2), the shipped call, and the
other eight are ``valis_mirage_<tier>_m<depth>``, checked by ``valis_mirage``'s selfcheck,
with no realcheck (only the primary row has one).

**Primary rows** (ruling R3.12). VALIS 1.2.0 and DeeperHistReg each run at upstream's
default setting AND at upstream's high-resolution one; the high-resolution row is the
primary of both, by the user's ruling. ``PRIMARY_VALIS120`` and ``PRIMARY_DEEPERHISTREG``
are the one line to change for each.

**The images.** ``image`` is the suffix of the variable that holds the image's path on the
cluster: ``REGBENCH_SIF_<image>``. For an image this repository builds, the suffix is its
directory under ``containers/`` in capitals with ``-`` written ``_`` (``BUILT_HERE``):
``REGBENCH_SIF_STARE_REGBENCH``, ``REGBENCH_SIF_VALIS120``, ``REGBENCH_SIF_DEEPERHISTREG``,
``REGBENCH_SIF_PALOM``; the two that are their authors' published images plus the slide reader's
codecs are ``REGBENCH_SIF_VALIS_MIRAGE`` and ``REGBENCH_SIF_ASHLAR`` (``PULLED``). ``SCORER_IMAGE`` names the one ``tissue`` and
``score`` run in.
"""

from __future__ import annotations

from typing import NamedTuple

from . import PUBLIC, base_name, methods, plugins, public_of, regimes
from .methods import ashlar as _ashlar
from .methods import valis_mirage as _valis_mirage

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
    # the regimes (regimes.REGIMES) whose datasets take the run: every regime by default
    regimes: tuple[str, ...] = tuple(regimes.REGIMES)


STARE_STRIDES = (64, 128, 256)


def stare_cell(stride, key, level):
    """``(run name, opts)`` of one STARE cell: a stride and a preprocessing level.

    The baseline level carries the stride alone (no proxy, no ``sprep``): the adapter's
    own default is the channel with no preprocessing, so the names ``stare``,
    ``stare_s128_none`` and ``stare_s256_none`` need no options. Any other level names
    its proxy and ``sprep``. Stride 64 is STARE's default and is left out of the name (and of the
    options of the plain row)."""
    lk = regimes.level_key(level)
    name = f"stare_{lk}" if stride == 64 else f"stare_s{stride}_{lk}"
    if key == "baseline":
        return ("stare", {}) if stride == 64 else (name, {"stride": str(stride)})
    opts = {"proxy": level.proxy, "sprep": level.sprep}
    if stride != 64:
        opts = {"stride": str(stride), **opts}
    return name, opts


def _stare_runs(regime):
    """``{run name: opts}`` of one regime's STARE runs: every stride x every level of the
    regime (``regimes.stare_levels``), deduplicated by name."""
    found = {}
    for key, level in regimes.stare_levels(regime):
        for s in STARE_STRIDES:
            name, opts = stare_cell(s, key, level)
            found.setdefault(name, opts)
    return found


def _stare_regime_runs():
    """The STARE runs of every regime, the cohort's first and in its own order (as at
    commit 5a46f778). A run lists the regimes whose levels produce it, so a baseline cell
    (STARE unchanged at a stride) is one run of every regime. Two regimes producing one
    name with different options is a bug and is refused."""
    found = {}
    order = (regimes.COHORT_REGIME,) + tuple(
        r for r in regimes.REGIMES if r != regimes.COHORT_REGIME
    )
    for reg in order:
        for name, opts in _stare_runs(reg).items():
            seen_opts, seen_regs = found.setdefault(name, (opts, []))
            if seen_opts != opts:
                raise ValueError(f"{name}: {seen_opts} in one regime, {opts} in {reg}")
            seen_regs.append(reg)
    return tuple(
        Run(
            "stare",
            None if name == "stare" else name,
            opts,
            "STARE_REGBENCH",
            False,
            None,
            True,
            tuple(regs),
        )
        for name, (opts, regs) in found.items()
    )


STARE_TIERS = ("medium", "low")  # beside high (the `stare` row): ruling 2026-10-16


def _stare_tier_runs():
    """STARE's medium and low tiers (``methods.stare.TIERS``, the rows of
    ``lib/RegPresets.groovy``) at stride 64 with no preprocessing: the high tier is the
    ``stare`` row itself. On the cohort and on the landmark sets' regimes
    (``LANDMARK_REGIMES``), where they are the stride-64 baseline cells of the full
    tier cross (``_stare_landmark_tier_runs``; ruling 2026-10-19)."""
    return tuple(
        Run(
            "stare",
            f"stare_{t}",
            {"tier": t},
            "STARE_REGBENCH",
            False,
            None,
            True,
            # the cohort's (ruling 2026-10-16) and the landmark sets' (ruling 2026-10-19)
            (regimes.COHORT_REGIME, *LANDMARK_REGIMES),
        )
        for t in STARE_TIERS
    )


# The regimes of the public landmark sets (``regimes.DATASET_REGIME``: HyReCo's
# ``bf_same``, the only one since the user's ruling of 2026-10-19). On them STARE runs its
# FULL cross (ruling 2026-10-19): tier {high, medium, low} x stride ``STARE_STRIDES`` x
# every level of the regime, 45 runs; every other method its defaults only
# (``_LANDMARK_DEFAULTS``).
LANDMARK_REGIMES = tuple(
    dict.fromkeys(
        regimes.DATASET_REGIME[name]
        for name, spec in PUBLIC.items()
        if spec["kind"] == "landmarks"
    )
)


def stare_tier_cell(tier, stride, key, level):
    """``(run name, opts)`` of one cell of the tier cross: ``stare_cell``'s with the tier
    after ``stare_`` (``stare_medium_s128_grey_clahe``); the high tier is ``stare_cell``
    itself, and the stride-64 baseline of a tier is ``stare_<tier>`` (``_stare_tier_runs``)."""
    name, opts = stare_cell(stride, key, level)
    if tier == "high":
        return name, opts
    rest = name[len("stare") :]
    return f"stare_{tier}{rest}", {"tier": tier, **opts}


def _stare_landmark_tier_runs():
    """The medium and low cells of the tier cross on the landmark sets' regimes, less the
    two that ``_stare_tier_runs`` already lists (``stare_medium``, ``stare_low``); the
    high cells are ``_stare_regime_runs``'. A run lists the regimes whose levels make it."""
    found = {}
    for tier in STARE_TIERS:
        for reg in LANDMARK_REGIMES:
            for key, level in regimes.stare_levels(reg):
                for s in STARE_STRIDES:
                    name, opts = stare_tier_cell(tier, s, key, level)
                    if name == f"stare_{tier}":
                        continue
                    seen_opts, seen_regs = found.setdefault(name, (opts, []))
                    if seen_opts != opts:
                        raise ValueError(f"{name}: {seen_opts} and {opts}")
                    if reg not in seen_regs:
                        seen_regs.append(reg)
    return tuple(
        Run("stare", name, opts, "STARE_REGBENCH", False, None, True, tuple(regs))
        for name, (opts, regs) in found.items()
    )


# STARE at 4 CPUs in ONE job (Task 31, ruling R2/P4): the `stare` configuration with
# `--stages local` and its tiles 4 at a time, every case of the cohort on the ordinary
# ladder, under the same resources.Monitor as every one-process method. A COST row: its
# accuracy must equal `stare`'s (the paper tables compare them case by case), it is
# never a best-row candidate (COST_ONLY), and its results have their own name, so no
# other row's reuse key moves. Column 16 of `runs` is `local` for it: the launcher runs
# it as one job per case whatever REGBENCH_STARE_EXECUTION is, and only at
# REGBENCH_CPUS=4.
STARE_LOCAL4 = "stare_local4"
LOCAL_4CPU = (STARE_LOCAL4,)
LOCAL_STAGES = "local"


def _stare_local4_runs():
    return (
        Run(
            "stare",
            STARE_LOCAL4,
            {"workers": "4"},
            "STARE_REGBENCH",
            False,
            None,
            True,
            (regimes.COHORT_REGIME,),
        ),
    )


def _valis_mirage_runs():
    """The pipeline's VALIS at its three tiers x micro-registration depth 0, 1, 2
    (``methods.valis_mirage.TIERS`` x ``MICRO``): ``valis_mirage`` (``high``, 2, the shipped
    call, no options) first, then the other eight as ``valis_mirage_<tier>_m<depth>``.
    Depth 1 is micro-rigid only; a non-rigid ``micro`` stage exists only at depth 2
    (``methods.valis_mirage.STAGES_AT_DEPTH``)."""
    out = [Run("valis_mirage", None, {}, "VALIS_MIRAGE", False, None)]
    for t in _valis_mirage.TIERS:
        for m in _valis_mirage.MICRO:
            if (t, m) == (
                _valis_mirage.ARGS["memory_mode"],
                _valis_mirage.ARGS["micro_reg"],
            ):
                continue  # the shipped call: `valis_mirage` itself
            out.append(
                Run(
                    "valis_mirage",
                    f"valis_mirage_{t}_m{m}",
                    {"memory_mode": t, "micro_reg": str(m)},
                    "VALIS_MIRAGE",
                    False,
                    None,
                )
            )
    return tuple(out)


ASHLAR_AFFINES = _ashlar.AFFINES
ASHLAR_NOISES = _ashlar.NOISE_FRACS
ASHLAR_SHIFTS = _ashlar.MAX_SHIFTS_UM


def _ashlar_runs():
    """The 24 ASHLAR runs, built FROM the adapter's cell table (``ASHLAR_CELLS``), so a
    label and its options cannot come apart. ``ashlar`` is stored under the method's own
    name (label None)."""
    return tuple(
        Run(
            "ashlar",
            None if label == "ashlar" else label,
            {"affine": a, "noise_frac": f, "max_shift_um": s},
            "ASHLAR",
            False,
            "stare" if a == "stare" else None,
        )
        for label, (a, f, s) in _ashlar.ASHLAR_CELLS.items()
    )


RUNS = (
    *_stare_regime_runs(),
    *_stare_tier_runs(),
    *_stare_landmark_tier_runs(),
    *_stare_local4_runs(),
    *_valis_mirage_runs(),
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
    *_ashlar_runs(),
    Run("palom", None, {}, "PALOM", False, None),
)

# Runs of a private method plug-in (regbench.plugins): none unless one is enabled.
RUNS += tuple(Run(**r) for r in plugins.get("RUNS", ()) or ())

# Ruling R3.12 (2026-10-11): the high-resolution row is primary for both methods that
# have one. Each is one line to switch back to upstream's default setting:
# "valis120" (register at 512 / 2048 px) and "deeperhistreg_gpu" (the 4096 px set). Both
# settings run either way. DeeperHistReg's one row per run is the image its writer puts
# on disk (user instruction 2026-10-17).
PRIMARY_VALIS120 = "valis120_hires"
PRIMARY_DEEPERHISTREG = "deeperhistreg_hr_gpu"
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
# family `valis120`, whose primary is `valis120_hires` (`primary_of`).
# `ashlar_clean` stays in ASHLAR's family; the two anchored rows share a family of their
# own, which has no primary row and is never offered as ASHLAR's best.
FAMILY = {
    "stare": "stare",
    "ashlar_astare": "affine_ashlar",
    "valis_mirage": "valis_mirage",
    "valis120": "valis120",
    "deeperhistreg": "deeperhistreg",
    "ashlar": "ashlar",
    "affine_ashlar": "affine_ashlar",
    "palom": "palom",
}

FAMILY.update(plugins.get("FAMILY", {}) or {})

# Which run's selfcheck checks a run, when it is not its own. Reported, never a gate.
CHECKED_BY = {
    # every ASHLAR cell is checked by the selfcheck of the cell that matches it in
    # `affine`; every generated STARE cell by `stare`'s (one adapter, one pointing path)
    **{
        n: ("affine_ashlar" if c[0] == "stare" else "ashlar")
        for n, c in _ashlar.ASHLAR_CELLS.items()
        if n not in ("ashlar", "affine_ashlar")
    },
    **{r.label: "stare" for r in _stare_regime_runs() if r.label is not None},
    # the tier runs too: the same adapter and pointing path at other tile sizes
    **{r.label: "stare" for r in _stare_tier_runs()},
    # and the tier cross of the landmark sets (ruling 2026-10-19)
    **{r.label: "stare" for r in _stare_landmark_tier_runs()},
    # STARE at 4 CPUs in one job: `stare` itself, another execution
    **{r.label: "stare" for r in _stare_local4_runs()},
    # the pipeline's VALIS at its other tier x depth cells: the same adapter and point path
    **{r.label: "valis_mirage" for r in _valis_mirage_runs() if r.label is not None},
}

# the private plug-in's key keeps its older name
CHECKED_BY.update(plugins.get("SELFCHECK_GATED_BY", {}) or {})

# Runs a public dataset does not take although their method has an input rule for its
# modality: ``{dataset: ((which runs, why), ...)}``, keyed by base name
# (``regbench.base_name``; a key ``<name>@<front end>`` holds that front end's own rules).
# Ported from 5a46f778, where the rule below was a removed landmark set's; it holds on
# every brightfield set, because the adapter refuses ``fluo=1`` on any brightfield case
# (``methods/deeperhistreg.py``). 5a46f778's HyReCo rule (no DeeperHistReg at all, its RGB
# rule being for JPEG / PNG only) is not ported: the adapter now reads an RGB TIFF too.
_FLUO_ON_RGB = (
    (
        lambda r: r.method == "deeperhistreg" and r.opts.get("fluo") == "1",
        "`fluo=1` is the benchmark's configuration of DeeperHistReg for a "
        "fluorescence channel (a black pad, no intensity flip); on an RGB "
        "brightfield image the shipped parameters are the ones to run",
    ),
)
# Ruling 2026-10-19: on a landmark set every method but STARE runs its defaults only:
# the pipeline's VALIS as shipped, VALIS 1.2.0 at both of upstream's settings, and
# DeeperHistReg's two shipped configurations on the GPU. Its other rows (the pipeline
# VALIS's tier x micro-registration cells, DeeperHistReg's CPU twins) are left out.
_LANDMARK_DEFAULTS = (
    "valis_mirage",
    "valis120",
    "valis120_hires",
    "deeperhistreg_gpu",
    "deeperhistreg_hr_gpu",
)
_DEFAULTS_ONLY = (
    (
        lambda r: r.method != "stare" and (r.label or r.method) not in _LANDMARK_DEFAULTS,
        "on a landmark set every method but STARE runs its defaults only (ruling "
        "2026-10-19: " + ", ".join(_LANDMARK_DEFAULTS) + "); STARE runs its full tier x "
        "stride x level cross",
    ),
)
NOT_ON = {
    name: _FLUO_ON_RGB
    + (_DEFAULTS_ONLY if spec["kind"] == "landmarks" else ())
    for name, spec in PUBLIC.items()
    if spec["modality"] == "brightfield"
}
for _dataset, _left_out in (plugins.get("NOT_ON", {}) or {}).items():
    NOT_ON[_dataset] = tuple(NOT_ON.get(_dataset, ())) + tuple(
        (lambda r, n=_name: (r.label or r.method) == n, _why)
        for _name, _why in _left_out.items()
    )

# The methods whose realcheck is run and REPORTED (never a gate): the two VALIS adapters.
# Their point path is derived by the benchmark, not handed over by the package, so it is
# also checked on a real case. It reads the run's PRIMARY row; a run of these methods
# without one has none.
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
    "VALIS_MIRAGE": "cdgatenbee/valis-wsi 1.0.0 (the digest of modules/local/register.nf), "
    "with the slide reader's codecs added: containers/valis100",
    "ASHLAR": "labsyspharm/ashlar:1.20.0, with imagecodecs added: containers/ashlar120",
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
    "checked_by",
    "selfcheck_mask",
    "primary",
    "variants",
    "active",
    "checked_rows",
    "realcheck",
    "stages",
)
NONE = "-"


def run_name(run):
    """What a run's results are stored under: its label, or the method's name."""
    return run.label or run.method


def variants_of(run):
    """The result names one run writes, by the method's own naming rule."""
    return methods.variant_names(run.method, run.label)


def run_variants():
    """``{run name: [the result names it writes]}`` for every run."""
    return {run_name(r): list(variants_of(r)) for r in RUNS}


NO_RECORD = "no selfcheck record"


def row_checks(selfcheck_dir, mine=None):
    """``{result name: what the selfcheck says of THAT row}`` for every run.

    A run's reported status is read from its checked rows; the others may never have
    been compared with an image (``no_stage_image``: a stage the method's writer cannot
    draw, such as ``stare_rigid``), and a run checked by another run's selfcheck has no
    check of its own at all. Tables carry this beside every number (column ``selfcheck_row``):

    ``ok`` / ``inconsistent`` / ``not_checked`` / ``crashed``   this row's own comparison
    ``no_stage_image``            NEVER CHECKED: no image of this stage exists
    ``via <run>: <status>``       not checked itself: its run is checked by ``<run>``'s
    ``no selfcheck record``       nothing was found (a run outside the launcher)

    ``mine(record)`` says whether a record belongs to what is being scored."""
    import json
    from pathlib import Path

    out = {}
    root = Path(selfcheck_dir) if selfcheck_dir else None
    for run in runs_for(None):
        name = run_name(run)
        checker = checked_by(name)
        record = None
        if root is not None:
            try:
                record = json.loads((root / checker / "selfcheck.json").read_text())
            except (OSError, ValueError):
                record = None
        if record is not None and mine is not None and not mine(record):
            record = None
        statuses = (record or {}).get("variants") or {}
        for variant in variants_of(run):
            if record is None:
                out[variant] = NO_RECORD
            elif checker != name:
                out[variant] = f"via {checker}: {record.get('status', 'not_checked')}"
            else:
                out[variant] = statuses.get(variant, NO_RECORD)
    return out


def _check_records(selfcheck_dir, mine=None):
    """``[(run, name, checking run, record or None)]`` for every run: the selfcheck
    record that checks it, when one is there and ``mine`` accepts it."""
    import json
    from pathlib import Path

    root = Path(selfcheck_dir) if selfcheck_dir else None
    out = []
    for run in runs_for(None):
        name = run_name(run)
        checker = checked_by(name)
        record = None
        if root is not None:
            try:
                record = json.loads((root / checker / "selfcheck.json").read_text())
            except (OSError, ValueError):
                record = None
        if record is not None and mine is not None and not mine(record):
            record = None
        out.append((run, name, checker, record))
    return out


MASK_NOT_OVER = "no"


def mask_flags(selfcheck_dir, mine=None):
    """``{result name: flag}`` for every run: did the selfcheck's mask hide more of the
    dots than its cap (``selfcheck.MASK_MAX``) on some case of the selfcheck that checks
    the row's run. Column ``selfcheck_mask_over_cap`` of every table (ruling R3.22: over
    the cap a run is flagged, never blocked):

    ``no``                         checked, and the mask stayed under its cap (or the
                                   method has no mask)
    ``yes: 0.57 masked, cap 0.50`` over the cap on the worst case; ``(via <run>)`` when
                                   the row's run is checked by another run's selfcheck
    empty                          no selfcheck record was found"""
    from . import selfcheck

    out = {}
    for run, name, checker, record in _check_records(selfcheck_dir, mine):
        over = selfcheck.over_cap(record) if record is not None else {}
        for variant in variants_of(run):
            if record is None:
                out[variant] = ""
                continue
            hit = over.get(variant) if checker == name else None
            if checker != name and over:
                hit = max(over.values(), key=lambda v: v["masked_frac"])
            if not hit:
                out[variant] = MASK_NOT_OVER
                continue
            cap = hit["cap"]
            out[variant] = (
                f"yes: {hit['masked_frac']:.2f} masked, cap "
                + ("?" if cap is None else f"{cap:.2f}")
                + (f" (via {checker})" if checker != name else "")
            )
    return out


def selfcheck_mask_rows(selfcheck_dir, mine=None):
    """The rows of ``selfcheck_masks.csv``: every selfcheck record of the run list, per
    case, result name and dot set (``selfcheck.mask_rows``). A record that checks several
    runs is listed once, under the run that made it."""
    from . import selfcheck

    rows, seen = [], set()
    for _, _, checker, record in _check_records(selfcheck_dir, mine):
        if record is None or checker in seen:
            continue
        seen.add(checker)
        rows += selfcheck.mask_rows(record, checker)
    return rows


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
    open on a row that does not exist), it cannot be the run whose selfcheck checks an
    active run (that run would have no verdict), and it cannot be what an
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
        if checked_by(name) not in active:
            raise ValueError(
                f"the run {checked_by(name)!r} is paused and its selfcheck checks the "
                f"active run {name!r}: pause that one too, or change CHECKED_BY"
            )
        if run.after is not None and run.after not in active:
            raise ValueError(
                f"the active run {name!r} waits for {run.after!r}, which is paused"
            )
    names = {run_name(r) for r in RUNS}
    if len(names) != len(RUNS):
        raise ValueError("two runs share a name")
    for dataset, rules in NOT_ON.items():
        if public_of(dataset) is None:
            raise ValueError(f"NOT_ON names no public dataset: {dataset!r}")
        for pick, _ in rules:
            if not any(pick(r) for r in RUNS):
                raise ValueError(f"a rule of NOT_ON[{dataset!r}] matches no run")
    for dataset in PUBLIC:
        kept = {run_name(r) for r in runs_for(dataset)}
        for run in runs_for(dataset):
            name = run_name(run)
            for needs in (checked_by(name), run.after):
                if needs is not None and needs not in kept:
                    raise ValueError(
                        f"on {dataset} the run {name!r} needs {needs!r}, which is left "
                        "out of it"
                    )


def _in_regime(dataset, run):
    """Whether the run's ``regimes`` take this dataset's regime; the cohort (``dataset``
    None) is ``regimes.COHORT_REGIME``. A dataset with no regime
    (``regimes.DATASET_REGIME``) is a KeyError."""
    reg = regimes.COHORT_REGIME if dataset is None else regimes.regime_of(dataset)
    return reg in run.regimes


def _not_on(dataset, run):
    """Why an active run is left out of a public dataset, or None: no input rule for the
    dataset's modality (``methods.input_rule``), or a rule of ``NOT_ON``."""
    spec = public_of(dataset)
    if spec is None:
        return None
    why = methods.input_rule(run.method, spec["modality"])
    if why is not None:
        return why
    keys = dict.fromkeys((base_name(dataset), str(dataset)))
    for key in keys:
        for pick, why in NOT_ON.get(key, ()):
            if pick(run):
                return why
    return None


def runs_for(dataset=None):
    """The active runs of a dataset: those whose ``regimes`` take its regime
    (``Run.regimes``), the cohort's (``regimes.COHORT_REGIME``) for None; on a public
    dataset (``regbench.PUBLIC``), less those left out of it (``excluded_for``). A
    dataset with no regime (``regimes.DATASET_REGIME``) is a KeyError."""
    return tuple(
        r
        for r in active_runs()
        if _in_regime(dataset, r) and _not_on(dataset, r) is None
    )


def excluded_for(dataset):
    """``[(run name, reason)]`` of the active runs of the dataset's regime that are left
    out of it (``_not_on``), in the order of ``RUNS``; ``[]`` for the cohort. Written
    to the dataset's ``excluded_runs.csv`` and its summary."""
    if public_of(dataset) is None:
        return []
    return [
        (run_name(r), _not_on(dataset, r))
        for r in active_runs()
        if _in_regime(dataset, r) and _not_on(dataset, r) is not None
    ]


def primary_for(dataset=None):
    """The primary rows that exist on ``dataset``, in the order of ``PRIMARY``."""
    written = {v for run in runs_for(dataset) for v in variants_of(run)}
    return [v for v in PRIMARY if v in written]


def expected_variants(dataset=None):
    """Every result name the cohort must have, in the order of ``RUNS``: the names of
    the active runs. For a public dataset, the names of ``runs_for(dataset)``."""
    check()
    return [v for run in runs_for(dataset) for v in variants_of(run)]


def scope(method):
    """``pair`` (run case by case) or ``group`` (one process per patient)."""
    return methods.scope_of(method)


def stages_of(run):
    """The scheduler jobs one round of this run is registered in, in order, or ``()``
    for a run that is one job per unit: the method's ``STAGES`` (``regbench.stages``).
    A ``LOCAL_4CPU`` run is one job per unit (its stages run inside it, T31 R2)."""
    if run_name(run) in LOCAL_4CPU:
        return ()
    return tuple(getattr(methods.load(run.method), "STAGES", ()) or ())


def counts():
    """``(runs, runs on a GPU, runs per patient)``, of the cohort's active runs."""
    runs = runs_for(None)
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
    """The cohort's active run names of one family, in the order of ``RUNS``; with
    ``paused``, every run of the family, paused or of another regime."""
    runs = RUNS if paused else runs_for(None)
    return [run_name(r) for r in runs if family(run_name(r)) == family_name]


def primary_of(family_name):
    """The primary result name of a family, or None (the anchored ASHLAR rows). It is
    looked up in ``PRIMARY``, never taken to be the family's own name: the family
    ``valis120`` has the primary ``valis120_hires``."""
    for variant in PRIMARY:
        if family(variant) == family_name:
            return variant
    return None


# rows that are a cost measurement of another row and never a best-row candidate
COST_ONLY = LOCAL_4CPU
# a GPU row's CPU twin: the same configuration on the CPU (R11, P7)
_TWIN = (("_gpu", "_cpu"),)


def cpu_twin(name):
    """The CPU twin of a GPU result name (``deeperhistreg_hr_gpu`` ->
    ``deeperhistreg_hr_cpu``), or None: for a name that is no GPU run's, or whose twin
    is no run."""
    run = None
    try:
        run = run_of(str(name))
    except (KeyError, SystemExit, ValueError):
        return None
    if run is None or not run.gpu:
        return None
    for gpu, cpu in _TWIN:
        if str(name).endswith(gpu):
            twin = str(name)[: -len(gpu)] + cpu
            try:
                other = run_of(twin)
            except (KeyError, SystemExit, ValueError):
                return None
            return twin if other is not None and not other.gpu else None
    return None


def checked_by(name):
    """The name of the run whose selfcheck checks the run ``name`` (reported, never a
    gate)."""
    return CHECKED_BY.get(name, name)


def checked_rows(name):
    """The result names a run's reported selfcheck status is read from: all the names it
    writes. For a run checked by another run's selfcheck (``checked_by``), ask for THAT
    run's names."""
    return tuple(variants_of(run_of(name)))


def realcheck_row(name):
    """The result name whose ``realcheck.json`` is reported for the run ``name``, or
    None: the run's primary row, for a method in ``REALCHECK_METHODS``. Not the exit
    code of ``run --realcheck``, which is also 3 when a non-primary row is flagged."""
    run = run_of(name)
    if run.method not in REALCHECK_METHODS:
        return None
    primary = [v for v in variants_of(run) if v in PRIMARY]
    return primary[0] if primary else None


def _statuses(record, names):
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
    return out


def _reference_status(record, names):
    """The worst status of the second comparison a method that redraws the reference
    has (ASHLAR): ``pairs[].variants.<row>.reference.status``. It is already folded
    into each row's own status; it is reported apart so the summary can name it."""
    from . import selfcheck

    found = [
        ((pair.get("variants") or {}).get(v) or {}).get("reference", {}).get("status")
        for pair in record.get("pairs") or []
        if isinstance(pair, dict)
        for v in names
    ]
    found = [f for f in found if f]
    return selfcheck.worst(found) if found else ""


def selfcheck_status(name, record):
    """``(status, rows, reference)``: what the selfcheck record (``selfcheck.json``) says
    of the run ``name``. REPORTED, never a gate: nothing waits for it and no row is left
    out because of it.

    ``status`` is the worst over ``checked_rows(name)`` (``rows``), read TWICE: from the
    record's summary (``variants``) and from every round of every case
    (``pairs[].variants``), a name missing anywhere counting as ``not_checked`` (as that
    round's own status when it is not ok). So the verdict is never better than what any
    case of the record says of a checked row. An empty record (none was found) is
    ``not_checked``. ``reference`` is the worst reference comparison
    (``_reference_status``), or ``""``."""
    from . import selfcheck

    names = checked_rows(name)
    record = record or {}
    statuses = _statuses(record, names)
    for pair in record.get("pairs") or []:
        rows = pair.get("variants") if isinstance(pair, dict) else None
        rows = rows if isinstance(rows, dict) else {}
        own = pair.get("status") if isinstance(pair, dict) else None
        missing = own if own in selfcheck.NOT_OK else selfcheck.NOT_CHECKED
        for v in names:
            entry = rows.get(v)
            if isinstance(entry, dict):
                statuses.append(entry.get("status", selfcheck.NOT_CHECKED))
            else:
                statuses.append(entry or missing)
    return selfcheck.worst(statuses), names, _reference_status(record, names)


REAL_OK, REAL_FLAGGED, REAL_NOT_CHECKED = "ok", "flagged", "not_checked"
REAL_UNREADABLE = "unreadable"


def realcheck_status(name, out, submission=""):
    """``(status, row, flagged case ids, unreadable case ids)``: what the realchecks of
    the run ``name`` say, from every ``realcheck.json`` of its realcheck row under
    ``out`` (only the run's first unit writes one). REPORTED, never a gate.

    ``flagged`` when one is flagged; ``unreadable`` when a realcheck.json (or the
    run.json beside it) cannot be read and none is flagged (Task 28, S14: it was
    skipped, and the run then read ``not_checked`` or ``ok``); ``not_checked`` when
    there is none of this submission beside an ``ok`` result (the unit failed, or
    wrote no image); else ``ok``. Status and row are ``-`` for a run that has no
    realcheck."""
    import json
    from pathlib import Path

    from . import ladder

    row = realcheck_row(name)
    if row is None:
        return NONE, NONE, [], []
    found, flagged, unreadable = 0, [], []
    for path in sorted(Path(out, row).glob("*/*/realcheck.json")):
        if not (path.parent / "run.json").exists():
            continue  # a realcheck beside no run answers for nothing (review item 13)
        try:
            run = json.loads((path.parent / "run.json").read_text())
            check = json.loads(path.read_text())
        except (OSError, ValueError):
            unreadable.append(path.parent.name)
            continue
        if not isinstance(run, dict) or not isinstance(check, dict):
            unreadable.append(path.parent.name)
            continue
        if not run.get("ok") or not ladder.mine(run, submission):
            continue  # a realcheck an earlier launch left answers for nothing here
        found += 1
        if check.get("flag"):
            flagged.append(path.parent.name)
    status = (
        REAL_FLAGGED
        if flagged
        else REAL_UNREADABLE
        if unreadable
        else REAL_OK
        if found
        else REAL_NOT_CHECKED
    )
    return status, row, flagged, unreadable


def selfcheck_mask(method):
    """What ``selfcheck --mask`` must be given for a method, or None. Read from the
    adapter's ``MASKS``, so the list and the adapter cannot name different masks."""
    masks = getattr(methods.load(method), "MASKS", None) or ()
    return masks[0] if masks else None


def opt_args(run):
    """A run's options as the ``key=value`` strings ``--opt`` takes."""
    return [f"{k}={v}" for k, v in run.opts.items()]


# What a cohort launch submits first when no order is asked for (`runs --order default`):
# ASHLAR, then the affine ASHLAR runs, which are moved behind the run they read.
DEFAULT_ORDER = ("ashlar", "affine_ashlar")
# On a public set, STARE's runs first (plan F addendum 4): the method the benchmark is
# about, then the cohort's default order, each family only where the set has it.
PUBLIC_DEFAULT_ORDER = ("stare", *DEFAULT_ORDER)


def default_order(dataset=None):
    """The family names ``runs --order default`` puts first: ``DEFAULT_ORDER`` on the
    cohort (unchanged), ``PUBLIC_DEFAULT_ORDER`` on a public set, without a family the
    set does not run (ASHLAR on brightfield)."""
    from . import public_of

    if dataset is None or public_of(dataset) is None:
        return list(DEFAULT_ORDER)
    present = {family(run_name(r)) for r in runs_for(dataset)}
    return [f for f in PUBLIC_DEFAULT_ORDER if f in present]


def submission_order(order, runs=None):
    """``(runs in the order they are to be SUBMITTED, notes)`` for a launcher.

    ``order`` is a list of family names (``FAMILY``'s values: every run of the family,
    in the order of ``RUNS``) or run names; a name that is both is the family. The runs
    it does not name follow, in the order of ``RUNS``. An unknown name is refused with
    the names there are.

    Then the one rule the order cannot break: a run that reads another's results
    (``after``) is put directly after that run when the order had it earlier, and a
    note says so. A launcher submits in this order and a job can only wait for a job
    that exists, so an anchored run placed before its anchor would start without it
    (or, on lanes, wait for a run that waits for it).

    This is the order of SUBMISSION only: ``RUNS``, ``runs_table`` without an order,
    every table and every result are untouched."""
    runs = list(runs_for(None) if runs is None else runs)
    by_name = {run_name(r): r for r in runs}
    families = {}
    for r in runs:
        families.setdefault(family(run_name(r)), []).append(run_name(r))
    families.pop(None, None)
    names = []
    for token in order or []:
        picked = families.get(token) or ([token] if token in by_name else None)
        if picked is None:
            raise SystemExit(
                f"run order: no family and no run named '{token}'. Families: "
                f"{', '.join(families)}. Runs: {', '.join(by_name)}."
            )
        names += [n for n in picked if n not in names]
    names += [n for n in by_name if n not in names]
    notes = []
    for name in [n for n in names if by_name[n].after in by_name]:
        anchor = by_name[name].after
        if names.index(name) > names.index(anchor):
            continue
        names.remove(name)
        at = names.index(anchor) + 1
        # behind the anchored runs already put there: they keep their own order
        while at < len(names) and by_name[names[at]].after == anchor:
            at += 1
        names.insert(at, name)
        notes.append(
            f"{name} reads the results of {anchor}: it is submitted directly after "
            f"{anchor}, not before it as the order asked"
        )
    return [by_name[n] for n in names], notes


def runs_table(paused=False, dataset=None, order=None):
    """One row of strings per active run, in the columns of ``RUNS_COLUMNS``; with
    ``paused``, the paused runs too, in their place in ``RUNS``. ``order`` (a list of names,
    possibly empty) puts the rows in ``submission_order``; None leaves them as they are."""
    check()
    table = []
    listed = RUNS if paused else runs_for(dataset)
    if order is not None:
        listed, _ = submission_order(order, listed)
    for run in listed:
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
                checked_by(run_name(run)),
                selfcheck_mask(run.method) or NONE,
                primary[0] if primary else NONE,
                ",".join(names),
                "1" if run.active else "0",
                ",".join(checked_rows(run_name(run))),
                realcheck_row(run_name(run)) or NONE,
                LOCAL_STAGES
                if run_name(run) in LOCAL_4CPU
                else ",".join(stages_of(run)) or NONE,
            )
        )
    return table


check()

"""The list of runs: every configuration the cohort is registered with, and nothing else.

One entry of ``RUNS`` is one ``python -m benchmarks.regbench run`` configuration: a method,
the label its results are stored under, its options, the container image it runs in,
whether it needs a GPU, and the run that must have finished first. The launcher reads the
list (``python -m benchmarks.regbench runs``); the scorer is told which result names to
expect (``expected_variants()``) and which are the primary rows (``PRIMARY``).

Totals are ``counts()`` and ``expected_variants()``; nothing here is a second copy to keep in
step. The STARE and ASHLAR runs are GENERATED from tables (``regimes.REGIMES`` x
``STARE_STRIDES``; ``methods.ashlar.ASHLAR_CELLS``), so a total moves with a table. A run
takes a dataset argument (``runs_for(dataset)``): the only dataset is the cohort's
``multiplex`` (``regimes.DATASET_REGIME``).

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
                       names, except for DeeperHistReg, whose status is read from the
                       row of the written image (``<label>_asdrawn``) alone. Every row
                       stays in every table whatever its selfcheck says.
15. ``realcheck``      the result name whose realcheck is reported for the run, or ``-``
                       (``realcheck_row()``): the run's first unit runs with ``run
                       --realcheck``, beside the rest of the run, which never waits
                       for it.
16. ``stages``         for a run whose registration is several scheduler jobs per round
                       (``stages_of()``): their names in order, joined by commas
                       (``coarse,tile,solve,stitch``); ``-`` for a run that is one job per unit.

The STARE runs of the cohort are its regime's levels x the three strides
(``regimes.DATASET_REGIME``), and two tier runs, ``stare_medium`` and ``stare_low``
(``STARE_TIERS``: stride 64, no preprocessing); ``runs --dataset multiplex`` prints the list.
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

from . import methods, plugins, regimes
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


def _stare_runs():
    """The STARE runs: every stride x every level of the cohort's regime
    (``regimes.COHORT_REGIME``), deduplicated by name."""
    found = {}
    for key, level in regimes.stare_levels(regimes.COHORT_REGIME):
        for s in STARE_STRIDES:
            name, opts = stare_cell(s, key, level)
            found.setdefault(name, opts)
    return tuple(
        Run(
            "stare",
            None if name == "stare" else name,
            opts,
            "STARE_REGBENCH",
            False,
            None,
        )
        for name, opts in found.items()
    )


STARE_TIERS = ("medium", "low")  # beside high (the `stare` row): ruling 2026-10-16


def _stare_tier_runs():
    """STARE's medium and low tiers (``methods.stare.TIERS``, the rows of
    ``lib/RegPresets.groovy``) at stride 64 with no preprocessing: the high tier is the
    ``stare`` row itself."""
    return tuple(
        Run("stare", f"stare_{t}", {"tier": t}, "STARE_REGBENCH", False, None)
        for t in STARE_TIERS
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
    *_stare_runs(),
    *_stare_tier_runs(),
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
    **{run_label: "stare" for r in _stare_runs() if (run_label := r.label) is not None},
    # the tier runs too: the same adapter and pointing path at other tile sizes
    **{r.label: "stare" for r in _stare_tier_runs()},
    # the pipeline's VALIS at its other tier x depth cells: the same adapter and point path
    **{r.label: "valis_mirage" for r in _valis_mirage_runs() if r.label is not None},
}

# the private plug-in's key keeps its older name
CHECKED_BY.update(plugins.get("SELFCHECK_GATED_BY", {}) or {})

# Method -> the suffixes of the result names a run's reported selfcheck status is read
# from, when that is not all of them. DeeperHistReg (ruling R3.12 b): only the row of the
# image its writer puts on disk. Its estimated-field row is checked and reported all the
# same, and stays in every table whatever its selfcheck says (user ruling, option A): its
# own verdict is beside its numbers (`selfcheck_row`).
CHECKED_BY_SUFFIX = {"deeperhistreg": ("_asdrawn",)}

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


def runs_for(dataset=None):
    """The active runs of a dataset: the same list for the cohort (None) and for
    ``multiplex``, the only dataset. A dataset with no regime
    (``regimes.DATASET_REGIME``) is a KeyError."""
    if dataset is not None:
        regimes.regime_of(dataset)
    return active_runs()


def primary_for(dataset=None):
    """The primary rows that exist on ``dataset``, in the order of ``PRIMARY``."""
    written = {v for run in runs_for(dataset) for v in variants_of(run)}
    return [v for v in PRIMARY if v in written]


def expected_variants(dataset=None):
    """Every result name the cohort must have, in the order of ``RUNS``: the names of
    the active runs."""
    check()
    return [v for run in runs_for(dataset) for v in variants_of(run)]


def scope(method):
    """``pair`` (run case by case) or ``group`` (one process per patient)."""
    return methods.scope_of(method)


def stages_of(run):
    """The scheduler jobs one round of this run is registered in, in order, or ``()``
    for a run that is one job per unit: the method's ``STAGES`` (``regbench.stages``)."""
    return tuple(getattr(methods.load(run.method), "STAGES", ()) or ())


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


def checked_by(name):
    """The name of the run whose selfcheck checks the run ``name`` (reported, never a
    gate)."""
    return CHECKED_BY.get(name, name)


def checked_rows(name):
    """The result names a run's reported selfcheck status is read from: all the names it
    writes, unless its method is in ``CHECKED_BY_SUFFIX``. For a run checked by another
    run's selfcheck (``checked_by``), ask for THAT run's names."""
    run = run_of(name)
    names = tuple(variants_of(run))
    suffixes = CHECKED_BY_SUFFIX.get(run.method)
    if not suffixes:
        return names
    rows = tuple(v for v in names if v.endswith(tuple(suffixes)))
    if not rows:
        raise ValueError(f"{name}: none of {names} ends in one of {suffixes}")
    return rows


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
    case of the record says of a checked row. It is the record's own ``status`` except
    where a method's status is read from only some of its rows. An empty record (none
    was found) is ``not_checked``. ``reference`` is the worst reference comparison
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


def realcheck_status(name, out, submission=""):
    """``(status, row, flagged case ids)``: what the realchecks of the run ``name`` say,
    from every ``realcheck.json`` of its realcheck row under ``out`` (only the run's
    first unit writes one). REPORTED, never a gate.

    ``flagged`` when one is flagged; ``not_checked`` when there is none of this
    submission beside an ``ok`` result (the unit failed, or wrote no image); else
    ``ok``. Status and row are ``-`` for a run that has no realcheck."""
    import json
    from pathlib import Path

    from . import ladder

    row = realcheck_row(name)
    if row is None:
        return NONE, NONE, []
    found, flagged = 0, []
    for path in sorted(Path(out, row).glob("*/*/realcheck.json")):
        try:
            run = json.loads((path.parent / "run.json").read_text())
            check = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not run.get("ok") or not ladder.mine(run, submission):
            continue  # a realcheck an earlier launch left answers for nothing here
        found += 1
        if check.get("flag"):
            flagged.append(path.parent.name)
    status = REAL_FLAGGED if flagged else REAL_OK if found else REAL_NOT_CHECKED
    return status, row, flagged


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
                ",".join(stages_of(run)) or NONE,
            )
        )
    return table


check()

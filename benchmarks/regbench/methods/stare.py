"""STARE, run the way the pipeline runs it, with the pipeline's parameters.

``stare_rigid`` is the coarse anchor ``M0`` alone and ``stare`` the full transform (``M0``
plus the mesh). Both come from one registration; the manifest reproduces each stage, so the
rigid one costs nothing extra. Both maps are moving -> reference
(``stare.stage_warp.make_warper``), STARE's own direction.

**Four stage commands per moving round**, as ``subworkflows/local/adapters/
tiled_adapter.nf`` runs them: ``stare coarse`` (the anchor and the tile plan), one
``stare reg-tile`` process per tile, ``stare solve`` (the manifest: the transform), then
``stare stitch`` (the image). Each is a process of its own (``cli``), with the argument
list ``stare.cli.register`` hands that stage (``stage_commands``), and leaves a record
(``regbench.stages``). The benchmark never runs ``stare register`` itself.

**One path to the transform.** COARSE, every tile and SOLVE each run through
``run_stage``, and ``register`` obtains the transform by reading the records they left
(``_collect``): the first stage, in order, that did not end ok for this round's command
fails the round. What differs between two runs is only WHO STARTS the stage processes,
and the run says so in ``REGBENCH_STAGES`` (``launcher``):

``jobs``   the scheduler (the cohort). Each stage is a job of its own (``regbench stage``
        in ``slurm/stage.sbatch``), with the allocation the pipeline gives that process
        and retried as the pipeline retries it. ``register`` starts nothing, and bills
        what the jobs recorded (``info["_cost"]``, ``info["jobs"]``).
``local``  ``register`` itself (``_launch``: the selfcheck, a test, a laptop). The same
        ``run_stage`` calls in the same order, the tiles ``workers`` at a time, each
        retried by the pipeline's rule (``stages.decide``); no stage has a memory or
        time limit of its own. The run's cost is this process tree's, as the driver
        measures any method; ``info["jobs"]`` holds what the stage records add up to.

There is no default. A run that states neither is refused before a stage process is
started, as the launcher's failure (``stages.NOT_SUBMITTED``): ``regbench run`` refuses
it before it touches a record and ends with ``REGBENCH_END ... cause=not_submitted
whose=launcher``; ``register`` refuses it for any other caller. ``slurm/run_rung.sbatch``
exports what the launcher's ``stages`` column says (``jobs``), ``regbench selfcheck``
states ``local`` for its own run, and a run by hand states it with ``regbench run
--stages local``. ``run.json: info.jobs.execution`` names the launcher that ran.

STITCH is the writer under both: ``write_native`` runs it once the transform exists, and
its time is ``write_s``.

**Inputs.** The slides as the case has them (``case.ref_image``, ``case.mov_image``), every
channel, read by STARE's own reader. ``--nuclear-index`` is ``case.extra["mov_channel"]``
(0 when the case names none). STARE takes ONE index for both slides, so a case whose
reference has its nuclei at another index is refused.

**Proxy and sprep.** ``--opt proxy=channel`` names the one-channel image STARE reads (the
cohort is fluorescence: the channel itself, the only value), and ``--opt sprep=none|clahe|
percentile`` is STARE's own ``--prep`` on it (STARE 1.3.0, not a departure): the cohort's
crossed preprocessing factor. ``sprep=clahe`` needs the case's pixel size (CLAHE's region is
in micrometres). A brightfield case is not part of this benchmark: the adapter refuses it.

**Parameters.** ``TIER_HIGH_STRIDE64``: the ``high`` row of the STARE table in
``lib/RegPresets.groovy`` plus stride 64, i.e. the pipeline's ``tiled_high_s64`` arm. All
five are passed explicitly. Every other TRANSFORM parameter of that arm is the command's
default (maximum displacement = halo, model, field step). Two STITCH flags are not, and the
pipeline always passes them (``modules/local/tiled_stitch.nf``): ``--pixel-size`` (the
case's) and ``--channel-names`` (``case.extra["channel_names"]``, else the names in the
moving slide's own header). SOLVE's ``--max-disp`` is the halo, as ``conf/modules.config``
gives ``TILED_SOLVE``. A tile is named by ``--plan``/``--row``: the row of COARSE's tile
plan whose values the pipeline's module passes one by one. COARSE's FFTs use every core
of the node whatever the allocation is, so ``os.cpu_count()`` is recorded.

**Stride.** ``--opt stride=<64|128|256>``, nothing else: 64 when unset. The stride is the
lattice REG-TILE measures on and SOLVE solves; COARSE never sees it (``stare/cli.py:79-100``
passes it no ``--stride``, ``stare/stages/coarse.py:170-211`` has no such argument), SOLVE
writes the anchor into the manifest as COARSE gave it (``stare/stages/solve.py:77-89``) and
the rigid stage applies that matrix alone (``stare/stage_warp.py:85-87``); stare-registration
1.2.1, installed package ``[PARTIAL: cli.py and stage_warp.py in full, main() of
stages/solve.py and stages/stitch.py, manifest.slide_entry; stages/coarse.py and
coarse_align.py only searched for "stride" (no match) and for the argument list]``. A test
finds the anchor identical to the last bit at stride 64 and 256. So ``stare_rigid`` is the same map at every stride, and a stride
run adds one new result, its final variant.

**Output.** STITCH runs after the transform exists, on SOLVE's manifest
(``write_native``). The image is on the reference grid (``out_shape`` = the reference's,
``stare/stages/solve.py:91``): no ``origin_xy``. Nothing is padded; where the moving
slide has no pixels the image is 0. ``stare_rigid`` has no image of its own: STITCH
writes one image.

Other options (``--opt key=value``), for tests and small slides, each recorded in
``info["implementation"]["changed_from_tier"]``: ``tile``, ``halo``, ``out_tile``,
``max_dim``, ``model``, ``max_disp``.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

from .. import imageio, ladder, resources, stages
from ..mapping import MOV_TO_REF, PointMap
from ..regimes import PROXIES, SPREPS
from . import require_fluorescence

SCOPE = "pair"
# the `--opt` keys this adapter reads (methods.known_options): any other is refused
OPTIONS = (
    "proxy",
    "sprep",
    "stride",
    "tile",
    "halo",
    "out_tile",
    "max_dim",
    "model",
    "max_disp",
)
VARIANTS = ("stare_rigid", "stare")
REF_NAME, MOV_NAME = "reference", "moving"
STRIDES = (64, 128, 256)
# the jobs a round is run as on the cohort (regbench.stages); the launcher reads it
STAGES = (stages.COARSE, stages.TILE, stages.SOLVE, stages.STITCH)
# who starts the stage processes of a run (module docstring): stated, never assumed
JOBS_ENV = "REGBENCH_STAGES"
JOBS, LOCAL = "jobs", "local"
# lib/RegPresets.groovy, STARE.high (coarse_max_dim is --max-dim), plus the stride;
# tests/regbench/test_stare_adapter.py reads the row from that file
TIER_HIGH_STRIDE64 = {
    "tile": 2048,
    "halo": 256,
    "out_tile": 1024,
    "max_dim": 1024,
    "stride": 64,
}
_FLAGS = {
    "tile": "--tile",
    "halo": "--halo",
    "out_tile": "--out-tile",
    "max_dim": "--max-dim",
    "stride": "--stride",
    "model": "--model",
    "max_disp": "--max-disp",
}


def settings(opts, case):
    """``(proxy, sprep)`` of a run (module docstring, "Proxy and sprep"). ``proxy`` is the
    one-channel image STARE reads: ``channel``, the slide's own nuclear channel, the only
    value. ``sprep`` is STARE's own ``--prep``, ``none`` when unset. Refused: a value
    outside ``regimes.PROXIES`` / ``SPREPS``; a brightfield case (not part of this
    benchmark); ``sprep=clahe`` where the pixel size is not established (CLAHE's region
    is in micrometres)."""
    require_fluorescence(case, "stare")
    proxy = str(opts.get("proxy", "channel")).strip()
    sprep = str(opts.get("sprep", "none")).strip()
    if proxy not in PROXIES:
        raise ValueError(f"proxy={proxy!r}: expected one of {', '.join(PROXIES)}")
    if sprep not in SPREPS:
        raise ValueError(f"sprep={sprep!r}: expected one of {', '.join(SPREPS)}")
    if sprep == "clahe" and not case.pixel_size_um:
        raise ValueError(
            f"sprep=clahe: {case.case_id} has no established pixel size, and CLAHE's "
            "region is in micrometres"
        )
    return proxy, sprep


def stride_of(opts):
    """The stride of a run: one of ``STRIDES``, 64 when the option is unset."""
    raw = opts.get("stride", TIER_HIGH_STRIDE64["stride"])
    if str(raw).strip() not in {str(s) for s in STRIDES}:
        raise ValueError(f"stride={raw!r}: expected one of 64, 128, 256")
    return int(str(raw).strip())


def nuclear_index(case):
    """The one channel index STARE reads on BOTH slides."""
    mov = int(case.extra.get("mov_channel", 0))
    ref = int(case.extra.get("ref_channel", mov))
    if ref != mov:
        raise ValueError(
            f"{case.case_id}: the nuclear channel is index {ref} on the reference and "
            f"{mov} on the moving slide; STARE takes one --nuclear-index for both"
        )
    return mov


def slides(case):
    """``(reference, moving, nuclear index)`` STARE is given for a case: the original
    slides, every channel."""
    return str(case.ref_image), str(case.mov_image), nuclear_index(case)


def channel_names(case):
    """The moving slide's channel names: the case's, else the ones in the header of the
    file STARE is given."""
    names = case.extra.get("channel_names")
    if names is None:
        try:
            names = imageio.channel_names(slides(case)[1])
        except (
            Exception
        ):  # an unreadable header names nothing; STITCH reads the file itself
            names = []
    return [str(n) for n in names]


def command(case, stage_dir, opts):
    """The parameters of one round, written as the arguments of ``stare register``: the
    package's own parser reads them (``namespace``), so every value has the type and
    the default STARE gives it, and every stage command is built from what it parsed,
    the way ``stare.cli.register`` builds them (``stage_commands``). The list is parsed
    and never run."""
    stage_dir = Path(stage_dir)
    values = {**TIER_HIGH_STRIDE64, "stride": stride_of(opts)}
    values.update({k: opts[k] for k in _FLAGS if k in opts and k != "stride"})
    ref, mov, index = slides(case)
    argv = [
        "register",
        "--reference",
        ref,
        "--moving",
        mov,
        "--out",
        str(stage_dir / "registered.ome.tif"),
        "--manifest",
        str(stage_dir / "manifest.json"),
        "--tre",
        str(stage_dir / "tre.json"),
        "--workdir",
        str(stage_dir),
        "--workers",
        str(opts.get("workers", 1)),
        "--reference-name",
        REF_NAME,
        "--moving-name",
        MOV_NAME,
        "--nuclear-index",
        str(index),
    ]
    for key, value in values.items():
        argv += [_FLAGS[key], str(value)]
    argv += ["--prep", settings(opts, case)[1]]
    if case.pixel_size_um:  # none in the case: the command's own default, `auto`
        argv += ["--pixel-size", repr(float(case.pixel_size_um))]
    names = channel_names(case)
    if names:
        argv += ["--channel-names", *names]
    return argv


def _parse(argv):
    """The command's own parser. Its error message goes to stderr and the exception holds
    only the exit code, so the message is put into the exception: it is the run's error."""
    from stare.cli import build_parser

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return build_parser().parse_args(argv)
    except SystemExit as exc:
        lines = [ln.strip() for ln in err.getvalue().splitlines() if ln.strip()]
        raise SystemExit(lines[-1] if lines else f"stare: exit {exc.code}") from None


# `stare coarse` takes this model and no other (stare/stages/coarse.py:200:
# `choices=["euclidean"]`), while `stare register` lists three (stare/cli.py:228);
# stare-registration 1.2.1, installed package `[PARTIAL: those two lines]`
COARSE_MODELS = ("euclidean",)


def namespace(case, stage_dir, opts):
    """The parameters of one round, as ``stare register``'s parser types them. A
    ``model`` the first stage would refuse is refused here, before any job runs."""
    ns = _parse(command(case, stage_dir, opts))
    if ns.model not in COARSE_MODELS:
        raise SystemExit(
            f"model={ns.model}: `stare coarse` accepts {', '.join(COARSE_MODELS)} and "
            "nothing else, so every round of this run would fail in its first job"
        )
    return ns


def stage_commands(ns):
    """The argument list of each stage, from the parsed ``stare register`` namespace:
    what ``stare.cli.register`` hands ``coarse.main``, ``reg_tile.main``, ``solve.main``
    and ``stitch.main`` (stare/cli.py:79-178; stare-registration 1.2.1, installed
    package ``[FULL: cli.py]``; the ``--prep`` and ``--pixel-size`` arguments added below
    exist from 1.3.0, the pinned version, and are checked against its source by
    ``tests/regbench/test_stages.py``), and what the pipeline's four modules render
    (modules/local/tiled_*.nf, with SOLVE's ``--max-disp`` from conf/modules.config).
    ``tests/regbench/test_stages.py`` evaluates the four lists in the source of
    ``stare.cli.register`` and fails when one differs. ``tile`` and ``stitch`` are
    functions: of a plan row, and of the moving file and the image to write."""
    work = Path(ns.workdir)
    m0, plan = work / f"{ns.moving_name}_m0.json", work / f"{ns.moving_name}_tiles.csv"
    coarse = [
        "--reference",
        str(ns.reference),
        "--moving",
        str(ns.moving),
        "--nuclear-index",
        str(ns.nuclear_index),
        "--tile",
        str(ns.tile),
        "--halo",
        str(ns.halo),
        "--max-dim",
        str(ns.max_dim),
        "--model",
        ns.model,
        "--out-m0",
        str(m0),
        "--out-tiles",
        str(plan),
        "--prep",
        ns.prep,
        "--pixel-size",
        str(ns.pixel_size),
    ]

    def tile(row, ix, iy):
        return [
            "--reference",
            str(ns.reference),
            "--moving",
            str(ns.moving),
            "--m0",
            str(m0),
            "--nuclear-index",
            str(ns.nuclear_index),
            "--plan",
            str(plan),
            "--row",
            str(row),
            "--stride",
            str(ns.stride),
            "--prep",
            ns.prep,
            "--pixel-size",
            str(ns.pixel_size),
            "--out",
            str(work / f"{ns.moving_name}_{ix}_{iy}_ctrl.json"),
        ]

    solve = [
        "--m0",
        str(m0),
        "--controls",
        str(work / f"{ns.moving_name}_*_ctrl.json"),
        "--reference-name",
        ns.reference_name,
        "--moving-name",
        ns.moving_name,
        "--out-manifest",
        str(ns.manifest),
        # the pipeline passes the halo unless reg_tiled_max_disp is set
        "--max-disp",
        str(ns.halo if ns.max_disp is None else ns.max_disp),
        "--out-tre",
        str(ns.tre),
    ]

    def stitch(moving, out):
        argv = [
            "--moving",
            str(moving),
            "--manifest",
            str(ns.manifest),
            "--moving-name",
            ns.moving_name,
            "--out",
            str(out),
            "--out-tile",
            str(ns.out_tile),
            "--pixel-size",
            ns.pixel_size,
        ]
        if ns.channel_names:
            argv += ["--channel-names", *ns.channel_names]
        return argv

    return {
        stages.COARSE: coarse,
        stages.TILE: tile,
        stages.SOLVE: solve,
        stages.STITCH: stitch,
        "m0": m0,
        "plan": plan,
    }


def cli(stage, argv):
    """The process that runs one stage: the package's own command."""
    name = {stages.TILE: "reg-tile"}.get(stage, stage)
    return [sys.executable, "-m", "stare.cli", name, *argv]


def tile_count(case, opts):
    """How many REG-TILE jobs a round has. STARE's tile plan is a function of the
    reference's size, the tile and the halo alone (``stare/stages/coarse.py:363``:
    ``tile_grid(w, h, tile, halo)``; no tile is dropped for its content), so it is
    known before COARSE has run: the launcher sizes the tile arrays from it."""
    from stare.tile_grid import tile_grid

    values = {
        **TIER_HIGH_STRIDE64,
        **{k: opts[k] for k in ("tile", "halo") if k in opts},
    }
    if case.ref_hw:
        h, w = case.ref_hw  # the slide STARE reads
    else:
        _, h, w = imageio.shape_of(slides(case)[0])
    return len(tile_grid(int(w), int(h), int(values["tile"]), int(values["halo"])))


def stage_argv(case, opts, stage_dir="."):
    """The argument list of COARSE and of the first REG-TILE job of a run, from the same
    parsing a round makes (``Round``): ``{"coarse": [...], "reg-tile": [...]}``. Nothing
    is run or written."""
    rnd = Round(case, stage_dir, dict(opts))
    return {
        "coarse": rnd.argv["coarse"],
        "reg-tile": rnd.argv["tile"](0, 0, 0),
        "solve": rnd.argv["solve"],
    }


def stage_dir_of(out, label, case):
    """Where a round's stage jobs keep their files: beside its written image."""
    from ..cases import result_dir
    from . import variant_names

    return result_dir(out, variant_names("stare", label)[-1], case) / "stages"


class Round:
    """One moving round: its parameters, its stage commands, and its job records."""

    def __init__(self, case, stage_dir, opts):
        self.case, self.dir, self.opts = case, Path(stage_dir), opts
        self._identity = {}
        self.proxy, self.sprep = settings(opts, case)
        self.ns = namespace(case, self.dir, opts)
        self.argv = stage_commands(self.ns)

    def command(self, stage, row=None):
        if stage == stages.TILE:
            r = self.plan()[row]
            return cli(stage, self.argv[stage](row, int(r["ix"]), int(r["iy"])))
        return cli(stage, self.argv[stage])

    def plan(self):
        from stare.stages.reg_tile import plan_rows

        return plan_rows(self.argv["plan"])

    def record(self, stage, row=None):
        return stages.read_record(self.dir, stage, row)

    def identity(self, stage, row=None):
        """What a stage job is computed FROM, beyond its command line, as one digest
        (``stages.decide``): the STARE version; for COARSE and a tile the bytes
        (``cases.file_fingerprint``) of the two files STARE reads; for a tile also the
        anchor and the tile plan COARSE wrote; for SOLVE the anchor and every control
        file of the plan. A stage whose inputs were replaced, or whose stage before
        ran again and wrote something else, is another job. The same for every row."""
        import hashlib

        import stare

        from ..cases import file_fingerprint

        if stage in self._identity:
            return self._identity[stage]

        def sha(path):
            try:
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError:
                return None

        parts = {"stage": stage, "stare": stare.__version__}
        if stage in (stages.COARSE, stages.TILE):
            parts["inputs"] = [
                file_fingerprint(self.ns.reference),
                file_fingerprint(self.ns.moving),
            ]
        if stage in (stages.TILE, stages.SOLVE):
            parts["m0"] = sha(self.argv["m0"])
        if stage == stages.TILE:
            parts["plan"] = sha(self.argv["plan"])
        if stage == stages.SOLVE:
            parts["controls"] = [
                [self.control(i).name, sha(self.control(i))]
                for i in range(len(self.plan()))
            ]
        text = json.dumps(parts, sort_keys=True)
        self._identity[stage] = hashlib.sha256(text.encode()).hexdigest()
        return self._identity[stage]

    def done(self, stage, row=None):
        """The job's record when it is ok FOR THIS JOB (the same command, computed
        from the same things: ``identity``), else None."""
        rec = self.record(stage, row)
        if (
            rec
            and rec.get("ok")
            and stages.same_job(rec, self.command(stage, row), self.identity(stage))
        ):
            return rec
        return None

    def control(self, row):
        return Path(self.command(stages.TILE, row)[-1])

    def tiles_done(self):
        """Every tile's record, or the reason SOLVE may not run: it reads a glob, so a
        missing tile would not stop it, and a file no tile of this plan wrote would
        be read with the others."""
        rows = self.plan()
        recs = [self.done(stages.TILE, i) for i in range(len(rows))]
        missing = [i for i, r in enumerate(recs) if r is None]
        if missing:
            return (
                None,
                f"{len(missing)} of {len(rows)} tiles are not ok (first: row {missing[0]})",
            )
        wanted = {str(self.control(i)) for i in range(len(rows))}
        found = set(glob.glob(self.argv[stages.SOLVE][3]))
        if found != wanted:
            return None, (
                f"the control files on disk are not this plan's {len(rows)}: "
                f"{len(wanted - found)} missing, {len(found - wanted)} of another plan"
            )
        return recs, ""


def run_stage(
    case,
    stage_dir,
    opts,
    stage,
    row=None,
    attempt=1,
    execution=JOBS,
    force=False,
    rnd=None,
):
    """One stage job of one round: ``coarse``, ``tile`` (one plan row; with ``row=None`` every row that is due, one after another)
    or ``solve``. Returns ``[(stage, row, ran?, why, record)]``. The one entry of every
    stage process, whoever starts it (``execution``: ``jobs`` or ``local``).

    A job runs only when the stage before it is ok for this command, and when
    ``stages.decide`` says this attempt is due; otherwise it leaves everything as it
    is. STITCH is not a job of this kind: it is the writer (``write_native``). ``rnd``
    is the round when the caller holds it already (``_launch``, whose tiles run in
    threads: one ``Round`` for all of them)."""
    rnd = rnd or Round(case, stage_dir, opts)
    submission = resources.host_info()["submission"]
    out = []

    def one(stage, row=None):
        rec = rnd.record(stage, row)
        command = rnd.command(stage, row)
        identity = rnd.identity(stage)
        go, why = stages.decide(rec, attempt, command, submission, force, identity)
        if go:
            rec = stages.run_command(
                rnd.dir, stage, command, row, attempt, execution, identity=identity
            )
        out.append((stage, row, go, why, rec))

    if stage == stages.COARSE:
        rnd.dir.mkdir(parents=True, exist_ok=True)
        one(stage)
        return out
    if not rnd.done(stages.COARSE):
        return [(stage, row, False, "COARSE is not ok", None)]
    if stage == stages.TILE:
        n = len(rnd.plan())
        if row is not None and not 0 <= int(row) < n:
            return [(stage, row, False, f"the plan has {n} tiles", None)]
        for i in range(n) if row is None else [int(row)]:
            one(stage, i)
        return out
    if stage == stages.SOLVE:
        _, why = rnd.tiles_done()
        if why:
            return [(stage, row, False, why, None)]
        one(stage)
        return out
    raise ValueError(f"stage {stage!r}: one of coarse, tile, solve")


def tiles_due(case, stage_dir, opts, attempt, force=False):
    """The plan rows of a round whose tile job is due at ``attempt``
    (``stages.decide``): what the launcher's tile submitter puts into the queue.
    None of a round whose COARSE is not ok."""
    rnd = Round(case, stage_dir, opts)
    if not rnd.done(stages.COARSE):
        return []
    submission = resources.host_info()["submission"]
    return [
        i
        for i in range(len(rnd.plan()))
        if stages.decide(
            rnd.record(stages.TILE, i),
            attempt,
            rnd.command(stages.TILE, i),
            submission,
            force,
            rnd.identity(stages.TILE),
        )[0]
    ]


def _failed(rnd, stage, row, jobs):
    """The error of a round whose job ``stage`` did not end ok for this command."""
    rec = rnd.record(stage, row)
    what = stage.upper() + (f" (tile row {row})" if row is not None else "")
    mine = stages.same_job(
        rec, rnd.command(stage, row), rnd.identity(stage)
    ) and ladder.mine(rec, resources.host_info()["submission"])
    # A tile job the launcher's submitter did not see through: never put into the
    # queue (no record of this submission), or left without the attempt the pipeline
    # would have given it next (its record is a placeholder of a task that had not
    # ended, or a failure the pipeline retries, below the last attempt). The submitter
    # waits for every tile task to leave the queue before it writes its `.done`, so
    # with a `.done` neither can be.
    unfinished = (
        mine
        and stages.retryable(rec)
        and int(rec.get("attempt") or 1) < (stages.ATTEMPTS)
    )
    never = ""
    if (not mine or unfinished) and stage == stages.TILE and jobs:
        never = stages.feed_problem(
            rnd.dir.parents[3],
            resources.host_info()["submission"],
            str(rnd.opts.get("label") or "") or "stare",
            rnd.case.dataset,
        )
    if never and mine:
        err = stages.StageFailed(
            f"{what}: attempt {rec.get('attempt')} of {stages.ATTEMPTS} "
            + (
                "had not recorded its end when the round's last job ran"
                if rec.get("exit") is None
                else f"ended with exit {rec.get('exit')}, which the pipeline retries"
            )
            + f", and the attempt after it was never submitted ({never})",
            stages.NOT_SUBMITTED,
        )
    elif never:
        # the launcher's tile submitter did not see this job through: it was never put
        # into the queue, or was still waiting there when the round's last job ran
        err = stages.StageFailed(
            f"{what}: this job never ran: it was not submitted, or was still waiting "
            f"in the queue when the round's last job ran ({never})",
            stages.NOT_SUBMITTED,
        )
    elif not mine:
        # no job of this submission wrote a record: none of them started
        err = stages.StageFailed(
            f"{what}: no attempt of this job left a record (lost to the scheduler)",
            "infrastructure",
        )
    else:
        err = stages.StageFailed(
            f"{what} failed at attempt {rec.get('attempt')} of {stages.ATTEMPTS}, "
            f"with exit {rec.get('exit')} at {rec.get('mem_gb')} GB (job "
            f"{rec.get('launch_id')}): {str(rec.get('error') or '')[-600:]}",
            rec.get("cause") or "exception",
        )
    # in its own job this failure was already retried as the pipeline retries it
    err.regbench_no_rerun = jobs
    return err


def _launch(rnd, workers):
    """The local launcher (``REGBENCH_STAGES=local``): start the round's stage processes
    from here, as the scheduler starts the cohort's. Every one goes through
    ``run_stage``, attempt after attempt while ``stages.decide`` says the next one is
    due (the pipeline's retry rule), the tiles ``workers`` at a time. It stops at the
    first stage that did not end ok and says nothing about it: ``_collect`` reads the
    records and fails the round."""
    from concurrent.futures import ThreadPoolExecutor

    def start(stage, row=None):
        for attempt in range(1, stages.ATTEMPTS + 1):
            done = run_stage(
                rnd.case, rnd.dir, rnd.opts, stage, row, attempt, LOCAL, rnd=rnd
            )
            rec = done[-1][4] if done else None
            if rec is None or rec.get("ok") or not stages.retryable(rec):
                break
        return bool(rec and rec.get("ok"))

    if not start(stages.COARSE):
        return
    rnd.identity(stages.TILE)  # once, before the pool: the same for every row
    with ThreadPoolExecutor(max(1, int(workers))) as pool:
        tiles = list(pool.map(lambda i: start(stages.TILE, i), range(len(rnd.plan()))))
    if all(tiles):
        start(stages.SOLVE)


def _collect(rnd, jobs):
    """The round's transform exists: every stage process left an ok record for this
    round's command. Nothing is run here, whoever started the stages (``jobs``: the
    scheduler). Raises for the first stage, in order, that is not ok."""
    if not rnd.done(stages.COARSE):
        raise _failed(rnd, stages.COARSE, None, jobs)
    for i in range(len(rnd.plan())):
        if not rnd.done(stages.TILE, i):
            raise _failed(rnd, stages.TILE, i, jobs)
    _, why = rnd.tiles_done()
    if why:
        raise stages.StageFailed(f"SOLVE's inputs are not complete: {why}")
    if not rnd.done(stages.SOLVE):
        raise _failed(rnd, stages.SOLVE, None, jobs)


def launcher():
    """Who starts the stage processes of this run: ``JOBS`` or ``LOCAL``, as
    ``REGBENCH_STAGES`` states it. A run that states neither is refused: it was
    launched without its stage jobs, and that is the launcher's failure."""
    stated = os.environ.get(JOBS_ENV) or ""
    if stated in (JOBS, LOCAL):
        return stated
    err = stages.StageFailed(
        f"{JOBS_ENV}={stated or 'unset'}: a STARE round is several stage processes, "
        f"and the run must say who starts them: `{JOBS}` (the scheduler: the cohort's "
        f"stage jobs, slurm/stage.sbatch) or `{LOCAL}` (this run itself: the "
        "selfcheck, `regbench run --stages local`). Nothing was started",
        stages.NOT_SUBMITTED,
    )
    err.regbench_no_rerun = True  # another rung of the same launch states no more
    raise err


def REUSE_KEY(cases, out, opts, label):
    """What a result of this run rests on beyond its options and its inputs' bytes
    (``methods.reuse_key``): the STARE version, and, for a round that ran as jobs of
    its own, the identity of the SOLVE job its stage records hold now. A round whose
    stages ran again on other bytes is then another result, not a reused one."""
    import stare

    key = {"stare": stare.__version__}
    if os.environ.get(JOBS_ENV) == JOBS:
        solved = {}
        for case in cases:
            rnd = Round(
                case, stage_dir_of(out, label, case), {**opts, "label": label or ""}
            )
            rec = rnd.record(stages.SOLVE) or {}
            solved[case.case_id] = rec.get("identity") if rec.get("ok") else None
        key["solve"] = solved
    return key


def register(case, work, opts):
    import stare
    from stare.stage_warp import make_warper

    jobs = launcher() == JOBS
    where = opts.get("native_dir")
    if jobs:
        if where is None:
            raise ValueError(f"{JOBS_ENV}=jobs needs the driver (its results folder)")
        stage_dir = Path(where(case.case_id, VARIANTS[-1])).parent / "stages"
    else:
        stage_dir = Path(work) / "stages"
    rnd = Round(case, stage_dir, opts)
    ns = rnd.ns
    try:
        if not jobs:
            _launch(rnd, ns.workers)
        _collect(rnd, jobs)
    except stages.StageFailed as err:
        # a round that failed still cost its stage jobs: what they recorded goes with
        # the failure (the driver keeps it in run.json: info.jobs)
        try:
            n = len(rnd.plan()) if rnd.argv["plan"].exists() else 0
            spent = stages.summarise(
                rnd.record(stages.COARSE),
                [rnd.record(stages.TILE, i) for i in range(n)],
                rnd.record(stages.SOLVE),
            )
            spent["execution"] = JOBS if jobs else LOCAL
            err.regbench_info = {"jobs": spent}
        except Exception:  # the records are unreadable: the failure stands alone
            pass
        raise
    cost = stages.summarise(
        rnd.record(stages.COARSE),
        [rnd.record(stages.TILE, i) for i in range(len(rnd.plan()))],
        rnd.record(stages.SOLVE),
    )
    cost["execution"] = JOBS if jobs else LOCAL
    man = json.loads(Path(ns.manifest).read_text())
    warp = make_warper(man)
    entry = man["slides"][MOV_NAME]
    changed = {
        k: str(getattr(ns, k))
        for k in _FLAGS
        if k in opts and getattr(ns, k) != TIER_HIGH_STRIDE64.get(k)
    }
    ref_in, mov_in, _ = slides(case)
    info = {
        "version": stare.__version__,
        "source": str(Path(stare.__file__).parent),
        # the anchor, moving px -> reference px: the whole of `stare_rigid`, and the same
        # matrix at every stride (module docstring)
        "m0": entry["M0"],
        "input": {
            "reference": ref_in,
            "moving": mov_in,
            "kind": "original slides",
            "proxy_option": rnd.proxy,
            "sprep": rnd.sprep,
        },
        "workers": int(ns.workers),
        "cpu_count": os.cpu_count(),  # what COARSE's FFTs use, whatever the allocation
        "stride": int(ns.stride),
        "max_disp": float(ns.halo if ns.max_disp is None else ns.max_disp),
        "model": ns.model,
        "nuclear_index": int(ns.nuclear_index),
        "pixel_size": ns.pixel_size,
        "channel_names": list(ns.channel_names or []),
        # what the round's stage records add up to (regbench.stages.summarise), and
        # who started the stage processes (`execution`)
        "jobs": cost,
        "implementation": {
            "package": "stare-registration",
            "source": str(Path(stare.__file__).parent),
            "registration": "the four stage commands `stare coarse`, `stare reg-tile` "
            "(one process per tile), `stare solve`, `stare stitch`, as the pipeline's "
            "TILED_COARSE, TILED_REG_TILE, TILED_SOLVE and TILED_STITCH run them",
            "parameters": "mirage's `high` STARE tier (lib/RegPresets.groovy), stride "
            f"{TIER_HIGH_STRIDE64['stride']}: {TIER_HIGH_STRIDE64}",
            "changed_from_tier": changed,
            "point_warp": "stare.stage_warp.make_warper",
            "benchmark_side": [
                "STITCH is the writer: it runs once the transform exists and is timed "
                "apart (write_s)",
                (
                    "each stage is a scheduler job of its own with the pipeline's "
                    "allocation for that process (regbench.stages.ALLOCATION); a tile "
                    "that is retried runs in one job per round and attempt with the "
                    "other retried tiles of that round, one after another"
                    if jobs
                    else "the stage processes were started by this run itself "
                    f"({JOBS_ENV}={LOCAL}), the tiles {int(ns.workers)} at a time, "
                    "retried by the pipeline's rule and with no memory or time "
                    "limit of their own: not the cohort's scheduler jobs, so this "
                    "run's cost is its own process tree's"
                ),
            ],
        },
    }
    if jobs:
        # the registration ran in other jobs: this process's own clock saw none of it
        info["_cost"] = {
            "wall_s": cost["wall_unlimited_parallel_s"],
            "cpu_s": cost["cpu_h_measured"] * 3600.0,
            "peak_rss_mb": cost["largest_job_peak_mb"],
            "rss_method": "ru_maxrss of each stage command (os.wait4), the largest",
        }
    if rnd.argv["m0"].exists():
        info["coarse_trusted"] = json.loads(rnd.argv["m0"].read_text()).get(
            "coarse_trusted"
        )
    yield (
        "stare_rigid",
        PointMap(lambda xy: warp(MOV_NAME, xy, "rigid"), MOV_TO_REF),
        info,
    )
    if Path(ns.tre).exists():
        info = {**info, "solve": json.loads(Path(ns.tre).read_text()).get("solve", {})}
    state = {
        "stage_dir": str(rnd.dir),
        "stitch": rnd.argv[stages.STITCH],
        "moving": ns.moving,
        "execution": cost["execution"],
        "out_shape": entry["out_shape"],
    }
    yield (
        "stare",
        PointMap(lambda xy: warp(MOV_NAME, xy, "refined"), MOV_TO_REF),
        {**info, "_state": state},
    )


def _own_rss_gb():
    """This process's own peak memory, in GB (the benchmark's part of the job)."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * (1.0 if sys.platform == "darwin" else 1024.0) / (1 << 30)


def write_native(state, case_id, dest_dir):
    """STARE's own STITCH, as a process of its own, on the manifest SOLVE wrote.

    On the cohort it runs inside the round's LAST JOB, which has the benchmark's
    allocation because the benchmark's own work is in it (the points, the tissue
    reduction). STITCH is still the pipeline's process: measured alone (``os.wait4``),
    ended at the pipeline's time limit for its attempt, and run again as the pipeline
    retries it (``stages.RETRY_EXIT``, the time limit included), up to four attempts.
    The job is not STITCH's cgroup, so the pipeline's MEMORY for an attempt is not
    enforced: ``stitch_job.over_allocation`` says when STITCH's peak was above it.

    What a STITCH killed for memory becomes depends on who sees it first. Where the
    driver's out-of-memory watcher can read the job's cgroup counter (a scheduler step's
    cgroup: the cohort), it interrupts this call during the attempt the kernel killed,
    usually the first: no later attempt runs, the STITCH process still running is
    terminated (``stages.run_command``), the row is ``writer_oom`` (the kill was inside
    the writer's stretch), and the job goes up a rung like any memory cause. Where
    there is no such counter the loop below decides: a STITCH killed for memory while
    its allocation plus this process's own memory was more than the job had is the
    job's limit, not STITCH's (``benchmark_oom``, and the job goes up a rung); else it
    is run again as the pipeline retries it, and a STITCH that fails after its attempts
    fails the row as the writer's (``writer`` / ``writer_oom``) with no rung running
    it again."""
    from stare.slide_io import open_lazy

    out = Path(dest_dir) / "registered.ome.tif"
    jobs = state["execution"] == JOBS
    job_gb = resources.host_info().get("mem_requested_gb")
    command = cli(stages.STITCH, state["stitch"](state["moving"], out))
    rec, failed = None, []
    for attempt in range(1, (stages.ATTEMPTS if jobs else 1) + 1):
        alloc = stages.allocation(stages.STITCH, attempt)
        rec = stages.run_command(
            state["stage_dir"],
            stages.STITCH,
            command,
            attempt=attempt,
            execution=state["execution"],
            timeout_s=alloc["hours"] * 3600 if jobs else None,
        )
        if rec["ok"]:
            break
        failed.append(
            {k: rec.get(k) for k in ("attempt", "exit", "cause", "wall_s", "cpus")}
        )
        if (
            jobs
            and rec.get("cause") == "oom"
            and job_gb
            and alloc["mem_gb"] + _own_rss_gb() > float(job_gb)
        ):
            raise stages.StageFailed(
                f"stare stitch was killed for memory at attempt {attempt} (the "
                f"pipeline gives it {alloc['mem_gb']} GB) in a job of {job_gb} GB that "
                f"also holds the benchmark's own {_own_rss_gb():.1f} GB: the job's "
                "limit, not STITCH's allocation",
                "benchmark_oom",
            )
        if not stages.retryable(rec):
            break
    if not rec["ok"]:
        err = stages.StageFailed(
            f"stare stitch failed at attempt {rec['attempt']} of "
            f"{stages.ATTEMPTS if jobs else 1} with exit {rec['exit']} "
            f"({rec.get('cause')}): {str(rec['error'])[-600:]}",
            "writer_oom" if rec.get("cause") == "oom" else "writer",
        )
        # retried here as the pipeline retries it: no rung runs the round again
        err.regbench_no_rerun = jobs
        err.failed_attempts = failed
        raise err
    arr, dtype, close = open_lazy(out)
    try:
        channels, hw = int(arr.shape[0]), [int(v) for v in arr.shape[1:]]
    finally:
        close()
    # No `origin_xy`: STITCH draws on the reference grid. Checked, not assumed.
    if hw != [int(v) for v in state["out_shape"]]:
        raise RuntimeError(
            f"stare stitch wrote {hw}, not the reference grid {state['out_shape']}"
        )
    native = {
        "files": [str(out)],
        "format": "ome-tiff",
        "channels": channels,
        "dtype": str(np.dtype(dtype)),
        "writer": "stare.stages.stitch.main",
        # STITCH's own process: the pipeline's allocation for the attempt that wrote
        # the image, its time, CPU time and peak memory; and the attempts before it
        "stitch_job": {
            **{
                k: rec[k]
                for k in ("attempt", "cpus", "mem_gb", "wall_s", "cpu_s", "peak_rss_mb")
            },
            "over_allocation": bool(rec["peak_rss_mb"] > rec["mem_gb"] * 1024.0),
            "failed_attempts": failed,
        },
        # the row's write_s: the attempt that wrote the image (a failed one is wasted)
        "write_s": rec["wall_s"],
    }
    return native

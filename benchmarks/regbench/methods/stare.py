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
(``regbench.stages``). Two ways to run them, one code path for the commands:

``REGBENCH_STAGES=jobs``  the cohort. COARSE, every tile and SOLVE ran as scheduler jobs
        of their own (``run_stage``, called by ``regbench stage`` in
        ``slurm/stage.sbatch``), each with the allocation the pipeline gives that
        process and retried as the pipeline retries it. ``register`` runs nothing: it
        reads their records (``_collect``), fails the round when a stage did not end
        ok, and bills what the jobs recorded (``info["_cost"]``, ``info["jobs"]``).
otherwise  the selfcheck, a laptop: ``register`` runs the same commands itself
        (``_local``), the tiles ``workers`` at a time, and the run's cost is its own
        process's.

STITCH is the writer in both: ``write_native`` runs it once the transform exists, and
its time is ``write_s``.

**Inputs.** The slides as the case has them (``case.ref_image``, ``case.mov_image``), every
channel, read by STARE's own reader. ``--nuclear-index`` is ``case.extra["mov_channel"]``
(0 when the case names none). STARE takes ONE index for both slides, so a case whose
reference has its nuclei at another index is refused.

**A brightfield case** (HyReCo, ANHIR: ``case.modality == "brightfield"``). ``stare
register`` reads ONE channel per slide, so an RGB slide cannot be handed over as it is.
The case carries a one-channel proxy of each slide, written by its dataset module
(``case.ref_nuclear`` / ``case.mov_nuclear``; ``datasets/hyreco.py``: inverted luminance
by default; ``datasets/anhir.py``: inverted luminance, ``regbench.prep.grey_plane``), and
that is what STARE is given, at index 0. This is benchmark-side
preprocessing and is recorded as such (``info["input"]``,
``implementation.benchmark_side``). STARE's archived benchmark harness did the same
(``benchmarks/regbench/datasets/hyreco.py`` at the tag ``archive/benchmarking-2026-10-08``
of sceriff0/stare, read from a local copy ``[FULL: that file; its methods/stare.py only
searched for the two slide arguments]``: an inverted-luminance OME-TIFF per slide, passed
as ``--reference`` / ``--moving``). When the proxy is reduced
(``case.extra["settings"]["reduce"]`` = f > 1, block-averaged), STARE works in the reduced
frame: ``--pixel-size`` is f times the slide's, and each point map takes a full-resolution
point to the reduced frame (``(x - (f - 1) / 2) / f``, the centre of an f x f block),
through STARE's own warper, and back. The written image is then the registered PROXY at
the reduced size (``native["reduced_by"]``), not an RGB slide.

**The preprocessing row** (``--label stare_prep --opt prep=<candidate>``, ruling R3.21).
STARE is unchanged: same package, same parameters, same stages. Before COARSE, both
images are rewritten as one-channel files by a candidate of ``regbench.prep`` (the stage
``prep``: a job of its own on the cohort with the benchmark's allocation, unbilled,
``info["prep"]["prep_s"]``) and STARE registers THOSE, at index 0. STITCH is
then run on what the plain row would stitch (the original moving slide; for a brightfield
case its plain grey proxy), so the written image is never the preprocessed pixels and
both rows write the same kind of image. A candidate changes no pixel grid
(``prep.frame_check``, run before every case, refuses otherwise), so the manifest is in
the slide's own frame. A label and its option come together: ``stare_prep`` takes the
candidate frozen in ``prep_frozen.json`` and no other, ``stare_prep_<candidate>`` (the
tuning runs) takes that candidate, and ``prep`` without one of these labels is refused,
so a preprocessed result can never be stored as ``stare``.

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
import time
from pathlib import Path

import numpy as np

from .. import imageio, ladder, prep, resources, stages
from ..mapping import MOV_TO_REF, PointMap

SCOPE = "pair"
VARIANTS = ("stare_rigid", "stare")
REF_NAME, MOV_NAME = "reference", "moving"
# STITCH needs a pixel size for the header of the image it writes and uses it for nothing
# else (`stare/cli.py:163-178`: it is passed to STITCH alone). A brightfield case whose
# pixel size is not established (ANHIR's mammary glands) gets this placeholder there;
# the transform and every score are in pixels and do not see it.
NOMINAL_PIXEL_SIZE = 1.0
STRIDES = (64, 128, 256)
# the jobs a round is run as on the cohort (regbench.stages); the launcher reads it
STAGES = (stages.COARSE, stages.TILE, stages.SOLVE, stages.STITCH)
# set to `jobs` by the launcher's last job of a round: the stages ran as jobs of their
# own and `register` collects them; unset, it runs them itself (`_local`)
JOBS_ENV = "REGBENCH_STAGES"
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


class _NullMonitor:
    def pause(self):
        pass

    def resume(self):
        pass


def prep_of(opts):
    """The preprocessing candidate of a run, or None (the plain row). Checked against
    the label (module docstring)."""
    name = opts.get("prep")
    label = str(opts.get("label") or "")
    mine = label == prep.ROW or label.startswith(prep.ROW + "_")
    if not name:
        if mine:
            raise ValueError(
                f"the label {label!r} names a preprocessing row; it needs --opt "
                "prep=<candidate>"
            )
        return None
    name = str(name)
    prep.candidate(name)
    if label == prep.ROW:
        frozen = prep.selected()
        if name != frozen:
            raise ValueError(
                f"the row {prep.ROW!r} runs the frozen candidate {frozen!r} "
                f"(prep_frozen.json), not {name!r}. A label and its option come together"
            )
    elif label != f"{prep.ROW}_{name}":
        raise ValueError(
            f"prep={name} needs --label {prep.ROW} (the frozen candidate) or --label "
            f"{prep.ROW}_{name} (a tuning run); with the label {label!r} its result "
            "would be stored under another row's name"
        )
    return name


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
            f"{mov} on the moving slide; `stare register` takes one --nuclear-index for both"
        )
    return mov


def slides(case):
    """``(reference, moving, nuclear index, reduction)`` STARE is given for a case: the
    original slides, or for a brightfield case its one-channel proxies (module
    docstring)."""
    if case.modality == "brightfield":
        f = int((case.extra.get("settings") or {}).get("reduce", 1))
        if f < 1 or not (case.ref_nuclear and case.mov_nuclear):
            raise ValueError(
                f"{case.case_id}: a brightfield case needs a one-channel proxy of each "
                "slide (ref_nuclear / mov_nuclear) and a reduction of at least 1"
            )
        return str(case.ref_nuclear), str(case.mov_nuclear), 0, f
    return str(case.ref_image), str(case.mov_image), nuclear_index(case), 1


def reduced_map(fn, f):
    """``fn`` (a map between two frames reduced f-fold by block averaging) as a map
    between the full-resolution frames: block ``i`` is centred on pixel ``i f + (f - 1) /
    2``."""
    if f == 1:
        return fn
    shift = (f - 1) / 2.0
    return lambda xy: np.asarray(fn((np.asarray(xy, float) - shift) / f)) * f + shift


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


def info_proxy(case):
    """What a brightfield case's proxy is, in a few words."""
    settings = case.extra.get("settings") or {}
    f = int(settings.get("reduce", 1))
    return f"{settings.get('proxy', 'proxy')}, " + (
        f"reduced {f} x {f}" if f > 1 else "full resolution"
    )


def command(case, stage_dir, opts, inputs=None):
    """The ONE ``stare register`` command line that is this round: the package's own
    parser reads it (``namespace``), and every stage command is built from what it
    parsed, the way ``stare.cli.register`` builds them (``stage_commands``). Running it
    (``stare.cli.main(command(...))``) is the same registration in one process.
    ``inputs`` = ``(reference, moving)`` replaces the two files STARE registers on (the
    preprocessing row: one-channel files, index 0); everything else is the case's."""
    stage_dir = Path(stage_dir)
    values = {**TIER_HIGH_STRIDE64, "stride": stride_of(opts)}
    values.update({k: opts[k] for k in _FLAGS if k in opts and k != "stride"})
    ref, mov, index, reduce = slides(case)
    if inputs is not None:
        ref, mov, index = str(inputs[0]), str(inputs[1]), 0
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
    if case.pixel_size_um:  # none in the case: the command's own default, `auto`
        argv += ["--pixel-size", repr(float(case.pixel_size_um) * reduce)]
    elif case.modality == "brightfield":
        # a proxy written here has no size in its header for `auto` to find
        argv += ["--pixel-size", repr(NOMINAL_PIXEL_SIZE)]
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


def namespace(case, stage_dir, opts, inputs=None):
    """The parameters of one round, as ``stare register``'s parser types them."""
    return _parse(command(case, stage_dir, opts, inputs))


def stage_commands(ns):
    """The argument list of each stage, from the parsed ``stare register`` namespace:
    what ``stare.cli.register`` hands ``coarse.main``, ``reg_tile.main``, ``solve.main``
    and ``stitch.main`` (stare/cli.py:79-178), and what the pipeline's four modules
    render (modules/local/tiled_*.nf, with SOLVE's ``--max-disp`` from
    conf/modules.config). ``tile`` and ``stitch`` are functions: of a plan row, and of
    the moving file and the image to write."""
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
    prep_of(opts)  # a label and its option come together: refused here too
    if case.modality == "fluorescence" and case.ref_hw:
        h, w = (
            case.ref_hw
        )  # the slide STARE reads, or its preprocessed copy (same grid)
    else:  # a brightfield case: the proxy STARE is given may be a reduced one
        _, h, w = imageio.shape_of(slides(case)[0])
    return len(tile_grid(int(w), int(h), int(values["tile"]), int(values["halo"])))


def stage_dir_of(out, label, case):
    """Where a round's stage jobs keep their files: beside its written image."""
    from ..cases import result_dir
    from . import variant_names

    return result_dir(out, variant_names("stare", label)[-1], case) / "stages"


class Round:
    """One moving round: its parameters, its stage commands, and its job records."""

    def __init__(self, case, stage_dir, opts):
        self.case, self.dir, self.opts = case, Path(stage_dir), opts
        self.candidate = prep_of(opts)
        self.inputs = None
        if self.candidate:
            if slides(case)[3] != 1:
                raise ValueError(
                    f"{case.case_id}: the preprocessing row is not defined on a "
                    "reduced proxy"
                )
            self.inputs = prep.input_paths(self.dir / "prep", self.candidate)
        self.ns = namespace(case, self.dir, opts, self.inputs)
        self.argv = stage_commands(self.ns)

    def command(self, stage, row=None):
        if stage == stages.PREP:
            return [
                "regbench.prep.write_inputs",
                self.candidate,
                *map(str, self.inputs),
            ]
        if stage == stages.TILE:
            r = self.plan()[row]
            return cli(stage, self.argv[stage](row, int(r["ix"]), int(r["iy"])))
        return cli(stage, self.argv[stage])

    def plan(self):
        from stare.stages.reg_tile import plan_rows

        return plan_rows(self.argv["plan"])

    def record(self, stage, row=None):
        return stages.read_record(self.dir, stage, row)

    def done(self, stage, row=None):
        """The job's record when it is ok FOR THIS COMMAND, else None."""
        rec = self.record(stage, row)
        if rec and rec.get("ok") and rec.get("command") == self.command(stage, row):
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


def _run_prep(rnd, attempt, execution):
    """The benchmark's preprocessing of both images (the `stare_prep` row), with a
    record like a stage job's."""
    path = stages.record_path(rnd.dir, stages.PREP)
    command = rnd.command(stages.PREP)
    host = resources.host_info()
    base = {
        "stage": stages.PREP,
        "row": None,
        "attempt": int(attempt),
        "command": command,
        "execution": execution,
        "launch_id": host["launch_id"],
        "log_path": host["log_path"],
        "submission": host["submission"],
        "started": time.time(),
    }
    stages._write(
        path, {**base, "ok": False, "exit": None, "cause": stages.KILLED, "error": ""}
    )
    try:
        made = prep.write_inputs(
            rnd.case, rnd.dir / "prep", rnd.candidate, slides(rnd.case)[2]
        )
    except (Exception, SystemExit) as exc:
        rec = {
            **base,
            "ok": False,
            "exit": 137 if isinstance(exc, MemoryError) else 1,
            "cause": resources.classify_error(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "ended": time.time(),
        }
        stages._write(path, rec)
        return rec
    rec = {
        **base,
        "ok": True,
        "exit": 0,
        "cause": "",
        "error": "",
        "ended": time.time(),
        "made": made["record"],
    }
    stages._write(path, rec)
    return rec


def run_stage(
    case, stage_dir, opts, stage, row=None, attempt=1, execution="jobs", force=False
):
    """One stage job of one round: ``prep`` (the preprocessing row only), ``coarse``,
    ``tile`` (one plan row; with ``row=None`` every row that is due, one after another)
    or ``solve``. Returns ``[(stage, row, ran?, why, record)]``.

    A job runs only when the stage before it is ok for this command, and when
    ``stages.decide`` says this attempt is due; otherwise it leaves everything as it
    is. STITCH is not a job of this kind: it is the writer (``write_native``)."""
    rnd = Round(case, stage_dir, opts)
    submission = resources.host_info()["submission"]
    out = []

    def one(stage, row=None):
        rec = rnd.record(stage, row)
        command = rnd.command(stage, row)
        go, why = stages.decide(rec, attempt, command, submission, force)
        if go and stage == stages.PREP:
            rec = _run_prep(rnd, attempt, execution)
        elif go:
            rec = stages.run_command(rnd.dir, stage, command, row, attempt, execution)
        out.append((stage, row, go, why, rec))

    if stage == stages.PREP:
        if rnd.candidate:
            one(stage)
        return out
    if rnd.candidate and not rnd.done(stages.PREP):
        return [(stage, row, False, "the preprocessing is not ok", None)]
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
    raise ValueError(f"stage {stage!r}: one of prep, coarse, tile, solve")


def _failed(rnd, stage, row, jobs):
    """The error of a round whose job ``stage`` did not end ok for this command."""
    rec = rnd.record(stage, row)
    what = stage.upper() + (f" (tile row {row})" if row is not None else "")
    mine = (
        rec is not None
        and rec.get("command") == rnd.command(stage, row)
        and ladder.mine(rec, resources.host_info()["submission"])
    )
    if not mine:
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


def _local(rnd, workers, mon):
    """The four stage commands inside ONE run (the selfcheck, a laptop): the same
    commands and the same records as the cohort's jobs, the tiles ``workers`` at a
    time, each command once."""
    from concurrent.futures import ThreadPoolExecutor

    def go(stage, row=None):
        if stage == stages.PREP:
            rec = _run_prep(rnd, 1, "local")
        else:
            rec = stages.run_command(
                rnd.dir, stage, rnd.command(stage, row), row, 1, "local"
            )
        if not rec.get("ok"):
            raise _failed(rnd, stage, row, False)

    rnd.dir.mkdir(parents=True, exist_ok=True)
    if rnd.candidate:
        mon.pause()  # benchmark-side work, not STARE's
        try:
            go(stages.PREP)
        finally:
            mon.resume()
    go(stages.COARSE)
    rows = range(len(rnd.plan()))
    with ThreadPoolExecutor(max(1, int(workers))) as pool:
        list(pool.map(lambda i: go(stages.TILE, i), rows))
    _, why = rnd.tiles_done()
    if why:
        raise stages.StageFailed(f"SOLVE was not run: {why}")
    go(stages.SOLVE)


def _collect(rnd):
    """The cohort: every stage ran in a job of its own; nothing is run here. Raises
    for the first stage, in order, that is not ok for this round's command."""
    if rnd.candidate and not rnd.done(stages.PREP):
        raise _failed(rnd, stages.PREP, None, True)
    if not rnd.done(stages.COARSE):
        raise _failed(rnd, stages.COARSE, None, True)
    for i in range(len(rnd.plan())):
        if not rnd.done(stages.TILE, i):
            raise _failed(rnd, stages.TILE, i, True)
    _, why = rnd.tiles_done()
    if why:
        raise stages.StageFailed(f"SOLVE's inputs are not complete: {why}")
    if not rnd.done(stages.SOLVE):
        raise _failed(rnd, stages.SOLVE, None, True)


def register(case, work, opts):
    import stare
    from stare.stage_warp import make_warper

    jobs = os.environ.get(JOBS_ENV) == "jobs"
    where = opts.get("native_dir")
    if jobs:
        if where is None:
            raise ValueError(f"{JOBS_ENV}=jobs needs the driver (its results folder)")
        stage_dir = Path(where(case.case_id, VARIANTS[-1])).parent / "stages"
    else:
        stage_dir = Path(work) / "stages"
    rnd = Round(case, stage_dir, opts)
    ns, candidate = rnd.ns, rnd.candidate
    mon = opts.get("monitor") or _NullMonitor()
    if jobs:
        _collect(rnd)
    else:
        _local(rnd, ns.workers, mon)
    made = None
    if candidate:
        made = {
            "reference": str(rnd.inputs[0]),
            "moving": str(rnd.inputs[1]),
            "record": rnd.record(stages.PREP)["made"],
        }
    cost = stages.summarise(
        rnd.record(stages.COARSE),
        [rnd.record(stages.TILE, i) for i in range(len(rnd.plan()))],
        rnd.record(stages.SOLVE),
    )
    cost["execution"] = "jobs" if jobs else "local"
    man = json.loads(Path(ns.manifest).read_text())
    warp = make_warper(man)
    entry = man["slides"][MOV_NAME]
    changed = {
        k: str(getattr(ns, k))
        for k in _FLAGS
        if k in opts and getattr(ns, k) != TIER_HIGH_STRIDE64.get(k)
    }
    ref_in, mov_in, _, reduce = slides(case)
    plain_mov = mov_in  # what the plain row registers on, and what STITCH draws
    if made:
        ref_in, mov_in = made["reference"], made["moving"]
    proxy = case.modality == "brightfield"
    info = {
        "version": stare.__version__,
        "source": str(Path(stare.__file__).parent),
        # the anchor, moving px -> reference px: the whole of `stare_rigid`, and the same
        # matrix at every stride (module docstring). In the frame STARE worked in: for
        # a reduced proxy, the reduced one (`input.reduced_by`)
        "m0": entry["M0"],
        "input": {
            "reference": ref_in,
            "moving": mov_in,
            "kind": (
                f"benchmark-side preprocessing `{candidate}` of the image the plain "
                "row registers on"
                if made
                else "one-channel proxy of an RGB slide"
                if proxy
                else "original slides"
            ),
            "proxy": (case.extra.get("settings") or {}).get("proxy") if proxy else None,
            "reduced_by": reduce,
        },
        **(
            {
                "prep": {
                    **made["record"],
                    "frozen": prep.load_frozen()["status"]
                    if str(opts.get("label") or "") == prep.ROW
                    else "a tuning run (not the frozen row)",
                    "frozen_sha256": prep.load_frozen()["sha256"],
                }
            }
            if made
            else {}
        ),
        "workers": int(ns.workers),
        "cpu_count": os.cpu_count(),  # what COARSE's FFTs use, whatever the allocation
        "stride": int(ns.stride),
        "max_disp": float(ns.halo if ns.max_disp is None else ns.max_disp),
        "model": ns.model,
        "nuclear_index": int(ns.nuclear_index),
        "pixel_size": ns.pixel_size,
        "channel_names": list(ns.channel_names or []),
        # the round's cost over its jobs (regbench.stages.summarise)
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
                    else "the stage commands run inside ONE job, the tiles "
                    f"{int(ns.workers)} at a time: not the cohort's separate jobs, "
                    "so this run's cost is this job's"
                ),
                *(
                    [
                        "brightfield: STARE is given the case's one-channel proxy of "
                        f"each RGB slide ({info_proxy(case)}), written by the dataset "
                        "module before the clock starts; the image it writes is the "
                        "registered proxy"
                    ]
                    if proxy
                    else []
                ),
                *(
                    [
                        "no pixel size is established for this case: STITCH is "
                        f"given {NOMINAL_PIXEL_SIZE} as a placeholder for the header "
                        "of the image it writes; nothing else reads it"
                    ]
                    if proxy and not case.pixel_size_um
                    else []
                ),
                *(
                    [
                        f"preprocessing `{candidate}` (regbench.prep; here: "
                        f"{'identity' if made['record']['identity_here'] else made['record']['base'] + ', ' + made['record']['operation']}): "
                        "both images rewritten as one-channel files before COARSE "
                        f"(prep_s = {made['record']['prep_s']:.1f}); STARE and "
                        "its parameters are unchanged; STITCH draws the image the "
                        "plain row draws, with this row's transform"
                    ]
                    if made
                    else []
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
    rigid = reduced_map(lambda xy: warp(MOV_NAME, xy, "rigid"), reduce)
    yield "stare_rigid", PointMap(rigid, MOV_TO_REF), info
    if Path(ns.tre).exists():
        info = {**info, "solve": json.loads(Path(ns.tre).read_text()).get("solve", {})}
    state = {
        "stage_dir": str(rnd.dir),
        # the preprocessing row: never the preprocessed pixels
        "stitch": rnd.argv[stages.STITCH],
        "moving": plain_mov if made else ns.moving,
        "execution": cost["execution"],
        "out_shape": entry["out_shape"],
        "reduced_by": reduce,
        "proxy": proxy,
    }
    refined = reduced_map(lambda xy: warp(MOV_NAME, xy, "refined"), reduce)
    yield "stare", PointMap(refined, MOV_TO_REF), {**info, "_state": state}


def write_native(state, case_id, dest_dir):
    """STARE's own STITCH, as a process of its own, on the manifest SOLVE wrote."""
    from stare.slide_io import open_lazy

    out = Path(dest_dir) / "registered.ome.tif"
    attempt = resources.host_info()["attempt"] if state["execution"] == "jobs" else 1
    rec = stages.run_command(
        state["stage_dir"],
        stages.STITCH,
        cli(stages.STITCH, state["stitch"](state["moving"], out)),
        attempt=min(attempt, stages.ATTEMPTS),
        execution=state["execution"],
    )
    if not rec["ok"]:
        text = f"stare stitch exited {rec['exit']}: {str(rec['error'])[-600:]}"
        raise MemoryError(text) if rec["cause"] == "oom" else RuntimeError(text)
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
        # the writer's own job: allocated CPUs, its CPU time and peak memory
        "stitch_job": {
            k: rec[k] for k in ("cpus", "mem_gb", "wall_s", "cpu_s", "peak_rss_mb")
        },
    }
    if state.get("proxy"):  # a brightfield case: not the slide, its registered proxy
        native["content"] = "the one-channel proxy STARE was given, registered"
        native["reduced_by"] = int(state.get("reduced_by", 1))
    return native

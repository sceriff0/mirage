"""STARE, through ``stare register``'s own code path, with the pipeline's parameters.

``stare_rigid`` is the coarse anchor ``M0`` alone and ``stare`` the full transform (``M0``
plus the mesh). Both come from one registration; the manifest reproduces each stage, so the
rigid one costs nothing extra. Both maps are moving -> reference
(``stare.stage_warp.make_warper``), STARE's own direction.

**Inputs.** The slides as the case has them (``case.ref_image``, ``case.mov_image``), every
channel, read by STARE's own reader. ``--nuclear-index`` is ``case.extra["mov_channel"]``
(0 when the case names none). STARE takes ONE index for both slides, so a case whose
reference has its nuclei at another index is refused.

**A brightfield case** (HyReCo: ``case.modality == "brightfield"``). ``stare register``
reads ONE channel per slide, so an RGB slide cannot be handed over as it is. The case
carries a one-channel proxy of each slide, written by its dataset module
(``case.ref_nuclear`` / ``case.mov_nuclear``; ``datasets/hyreco.py``: inverted luminance
by default), and that is what STARE is given, at index 0. This is benchmark-side
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

**Parameters.** ``TIER_HIGH_STRIDE64``: the ``high`` row of the STARE table in
``lib/RegPresets.groovy`` plus stride 64, i.e. the pipeline's ``tiled_high_s64`` arm. All
five are passed explicitly. Every other TRANSFORM parameter of that arm is the command's
default (maximum displacement = halo, model, field step). Two STITCH flags are not, and the
pipeline always passes them (``modules/local/tiled_stitch.nf``): ``--pixel-size`` (the
case's) and ``--channel-names`` (``case.extra["channel_names"]``, else the names in the
moving slide's own header). ``--workers`` is the allocation (``resources.n_cpus()``);
COARSE's FFTs use every core of the node whatever it is, so ``os.cpu_count()`` is recorded
beside it.

**Stride.** ``--opt stride=<64|128|256>``, nothing else: 64 when unset. The stride is the
lattice REG-TILE measures on and SOLVE solves; COARSE never sees it (``stare/cli.py:79-100``
passes it no ``--stride``, ``stare/stages/coarse.py:170-211`` has no such argument), SOLVE
writes the anchor into the manifest as COARSE gave it (``stare/stages/solve.py:77-89``) and
the rigid stage applies that matrix alone (``stare/stage_warp.py:85-87``); stare-registration
1.2.1, installed package ``[PARTIAL: cli.py and stage_warp.py in full, main() of
stages/solve.py and stages/stitch.py, manifest.slide_entry; stages/coarse.py and
coarse_align.py only searched for "stride" (no match) and for the argument list]``. A test
finds the anchor identical to the last bit at stride 64 and 256. So ``stare_rigid`` is the same map at every stride, and a stride
run adds one new result, its final variant. (Its rigid row's COST still differs: both
variants are yielded after SOLVE.)

**Output.** STITCH is deferred, not dropped: while ``stare register`` runs,
``stare.stages.stitch.main`` is replaced by a stub that records its arguments, so the clock
stops when the manifest exists; ``write_native`` then calls the real ``stitch.main`` with
those arguments (only ``--out`` is moved). The image is on the reference grid
(``out_shape`` = the reference's, ``stare/stages/solve.py:91``): no ``origin_xy``. Nothing
is padded; where the moving slide has no pixels the image is 0. ``stare_rigid`` has no
image of its own: ``stare register`` writes one image.

Other options (``--opt key=value``), for tests and small slides, each recorded in
``info["implementation"]["changed_from_tier"]``: ``tile``, ``halo``, ``out_tile``,
``max_dim``, ``model``, ``max_disp``.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np

from .. import imageio
from ..mapping import MOV_TO_REF, PointMap

SCOPE = "pair"
VARIANTS = ("stare_rigid", "stare")
REF_NAME, MOV_NAME = "reference", "moving"
STRIDES = (64, 128, 256)
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


def command(case, work, opts):
    """The ``stare`` command line of one case: ``stare.cli.main(command(...))`` is the run."""
    work = Path(work)
    values = {**TIER_HIGH_STRIDE64, "stride": stride_of(opts)}
    values.update({k: opts[k] for k in _FLAGS if k in opts and k != "stride"})
    ref, mov, index, reduce = slides(case)
    argv = [
        "register",
        "--reference",
        ref,
        "--moving",
        mov,
        "--out",
        str(work / "registered.ome.tif"),
        "--manifest",
        str(work / "manifest.json"),
        "--tre",
        str(work / "tre.json"),
        "--workdir",
        str(work / "stages"),
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


def register(case, work, opts):
    import stare
    import stare.cli
    import stare.stages.stitch as stitch_stage
    from stare.stage_warp import make_warper

    work = Path(work)
    manifest, tre = work / "manifest.json", work / "tre.json"
    ns = _parse(command(case, work, opts))
    stitch_argv = []
    real = stitch_stage.main
    # looked up at call time by `stare register`, so this is what it calls
    stitch_stage.main = lambda argv: stitch_argv.append(list(argv)) or 0
    try:
        rc = stare.cli.register(ns)
    finally:
        stitch_stage.main = real
    if rc:
        raise RuntimeError(f"stare register exited {rc}")
    man = json.loads(manifest.read_text())
    warp = make_warper(man)
    entry = man["slides"][MOV_NAME]
    changed = {
        k: str(getattr(ns, k))
        for k in _FLAGS
        if k in opts and getattr(ns, k) != TIER_HIGH_STRIDE64.get(k)
    }
    ref_in, mov_in, _, reduce = slides(case)
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
            "kind": "one-channel proxy of an RGB slide" if proxy else "original slides",
            "proxy": (case.extra.get("settings") or {}).get("proxy") if proxy else None,
            "reduced_by": reduce,
        },
        "workers": int(ns.workers),
        "cpu_count": os.cpu_count(),  # what COARSE's FFTs use, whatever --workers is
        "stride": int(ns.stride),
        "max_disp": float(ns.halo if ns.max_disp is None else ns.max_disp),
        "model": ns.model,
        "nuclear_index": int(ns.nuclear_index),
        "pixel_size": ns.pixel_size,
        "channel_names": list(ns.channel_names or []),
        "implementation": {
            "package": "stare-registration",
            "source": str(Path(stare.__file__).parent),
            "registration": "stare.cli.register (the `stare register` command's function)",
            "parameters": "mirage's `high` STARE tier (lib/RegPresets.groovy), stride "
            f"{TIER_HIGH_STRIDE64['stride']}: {TIER_HIGH_STRIDE64}",
            "changed_from_tier": changed,
            "point_warp": "stare.stage_warp.make_warper",
            "benchmark_side": [
                "STITCH deferred until the clock has stopped: stare.stages.stitch.main "
                "is called afterwards with the arguments `stare register` gave it",
                "one process with --workers = the allocation, not the pipeline's four "
                "cluster stages; COARSE's FFTs use every core of the node regardless",
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
            ],
        },
    }
    m0 = work / "stages" / f"{MOV_NAME}_m0.json"
    if m0.exists():
        info["coarse_trusted"] = json.loads(m0.read_text()).get("coarse_trusted")
    rigid = reduced_map(lambda xy: warp(MOV_NAME, xy, "rigid"), reduce)
    yield "stare_rigid", PointMap(rigid, MOV_TO_REF), info
    if tre.exists():
        info = {**info, "solve": json.loads(tre.read_text()).get("solve", {})}
    if stitch_argv:
        state = {
            "stitch_argv": stitch_argv[-1],
            "out_shape": entry["out_shape"],
            "reduced_by": reduce,
            "proxy": proxy,
        }
        info = {**info, "_state": state}
    refined = reduced_map(lambda xy: warp(MOV_NAME, xy, "refined"), reduce)
    yield "stare", PointMap(refined, MOV_TO_REF), info


def write_native(state, case_id, dest_dir):
    """STARE's own STITCH, with the arguments ``stare register`` handed it."""
    import stare.stages.stitch as stitch_stage
    from stare.slide_io import open_lazy

    argv = list(state["stitch_argv"])
    out = Path(dest_dir) / "registered.ome.tif"
    argv[argv.index("--out") + 1] = str(out)
    rc = stitch_stage.main(argv)
    if rc:
        raise RuntimeError(f"stare stitch exited {rc}")
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
    }
    if state.get("proxy"):  # a brightfield case: not the slide, its registered proxy
        native["content"] = "the one-channel proxy STARE was given, registered"
        native["reduced_by"] = int(state.get("reduced_by", 1))
    return native

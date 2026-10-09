"""STARE, through ``stare register``'s own code path.

``stare_rigid`` is the coarse anchor ``M0`` alone and ``stare`` the full transform (``M0``
plus the mesh). Both come from one registration; the manifest reproduces each stage, so the
rigid one costs nothing extra. STITCH is deferred, not dropped: while ``stare register``
runs, ``stare.stages.stitch.main`` is replaced by a stub that records its arguments, so the
clock stops when the manifest exists; ``write_native`` then calls the real ``stitch.main``
with those arguments (only ``--out`` is moved). ``stare_rigid`` has no image of its own:
``stare register`` writes one image.

Both maps are moving -> reference (``stare.stage_warp.make_warper``), STARE's own direction.
``--workers`` is the allocation (``resources.n_cpus()``).

Options (``--opt key=value``), each defaulting to ``stare register``'s own default:
``tile``, ``halo``, ``stride``, ``max_dim``, ``model``, ``max_disp``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..mapping import MOV_TO_REF, PointMap

SCOPE = "pair"
VARIANTS = ("stare_rigid", "stare")
REF_NAME, MOV_NAME = "reference", "moving"
_FLAGS = {
    "tile": "--tile",
    "halo": "--halo",
    "stride": "--stride",
    "max_dim": "--max-dim",
    "model": "--model",
    "max_disp": "--max-disp",
}


def register(case, work, opts):
    import stare
    import stare.stages.stitch as stitch_stage
    from stare.cli import build_parser
    from stare.cli import register as stare_register
    from stare.stage_warp import make_warper

    work = Path(work)
    manifest, tre = work / "manifest.json", work / "tre.json"
    argv = [
        "register",
        "--reference",
        case.ref_nuclear,
        "--moving",
        case.mov_nuclear,
        "--out",
        str(work / "registered.ome.tif"),
        "--manifest",
        str(manifest),
        "--tre",
        str(tre),
        "--workdir",
        str(work / "stages"),
        "--workers",
        str(opts.get("workers", 1)),
        "--reference-name",
        REF_NAME,
        "--moving-name",
        MOV_NAME,
    ]
    for key, flag in _FLAGS.items():
        if key in opts:
            argv += [flag, str(opts[key])]
    ns = build_parser().parse_args(argv)
    stitch_argv = []
    real = stitch_stage.main
    # looked up at call time by `stare register`, so this is what it calls
    stitch_stage.main = lambda argv: stitch_argv.append(list(argv)) or 0
    try:
        rc = stare_register(ns)
    finally:
        stitch_stage.main = real
    if rc:
        raise RuntimeError(f"stare register exited {rc}")
    man = json.loads(manifest.read_text())
    warp = make_warper(man)
    info = {
        "version": stare.__version__,
        "source": str(Path(stare.__file__).parent),
        "implementation": {
            "package": "stare-registration",
            "source": str(Path(stare.__file__).parent),
            "registration": "stare.cli.register (the `stare register` command's function)",
            "parameters": "the command's defaults",
            "changed_from_defaults": {k: str(opts[k]) for k in _FLAGS if k in opts},
            "point_warp": "stare.stage_warp.make_warper",
            "benchmark_side": [
                "STITCH deferred until the clock has stopped: stare.stages.stitch.main "
                "is called afterwards with the arguments `stare register` gave it"
            ],
        },
    }
    m0 = work / "stages" / f"{MOV_NAME}_m0.json"
    if m0.exists():
        info["coarse_trusted"] = json.loads(m0.read_text()).get("coarse_trusted")
    rigid = PointMap(lambda xy: warp(MOV_NAME, xy, "rigid"), MOV_TO_REF)
    yield "stare_rigid", rigid, info
    if tre.exists():
        info = {**info, "solve": json.loads(tre.read_text()).get("solve", {})}
    if stitch_argv:
        info = {**info, "_state": {"stitch_argv": stitch_argv[-1]}}
    yield "stare", PointMap(lambda xy: warp(MOV_NAME, xy, "refined"), MOV_TO_REF), info


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
        channels = int(arr.shape[0])
    finally:
        close()
    return {
        "files": [str(out)],
        "format": "ome-tiff",
        "channels": channels,
        "dtype": str(np.dtype(dtype)),
        "writer": "stare.stages.stitch.main",
    }

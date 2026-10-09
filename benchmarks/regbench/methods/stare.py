"""STARE, through ``stare register``'s own code path.

``stare_rigid`` is the coarse anchor ``M0`` alone and ``stare`` the full transform (``M0``
plus the mesh). Both come from one registration; the manifest reproduces each stage, so the
rigid one costs nothing extra. STITCH is skipped: the benchmark scores points, and the
manifest is the transform.

Options (``--opt key=value``), each defaulting to ``stare register``'s own default:
``tile``, ``halo``, ``stride``, ``max_dim``, ``model``, ``max_disp``.
"""

from __future__ import annotations

import json
from pathlib import Path

VARIANTS = ("stare_rigid", "stare")
REF_NAME, MOV_NAME = "reference", "moving"
_FLAGS = {"tile": "--tile", "halo": "--halo", "stride": "--stride", "max_dim": "--max-dim",
          "model": "--model", "max_disp": "--max-disp"}


def register(case, work, opts):
    import stare
    import stare.stages.stitch as stitch_stage
    from stare.cli import build_parser
    from stare.cli import register as stare_register
    from stare.stage_warp import make_warper

    work = Path(work)
    manifest, tre = work / "manifest.json", work / "tre.json"
    argv = ["register", "--reference", case.ref_nuclear, "--moving", case.mov_nuclear,
            "--out", str(work / "registered.ome.tif"), "--manifest", str(manifest),
            "--tre", str(tre), "--workdir", str(work / "stages"),
            "--workers", str(opts.get("workers", 1)),
            "--reference-name", REF_NAME, "--moving-name", MOV_NAME]
    for key, flag in _FLAGS.items():
        if key in opts:
            argv += [flag, str(opts[key])]
    ns = build_parser().parse_args(argv)
    real = stitch_stage.main
    stitch_stage.main = lambda argv: 0
    try:
        rc = stare_register(ns)
    finally:
        stitch_stage.main = real
    if rc:
        raise RuntimeError(f"stare register exited {rc}")
    man = json.loads(manifest.read_text())
    warp = make_warper(man)
    info = {"version": stare.__version__, "source": str(Path(stare.__file__).parent),
            "implementation": {
                "package": "stare-registration", "source": str(Path(stare.__file__).parent),
                "registration": "stare.cli.register (the `stare register` command's function)",
                "parameters": "the command's defaults",
                "changed_from_defaults": {k: str(opts[k]) for k in _FLAGS if k in opts},
                "point_warp": "stare.stage_warp.make_warper",
                "benchmark_side": ["STITCH skipped: points are scored, the manifest is the transform"]}}
    m0 = work / "stages" / f"{MOV_NAME}_m0.json"
    if m0.exists():
        info["coarse_trusted"] = json.loads(m0.read_text()).get("coarse_trusted")
    yield "stare_rigid", lambda xy: warp(MOV_NAME, xy, "rigid"), info
    if tre.exists():
        info = {**info, "solve": json.loads(tre.read_text()).get("solve", {})}
    yield "stare", lambda xy: warp(MOV_NAME, xy, "refined"), info

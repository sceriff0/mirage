"""The ``stare`` command: the four stages as subcommands, plus ``register``.

``stare coarse``, ``stare reg-tile``, ``stare solve`` and ``stare stitch`` pass
their argv straight to the stage ``main`` functions in ``stare.stages`` -- the
same functions the mirage pipeline's ``bin/tiled_*.py`` shims call, one Nextflow
task each. ``stare register`` runs all four in one process: the coarse anchor,
then the tile plan mapped over a local ``multiprocessing`` pool (``--workers``),
then the solve, then the stitch. It writes the same ``*_manifest.json`` and
``*_registered.ome.tif`` the pipeline publishes, because it runs the same stage
functions over the same tile rows; mirage's ``tests/test_stare_package_parity.py``
asserts the two paths agree to the byte.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

from stare import __version__

# The package's own default for the anchor thumbnail bound. The stage itself has
# NO default (the pipeline resolves it from its preset tiers and passes it
# explicitly); this is the value `stare register` uses when the caller does not
# say, and it equals the pipeline's `high` STARE tier.
DEFAULT_MAX_DIM = 2048


def _slide_name(path) -> str:
    """The slide name a file gets in the manifest: the basename before its first dot.

    Matches what the mirage pipeline uses (``meta.id`` = Nextflow's ``simpleName``),
    so ``mov.ome.tif`` is ``mov`` on both paths.
    """
    return Path(path).name.split(".", 1)[0]


def _run_tile(argv) -> str:
    """Pool worker: one reg-tile stage over one plan row; returns the control JSON path."""
    from stare.stages import reg_tile

    rc = reg_tile.main(argv)
    if rc:
        raise RuntimeError(f"reg-tile exited {rc} for {argv}")
    return argv[argv.index("--out") + 1]


def register(a) -> int:
    """``stare register``: coarse -> reg-tile x N (pool) -> solve -> stitch, one process.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed ``register`` arguments; see ``build_parser``.

    Returns
    -------
    int
        0 on success; a stage's non-zero return code otherwise.
    """
    from stare.stages import coarse, reg_tile, solve, stitch

    ref_name = a.reference_name or _slide_name(a.reference)
    mov_name = a.moving_name or _slide_name(a.moving)

    tmp = None
    if a.workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="stare-")
        work = Path(tmp.name)
    else:
        work = Path(a.workdir)
        work.mkdir(parents=True, exist_ok=True)
    try:
        m0_f = work / f"{mov_name}_m0.json"
        tiles_f = work / f"{mov_name}_tiles.csv"
        rc = coarse.main(
            [
                "--reference",
                str(a.reference),
                "--moving",
                str(a.moving),
                "--nuclear-index",
                str(a.nuclear_index),
                "--tile",
                str(a.tile),
                "--halo",
                str(a.halo),
                "--max-dim",
                str(a.max_dim),
                "--model",
                a.model,
                "--out-m0",
                str(m0_f),
                "--out-tiles",
                str(tiles_f),
            ]
        )
        if rc:
            return rc

        # THE FAN-OUT. One reg-tile invocation per plan row, addressed by --plan/--row
        # exactly as a SLURM array job would, over a local pool. The rows, the function
        # and the per-tile arguments are the ones the pipeline's TILED_REG_TILE renders,
        # which is why the manifest comes out identical.
        n_rows = len(reg_tile.plan_rows(tiles_f))
        jobs = []
        for row in range(n_rows):
            r = reg_tile.plan_row(tiles_f, row)
            out = work / f"{mov_name}_{r['ix']}_{r['iy']}_ctrl.json"
            jobs.append(
                [
                    "--reference",
                    str(a.reference),
                    "--moving",
                    str(a.moving),
                    "--m0",
                    str(m0_f),
                    "--nuclear-index",
                    str(a.nuclear_index),
                    "--plan",
                    str(tiles_f),
                    "--row",
                    str(row),
                    "--upsample",
                    str(a.upsample),
                    "--out",
                    str(out),
                ]
            )
        if a.workers <= 1:
            for argv in jobs:
                _run_tile(argv)
        else:
            with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
                list(pool.imap_unordered(_run_tile, jobs))

        solve_argv = [
            "--m0",
            str(m0_f),
            "--controls",
            str(work / f"{mov_name}_*_ctrl.json"),
            "--gate-tre",
            str(a.gate_tre),
            "--max-error",
            str(a.max_error),
            "--solver",
            a.solver,
            "--reference-name",
            ref_name,
            "--moving-name",
            mov_name,
            "--out-manifest",
            str(a.manifest),
        ]
        # --max-disp None means "no range bound" at the stage; the pipeline passes the
        # halo, and so does this command unless told otherwise.
        max_disp = a.halo if a.max_disp is None else a.max_disp
        solve_argv += ["--max-disp", str(max_disp)]
        if a.tre:
            solve_argv += ["--out-tre", str(a.tre)]
        rc = solve.main(solve_argv)
        if rc:
            return rc

        stitch_argv = [
            "--moving",
            str(a.moving),
            "--manifest",
            str(a.manifest),
            "--moving-name",
            mov_name,
            "--out",
            str(a.out),
            "--out-tile",
            str(a.out_tile),
            "--pixel-size",
            a.pixel_size,
        ]
        if a.channel_names:
            stitch_argv += ["--channel-names", *a.channel_names]
        return stitch.main(stitch_argv)
    finally:
        if tmp is not None:
            tmp.cleanup()


def build_parser() -> argparse.ArgumentParser:
    """The ``stare`` argument parser: four passthrough stages and ``register``."""
    ap = argparse.ArgumentParser(
        prog="stare",
        description="STARE: tile-parallel, JVM-free, non-rigid registration of "
        "whole-slide images.",
    )
    ap.add_argument("--version", action="version", version=f"stare {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    for name, help_ in (
        ("coarse", "global rigid anchor M0 + the tile plan (stage 1/4)"),
        ("reg-tile", "one tile's residual displacement (stage 2/4, the fan-out)"),
        ("solve", "control points -> transform manifest (stage 3/4)"),
        ("stitch", "stream the moving slide through the manifest (stage 4/4)"),
    ):
        # No arguments of its own: everything after the stage name is the stage's
        # argv (`parse_known_args` in `main` hands it over untouched, `--help` included).
        sub.add_parser(name, help=help_, add_help=False)

    r = sub.add_parser(
        "register",
        help="all four stages in one process, the tile stage over a local worker pool",
    )
    r.add_argument("--reference", required=True)
    r.add_argument("--moving", required=True)
    r.add_argument("--out", required=True, help="registered OME-TIFF to write")
    r.add_argument("--manifest", required=True, help="transform manifest JSON to write")
    r.add_argument("--tre", default=None, help="optional intrinsic-TRE report JSON")
    r.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    r.add_argument(
        "--workdir",
        default=None,
        help="where the M0 JSON, tile plan and per-tile control JSONs go "
        "(default: a temporary directory removed on exit)",
    )
    r.add_argument("--reference-name", default=None)
    r.add_argument("--moving-name", default=None)
    # coarse
    r.add_argument("--nuclear-index", type=int, default=0)
    r.add_argument("--tile", type=int, default=2048)
    r.add_argument("--halo", type=int, default=256)
    r.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    r.add_argument(
        "--model", default="euclidean", choices=["euclidean", "similarity", "affine"]
    )
    # reg-tile
    r.add_argument("--upsample", type=int, default=10)
    # solve
    r.add_argument("--gate-tre", type=float, default=1.0)
    r.add_argument("--max-error", type=float, default=0.99)
    r.add_argument(
        "--max-disp", type=float, default=None, help="default: --halo, as the pipeline"
    )
    r.add_argument("--solver", choices=["legacy", "robust"], default="robust")
    # stitch
    r.add_argument("--out-tile", type=int, default=1024)
    r.add_argument("--pixel-size", default="auto")
    r.add_argument("--channel-names", nargs="*", default=None)
    return ap


def main(argv=None) -> int:
    """Console entry point.

    Returns
    -------
    int
        The stage's (or ``register``'s) return code.
    """
    ap = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    a, stage_argv = ap.parse_known_args(argv)
    if a.command == "register":
        if stage_argv:
            ap.error(f"unrecognized arguments: {' '.join(stage_argv)}")
        return register(a)
    from stare.stages import coarse, reg_tile, solve, stitch

    stage = {
        "coarse": coarse,
        "reg-tile": reg_tile,
        "solve": solve,
        "stitch": stitch,
    }[a.command]
    if stage_argv[:1] == ["--"]:
        stage_argv = stage_argv[1:]
    return stage.main(stage_argv)


if __name__ == "__main__":
    raise SystemExit(main())

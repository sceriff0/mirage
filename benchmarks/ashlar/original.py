#!/usr/bin/env python3
"""ASHLAR as published: its own command, its own stitching, its own mosaic.

``solve.py`` drives ASHLAR's cross-cycle class alone, places the reference tiles by fiat and
leaves the image to the pipeline's stitcher. That answers "what does ASHLAR's cycle
alignment do on our slides" and is not what a reader means by "ASHLAR". This module runs
the ORIGINAL program, ``ashlar.scripts.ashlar.main`` -- the function behind the ``ashlar``
command -- unmodified, on one patient's cycles:

    ashlar 'filepattern|<ref tiles>|...' 'filepattern|<cycle 1 tiles>|...' ... \\
        -o ashlar_output.ome.tif -c <nuclear channel> -m <maximum shift>

so the reference cycle is stitched by ``EdgeAligner`` (edge registration, permutation
threshold, spanning tree, linear model), every later cycle is aligned by ``LayerAligner``
against it, and the registered image is the pyramid ``PyramidWriter`` assembles from
``Mosaic`` with ASHLAR's own blending. Nothing of ASHLAR's is patched or replaced.

WHAT IS NOT ASHLAR'S, and cannot be: the INPUT. The slides exist only stitched, so
``retile.py`` synthesises the raw tiles ASHLAR needs (a regular grid with a stage error and
per-tile sensor noise, see there), and they are read with ASHLAR's own ``filepattern``
reader, selected with its documented ``reader|path|key=value`` syntax.

TWO THINGS ARE ADDED AROUND IT, neither changing what it computes:

1. A RECORD of where it put the tiles. The command writes an image and keeps no transform,
   and the benchmark scores every method by carrying the SAME nuclei through its transform.
   So ``EdgeAligner.run`` / ``LayerAligner.run`` are wrapped to note the aligner after it
   has run (positions, discarded tiles, errors) -- the original method is called first and
   its result returned untouched.

2. A MANIFEST per moving cycle, for ``bin/warp_seg_qc.py --method tiled``. ASHLAR moves
   each tile rigidly, so the field is piecewise constant: every tile carries its own
   displacement over the part of it no neighbour covers, and the displacement ramps
   linearly across the overlap band, where ASHLAR's mosaic blends the two tiles
   (:func:`piecewise_mesh`). A point's displacement is

       D(t) = (L[t] - c_mov[t]) - (E[s] - c_ref[s]),   s = reference_idx[t]

   with ``L`` / ``E`` the positions ASHLAR gave the moving / reference tiles in its mosaic
   and ``c`` where each tile's pixels really came from in its slide (``grid.json``'s
   ``true_positions_yx``). The reference term is ASHLAR's own stitching of the reference,
   which is part of what its output looks like and is therefore part of its score.

Optionally the mosaic is also split into one OME-TIFF per cycle with channel names
(:func:`split_mosaic`): a repackaging of ASHLAR's pixels for the figure tools, which read
one slide per file.

Runs inside ``labsyspharm/ashlar:1.20.0`` (ashlar, numpy, tifffile); the manifest assembly
is the pipeline's (``bin/utils/tiled_manifest.py``), as in ``solve.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bin" / "utils"))

from tiled_manifest import build_manifest, slide_entry  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_NAME = "ashlar_output.ome.tif"
SPLIT_TILE = 512


# --------------------------------------------------------------------- the command --
def reader_spec(tile_dir, grid) -> str:
    """ASHLAR's ``reader|path|key=value`` argument for one retiled cycle.

    ``pixel_size`` is not optional: the reader defaults it to 1.0 and ASHLAR converts
    --maximum-shift with it, so leaving it out reads a micron budget as a pixel budget.
    """
    for bad in "|=":
        if bad in str(tile_dir):
            raise ValueError(
                f"tile directory {tile_dir} contains {bad!r}, which ASHLAR's reader "
                "syntax uses as a separator"
            )
    return (
        f"filepattern|{tile_dir}|pattern={grid['pattern']}"
        f"|overlap={float(grid['overlap'])!r}|pixel_size={float(grid['pixel_size_um'])!r}"
    )


def ashlar_argv(tile_dirs, grids, output, align_channel, maximum_shift_um, extra=()):
    """The command line, as a list: what a user would type, plus ``extra`` flags."""
    return [
        "ashlar",
        *[reader_spec(d, g) for d, g in zip(tile_dirs, grids)],
        "-o",
        str(output),
        "-c",
        str(int(align_channel)),
        "-m",
        repr(float(maximum_shift_um)),
        *[str(e) for e in extra],
    ]


def nuclear_index(explicit, channels, name, cycles) -> int:
    """The ONE channel index ashlar's ``-c`` aligns every cycle on.

    ASHLAR takes a single index for all cycles, so the nuclear stain must sit at the same
    position in each. Read from the channel lists when they are given; a cycle that has
    it elsewhere (or not at all) is refused, since ashlar would silently align a marker.
    """
    if explicit is not None:
        return int(explicit)
    if not channels:
        return 0
    found = []
    for cyc, spec in zip(cycles, channels):
        names = [c.strip().lower() for c in spec.split("|")]
        if name.lower() not in names:
            raise SystemExit(f"cycle {cyc}: no {name!r} among its channels {spec!r}")
        found.append(names.index(name.lower()))
    if len(set(found)) != 1:
        raise SystemExit(
            f"{name} is not at one index in every cycle ({dict(zip(cycles, found))}); "
            "ashlar aligns ONE channel index across cycles (-c)"
        )
    return found[0]


def _note(aligner) -> dict:
    """What an aligner knows once it has run, as plain lists."""

    def arr(name):
        v = getattr(aligner, name, None)
        return None if v is None else np.asarray(v, dtype=float).tolist()

    out = {
        "kind": type(aligner).__name__,
        "positions_yx": arr("positions"),
        "metadata_positions_yx": np.asarray(
            aligner.metadata.positions, dtype=float
        ).tolist(),
        "tile_size_yx": np.asarray(aligner.metadata.size, dtype=float).tolist(),
        "shifts_yx": arr("shifts"),
    }
    if out["kind"] == "EdgeAligner":
        tree = getattr(aligner, "spanning_tree", None)
        out.update(
            origin_yx=arr("origin"),
            mosaic_shape=[int(v) for v in aligner.mosaic_shape],
            model_coef=np.asarray(aligner.lr.coef_, dtype=float).tolist(),
            model_intercept=np.asarray(aligner.lr.intercept_, dtype=float).tolist(),
            max_error=float(aligner.max_error),
            n_edges=int(aligner.neighbors_graph.size()),
            n_edges_in_tree=None if tree is None else int(tree.size()),
        )
    else:
        out.update(
            reference_idx=[int(v) for v in aligner.reference_idx],
            cycle_offset_yx=arr("cycle_offset"),
            discard=[bool(v) for v in aligner.discard],
            errors=[
                None if not np.isfinite(e) else float(e)
                for e in np.asarray(aligner.errors, dtype=float)
            ],
        )
    return out


def run_ashlar(argv) -> tuple[int, list[dict], list[str]]:
    """Run the original command; returns (its exit code, one note per cycle, its data
    warnings). The aligners' ``run`` methods are wrapped only to be observed."""
    import warnings

    from ashlar import reg
    from ashlar.scripts import ashlar as cli

    notes: list[dict] = []
    originals = {cls: cls.run for cls in (reg.EdgeAligner, reg.LayerAligner)}

    def observed(cls):
        original = originals[cls]

        def run(self):
            result = original(self)
            notes.append(_note(self))
            return result

        return run

    for cls in originals:
        cls.run = observed(cls)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            code = cli.main(list(argv))
    finally:
        for cls, original in originals.items():
            cls.run = original
    warned = [str(w.message) for w in caught if issubclass(w.category, reg.Warning)]
    return int(code or 0), notes, warned


# ------------------------------------------------------------------- the manifest --
def _yx_to_xy(a):
    return np.asarray(a, dtype=float)[..., ::-1]


def tile_displacements(layer_positions, moving_true, ref_positions, ref_true, idx):
    """Per-tile ``D(t)`` in ``(y, x)``, moving slide -> reference slide (module docstring).

    ``*_true`` are where the tiles' pixels came from in their own slides -- NOT the stage
    positions ASHLAR was given, which differ from them by the stage error it had to find.
    """
    idx = np.asarray(idx, dtype=int)
    ref_frame = np.asarray(ref_positions, float)[idx] - np.asarray(ref_true, float)[idx]
    return (
        np.asarray(layer_positions, float) - np.asarray(moving_true, float) - ref_frame
    )


def piecewise_mesh(n_rows, n_cols, tile_size, stride, residual_xy, translation_xy):
    """``(grid_x, grid_y, displacements)`` of a field that is CONSTANT over each tile's own
    area and ramps linearly across the overlap bands.

    Two nodes per tile and axis, at the edges of the part of the tile no neighbour covers
    (``[k*stride + overlap, (k+1)*stride]``), both carrying that tile's residual. Between
    one tile's last node and the next tile's first lies the overlap band, where ASHLAR's
    mosaic blends the two tiles and a bilinear field ramps from one displacement to the
    other. Nodes are pushed through the rigid translation, because the warper samples the
    mesh at the rigid position.
    """
    overlap = tile_size - stride
    if not 0 <= overlap < tile_size / 2:
        raise ValueError(
            f"overlap of {overlap} px on a {tile_size} px tile leaves no part of a tile "
            "to itself; the piecewise field needs overlap < half a tile"
        )
    tx, ty = (float(v) for v in translation_xy)

    def nodes(n, shift):
        out = []
        for k in range(n):
            out += [k * stride + overlap + shift, (k + 1) * stride + shift]
        return out

    res = np.asarray(residual_xy, dtype=float).reshape(n_rows, n_cols, 2)
    return (
        nodes(n_cols, tx),
        nodes(n_rows, ty),
        np.repeat(np.repeat(res, 2, axis=0), 2, axis=1),
    )


def cycle_entry(edge, layer, ref_grid, mov_grid):
    """(manifest entry, rigid translation xy, per-tile D yx) of one moving cycle."""
    n_rows, n_cols = int(mov_grid["n_rows"]), int(mov_grid["n_cols"])
    d_yx = tile_displacements(
        layer["positions_yx"],
        mov_grid["true_positions_yx"],
        edge["positions_yx"],
        ref_grid["true_positions_yx"],
        layer["reference_idx"],
    )
    if len(d_yx) != n_rows * n_cols:
        raise ValueError(
            f"ashlar placed {len(d_yx)} tiles but the grid is {n_rows}x{n_cols}; the "
            "row-major reshape would mis-place every control point"
        )
    keep = ~np.asarray(layer["discard"], dtype=bool)
    if not keep.any():
        keep[:] = True  # every tile discarded: no better estimate exists
    d_xy = _yx_to_xy(d_yx)
    translation = np.median(d_xy[keep], axis=0)
    grid_x, grid_y, disp = piecewise_mesh(
        n_rows,
        n_cols,
        float(mov_grid["tile_size"]),
        float(mov_grid["stride"]),
        d_xy - translation,
        translation,
    )
    tx, ty = (float(v) for v in translation)
    entry = slide_entry(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], grid_x, grid_y, disp
    )
    entry["out_shape"] = [int(v) for v in ref_grid["orig_shape"]]
    return entry, translation, d_yx


def reference_stitch_error(edge, ref_grid) -> dict:
    """How far ASHLAR's stitched reference is from the truth, in px.

    Its positions minus where the tiles really came from should be ONE constant offset
    (the mosaic's origin); what is left after removing the median is its stitching error,
    which the synthetic tiles make measurable.
    """
    off = np.asarray(edge["positions_yx"], float) - np.asarray(
        ref_grid["true_positions_yx"], float
    )
    err = np.linalg.norm(off - np.median(off, axis=0), axis=1)
    return {
        "p50": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(err.max()),
    }


def cycle_diagnostics(layer, d_yx, translation_xy) -> dict:
    discard = np.asarray(layer["discard"], dtype=bool)
    errors = np.asarray([np.inf if e is None else e for e in layer["errors"]], float)
    finite = errors[np.isfinite(errors)]
    residual = np.linalg.norm(_yx_to_xy(d_yx) - np.asarray(translation_xy), axis=1)
    off = int((np.asarray(layer["reference_idx"]) != np.arange(len(discard))).sum())
    return {
        "n_tiles": int(len(discard)),
        "n_discarded": int(discard.sum()),
        "discard_fraction": float(discard.mean()) if len(discard) else 0.0,
        "n_reference_idx_off_diagonal": off,
        "cycle_offset_yx": layer["cycle_offset_yx"],
        "error_p50": float(np.median(finite)) if finite.size else None,
        "error_p90": float(np.percentile(finite, 90)) if finite.size else None,
        "n_error_infinite": int((~np.isfinite(errors)).sum()),
        "rigid_translation_xy": [float(v) for v in translation_xy],
        "mesh_residual_px": {
            "p50": float(np.median(residual)),
            "p90": float(np.percentile(residual, 90)),
            "max": float(residual.max()),
        },
    }


# --------------------------------------------------------------------- the mosaic --
def split_mosaic(mosaic_path, outputs, pixel_size_um):
    """Write each cycle of ASHLAR's mosaic as its own OME-TIFF.

    ``outputs`` is ``[(path, [channel names]), ...]`` in cycle order; the mosaic's planes
    are the cycles' channels concatenated in that order. Pixels are copied, never
    resampled: this is ASHLAR's image, one slide per file, with the channel names the
    figure tools find the nuclear plane by.
    """
    import tifffile

    with tifffile.TiffFile(str(mosaic_path)) as tf:
        series = tf.series[0]
        planes = series.shape[0] if series.ndim == 3 else 1
        wanted = sum(len(names) for _, names in outputs)
        if planes != wanted:
            raise ValueError(
                f"{mosaic_path} has {planes} planes but the cycles list {wanted} "
                "channels; refusing to guess which plane is which"
            )
        at = 0
        for path, names in outputs:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "axes": "CYX",
                "Channel": {"Name": list(names)},
                "PhysicalSizeX": float(pixel_size_um),
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": float(pixel_size_um),
                "PhysicalSizeYUnit": "µm",
            }
            pages = [series.pages[at + k] for k in range(len(names))]
            at += len(names)

            def tiles(pages=pages):
                # a tiled write from an iterator takes TILES, edge ones padded to size
                h, w = series.shape[-2:]
                for page in pages:
                    plane = page.asarray()
                    for y in range(0, h, SPLIT_TILE):
                        for x in range(0, w, SPLIT_TILE):
                            tile = plane[y : y + SPLIT_TILE, x : x + SPLIT_TILE]
                            if tile.shape != (SPLIT_TILE, SPLIT_TILE):
                                full = np.zeros((SPLIT_TILE, SPLIT_TILE), plane.dtype)
                                full[: tile.shape[0], : tile.shape[1]] = tile
                                tile = full
                            yield tile

            with tifffile.TiffWriter(str(path), bigtiff=True, ome=True) as tw:
                tw.write(
                    tiles(),
                    shape=(len(names), *series.shape[-2:]),
                    dtype=series.dtype,
                    tile=(SPLIT_TILE, SPLIT_TILE),
                    compression="zlib",
                    photometric="minisblack",
                    metadata=metadata,
                )
    return [Path(p) for p, _ in outputs]


# ------------------------------------------------------------------------- the CLI --
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the original ASHLAR command on retiled cycles, record where it "
        "placed the tiles, and write one scoring manifest per moving cycle."
    )
    ap.add_argument(
        "--tiles",
        nargs="+",
        required=True,
        metavar="DIR",
        help="retiled cycles (retile.py output); the FIRST is the reference, as the "
        "first file of an ashlar command is",
    )
    ap.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="slide name of each cycle, in the same order (the manifest's keys)",
    )
    ap.add_argument("--outdir", required=True, help="ashlar_output.ome.tif and records")
    ap.add_argument(
        "--nuclear-index",
        type=int,
        default=None,
        help="the channel ashlar aligns on (-c). Default: where --nuclear-name sits in "
        "--channels, which must be the same place in every cycle; 0 without --channels",
    )
    ap.add_argument("--nuclear-name", default="DAPI")
    ap.add_argument("--maximum-shift", type=float, default=15.0, dest="max_shift_um")
    ap.add_argument(
        "--max-discard-fraction",
        type=float,
        default=1.0,
        help="fail when ASHLAR replaced more than this fraction of a cycle's tiles with "
        "a model prediction (default 1 = report it, never fail)",
    )
    ap.add_argument(
        "--split",
        nargs="*",
        default=None,
        metavar="PATH",
        help="also write each cycle of the mosaic to PATH (one per cycle, same order)",
    )
    ap.add_argument(
        "--channels",
        nargs="*",
        default=None,
        metavar="A|B|C",
        help="with --split: each cycle's channel names, '|'-separated, same order",
    )
    ap.add_argument(
        "--ashlar-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="everything after this is passed to ashlar as typed (e.g. --filter-sigma 1)",
    )
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dirs = [Path(d) for d in a.tiles]
    if len(dirs) < 2 or len(a.names) != len(dirs):
        raise SystemExit(
            "--tiles needs a reference and at least one cycle, one --names each"
        )
    grids = [json.loads((d / "grid.json").read_text()) for d in dirs]
    for key in ("n_rows", "n_cols", "tile_size", "overlap", "pixel_size_um"):
        if len({json.dumps(g[key]) for g in grids}) != 1:
            raise SystemExit(
                f"the cycles disagree on {key}: {[g[key] for g in grids]}. Retile every "
                "cycle on one canvas with identical settings (retile.py --canvas-like)."
            )
    if any("true_positions_yx" not in g for g in grids):
        raise SystemExit(
            "grid.json has no true_positions_yx: retile with this checkout"
        )

    nuclear = nuclear_index(a.nuclear_index, a.channels, a.nuclear_name, a.names)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / OUTPUT_NAME
    command = ashlar_argv(dirs, grids, output, nuclear, a.max_shift_um, a.ashlar_args)
    logger.info("running: %s", " ".join(command))
    code, notes, warned = run_ashlar(command)
    if code != 0:
        raise SystemExit(f"ashlar exited {code}")
    if len(notes) != len(dirs) or notes[0]["kind"] != "EdgeAligner":
        raise SystemExit(
            f"expected one aligner per cycle, reference first; saw "
            f"{[n['kind'] for n in notes]}"
        )
    edge, layers = notes[0], notes[1:]

    import ashlar

    stitch = reference_stitch_error(edge, grids[0])
    record = {
        "ashlar_version": ashlar.__version__,
        "command": command,
        "warnings": warned,
        "cycles": a.names,
        "reference_stitch_error_px": stitch,
        "aligners": notes,
    }
    (outdir / "positions.json").write_text(json.dumps(record))
    logger.info(
        "reference stitched: %d/%d edges in the spanning tree, error vs the true tile "
        "positions p50=%.2f px, max=%.2f px",
        edge["n_edges_in_tree"] or 0,
        edge["n_edges"],
        stitch["p50"],
        stitch["max"],
    )

    failed = []
    for name, layer, grid in zip(a.names[1:], layers, grids[1:]):
        entry, translation, d_yx = cycle_entry(edge, layer, grids[0], grid)
        manifest = build_manifest(
            a.names[0], {a.names[0]: slide_entry(np.eye(3)), name: entry}
        )
        cdir = outdir / name
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        diag = cycle_diagnostics(layer, d_yx, translation)
        diag.update(
            method="ashlar",
            mode="original",
            ashlar_version=ashlar.__version__,
            reference=a.names[0],
            moving=name,
            pixel_size_um=float(grid["pixel_size_um"]),
            maximum_shift_um=a.max_shift_um,
            stage_jitter_um=grid.get("stage_jitter_um"),
            noise_frac=grid.get("noise_frac"),
            reference_stitch_error_px=stitch,
            data_warnings=warned,
        )
        (cdir / "tre.json").write_text(json.dumps(diag, indent=2))
        logger.info(
            "%s: %d tiles, %d discarded, rigid translation (x,y)=(%.2f, %.2f) px",
            name,
            diag["n_tiles"],
            diag["n_discarded"],
            translation[0],
            translation[1],
        )
        if diag["discard_fraction"] > a.max_discard_fraction:
            failed.append(f"{name} ({diag['discard_fraction']:.0%})")

    if a.split is not None:
        if len(a.split) != len(dirs) or len(a.channels or []) != len(dirs):
            raise SystemExit("--split and --channels each need one value per cycle")
        split_mosaic(
            output,
            [(p, c.split("|")) for p, c in zip(a.split, a.channels)],
            float(grids[0]["pixel_size_um"]),
        )
        logger.info("split %s into %d slide(s)", output.name, len(a.split))
    if failed:
        raise SystemExit(
            f"ASHLAR discarded more than {a.max_discard_fraction:.0%} of the tiles of: "
            f"{', '.join(failed)}. They are model predictions, not measurements."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

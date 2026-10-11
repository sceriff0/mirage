"""ASHLAR 1.20.0 (labsyspharm/ashlar), its own command, once per patient.

    ashlar 'filepattern|<reference tiles>|...' 'filepattern|<round 1 tiles>|...' ... \\
        -o ashlar_output.ome.tif -c <nuclear channel> -m 15

through ``benchmarks.ashlar.original.run_ashlar`` (``ashlar.scripts.ashlar.main``,
unmodified): the reference cycle is stitched by ``EdgeAligner``, every later cycle aligned
to it by ``LayerAligner``, and ONE mosaic written by ``PyramidWriter``. Nothing but the two
documented flags above is set.

**Four rows, one module.** ``noise`` (``ref`` | ``none``) x ``affine`` (``none`` |
``stare``), run with these labels (a label and its options must come together):

``ashlar``               noise=ref, affine=none. ASHLAR's primary row
``ashlar_clean``         noise=none, affine=none
``affine_ashlar``        noise=ref, affine=stare. A hybrid, never ranked as ASHLAR
``affine_ashlar_clean``  noise=none, affine=stare

**What is not ASHLAR's, and cannot be: the tiles.** The slides exist only stitched, and
ASHLAR "requires unstitched individual tile images as input" (README at v1.20.0
``[PARTIAL: lines 1-60]``). ``benchmarks.ashlar.retile.write_tiles`` cuts every slide of the
patient on one regular grid (1024 px tiles, 922 px pitch, all channels, whole pixels, no
stage error). ``noise=ref`` adds per-tile Gaussian noise to the REFERENCE cycle's tiles
only, s.d. = max(0.2 % of the channel's 1st-99th percentile range, 3 counts): on
pixel-identical overlaps ASHLAR's error metric can raise (``utils.py:74-86``), which its
command does not catch (``scripts/ashlar.py:231`` catches ``ProcessingError`` only), and
only the reference cycle's edge alignment can (``:261-262``). ``affine=stare`` cuts each
moving tile, unrotated and uninterpolated, where STARE's rigid matrix (``info["m0"]`` of
the ``stare`` run of that round) puts the matching reference tile's centre; ASHLAR is told
the regular grid. A round whose ``stare`` result is missing fails alone ("anchor
unavailable"). Reference tiles are cut once per noise level and shared on disk by the plain
and the anchored row (``<out>/cache/ashlar_tiles``); ASHLAR's reference stitch is not
shared, each command does its own.

**Points: exactly where ASHLAR drew each tile.** ``Mosaic.assemble_channel`` pastes tile
``i`` at ``aligner.positions[i]`` (``reg.py:1147-1153``) and ``utils.paste`` puts tile
pixel ``l`` on mosaic pixel ``position + l`` (``utils.py:149-194``). The wrapper records
both cycles' ``positions`` after each aligner has run, and ``grid.json`` says where each
tile's pixels came from. A moving point therefore goes, in ASHLAR's own direction,

    moving slide --(+ L[t] - c_mov[t])--> mosaic --(- (E[s] - c_ref[s]))--> reference slide

``t`` the moving tile holding the point, ``s`` the reference tile drawn where it lands in
the mosaic (``regbench.tilemap``). Both steps are one translation per tile: nothing is
inverted numerically, resampled or fitted. The mosaic is NOT on the reference slide's grid
(ASHLAR re-stitches the reference and moves its origin to zero, ``reg.py:713-716``): the
native record carries ``origin_xy``, where the mosaic's first pixel sits in the reference
slide's frame (minus the median of ``E - c_ref`` over tiles), and ``info`` the spread of
``E - c_ref`` around it (``reference_stitch_error``).

**Where that is not exact: seams.** Where two tiles of a cycle are drawn over the same
mosaic pixel ASHLAR cross-fades them (``utils.py:197-216``) and a point may be drawn twice;
the map there is a depth-weighted mean, an approximation (``regbench.tilemap``).
``info["band_mov"]`` marks the moving nuclei with a vertex on such a seam (ASHLAR's
geometry only: the tile grid and the positions it accepted), and the scorer scores every
method a second time without them (``<variant>:non_ashlar``).

**Selfcheck** (``selfcheck --method ashlar --mask overlap``). Two comparisons against the
mosaic file ASHLAR itself wrote, both needed and neither assumed: the moving dots through
the scored map against the moving cycle's planes, located by ``origin_xy``; and the
REFERENCE cycle's own dots (``SELFCHECK_REF_DOTS``) against the reference cycle's planes
through the second step of the map. The first leaves out the per-tile departure of
``E - c_ref`` from its median (declared, ``SELFCHECK_OFFSET``), and with it ``origin_xy``
cancels out of the first comparison; the second is what proves both from the image: every
reference dot found in the mosaic must go back onto a dot of the reference slide, and the
reported ``origin_xy`` must equal the origin read off those matched dots. Dots within 8 px
of a seam are masked and counted.

**Clock.** Every aligner finishes before the writer starts (``scripts/ashlar.py:262-296``),
so the clock and the memory peak are paused after the last aligner; tile cutting is paused
too and reported as ``retile_s``. ``register_s`` is the whole command up to the writer;
``info["register_s_round"]`` = that round's ``layer_s`` + ``edge_s`` / rounds.

ASHLAR has no core-count option and the harness sets none. It starts a JVM on import with
``-Xms10m -XX:+UseSerialGC`` and no heap limit (``reg.py:29-38``); the file-pattern reader
does not use it.

Source: labsyspharm/ashlar at the tag v1.20.0, read, never executed here: ``scripts/
ashlar.py``, ``utils.py``, ``thumbnail.py``, ``filepattern.py`` ``[FULL]``; ``reg.py``
``[PARTIAL: lines 1-119, 436-810, 836-1375; the BioFormats reader (120-435) and the plot
functions (1376-1583) not read]``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .. import imageio, resources, tilemap
from ..cases import load_points, polys
from ..mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from . import require_fluorescence

SCOPE = "group"
VARIANTS = ("ashlar",)
VERSION = "1.20.0"
SELFCHECK_REF_DOTS = True

# Frozen: changing one after a cohort run is a post hoc change.
TILE = 1024
OVERLAP = 0.1
MAX_SHIFT_UM = 15.0  # ASHLAR's own default (scripts/ashlar.py:63)
NOISE_FRAC = 0.002
NOISE_MIN_COUNTS = 3.0  # assumes 16-bit data
SEED = 0
NUCLEUS_MARGIN_PX = 1.0  # a vertex this close to a second tile is on the seam
DOT_MARGIN_PX = 8.0  # the selfcheck's dots are 7 px across

NOISES, AFFINES = ("ref", "none"), ("none", "stare")
# label -> (noise, affine)
LABELS = {
    "ashlar": ("ref", "none"),
    "ashlar_clean": ("none", "none"),
    "affine_ashlar": ("ref", "stare"),
    "affine_ashlar_clean": ("none", "stare"),
}
MASKS = ("overlap",)
WRITER = (
    "ashlar.reg.PyramidWriter (one mosaic per patient); the cycle's planes copied out "
    "of it, pixel for pixel, by benchmarks.ashlar.original.split_mosaic"
)
REFERENCE_SPLIT = "ashlar_cycle0_reference.ome.tif"


class _NullMonitor:
    def pause(self):
        pass

    def resume(self):
        pass

    def snapshot(self):
        return {}


def _truthy(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def _error(exc):
    return f"{type(exc).__name__}: {exc}"


def settings(opts):
    """``(noise, affine)`` of this run, checked against its label."""
    noise = str(opts.get("noise", "ref"))
    affine = str(opts.get("affine", "none"))
    if noise not in NOISES or affine not in AFFINES:
        raise ValueError(
            f"noise={noise!r} (one of {NOISES}), affine={affine!r} (one of {AFFINES})"
        )
    label = opts.get("label") or "ashlar"
    if label in LABELS and LABELS[label] != (noise, affine):
        want = LABELS[label]
        raise ValueError(
            f"the label {label!r} names the run noise={want[0]}, affine={want[1]}; "
            f"this one has noise={noise}, affine={affine}. A label and its options "
            "come together"
        )
    return noise, affine


def out_root(opts, case):
    """The results folder: ``opts["out"]``, else where the driver puts this case's image."""
    if opts.get("out"):
        return Path(opts["out"])
    native_dir = opts.get("native_dir")
    if native_dir is None:
        return None
    return Path(native_dir(case.case_id, VARIANTS[0])).parents[3]


# ── the anchor ────────────────────────────────────────────────────────────────
def anchor_paths(opts, case, root):
    """Where the ``stare`` run of this round may have left its ``run.json``.

    On the cohort: ``<out>/<affine_from>/<dataset>/<case>/run.json``. In the selfcheck
    (``opts["m0_from"]`` = the ``--work`` of a ``stare`` selfcheck): that folder's ``out``,
    directly, under the pair's size, or under the patient's name (the folder
    ``selfcheck.case_folder`` gives a case of several rounds: its group)."""
    variant = str(opts.get("affine_from", "stare"))
    tail = Path(variant) / case.dataset / str(case.case_id) / "run.json"
    if opts.get("m0_from"):
        base = Path(opts["m0_from"])
        h, w = (int(v) for v in case.ref_hw)
        return [
            base / f"{h}x{w}" / "out" / tail,
            base / str(case.group) / "out" / tail,
            base / "out" / tail,
        ]
    return [Path(root) / tail] if root is not None else []


def read_anchor(opts, case, root):
    """``(m0 (3, 3), path)`` or ``(None, message)``: STARE's rigid matrix, moving px ->
    reference px, from the ``run.json`` of its run on this round."""
    paths = anchor_paths(opts, case, root)
    for path in paths:
        try:
            run = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        m0 = (run.get("info") or {}).get("m0")
        if not run.get("ok") or m0 is None:
            return None, f"anchor unavailable: {path} holds no info.m0 of a run that ok"
        m0 = np.asarray(m0, dtype=float)
        if m0.shape != (3, 3) or not np.isfinite(m0).all():
            return None, f"anchor unavailable: {path} has no 3 x 3 finite info.m0"
        return m0, str(path)
    where = " or ".join(str(p) for p in paths) or "(no results folder is known)"
    return None, f"anchor unavailable: {where}"


# ── tiles ─────────────────────────────────────────────────────────────────────
def _file_key(path):
    st = os.stat(path)
    return [str(Path(path).resolve()), int(st.st_size), int(st.st_mtime_ns)]


def reference_tiles(ref, slides, px, noise, cache, work):
    """The reference cycle's tiles, cut once per noise level: ``(dir, seconds, reused)``.

    The plain and the anchored row of a patient read the same reference tiles (the noise
    seed is ``[seed, cycle, 1, row, col]`` and the reference is cycle 0 in every row), so
    they are cut once into ``cache`` and found there by the second row. Two jobs cutting
    at once each write their own folder and one rename wins.

    ``seconds`` is what cutting THESE tiles cost, whoever cut them: it is written beside
    the tiles (``retile_seconds.json``) and read back on a cache hit, so a row's
    ``retile_s`` does not depend on which row of the patient ran first.
    """
    from benchmarks.ashlar import retile

    frac, floor = (NOISE_FRAC, NOISE_MIN_COUNTS) if noise == "ref" else (0.0, 0.0)
    key = [
        _file_key(ref),
        sorted(retile._yx_shape(p) for p in slides),
        [TILE, OVERLAP, float(px), frac, floor, SEED],
    ]
    name = f"ref_{noise}_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
    home = Path(cache) if cache is not None else Path(work) / "tiles"
    final = home / name
    if (final / "grid.json").exists():
        return final, _cut_seconds(final), True
    home.mkdir(parents=True, exist_ok=True)
    tmp = home / f"tmp{os.getpid()}_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.perf_counter()
    try:
        retile.write_tiles(
            ref,
            tmp,
            TILE,
            OVERLAP,
            cycle=0,
            canvas_like=slides,
            pixel_size_um=px,
            noise_frac=frac,
            noise_min_counts=floor,
            seed=SEED,
            exact_overlap=True,
        )
        (tmp / RETILE_SECONDS).write_text(
            json.dumps({"seconds": time.perf_counter() - t0})
        )
        try:
            os.rename(tmp, final)
        except OSError:  # another job finished first: its tiles are the same tiles
            if not (final / "grid.json").exists():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final, _cut_seconds(final), False


RETILE_SECONDS = "retile_seconds.json"


def _cut_seconds(tiles):
    """What cutting the tiles in this folder cost, as the job that cut them recorded it;
    None for a folder without the record (cut before the record existed)."""
    try:
        return float(json.loads((Path(tiles) / RETILE_SECONDS).read_text())["seconds"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def nominal_origins(slides):
    """The regular grid every cycle is told, ``(n_tiles, 2)`` in ``(y, x)``, row-major."""
    from benchmarks.ashlar import retile

    shapes = [retile._yx_shape(p) for p in slides]
    h, w = max(s[0] for s in shapes), max(s[1] for s in shapes)
    return np.array([[t.y, t.x] for t in retile.tile_grid(w, h, TILE, OVERLAP)])


# ── what ASHLAR did, from its aligners' notes ─────────────────────────────────
def tile_order(metadata_positions_yx, grid):
    """For each of ASHLAR's tiles, its row-major index in ``grid.json``.

    ASHLAR's file-pattern reader numbers tiles row-major (``filepattern.py:97-100``) and
    computes their positions from row and column (``:90-92``); this reads the order back
    from those positions instead of assuming it, and refuses anything that is not the grid.
    """
    pos = np.asarray(metadata_positions_yx, dtype=float)
    rc = np.rint(pos / float(grid["stride"])).astype(int)
    index = rc[:, 0] * int(grid["n_cols"]) + rc[:, 1]
    n = int(grid["n_rows"]) * int(grid["n_cols"])
    off = np.abs(pos - rc * float(grid["stride"])).max() if len(pos) else 0.0
    if len(pos) != n or sorted(index.tolist()) != list(range(n)) or off > 1e-3:
        raise RuntimeError(
            f"ASHLAR's {len(pos)} tile positions are not the {grid['n_rows']} x "
            f"{grid['n_cols']} grid of pitch {grid['stride']} px that was cut (largest "
            f"departure {off:.3g} px): the tiles cannot be told apart"
        )
    return index


def placement_of(edge, layer, ref_grid, mov_grid):
    """Where each tile's pixels came from and where ASHLAR drew it, in ASHLAR's tile
    order, as plain lists (it is stored in ``run.json``: it IS the transform)."""
    ref_true = np.asarray(ref_grid["true_positions_yx"], dtype=float)
    mov_true = np.asarray(mov_grid["true_positions_yx"], dtype=float)
    ref_true = ref_true[tile_order(edge["metadata_positions_yx"], ref_grid)]
    mov_true = mov_true[tile_order(layer["metadata_positions_yx"], mov_grid)]
    return {
        "tile": int(mov_grid["tile_size"]),
        "mov_true_yx": mov_true.tolist(),
        "mov_pos_yx": np.asarray(layer["positions_yx"], dtype=float).tolist(),
        "ref_true_yx": ref_true.tolist(),
        "ref_pos_yx": np.asarray(edge["positions_yx"], dtype=float).tolist(),
        "mosaic_shape": [int(v) for v in edge["mosaic_shape"]],
    }


def mosaic_origin_xy(placement):
    """Where the mosaic's first pixel sits in the reference slide's frame, ``(x, y)``:
    minus the median over reference tiles of (drawn at - cut from)."""
    off = np.asarray(placement["ref_pos_yx"]) - np.asarray(placement["ref_true_yx"])
    return (-np.median(off, axis=0)[::-1]).tolist()


def point_map(placement):
    """The scored map: moving slide -> reference slide, through the mosaic."""
    to_reference, _ = tilemap.placement_maps(placement)
    return PointMap(to_reference, MOV_TO_REF)


def discards(layer, edge, max_shift_px):
    """Why ASHLAR replaced a tile's position by its linear model, one cause per tile.

    ``LayerAligner.constrain_positions`` (``reg.py:910-946``) keeps only the combined
    flag. The three causes are rebuilt from the same quantities, with a fixed precedence
    because they overlap: an infinite error (``:923``); then a final position equal to its
    reference tile's to 0.1 px on both axes (``:911-921``; before the model replaces it
    that difference is metadata position + cycle offset + shift - the reference tile's
    metadata position); then further than the shift budget from the model (``:935-939``).
    """
    meta = np.asarray(layer["metadata_positions_yx"], dtype=float)
    shifts = np.asarray(layer["shifts_yx"], dtype=float)
    idx = np.asarray(layer["reference_idx"], dtype=int)
    ref_meta = np.asarray(edge["metadata_positions_yx"], dtype=float)[idx]
    ref_pos = np.asarray(edge["positions_yx"], dtype=float)[idx]
    nominal = meta + np.asarray(layer["cycle_offset_yx"], dtype=float)
    infinite = np.array([e is None for e in layer["errors"]], dtype=bool)
    zero = (np.rint(np.abs(nominal + shifts - ref_meta) * 10) / 10 == 0).all(axis=1)
    first = zero | infinite
    offset = 0.0 if first.all() else np.nan_to_num(np.median(shifts[~first], axis=0))
    coef = np.asarray(edge["model_coef"], dtype=float)
    predictions = nominal @ coef.T + np.asarray(edge["model_intercept"], dtype=float)
    measured = nominal + shifts + ref_pos - ref_meta
    over = np.linalg.norm(measured - predictions - offset, axis=1) > max_shift_px
    combined = infinite | zero | over
    return {
        "discard_infinite_error": int(infinite.sum()),
        "discard_zero_offset": int((zero & ~infinite).sum()),
        "discard_over_budget": int((over & ~zero & ~infinite).sum()),
        "n_tiles": int(len(meta)),
        "n_discarded": int(np.asarray(layer["discard"], dtype=bool).sum()),
        # the rebuilt flag against the one ASHLAR kept: False means this reading is wrong
        "discard_rebuild_matches": bool(
            np.array_equal(combined, np.asarray(layer["discard"], dtype=bool))
        ),
    }


def nucleus_band(placement, points, which="outlines"):
    """One bool per moving nucleus: True where any vertex is on one of ASHLAR's seams.
    ``which`` is the polygon set (``cases.SETS``): the outlines, or the bare nuclei."""
    ps = polys(points, "mov", which)
    if ps is None or not ps.n_features:
        return None
    on = tilemap.on_seam(placement, ps.xy, NUCLEUS_MARGIN_PX)
    feat = np.repeat(ps.ring_feat, np.diff(ps.ring_off))
    band = np.zeros(ps.n_features, dtype=bool)
    np.logical_or.at(band, feat, on)
    return band


# ── selfcheck hooks ───────────────────────────────────────────────────────────
def SELFCHECK_MASK(info, mask):
    """``--mask overlap``: the dots within ``DOT_MARGIN_PX`` of a seam are left out."""
    if mask not in MASKS:
        raise SystemExit(f"ashlar knows the masks {MASKS}, not {mask!r}")
    placement = info["placement"]
    return lambda xy: tilemap.on_seam(placement, xy, DOT_MARGIN_PX)


def SELFCHECK_OFFSET(info):
    """The scored map minus (mosaic position + ``origin_xy``): how far the reference tile
    under each point was drawn from the median placement. ``SELFCHECK_REFERENCE`` is the
    comparison that proves this part from the image."""
    placement = info["placement"]
    to_reference, to_mosaic = tilemap.placement_maps(placement)
    origin = np.asarray(mosaic_origin_xy(placement), dtype=float)

    def declared(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return to_reference(xy) - (to_mosaic(xy) + origin)

    return {VARIANTS[0]: declared}


def SELFCHECK_REFERENCE(info, native, pair):
    """The reference cycle as ASHLAR redrew it: every dot found in its planes of the
    mosaic, sent back by the SECOND step of the scored map (mosaic -> reference slide),
    must land on a dot of the reference slide."""
    spec = (native or {}).get("reference_dots")
    if not spec:
        return None
    placement = info["placement"]
    tile = float(placement["tile"])
    ref_pos, ref_true = placement["ref_pos_yx"], placement["ref_true_yx"]
    back = tilemap.tile_point_map(
        ref_pos, tile, np.asarray(ref_true, dtype=float) - np.asarray(ref_pos)
    )

    def drop(xy):
        return ~tilemap.single_cover(ref_pos, tile, xy, DOT_MARGIN_PX)

    return {
        "point_map": PointMap(back, REF_TO_MOV),
        "file": spec["file"],
        "channel": spec["channel"],
        "drop": drop,
        "expect_origin_xy": native.get("origin_xy"),
        # where a reference dot belongs in the mosaic, to within the tiles' shifts: for
        # the coverage half of the comparison (every drawn dot must be shown)
        "expect": lambda xy: (
            np.asarray(xy, dtype=float).reshape(-1, 2)
            - np.asarray(native.get("origin_xy") or (0.0, 0.0), dtype=float)
        ),
    }


# ── the run ───────────────────────────────────────────────────────────────────
def _channel_names(case, side, n):
    names = case.extra.get(f"channels_{side}")
    if not names and side == "mov":
        names = case.extra.get("channel_names")
    names = [str(v) for v in names or []]
    return names if len(names) == n else [f"channel_{i}" for i in range(n)]


def _align_channel(cases):
    """The ONE index ``-c`` aligns every cycle on, through the harness's own rule
    (``original.nuclear_index``: refused unless it is the same index in every cycle)."""
    from benchmarks.ashlar import original

    def spec(i):
        return "|".join("nuclear" if k == i else f"c{k}" for k in range(int(i) + 1))

    ref = int(cases[0].extra.get("ref_channel", 0))
    movs = [int(c.extra.get("mov_channel", 0)) for c in cases]
    cycles = ["reference", *[c.case_id for c in cases]]
    return original.nuclear_index(
        None, [spec(ref), *[spec(i) for i in movs]], "nuclear", cycles
    )


def _valid_box(placement, ref_hw):
    """The reference slide's extent inside the mosaic, ``[x0, y0, x1, y1]`` in the image's
    own pixels: outside it the mosaic is the tiles' zero padding."""
    ox, oy = mosaic_origin_xy(placement)
    h, w = placement["mosaic_shape"]
    x0, y0 = max(0, int(np.ceil(-ox))), max(0, int(np.ceil(-oy)))
    x1 = min(int(w), int(np.floor(-ox + ref_hw[1])))
    y1 = min(int(h), int(np.floor(-oy + ref_hw[0])))
    return [x0, y0, x1, y1]


def _edge_identity(cache_dir, tiles_name, base, edge):
    """Whether this command's reference stitch equals the other row's on the same
    reference tiles (None when no other row has run): the tiles are shared between the
    plain and the anchored row, the stitch is NOT, each command does its own."""
    if cache_dir is None:
        return None
    label = base["label"]
    mine = np.asarray(edge["positions_yx"], dtype=float)
    same = None
    for other in sorted(Path(cache_dir).glob(f"{tiles_name}.edge_*.json")):
        if other.name == f"{tiles_name}.edge_{label}.json":
            continue
        try:
            theirs = np.asarray(json.loads(other.read_text()), dtype=float)
        except (OSError, ValueError):
            continue
        same = bool(theirs.shape == mine.shape and np.array_equal(theirs, mine))
    try:
        path = Path(cache_dir) / f"{tiles_name}.edge_{label}.json"
        path.write_text(json.dumps(mine.tolist()))
    except OSError:
        pass
    return same


def register_group(cases, work, opts):
    work = Path(work)
    noise, affine = settings(opts)
    for case in cases:
        require_fluorescence(case, "ashlar")
    label = opts.get("label") or "ashlar"
    mon = opts.get("monitor") or _NullMonitor()
    selfcheck = _truthy(opts.get("stage_images", 0))
    variant = VARIANTS[0]

    import ashlar

    from benchmarks.ashlar import original, retile

    version = str(getattr(ashlar, "__version__", "?"))
    if version != VERSION and not _truthy(opts.get("any_version", 0)):
        raise RuntimeError(
            f"this adapter is ASHLAR {VERSION} as shipped; the installed ashlar is "
            f"{version}"
        )
    refs = sorted({str(c.ref_image) for c in cases})
    if len(refs) != 1:
        raise ValueError(
            f"one patient has one reference slide, got {len(refs)}: {refs}"
        )
    ref = refs[0]
    slides = [ref, *[str(c.mov_image) for c in cases]]
    px = float(cases[0].pixel_size_um or retile.DEFAULT_PIXEL_SIZE_UM)
    channel = _align_channel(
        cases
    )  # SystemExit when the cycles disagree: fails the patient
    root = out_root(opts, cases[0])
    cache = opts.get("tiles_cache")
    if cache is None and root is not None:
        cache = (
            root / "cache" / "ashlar_tiles" / f"{cases[0].dataset}__{cases[0].group}"
        )

    base = {
        "version": version,
        "source": str(Path(getattr(ashlar, "__file__", "?")).parent),
        "label": label,
        "noise": noise,
        "affine": affine,
        "tile": TILE,
        "overlap": OVERLAP,
        "max_shift_um": MAX_SHIFT_UM,
        "noise_frac": NOISE_FRAC if noise == "ref" else 0.0,
        "noise_min_counts": NOISE_MIN_COUNTS if noise == "ref" else 0.0,
        "seed": SEED,
        "align_channel": int(channel),
        "pixel_size_um": px,
        "workers_seen": "ASHLAR has no core-count option; none is set",
        "implementation": {
            "package": "ashlar",
            "registration": "ashlar.scripts.ashlar.main([... one filepattern reader "
            "per cycle ..., '-o', mosaic, '-c', channel, '-m', '15.0'])",
            "parameters": "ASHLAR's defaults",
            "point_warp": "one translation per tile, from the positions ASHLAR's own "
            "aligners hold when its writer pastes the tiles (regbench.tilemap)",
            "benchmark_side": [
                "tiles: every slide of the patient cut on one regular grid of "
                f"{TILE} px tiles at a {retile._stride(TILE, OVERLAP)} px pitch, all "
                "channels, whole pixels, read by ASHLAR's own filepattern reader "
                "(the slides exist only stitched)",
                "noise on the reference cycle's tiles only: s.d. = max(0.2 % of the "
                "channel's 1st-99th percentile range, 3 counts), one draw per tile"
                if noise == "ref"
                else "no noise on any tile",
                "the align channel index (-c) and the pixel size of the reader "
                "(ASHLAR converts --maximum-shift with it)",
                "EdgeAligner.run and LayerAligner.run wrapped to be OBSERVED after "
                "they return: positions, shifts, errors, discards, seconds",
                "the mosaic's planes of each cycle copied into one OME-TIFF per "
                "cycle (pixels untouched)",
                "the clock and the memory peak paused while tiles are cut and from "
                "the last aligner's return on (ASHLAR's writer)",
                *(
                    [
                        "moving tiles cut where STARE's rigid matrix puts the "
                        "matching reference tile's centre, unrotated, whole pixels; "
                        "ASHLAR is told the regular grid"
                    ]
                    if affine == "stare"
                    else []
                ),
            ],
        },
    }

    # ── tiles: the clock is paused, this is the benchmark making ASHLAR's input ──
    mon.pause()
    try:
        ref_dir, ref_s, reused = reference_tiles(ref, slides, px, noise, cache, work)
        t0 = time.perf_counter()  # the moving cycles' tiles, cut by every row
        ref_grid = json.loads((ref_dir / "grid.json").read_text())
        nominal = nominal_origins(slides)
        included, anchors, dropped = [], {}, []
        for case in cases:
            if affine == "stare":
                m0, where = read_anchor(opts, case, root)
                if m0 is None:
                    dropped.append((case, where))
                    continue
                anchors[case.case_id] = (m0, where)
            included.append(case)
        mov_dirs, mov_grids = [], []
        for k, case in enumerate(included, 1):
            origins = None
            if affine == "stare":
                origins = tilemap.anchor_tile_origins(
                    anchors[case.case_id][0], nominal, TILE
                )
            d = work / "tiles" / f"cycle_{k}"
            # moving cycles get no noise in any row
            retile.write_tiles(
                case.mov_image,
                d,
                TILE,
                OVERLAP,
                cycle=k,
                canvas_like=slides,
                pixel_size_um=px,
                seed=SEED,
                exact_overlap=True,
                tile_origins_yx=origins,
            )
            mov_dirs.append(d)
            mov_grids.append(json.loads((d / "grid.json").read_text()))
        moving_s = time.perf_counter() - t0
        # the reference's share is what cutting it cost, cache hit or not: the row
        # would have paid it had it run first (or alone)
        retile_s = moving_s + ref_s if ref_s is not None else None
    finally:
        mon.resume()
    base.update(
        retile_s=retile_s,
        retile_moving_s=moving_s,
        retile_reference_s=ref_s,
        reference_tiles_reused=reused,
        reference_tiles=str(ref_dir),
        reference_noise_sd=ref_grid.get("noise_sd"),
        rounds_in_command=[c.case_id for c in included],
    )
    for case, why in dropped:  # the patient's other rounds proceed
        yield case.case_id, variant, None, {**base, "error": why}
    if not included:
        return

    # ── ASHLAR's own command: reference first, then every round ──────────────────
    mosaic = work / original.OUTPUT_NAME
    work.mkdir(parents=True, exist_ok=True)
    grids = [ref_grid, *mov_grids]
    argv = original.ashlar_argv(
        [ref_dir, *mov_dirs], grids, mosaic, channel, MAX_SHIFT_UM
    )
    base["command"] = argv
    n_cycles = len(grids)
    seen = {}

    def after_aligner(index, note):
        if index == n_cycles - 1:
            # registration is over: what follows inside the command is the writer
            seen["peak_rss_mb"] = (mon.snapshot() or {}).get("peak_rss_mb")
            mon.pause()
            seen["write_t0"] = time.perf_counter()

    def failed(message, **extra):
        return [
            (c.case_id, variant, None, {**base, "error": message, **extra})
            for c in included
        ]

    ready = None
    try:
        try:
            code, notes, warned = original.run_ashlar(
                argv, notes_path=work / "aligner_notes.json", on_aligner=after_aligner
            )
        except Exception as exc:
            # a memory error keeps its cause (the ladder reads it); anything else, the
            # clean tiles' RuntimeError first of all, fails the patient with its message
            if resources.classify_error(exc) != "exception":
                raise
            ready = failed(_error(exc), ashlar_raise=_error(exc))
        else:
            mon.pause()
            write_s = time.perf_counter() - seen.get("write_t0", time.perf_counter())
            if code != 0:
                ready = failed(
                    f"ashlar exited {code} (it printed its error; see the log)"
                )
            elif len(notes) != n_cycles or notes[0]["kind"] != "EdgeAligner":
                ready = failed(
                    "expected one aligner per cycle, reference first; saw "
                    f"{[n['kind'] for n in notes]}"
                )
            else:
                ready = _results(
                    included,
                    anchors,
                    notes,
                    warned,
                    grids,
                    mosaic,
                    base,
                    opts,
                    write_s,
                    seen.get("peak_rss_mb"),
                    selfcheck,
                    cache,
                )
    finally:
        mon.resume()
    yield from ready


def _results(
    cases,
    anchors,
    notes,
    warned,
    grids,
    mosaic,
    base,
    opts,
    write_s,
    peak,
    selfcheck,
    cache_dir,
):
    """One result per round, built with the clock paused."""
    from benchmarks.ashlar import original

    edge, layers = notes[0], notes[1:]
    ref_grid = grids[0]
    px = float(ref_grid["pixel_size_um"])
    rounds = len(cases)
    edge_s = float(edge["seconds"])
    layer_s = [float(n["seconds"]) for n in layers]
    identical = _edge_identity(
        cache_dir, Path(base["reference_tiles"]).name, base, edge
    )
    shared = {
        **base,
        "warnings": warned,
        "edge_s": edge_s,
        "layer_s": layer_s,
        "edge_positions_identical": identical,
        "reference_stitch_error": original.reference_stitch_error(edge, ref_grid),
        "reference_stitch": {
            "n_edges": edge.get("n_edges"),
            "n_edges_in_tree": edge.get("n_edges_in_tree"),
            "max_error": edge.get("max_error"),
        },
        "peak_rss_after_register_mb": peak,
    }

    # ── the image: ASHLAR's mosaic, each cycle's planes copied to its own file ───
    native_dir = opts.get("native_dir")
    homes = {}
    for case in cases:
        home = (
            Path(native_dir(case.case_id, VARIANTS[0]))
            if native_dir is not None
            else mosaic.parent / "native" / str(case.case_id)
        )
        home.mkdir(parents=True, exist_ok=True)
        homes[case.case_id] = home
    first = homes[cases[0].case_id]
    n_ref = int(ref_grid["n_channels"])
    names = [_channel_names(cases[0], "ref", n_ref)]
    files = [first / REFERENCE_SPLIT]
    for case, grid in zip(cases, grids[1:]):
        names.append(_channel_names(case, "mov", int(grid["n_channels"])))
        files.append(homes[case.case_id] / f"{case.case_id}.ome.tif")
    # what ASHLAR itself wrote, before the benchmark's copies are cut from it: one file
    # for the patient, every cycle's channels
    own = imageio.describe_written(mosaic)
    if "unread" not in own:
        own["of"] = "ASHLAR's own mosaic: every cycle of the patient in one file"
    writer_s = write_s
    t0 = time.perf_counter()
    split_error = None
    try:
        original.split_mosaic(mosaic, list(zip(files, names)), px)
    except Exception as exc:
        split_error = _error(exc)
    kept = None
    if selfcheck or _truthy(opts.get("keep_mosaic", 0)):
        # the selfcheck reads ASHLAR's own file, not the copy
        kept = first / mosaic.name
        try:
            shutil.move(str(mosaic), str(kept))
        except OSError as exc:
            split_error = split_error or _error(exc)
            kept = None
    split_s = time.perf_counter() - t0
    write_s_patient = write_s + split_s
    write_s = write_s_patient / rounds  # one writer call draws the whole patient
    # a round's share of the two parts: ASHLAR's writer, and the benchmark's own split
    parts = {"writer": writer_s / rounds, "convert": 0.0, "benchmark": split_s / rounds}

    out = []
    plane = n_ref
    for k, (case, layer, grid) in enumerate(zip(cases, layers, grids[1:])):
        cid = case.case_id
        own_plane, plane = plane, plane + int(grid["n_channels"])
        try:
            placement = placement_of(edge, layer, ref_grid, grid)
            d_yx = original.tile_displacements(
                layer["positions_yx"],
                placement["mov_true_yx"],
                edge["positions_yx"],
                placement["ref_true_yx"],
                layer["reference_idx"],
            )
            jumps = tilemap.neighbour_jumps(
                d_yx, int(grid["n_rows"]), int(grid["n_cols"])
            )
            case_points = load_points(case)
            band = nucleus_band(placement, case_points)
            band_nuc = nucleus_band(placement, case_points, "nuclei")
            info = {
                **shared,
                "placement": placement,
                "register_s_round": layer_s[k] + edge_s / rounds,
                "cycle": k + 1,
                "cycle_offset_yx": layer["cycle_offset_yx"],
                **discards(layer, edge, MAX_SHIFT_UM / px),
                "band_fraction": float(band.mean()) if band is not None else None,
                "jump_px_p50": float(np.median(jumps)) if len(jumps) else None,
                "jump_px_p95": float(np.percentile(jumps, 95)) if len(jumps) else None,
                "tile_origins": grid.get("tile_origins"),
                "write_s": write_s,
                "write_s_patient": write_s_patient,
            }
            if cid in anchors:
                info["m0"] = anchors[cid][0].tolist()
                info["anchor_from"] = anchors[cid][1]
            if band is not None:
                info["band_mov"] = band
            if band_nuc is not None:  # the same rule on the bare nuclei
                info["band_mov_nuc"] = band_nuc
            if split_error:
                info["native"] = {"error": split_error}
            else:
                origin = mosaic_origin_xy(placement)
                native = {
                    "files": [str(files[k + 1])],
                    "format": "ome-tiff",
                    "writer": WRITER,
                    "written": own,
                    "write_parts_s": parts,
                    "fill": "zero (the tiles' padding and what no tile covers)",
                    "channels": names[k + 1],
                    "dtype": str(np.dtype(_dtype_of(files[k + 1]))),
                    # where this image's first pixel sits in the reference slide's frame
                    "origin_xy": origin,
                    "valid_box": _valid_box(placement, case.ref_hw)
                    if case.ref_hw
                    else None,
                    "nuclear_channel": int(base["align_channel"]),
                    "mosaic_shape": placement["mosaic_shape"],
                    "reference_file": str(files[0]),
                    "frame": "ASHLAR's mosaic: the reference cycle as ASHLAR "
                    "re-stitched it, not the reference slide's grid",
                }
                if kept is not None:
                    native["mosaic"] = str(kept)
                    native["mosaic_planes"] = [own_plane, plane]
                    dots = case.extra.get("dots_channel")
                    if selfcheck and dots is not None:
                        native["dots_file"] = str(kept)
                        native["dots_channel"] = own_plane + int(dots)
                        native["reference_dots"] = {
                            "file": str(kept),
                            "channel": int(dots),
                        }
                info["native"] = native
            out.append((cid, VARIANTS[0], point_map(placement), info))
        except Exception as exc:  # this round's bookkeeping failed; the others stand
            out.append((cid, VARIANTS[0], None, {**shared, "error": _error(exc)}))
    return out


def _dtype_of(path):
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        return tf.series[0].dtype

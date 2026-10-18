"""The overlay zooms: one field of a cohort case before registration, after the rigid
stage, and after the whole transform, for each method's best row (Fig. 3a), STARE's alone
(Fig. 2b) and the ``before`` panel over the whole slide (Fig. 1c).

**Residuals** (``residuals``). Per stage (``initial``, ``rigid``, ``final``), one distance
per REFERENCE nucleus (outlines), in um: from the reference centroid to the nearest moving
centroid in THAT stage's frame. A ``mov_to_ref`` row moves the moving outlines
(``warped.npz: cells_xy``, regrouped into polygons by the case's ring offsets) and is
compared with the native reference centroids; a ``ref_to_mov`` row (VALIS, PALOM,
DeeperHistReg) moves the REFERENCE outlines into the moving slide, and each moved
reference centroid is compared with the native moving centroids. ``initial`` is native
against native. Every distance is capped at the case's gate,
``pairs.gate_px(ref_native, mov_native)[0]`` (``pairs.GATE_FACTOR`` x the median equivalent
radius of the outlines), and converted to um by the case's pixel size. A reference nucleus
whose moved centroid is not finite is at the cap.

**The field** (``pick_roi``, rule ``ROI_RULE``) is chosen ONCE per case on the STARE best
row (``paper/method_best.csv``) and drawn for every family's best row in that same field;
``overlay_rois.csv`` records each drawn row's own three medians there. The case is the one
whose chosen window has the largest ``d_rigid - d_final`` among the cases with a monotone
window (``pick_case``, rule ``CASE_RULE``), unless ``--case-id`` names one. The candidate
cases are recorded too (their STARE row, ``rule`` beginning ``candidate case``).

**The rigid stage** of a row is read from the run list, never from a hand list
(``rigid_of``): the first result name of the row's run (``variants.variants_of``) when the
run writes more than one and that name is a rigid stage (``tables.stage_of``: ``rigid`` or
``micro_rigid``). STARE's ``<run>_rigid``, VALIS's ``<run>_rigid`` (at micro-registration
depth 1 and 2 that row already holds the micro-rigid pass: titled ``after rigid (+
micro-rigid)``), PALOM's ``palom_affine``. ASHLAR and DeeperHistReg write one row per run:
their rigid panel says ``no rigid stage`` and nothing is drawn in it.

**Panels**, each a two-colour nuclear overlay of the field (reference cyan, moving
magenta) on the reference grid, with the scored outlines on top (reference cyan, the
row's moved moving outlines magenta) and a scale bar in um:

``before``                    the native moving nuclear channel at the same coordinates
``after rigid``               the moving nuclear channel resampled through the rigid
                              variant's map: ``warped.npz: grid`` on its lattice,
                              interpolated bilinearly (exact for an affine map, and
                              extrapolated linearly past the lattice). A ``ref_to_mov`` map
                              gives each field pixel its moving position directly; a
                              ``mov_to_ref`` lattice is inverted by fixed-point iteration to
                              ``INVERT_TOL_PX`` (a pixel that does not converge is blank)
``after the whole transform`` the crop of the image the METHOD wrote (``run.json:
                              native.files[0]``, or ``native.nuclear_file`` when the writer
                              names one; its nuclear channel as ``tissue.native_channel``
                              reads it), placed by ``native.origin_xy`` (default ``(0, 0)``,
                              on the reference grid). DeeperHistReg's one row is this image.

The moved outlines of a ``ref_to_mov`` row are the native moving outlines brought onto the
reference grid through the inverse of its lattice (drawing only; its residuals stay in the
moving frame). An overview panel shows the whole reference as nucleus density (reference
cyan, native moving magenta, from the case's points, no pixel read) with the field's box,
unless the reference slide has a reduced pyramid level (a series level / SubIFD, never level
0) whose long side is at most ``OVERVIEW_MAX_SIDE``: then the largest such level is read
whole (``read_level``, a bounded read) and its nuclear channel drawn in cyan, the box in that
level's scale. The fallback is labelled ``NO_LEVEL`` in the panel and the source data.

**Reading.** Every crop is read by region (``read_region``: ``tifffile`` ``aszarr`` +
``zarr``), never the whole slide: the field itself, or for a resampled panel the box its
pixels come from plus ``READ_PAD_PX``. A box larger than ``read_limit(roi_px)`` is not read
and the panel says so.

**Placeholders (R22).** Without ``method_best.csv``, without a STARE best row, or without a
STARE result on any case, each figure is the placeholder page of ``plots`` naming what is
missing. A row whose ``warped.npz`` or written image is missing keeps its panel, which
names the missing file; the source-data CSV says the same.

Figures: ``OVERLAYS`` (``<out>/paper/figures/<name>.{pdf,png,svg}`` and
``<name>_source_data.csv``, through ``plotstyle.save``); table: ``<out>/paper/
overlay_rois.csv`` (``paper.schema``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .. import pairs, variants  # noqa: E402
from ..cases import list_cases, load_points, polys, result_dir  # noqa: E402
from ..mapping import MOV_TO_REF, REF_TO_MOV  # noqa: E402
from . import plots, schema, tables  # noqa: E402
from . import plotstyle as ps  # noqa: E402

STAGES = ("initial", "rigid", "final")
OVERLAYS = ("fig3a_overlays", "fig1c_overlay_before", "fig2b_overlay_stare")
FIG3A_FAMILIES = ("stare", "valis_mirage", "palom", "deeperhistreg", "ashlar")
STARE_FAMILY = "stare"
COHORT = tables.COHORT
ROI_UM = 300.0
MIN_NUCLEI = 40
INVERT_TOL_PX = 1e-3
INVERT_MAX_ITER = 200
READ_PAD_PX = 8
OVERVIEW_MAX_SIDE = 4096  # a reduced pyramid level this small is read whole for the overview
NO_LEVEL = "nucleus density (no pyramid level)"
READ_GROWTH = 1.5  # a resampled panel may read a box this much larger than the field
REF_RGB = np.array([0.0, 1.0, 1.0])  # cyan
MOV_RGB = np.array([1.0, 0.0, 1.0])  # magenta
REF_LINE, MOV_LINE = "#00FFFF", "#FF00FF"
SCALE_BARS_UM = (1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000)

BEFORE = "before"
WHOLE = "after the whole transform"
RIGID = "after rigid"
RIGID_MICRO = "after rigid (+ micro-rigid)"
NO_RIGID = "no rigid stage"

ROI_RULE = (
    "windows of roi_px on a grid of step roi_px // 2 over the reference; a window counts "
    "with at least {min_nuclei} reference nuclei; per window the medians d_initial, "
    "d_rigid, d_final of the distance from each reference nucleus to the nearest moving "
    "one in that stage's frame (capped at the gate, um); among windows with "
    "d_initial > d_rigid > d_final and d_final <= the median d_final over the counted "
    "windows, the largest d_rigid - d_final, ties by the larger d_initial - d_rigid; "
    "chosen once per case on the STARE best row"
)
NO_MONOTONE = (
    "no window is monotone (d_initial > d_rigid > d_final): the largest d_rigid - d_final "
    "over all counted windows; "
)
NO_WINDOW = (
    "no window holds {min_nuclei} reference nuclei: the window with the most reference "
    "nuclei (the field at (0, 0) when there is none); "
)
CASE_RULE = (
    "the case whose chosen window has the largest d_rigid - d_final among the cases with "
    "a monotone window; with none monotone, the largest over all cases"
)
CASE_NAMED = "the case named by --case-id"


# ── reading ──────────────────────────────────────────────────────────────────
def _level(tif):
    series = tif.series[0]
    levels = getattr(series, "levels", None) or [series]
    return levels[0]


def _levels(tif):
    series = tif.series[0]
    return list(getattr(series, "levels", None) or [series])


def overview_level(path, max_side=OVERVIEW_MAX_SIDE):
    """The index of the largest REDUCED pyramid level (SubIFD / series level >= 1) of a
    TIFF whose long side is at most ``max_side``, or None. Level 0 never counts."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        levels = _levels(tif)
        best = None
        for k, level in enumerate(levels[1:], start=1):
            axes, shape = str(level.axes), tuple(int(v) for v in level.shape)
            if max(_hw(axes, shape)) <= max_side and (
                best is None or max(_hw(axes, shape)) > best[1]
            ):
                best = (k, max(_hw(axes, shape)))
        return best[0] if best else None


def read_level(path, channel, k):
    """One channel of pyramid level ``k`` of a TIFF, whole, as float64 (an ``S`` axis is
    averaged). Called only for a reduced level within ``OVERVIEW_MAX_SIDE``."""
    import tifffile
    import zarr

    with tifffile.TiffFile(str(path)) as tif:
        level = _levels(tif)[int(k)]
        axes = str(level.axes)
        store = tif.series[0].aszarr(level=int(k))
        try:
            arr = zarr.open(store, mode="r")
            idx = tuple(
                int(channel) if ax == "C" else slice(None) if ax in "YXS" else 0
                for ax in axes
            )
            block = np.asarray(arr[idx], dtype=np.float64)
        finally:
            store.close()
    if "S" in axes:
        block = block.mean(axis=-1)
    return block


def level_hw(path, k):
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        level = _levels(tif)[int(k)]
        return _hw(str(level.axes), tuple(int(v) for v in level.shape))


def image_info(path):
    """``(axes, shape)`` of the full-resolution level of a TIFF, no pixel read."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        level = _level(tif)
        return str(level.axes), tuple(int(v) for v in level.shape)


def _hw(axes, shape):
    return shape[axes.index("Y")], shape[axes.index("X")]


def n_channels(axes, shape):
    return shape[axes.index("C")] if "C" in axes else 1


def read_region(path, channel, y0, y1, x0, x1):
    """Rows ``y0:y1`` and columns ``x0:x1`` of one channel of a TIFF's full-resolution
    level, as float64: only the tiles the box touches are decoded (``aszarr``). An RGB
    image (an ``S`` axis) is read as the mean of its samples."""
    import tifffile
    import zarr

    with tifffile.TiffFile(str(path)) as tif:
        level = _level(tif)
        axes = str(level.axes)
        store = tif.series[0].aszarr(level=0)
        try:
            arr = zarr.open(store, mode="r")
            idx = []
            for ax in axes:
                if ax == "Y":
                    idx.append(slice(int(y0), int(y1)))
                elif ax == "X":
                    idx.append(slice(int(x0), int(x1)))
                elif ax == "C":
                    idx.append(int(channel))
                elif ax == "S":
                    idx.append(slice(None))
                else:
                    idx.append(0)
            block = np.asarray(arr[tuple(idx)], dtype=np.float64)
        finally:
            store.close()
    if "S" in axes:
        block = block.mean(axis=-1)
    return block


def read_limit(roi_px):
    """The largest box side a panel of a field of ``roi_px`` may read."""
    return int(READ_GROWTH * int(roi_px)) + 2 * READ_PAD_PX


def _channel(path, case, side):
    """The nuclear channel of a case's slide: its only one, else the case's index."""
    axes, shape = image_info(path)
    return 0 if n_channels(axes, shape) == 1 else int(case.extra.get(f"{side}_channel", 0))


def _placed(path, channel, x0, y0, s, origin=(0.0, 0.0)):
    """The field ``[x0, x0 + s) x [y0, y0 + s)`` of the reference grid cut from an image
    whose first pixel sits at ``origin`` there; what the image does not cover is 0.
    Reads at most the field itself."""
    axes, shape = image_info(path)
    h, w = _hw(axes, shape)
    ox, oy = (int(round(float(v))) for v in origin)
    ix0, iy0 = x0 - ox, y0 - oy
    cx0, cy0 = max(ix0, 0), max(iy0, 0)
    cx1, cy1 = min(ix0 + s, w), min(iy0 + s, h)
    out = np.zeros((s, s), dtype=np.float64)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0 - iy0 : cy1 - iy0, cx0 - ix0 : cx1 - ix0] = read_region(
            path, channel, cy0, cy1, cx0, cx1
        )
    return out


def _resampled(path, channel, src_xy, s):
    """The ``s x s`` field whose pixel ``k`` (row-major) comes from ``src_xy[k]`` of the
    image (bilinear, 0 outside it); ``(plane, None)`` or ``(None, why)`` when the box it
    needs is larger than ``read_limit(s)``."""
    from scipy.ndimage import map_coordinates

    axes, shape = image_info(path)
    h, w = _hw(axes, shape)
    out = np.zeros(s * s, dtype=np.float64)
    ok = np.isfinite(src_xy).all(axis=1)
    if not ok.any():
        return out.reshape(s, s), None
    xs, ys = src_xy[ok, 0], src_xy[ok, 1]
    bx0 = max(int(math.floor(xs.min())) - READ_PAD_PX, 0)
    by0 = max(int(math.floor(ys.min())) - READ_PAD_PX, 0)
    bx1 = min(int(math.ceil(xs.max())) + READ_PAD_PX + 1, w)
    by1 = min(int(math.ceil(ys.max())) + READ_PAD_PX + 1, h)
    limit = read_limit(s)
    if bx1 - bx0 > limit or by1 - by0 > limit:
        return None, (
            f"the map spreads the field over {bx1 - bx0} x {by1 - by0} px of the moving "
            f"slide, more than the read limit {limit} px: not read"
        )
    if bx1 <= bx0 or by1 <= by0:
        return out.reshape(s, s), None
    block = read_region(path, channel, by0, by1, bx0, bx1)
    out[ok] = map_coordinates(block, [ys - by0, xs - bx0], order=1, mode="constant", cval=0.0)
    return out.reshape(s, s), None


# ── the rows ─────────────────────────────────────────────────────────────────
def _run_of_variant(variant):
    """The run whose result names include ``variant`` (paused runs too), or None."""
    for run in variants.RUNS:
        if variant in list(variants.variants_of(run)):
            return run
    return None


def rigid_of(row):
    """The rigid result name of ``row``'s run (module docstring), or None."""
    run = _run_of_variant(row)
    if run is None:
        return None
    names = list(variants.variants_of(run))
    if len(names) < 2:
        return None
    first = names[0]
    return first if tables.stage_of(first) in ("rigid", "micro_rigid") else None


def rigid_title(row):
    rigid = rigid_of(row)
    if rigid is None:
        return NO_RIGID
    return RIGID_MICRO if tables.stage_of(rigid) == "micro_rigid" else RIGID


def stage_rows_of(row):
    return {"initial": "initial", "rigid": rigid_of(row), "final": row}


def _record(out, name, case):
    """``(run.json, warped, why)``: ``warped`` None (with ``why``) when absent or failed."""
    d = result_dir(out, name, case)
    run = {}
    if (d / "run.json").is_file():
        try:
            run = json.loads((d / "run.json").read_text())
        except ValueError:
            return {}, None, f"run.json is not readable: {d / 'run.json'}"
    if run and not run.get("ok", False):
        return run, None, f"the run failed ({run.get('cause') or 'error'}): {d}"
    if not (d / "warped.npz").is_file():
        return run, None, f"no warped.npz: {d / 'warped.npz'}"
    with np.load(d / "warped.npz") as z:
        warped = {k: z[k] for k in z.files}
    return run, warped, ""


def _direction(warped, run):
    if "direction" in warped:
        return str(np.asarray(warped["direction"]).item())
    return str(run.get("direction") or MOV_TO_REF)


class Sets:
    """A case's native outlines both sides, their centroids and its gate (px)."""

    def __init__(self, case):
        points = load_points(case)
        self.case = case
        self.ref, self.mov = polys(points, "ref"), polys(points, "mov")
        if self.ref is None or self.mov is None:
            raise ValueError(f"{case.case_id}: the case has no outlines on both sides")
        geo = pairs.Geometry()
        self.gate = float(pairs.gate_px(self.ref, self.mov, geo)[0])
        _, self.ref_c = pairs.area_centroid(self.ref, geo)
        _, self.mov_c = pairs.area_centroid(self.mov, geo)
        self.um = float(case.pixel_size_um)


def _nearest(query, target, cap):
    """Per query point, the distance to the nearest target point, capped at ``cap``; a
    non-finite query point is at the cap."""
    from scipy.spatial import cKDTree

    query = np.asarray(query, dtype=float).reshape(-1, 2)
    target = np.asarray(target, dtype=float).reshape(-1, 2)
    out = np.full(len(query), float(cap))
    tf = target[np.isfinite(target).all(axis=1)]
    qf = np.isfinite(query).all(axis=1)
    if len(tf) and qf.any():
        dist, _ = cKDTree(tf).query(query[qf], k=1, distance_upper_bound=float(cap))
        out[qf] = np.minimum(dist, float(cap))
    return out


def _moved_set(sets, warped, direction):
    """The outlines a variant moved: the moving ones (``mov_to_ref``) or the reference
    ones (``ref_to_mov``), regrouped by the case's ring offsets; None when the vertex
    count does not fit the case."""
    src = sets.mov if direction == MOV_TO_REF else sets.ref
    xy = np.asarray(warped.get("cells_xy", np.empty((0, 2))), dtype=float).reshape(-1, 2)
    if len(xy) != len(src.xy):
        return None
    return pairs.with_xy(src, xy)


def residuals(case, out, row, stage_rows=None, sets=None):
    """``{stage: distance per reference nucleus (um)}`` for ``initial``, ``rigid`` and
    ``final`` (module docstring). ``stage_rows`` maps a stage to the result name read
    for it (default ``stage_rows_of(row)``); a stage with no result name, or whose
    result is missing, is all NaN."""
    sets = sets or Sets(case)
    stage_rows = stage_rows or stage_rows_of(row)
    n = len(sets.ref_c)
    got = {}
    for stage in STAGES:
        name = stage_rows.get(stage)
        d = np.full(n, np.nan)
        if name == "initial":
            d = _nearest(sets.ref_c, sets.mov_c, sets.gate)
        elif name:
            run, warped, _ = _record(out, name, case)
            if warped is not None:
                direction = _direction(warped, run)
                moved = _moved_set(sets, warped, direction)
                if moved is not None:
                    _, cent = pairs.area_centroid(moved)
                    if direction == REF_TO_MOV:
                        # moved REFERENCE nuclei, in the moving frame, against native moving
                        d = _nearest(cent, sets.mov_c, sets.gate)
                    else:
                        # moved moving nuclei, in the reference frame, against native reference
                        d = _nearest(sets.ref_c, cent, sets.gate)
        got[stage] = d * sets.um
    return got


# ── the field ────────────────────────────────────────────────────────────────
def _nanmedian(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else math.nan


def _gain(w):
    g = w["d_rigid_um"] - w["d_final_um"]
    return g if math.isfinite(g) else -math.inf


def _step_gain(w):
    g = w["d_initial_um"] - w["d_rigid_um"]
    return g if math.isfinite(g) else -math.inf


def _monotone(di, dr, df):
    return bool(di > dr > df)


def pick_roi(ref_xy, d, roi_px, step_px=None, min_nuclei=MIN_NUCLEI):
    """The zoom field of one case (rule ``ROI_RULE``): ``roi_x``, ``roi_y``, ``roi_px``,
    ``d_initial_um``, ``d_rigid_um``, ``d_final_um``, ``monotone``, ``rule``,
    ``n_nuclei``. ``ref_xy`` are the reference centroids (px), ``d`` the ``residuals``."""
    roi = int(roi_px)
    step = int(step_px or max(1, roi // 2))
    xy = np.asarray(ref_xy, dtype=float).reshape(-1, 2)
    # windows start at 0: a centroid left of or above the slide is in none
    fin = np.isfinite(xy).all(axis=1) & (np.nan_to_num(xy, nan=-1.0) >= 0).all(axis=1)
    idx_all = np.flatnonzero(fin)
    rule_head = ROI_RULE.format(min_nuclei=min_nuclei)
    if not idx_all.size:
        return _field(0, 0, roi, {}, 0, False,
                      NO_WINDOW.format(min_nuclei=min_nuclei) + rule_head)
    pts = xy[idx_all]
    xmax, ymax = pts.max(axis=0)
    # every window lies inside the nuclei's extent; the last one is flush with its edge
    xs, ys = (_starts(v, roi, step) for v in (xmax, ymax))
    # nuclei by step cell, so a window only looks at the few cells it covers
    cx = np.floor(pts[:, 0] / step).astype(np.int64)
    cy = np.floor(pts[:, 1] / step).astype(np.int64)
    order = np.lexsort((cx, cy))
    keys = cy[order] * (len(xs) + roi // step + 4) + cx[order]
    uniq, start = np.unique(keys, return_index=True)
    bounds = dict(zip(uniq.tolist(), zip(start.tolist(), [*start[1:].tolist(), len(order)])))
    span = -(-roi // step) + 1
    stride = len(xs) + roi // step + 4
    stages = {s: np.asarray(d[s], dtype=float) for s in STAGES}
    windows = []
    for y0 in ys:
        for x0 in xs:
            i0, j0 = int(x0 // step), int(y0 // step)
            got = [
                order[a:b]
                for j in range(j0, j0 + span)
                for i in range(i0, i0 + span)
                if (j * stride + i) in bounds
                for a, b in [bounds[j * stride + i]]
            ]
            if not got:
                continue
            cand = np.concatenate(got)
            p = pts[cand]
            inside = cand[
                (p[:, 0] >= x0) & (p[:, 0] < x0 + roi) & (p[:, 1] >= y0) & (p[:, 1] < y0 + roi)
            ]
            if not inside.size:
                continue
            nuc = idx_all[inside]
            med = {s: _nanmedian(stages[s][nuc]) for s in STAGES}
            windows.append(_field(int(x0), int(y0), roi, med, int(inside.size), None, ""))
    counted = [w for w in windows if w["n_nuclei"] >= min_nuclei]
    if not counted:
        if not windows:
            return _field(0, 0, roi, {}, 0, False,
                          NO_WINDOW.format(min_nuclei=min_nuclei) + rule_head)
        best = dict(max(windows, key=lambda w: (w["n_nuclei"], -w["roi_y"], -w["roi_x"])))
        best["monotone"] = False
        best["rule"] = NO_WINDOW.format(min_nuclei=min_nuclei) + rule_head
        return best
    windows = counted
    final_med = _nanmedian([w["d_final_um"] for w in windows])
    mono = [
        w for w in windows
        if _monotone(w["d_initial_um"], w["d_rigid_um"], w["d_final_um"])
        and w["d_final_um"] <= final_med
    ]
    pool = mono or windows
    best = max(pool, key=lambda w: (_gain(w), _step_gain(w), -w["roi_y"], -w["roi_x"]))
    best = dict(best)
    best["monotone"] = bool(mono)
    best["rule"] = rule_head if mono else NO_MONOTONE + rule_head
    return best


def _starts(extent, roi, step):
    last = max(int(math.floor(extent)) + 1 - roi, 0)
    return np.unique(np.append(np.arange(0, last + 1, step), last)).astype(np.int64)


def _field(x0, y0, roi, med, n, monotone, rule):
    return {
        "roi_x": int(x0),
        "roi_y": int(y0),
        "roi_px": int(roi),
        "d_initial_um": med.get("initial", math.nan),
        "d_rigid_um": med.get("rigid", math.nan),
        "d_final_um": med.get("final", math.nan),
        "monotone": monotone,
        "rule": rule,
        "n_nuclei": n,
    }


def medians_in(ref_xy, d, roi):
    """A row's own three medians in a given field."""
    xy = np.asarray(ref_xy, dtype=float).reshape(-1, 2)
    x0, y0, s = roi["roi_x"], roi["roi_y"], roi["roi_px"]
    inside = (xy[:, 0] >= x0) & (xy[:, 0] < x0 + s) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + s)
    return {s_: _nanmedian(np.asarray(d[s_], dtype=float)[inside]) for s_ in STAGES}


def pick_case(rois, case_id=None):
    """``(case_id, rule)``: the case drawn (rule ``CASE_RULE``) among ``{case_id: the
    field pick_roi chose on its STARE best row}``."""
    if case_id is not None:
        return str(case_id), CASE_NAMED
    if not rois:
        return None, CASE_RULE
    pool = {k: r for k, r in rois.items() if r.get("monotone")} or rois
    best = max(sorted(pool), key=lambda k: (_gain(pool[k]), _step_gain(pool[k])))
    return best, CASE_RULE


# ── the lattice ──────────────────────────────────────────────────────────────
class Lattice:
    """A variant's map, sampled on the driver's lattice (``warped.npz: grid``,
    ``grid_shape``, ``grid_step``; ``methods.jacobian_grid``), interpolated bilinearly and
    extrapolated linearly past its edge: exact for an affine map."""

    def __init__(self, warped):
        rows, cols = (int(v) for v in np.ravel(warped["grid_shape"])[:2])
        if rows < 2 or cols < 2:
            raise ValueError(f"a lattice of {rows} x {cols} points cannot be interpolated")
        self.rows, self.cols = rows, cols
        self.step = float(np.ravel(warped["grid_step"])[0])
        self.grid = np.asarray(warped["grid"], dtype=np.float64).reshape(rows, cols, 2)

    def __call__(self, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        ok = np.isfinite(xy).all(axis=1)
        fx = np.where(ok, (xy[:, 0] - self.step / 2.0) / self.step, 0.0)
        fy = np.where(ok, (xy[:, 1] - self.step / 2.0) / self.step, 0.0)
        i = np.clip(np.floor(fx), 0, self.cols - 2).astype(np.int64)
        j = np.clip(np.floor(fy), 0, self.rows - 2).astype(np.int64)
        tx, ty = (fx - i)[:, None], (fy - j)[:, None]
        g = self.grid
        out = (
            (1 - ty) * ((1 - tx) * g[j, i] + tx * g[j, i + 1])
            + ty * ((1 - tx) * g[j + 1, i] + tx * g[j + 1, i + 1])
        )
        out[~ok] = np.nan
        return out

    def invert(self, target, tol=INVERT_TOL_PX, max_iter=INVERT_MAX_ITER):
        """``m`` with ``self(m) == target`` to ``tol`` px, by fixed-point iteration
        ``m <- m + (target - self(m))``; NaN where it does not converge."""
        t = np.asarray(target, dtype=np.float64).reshape(-1, 2)
        m = 2.0 * t - self(t)
        bad = ~np.isfinite(m).all(axis=1)
        m[bad] = t[bad]
        for _ in range(max_iter):
            err = t - self(m)
            step = np.where(np.isfinite(err), err, 0.0)
            m = m + step
            if not (np.hypot(step[:, 0], step[:, 1]) >= tol).any():
                break
        err = t - self(m)
        m[~(np.hypot(err[:, 0], err[:, 1]) < tol)] = np.nan
        return m


def _field_points(x0, y0, s):
    ys, xs = np.mgrid[y0 : y0 + s, x0 : x0 + s]
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


# ── the outlines ─────────────────────────────────────────────────────────────
def _rings(ps, xy=None):
    """``(rings, ring means)`` of the outer rings of a polygon set (``xy`` replaces its
    vertices)."""
    ps = pairs.as_polyset(ps)
    xy = np.asarray(ps.xy if xy is None else xy, dtype=float).reshape(-1, 2)
    off = np.asarray(ps.ring_off, dtype=np.int64)
    keep = ~np.asarray(ps.ring_hole, dtype=bool)
    rings = [xy[off[r] : off[r + 1]] for r in range(len(off) - 1) if keep[r]]
    means = np.array([r.mean(axis=0) if len(r) else (np.nan, np.nan) for r in rings])
    return rings, means.reshape(-1, 2)


def _in_field(rings, means, x0, y0, s, margin=0.0):
    m = means
    hit = (
        (m[:, 0] >= x0 - margin) & (m[:, 0] < x0 + s + margin)
        & (m[:, 1] >= y0 - margin) & (m[:, 1] < y0 + s + margin)
    ) if len(m) else np.zeros(0, bool)
    return [rings[k] - (x0, y0) for k in np.flatnonzero(hit) if np.isfinite(rings[k]).all()]


def _moving_outlines_on_ref(sets, warped, direction, x0, y0, s):
    """The moving outlines of the field on the reference grid, through one variant."""
    if direction == MOV_TO_REF:
        moved = _moved_set(sets, warped, direction)
        if moved is None:
            return []
        rings, means = _rings(moved)
        return _in_field(rings, means, x0, y0, s)
    lat = Lattice(warped)
    border = _field_points(x0, y0, s)[:: max(1, s // 8)]
    where = lat(border)
    where = where[np.isfinite(where).all(axis=1)]
    if not len(where):
        return []
    rings, means = _rings(sets.mov)
    (lx, ly), (hx, hy) = where.min(axis=0), where.max(axis=0)
    margin = 0.25 * s
    near = (
        (means[:, 0] >= lx - margin) & (means[:, 0] <= hx + margin)
        & (means[:, 1] >= ly - margin) & (means[:, 1] <= hy + margin)
    )
    picked = [rings[k] for k in np.flatnonzero(near)]
    if not picked:
        return []
    flat = lat.invert(np.concatenate(picked))
    cuts = np.cumsum([len(r) for r in picked])[:-1]
    back = np.split(flat, cuts)
    return _in_field(back, np.array([b.mean(axis=0) for b in back]), x0, y0, s)


# ── one row's panels ─────────────────────────────────────────────────────────
def _panel(status, image=None, rings=None, d_um=math.nan, stage_row="", direction=""):
    return {"status": status, "image": image, "rings": rings or [], "d_um": d_um,
            "stage_row": stage_row, "direction": direction}


def _rigid_panel(case, sets, out, rigid, roi, mov_path, mov_ch):
    x0, y0, s = roi["roi_x"], roi["roi_y"], roi["roi_px"]
    run, warped, why = _record(out, rigid, case)
    if warped is None:
        return _panel(why, stage_row=rigid)
    direction = _direction(warped, run)
    try:
        lat = Lattice(warped)
    except (KeyError, ValueError) as exc:
        return _panel(f"no lattice in warped.npz: {exc}", stage_row=rigid)
    pts = _field_points(x0, y0, s)
    src = lat(pts) if direction == REF_TO_MOV else lat.invert(pts)
    image, why = _resampled(mov_path, mov_ch, src, s)
    if image is None:
        return _panel(why, stage_row=rigid, direction=direction)
    rings = _moving_outlines_on_ref(sets, warped, direction, x0, y0, s)
    return _panel("drawn", image, rings, stage_row=rigid, direction=direction)


def _whole_panel(case, sets, out, row, roi):
    x0, y0, s = roi["roi_x"], roi["roi_y"], roi["roi_px"]
    run, warped, why = _record(out, row, case)
    native = (run or {}).get("native") or {}
    files = native.get("files") if isinstance(native, dict) else None
    d = result_dir(out, row, case)
    if not files:
        return _panel(f"no image written: run.json native.files is absent ({d / 'run.json'})",
                      stage_row=row)
    path = Path(native.get("nuclear_file") or files[0])
    if not path.is_file():
        return _panel(f"no image written: {path} is not on disk", stage_row=row)
    from ..tissue import native_channel

    axes, shape = image_info(path)
    channel = native_channel(native, case, n_channels(axes, shape))
    image = _placed(path, channel, x0, y0, s, native.get("origin_xy") or (0.0, 0.0))
    rings, direction = [], ""
    if warped is not None:
        direction = _direction(warped, run)
        try:
            rings = _moving_outlines_on_ref(sets, warped, direction, x0, y0, s)
        except (KeyError, ValueError):
            rings = []
    status = "drawn" if warped is not None else f"drawn without outlines: {why}"
    return _panel(status, image, rings, stage_row=row, direction=direction)


def row_panels(case, sets, out, row, roi, d=None):
    """``{stage: panel}`` of one row in the field (``_panel`` dicts)."""
    x0, y0, s = roi["roi_x"], roi["roi_y"], roi["roi_px"]
    mov_path, ref_path = Path(case.mov_nuclear), Path(case.ref_nuclear)
    panels = {}
    if mov_path.is_file():
        mov_ch = _channel(mov_path, case, "mov")
        rings, means = _rings(sets.mov)
        panels["initial"] = _panel("drawn", _placed(mov_path, mov_ch, x0, y0, s),
                                   _in_field(rings, means, x0, y0, s), stage_row="initial")
    else:
        mov_ch = 0
        panels["initial"] = _panel(f"no moving slide: {mov_path}", stage_row="initial")
    rigid = rigid_of(row)
    if rigid is None:
        panels["rigid"] = _panel(NO_RIGID)
    elif not mov_path.is_file():
        panels["rigid"] = _panel(f"no moving slide: {mov_path}", stage_row=rigid)
    else:
        panels["rigid"] = _rigid_panel(case, sets, out, rigid, roi, mov_path, mov_ch)
    panels["final"] = _whole_panel(case, sets, out, row, roi)
    if d is not None:
        med = medians_in(sets.ref_c, d, roi)
        for stage in STAGES:
            panels[stage]["d_um"] = med[stage]
    ref = None
    if ref_path.is_file():
        ref = _placed(ref_path, _channel(ref_path, case, "ref"), x0, y0, s)
    rings, means = _rings(sets.ref)
    return panels, ref, _in_field(rings, means, x0, y0, s)


# ── drawing ──────────────────────────────────────────────────────────────────
def _norm(a):
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    v = a[np.isfinite(a) & (a > 0)]
    if not v.size:
        return np.zeros_like(a)
    lo, hi = np.percentile(v, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _rgb(ref, mov, shape):
    out = np.zeros((*shape, 3))
    for plane, color in ((_norm(ref), REF_RGB), (_norm(mov), MOV_RGB)):
        if plane is not None:
            out += plane[..., None] * color
    return np.clip(out, 0.0, 1.0)


def _scale_bar(ax, s, um_per_px):
    want = s * um_per_px / 5.0
    bar = min(SCALE_BARS_UM, key=lambda b: abs(b - want))
    length = bar / um_per_px
    x, y = 0.06 * s, 0.92 * s
    ax.plot([x, x + length], [y, y], color="white", lw=1.2, solid_capstyle="butt")
    ax.text(x + length / 2.0, y - 0.03 * s, f"{bar:g} µm", color="white",
            ha="center", va="bottom", fontsize=ps.TICK_PT)


def _wrap(text, width=34):
    import textwrap

    return "\n".join(textwrap.wrap(str(text), width, break_long_words=True))


def _draw_panel(ax, title, panel, ref, ref_rings, s, um_per_px):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=ps.TICK_PT)
    if panel["image"] is None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, _wrap(panel["status"]), ha="center", va="center",
                fontsize=ps.TICK_PT - 1, color=ps.GREY, transform=ax.transAxes)
        return
    ax.imshow(_rgb(ref, panel["image"], (s, s)), interpolation="nearest")
    if ref_rings:
        ax.add_collection(LineCollection(ref_rings, colors=REF_LINE, linewidths=0.3))
    if panel["rings"]:
        ax.add_collection(LineCollection(panel["rings"], colors=MOV_LINE, linewidths=0.3))
    ax.set_xlim(-0.5, s - 0.5)
    ax.set_ylim(s - 0.5, -0.5)
    _scale_bar(ax, s, um_per_px)
    if math.isfinite(panel["d_um"]):
        ax.text(0.97 * s, 0.04 * s, f"median {panel['d_um']:.2f} µm", color="white",
                ha="right", va="top", fontsize=ps.TICK_PT - 1)


def _overview(ax, sets, roi, title="whole reference"):
    """The whole reference with the field boxed: its own reduced pyramid level when it
    has one (``overview_level``), else nucleus density. Returns what was drawn."""
    case = sets.case
    path = Path(case.ref_nuclear)
    k = overview_level(path) if path.is_file() else None
    if k is None:
        _density_overview(ax, sets, roi, f"{title}: {NO_LEVEL}")
        return NO_LEVEL
    lh, lw = level_hw(path, k)
    h0, w0 = _hw(*image_info(path))
    fx, fy = w0 / lw, h0 / lh
    plane = read_level(path, _channel(path, case, "ref"), k)
    ax.imshow(_norm(plane)[..., None] * REF_RGB, interpolation="nearest")
    ax.add_patch(Rectangle((roi["roi_x"] / fx - 0.5, roi["roi_y"] / fy - 0.5),
                           roi["roi_px"] / fx, roi["roi_px"] / fy,
                           fill=False, edgecolor="#F0E442", linewidth=0.8))
    status = f"pyramid level {k} ({lw} x {lh} px)"
    _overview_frame(ax, f"{title}, {status}", lw, lh, float(case.pixel_size_um) * fx)
    return status


def _overview_frame(ax, title, w, h, um):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(_wrap(title, 44), fontsize=ps.TICK_PT)
    bar = min(SCALE_BARS_UM, key=lambda b: abs(b - w * um / 4.0))
    ax.plot([0.05 * w, 0.05 * w + bar / um], [0.93 * h, 0.93 * h], color="white", lw=1.2)
    ax.text(0.05 * w + bar / um / 2.0, 0.90 * h, f"{bar:g} \u00b5m", color="white",
            ha="center", va="bottom", fontsize=ps.TICK_PT - 1,
            bbox={"facecolor": "black", "edgecolor": "none", "pad": 0.6, "alpha": 0.7})


def _density_overview(ax, sets, roi, title):
    case = sets.case
    h, w = (int(v) for v in (case.ref_hw or (0, 0)))
    pts = np.concatenate([sets.ref_c, sets.mov_c])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not (h and w) and len(pts):
        w, h = (int(math.ceil(v)) + 1 for v in pts.max(axis=0))
    h, w = max(h, 1), max(w, 1)
    bins = 64
    sy, sx = (bins, max(1, int(bins * w / h))) if h >= w else (max(1, int(bins * h / w)), bins)

    def density(c):
        c = c[np.isfinite(c).all(axis=1)]
        hist, _, _ = np.histogram2d(c[:, 1], c[:, 0], bins=(sy, sx), range=((0, h), (0, w)))
        return hist

    def scaled(hist):
        top = np.percentile(hist[hist > 0], 99.0) if (hist > 0).any() else 1.0
        return np.clip(hist / max(top, 1e-9), 0.0, 1.0)

    img = np.clip(scaled(density(sets.ref_c))[..., None] * REF_RGB
                  + scaled(density(sets.mov_c))[..., None] * MOV_RGB, 0.0, 1.0)
    ax.imshow(img, extent=(0, w, h, 0), interpolation="nearest")
    ax.add_patch(Rectangle((roi["roi_x"], roi["roi_y"]), roi["roi_px"], roi["roi_px"],
                           fill=False, edgecolor="#F0E442", linewidth=0.8))
    _overview_frame(ax, title, w, h, float(case.pixel_size_um))


def _title(stage, row):
    return {"initial": BEFORE, "rigid": rigid_title(row), "final": WHOLE}[stage]


def _source(figure, family, row, case, roi, stage, panel, title):
    return {
        "figure": figure, "family": family, "row": row, "case_id": case.case_id,
        "panel": title, "stage": stage, "stage_row": panel.get("stage_row") or "",
        "direction": panel.get("direction") or "", "roi_x": roi["roi_x"],
        "roi_y": roi["roi_y"], "roi_px": roi["roi_px"],
        "d_median_um": panel.get("d_um", math.nan), "status": panel["status"],
    }


SOURCE_COLS = ("figure", "family", "row", "case_id", "panel", "stage", "stage_row",
               "direction", "roi_x", "roi_y", "roi_px", "d_median_um", "status")


def _row_label(family, row):
    return f"{plots._label(family)}\n{row or 'no best row'}"


def _draw_rows(name, drawn, sets, roi, fig_dir, overview=True):
    """One figure of ``drawn`` = ``[(family, row, panels, ref, ref_rings)]``."""
    plots._style()
    ncols = 4 if overview else 3
    n = len(drawn)
    height = min(ps.MAX_HEIGHT_MM, 180.0 / ncols * n * 1.0 + 8.0)
    fig, axes = ps.figure(width="double", height_mm=height, nrows=n, ncols=ncols)
    s, um = roi["roi_px"], float(sets.case.pixel_size_um)
    data = []
    for k, (family, row, panels, ref, ref_rings) in enumerate(drawn):
        line = axes[k * ncols : (k + 1) * ncols]
        if overview:
            if k == 0:
                drew = _overview(line[0], sets, roi)
                data.append(_source(name, family, row, sets.case, roi, "overview",
                                    _panel(drew), "overview"))
            else:
                line[0].axis("off")
            line = line[1:]
        for ax, stage in zip(line, STAGES):
            title = _title(stage, row) if row else (BEFORE, RIGID, WHOLE)[STAGES.index(stage)]
            _draw_panel(ax, title, panels[stage], ref, ref_rings, s, um)
            data.append(_source(name, family, row, sets.case, roi, stage, panels[stage], title))
        line[0].set_ylabel(_row_label(family, row), fontsize=ps.TICK_PT)
    files = plots._save(fig, fig_dir, name, plots._Table(SOURCE_COLS, data))
    return plots._result(files, {"rows": n})


def _draw_before(name, sets, roi, panels, ref, ref_rings, fig_dir):
    plots._style()
    fig, axes = ps.figure(width="onehalf", height_mm=62.0, nrows=1, ncols=2)
    drew = _overview(axes[0], sets, roi, "whole slide, field boxed")
    _draw_panel(axes[1], BEFORE, panels["initial"], ref, ref_rings, roi["roi_px"],
                float(sets.case.pixel_size_um))
    data = [
        _source(name, STARE_FAMILY, "", sets.case, roi, "overview", _panel(drew), "overview"),
        _source(name, STARE_FAMILY, "", sets.case, roi, "initial", panels["initial"], BEFORE),
    ]
    ps.panel_labels(axes)
    files = plots._save(fig, fig_dir, name, plots._Table(SOURCE_COLS, data))
    return plots._result(files, {"panels": 2})


# ── the whole job ────────────────────────────────────────────────────────────
def _roi_row(case, family, row, roi, med, rule):
    di, dr, df = med["initial"], med["rigid"], med["final"]
    mono = _monotone(di, dr, df) if all(math.isfinite(v) for v in (di, dr, df)) else None
    from ..score import panel_of

    return schema.empty_row(
        "overlay_rois", source=tables.SOURCE, dataset=case.dataset, patient_id=case.group,
        panel=panel_of(case), case_id=case.case_id, family=family, row=row,
        roi_x=roi["roi_x"], roi_y=roi["roi_y"], roi_px=roi["roi_px"], d_initial_um=di,
        d_rigid_um=dr, d_final_um=df, monotone=mono, rule=rule,
    )


def _placeholders(fig_dir, missing, paper_dir):
    schema.write(paper_dir / "overlay_rois.csv", "overlay_rois", [])
    return {name: plots._placeholder(fig_dir, name, missing) for name in OVERLAYS}


def build(cases_root, out, roi_um=ROI_UM, case_id=None, dataset=COHORT):
    """Write ``OVERLAYS`` and ``overlay_rois.csv``; ``{figure: result}`` as ``plots``."""
    paper_dir = Path(out) / "paper"
    fig_dir = paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    best_path = paper_dir / "method_best.csv"
    best = plots._read(best_path)
    if best is None:
        return _placeholders(fig_dir, [("method_best", best_path)], paper_dir)
    best_rows = {r["family"]: r.get("best_row") for r in best
                 if (r.get("dataset") or dataset) == dataset}
    stare_row = best_rows.get(STARE_FAMILY)
    if not stare_row:
        return _placeholders(fig_dir, [("method_best: the stare family's best_row", best_path)],
                             paper_dir)
    found = list_cases(cases_root, dataset)
    if case_id is not None:
        found = [c for c in found if str(c.case_id) == str(case_id)]
    rois, sets_of, d_of = {}, {}, {}
    for case in found:
        _, warped, _ = _record(out, stare_row, case)
        if warped is None:
            continue
        try:
            sets = Sets(case)
        except ValueError:
            continue
        d = residuals(case, out, stare_row, sets=sets)
        roi_px = max(8, int(round(float(roi_um) / float(case.pixel_size_um))))
        rois[case.case_id] = pick_roi(sets.ref_c, d, roi_px, roi_px // 2)
        sets_of[case.case_id], d_of[case.case_id] = sets, d
    if not rois:
        where = Path(out) / stare_row / dataset / (str(case_id) if case_id else "*") / "warped.npz"
        return _placeholders(fig_dir, [(f"a {stare_row} result", where)], paper_dir)
    chosen, case_rule = pick_case(rois, case_id)
    sets, roi = sets_of[chosen], rois[chosen]
    case = sets.case
    rule = f"{roi['rule']} | case: {case_rule}"
    roi_rows = [
        _roi_row(sets_of[k].case, STARE_FAMILY, stare_row, r,
                 {"initial": r["d_initial_um"], "rigid": r["d_rigid_um"],
                  "final": r["d_final_um"]},
                 f"candidate case (not drawn): {r['rule']}")
        for k, r in sorted(rois.items()) if k != chosen
    ]
    drawn = []
    for family in FIG3A_FAMILIES:
        row = best_rows.get(family)
        if not row:
            missing = _panel(f"no best row for {family} in {best_path}")
            drawn.append((family, None, {s: dict(missing) for s in STAGES}, None, []))
            continue
        d = d_of[chosen] if row == stare_row else residuals(case, out, row, sets=sets)
        panels, ref, ref_rings = row_panels(case, sets, out, row, roi, d)
        drawn.append((family, row, panels, ref, ref_rings))
        roi_rows.append(_roi_row(case, family, row, roi, medians_in(sets.ref_c, d, roi), rule))
    schema.write(paper_dir / "overlay_rois.csv", "overlay_rois", roi_rows)
    stare = next(x for x in drawn if x[0] == STARE_FAMILY)
    return {
        "fig3a_overlays": _draw_rows("fig3a_overlays", drawn, sets, roi, fig_dir),
        "fig1c_overlay_before": _draw_before("fig1c_overlay_before", sets, roi, stare[2],
                                             stare[3], stare[4], fig_dir),
        "fig2b_overlay_stare": _draw_rows("fig2b_overlay_stare", [stare], sets, roi, fig_dir),
    }


def main(a):
    got = build(a.cases, a.out, a.roi_um, a.case_id, a.dataset)
    for name in OVERLAYS:
        res = got[name]
        tag = "placeholder" if res.get("placeholder") else "drawn"
        print(f"{name}\t{tag}\t{res['files'][0]}")
    print(f"overlay_rois\t{Path(a.out) / 'paper' / 'overlay_rois.csv'}")
    return 0

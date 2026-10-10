"""PALOM 2026.4.1 (labsyspharm/palom, ``pip install palom==2026.4.1``) by its README's
scripting recipe: one affine from thumbnails, then one whole-pixel shift per block of the
reference, pairwise.

Rows (``--label`` with the options that go with it; a label and its options must agree):

==================  ==========================================================
``palom``           no option: the README's recipe (PALOM's primary row)
``palom_b1024``     ``--opt block=1024``: the same recipe with the REFERENCE array
                    rechunked to 1024 px blocks before ``Aligner`` sees it.
                    Configured by the benchmark, never a primary row
==================  ==========================================================

Each run yields two results, both reference -> moving (``REF_TO_MOV``):

``palom_affine``  the coarse affine alone (``aligner.affine_matrix``)
``palom``         the affine and each block's shift: the image PALOM writes

**The recipe** (``README.md:220-284`` at the tag; ``RECIPE``): ``OmePyramidReader(path)``
for each slide, ``LEVEL = 0``, ``THUMBNAIL_LEVEL = ref_reader.get_thumbnail_level_of_size(
2000)`` (the README's suggestion, used for both readers as the README uses one level for
both), ``Aligner`` with its six keywords, ``coarse_register_affine(n_keypoints=4000)``,
``compute_shifts()``, ``constrain_shifts()``. Nothing else is passed and no dask, OpenCV
or thread setting is touched. The clock stops after ``constrain_shifts()``.

**A block is one dask chunk of the reference array** (``align.py:20-28, 43-51, 213-215``),
which for an OME-TIFF is the file's TIFF tile (``reader.py:111-136``: ``tifffile``'s zarr
view through ``da.from_zarr``): 2048 px on the cohort's preprocessed slides, 1024 px on
this benchmark's generated cases. There is no block-size parameter. ``block=N`` therefore
changes the one thing PALOM reads the block from: ``ref_img.rechunk((N, N))`` on the array
its reader returned, before ``Aligner``; the same array is handed to the writer, so the
image is drawn on the same blocks as the shifts were measured on.

**The exact point map** (``Geometry``, ``block_map``). Read from the source:

* ``aligner.affine_matrix`` is 3 x 3 in homogeneous (x, y, 1) and maps a MOVING pixel to a
  REFERENCE pixel at ``LEVEL``: ``cv2.estimateAffine2D(all_dst, all_src)`` is fitted from
  the right image's keypoints (``img_right``, the moving thumbnail; ``.pt`` is (x, y)) to
  the left image's (``register.py:226-234, 295-300``), and ``align.py:172-184`` rescales
  it by the two thumbnail down-factors. PALOM's own writer uses it in that direction: it
  takes each output block's corners through ``transformation.inverse`` to find the source
  region (``block_affine.py:56-71``).
* ``aligner.shifts`` is ``(n_blocks, 2)``, each row ``(dy, dx)`` in reference pixels,
  row-major over ``aligner.grid_shape = ref_img.numblocks`` (rows, columns):
  ``align.py:210-215``; ``shift_affine_mx`` unpacks ``y, x = shift`` and builds
  ``T(x, y) @ mx`` (``align.py:113-125``), and ``block_affine_matrices_da`` lays the
  matrices out with ``np.arange(rows * cols).reshape(grid_shape)`` (``align.py:128-135``).
* The writer draws output block ``b`` with ``cv2.warpAffine(src, M_b, ...)`` and no
  inverse flag (``block_affine.py:81-99``), ``M_b = T(shift_b) @ affine``. A block is the
  pixels of one reference chunk: its position and shape come from dask's ``block_info``
  (``block_affine.py:18-29``) for ``chunks=ref_img.chunks`` (``align.py:25``).

So ``written(P) = bilinear(moving, A^-1 (P - s_b(P)))`` with ``b(P)`` the block that holds
output pixel ``P``: an exact reference -> moving map in closed form. The only inverse is
of the 3 x 3 affine, which PALOM's own code takes (``block_affine.py:59``). The other
direction is not a function: where neighbouring shifts differ, some moving points are
drawn twice and some never. Pixel centres are on the integers; pixel ``X_edge - 1`` is the
last of the left block, so for a point between pixel centres the edge is at
``X_edge - 0.5``. That ``cv2.warpAffine`` without ``WARP_INVERSE_MAP`` samples the source
at ``M^-1`` of each output pixel centre is OpenCV's documented behaviour ``[MEMORY]``;
``test_palom_adapter.py`` measures it on the OpenCV installed where the suite runs, and
``test_external_methods.py`` on PALOM's own function in its image.

**Seams** (``band_ref``, ``band_truth``; ``SELFCHECK_MASK``). The map jumps by
``A^-1 (s_b - s_b')`` at an edge between two blocks whose shifts differ, and the image is
torn there. A reference nucleus is on a seam when the blocks its outline's bounding box
touches, grown by ``SEAM_MARGIN_PX``, do not all carry the same shift. A known-truth probe
is on a seam by the same rule applied to its own nucleus: the bounding box of the moving
nucleus the probe is the centroid of, carried to where the truth puts that centroid on
the reference (``truth_ref``). Only the block grid and the accepted shifts enter. The
scorer scores every method a second time without those nuclei and probes
(``<variant>:non_palom``).

**Written image.** ``palom.align.block_affine_transformed_moving_img(ref_img=
aligner.ref_img, moving_img=mov_reader.pyramid[0], mxs=aligner.block_affine_matrices_da)``
then ``palom.pyramid.write_pyramid(mosaics=[that], output_path=..., pixel_size=
ref_reader.pixel_size * ref_reader.level_downsamples[0])``: every channel of the moving
slide, on the reference grid, zero where nothing was drawn, channels anonymous (the
writer's default). The affine stage is drawn by the same two calls with
``mxs=aligner.affine_matrix``, the package's own coarse-only path
(``cli/align_he.py:165-172``); it is written for the selfcheck only.

**Known risks, reported as they are.** Failed feature matching continues with the identity
and a log line (``register.py:152-154``): the row is kept and ``problem`` says so.
``constrain_shifts`` thresholds the shift lengths with ``threshold_triangle`` and fits a
regression on the blocks under it (``align.py:54-80``). With every finite shift equal no
block is under the threshold and the regression has nothing to fit: a transcription of
those lines run with scikit-image 0.24.0 and scikit-learn 1.7.2 (not PALOM, and not its
pinned scikit-image 0.19.3) raises ``ValueError: Found array with 0 sample(s)``. That
fails the ``palom`` row and keeps ``palom_affine``. A single-level OME-TIFF was reported to
raise ``ValueError: seek of closed file`` (issue #12); see ``first_block_persist``.

Sources, with what was read for this file. labsyspharm/palom at tag ``v2026.4.1`` (commit
``f5f9c61a``; the PyPI sdist ``palom-2026.4.1`` is file-identical per the port's PALOM
report) ``[PARTIAL: read in full align.py, block_affine.py, pyramid.py, __init__.py;
register.py:1-316; reader.py:1-175; img_util.py:1-47; README.md:214-284;
cli/align_he.py:150-180; missing: align_multi_res.py, align_multi_obj.py, register_dev.py,
register_util.py, the stores and the other command-line tools, none on this path]``.
PALOM issues #12 and #13 ``[PARTIAL: the maintainer's comments, as saved by the port's
PALOM report]``. The maintainer's example script, ``Yu-AnChen/dump``
``2023-09/palom-example-script.py`` at commits ``61c8e1a2`` and ``333ebbb5`` ``[FULL]``:
the maintainer's personal script, not an official PALOM source. OpenCV, scikit-image and
dask were not read for this file. Nothing here was run with PALOM: it is not installed
where this was written.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np

from ..cases import load_points, polys
from ..mapping import REF_TO_MOV, PointMap

SCOPE = "pair"
VARIANTS = ("palom_affine", "palom")
VERSION = "2026.4.1"
DIRECTION = REF_TO_MOV
SOURCE = "labsyspharm/palom v2026.4.1 (PyPI palom==2026.4.1)"
LEVEL = 0
THUMBNAIL_SIZE = 2000
N_KEYPOINTS = 4000
ALIGNER_KEYWORDS = (
    "ref_img",
    "moving_img",
    "ref_thumbnail",
    "moving_thumbnail",
    "ref_thumbnail_down_factor",
    "moving_thumbnail_down_factor",
)
RECIPE = {
    "source": "README.md:220-284 at v2026.4.1 (For TIFF and OME-TIFF files)",
    "reader": "palom.reader.OmePyramidReader(path)",
    "level": LEVEL,
    "thumbnail_level": f"ref_reader.get_thumbnail_level_of_size({THUMBNAIL_SIZE})",
    "aligner_keywords": list(ALIGNER_KEYWORDS),
    "calls": [
        f"coarse_register_affine(n_keypoints={N_KEYPOINTS})",
        "compute_shifts()",
        "constrain_shifts()",
    ],
}
# label -> the block size the benchmark sets (None: PALOM's own, the reference's chunks)
LABELS = {"palom": None, "palom_b1024": 1024}
WRITER = (
    "palom.align.block_affine_transformed_moving_img, then palom.pyramid.write_pyramid "
    "with the pixel size and its defaults (README.md:263-284)"
)
SEAM_MARGIN_PX = (
    1.0  # a nucleus this close to a block with another shift is on the seam
)
DOT_MARGIN_PX = 8.0  # the selfcheck's dots are 7 px across
MIN_SHIFTED_BLOCKS = 0.10  # below it the selfcheck cannot see a wrong sign or order
MASKS = ("seams",)
PREREAD = ("auto", "0", "1")
CLOSED_FILE = "closed file"
IDENTITY_PROBLEM = "coarse matching failed; PALOM continued with the identity"


class _NullMonitor:
    def pause(self):
        pass

    def resume(self):
        pass


# ── settings ──────────────────────────────────────────────────────────────────
def block_option(opts):
    """``block`` as a number of pixels, or None when the option is not given."""
    raw = (opts or {}).get("block")
    if raw in (None, ""):
        return None
    try:
        block = int(str(raw))
    except ValueError:
        raise ValueError(f"block={raw!r}: expected a number of pixels") from None
    if block < 16:
        raise ValueError(f"block={block}: expected at least 16 px")
    return block


def settings(opts):
    """``(block, preread)`` of this run, checked against its label.

    ``block`` is None (PALOM's own blocks) or the size the reference is rechunked to. A
    run with ``block=N`` must be labelled ``palom_b<N>`` and the other way round, so a row
    cannot carry another row's name. ``preread`` is ``auto``, ``0`` or ``1``
    (``first_block_persist``).
    """
    block = block_option(opts)
    label = str(opts.get("label") or "") or "palom"
    named = re.fullmatch(r"palom_b(\d+)", label)
    if label == "palom" or named:
        want = int(named.group(1)) if named else None
        if want != block:

            def say(b):
                return "no block option" if b is None else f"block={b}"

            raise ValueError(
                f"the label {label!r} names a run with {say(want)}; this one has "
                f"{say(block)}. A label and its options come together"
            )
    elif block is not None:
        raise ValueError(
            f"block={block} belongs to the label palom_b{block}, not to {label!r}"
        )
    preread = str(opts.get("preread", "auto")).strip().lower()
    if preread not in PREREAD:
        raise ValueError(f"preread={preread!r}: expected one of {PREREAD}")
    return block, preread


def _nuclear(case, side):
    """``(path, channel index)`` of a slide's nuclear channel, in its original file."""
    path = getattr(case, f"{side}_image") or getattr(case, f"{side}_nuclear")
    return str(path), int(case.extra.get(f"{side}_channel", 0))


# ── the exact map ─────────────────────────────────────────────────────────────
def block_index(edges, values):
    """The block each coordinate falls in, for block edges ``edges`` (``[0, ..., size]``).

    Pixel ``edge - 1`` is the last of the block before the edge, and pixel centres are on
    the integers: between two pixel centres the edge is at ``edge - 0.5``. A coordinate
    outside the image belongs to the nearest block.
    """
    inner = np.asarray(edges, dtype=float)[1:-1] - 0.5
    index = np.searchsorted(inner, np.asarray(values, dtype=float), side="right")
    return np.clip(index, 0, len(edges) - 2)


class Geometry:
    """What the two maps and the seam masks need, copied from the aligner.

    ``affine``  3 x 3, moving (x, y, 1) -> reference (x, y, 1)  (``aligner.affine_matrix``)
    ``shifts``  ``(rows * cols, 2)`` as ``(dy, dx)``, row-major    (``aligner.shifts``)
    ``chunks``  the reference array's ``(row chunks, column chunks)``: the real blocks,
                whose last one may be short                        (``aligner.ref_img.chunks``)
    """

    def __init__(self, affine, shifts, grid_shape, chunks):
        self.affine = np.array(affine, dtype=float).reshape(3, 3)
        self.inverse = np.linalg.inv(self.affine)  # of the affine alone, as PALOM does
        self.grid_shape = tuple(int(v) for v in grid_shape)
        rows, cols = self.grid_shape
        self.edges_y = np.concatenate([[0], np.cumsum([int(v) for v in chunks[0]])])
        self.edges_x = np.concatenate([[0], np.cumsum([int(v) for v in chunks[1]])])
        if (len(self.edges_y) - 1, len(self.edges_x) - 1) != (rows, cols):
            raise RuntimeError(
                f"the reference has {len(self.edges_y) - 1} x {len(self.edges_x) - 1} "
                f"chunks and the aligner's grid is {rows} x {cols}"
            )
        self.shifts = np.array(shifts, dtype=float).reshape(-1, 2)
        if len(self.shifts) != rows * cols:
            raise RuntimeError(
                f"{len(self.shifts)} shifts for a grid of {rows} x {cols} blocks"
            )
        # blocks with the same shift carry the same id: no seam between them
        _, ids = np.unique(self.shifts, axis=0, return_inverse=True)
        self.shift_id = np.asarray(ids).reshape(rows, cols)

    @classmethod
    def from_info(cls, blocks):
        """Rebuilt from ``info["blocks"]`` of a run (``describe``)."""
        chunks = (np.diff(blocks["edges_y"]), np.diff(blocks["edges_x"]))
        return cls(blocks["affine"], blocks["shifts_yx"], blocks["grid_shape"], chunks)

    def block_of(self, xy):
        """``(row, column)`` of the block holding each reference point."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return block_index(self.edges_y, xy[:, 1]), block_index(self.edges_x, xy[:, 0])


def affine_map(geo):
    """reference px -> moving px through the coarse affine alone: ``A^-1 P``."""
    linear, offset = geo.inverse[:2, :2].T.copy(), geo.inverse[:2, 2].copy()

    def fn(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return xy @ linear + offset

    return fn


def block_map(geo):
    """reference px -> moving px, the image as PALOM writes it: ``A^-1 (P - s_b(P))``."""
    linear, offset = geo.inverse[:2, :2].T.copy(), geo.inverse[:2, 2].copy()
    shifts_xy = geo.shifts[:, ::-1].copy()  # (dy, dx) -> (dx, dy)
    cols = geo.grid_shape[1]

    def fn(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(xy):
            return xy.copy()
        row, col = geo.block_of(xy)
        return (xy - shifts_xy[row * cols + col]) @ linear + offset

    return fn


# ── seams ─────────────────────────────────────────────────────────────────────
def on_seam(geo, lo_xy, hi_xy, margin=SEAM_MARGIN_PX):
    """True for each box ``[lo, hi]`` (reference px) that, grown by ``margin``, touches
    blocks that do not all carry the same shift. A point is a box with ``lo == hi``."""
    lo = np.asarray(lo_xy, dtype=float).reshape(-1, 2) - margin
    hi = np.asarray(hi_xy, dtype=float).reshape(-1, 2) + margin
    out = np.zeros(len(lo), dtype=bool)
    ok = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
    if not ok.any():
        return out
    r0, c0 = geo.block_of(lo[ok])
    r1, c1 = geo.block_of(hi[ok])
    ids = geo.shift_id
    first = ids[r0, c0]
    torn = (ids[r0, c1] != first) | (ids[r1, c0] != first) | (ids[r1, c1] != first)
    wide = np.flatnonzero((r1 - r0 > 1) | (c1 - c0 > 1))  # a box over three blocks
    for i in wide:
        part = ids[r0[i] : r1[i] + 1, c0[i] : c1[i] + 1]
        torn[i] = bool((part != part.flat[0]).any())
    out[ok] = torn
    return out


def _boxes(ps):
    """``(lo (n, 2), hi (n, 2))``: each nucleus's bounding box (NaN for one with no vertex)."""
    xy = np.asarray(ps.xy, dtype=float).reshape(-1, 2)
    feat = np.repeat(ps.ring_feat, np.diff(ps.ring_off))
    lo = np.full((ps.n_features, 2), np.inf)
    hi = np.full((ps.n_features, 2), -np.inf)
    np.minimum.at(lo, feat, xy)
    np.maximum.at(hi, feat, xy)
    empty = ~np.isfinite(lo).all(axis=1)
    lo[empty], hi[empty] = np.nan, np.nan
    return lo, hi


def nucleus_band(geo, points):
    """One bool per REFERENCE nucleus: True where its outline's bounding box, grown by
    ``SEAM_MARGIN_PX``, touches blocks with different shifts. None for a case without
    reference nuclei."""
    ps = polys(points, "ref")
    if ps is None or not ps.n_features:
        return None
    return on_seam(geo, *_boxes(ps))


def probe_band(geo, points):
    """One bool per known-truth probe, or None for a case without probes.

    A probe is the pair (``truth_mov``, ``truth_ref``): the centroid of moving nucleus
    ``truth_cell`` and where the truth puts it on the reference. It is on a seam when
    that nucleus's bounding box, moved so that its centroid sits at ``truth_ref``, is on a
    seam by the rule of ``nucleus_band``. Without ``truth_cell`` or moving nuclei the
    probe is judged as the point ``truth_ref``.
    """
    if "truth_ref" not in points or not len(points["truth_ref"]):
        return None
    t_ref = np.asarray(points["truth_ref"], dtype=float).reshape(-1, 2)
    lo = hi = t_ref
    ps = polys(points, "mov")
    if ps is not None and ps.n_features and "truth_cell" in points:
        cell = np.asarray(points["truth_cell"], dtype=np.int64)
        t_mov = np.asarray(points["truth_mov"], dtype=float).reshape(-1, 2)
        box_lo, box_hi = _boxes(ps)
        lo = t_ref + np.nan_to_num(box_lo[cell] - t_mov)
        hi = t_ref + np.nan_to_num(box_hi[cell] - t_mov)
    return on_seam(geo, lo, hi)


def seam_jumps(geo):
    """The length of the shift difference across every edge between two neighbouring
    blocks, in reference px (the map jumps by ``A^-1`` of it)."""
    rows, cols = geo.grid_shape
    s = geo.shifts.reshape(rows, cols, 2)
    return np.concatenate(
        [
            np.hypot(*(s[:, 1:] - s[:, :-1]).reshape(-1, 2).T),
            np.hypot(*(s[1:] - s[:-1]).reshape(-1, 2).T),
        ]
    )


# ── the package ───────────────────────────────────────────────────────────────
def installed_version(package):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("palom")
    except PackageNotFoundError:
        return str(getattr(package, "__version__", "?"))


def first_block_persist(reader):
    """The maintainer's workaround for ``ValueError: seek of closed file`` on a file whose
    pyramid PALOM has to build itself (issue #12): read the first block of level 0 once
    and drop it. From the maintainer's personal script (``Yu-AnChen/dump``
    ``2023-09/palom-example-script.py`` at ``333ebbb5``, lines 46-48 ``[FULL]``), not an
    official PALOM source. Benchmark-side when it is used; see ``_Reads``."""
    _ = reader.pyramid[0].blocks.ravel()[0].persist()


class _Reads:
    """Runs PALOM's calls and, on ``seek of closed file`` only, applies the maintainer's
    workaround to both readers once and repeats the call that failed.

    ``preread=auto`` (default) does that; ``1`` applies the workaround before the first
    read; ``0`` never does. What happened is recorded (``describe``) and, when the
    workaround was applied, listed as a benchmark-side step. The failed call's time stays
    on the clock.
    """

    def __init__(self, readers, mode):
        self.readers, self.mode = list(readers), mode
        self.applied, self.first_error, self.lost_s = False, "", 0.0
        if mode == "1":
            self.apply()

    def apply(self):
        for reader in self.readers:
            first_block_persist(reader)
        self.applied = True

    def call(self, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        except ValueError as exc:
            if CLOSED_FILE not in str(exc) or self.mode != "auto" or self.applied:
                raise
            self.first_error = f"{type(exc).__name__}: {exc}"
            self.lost_s = time.perf_counter() - t0
        self.apply()
        return fn()

    def describe(self):
        return {
            "mode": self.mode,
            "applied": bool(self.applied),
            "first_error": self.first_error,
            "failed_call_s": float(self.lost_s),
        }


def save_figures(dest, prefix):
    """Save and close every matplotlib figure that is open (the documented
    ``coarse_register_affine`` call opens one: ``plot_match_result`` defaults to True,
    ``align.py:156-160``, ``register.py:235-263``). Returns ``{"files", "error"}``; a
    failure to save never fails the run, and the figures are closed either way."""
    import sys

    out = {"files": [], "error": ""}
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:  # the package never imported it: nothing can be open
        return out
    try:
        numbers = list(plt.get_fignums())
        if numbers:
            Path(dest).mkdir(parents=True, exist_ok=True)
        for n in numbers:
            path = Path(dest) / f"{prefix}_{n}.png"
            plt.figure(n).savefig(path, dpi=144, bbox_inches="tight")
            out["files"].append(str(path))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            plt.close("all")
        except Exception as exc:
            out["error"] = out["error"] or f"{type(exc).__name__}: {exc}"
    return out


def _concurrency():
    """What the pools PALOM uses could see. Nothing is set."""
    seen = {"os_cpu_count": os.cpu_count()}
    try:
        import psutil  # noqa: F401

        seen["psutil"] = True
    except ImportError:
        seen["psutil"] = False
    try:
        import dask.system

        seen["dask_cpu_count"] = int(dask.system.CPU_COUNT)
    except Exception as exc:
        seen["dask_cpu_count"] = f"{type(exc).__name__}: {exc}"
    try:
        import cv2

        seen["cv2_num_threads"] = int(cv2.getNumThreads())
    except Exception as exc:
        seen["cv2_num_threads"] = f"{type(exc).__name__}: {exc}"
    return seen


def _file_levels(path):
    """How many pyramid levels the FILE has, as PALOM's reader counts them
    (``reader.py:112-117``). None when it cannot be read that way."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            n = len(tif.series)
            return len(tif.series[0].levels) if n == 1 else n
    except Exception:
        return None


def _chunk_summary(chunks):
    """``{"y": [...], "x": [...]}``: the distinct block sizes on each axis, in order."""
    return {
        axis: [int(v) for v in dict.fromkeys(int(c) for c in part)]
        for axis, part in zip(("y", "x"), chunks)
    }


def describe(geo, original_shifts):
    """What a run records about its blocks: counts, shift statistics, and the arrays the
    map is made of (``blocks``), so the map and the seam mask can be rebuilt from
    ``run.json``."""
    original = np.asarray(original_shifts, dtype=float).reshape(-1, 2)
    blank = ~np.isfinite(original).all(axis=1)
    replaced = (original != geo.shifts).any(axis=1)
    kept = ~replaced
    length = np.hypot(geo.shifts[:, 0], geo.shifts[:, 1])
    jumps = seam_jumps(geo)
    whole = (geo.shifts == np.rint(geo.shifts)).all(axis=1)

    def pct(values, q):
        return float(np.percentile(values, q)) if len(values) else float("nan")

    return {
        "grid_shape": list(geo.grid_shape),
        "n_blocks": int(len(geo.shifts)),
        "n_blocks_blank": int(blank.sum()),
        "n_blocks_replaced": int(replaced.sum()),
        "integer_shifts": float(whole[kept].mean()) if kept.any() else float("nan"),
        "shift_nonzero_fraction": float((length > 0).mean()),
        "shift_px_p50": pct(length, 50),
        "shift_px_p95": pct(length, 95),
        "shift_px_max": float(length.max()) if len(length) else float("nan"),
        "n_block_edges": int(len(jumps)),
        "n_seams": int((jumps > 0).sum()),
        "seam_jump_px_p50": pct(jumps, 50),
        "seam_jump_px_p95": pct(jumps, 95),
        "blocks": {
            "affine": geo.affine.tolist(),
            "grid_shape": list(geo.grid_shape),
            "edges_y": [int(v) for v in geo.edges_y],
            "edges_x": [int(v) for v in geo.edges_x],
            "shifts_yx": geo.shifts.tolist(),
            "original_shifts_yx": [
                [float(v) if np.isfinite(v) else None for v in row] for row in original
            ],
        },
    }


def register(case, work, opts):
    import palom
    import psutil  # noqa: F401  (dask reads the affinity mask only through it)

    block, preread = settings(opts)
    version = installed_version(palom)
    if version != VERSION:
        raise RuntimeError(f"palom {VERSION} is the pinned release; found {version}")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    mon = opts.get("monitor") or _NullMonitor()
    ref_path, ref_channel = _nuclear(case, "ref")
    mov_path, mov_channel = _nuclear(case, "mov")

    ref_reader = palom.reader.OmePyramidReader(ref_path)
    mov_reader = palom.reader.OmePyramidReader(mov_path)
    reads = _Reads((ref_reader, mov_reader), preread)
    thumbnail_level = int(ref_reader.get_thumbnail_level_of_size(THUMBNAIL_SIZE))
    if thumbnail_level >= len(mov_reader.pyramid):
        raise RuntimeError(
            f"the reference's thumbnail level is {thumbnail_level} and the moving slide "
            f"has {len(mov_reader.pyramid)} level(s): the README's recipe uses one "
            "level for both"
        )
    ref_img = ref_reader.read_level_channels(LEVEL, ref_channel)
    own_chunks = _chunk_summary(ref_img.chunks)
    if block is not None:
        ref_img = ref_img.rechunk((block, block))
    ref_factor = (
        ref_reader.level_downsamples[thumbnail_level]
        / ref_reader.level_downsamples[LEVEL]
    )
    mov_factor = (
        mov_reader.level_downsamples[thumbnail_level]
        / mov_reader.level_downsamples[LEVEL]
    )
    aligner = palom.align.Aligner(
        ref_img=ref_img,
        moving_img=mov_reader.read_level_channels(LEVEL, mov_channel),
        ref_thumbnail=reads.call(
            lambda: ref_reader.read_level_channels(
                thumbnail_level, ref_channel
            ).compute()
        ),
        moving_thumbnail=reads.call(
            lambda: mov_reader.read_level_channels(
                thumbnail_level, mov_channel
            ).compute()
        ),
        ref_thumbnail_down_factor=ref_factor,
        moving_thumbnail_down_factor=mov_factor,
    )
    aligner.coarse_register_affine(n_keypoints=N_KEYPOINTS)
    affine = np.array(aligner.affine_matrix, dtype=float)

    mon.pause()  # the benchmark's own bookkeeping, not the method's work
    try:
        t0 = time.perf_counter()
        figures = save_figures(work / "qc", "coarse")
        figures["save_s"] = time.perf_counter() - t0
        identity = bool(
            np.array_equal(np.asarray(aligner.coarse_affine_matrix), np.eye(3))
        )
        pixel_size = float(ref_reader.pixel_size)  # the README passes it to the writer
        seen = _concurrency()
        levels = {"reference": _file_levels(ref_path), "moving": _file_levels(mov_path)}
        chunks = tuple(tuple(int(v) for v in c) for c in aligner.ref_img.chunks)
        rows, cols = (int(v) for v in aligner.grid_shape)
        coarse = Geometry(affine, np.zeros((rows * cols, 2)), (rows, cols), chunks)
    finally:
        mon.resume()
    px = float(case.pixel_size_um) if case.pixel_size_um else None
    benchmark_side = [
        f"the nuclear channel of each slide by index (reference {ref_channel}, moving "
        f"{mov_channel}): PALOM selects channels by index only",
        "LEVEL = 0 (the README: 'set LEVEL = 0 for processing lowest level pyramid "
        "(full resolution)'; its example uses 1 'for a quick test')",
        f"the thumbnail level from get_thumbnail_level_of_size({THUMBNAIL_SIZE}), the "
        "README's suggestion (its example uses the literal 3)",
        "the writer runs after the clock stops, on the moving slide alone (the "
        "README's example also writes three reference channels)",
        "the point map assembled from the aligner's attributes: affine_matrix, shifts, "
        "grid_shape and ref_img.chunks",
        "the matplotlib figure the documented call opens is saved to the work folder "
        "and closed, clock paused",
    ]
    if block is not None:
        benchmark_side.append(
            f"block={block}: the reference array PALOM's reader returned is rechunked "
            f"to ({block}, {block}) before Aligner (ref_img.rechunk); PALOM has no "
            "block-size parameter, a block is one chunk of the reference. Configured "
            "by the benchmark"
        )
    info = {
        "version": version,
        "source": SOURCE,
        "RECIPE": RECIPE,
        "affine": affine.tolist(),
        "coarse_identity_placeholder": identity,
        "thumbnail_level": thumbnail_level,
        "thumb_um_per_px": None if px is None else px * float(ref_factor),
        "ref_down_factor": float(ref_factor),
        "mov_down_factor": float(mov_factor),
        "n_levels": {
            "reference": len(ref_reader.pyramid),
            "moving": len(mov_reader.pyramid),
        },
        "file_levels": levels,
        # PALOM built the levels itself (reader.py:46-68): a one-level file
        "auto_pyramid": {
            side: bool(
                levels[side] == 1 and len(reader.pyramid) > 1
                if levels[side] is not None
                else False
            )
            for side, reader in (("reference", ref_reader), ("moving", mov_reader))
        },
        "block_px": _chunk_summary(chunks),
        "file_block_px": own_chunks,
        "block_option": block,
        "grid_shape": [rows, cols],
        "n_blocks": rows * cols,
        "pixel_size_detected": pixel_size,
        "workers_seen": seen.get("dask_cpu_count"),
        "concurrency": seen,
        "figures": figures,
        "implementation": {
            "package": "palom",
            "source": str(Path(getattr(palom, "__file__", "?")).parent),
            "method": "PALOM, the README's scripting recipe (no publication)",
            "registration": "palom.align.Aligner: coarse_register_affine, "
            "compute_shifts, constrain_shifts",
            "point_warp": "A^-1 (P - s_b(P)) from the aligner's own attributes; the "
            "only inverse is the 3 x 3 affine's, which the package's writer takes too",
            "writer": WRITER,
            "benchmark_side": benchmark_side,
        },
    }
    if identity:
        info["problem"] = IDENTITY_PROBLEM
    yield (
        VARIANTS[0],
        PointMap(affine_map(coarse), DIRECTION),
        {**info, "closed_file": reads.describe()},
    )

    reads.call(aligner.compute_shifts)
    aligner.constrain_shifts()
    shifts = np.array(aligner.shifts, dtype=float).reshape(-1, 2)

    mon.pause()
    try:
        if not np.isfinite(shifts).all():
            bad = int((~np.isfinite(shifts).all(axis=1)).sum())
            raise RuntimeError(
                f"{bad} of {len(shifts)} block shifts are not finite after "
                "constrain_shifts: the image cannot be drawn there"
            )
        geo = Geometry(affine, shifts, (rows, cols), chunks)
        original = np.array(
            getattr(aligner, "original_shifts", shifts), dtype=float
        ).reshape(-1, 2)
        info.update(describe(geo, original))
        points = load_points(case)
        band_ref, band_truth = nucleus_band(geo, points), probe_band(geo, points)
        info["band_fraction"] = None if band_ref is None else float(band_ref.mean())
        info["closed_file"] = reads.describe()
        if reads.applied:
            benchmark_side.append(
                "the first block of level 0 of each slide was read once and dropped "
                "before PALOM's calls (the maintainer's workaround for 'seek of closed "
                "file' on a file without pyramid levels, from their personal script; "
                "not an official PALOM source)"
                + (
                    ""
                    if preread == "1"
                    else f", after the recipe raised {reads.first_error!r}; the failed "
                    f"call's {reads.lost_s:.1f} s stay on the clock"
                )
            )
        fn = block_map(geo)
    finally:
        mon.resume()
    state = {
        "case": case,
        "palom": palom,
        "aligner": aligner,
        "ref_reader": ref_reader,
        "mov_reader": mov_reader,
        "reads": reads,
        "geo": geo,
        "pixel_size": pixel_size,
        "mov_channel": mov_channel,
        "maps": {VARIANTS[0]: affine_map(geo), VARIANTS[1]: fn},
    }
    final = {**info, "_state": state}
    if band_ref is not None:
        final["band_ref"] = band_ref
    if band_truth is not None:
        final["band_truth"] = band_truth
    yield VARIANTS[1], PointMap(fn, DIRECTION), final


# ── the writer ────────────────────────────────────────────────────────────────
def _still(state):
    """Refuse to draw an image the yielded map no longer describes."""
    aligner, geo = state["aligner"], state["geo"]
    same = (
        np.array_equal(
            np.asarray(aligner.shifts, dtype=float).reshape(-1, 2), geo.shifts
        )
        and np.array_equal(np.asarray(aligner.affine_matrix, dtype=float), geo.affine)
        and tuple(int(v) for v in aligner.grid_shape) == geo.grid_shape
        and np.array_equal(np.cumsum(aligner.ref_img.chunks[0]), geo.edges_y[1:])
        and np.array_equal(np.cumsum(aligner.ref_img.chunks[1]), geo.edges_x[1:])
    )
    if not same:
        raise RuntimeError(
            "the aligner's affine, shifts or blocks changed after the point map was "
            "handed over: the image would not be the one that was scored"
        )


def _write(state, variant, dest_dir):
    from ._valis_points import footprint_box

    _still(state)
    palom, aligner = state["palom"], state["aligner"]
    ref_reader, mov_reader = state["ref_reader"], state["mov_reader"]
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "registered.ome.tif"
    final = variant == VARIANTS[1]
    # README.md:263-284; the affine stage as cli/align_he.py:165-172 (only_coarse)
    mxs = aligner.block_affine_matrices_da if final else aligner.affine_matrix
    moving = mov_reader.pyramid[LEVEL]

    def draw():
        mosaic = palom.align.block_affine_transformed_moving_img(
            ref_img=aligner.ref_img, moving_img=moving, mxs=mxs
        )
        palom.pyramid.write_pyramid(
            mosaics=[mosaic],
            output_path=str(out),
            pixel_size=state["pixel_size"] * ref_reader.level_downsamples[LEVEL],
        )

    state["reads"].call(draw)
    case = state["case"]
    n_channels = int(moving.shape[0])
    return {
        "files": [str(out)],
        "format": "pyramidal OME-TIFF (BigTIFF, zlib, 1024 px tiles: the writer's "
        "defaults), channels anonymous",
        "writer": WRITER
        if final
        else WRITER + ", with mxs = aligner.affine_matrix (cli/align_he.py:165-172)",
        "fill": "0 where no moving pixel lands",
        "valid_box": footprint_box(state["maps"][variant], case.ref_hw, case.mov_hw),
        "channels": n_channels,
        "dtype": str(moving.dtype),
        "nuclear_channel": int(state["mov_channel"]),
        "frame": "the reference slide's grid",
    }


def write_native(state, case_id, dest_dir):
    """PALOM's own writer on every channel of the moving slide: the README's two calls,
    with the aligner's block matrices."""
    return _write(state, VARIANTS[1], dest_dir)


def write_stage(state, case_id, variant, dest_dir):
    """The affine stage's image, by the same two calls with the affine alone (selfcheck)."""
    if variant != VARIANTS[0]:
        return None
    return _write(state, variant, dest_dir)


# ── selfcheck hooks ───────────────────────────────────────────────────────────
def SELFCHECK_MASK(info, mask):
    """``--mask seams``: detections within ``DOT_MARGIN_PX`` of an edge between two blocks
    with different shifts are left out. A dot drawn across such an edge is torn in the
    image (two halves, one per block), and its centroid is neither half's. The affine
    stage has no seams: nothing is left out there."""
    if mask not in MASKS:
        raise SystemExit(f"palom knows the masks {MASKS}, not {mask!r}")
    blocks = (info or {}).get("blocks")
    if not blocks:
        return lambda xy: np.zeros(len(xy), dtype=bool)
    geo = Geometry.from_info(blocks)
    return lambda xy: on_seam(geo, xy, xy, DOT_MARGIN_PX)


def SELFCHECK_NOT_CHECKED(info, variant):
    """Why an ``ok`` comparison of ``variant`` proves too little, or None.

    With (almost) every block shift at zero the block map IS the affine map: a swapped
    axis, a wrong sign or a wrong block order would pass unseen."""
    if variant != VARIANTS[1]:
        return None
    moved = (info or {}).get("shift_nonzero_fraction")
    if moved is None or moved >= MIN_SHIFTED_BLOCKS:
        return None
    return (
        f"{100 * moved:.0f} % of the blocks have a non-zero shift (under "
        f"{100 * MIN_SHIFTED_BLOCKS:.0f} %): the block map equals the affine map here, "
        "so its sign and block order were not tested"
    )


def SELFCHECK_PAIR_TILE(opts):
    """The TIFF tile of the selfcheck pair, or None for the generator's own (1024 px).

    The rechunked row is checked on files with 2048 px tiles, the cohort's layout, so
    that the rechunk does something: 1024 px blocks cut from 2048 px tiles."""
    try:
        block = block_option(opts)
    except ValueError:
        return None
    return None if block is None else 2 * block

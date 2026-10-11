"""Synthetic nuclear slides with a KNOWN moving -> reference map.

The reference is a field of Gaussian nuclei inside a blobby tissue silhouette. The moving
slide is the same tissue seen through a forward map

    g(p) = c + R(theta) (p - c) + t + w(p)            moving pixel p  ->  reference position

with ``w`` a displacement field: ``mov(p) = base(g(p))``. ``w`` is a smooth two-wave field of
peak amplitude ``amp`` px and wavelength ``wavelength`` px, plus, by ``field``:

``multiscale``  three random fields of correlation length ``scales`` px and RMS ``rms`` px
                each, summed (energy at several scales at once)
``grid``        a ``grid_nodes`` x ``grid_nodes`` (9 x 9) grid of i.i.d. Gaussian
                control-point moves of SD ``grid_sd`` px, cubic-spline interpolated
                (Casamitjana et al., arXiv:2104.14873)
``bumps``       ``bump_count`` local Gaussian bumps of width ``bump_sigma`` px, each peaking at
                0.2-0.6 sigma (at most 50 px), so no single bump folds the map. A bump is
                summed out to ``BUMP_REACH`` (8) sigma and is zero beyond, point by point
``seams``       an independent shift of SD ``seam_sd`` px per ``seam_tile`` px tile: the jump a
                stitching error leaves at a tile boundary. Not continuous, on purpose.
 Because ``g`` is written in the direction a
registration is scored in, the truth for any moving point is one evaluation of ``g``, with no
inversion and no interpolation.

Two things besides geometry differ between the slides, as between two staining rounds:
independent noise, and (``dropout``) a fraction of nuclei missing from the moving slide.
``mov_hw`` gives the moving slide a smaller canvas than the reference (its top-left
``(H, W)`` pixels), because real pairs rarely share a shape and methods that pad or rescale
must get the bookkeeping right. ``ref_hw`` gives the reference a canvas that is not square
(the top-left ``(H, W)`` of the ``n`` x ``n`` scene) and ``tissue_box`` keeps the nuclei
inside a box, so the tissue need not start at the origin; the selfcheck's pair uses both.

**Patients.** A spec with ``patient`` is one ROUND of that patient: ``scene_seed`` fixes the
tissue and the reference slide, which every round of the patient shares (one file), and
``seed`` fixes that round's deformation and noise. ``truth_suite`` builds the frozen
known-truth cases of ``truth_spec.json`` this way: one patient per deformation family, two
rounds each (``regbench.truth``). The ``seed`` of such a plan is the seed ASKED for:
``prepare_one`` first settles it (``settle``), skipping a seed whose true map folds, and
``case.json`` holds the seed used (``extra.spec.seed``) and the record (``extra.reseed``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from .. import cells, truth
from ..cases import Case, case_dir, write_case
from ..imageio import write_nuclear_tiles

# The cohort's pixel size (benchmarks/configs/benchmark.config). The generator works in
# pixels; this is what turns its scores into micrometres.
PIXEL_SIZE_UM = 0.325
NUCLEUS_SIGMA = 5.0
NUCLEUS_DIAMETER = 12.0  # what the segmenter is told; ~FWHM of a sigma-5 blob
PEAK = 20000.0
TILE = 1024  # generation and TIFF tile; also the unit the noise is seeded by
_PAD = int(4.0 * NUCLEUS_SIGMA + 0.5) + 1  # scipy truncates the Gaussian at 4 sigma
_MARGIN = (
    16  # around a resampled patch, so the cubic prefilter's edge transient has died out
)
SCALE_SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)

_BASE = dict(
    n=4096,
    theta=0.0,
    shift=(0.0, 0.0),
    amp=0.0,
    wavelength=1500.0,
    noise=0.02,
    dropout=0.0,
    gain=1.0,
    keep=0.7,
    seed=0,
    mov_hw=None,
    max_cells=None,
    field="wave",
    scales=(2000.0, 500.0, 125.0),
    rms=(8.0, 3.0, 1.0),
    grid_sd=5.0,
    grid_nodes=9,
    bump_sigma=100.0,
    bump_count=20,
    seam_tile=1024,
    seam_sd=3.0,
)
_POSE = dict(
    theta=2.0, shift=(30.0, -20.0)
)  # the rigid part every non-rigid family shares


def _suite(rows):
    return {
        name: {**_BASE, "family": name.rsplit("_", 1)[0].rstrip("0123456789_"), **kw}
        for name, kw in rows
    }


SUITES = {
    # One deformation at every size from 1024 to 65536 px a side (1 Mpx to 4.3 Gpx per
    # slide): the same rotation, shift and smooth warp, so what changes down the suite is
    # the slide, and accuracy and cost can be read against size. Names sort by size.
    "scale": _suite(
        [
            (
                f"scale_{n:05d}",
                dict(
                    n=n,
                    theta=1.0,
                    shift=(30.0, -20.0),
                    amp=10.0,
                    wavelength=1500.0,
                    seed=100,
                    max_cells=200_000,
                ),
            )
            for n in SCALE_SIZES
        ]
    ),
    # What the test-suite registers: seconds per case, one with a smaller moving slide, and
    # two that pin the conditions under which STARE 1.2.1's SOLVE picks a flat field.
    "test": _suite(
        [
            ("rigid", dict(n=1024, theta=2.0, shift=(14.0, -9.0), seed=1)),
            ("smooth", dict(n=1024, amp=5.0, wavelength=600.0, seed=4)),
            (
                "crop",
                dict(
                    n=1024,
                    theta=1.0,
                    shift=(10.0, 6.0),
                    amp=4.0,
                    wavelength=600.0,
                    mov_hw=(832, 928),
                    seed=4,
                ),
            ),
            ("smooth_border", dict(n=1024, amp=5.0, wavelength=600.0, seed=2)),
        ]
    ),
    # Small enough for a CI runner: one rigid, one smooth, one both.
    "ci": _suite(
        [
            ("rigid_0", dict(n=1536, theta=2.0, shift=(18.0, -11.0), seed=1)),
            ("smooth_0", dict(n=1536, amp=6.0, wavelength=700.0, seed=2)),
            (
                "mixed_0",
                dict(
                    n=1536,
                    theta=-1.5,
                    shift=(-9.0, 14.0),
                    amp=5.0,
                    wavelength=700.0,
                    dropout=0.1,
                    seed=3,
                ),
            ),
            (
                "crop_0",
                dict(
                    n=1536,
                    theta=1.0,
                    shift=(12.0, 8.0),
                    amp=4.0,
                    wavelength=700.0,
                    mov_hw=(1280, 1408),
                    seed=4,
                ),
            ),
        ]
    ),
    "full": _suite(
        [
            (f"rigid_{i}", dict(theta=th, shift=sh, seed=10 + i))
            for i, (th, sh) in enumerate(
                [
                    (0.0, (40.0, -25.0)),
                    (3.0, (60.0, 30.0)),
                    (15.0, (-80.0, 50.0)),
                    (90.0, (20.0, 20.0)),
                    (180.0, (-30.0, 10.0)),
                ]
            )
        ]
        + [
            (f"smooth_{i}", dict(amp=a, wavelength=wl, seed=20 + i))
            for i, (a, wl) in enumerate(
                [
                    (3.0, 2000.0),
                    (8.0, 1500.0),
                    (15.0, 1500.0),
                    (15.0, 600.0),
                    (30.0, 1500.0),
                    (60.0, 2500.0),
                ]
            )
        ]
        + [
            (f"mixed_{i}", dict(theta=th, shift=sh, amp=a, wavelength=wl, seed=30 + i))
            for i, (th, sh, a, wl) in enumerate(
                [
                    (2.0, (30.0, -20.0), 8.0, 1500.0),
                    (5.0, (-50.0, 40.0), 15.0, 1000.0),
                    (30.0, (60.0, 60.0), 15.0, 1500.0),
                    (180.0, (10.0, -10.0), 30.0, 2000.0),
                ]
            )
        ]
        + [
            (
                f"noisy_{i}",
                dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, noise=nz, seed=40 + i),
            )
            for i, nz in enumerate([0.05, 0.15, 0.4])
        ]
        + [
            (
                f"dropout_{i}",
                dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, dropout=d, seed=50 + i),
            )
            for i, d in enumerate([0.1, 0.3, 0.6])
        ]
        + [
            (
                f"sparse_{i}",
                dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, keep=k, seed=60 + i),
            )
            for i, k in enumerate([0.4, 0.15])
        ]
        + [
            (
                f"dim_{i}",
                dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, gain=g, seed=70 + i),
            )
            for i, g in enumerate([0.4, 0.1])
        ]
        + [
            (f"multiscale_{i}", dict(**_POSE, field="multiscale", rms=r, seed=80 + i))
            for i, r in enumerate([(4.0, 1.5, 0.5), (8.0, 3.0, 1.0), (16.0, 6.0, 2.0)])
        ]
        + [
            (f"grid_{i}", dict(**_POSE, field="grid", grid_sd=sd, seed=90 + i))
            for i, sd in enumerate([3.0, 5.0, 7.0])
        ]
        # sigma 25 px is finer than STARE's 64 px vector lattice: a case it cannot represent
        + [
            (
                f"bumps_{i}",
                dict(**_POSE, field="bumps", bump_sigma=sg, bump_count=k, seed=110 + i),
            )
            for i, (sg, k) in enumerate([(25.0, 60), (100.0, 20), (400.0, 4)])
        ]
        + [
            (f"seams_{i}", dict(**_POSE, field="seams", seam_sd=sd, seed=120 + i))
            for i, sd in enumerate([1.5, 3.0, 6.0])
        ]
    ),
    # Development cases: what method options may be chosen on. Their
    # seeds are used nowhere else and the scorer leaves `dev_*` cases out of every table.
    "dev": _suite(
        [
            ("dev_wave", dict(**_POSE, amp=10.0, seed=900)),
            ("dev_multiscale", dict(**_POSE, field="multiscale", seed=901)),
            ("dev_grid", dict(**_POSE, field="grid", seed=902)),
            ("dev_bumps", dict(**_POSE, field="bumps", seed=903)),
            ("dev_seams", dict(**_POSE, field="seams", seed=904)),
            ("dev_dropout", dict(**_POSE, amp=10.0, dropout=0.3, seed=905)),
        ]
    ),
}
FIELDS = ("wave", "multiscale", "grid", "bumps", "seams")
# A bump is summed out to this many sigma: beyond, it is under 1.3e-14 of its peak.
BUMP_REACH = 8.0
TRUTH_SUITE = "truth"
TRUTH_WINDOW = "blob"  # the synthetic scene's name in a patient id


def truth_suite(spec=None, n=None, families=None):
    """``{case_id: generator spec}`` of the frozen known-truth cases: one patient per
    family (its own blob scene), one round per pose.

    ``n`` and ``families`` make a reduced copy for a test; a reduced copy is not the frozen
    suite and its cases carry no spec hash.
    """
    spec = spec or truth.load_spec()
    names = truth.family_names(spec)
    frozen = n is None and families is None
    out = {}
    for family in names:
        if families is not None and family not in families:
            continue
        # the family's own index, not its position: see truth.family_index
        j = truth.family_index(spec, family)
        scene_seed = spec["synthetic"]["scene_seed"] + 100 * j
        patient = f"{TRUTH_WINDOW}_{family}"
        for k, pose in enumerate(truth.pose_names(spec)):
            out[f"{patient}_{pose}"] = {
                **_BASE,
                "n": int(n or spec["n_px"]),
                "noise": spec["synthetic"]["noise"],
                "family": family,
                "pose": pose,
                "patient": patient,
                "scene_seed": scene_seed,
                "seed": scene_seed + 1 + k,
                **truth.pose_kwargs(spec, pose),
                **truth.family_kwargs(spec, family, spec["pixel_size_um"], n),
                **truth.case_extra(spec, family, frozen),
            }
    return out


def suite_specs(name):
    """The cases of a suite by name; ``truth`` is read from the frozen spec."""
    return truth_suite() if name == TRUTH_SUITE else SUITES[name]


def _spline_field(rng, n, spacing, sd):
    """A random field from i.i.d. N(0, sd) nodes ``spacing`` px apart, cubic-interpolated."""
    from scipy.ndimage import spline_filter

    m = int(math.ceil(n / spacing)) + 3
    coef = [
        spline_filter(rng.normal(0.0, sd, (m, m)), order=3, mode="nearest")
        for _ in (0, 1)
    ]

    def f(x, y):
        from scipy.ndimage import map_coordinates

        at = [y / spacing + 1.0, x / spacing + 1.0]
        return [
            map_coordinates(c, at, order=3, mode="nearest", prefilter=False)
            for c in coef
        ]

    return f


def _bump_sum(x, y, cen, ax, ay, sg):
    """Sum of Gaussian bumps at the points ``(x, y)``: each bump counts out to
    ``BUMP_REACH`` sigma and is exactly zero beyond.

    The rule is per point, so a point gets the same value in whatever batch it is asked
    for (a tile of the image, or one probe). The points are taken a block at a time and
    each block meets only the bumps that can reach it: thousands of bumps on a 8192 px
    slide cost what the few near each block cost.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    shape = np.broadcast(x, y).shape
    xf, yf = (np.broadcast_to(v, shape).ravel() for v in (x, y))
    dx, dy = np.full(xf.shape, np.nan), np.full(xf.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(xf) & np.isfinite(yf))
    if ok.size:
        reach = BUMP_REACH * sg
        block = max(512.0, 4.0 * reach)
        kx = np.floor(xf[ok] / block).astype(np.int64)
        ky = np.floor(yf[ok] / block).astype(np.int64)
        _, group = np.unique(np.stack([ky, kx], axis=1), axis=0, return_inverse=True)
        group = group.ravel()
        order = np.argsort(group, kind="stable")
        cuts = np.flatnonzero(np.diff(group[order])) + 1
        for part in np.split(order, cuts):
            idx = ok[part]
            px, py = xf[idx], yf[idx]
            near = (
                (cen[:, 0] >= px.min() - reach)
                & (cen[:, 0] <= px.max() + reach)
                & (cen[:, 1] >= py.min() - reach)
                & (cen[:, 1] <= py.max() + reach)
            )
            sx, sy = np.zeros(len(idx)), np.zeros(len(idx))
            for i in np.flatnonzero(near):  # ascending: one order of summation
                d2 = (px - cen[i, 0]) ** 2 + (py - cen[i, 1]) ** 2
                e = np.where(d2 <= reach * reach, np.exp(-d2 / (2.0 * sg * sg)), 0.0)
                sx, sy = sx + ax[i] * e, sy + ay[i] * e
            dx[idx], dy[idx] = sx, sy
    return dx.reshape(shape), dy.reshape(shape)


def extra_field(spec):
    """``w(x, y) -> (dx, dy)`` of ``spec["field"]``, or None for the plain two-wave field."""
    kind, n = spec.get("field", "wave"), spec["n"]
    if kind == "wave":
        return None
    if kind not in FIELDS:
        raise ValueError(f"unknown field {kind!r}; expected one of {FIELDS}")
    rng = np.random.default_rng(spec["seed"] + 15485863)
    if kind == "multiscale":
        probe = np.linspace(0, n - 1, 96)
        px, py = np.meshgrid(probe, probe)
        parts = []
        for length, rms in zip(spec["scales"], spec["rms"]):
            f = _spline_field(rng, n, float(length), 1.0)
            # the interpolant of unit nodes has RMS below 1: scale each scale to its target
            k = [rms / max(float(np.sqrt(np.mean(v**2))), 1e-9) for v in f(px, py)]
            parts.append((f, k))

        def w(x, y):
            dx = dy = 0.0
            for f, k in parts:
                vx, vy = f(x, y)
                dx, dy = dx + k[0] * vx, dy + k[1] * vy
            return dx, dy

        return w
    if kind == "grid":
        nodes = int(spec.get("grid_nodes", 9))
        return _spline_field(rng, n, n / (nodes - 1.0), float(spec["grid_sd"]))
    if kind == "bumps":
        sg, k = float(spec["bump_sigma"]), int(spec["bump_count"])
        cen = rng.uniform(0, n, (k, 2))
        peak = np.minimum(rng.uniform(0.2, 0.6, k) * sg, 50.0)
        ang = rng.uniform(0, 2 * math.pi, k)
        ax, ay = peak * np.cos(ang), peak * np.sin(ang)

        def w(x, y):
            return _bump_sum(x, y, cen, ax, ay, sg)

        return w
    t = int(spec["seam_tile"])
    m = int(math.ceil(n / t))
    shifts = rng.normal(0.0, float(spec["seam_sd"]), (m, m, 2))

    def w(x, y):
        ix = np.clip(np.floor(np.asarray(x) / t).astype(np.int64), 0, m - 1)
        iy = np.clip(np.floor(np.asarray(y) / t).astype(np.int64), 0, m - 1)
        return shifts[iy, ix, 0], shifts[iy, ix, 1]

    return w


def field_stats(spec, pixel_size_um, step=8, threshold_um=1.0):
    """What a case's non-rigid field does to its window, on a lattice every ``step`` px:

    ``moved_frac_over_1um``  share of the window the field ``w`` moves by more than
                             ``threshold_um`` (the pose is not counted)
    ``min_jacobian``         smallest determinant of the Jacobian of the whole map ``g``,
                             by central differences; a value <= 0 is a fold
    ``fold_pct``             share of the lattice where it is <= 0
    """
    n = spec["n"]
    at = np.arange(step / 2.0, n, step)
    xx, yy = np.meshgrid(at, at)
    q = forward_map(spec)(np.stack([xx, yy], axis=-1))
    rigid = forward_map({**spec, "amp": 0.0, "field": "wave"})(
        np.stack([xx, yy], axis=-1)
    )
    moved = np.hypot(*np.moveaxis(q - rigid, -1, 0)) * float(pixel_size_um)
    det = np.gradient(q[..., 0], step, axis=1) * np.gradient(
        q[..., 1], step, axis=0
    ) - np.gradient(q[..., 0], step, axis=0) * np.gradient(q[..., 1], step, axis=1)
    return {
        "moved_frac_over_1um": float(np.mean(moved > threshold_um)),
        "min_jacobian": float(det.min()),
        "fold_pct": float(100.0 * np.mean(det <= 0)),
    }


def settle(spec, pixel_size_um):
    """The spec a known-truth case is really made with, and the record of how its seed
    was settled: ``(spec, record)``.

    A spec carrying a re-seed rule (``spec["reseed"]``, from ``truth_spec.json`` through
    ``truth.case_extra``) is measured before anything is drawn: the smallest Jacobian
    determinant of its true map on a lattice every ``lattice_px`` px. Above
    ``min_jacobian`` the seed is used. Otherwise the map folds, or comes closer to
    folding than a lattice can tell apart from it, and the seed is skipped for
    ``seed + step``, up to ``max_tries`` seeds. The answer depends on the spec alone, so
    every process that prepares the case settles on the same seed.

    The record: ``seed_asked``, ``seed`` (the one used), ``skipped`` (each skipped seed
    with its determinant, in order), ``step``, ``lattice_px``, ``min_jacobian`` (of the
    map used) and ``min_jacobian_required``. A spec without the rule comes back as it
    is, with None.
    """
    rule = spec.get("reseed")
    if not rule:
        return spec, None
    asked, skipped = int(spec["seed"]), []
    for attempt in range(int(rule["max_tries"])):
        seed = asked + attempt * int(rule["step"])
        trial = {**spec, "seed": seed}
        low = field_stats(trial, pixel_size_um, step=rule["lattice_px"])["min_jacobian"]
        if low > rule["min_jacobian"]:
            return trial, {
                "seed_asked": asked,
                "seed": seed,
                "skipped": skipped,
                "step": int(rule["step"]),
                "lattice_px": rule["lattice_px"],
                "min_jacobian": low,
                "min_jacobian_required": rule["min_jacobian"],
            }
        skipped.append({"seed": seed, "min_jacobian": low})
    raise SystemExit(
        f"{spec.get('family')} case with seed {asked}: the true map folds (smallest "
        f"Jacobian determinant not above {rule['min_jacobian']} on a "
        f"{rule['lattice_px']} px lattice) for all {rule['max_tries']} seeds tried: "
        + ", ".join(f"{s['seed']} ({s['min_jacobian']:.3f})" for s in skipped)
        + ". That is not chance: the family's amplitude or count makes it fold."
    )


def known_truth_field(spec, pixel_size_um):
    """``{"field": field_stats}`` for a known-truth case (one with a pose), else nothing:
    each such case says in its own ``case.json`` how much of it moves and that its true
    map does not fold, measured on the map it was really made with."""
    if not spec.get("pose"):
        return {}
    return {"field": {"lattice_px": 8, **field_stats(spec, pixel_size_um, step=8)}}


def forward_map(spec):
    """``g(xy (N, 2)) -> (N, 2)``: where a moving pixel belongs in the reference."""
    n = spec["n"]
    c = (n - 1) / 2.0
    th = math.radians(spec["theta"])
    cos, sin = math.cos(th), math.sin(th)
    tx, ty = spec["shift"]
    amp, k = spec["amp"], 2.0 * math.pi / spec["wavelength"]
    ph = np.random.default_rng(spec["seed"] + 7919).uniform(0, 2 * math.pi, 4)
    extra = extra_field(spec)

    def g(xy):
        xy = np.asarray(xy, dtype=float)
        x, y = xy[..., 0], xy[..., 1]
        gx = c + cos * (x - c) - sin * (y - c) + tx
        gy = c + sin * (x - c) + cos * (y - c) + ty
        if amp:
            gx = gx + amp * np.sin(k * x + ph[0]) * np.cos(k * y + ph[1])
            gy = gy + amp * np.cos(k * x + ph[2]) * np.sin(k * y + ph[3])
        if extra is not None:
            dx, dy = extra(x, y)
            gx, gy = gx + dx, gy + dy
        return np.stack([gx, gy], axis=-1)

    return g


class Scene:
    """The nuclei and tissue of one case, cheap to rebuild from the seed in any process."""

    def __init__(self, spec):
        from scipy.ndimage import gaussian_filter, zoom

        self.spec = spec
        n = self.n = spec["n"]
        # the tissue and the reference belong to the patient, the map to the round
        self.scene_seed = spec.get("scene_seed", spec["seed"])
        rng = np.random.default_rng(self.scene_seed)
        # The silhouette is kept at <= 2048 px a side and looked up by index, so a 65536 px
        # slide does not need a 4 GB mask.
        m = self.m = min(n, 2048)
        g = zoom(gaussian_filter(rng.random((32, 32)), 2), m / 32, order=1)[:m, :m]
        self.mask = g > np.percentile(g, 100 * (1 - spec["keep"]))
        k = n * n // 900
        ys, xs = rng.integers(0, n, k), rng.integers(0, n, k)
        inside = self.on_tissue(xs, ys)
        box = spec.get("tissue_box")
        if box:  # tissue that does not fill the canvas: (x0, y0, x1, y1), reference px
            inside &= (xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3])
        ys, xs = ys[inside], xs[inside]
        amp = rng.uniform(0.5, 1.5, len(ys)).astype(np.float32)
        kept = rng.random(len(ys)) >= spec["dropout"]
        order = np.argsort(
            ys, kind="stable"
        )  # sorted by row: a patch is one searchsorted
        self.ref = (ys[order], xs[order], amp[order])
        self.mov = tuple(v[order][kept[order]] for v in (ys, xs, amp))
        # a reference that is not square: its top-left (H, W) of the n x n scene
        self.ref_hw = tuple(spec.get("ref_hw") or (n, n))
        if max(self.ref_hw) > n:
            raise ValueError(f"ref_hw {self.ref_hw} does not fit the scene (n = {n})")
        self.mov_hw = tuple(spec.get("mov_hw") or self.ref_hw)
        self.g = forward_map(spec)

    def on_tissue(self, x, y):
        s = self.m / self.n
        iy = np.clip((np.asarray(y) * s).astype(np.int64), 0, self.m - 1)
        ix = np.clip((np.asarray(x) * s).astype(np.int64), 0, self.m - 1)
        return self.mask[iy, ix]

    def patch(self, which, x0, y0, x1, y1):
        """The noiseless nuclei image on ``[y0, y1) x [x0, x1)``, exactly as a whole-slide
        render would give it: the Gaussian is truncated at 4 sigma, so nuclei farther than
        that from the window cannot contribute and the window is rendered with that halo."""
        from scipy.ndimage import gaussian_filter

        ys, xs, amp = self.ref if which == "ref" else self.mov
        ax0, ay0, ax1, ay1 = x0 - _PAD, y0 - _PAD, x1 + _PAD, y1 + _PAD
        lo, hi = np.searchsorted(ys, ay0, "left"), np.searchsorted(ys, ay1, "left")
        sel = (xs[lo:hi] >= ax0) & (xs[lo:hi] < ax1)
        im = np.zeros((ay1 - ay0, ax1 - ax0), np.float32)
        np.add.at(im, (ys[lo:hi][sel] - ay0, xs[lo:hi][sel] - ax0), amp[lo:hi][sel])
        im = gaussian_filter(im, NUCLEUS_SIGMA, mode="constant")
        # a unit impulse peaks at PEAK
        return im[_PAD:-_PAD, _PAD:-_PAD] * (PEAK * 2.0 * math.pi * NUCLEUS_SIGMA**2)

    def tile(self, which, x0, y0, x1, y1):
        """One uint16 tile of the reference or the moving slide, noise included."""
        from scipy.ndimage import map_coordinates

        spec = self.spec
        if which == "ref":
            img = self.patch("ref", x0, y0, x1, y1)
        else:
            yy = np.arange(y0, y1, dtype=float)[:, None]
            xx = np.arange(x0, x1, dtype=float)[None, :]
            q = self.g(np.stack(np.broadcast_arrays(xx, yy), axis=-1))
            bx0, by0 = (int(math.floor(q[..., i].min())) - _MARGIN for i in (0, 1))
            bx1, by1 = (int(math.ceil(q[..., i].max())) + _MARGIN + 1 for i in (0, 1))
            base = self.patch("mov", bx0, by0, bx1, by1)
            img = spec["gain"] * map_coordinates(
                base, [q[..., 1] - by0, q[..., 0] - bx0], order=3, mode="constant"
            )
        sig = spec["noise"] * PEAK
        seed = spec["seed"] if which == "mov" else self.scene_seed
        rng = np.random.default_rng([seed, which == "mov", y0, x0])
        # a pedestal, so the noise is not clipped to half-normal at zero
        img = img + 4.0 * sig + rng.normal(0, sig, img.shape).astype(np.float32)
        return img.clip(0, 65535).astype(np.uint16)

    def shape(self, which):
        return self.ref_hw if which == "ref" else self.mov_hw


def tile_boxes(hw, tile=TILE):
    h, w = hw
    return [
        (x0, y0, min(x0 + tile, w), min(y0 + tile, h))
        for y0 in range(0, h, tile)
        for x0 in range(0, w, tile)
    ]


_SCENE = None


def _init(spec):
    global _SCENE
    _SCENE = Scene(
        spec
    )  # rebuilt from the seed in each worker: nothing large is pickled


def _job(job):
    which, box = job
    return _SCENE.tile(which, *box)


def _tiles(spec, which, workers, tile=TILE):
    """Tiles of one slide in row-major order, generated a tile row at a time."""
    scene = Scene(spec)
    boxes = tile_boxes(scene.shape(which), tile)
    per_row = len(range(0, scene.shape(which)[1], tile))
    if workers > 1 and len(boxes) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(workers, _init, (spec,)) as pool:
            for i in range(0, len(boxes), per_row):
                yield from pool.map(_job, [(which, b) for b in boxes[i : i + per_row]])
    else:
        for b in boxes:
            yield scene.tile(which, *b)


def make_pair(spec, tile=TILE):
    """``(ref, mov, scene)`` in memory: uint16 slides (small sizes, tests)."""
    scene = Scene(spec)
    out = []
    for which in ("ref", "mov"):
        img = np.empty(scene.shape(which), np.uint16)
        for x0, y0, x1, y1 in tile_boxes(scene.shape(which), tile):
            img[y0:y1, x0:x1] = scene.tile(which, x0, y0, x1, y1)
        out.append(img)
    return out[0], out[1], scene


def _write_slide(spec, which, path, workers):
    scene = Scene(spec)
    write_nuclear_tiles(
        path, scene.shape(which), np.uint16, _tiles(spec, which, workers), TILE
    )
    return scene


def write_pair(spec, ref_path, mov_path, workers=1):
    """Write both slides tile by tile: memory is one row of tiles, whatever the slide size."""
    _write_slide(spec, "ref", ref_path, workers)
    return _write_slide(spec, "mov", mov_path, workers)


def shared_reference(cases_root, dataset, patient):
    """Where the one reference slide of a patient with several rounds is kept."""
    return Path(cases_root) / "_cache" / dataset / str(patient) / "ref.ome.tif"


def write_once(path, write):
    """``write(tmp_path)`` then rename, unless ``path`` exists: two array tasks preparing
    two rounds of one patient both find a complete file."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"tmp{os.getpid()}_{path.name}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def prepare_one(cases_root, name, spec, workers=1, force=False):
    d = case_dir(cases_root, "synthetic", name)
    if (d / "case.json").exists() and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    # a known-truth case whose true map folds is made with the next seed that does not
    spec, reseed = settle(spec, PIXEL_SIZE_UM)
    patient = spec.get("patient")
    ref_p, mov_p = d / "ref.ome.tif", d / "mov.ome.tif"
    if patient:
        ref_p = write_once(
            shared_reference(cases_root, "synthetic", patient),
            lambda tmp: _write_slide(spec, "ref", tmp, workers),
        )
    else:
        _write_slide(spec, "ref", ref_p, workers)
    scene = _write_slide(spec, "mov", mov_p, workers)
    seg = dict(diameter=NUCLEUS_DIAMETER, workers=workers)
    mov_cells, ref_cells = (
        cells.segment_slide(mov_p, **seg),
        cells.segment_slide(ref_p, **seg),
    )
    cap = spec.get("max_cells")
    if cap and len(mov_cells) > cap:
        # only the moving side is thinned: each sampled cell still meets ALL reference cells
        keep = np.sort(
            np.random.default_rng(0).choice(len(mov_cells), cap, replace=False)
        )
        mov_cells = mov_cells.take(keep)
    hw = [spec["n"], spec["n"]]
    case = Case(
        dataset="synthetic",
        case_id=name,
        group=patient or name,
        family=spec["family"],
        pose=spec.get("pose", ""),
        ref_nuclear=str(ref_p),
        mov_nuclear=str(mov_p),
        ref_image=str(ref_p),
        mov_image=str(mov_p),
        modality="fluorescence",
        diagonal=math.hypot(spec["n"], spec["n"]),
        pixel_size_um=PIXEL_SIZE_UM,
        ref_hw=hw,
        mov_hw=list(scene.mov_hw),
        extra={
            "spec": {k: v for k, v in spec.items()},
            "in_stare_model": spec.get("in_stare_model", ""),
            "truth_spec_sha256": spec.get("truth_spec_sha256", ""),
            **known_truth_field(spec, PIXEL_SIZE_UM),
            **({"reseed": reseed} if reseed else {}),
        },
    )
    g = forward_map(spec)
    # Paired probes for the known-truth error: moving nuclei and where the map puts them.
    _, centroids = cells.area_centroid(mov_cells)
    probes = truth.probe_arrays(centroids, g, hw, spec["seed"])
    # Where a perfect registration puts the moving outlines: scoring them gives the cell
    # metric's ceiling for this pair (noise, resampling and dropout, no registration error).
    return write_case(
        cases_root,
        case,
        probes["truth_mov"],
        probes["truth_ref"],
        mov_cells,
        ref_cells,
        truth_cell=probes["truth_cell"],
        cells_mov_xy_truth=g(mov_cells.xy).astype(np.float32),
    )


def prepare(a):
    suite = suite_specs(a.suite)
    names = sorted(suite)
    if a.index is not None:
        names = names[a.index : a.index + 1]
    for name in names:
        d = prepare_one(a.cases, name, suite[name], workers=a.workers, force=a.force)
        print(f"synthetic/{name} -> {d}")
    return 0

"""Synthetic nuclear slides with a KNOWN moving -> reference map.

The reference is a field of Gaussian nuclei inside a blobby tissue silhouette. The moving
slide is the same tissue seen through a forward map

    g(p) = c + R(theta) (p - c) + t + w(p)            moving pixel p  ->  reference position

with ``w`` a displacement field: ``mov(p) = base(g(p))``. ``w`` is a smooth two-wave field of
peak amplitude ``amp`` px and wavelength ``wavelength`` px, plus, by ``field``:

``multiscale``  three random fields of correlation length ``scales`` px and RMS ``rms`` px
                each, summed (energy at several scales at once)
``grid``        a 9 x 9 grid of i.i.d. Gaussian control-point moves of SD ``grid_sd`` px,
                cubic-spline interpolated (Casamitjana et al., arXiv:2104.14873)
``bumps``       ``bump_count`` local Gaussian bumps of width ``bump_sigma`` px, each peaking at
                0.2-0.6 sigma (at most 50 px), so no single bump folds the map
``seams``       an independent shift of SD ``seam_sd`` px per ``seam_tile`` px tile: the jump a
                stitching error leaves at a tile boundary. Not continuous, on purpose.
 Because ``g`` is written in the direction a
registration is scored in, the truth for any moving point is one evaluation of ``g``, with no
inversion and no interpolation.

Two things besides geometry differ between the slides, as between two staining rounds:
independent noise, and (``dropout``) a fraction of nuclei missing from the moving slide.
``mov_hw`` gives the moving slide a smaller canvas than the reference (its top-left
``(H, W)`` pixels), because real pairs rarely share a shape and methods that pad or rescale
must get the bookkeeping right.
"""

from __future__ import annotations

import math

import numpy as np

from .. import cells
from ..cases import Case, case_dir, write_case
from ..imageio import write_nuclear_tiles

NUCLEUS_SIGMA = 5.0
NUCLEUS_DIAMETER = 12.0  # what the segmenter is told; ~FWHM of a sigma-5 blob
PEAK = 20000.0
TILE = 1024  # generation and TIFF tile; also the unit the noise is seeded by
_PAD = int(4.0 * NUCLEUS_SIGMA + 0.5) + 1  # scipy truncates the Gaussian at 4 sigma
_MARGIN = 16  # around a resampled patch, so the cubic prefilter's edge transient has died out
SCALE_SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)

_BASE = dict(n=4096, theta=0.0, shift=(0.0, 0.0), amp=0.0, wavelength=1500.0, noise=0.02,
             dropout=0.0, gain=1.0, keep=0.7, seed=0, mov_hw=None, max_cells=None,
             field="wave", scales=(2000.0, 500.0, 125.0), rms=(8.0, 3.0, 1.0), grid_sd=5.0,
             bump_sigma=100.0, bump_count=20, seam_tile=1024, seam_sd=3.0)
_POSE = dict(theta=2.0, shift=(30.0, -20.0))  # the rigid part every non-rigid family shares


def _suite(rows):
    return {name: {**_BASE, "family": name.rsplit("_", 1)[0].rstrip("0123456789_"), **kw}
            for name, kw in rows}


SUITES = {
    # One deformation at every size from 1024 to 65536 px a side (1 Mpx to 4.3 Gpx per
    # slide): the same rotation, shift and smooth warp, so what changes down the suite is
    # the slide, and accuracy and cost can be read against size. Names sort by size.
    "scale": _suite([
        (f"scale_{n:05d}", dict(n=n, theta=1.0, shift=(30.0, -20.0), amp=10.0, wavelength=1500.0,
                                seed=100, max_cells=200_000)) for n in SCALE_SIZES]),
    # What the test-suite registers: seconds per case, one with a smaller moving slide, and
    # two that pin the conditions under which STARE 1.2.1's SOLVE picks a flat field.
    "test": _suite([
        ("rigid", dict(n=1024, theta=2.0, shift=(14.0, -9.0), seed=1)),
        ("smooth", dict(n=1024, amp=5.0, wavelength=600.0, seed=4)),
        ("crop", dict(n=1024, theta=1.0, shift=(10.0, 6.0), amp=4.0, wavelength=600.0,
                      mov_hw=(832, 928), seed=4)),
        ("smooth_border", dict(n=1024, amp=5.0, wavelength=600.0, seed=2)),
    ]),
    # Small enough for a CI runner: one rigid, one smooth, one both.
    "ci": _suite([
        ("rigid_0", dict(n=1536, theta=2.0, shift=(18.0, -11.0), seed=1)),
        ("smooth_0", dict(n=1536, amp=6.0, wavelength=700.0, seed=2)),
        ("mixed_0", dict(n=1536, theta=-1.5, shift=(-9.0, 14.0), amp=5.0, wavelength=700.0,
                         dropout=0.1, seed=3)),
        ("crop_0", dict(n=1536, theta=1.0, shift=(12.0, 8.0), amp=4.0, wavelength=700.0,
                        mov_hw=(1280, 1408), seed=4)),
    ]),
    "full": _suite(
        [(f"rigid_{i}", dict(theta=th, shift=sh, seed=10 + i)) for i, (th, sh) in enumerate(
            [(0.0, (40.0, -25.0)), (3.0, (60.0, 30.0)), (15.0, (-80.0, 50.0)),
             (90.0, (20.0, 20.0)), (180.0, (-30.0, 10.0))])]
        + [(f"smooth_{i}", dict(amp=a, wavelength=wl, seed=20 + i)) for i, (a, wl) in enumerate(
            [(3.0, 2000.0), (8.0, 1500.0), (15.0, 1500.0), (15.0, 600.0), (30.0, 1500.0),
             (60.0, 2500.0)])]
        + [(f"mixed_{i}", dict(theta=th, shift=sh, amp=a, wavelength=wl, seed=30 + i))
           for i, (th, sh, a, wl) in enumerate(
               [(2.0, (30.0, -20.0), 8.0, 1500.0), (5.0, (-50.0, 40.0), 15.0, 1000.0),
                (30.0, (60.0, 60.0), 15.0, 1500.0), (180.0, (10.0, -10.0), 30.0, 2000.0)])]
        + [(f"noisy_{i}", dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, noise=nz, seed=40 + i))
           for i, nz in enumerate([0.05, 0.15, 0.4])]
        + [(f"dropout_{i}", dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, dropout=d,
                                 seed=50 + i)) for i, d in enumerate([0.1, 0.3, 0.6])]
        + [(f"sparse_{i}", dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, keep=k, seed=60 + i))
           for i, k in enumerate([0.4, 0.15])]
        + [(f"dim_{i}", dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, gain=g, seed=70 + i))
           for i, g in enumerate([0.4, 0.1])]
        + [(f"multiscale_{i}", dict(**_POSE, field="multiscale", rms=r, seed=80 + i))
           for i, r in enumerate([(4.0, 1.5, 0.5), (8.0, 3.0, 1.0), (16.0, 6.0, 2.0)])]
        + [(f"grid_{i}", dict(**_POSE, field="grid", grid_sd=sd, seed=90 + i))
           for i, sd in enumerate([3.0, 5.0, 7.0])]
        # sigma 25 px is finer than STARE's 64 px vector lattice: a case it cannot represent
        + [(f"bumps_{i}", dict(**_POSE, field="bumps", bump_sigma=sg, bump_count=k, seed=110 + i))
           for i, (sg, k) in enumerate([(25.0, 60), (100.0, 20), (400.0, 4)])]
        + [(f"seams_{i}", dict(**_POSE, field="seams", seam_sd=sd, seed=120 + i))
           for i, sd in enumerate([1.5, 3.0, 6.0])]
    ),
    # Development cases: what `regbench calibrate` chooses each method's options on. Their
    # seeds are used nowhere else and the scorer leaves `dev_*` cases out of every table.
    "dev": _suite([
        ("dev_wave", dict(**_POSE, amp=10.0, seed=900)),
        ("dev_multiscale", dict(**_POSE, field="multiscale", seed=901)),
        ("dev_grid", dict(**_POSE, field="grid", seed=902)),
        ("dev_bumps", dict(**_POSE, field="bumps", seed=903)),
        ("dev_seams", dict(**_POSE, field="seams", seed=904)),
        ("dev_dropout", dict(**_POSE, amp=10.0, dropout=0.3, seed=905)),
    ]),
}
FIELDS = ("wave", "multiscale", "grid", "bumps", "seams")


def _spline_field(rng, n, spacing, sd):
    """A random field from i.i.d. N(0, sd) nodes ``spacing`` px apart, cubic-interpolated."""
    from scipy.ndimage import spline_filter

    m = int(math.ceil(n / spacing)) + 3
    coef = [spline_filter(rng.normal(0.0, sd, (m, m)), order=3, mode="nearest") for _ in (0, 1)]

    def f(x, y):
        from scipy.ndimage import map_coordinates

        at = [y / spacing + 1.0, x / spacing + 1.0]
        return [map_coordinates(c, at, order=3, mode="nearest", prefilter=False) for c in coef]

    return f


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
        return _spline_field(rng, n, n / 8.0, float(spec["grid_sd"]))
    if kind == "bumps":
        sg, k = float(spec["bump_sigma"]), int(spec["bump_count"])
        cen = rng.uniform(0, n, (k, 2))
        peak = np.minimum(rng.uniform(0.2, 0.6, k) * sg, 50.0)
        ang = rng.uniform(0, 2 * math.pi, k)
        ax, ay = peak * np.cos(ang), peak * np.sin(ang)

        def w(x, y):
            dx, dy = np.zeros(np.shape(x)), np.zeros(np.shape(x))
            for (cx, cy), bx, by in zip(cen, ax, ay):
                e = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sg * sg))
                dx, dy = dx + bx * e, dy + by * e
            return dx, dy

        return w
    t = int(spec["seam_tile"])
    m = int(math.ceil(n / t))
    shifts = rng.normal(0.0, float(spec["seam_sd"]), (m, m, 2))

    def w(x, y):
        ix = np.clip(np.floor(np.asarray(x) / t).astype(np.int64), 0, m - 1)
        iy = np.clip(np.floor(np.asarray(y) / t).astype(np.int64), 0, m - 1)
        return shifts[iy, ix, 0], shifts[iy, ix, 1]

    return w


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
        rng = np.random.default_rng(spec["seed"])
        # The silhouette is kept at <= 2048 px a side and looked up by index, so a 65536 px
        # slide does not need a 4 GB mask.
        m = self.m = min(n, 2048)
        g = zoom(gaussian_filter(rng.random((32, 32)), 2), m / 32, order=1)[:m, :m]
        self.mask = g > np.percentile(g, 100 * (1 - spec["keep"]))
        k = n * n // 900
        ys, xs = rng.integers(0, n, k), rng.integers(0, n, k)
        inside = self.on_tissue(xs, ys)
        ys, xs = ys[inside], xs[inside]
        amp = rng.uniform(0.5, 1.5, len(ys)).astype(np.float32)
        kept = rng.random(len(ys)) >= spec["dropout"]
        order = np.argsort(ys, kind="stable")  # sorted by row: a patch is one searchsorted
        self.ref = (ys[order], xs[order], amp[order])
        self.mov = tuple(v[order][kept[order]] for v in (ys, xs, amp))
        self.mov_hw = tuple(spec.get("mov_hw") or (n, n))
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
            img = spec["gain"] * map_coordinates(base, [q[..., 1] - by0, q[..., 0] - bx0],
                                                 order=3, mode="constant")
        sig = spec["noise"] * PEAK
        rng = np.random.default_rng([spec["seed"], which == "mov", y0, x0])
        # a pedestal, so the noise is not clipped to half-normal at zero
        img = img + 4.0 * sig + rng.normal(0, sig, img.shape).astype(np.float32)
        return img.clip(0, 65535).astype(np.uint16)

    def shape(self, which):
        return (self.n, self.n) if which == "ref" else self.mov_hw


def tile_boxes(hw, tile=TILE):
    h, w = hw
    return [(x0, y0, min(x0 + tile, w), min(y0 + tile, h))
            for y0 in range(0, h, tile) for x0 in range(0, w, tile)]


_SCENE = None


def _init(spec):
    global _SCENE
    _SCENE = Scene(spec)  # rebuilt from the seed in each worker: nothing large is pickled


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


def write_pair(spec, ref_path, mov_path, workers=1):
    """Write both slides tile by tile: memory is one row of tiles, whatever the slide size."""
    scene = Scene(spec)
    for which, path in (("ref", ref_path), ("mov", mov_path)):
        write_nuclear_tiles(path, scene.shape(which), np.uint16, _tiles(spec, which, workers), TILE)
    return scene


def landmarks(spec, scene, count=2000, margin=32):
    """Moving-frame probes whose true reference position is on tissue, and that position."""
    n = spec["n"]
    g = forward_map(spec)
    rng = np.random.default_rng(spec["seed"] + 104729)
    mh, mw = scene.mov_hw
    p = rng.uniform(margin, [mw - 1 - margin, mh - 1 - margin], size=(count * 8, 2))
    x = g(p)
    ok = ((x >= margin) & (x <= n - 1 - margin)).all(axis=1)
    p, x = p[ok], x[ok]
    on = scene.on_tissue(np.round(x[:, 0]), np.round(x[:, 1]))
    return p[on][:count], x[on][:count]


def prepare_one(cases_root, name, spec, workers=1, force=False):
    d = case_dir(cases_root, "synthetic", name)
    if (d / "case.json").exists() and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    ref_p, mov_p = d / "ref.ome.tif", d / "mov.ome.tif"
    scene = write_pair(spec, ref_p, mov_p, workers)
    lm_mov, lm_ref = landmarks(spec, scene)
    seg = dict(diameter=NUCLEUS_DIAMETER, workers=workers)
    mov_cells, ref_cells = cells.segment_slide(mov_p, **seg), cells.segment_slide(ref_p, **seg)
    cap = spec.get("max_cells")
    if cap and len(mov_cells) > cap:
        # only the moving side is thinned: each sampled cell still meets ALL reference cells
        keep = np.sort(np.random.default_rng(0).choice(len(mov_cells), cap, replace=False))
        mov_cells = mov_cells.take(keep)
    case = Case(
        dataset="synthetic", case_id=name, group=spec["family"], ref_nuclear=str(ref_p),
        mov_nuclear=str(mov_p), ref_image=str(ref_p), mov_image=str(mov_p),
        modality="fluorescence", diagonal=math.hypot(spec["n"], spec["n"]),
        ref_hw=[spec["n"], spec["n"]], mov_hw=list(scene.mov_hw),
        extra={"spec": {k: v for k, v in spec.items()}},
    )
    # Where a perfect registration puts the moving outlines: scoring them gives the cell
    # metric's ceiling for this pair (noise, resampling and dropout, no registration error).
    truth = forward_map(spec)(mov_cells.xy)
    return write_case(cases_root, case, lm_mov, lm_ref, mov_cells, ref_cells,
                      cells_mov_xy_truth=truth.astype(np.float32))


def prepare(a):
    suite = SUITES[a.suite]
    names = sorted(suite)
    if a.index is not None:
        names = names[a.index : a.index + 1]
    for name in names:
        d = prepare_one(a.cases, name, suite[name], workers=a.workers, force=a.force)
        print(f"synthetic/{name} -> {d}")
    return 0

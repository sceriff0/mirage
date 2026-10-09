"""Real nuclear images under a KNOWN map: the synthetic suite's geometry on real texture.

A window of a real nuclear channel is the reference, untouched. The moving slide is the same
image seen through the synthetic suite's forward map ``g`` (``datasets/synthetic.py``), so
the truth for any moving point is one evaluation of ``g``, exactly as there::

    mov(p) = image(origin + g(p)) + noise

The pixels ``g`` reaches outside the window are read from the slide around it, so the moving
slide has real tissue up to its edges.

Two variants, by where the moving pixels come from:

``same``   (default) the reference image itself. The moving slide gets fresh noise, at
           ``--noise`` times the image's own background noise (robust SD of the pixels below
           the tissue threshold); its real noise and texture are otherwise the reference's,
           resampled.
``cross``  ``--mov-image``: another staining round of the same section, ALREADY registered to
           the reference. Noise, bleaching and lost cells are then real and independent. The
           truth is ``g`` composed with that upstream registration, so its residual is a floor
           under every method's error here.

**Deformations.** Two families have their parameters from published work and make the
headline tier:

``grid``        a 9 x 9 grid of i.i.d. Gaussian control-point moves, SD drawn from 3-7 px,
                spline-interpolated (Casamitjana et al., arXiv:2104.14873)
``multiscale``  random fields at 1/8, 1/16 and 1/32 of the window (the resolutions SynthMorph
                sums, arXiv:2004.10282), scaled so the displacement's RMS is ``--residual-um``
                (2 um): the non-rigid residual measured between two scans of one section is a
                median of 1.6 um (Lotz et al., arXiv:2106.13150) to 1.9 um (Muhlich et al.
                2022). The equal split of energy between the three scales is this file's
                choice; nobody has measured the spectrum of real inter-cycle deformation.

``wave``, ``bumps`` and ``seams`` have parameters without a source and are scored in the
secondary tier, never pooled with the two above.

``--windows K`` test windows are taken where there is most tissue, each paired with every
deformation family; one more window makes the ``dev_*`` cases that ``regbench calibrate``
chooses options on. Test and development windows never overlap.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np

from .. import cells
from ..cases import Case, case_dir, write_case
from ..imageio import pixel_size_um, write_nuclear_tiles
from . import synthetic

_POSE = dict(theta=2.0, shift=(30.0, -20.0))
HEADLINE = ("grid", "multiscale")
FAMILIES = ("grid", "multiscale", "wave", "bumps", "seams")


def family_spec(family, n, px_um, residual_um, seed):
    """The deformation parameters of one case; ``px_um`` turns micrometres into pixels."""
    if family == "grid":
        return dict(field="grid", grid_sd=float(np.random.default_rng(seed).uniform(3.0, 7.0)))
    if family == "multiscale":
        # three scales, two components each, equal energy: RMS of |w| is residual_um
        rms = residual_um / px_um / math.sqrt(6.0)
        return dict(field="multiscale", scales=(n / 8.0, n / 16.0, n / 32.0), rms=(rms,) * 3)
    return {"wave": dict(amp=10.0, wavelength=1500.0), "bumps": dict(field="bumps"),
            "seams": dict(field="seams")}[family]
PROBE = 256  # side of the patches a candidate window is sampled with
_MARGIN = 16


class Source:
    """One channel of a slide, read by window; pixels outside the slide are zero."""

    def __init__(self, path, channel=0):
        from stare.slide_io import open_lazy

        self.arr, self.dtype, self.close = open_lazy(path)
        self.channel = int(channel)
        c, self.h, self.w = self.arr.shape
        if not 0 <= self.channel < c:
            raise SystemExit(f"{path}: channel {channel} out of range for C={c}")

    def read(self, x0, y0, x1, y1):
        out = np.zeros((y1 - y0, x1 - x0), self.dtype)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if cx1 > cx0 and cy1 > cy0:
            out[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = np.asarray(
                self.arr[self.channel, slice(cy0, cy1), slice(cx0, cx1)])
        return out


def pick_windows(src, n, count, threshold):
    """Origins ``(x, y)`` of the ``count`` non-overlapping ``n`` px windows with most tissue,
    best first, and the robust SD of the background pixels met while looking."""
    scored, background = [], []
    for y0 in range(0, max(src.h - n, 0) + 1, n):
        for x0 in range(0, max(src.w - n, 0) + 1, n):
            at = np.linspace(0, n - PROBE, 4).astype(int)
            patches = [src.read(x0 + dx, y0 + dy, x0 + dx + PROBE, y0 + dy + PROBE)
                       for dy in at for dx in at]
            v = np.concatenate([p.ravel() for p in patches]).astype(np.float32)
            scored.append((float((v > threshold).mean()), x0, y0))
            background.append(v[v <= threshold][::16])
    if not scored:
        raise SystemExit(f"the image is smaller than one {n} px window")
    scored.sort(key=lambda t: (-t[0], t[2], t[1]))
    bg = np.concatenate(background) if background else np.zeros(1)
    sd = 1.4826 * float(np.median(np.abs(bg - np.median(bg)))) if len(bg) else 0.0
    return [(x, y, f) for f, x, y in scored[:count]], sd


class RealScene:
    """The reference window and its moving counterpart, a tile at a time."""

    def __init__(self, spec, ref, mov, origin, noise_sd):
        self.spec, self.ref, self.mov = spec, ref, mov
        self.ox, self.oy = origin
        self.noise_sd = float(noise_sd)
        self.n = spec["n"]
        self.g = synthetic.forward_map(spec)

    def tile(self, which, x0, y0, x1, y1):
        from scipy.ndimage import map_coordinates

        if which == "ref":
            return self.ref.read(self.ox + x0, self.oy + y0, self.ox + x1, self.oy + y1)
        yy = np.arange(y0, y1, dtype=float)[:, None]
        xx = np.arange(x0, x1, dtype=float)[None, :]
        q = self.g(np.stack(np.broadcast_arrays(xx, yy), axis=-1))
        bx0, by0 = (int(math.floor(q[..., i].min())) - _MARGIN for i in (0, 1))
        bx1, by1 = (int(math.ceil(q[..., i].max())) + _MARGIN + 1 for i in (0, 1))
        base = self.mov.read(self.ox + bx0, self.oy + by0, self.ox + bx1, self.oy + by1)
        img = map_coordinates(base.astype(np.float32), [q[..., 1] - by0, q[..., 0] - bx0],
                              order=3, mode="nearest")
        if self.noise_sd:
            rng = np.random.default_rng([self.spec["seed"], y0, x0])
            img = img + rng.normal(0, self.noise_sd, img.shape).astype(np.float32)
        info = np.iinfo(self.ref.dtype) if np.issubdtype(self.ref.dtype, np.integer) else None
        if info is not None:
            img = np.rint(img).clip(info.min, info.max)
        return img.astype(self.ref.dtype)


def landmarks(spec, ref_cells, diameter, count=2000, margin=32):
    """Moving-frame probes whose true reference position is within 3 nuclear diameters of a
    reference nucleus (so they sit on tissue), and that position."""
    from scipy.spatial import cKDTree

    n = spec["n"]
    g = synthetic.forward_map(spec)
    rng = np.random.default_rng(spec["seed"] + 104729)
    p = rng.uniform(margin, n - 1 - margin, size=(count * 8, 2))
    x = g(p)
    ok = ((x >= margin) & (x <= n - 1 - margin)).all(axis=1)
    p, x = p[ok], x[ok]
    _, centroids = cells.area_centroid(ref_cells)
    near = cKDTree(centroids).query(x, distance_upper_bound=3.0 * diameter)[0] < np.inf
    return p[near][:count], x[near][:count]


def prepare_one(cases_root, name, spec, a, origin, noise_sd, px, force=False):
    d = case_dir(cases_root, "semisynth", name)
    if (d / "case.json").exists() and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    n = spec["n"]
    ref_src = Source(a.image, a.channel)
    mov_src = Source(a.mov_image, a.mov_channel) if a.mov_image else ref_src
    try:
        scene = RealScene(spec, ref_src, mov_src, origin, noise_sd)
        ref_p, mov_p = d / "ref.ome.tif", d / "mov.ome.tif"
        boxes = synthetic.tile_boxes((n, n))
        for which, path in (("ref", ref_p), ("mov", mov_p)):
            write_nuclear_tiles(path, (n, n), ref_src.dtype,
                                (scene.tile(which, *b) for b in boxes), synthetic.TILE, px)
    finally:
        ref_src.close()
        if mov_src is not ref_src:
            mov_src.close()
    seg = dict(diameter=a.diameter, workers=a.workers)
    mov_cells, ref_cells = cells.segment_slide(mov_p, **seg), cells.segment_slide(ref_p, **seg)
    if not len(ref_cells) or not len(mov_cells):
        raise RuntimeError(f"{name}: no nuclei segmented; check --channel and --diameter")
    lm_mov, lm_ref = landmarks(spec, ref_cells, a.diameter)
    truth = synthetic.forward_map(spec)(mov_cells.xy)
    case = Case(
        dataset="semisynth", case_id=name, group=spec["family"], ref_nuclear=str(ref_p),
        mov_nuclear=str(mov_p), ref_image=str(ref_p), mov_image=str(mov_p),
        modality="fluorescence", diagonal=math.hypot(n, n), pixel_size_um=px,
        ref_hw=[n, n], mov_hw=[n, n],
        extra={"spec": dict(spec), "image": str(a.image), "mov_image": str(a.mov_image or a.image),
               "variant": "cross" if a.mov_image else "same", "origin_xy": list(origin),
               "noise_sd": noise_sd},
    )
    return write_case(cases_root, case, lm_mov, lm_ref, mov_cells, ref_cells,
                      cells_mov_xy_truth=truth.astype(np.float32))


def plan(a):
    """``[(name, spec, origin)]`` and the noise SD: the dev window's cases, then the test ones."""
    src = Source(a.image, a.channel)
    try:
        threshold = cells.slide_threshold(a.image, a.channel)
        windows, bg_sd = pick_windows(src, a.n, a.windows + 1, threshold)
    finally:
        src.close()
    px = a.pixel_size_um or pixel_size_um(a.image)
    if not px:
        raise SystemExit(f"{a.image}: no pixel size in the header; give --pixel-size-um "
                         "(the multiscale deformation is defined in micrometres)")
    stem = a.name or Path(a.image).name.split(".")[0]
    key = zlib.crc32(stem.encode()) % 100_000
    # the window with the least tissue of those picked is the development one
    n_test = min(a.windows, len(windows) - 1) if len(windows) > 1 else 1
    order = [(len(windows) - 1, "dev")] if len(windows) > 1 else []
    order += [(i, "test") for i in range(n_test)]
    jobs = []
    for i, role in order:
        x, y, _ = windows[i]
        for j, family in enumerate(FAMILIES):
            seed = 1_000_000 + key * 100 + i * 10 + j
            spec = {**synthetic._BASE, **_POSE, "n": a.n, "family": family, "seed": seed,
                    **family_spec(family, a.n, px, a.residual_um, seed)}
            name = f"{stem}_w{i}_{family}"
            jobs.append((f"dev_{name}" if role == "dev" else name, spec, (x, y)))
    noise = 0.0 if a.mov_image else a.noise * bg_sd
    return jobs, noise


def prepare(a):
    jobs, noise = plan(a)
    px = a.pixel_size_um or pixel_size_um(a.image)
    if a.index is not None:
        jobs = jobs[a.index : a.index + 1]
    for name, spec, origin in jobs:
        d = prepare_one(a.cases, name, spec, a, origin, noise, px, force=a.force)
        print(f"semisynth/{name} -> {d}")
    return 0

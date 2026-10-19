"""The synthetic scaling set (T32): slides of a controlled size, for COST only.

``python -m benchmarks.regbench synth --source <slide.ome.tif> --to <CASES>`` writes the
dataset ``synthetic`` (``regbench.PUBLIC``) under ``<CASES>/synthetic/``. It measures how
the cost of STARE and VALIS grows with the size of a slide. **Accuracy is not measured on
it**: there is no truth map, no field error and no accuracy table; the transform each moving
panel was made with is written to its case's ``synth_transform.json`` for reproducibility
only, and nothing reads it for scoring.

THE DESIGN (user, 2026-10-19):

- one synthetic patient per size, square, long side ``GRIDS["default"]`` = 4096, 8192,
  16384, 32768 and 65536 px (``--grid small`` is 4096 only, a subset);
- 11 panels per patient, as the cohort has: panel 0 the reference, 1..10 moving. Each panel
  is ONE image of 3 channels named ``DAPI``, ``S1``, ``S2``, uint16, at 0.325 µm;
- the texture is real tissue: the source slide's nuclear channel, a centred crop of up to
  ``TEXTURE_SIDE`` px (resampled to 0.325 µm first when the source states another pixel
  size), mirror-tiled to the size (the crop, its mirror, the crop, ...), so nuclei keep their
  size at every slide size. A uint8 source is widened by 257 (255 -> 65535);
- the reference's DAPI is that tiled crop, untouched. S1 and S2 are the archive's perturbed
  copies (``synthesize_channels``): channel c is the texture shifted c columns, times a gain
  uniform in 1 +- 0.1, plus Gaussian noise of s.d. 3 counts, clipped;
- moving panel k is the reference moved by one fixed transform seeded by (seed, size, k):
  a rotation uniform in +-1 degree and a translation uniform in +-2 % of the long side per
  axis, both about the image centre, then a smooth deformation, the sum of 4 sine modes of
  wavelength 1 mm (3077 px) whose amplitudes sum to 5 µm (~15 px) -- the same in µm at every
  size. Every channel then gets Gaussian noise of s.d. 3 counts, clipped to uint16.

PORTED: ``compute_target_shape`` and ``synthesize_channels`` are the archive's, from
``benchmarks/generate_matrix.py`` at commit ``774e1c9d`` (tag
``archive/benchmarks-arms-sweep``), unchanged. The archive bounded memory by building the
noise in ROW BLOCKS; this module keeps that bound by rendering one TILE at a time
(``panel_tiles``), so a 65536 px panel is never held in memory: peak memory is the texture
(at most 4096 px square) plus a few tiles. The moving panels are rendered tile by tile from
the texture directly (bilinear), which the archive did not do (it had no transform).

ONE WRITE, MANY READERS. The set is generated once and shared by every run: ``plan``
refuses an existing ``<CASES>/synthetic`` folder and prints the bytes it will write before
anything is written; ``render_unit`` writes one (size, panel) slide -- one array task of
``slurm/synth.sbatch`` -- and keeps a slide already written (the write is atomic, so a
present slide is a complete one); ``finish`` writes the cases, ``case.json`` last.

The slides are single-resolution tiled OME-TIFFs, like the cohort's own preprocessed slides
(``bin/convert_image.py`` writes one level). The OME UUID is derived from (seed, size,
panel), so the same seed gives byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path

import numpy as np

DATASET = "synthetic"
GRIDS = {"default": (4096, 8192, 16384, 32768, 65536), "small": (4096,)}
CHANNELS = ("DAPI", "S1", "S2")
PIXEL_SIZE_UM = 0.325
DTYPE = np.uint16
N_MOVING = 10
N_PANELS = N_MOVING + 1
ROTATION_DEG = 1.0
SHIFT_FRACTION = 0.02
WAVELENGTH_UM = 1000.0
AMPLITUDE_UM = 5.0
N_MODES = 4
NOISE_SD = 3.0
GAIN_JITTER = 0.1  # the archive's: 1 + U(-0.1, 0.1)
TILE = 1024
TEXTURE_SIDE = 4096
SLIDES = "_slides"
PLAN = "synth_plan.json"
TRANSFORM = "synth_transform.json"
CAVEAT = (
    "moving panels carry a cohort-like rigid offset and a mild deformation so methods do "
    "real work; accuracy is not measured on this set; one repeat."
)
NUCLEAR_RE = ("dapi", "hoechst")


# ---- ported from the archive (774e1c9d:benchmarks/generate_matrix.py), unchanged ---------


def compute_target_shape(
    src_hw: tuple[int, int], target_long_edge: int
) -> tuple[int, int]:
    """Scale (height, width) so the longer edge equals target_long_edge, preserving aspect."""
    h, w = src_hw
    long_edge = max(h, w)
    if long_edge == 0:
        raise ValueError("src_hw must have positive dimensions")
    scale = target_long_edge / long_edge
    return (round(h * scale), round(w * scale))


def synthesize_channels(
    src_2d: np.ndarray, n_channels: int, seed: int = 0, block_rows: int = 4096
) -> np.ndarray:
    """Replicate a single 2-D channel into n_channels with per-channel perturbation.

    Channel 0 is the unmodified source. Channels 1..N-1 add intensity jitter,
    Gaussian noise, and a c-px roll offset (by channel index) so each channel is non-identical.

    Memory: the per-channel float64 noise/scale intermediate is built in ROW BLOCKS
    (``block_rows`` at a time), not for the whole image at once. Deterministic per seed.
    (The panels here are rendered per tile by ``panel_tiles``, which applies the same
    perturbation to the mirror-tiled texture; this function is kept as the reference.)
    """
    if src_2d.ndim != 2:
        raise ValueError("src_2d must be 2-D (H, W)")
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1")
    if not np.issubdtype(src_2d.dtype, np.integer):
        raise ValueError(f"src_2d must be an integer dtype, got {src_2d.dtype}")
    rng = np.random.default_rng(seed)
    info = np.iinfo(src_2d.dtype)
    h = src_2d.shape[0]
    out = np.empty((n_channels,) + src_2d.shape, dtype=src_2d.dtype)
    out[0] = src_2d
    for c in range(1, n_channels):
        gain = 1.0 + rng.uniform(-0.1, 0.1)
        shifted = np.roll(src_2d, shift=c, axis=1)
        for r0 in range(0, h, block_rows):
            r1 = min(r0 + block_rows, h)
            noise = rng.normal(0.0, 3.0, size=(r1 - r0, src_2d.shape[1]))
            vals = np.clip(
                shifted[r0:r1].astype(np.float64) * gain + noise, info.min, info.max
            )
            out[c, r0:r1] = vals.astype(src_2d.dtype)
            del noise, vals
        del shifted
    return out


# ---- the design ---------------------------------------------------------------------------


def estimate(sizes):
    """The bytes the set takes, uncompressed: ``{"sizes": [...], "total_bytes": n}``."""
    rows, total = [], 0
    for s in sizes:
        per_image = len(CHANNELS) * np.dtype(DTYPE).itemsize * int(s) * int(s)
        rows.append(
            {
                "size": int(s),
                "bytes_per_image": per_image,
                "bytes_per_patient": per_image * N_PANELS,
            }
        )
        total += per_image * N_PANELS
    return {"sizes": rows, "total_bytes": total}


def estimate_lines(sizes):
    est = estimate(sizes)
    lines = [
        f"synth: {len(sizes)} patient(s) x {N_PANELS} panels x {len(CHANNELS)} channels "
        f"uint16 at {PIXEL_SIZE_UM} um; uncompressed bytes it will write:"
    ]
    for r in est["sizes"]:
        lines.append(
            f"  {patient_of(r['size'])}: {r['bytes_per_image']} bytes per image "
            f"({r['bytes_per_image'] / 1e9:.1f} GB), {r['bytes_per_patient']} bytes per "
            f"patient ({r['bytes_per_patient'] / 1e9:.1f} GB)"
        )
    lines.append(
        f"  total: {est['total_bytes']} bytes ({est['total_bytes'] / 1e9:.1f} GB) "
        "uncompressed; the slides are zlib-compressed on disk"
    )
    return lines


def patient_of(size):
    return f"s{int(size):05d}"


def case_id_of(size, panel):
    return f"{patient_of(size)}_r{int(panel):02d}"


def panel_path(to, size, panel):
    return (
        Path(to)
        / DATASET
        / SLIDES
        / patient_of(size)
        / f"panel_{int(panel):02d}.ome.tif"
    )


def transform(seed, size, panel):
    """Moving panel ``panel``'s transform, seeded by (seed, size, panel): plain JSON values.

    Moving at output pixel q (row, col) is the reference texture at
    ``p = R(-rotation)(q - centre - translation) + centre + d(q)``, where ``d`` is the sum of
    the sine ``modes``: ``amplitude * (sin(2 pi (k . q) / wavelength + phase)) * u`` with the
    wave direction ``k`` and the displacement direction ``u`` unit vectors.
    """
    if not 1 <= int(panel) <= N_MOVING:
        raise ValueError(f"panel {panel}: the moving panels are 1..{N_MOVING}")
    rng = np.random.default_rng([int(seed), int(size), int(panel)])
    rot = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
    shift = SHIFT_FRACTION * int(size)
    trans = [float(v) for v in rng.uniform(-shift, shift, 2)]
    amp_px = AMPLITUDE_UM / PIXEL_SIZE_UM
    weights = rng.dirichlet(np.ones(N_MODES))
    weights = weights / weights.sum()
    modes = []
    for w in weights:
        modes.append(
            {
                "amplitude_px": float(w * amp_px),
                "wavelength_px": WAVELENGTH_UM / PIXEL_SIZE_UM,
                "direction_deg": float(rng.uniform(0.0, 360.0)),
                "displacement_deg": float(rng.uniform(0.0, 360.0)),
                "phase_rad": float(rng.uniform(0.0, 2 * math.pi)),
            }
        )
    c = (int(size) - 1) / 2.0
    return {
        "seed": int(seed),
        "size": int(size),
        "panel": int(panel),
        "pixel_size_um": PIXEL_SIZE_UM,
        "centre_px": [c, c],
        "rotation_deg": rot,
        "translation_px": trans,
        "modes": modes,
        "noise_sd": NOISE_SD,
        "convention": transform.__doc__.split("\n\n")[1].strip(),
    }


def displacement(t, yy, xx):
    """``(dy, dx)`` of the transform's sine modes at output pixels ``(yy, xx)``."""
    dy = np.zeros(np.shape(yy), dtype=np.float64)
    dx = np.zeros(np.shape(xx), dtype=np.float64)
    for m in t["modes"]:
        a = math.radians(m["direction_deg"])
        u = math.radians(m["displacement_deg"])
        s = m["amplitude_px"] * np.sin(
            2 * math.pi * (math.sin(a) * yy + math.cos(a) * xx) / m["wavelength_px"]
            + m["phase_rad"]
        )
        dy += s * math.sin(u)
        dx += s * math.cos(u)
    return dy, dx


def source_coords(t, yy, xx):
    """The reference texture coordinates ``(py, px)`` moving pixels ``(yy, xx)`` show."""
    cy, cx = t["centre_px"]
    ty, tx = t["translation_px"]
    th = math.radians(t["rotation_deg"])
    cos, sin = math.cos(th), math.sin(th)
    u = np.asarray(yy, dtype=np.float64) - cy - ty
    v = np.asarray(xx, dtype=np.float64) - cx - tx
    dy, dx = displacement(t, yy, xx)
    py = cos * u - sin * v + cy + dy
    px = sin * u + cos * v + cx + dx
    return py, px


def gains(seed, size):
    """The gains of S1 and S2 (the archive's 1 + U(-0.1, 0.1)), seeded by (seed, size)."""
    rng = np.random.default_rng([int(seed), int(size), 0])
    return [1.0] + [
        float(1.0 + rng.uniform(-GAIN_JITTER, GAIN_JITTER))
        for _ in range(len(CHANNELS) - 1)
    ]


# ---- the texture --------------------------------------------------------------------------


def _fold(i, n):
    """Mirror-periodic index: 0..n-1, then n-1..0, then 0..n-1, ..."""
    m = np.mod(i, 2 * n)
    return np.where(m >= n, 2 * n - 1 - m, m)


def bilinear(tex, py, px):
    """The mirror-tiled texture at float coordinates, bilinear, as float64."""
    h, w = tex.shape
    y0 = np.floor(py)
    x0 = np.floor(px)
    fy, fx = py - y0, px - x0
    y0, x0 = y0.astype(np.int64), x0.astype(np.int64)
    ya, yb = _fold(y0, h), _fold(y0 + 1, h)
    xa, xb = _fold(x0, w), _fold(x0 + 1, w)
    t = tex
    top = t[ya, xa] * (1 - fx) + t[ya, xb] * fx
    bot = t[yb, xa] * (1 - fx) + t[yb, xb] * fx
    return top * (1 - fy) + bot * fy


def _nuclear_index(path, channel):
    from . import imageio

    if channel is not None:
        return int(channel)
    names = [n.lower() for n in imageio.channel_names(path)]
    for i, n in enumerate(names):
        if any(k in n for k in NUCLEAR_RE):
            return i
    return 0


def texture(source, texture_side=TEXTURE_SIDE, channel=None):
    """The source's nuclear channel as a centred square crop, uint16, at 0.325 µm.

    ``channel`` None picks the first channel named DAPI or Hoechst (channel 0 when none is).
    A source that states another pixel size is resampled (bilinear) to 0.325 µm first, so a
    nucleus keeps its size in µm."""
    from stare.slide_io import open_lazy

    from . import imageio

    idx = _nuclear_index(source, channel)
    src_px = imageio.pixel_size_um(source)
    scale = (src_px / PIXEL_SIZE_UM) if src_px else 1.0
    if abs(scale - 1.0) < 0.01:
        scale = 1.0
    arr, dtype, close = open_lazy(source)
    try:
        c, h, w = arr.shape
        if not 0 <= idx < c:
            raise SystemExit(f"synth: {source}: channel {idx} out of range for C={c}")
        side_src = min(h, w, max(2, int(math.ceil(texture_side / scale))))
        y0, x0 = (h - side_src) // 2, (w - side_src) // 2
        crop = np.asarray(arr[idx, slice(y0, y0 + side_src), slice(x0, x0 + side_src)])
    finally:
        close()
    if crop.dtype == np.uint8:
        crop = crop.astype(np.uint16) * 257
    elif crop.dtype != np.uint16:
        raise SystemExit(
            f"synth: {source}: dtype {crop.dtype}; the texture must be uint8 or uint16"
        )
    if scale != 1.0:
        from scipy import ndimage

        th, tw = compute_target_shape(crop.shape, round(side_src * scale))
        zoomed = ndimage.zoom(
            crop.astype(np.float64), (th / crop.shape[0], tw / crop.shape[1]), order=1
        )
        crop = np.clip(np.rint(zoomed), 0, 65535).astype(np.uint16)
    side = min(crop.shape[0], crop.shape[1], int(texture_side))
    return np.ascontiguousarray(crop[:side, :side])


# ---- the panels ---------------------------------------------------------------------------


def _tile_noise(seed, size, panel, ch, ty, tx, shape):
    rng = np.random.default_rng([int(seed), int(size), int(panel), ch, ty, tx, 7])
    return rng.normal(0.0, NOISE_SD, size=shape)


def panel_tiles(tex, seed, size, panel, tile=TILE):
    """Every tile of the panel, channel 0 first, row-major within a channel: what
    ``imageio.write_channel_tiles`` takes. One tile is rendered at a time."""
    tex = np.asarray(tex)
    t = transform(seed, size, panel) if panel else None
    g = gains(seed, size)
    texf = tex.astype(np.float64)
    for ch in range(len(CHANNELS)):
        for ty, y0 in enumerate(range(0, size, tile)):
            y1 = min(y0 + tile, size)
            for tx, x0 in enumerate(range(0, size, tile)):
                x1 = min(x0 + tile, size)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                if t is None and ch == 0:  # the reference's DAPI: the tiled crop itself
                    yield tex[_fold(yy, tex.shape[0]), _fold(xx, tex.shape[1])]
                    continue
                if t is None:
                    py, px = yy.astype(np.float64), xx.astype(np.float64)
                else:
                    py, px = source_coords(t, yy, xx)
                val = bilinear(texf, py, px - ch) * g[ch]  # S_c: shifted c columns
                val += _tile_noise(seed, size, panel, ch, ty, tx, val.shape)
                yield np.clip(np.rint(val), 0, 65535).astype(DTYPE)
                del val, py, px


def _write_panel(path, tex, seed, size, panel, tile):
    import tifffile

    from . import imageio

    def padded():
        for a in panel_tiles(tex, seed, size, panel, tile):
            if a.shape != (tile, tile):
                full = np.zeros((tile, tile), DTYPE)
                full[: a.shape[0], : a.shape[1]] = a
                a = full
            yield a

    meta = imageio._ome_meta(CHANNELS, PIXEL_SIZE_UM)
    # a fixed UUID: tifffile draws a fresh one otherwise, and the same seed must give the
    # same bytes
    meta["UUID"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"regbench-synth/{seed}/{size}/{panel}")
    )
    return imageio._atomic_tiff(
        path,
        lambda tmp: tifffile.imwrite(
            tmp,
            padded(),
            shape=(len(CHANNELS), size, size),
            dtype=DTYPE,
            ome=True,
            bigtiff=True,
            tile=(tile, tile),
            compression="zlib",
            photometric="minisblack",
            metadata=meta,
            maxworkers=1,  # a memory bound (imageio.write_channel_tiles says why)
        ),
    )


# ---- plan / unit / finish -----------------------------------------------------------------


def _root(to):
    return Path(to) / DATASET


def _read_plan(to):
    p = _root(to) / PLAN
    if not p.exists():
        raise SystemExit(f"synth: {p} is absent; run `synth --plan` first")
    return json.loads(p.read_text())


def _write_plan(to, plan_rec, tex):
    from .cases import write_json

    slides = _root(to) / SLIDES
    slides.mkdir(parents=True, exist_ok=False)
    np.save(slides / "texture.npy", tex)
    write_json(_root(to) / PLAN, plan_rec)


def plan(
    source,
    to,
    sizes=GRIDS["default"],
    seed=0,
    tile=TILE,
    texture_side=TEXTURE_SIDE,
    channel=None,
    grid="",
):
    """Print the bytes, then fix the set: its texture and its units. Refuses an existing
    ``<to>/synthetic`` (the set is generated once and shared by every run)."""
    root = _root(to)
    if root.exists():
        raise SystemExit(
            f"synth: {root} exists. The synthetic set is generated once and shared by every "
            "run; it is never regenerated over. Remove it by hand to make a new one."
        )
    sizes = tuple(int(s) for s in sizes)
    for line in estimate_lines(sizes):
        print(line, flush=True)
    tex = texture(source, texture_side, channel)
    rec = {
        "source": str(source),
        "channel": _nuclear_index(source, channel),
        "seed": int(seed),
        "grid": grid,
        "sizes": list(sizes),
        "tile": int(tile),
        "texture_side": int(tex.shape[0]),
        "texture_sha256": hashlib.sha256(tex.tobytes()).hexdigest(),
        "units": [[s, k] for s in sizes for k in range(N_PANELS)],
        "estimate": estimate(sizes),
    }
    _write_plan(to, rec, tex)
    return rec


def n_units(to):
    return len(_read_plan(to)["units"])


def render_unit(to, i):
    """Write unit ``i``'s slide; a slide already written is kept (the write is atomic)."""
    rec = _read_plan(to)
    size, panel = rec["units"][int(i)]
    path = panel_path(to, size, panel)
    if path.exists():
        print(f"synth: {path} is already written; kept", flush=True)
        return path
    tex = np.load(_root(to) / SLIDES / "texture.npy")
    print(f"synth: writing {path} ({size} px, panel {panel})", flush=True)
    return _write_panel(path, tex, rec["seed"], size, panel, rec["tile"])


def finish(to):
    """Write the cases: one per moving panel, ``case.json`` last. Refuses a set with a
    missing slide."""
    from .cases import Case, case_dir, write_case, write_json

    rec = _read_plan(to)
    missing = [
        str(panel_path(to, s, k))
        for s, k in rec["units"]
        if not panel_path(to, s, k).exists()
    ]
    if missing:
        raise SystemExit(
            f"synth: {len(missing)} slide(s) missing (their array tasks did not finish): "
            + ", ".join(missing[:5])
        )
    for size in rec["sizes"]:
        ref = panel_path(to, size, 0)
        for k in range(1, N_PANELS):
            cid = case_id_of(size, k)
            write_json(
                case_dir(to, DATASET, cid) / TRANSFORM, transform(rec["seed"], size, k)
            )
            mov = panel_path(to, size, k)
            names = list(CHANNELS)
            case = Case(
                dataset=DATASET,
                case_id=cid,
                group=patient_of(size),
                ref_nuclear="",
                mov_nuclear="",
                ref_image=str(ref),
                mov_image=str(mov),
                modality="fluorescence",
                diagonal=math.hypot(size, size),
                pixel_size_um=PIXEL_SIZE_UM,
                ref_hw=[size, size],
                mov_hw=[size, size],
                extra={
                    "reference": str(ref),
                    "moving": str(mov),
                    "ref_channel": 0,
                    "mov_channel": 0,
                    "channels_ref": names,
                    "channels_mov": names,
                    "channel_names": names,
                    "cells": "none: cost only, accuracy is not measured on this set",
                    "synthetic": {
                        "size": size,
                        "panel": k,
                        "seed": rec["seed"],
                        "source": rec["source"],
                        "caveat": CAVEAT,
                    },
                },
            )
            write_case(to, case)
    print(f"synth: {len(rec['sizes']) * N_MOVING} cases written under {_root(to)}")


def status(to):
    """``complete`` (every case written), ``planned`` (a plan, not every case) or
    ``absent`` (no ``<to>/synthetic``); ``partial`` for a folder with no plan."""
    root = _root(to)
    if not root.exists():
        return "absent"
    if not (root / PLAN).exists():
        return "partial"
    rec = _read_plan(to)
    want = {case_id_of(s, k) for s in rec["sizes"] for k in range(1, N_PANELS)}
    have = {p.parent.name for p in root.glob("*/case.json")}
    return "complete" if want <= have else "planned"


def generate(source, to, sizes=GRIDS["default"], seed=0, **kw):
    """The whole set in one process (``synth`` without ``--plan``/``--unit``/``--finish``)."""
    plan(source, to, sizes, seed, **kw)
    for i in range(n_units(to)):
        render_unit(to, i)
    finish(to)


def main(a):
    """``regbench synth``: see the module docstring."""
    sizes = GRIDS[a.grid]
    if a.estimate:
        for line in estimate_lines(sizes):
            print(line)
        return 0
    if a.status:
        print(status(a.to))
        return 0
    if a.units:
        print(n_units(a.to))
        return 0
    if a.plan:
        plan(a.source, a.to, sizes, a.seed, channel=a.channel, grid=a.grid)
        return 0
    if a.unit is not None:
        render_unit(a.to, a.unit)
        return 0
    if a.finish:
        finish(a.to)
        return 0
    if not a.source:
        raise SystemExit("synth: --source is needed to generate")
    generate(a.source, a.to, sizes, a.seed, channel=a.channel, grid=a.grid)
    return 0

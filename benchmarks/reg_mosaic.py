#!/usr/bin/env python3
"""reg_mosaic.py -- before/after registration patch mosaic across benchmark arms.

Takes one or more ARM DIRECTORIES of the real-sample arm benchmark
(``<arm_results>/<arm>``, each a mirage ``--outdir`` that stopped at registration)
and writes, per patient, a mosaic of nuclear-channel two-colour overlays:

    rows    = (moving round, ROI) pairs -- exactly ``--rows`` of them
    columns = Before | <arm 1> | <arm 2> ...   (one column per arm directory)

Everything is read from the checkpoints the pipeline wrote, so no samplesheet
and no raw acquisitions are needed:

    <arm>/csv/registered.csv                    registered_image per moving slide
                                                (the reference row names the
                                                preprocessed reference: the frame)
    <root>/preprocess_shared/csv/preprocessed.csv   the "Before" column: every
                                                moving slide as it entered
                                                registration, put on the reference
                                                canvas at the origin with no
                                                transform (pad-or-crop, never
                                                rescaled) -- the same "before"
                                                mirage's own registration QC draws

The moving slides of different arms are matched by their channel set (the
``channels`` column), which every arm inherits from the same preprocessing run.
Every cell of a row uses the same ROI, the same pixel scale and the same
per-channel LUT. A per-cell metric (default: Dice of the Otsu nuclear masks; also
residual shift by phase correlation and NCC) is printed in the corner.

Outputs in OUTDIR:
    <patient>_mosaic.png/.pdf   the figure (PDF keeps native pixels, fonts editable)
    <patient>_locator.png/.pdf  low-res reference with numbered ROI boxes
    <patient>_rois.json         ROIs (reference frame, full-res px), files,
                                stretch limits and metrics; pass back with
                                --rois-json to reuse identical ROIs in another run
    <patient>_patches/          every cell as a PNG at native resolution
                                (+ the uint16 ref/mov crops as CYX TIFFs with --save-raw)

Recipes:
    # six rows (every moving round x two ROIs), VALIS best cell vs STARE best cell
    python -m benchmarks.reg_mosaic arm_results/valis_high_micro2 arm_results/tiled_high_gate1 \\
        --rows 6 --patient 5456 -o figs/mosaic
    # the legacy vs robust SOLVE stage on the same tiles, checkerboard too
    python -m benchmarks.reg_mosaic arm_results/tiled_high_gate1 arm_results/tiled_high_gate1_solver_robust \\
        --rows 4 --kinds overlay,checker -o figs/solver

Requires numpy, scipy, scikit-image, tifffile and matplotlib -- the benchmarks
analysis stack (requirements/segeval.txt).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu

log = logging.getLogger("reg_mosaic")

NUCLEAR_RE = re.compile(r"DAPI|HOECHST|CELLTOX", re.I)
DEFAULT_NUCLEAR = ("DAPI", "HOECHST", "CELLTOX")
REGISTERED_CSV = Path("csv") / "registered.csv"
PREPROCESSED_CSV = Path("csv") / "preprocessed.csv"
BEFORE_ARM = (
    "preprocess_shared"  # build_arm_plan.PREPROCESS_ARM: the shared preprocessing run
)
BEFORE_LABEL = "Before"

# (moving colour, reference colour); additive, so overlap = sum
PALETTES = {
    "magenta-green": ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0)),  # overlap -> white
    "red-green": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # overlap -> yellow (mirage QC)
    "cyan-magenta": ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0)),
    "cyan-red": ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
}
NICE_BARS_UM = (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)
MIN_PC_RESPONSE = 0.02  # phase-correlation peak below this = no reliable match


# --- checkpoints --------------------------------------------------------------
@dataclass
class Slide:
    """One row of a mirage checkpoint CSV."""

    patient: str
    slide_id: str
    image: Path
    is_reference: bool
    channels: list[str]  # the pipeline's channel list, e.g. ['DAPI', 'CD45', 'CD163']
    pixel_size: float | None
    index: int = 0  # position among the patient's rows

    @property
    def stains(self) -> list[str]:
        return [m for m in self.channels if not NUCLEAR_RE.search(m)]

    @property
    def key(self) -> str:
        """What identifies the same round across arms: its non-nuclear channel set."""
        return "_".join(self.stains) or self.slide_id

    @property
    def label(self) -> str:
        return " / ".join(self.stains) or self.slide_id


def _truthy(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes", "y", "t")


def _float_or_none(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_checkpoint(path: Path, image_col: str) -> dict[str, list[Slide]]:
    """A mirage checkpoint (lib/Checkpoint.groovy columns) as patient -> slides."""
    if not path.is_file():
        raise SystemExit(
            f"{path}: not found -- is this a mirage --outdir that reached registration?"
        )
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"patient_id", "id", image_col, "is_reference", "channels"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
        per: dict[str, list[Slide]] = {}
        for row in reader:
            pid = row["patient_id"].strip()
            if not pid:
                continue
            per.setdefault(pid, []).append(
                Slide(
                    patient=pid,
                    slide_id=row["id"].strip(),
                    image=Path(row[image_col].strip()),
                    is_reference=_truthy(row["is_reference"]),
                    channels=[c for c in row["channels"].split("|") if c],
                    pixel_size=_float_or_none(row.get("pixel_size", "")),
                    index=len(per.get(pid, [])),
                )
            )
    return per


def round_matches(sl: Slide, tokens) -> bool:
    for tok in tokens:
        t = tok.lower()
        if t in (sl.key.lower(), sl.slide_id.lower()) or any(
            t == s.lower() for s in sl.stains
        ):
            return True
        if t in sl.key.lower():
            return True
    return False


# --- image sources ------------------------------------------------------------
def pick_channel(
    names: list[str], wanted: str | None = None, prefer=DEFAULT_NUCLEAR
) -> int:
    """Index of the nuclear channel: `wanted` (name or index) or the first of
    `prefer` that appears in the channel names, case-insensitive."""
    low = [n.upper() for n in names]
    if wanted is not None:
        if wanted.isdigit():
            return int(wanted)
        if wanted.upper() in low:
            return low.index(wanted.upper())
        hits = [i for i, n in enumerate(low) if wanted.upper() in n]
        if len(hits) == 1:
            return hits[0]
        raise SystemExit(f"channel {wanted!r} not found in {names}")
    for p in prefer:
        for i, n in enumerate(low):
            if p in n:
                return i
    raise SystemExit(f"no nuclear channel among {names}; pass --channel")


class TiffSource:
    """TIFF / OME-TIFF (pyramidal or not); crops decode only the tiles they touch."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise SystemExit(
                f"{self.path}: not found (checkpoint names a file that is gone)"
            )
        self.tf = tifffile.TiffFile(str(self.path))
        s = self.tf.series[0]
        self.series = s
        self.axes = s.axes
        self.levels = list(s.levels) if s.is_pyramidal else [s]
        self.nchannels = s.shape[self.axes.index("C")] if "C" in self.axes else 1
        self.shape = self._yx(s.shape, self.axes)
        self.names = self._channel_names()
        self.px = self._pixel_size()

    @staticmethod
    def _yx(shape, axes):
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])

    def _channel_names(self) -> list[str]:
        names: list[str] = []
        if self.tf.ome_metadata:
            names = re.findall(
                r'<Channel\b[^>]*?\bName="([^"]*)"', self.tf.ome_metadata
            )
        ij = self.tf.imagej_metadata or {}
        if not names and ij.get("Labels"):
            names = list(ij["Labels"])
        names = names[: self.nchannels]
        names += [f"C{i}" for i in range(len(names), self.nchannels)]
        return names

    def _pixel_size(self) -> float | None:
        ome = self.tf.ome_metadata
        if ome:
            m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
            if m:
                u = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome)
                unit = u.group(1) if u else "µm"
                return float(m.group(1)) * {
                    "nm": 1e-3,
                    "µm": 1.0,
                    "um": 1.0,
                    "mm": 1e3,
                }.get(unit, 1.0)
        page = self.tf.pages[0]
        xres, unit = page.tags.get("XResolution"), page.tags.get("ResolutionUnit")
        if xres and xres.value[0]:
            num, den = xres.value
            ppu = num / den
            uval = int(unit.value) if unit else 1
            if uval == 3:
                return 1e4 / ppu
            if uval == 2:
                return 25400.0 / ppu
            ij = self.tf.imagej_metadata or {}
            if ij.get("unit") in ("um", "µm", "micron"):
                return 1.0 / ppu
        return None

    def _page_index(self, ci: int) -> int:
        """Flat page index of the plane with channel `ci` (all other non-YX axes at 0)."""
        dims = [
            (ax, n) for ax, n in zip(self.axes, self.series.shape) if ax not in "YX"
        ]
        if not dims:
            return 0
        idx = [ci if ax == "C" else 0 for ax, _ in dims]
        return int(np.ravel_multi_index(idx, [n for _, n in dims]))

    def _read_region(
        self, level: int, ci: int, y0: int, y1: int, x0: int, x1: int
    ) -> np.ndarray:
        """Decode only the tiles/strips overlapping [y0:y1, x0:x1] of one plane."""
        pg = self.levels[level].pages[self._page_index(ci)]
        if pg is None:
            raise SystemExit(
                f"{self.path.name}: missing page for channel {ci} at level {level}"
            )
        kf = pg.keyframe
        fh = pg.parent.filehandle
        Hl, Wl = kf.imagelength, kf.imagewidth
        out = np.zeros((y1 - y0, x1 - x0), kf.dtype)
        if kf.is_tiled:
            th, tw = kf.tilelength, kf.tilewidth
            ntx = -(-Wl // tw)
            segs = [
                ty * ntx + tx
                for ty in range(y0 // th, (y1 - 1) // th + 1)
                for tx in range(x0 // tw, (x1 - 1) // tw + 1)
            ]
        else:
            th, tw = (kf.rowsperstrip or Hl), Wl
            segs = list(range(y0 // th, (y1 - 1) // th + 1))
        offsets, counts = pg.dataoffsets, pg.databytecounts
        for i in segs:
            if i >= len(offsets) or counts[i] == 0:
                continue  # sparse / empty segment -> zeros
            fh.seek(offsets[i])
            arr, pos, _ = kf.decode(fh.read(counts[i]), i)
            if arr is None:
                continue
            sy, sx = int(pos[2]), int(pos[3])
            seg = (
                arr[0, :, :, 0]
                if arr.ndim == 4
                else np.asarray(arr).reshape(arr.shape[-2:])
            )
            ay0, ax0 = max(sy, y0), max(sx, x0)
            ay1, ax1 = min(sy + seg.shape[0], Hl, y1), min(sx + seg.shape[1], Wl, x1)
            if ay1 > ay0 and ax1 > ax0:
                out[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = seg[
                    ay0 - sy : ay1 - sy, ax0 - sx : ax1 - sx
                ]
        return out

    def level_shape(self, level: int) -> tuple[int, int]:
        return self._yx(self.levels[level].shape, self.axes)

    def _read(self, ci, ys, xs, level=0):
        Hl, Wl = self.level_shape(level)
        y0, y1, sy = ys.indices(Hl)
        x0, x1, sx = xs.indices(Wl)
        return self._read_region(level, ci, y0, y1, x0, x1)[::sy, ::sx]

    def read_patch(self, ci: int, y: int, x: int, h: int, w: int) -> np.ndarray:
        """Crop with mirage's pad-or-crop rule: zero outside the image, never resampled."""
        H, W = self.shape
        y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + h, H), min(x + w, W)
        if y1 <= y0 or x1 <= x0:
            return np.zeros((h, w), np.uint16)
        sub = self._read(ci, slice(y0, y1), slice(x0, x1))
        if (y0, x0, y1, x1) == (y, x, y + h, x + w):
            return sub
        out = np.zeros((h, w), sub.dtype)
        out[y0 - y : y1 - y, x0 - x : x1 - x] = sub
        return out

    def read_lowres(self, ci, factor):
        """Whole plane downsampled ~`factor` (from the closest pyramid level); (plane, actual factor)."""
        H = self.shape[0]
        best = 0
        for i in range(len(self.levels)):
            if H / self.level_shape(i)[0] <= factor + 1e-6:
                best = i
        f = H / self.level_shape(best)[0]
        step = max(1, int(round(factor / f)))
        return self._read(
            ci, slice(None, None, step), slice(None, None, step), level=best
        ), f * step

    def close(self):
        self.tf.close()


# --- columns: arms, and the Before column -------------------------------------
class Column:
    """One column of the mosaic: a directory carrying a mirage checkpoint whose rows
    name, per slide, the image to draw from -- ``registered_image`` for an arm
    (the reference row names the preprocessed reference, i.e. the frame) or
    ``preprocessed_image`` for the Before column."""

    checkpoint = REGISTERED_CSV
    image_col = "registered_image"

    def __init__(
        self, root: Path, label: str | None = None, channel: str | None = None
    ):
        self.root = Path(root)
        self.name = label or self.root.name
        self.channel = channel
        self.per = read_checkpoint(self.root / self.checkpoint, self.image_col)
        self._src: dict[str, TiffSource] = {}
        self._ci: dict[str, int] = {}
        self.ref: Slide | None = None
        self.moving: dict[str, Slide] = {}

    def patients(self) -> list[str]:
        return list(self.per)

    def open(self, patient: str) -> None:
        rows = self.per.get(patient)
        if not rows:
            raise SystemExit(
                f"[{self.name}] patient {patient!r} not in {self.root / self.checkpoint}"
            )
        refs = [r for r in rows if r.is_reference]
        if len(refs) != 1:
            raise SystemExit(
                f"[{self.name}] {patient}: expected one reference row, found {len(refs)}"
            )
        self.ref = refs[0]
        self.moving = {r.key: r for r in rows if not r.is_reference}
        self._src, self._ci = {}, {}

    def source(self, sl: Slide) -> tuple[TiffSource, int]:
        if sl.key not in self._src:
            src = TiffSource(sl.image)
            names = sl.channels if len(sl.channels) == src.nchannels else src.names
            ci = pick_channel(names, self.channel)
            if ci >= src.nchannels:
                raise SystemExit(
                    f"[{self.name}] {sl.image.name}: nuclear channel index {ci} but the file has "
                    f"{src.nchannels} channel(s); channels column says {sl.channels}"
                )
            self._src[sl.key], self._ci[sl.key] = src, ci
            log.info(
                "[%s] %s: %dx%d px, %s µm/px, nuclear channel %s",
                self.name,
                src.path.name,
                *src.shape,
                src.px,
                names[ci],
            )
        return self._src[sl.key], self._ci[sl.key]

    @property
    def frame(self) -> TiffSource:
        assert self.ref is not None
        return self.source(self.ref)[0]

    @property
    def px(self) -> float | None:
        assert self.ref is not None
        return self.ref.pixel_size or self.frame.px

    def slide(self, key: str) -> Slide:
        if key not in self.moving:
            raise SystemExit(
                f"[{self.name}] no moving slide with channel set {key!r}; this arm has {sorted(self.moving)}"
            )
        return self.moving[key]

    def crops(self, key: str, y, x, h, w) -> tuple[np.ndarray, np.ndarray]:
        """(reference crop, moving crop) at the same reference-frame coordinates."""
        rs, rc = self.source(self.ref)
        ms, mc = self.source(self.slide(key))
        return rs.read_patch(rc, y, x, h, w), ms.read_patch(mc, y, x, h, w)

    def lowres(self, key: str, factor) -> tuple[np.ndarray, np.ndarray]:
        rs, rc = self.source(self.ref)
        ms, mc = self.source(self.slide(key))
        return rs.read_lowres(rc, factor)[0], ms.read_lowres(mc, factor)[0]

    def lowres_ref(self, factor):
        rs, rc = self.source(self.ref)
        return rs.read_lowres(rc, factor)

    def files(self, keys) -> dict[str, str]:
        assert self.ref is not None
        out = {"__ref__": str(self.ref.image)}
        out.update({k: str(self.moving[k].image) for k in keys if k in self.moving})
        return out

    def close(self):
        for s in self._src.values():
            s.close()
        self._src, self._ci = {}, {}


class Before(Column):
    """The shared preprocessing run: every slide as it entered registration."""

    checkpoint = PREPROCESSED_CSV
    image_col = "preprocessed_image"

    def __init__(self, root: Path, channel: str | None = None):
        super().__init__(root, BEFORE_LABEL, channel)


def find_before(arm_dirs: list[Path], explicit: Path | None) -> Path:
    """The preprocessing run the arms resumed from: --before, else <root>/preprocess_shared."""
    if explicit is not None:
        return explicit
    roots = {d.resolve().parent for d in arm_dirs}
    for root in sorted(roots):
        cand = root / BEFORE_ARM
        if (cand / PREPROCESSED_CSV).is_file():
            return cand
    raise SystemExit(
        f"no {BEFORE_ARM}/{PREPROCESSED_CSV} beside the arm directories ({sorted(str(r) for r in roots)}); "
        "pass --before <the --outdir of the preprocessing run the arms resumed from>"
    )


# --- rows -----------------------------------------------------------------------
def plan_rows(round_keys: list[str], n_rois: int, n_rows: int) -> list[tuple[str, int]]:
    """Exactly ``n_rows`` (round, roi) pairs, ROI-major.

    ROI-major so a truncated plan still shows every round at the first ROI
    before it shows any round at a second one: ``--rows 4`` over three rounds
    gives (ROI 1 x all three rounds) + (ROI 2 x the first round), never three ROIs
    of one round and none of the others. ``n_rois`` should be
    ``ceil(n_rows / len(round_keys))`` when the ROIs are auto-selected.
    """
    if n_rows < 1:
        raise ValueError(f"--rows must be >= 1, got {n_rows}")
    if not round_keys:
        raise ValueError("no moving rounds to plan rows over")
    plan = [(k, i) for i in range(n_rois) for k in round_keys]
    if len(plan) < n_rows:
        raise ValueError(
            f"{len(round_keys)} round(s) x {n_rois} ROI(s) = {len(plan)} rows, fewer than --rows {n_rows}"
        )
    return plan[:n_rows]


# --- ROI selection ------------------------------------------------------------
def _otsu_mask(u8: np.ndarray, sigma: float) -> np.ndarray:
    blurred = ndimage.gaussian_filter(u8.astype(np.float32), sigma)
    if blurred.max() == blurred.min():
        return np.zeros(u8.shape, bool)
    return blurred > threshold_otsu(blurred)


def _box(img: np.ndarray, win: int) -> np.ndarray:
    return ndimage.uniform_filter(np.asarray(img, np.float32), size=win, mode="reflect")


def select_rois(
    low: np.ndarray,
    factor: float,
    patch_px: int,
    n: int,
    min_sep: float,
    spread: float,
    full_shape,
) -> list[tuple[int, int]]:
    """Pick `n` patch positions on a low-res nuclear image.

    Score = tissue coverage (window mostly inside tissue, Otsu on the log image)
          x local texture (std of log intensity within the tissue pixels: glands,
            vessels, density changes -- not a uniform nuclear sheet or empty glass)
          x penalty for saturated pixels (debris, folds).
    Greedy: best window first; later picks are pushed away from earlier ones
    (`spread`) and never closer than `min_sep` x image diagonal.
    """
    H, W = full_shape
    win = max(3, int(round(patch_px / factor)))
    img = np.asarray(low, np.float32)
    f = np.log1p(np.maximum(img - float(np.percentile(img, 1.0)), 0.0))
    rng = float(f.max() - f.min()) or 1.0
    u8 = ((f - f.min()) * (255.0 / rng)).astype(np.uint8)
    fg = _otsu_mask(u8, 1.5).astype(np.float32)
    fg_frac = _box(fg, win)
    # texture = std of log intensity over the TISSUE pixels of the window only, so the
    # glass/tissue edge does not dominate and a uniform nuclear sheet scores low
    den = np.maximum(fg_frac, 1e-3)
    m1 = _box(f * fg, win) / den
    m2 = _box(f * f * fg, win) / den
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    sat = _box((img >= np.percentile(img, 99.98)).astype(np.float32), win)
    cover = np.clip((fg_frac - 0.5) / 0.4, 0.0, 1.0)  # 0 at <=50 % tissue, 1 at >=90 %
    norm = float(np.percentile(std[fg > 0], 95)) if fg.any() else float(std.max())
    texture = np.clip(std / (norm + 1e-6), 0.0, 1.0)
    score = cover * texture * np.clip(1.0 - 25.0 * sat, 0.0, 1.0)
    half = win // 2
    ok = np.zeros(score.shape, bool)
    ok[half : score.shape[0] - half, half : score.shape[1] - half] = True
    score[~ok] = 0.0
    yy, xx = np.indices(score.shape)
    diag = float(np.hypot(*score.shape))
    sep = max(win, int(round(min_sep * diag)))
    chosen: list[tuple[int, int]] = []
    for _ in range(n):
        cur = score
        if chosen and spread > 0:
            dmin = (
                np.min([np.hypot(yy - py, xx - px) for py, px in chosen], axis=0) / diag
            )
            cur = score * (1.0 + spread * dmin)
        py, px = np.unravel_index(int(np.argmax(cur)), cur.shape)
        if cur[py, px] <= 0:
            log.warning(
                "only %d ROI(s) with tissue found; lower --min-sep or --rows",
                len(chosen),
            )
            break
        chosen.append((int(py), int(px)))
        score[max(py - sep, 0) : py + sep + 1, max(px - sep, 0) : px + sep + 1] = 0.0
    rois = []
    for py, px in chosen:
        y = int(round(py * factor)) - patch_px // 2
        x = int(round(px * factor)) - patch_px // 2
        rois.append(
            (min(max(y, 0), max(H - patch_px, 0)), min(max(x, 0), max(W - patch_px, 0)))
        )
    return rois


# --- rendering ----------------------------------------------------------------
def percentile_limits(img, pmin, pmax):
    lo, hi = np.percentile(np.asarray(img, np.float32), (pmin, pmax))
    return float(lo), float(hi)


def stretch(img, limits, gamma=1.0) -> np.ndarray:
    lo, hi = limits
    if hi <= lo:
        return np.zeros(np.shape(img), np.float32)
    out = np.clip((np.asarray(img, np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return out**gamma if gamma != 1.0 else out


def overlay(mov01, ref01, palette) -> np.ndarray:
    cm, cr = (np.asarray(c, np.float32) for c in PALETTES[palette])
    return np.clip(mov01[..., None] * cm + ref01[..., None] * cr, 0.0, 1.0)


def checkerboard(mov01, ref01, tiles) -> np.ndarray:
    h, w = ref01.shape
    ty, tx = max(1, -(-h // tiles)), max(1, -(-w // tiles))
    yy, xx = np.indices((h, w))
    g = np.where(((yy // ty + xx // tx) % 2) == 0, ref01, mov01)
    return np.repeat(g[..., None], 3, axis=-1)


def nuclear_mask(img01: np.ndarray) -> np.ndarray:
    """Otsu on the (stretched) nuclear channel after a light blur."""
    u8 = np.round(np.clip(img01, 0, 1) * 255).astype(np.uint8)
    if u8.max() == 0:
        return np.zeros(u8.shape, bool)
    return _otsu_mask(u8, 1.0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


def residual_shift(ref, mov):
    """(dx, dy, response): translation of `mov` relative to `ref` by phase correlation
    with a Hann window and parabolic sub-pixel refinement. `response` is the peak of the
    normalised cross-power spectrum (1 = identical, ~0.01 = unrelated content)."""
    a = np.asarray(ref, np.float64)
    b = np.asarray(mov, np.float64)
    if a.max() == a.min() or b.max() == b.min():
        return float("nan"), float("nan"), 0.0
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fft2((a - a.mean()) * win)
    G = np.fft.fft2((b - b.mean()) * win)
    R = F * np.conj(G)
    R /= np.abs(R) + 1e-12
    r = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    resp = float(r[py, px])

    def refine(vm, v0, vp):
        d = vm - 2.0 * v0 + vp
        return 0.0 if d == 0 else 0.5 * (vm - vp) / d

    dy = py + refine(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    dx = px + refine(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    dy, dx = (dy - h if dy > h / 2 else dy), (dx - w if dx > w / 2 else dx)
    return float(-dx), float(-dy), resp


def ncc(ref, mov) -> float:
    a = np.asarray(ref, np.float64).ravel()
    b = np.asarray(mov, np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics(ref_raw, mov_raw, ref01, mov01, px, which) -> tuple[str, dict]:
    """Text for the corner of a cell + numbers for the manifest."""
    vals: dict = {}
    parts = []
    if "dice" in which:
        d = dice(nuclear_mask(ref01), nuclear_mask(mov01))
        vals["dice"] = d
        parts.append("Dice n/a" if np.isnan(d) else f"Dice {d:.2f}")
    if "shift" in which:
        dx, dy, resp = residual_shift(ref_raw, mov_raw)
        vals.update(shift_px=[dx, dy], pc_response=resp)
        if np.isnan(dx) or resp < MIN_PC_RESPONSE:
            parts.append("Δ n/a")
        else:
            d = float(np.hypot(dx, dy))
            vals["shift_um"] = d * px if px else None
            parts.append(f"Δ {d * px:.1f} µm" if px else f"Δ {d:.1f} px")
    if "ncc" in which:
        r = ncc(ref_raw, mov_raw)
        vals["ncc"] = r
        parts.append("r n/a" if np.isnan(r) else f"r {r:.2f}")
    return "  ".join(parts), vals


def auto_scalebar_um(patch_um: float) -> float:
    target = patch_um / 4.0 * 1.05
    return float(max([b for b in NICE_BARS_UM if b <= target] or [NICE_BARS_UM[0]]))


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _outline(t):
    from matplotlib import patheffects

    t.set_path_effects([patheffects.withStroke(linewidth=1.6, foreground="black")])


def draw_scalebar(ax, h, w, bar_px, label, font, thick=0.014):
    from matplotlib.patches import Rectangle

    x0, y0 = 0.05 * w, 0.92 * h
    ax.add_patch(Rectangle((x0, y0), bar_px, max(1.0, thick * h), color="white", lw=0))
    _outline(
        ax.text(
            x0 + bar_px / 2,
            y0 - 0.012 * h,
            label,
            color="white",
            ha="center",
            va="bottom",
            fontsize=font,
        )
    )


def assemble_figure(
    grid,
    notes,
    row_labels,
    col_labels,
    out_stem: Path,
    formats,
    cell_in,
    dpi,
    scalebar,
    scalebar_where,
    font=8.0,
):
    plt = _mpl()
    nr, nc = len(grid), len(grid[0])
    fig, axs = plt.subplots(nr, nc, figsize=(nc * cell_in, nr * cell_in), squeeze=False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03, hspace=0.03)
    for r in range(nr):
        for c in range(nc):
            ax = axs[r][c]
            ax.imshow(grid[r][c], interpolation="none")
            ax.set_axis_off()
            if r == 0:
                ax.set_title(col_labels[c], fontsize=font, pad=3)
            if c == 0:
                ax.text(
                    -0.04,
                    0.5,
                    row_labels[r],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=font,
                )
            if notes[r][c]:
                _outline(
                    ax.text(
                        0.97,
                        0.04,
                        notes[r][c],
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=font * 0.8,
                        color="white",
                    )
                )
            if scalebar and c == 0 and (scalebar_where == "all" or r == nr - 1):
                h, w = grid[r][c].shape[:2]
                draw_scalebar(ax, h, w, scalebar[0], scalebar[1], font * 0.8)
    for fmt in formats:
        fig.savefig(f"{out_stem}.{fmt}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_locator(low, factor, rois, patch_px, out_stem: Path, formats, px, title, dpi):
    plt = _mpl()
    from matplotlib.patches import Rectangle

    disp = stretch(low, percentile_limits(low, 1.0, 99.8), 0.7)
    h, w = disp.shape
    fig, ax = plt.subplots(figsize=(5.0, 5.0 * h / w))
    ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="none")
    for i, (y, x) in enumerate(rois, 1):
        s = patch_px / factor
        ax.add_patch(
            Rectangle((x / factor, y / factor), s, s, fill=False, ec="#ffd400", lw=1.2)
        )
        _outline(
            ax.text(
                x / factor + s / 2,
                y / factor - 0.01 * h,
                str(i),
                color="#ffd400",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        )
    if px:
        bar_um = auto_scalebar_um(w * factor * px)
        draw_scalebar(
            ax, h, w, bar_um / (px * factor), f"{bar_um:g} µm", 7, thick=0.006
        )
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    for fmt in formats:
        fig.savefig(f"{out_stem}.{fmt}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_png(path: Path, rgb01):
    import matplotlib.image

    matplotlib.image.imsave(str(path), np.clip(np.asarray(rgb01, np.float32), 0.0, 1.0))


# --- per patient --------------------------------------------------------------
@dataclass
class Options:
    """Everything `process_patient` needs, decoupled from argparse for the tests."""

    outdir: Path
    rows: int
    rounds: list[str] | None = None
    patch_um: float = 200.0
    patch_px: int | None = None
    roi: list[str] = field(default_factory=list)
    rois_json: Path | None = None
    min_sep: float = 0.15
    spread: float = 1.0
    kinds: str = "overlay"
    palette: str = "magenta-green"
    stretch: str = "patch"
    pmin: float = 0.5
    pmax: float = 99.7
    gamma: float = 0.8
    checker_tiles: int = 6
    annotate: str = "dice"
    annotate_where: str = "after"
    scalebar_um: float | None = None
    scalebar_where: str = "last"
    cell_in: float = 1.4
    dpi: int = 300
    formats: str = "png,pdf"
    pixel_size_um: float | None = None
    lowres_um: float = 5.0
    save_raw: bool = False


def process_patient(pid: str, before: Before, arms: list[Column], opt: Options) -> dict:
    columns: list[Column] = [before, *arms]
    for col in columns:
        col.open(pid)
    ref = before.ref
    assert ref is not None
    moving = [before.moving[k] for k in before.moving]
    if opt.rounds:
        moving = [sl for sl in moving if round_matches(sl, opt.rounds)]
    if not moving:
        raise SystemExit(f"{pid}: no moving round matches {opt.rounds}")
    for arm in arms:
        absent = [sl.key for sl in moving if sl.key not in arm.moving]
        if absent:
            raise SystemExit(
                f"[{arm.name}] {pid}: no registered slide for round(s) {absent}; "
                f"the arm has {sorted(arm.moving)} -- did that arm finish for this patient?"
            )
    log.info(
        "== %s: reference %s; %d moving round(s): %s",
        pid,
        ref.image.name,
        len(moving),
        ", ".join(sl.key for sl in moving),
    )

    frame = before.frame
    px = opt.pixel_size_um or before.px
    if opt.patch_px:
        patch_px = opt.patch_px
    elif px:
        patch_px = int(round(opt.patch_um / px))
    else:
        raise SystemExit("pixel size unknown; pass --pixel-size-um or --patch-px")
    H, W = frame.shape
    patch_um = patch_px * px if px else None
    log.info(
        "reference frame %dx%d px, %s µm/px; patch %d px%s",
        H,
        W,
        px,
        patch_px,
        f" = {patch_um:.0f} µm" if patch_um else "",
    )

    for arm in arms:
        for sl in moving:
            s, _ = arm.source(arm.slide(sl.key))
            if s.shape != frame.shape:
                log.warning(
                    "[%s] %s is %s but the reference frame is %s; ROI coordinates assume the registered "
                    "output is on the reference canvas",
                    arm.name,
                    s.path.name,
                    s.shape,
                    frame.shape,
                )

    # low-res reference for ROI selection and the locator
    factor = max(1, int(round(opt.lowres_um / px))) if px else 16
    low, f = before.lowres_ref(factor)

    # ROIs: reuse, manual, or auto -- enough of them for --rows over the rounds
    n_rois = max(1, math.ceil(opt.rows / len(moving)))
    if opt.rois_json:
        prev = json.loads(Path(opt.rois_json).read_text())
        rois = [(int(r["y"]), int(r["x"])) for r in prev["rois"]]
        if prev.get("patch_px") and prev["patch_px"] != patch_px:
            log.warning(
                "patch size taken from %s: %d px", opt.rois_json, prev["patch_px"]
            )
            patch_px = int(prev["patch_px"])
    elif opt.roi:
        rois = [tuple(int(v) for v in r.split(","))[:2] for r in opt.roi]
    else:
        rois = select_rois(low, f, patch_px, n_rois, opt.min_sep, opt.spread, (H, W))
    if not rois:
        raise SystemExit(f"{pid}: no ROI could be selected")
    log.info("ROIs (y, x, %d px): %s", patch_px, rois)
    plan = plan_rows([sl.key for sl in moving], len(rois), opt.rows)
    by_key = {sl.key: sl for sl in moving}

    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip() for x in opt.formats.split(",") if x.strip()]
    save_locator(
        low,
        f,
        rois,
        patch_px,
        outdir / f"{pid}_locator",
        formats,
        px,
        f"{pid} — {ref.label} (reference)",
        opt.dpi,
    )

    glob_cache: dict = {}

    def limits(col: Column, chan: str, key: str, crop) -> tuple[float, float]:
        if opt.stretch == "patch":
            return percentile_limits(crop, opt.pmin, opt.pmax)
        ck = (col.name, chan, key if chan == "mov" else "")
        if ck not in glob_cache:
            plane = col.lowres(key, factor)[chan == "mov"]
            glob_cache[ck] = percentile_limits(plane, opt.pmin, opt.pmax)
        return glob_cache[ck]

    kinds = [k.strip() for k in opt.kinds.split(",") if k.strip()]
    which = (
        set()
        if opt.annotate == "none"
        else {a.strip() for a in opt.annotate.split(",")}
    )
    col_names = [c.name for c in columns]
    col_labels = [
        c if k == "overlay" else f"{c} (checker)" for k in kinds for c in col_names
    ]
    pdir = outdir / f"{pid}_patches"
    pdir.mkdir(exist_ok=True)

    grid, notes, row_labels, rows_meta = [], [], [], []
    for key, ri in plan:
        sl = by_key[key]
        y, x = rois[ri]
        crops = {col.name: col.crops(key, y, x, patch_px, patch_px) for col in columns}
        lim = {
            c.name: (
                limits(c, "ref", key, crops[c.name][0]),
                limits(c, "mov", key, crops[c.name][1]),
            )
            for c in columns
        }
        st = {
            n: (stretch(rc, lim[n][0], opt.gamma), stretch(mc, lim[n][1], opt.gamma))
            for n, (rc, mc) in crops.items()
        }
        cells_meta = {}
        row_grid, row_notes = [], []
        for k in kinds:
            for n in col_names:
                r01, m01 = st[n]
                img = (
                    overlay(m01, r01, opt.palette)
                    if k == "overlay"
                    else checkerboard(m01, r01, opt.checker_tiles)
                )
                note = ""
                if k == "overlay":
                    text, vals = metrics(crops[n][0], crops[n][1], r01, m01, px, which)
                    cells_meta[n] = {
                        "ref_limits": lim[n][0],
                        "mov_limits": lim[n][1],
                        **vals,
                    }
                    if opt.annotate_where == "all" or n != BEFORE_LABEL:
                        note = text
                row_grid.append(img)
                row_notes.append(note)
                stem = f"r{sl.index:02d}_{key}_roi{ri + 1}_{n}_{k}"
                write_png(pdir / f"{stem}.png", img)
                if opt.save_raw and k == "overlay":
                    raw = np.stack([np.asarray(a) for a in crops[n]])
                    meta = {"axes": "CYX", "Labels": ["reference", "moving"]}
                    kw = {}
                    if px:
                        meta["unit"] = "um"
                        kw = {
                            "resolution": (1.0 / px, 1.0 / px),
                            "resolutionunit": "MICROMETER",
                        }
                    tifffile.imwrite(
                        str(pdir / f"{stem}_raw.tif"),
                        raw,
                        imagej=True,
                        metadata=meta,
                        compression="zlib",
                        **kw,
                    )
        grid.append(row_grid)
        notes.append(row_notes)
        parts = ([sl.label] if len(moving) > 1 or len(rois) == 1 else []) + (
            [f"ROI {ri + 1}"] if len(rois) > 1 else []
        )
        row_labels.append("\n".join(parts))
        rows_meta.append(
            {"round": key, "roi": ri + 1, "y": y, "x": x, "cells": cells_meta}
        )

    scalebar = None
    if px and opt.scalebar_um != 0:
        bar_um = opt.scalebar_um or auto_scalebar_um(patch_um)
        scalebar = (bar_um / px, f"{bar_um:g} µm")
    assemble_figure(
        grid,
        notes,
        row_labels,
        col_labels,
        outdir / f"{pid}_mosaic",
        formats,
        opt.cell_in,
        opt.dpi,
        scalebar,
        opt.scalebar_where,
    )

    manifest = {
        "patient": pid,
        "reference": str(ref.image),
        "pixel_size_um": px,
        "patch_px": patch_px,
        "patch_um": patch_um,
        "rows": opt.rows,
        "palette": opt.palette,
        "kinds": kinds,
        "stretch": opt.stretch,
        "pmin": opt.pmin,
        "pmax": opt.pmax,
        "gamma": opt.gamma,
        "annotate": sorted(which),
        "scalebar_um": scalebar and float(scalebar[1].split()[0]),
        "rois": [{"id": i, "y": y, "x": x} for i, (y, x) in enumerate(rois, 1)],
        "columns": {
            c.name: {"dir": str(c.root), "files": c.files([sl.key for sl in moving])}
            for c in columns
        },
        "row_plan": rows_meta,
    }
    (outdir / f"{pid}_rois.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "wrote %s_mosaic.{%s}, %s_locator.*, %s_rois.json and %d cell PNGs in %s",
        pid,
        ",".join(formats),
        pid,
        pid,
        sum(len(r) for r in grid),
        pdir,
    )
    for col in columns:
        col.close()
    return manifest


# --- main ---------------------------------------------------------------------
def parse_labels(specs) -> dict[str, str]:
    out = {}
    for spec in specs or []:
        name, sep, title = spec.partition("=")
        if not sep:
            raise SystemExit(f"--label expects ARM=Title, got {spec!r}")
        out[name] = title
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "arm",
        nargs="+",
        type=Path,
        metavar="ARM_DIR",
        help="one or more <arm_results>/<arm> directories (one column each)",
    )
    ap.add_argument(
        "--rows",
        type=int,
        required=True,
        help="number of rows: (moving round, ROI) pairs, ROI-major",
    )
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument(
        "--patient",
        action="append",
        default=[],
        help="patient_id(s) to process (default: every patient of the first arm)",
    )
    ap.add_argument(
        "--before",
        type=Path,
        default=None,
        help=f"the preprocessing run's --outdir (default: <root>/{BEFORE_ARM} beside the arms)",
    )
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="ARM=Title",
        help="column title for an arm directory name (default: the directory name)",
    )
    g = ap.add_argument_group("rows")
    g.add_argument(
        "--rounds",
        nargs="*",
        default=None,
        help="moving rounds to include: channel-set key (CD8_CD4), a marker (CD8) or the slide id",
    )
    g.add_argument("--patch-um", type=float, default=200.0, help="patch side in µm")
    g.add_argument(
        "--patch-px",
        type=int,
        default=None,
        help="patch side in px (overrides --patch-um)",
    )
    g.add_argument(
        "--roi",
        action="append",
        default=[],
        metavar="Y,X",
        help="manual ROI top-left in the reference frame (full-res px); repeatable",
    )
    g.add_argument(
        "--rois-json",
        type=Path,
        default=None,
        help="reuse the ROIs of a previous run's *_rois.json",
    )
    g.add_argument(
        "--min-sep",
        type=float,
        default=0.15,
        help="min ROI separation, fraction of the image diagonal",
    )
    g.add_argument(
        "--spread",
        type=float,
        default=1.0,
        help="weight pushing later ROIs away from earlier ones (0 = pure quality score)",
    )
    g = ap.add_argument_group("look")
    g.add_argument("--kinds", default="overlay", help="comma list of: overlay, checker")
    g.add_argument(
        "--palette",
        choices=list(PALETTES),
        default="magenta-green",
        help="moving/reference colours (red-green = mirage QC)",
    )
    g.add_argument(
        "--stretch",
        choices=("patch", "global"),
        default="patch",
        help="percentile limits per crop and source, or per whole image and source",
    )
    g.add_argument("--pmin", type=float, default=0.5)
    g.add_argument("--pmax", type=float, default=99.7)
    g.add_argument("--gamma", type=float, default=0.8)
    g.add_argument("--checker-tiles", type=int, default=6)
    g.add_argument(
        "--annotate",
        default="dice",
        help="comma list of: dice (Otsu nuclear masks), shift (phase correlation, µm), ncc; or none",
    )
    g.add_argument(
        "--annotate-where",
        choices=("after", "all"),
        default="after",
        help="print the metric only in the arm columns or in Before too",
    )
    g.add_argument(
        "--scalebar-um",
        type=float,
        default=None,
        help="scale bar length (default auto ≈ patch/4; 0 = none)",
    )
    g.add_argument(
        "--scalebar-where",
        choices=("last", "all"),
        default="last",
        help="one bar in the bottom-left cell, or one per row",
    )
    g.add_argument(
        "--cell-in", type=float, default=1.4, help="cell size in inches in the figure"
    )
    g.add_argument("--dpi", type=int, default=300)
    g.add_argument("--formats", default="png,pdf")
    g = ap.add_argument_group("inputs")
    g.add_argument(
        "--channel",
        default=None,
        help="nuclear channel (name or index; default first of DAPI/HOECHST/CELLTOX in the checkpoint's channels)",
    )
    g.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="override the pixel size of the reference",
    )
    g.add_argument(
        "--lowres-um",
        type=float,
        default=5.0,
        help="pixel size of the low-res image used for ROI selection, the locator and --stretch global",
    )
    g.add_argument(
        "--save-raw",
        action="store_true",
        help="also write the uint16 ref/mov crops as CYX TIFFs",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for k in [x.strip() for x in args.kinds.split(",")]:
        if k not in ("overlay", "checker"):
            raise SystemExit(f"unknown kind {k!r}")
    if args.rows < 1:
        raise SystemExit("--rows must be >= 1")

    labels = parse_labels(args.label)
    arms = [Column(d, labels.get(d.name), args.channel) for d in args.arm]
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit(
            f"two arm columns would share a title {names}; pass --label ARM=Title"
        )
    before = Before(find_before(args.arm, args.before), args.channel)

    opt = Options(
        outdir=args.outdir,
        rows=args.rows,
        rounds=args.rounds,
        patch_um=args.patch_um,
        patch_px=args.patch_px,
        roi=args.roi,
        rois_json=args.rois_json,
        min_sep=args.min_sep,
        spread=args.spread,
        kinds=args.kinds,
        palette=args.palette,
        stretch=args.stretch,
        pmin=args.pmin,
        pmax=args.pmax,
        gamma=args.gamma,
        checker_tiles=args.checker_tiles,
        annotate=args.annotate,
        annotate_where=args.annotate_where,
        scalebar_um=args.scalebar_um,
        scalebar_where=args.scalebar_where,
        cell_in=args.cell_in,
        dpi=args.dpi,
        formats=args.formats,
        pixel_size_um=args.pixel_size_um,
        lowres_um=args.lowres_um,
        save_raw=args.save_raw,
    )
    patients = args.patient or arms[0].patients()
    for pid in patients:
        process_patient(pid, before, arms, opt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

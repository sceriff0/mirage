#!/usr/bin/env python3
"""reg_mosaic.py -- before/after registration patch mosaic across benchmark arms.

Takes one or more ARM DIRECTORIES of the real-sample arm benchmark
(``<arm_results>/<arm>``, each a mirage ``--outdir`` that ran registration with
its QC) and writes, per patient, a mosaic of nuclear-channel overlays:

    columns = (moving round, ROI) pairs -- exactly ``--rows`` of them
    rows    = Before | <arm 1> | <arm 2> ...   (one row per arm directory)
    (--orient rounds-as-rows transposes it; "--rows" counts the (round, ROI) pairs either way)

Nothing is re-registered, re-warped or re-scored. The PIXELS come from the arm's
own slides, at full 16-bit depth (``--source auto``, the default):

    <arm>/csv/registered.csv                         the registered slides + reference
    <arm>/csv/preprocessed.csv  or  <arm>/../preprocess_shared/csv/preprocessed.csv
                                                     the native slides as they entered registration

each nuclear channel found by its OME channel name, the native slide pad-or-cropped
onto the reference canvas at the origin exactly as the QC composite does. The 8-bit
QC composite is the per-round FALLBACK when a slide is missing: it is min-max scaled
over the whole slide, so a few extreme pixels squash all real tissue into 2-3 grey
levels, and a crop of that re-stretched is a flat speckled field with a contour edge
(5 of 9 rounds of a real mosaic, 2026-09-17). Each cell is drawn one image pixel per
output pixel (--cell-in defaults to patch px / dpi). What else the QC wrote:

    <arm>/<patient>/qc/registration[/qc]/<slide>_QC_RGB_fullres.tif
        the pipeline's two-panel composite (bin/utils/qc.py render_before_after):
        left = Before (red: the moving slide as it entered registration, green:
        the reference), a blue separator, right = After (red: the registered
        moving slide, green: the reference). Both panels sit on the reference
        canvas, so one (y, x) is the same tissue in every panel of every arm.
        Its full-res plane is the fallback pixel source; its reference plane
        picks the ROIs.
    <arm>/<patient>/qc/registration/<slide>_QC_RGB.tif
        its downsampled preview, used to pick ROIs and to draw the locator.
    <arm>/<patient>/qc/registration/*_seg_qc.json  and  *_reg_residuals.csv
        the reg_qc=2 scorer's numbers: ``dice_matched`` and the paired-nucleus
        displacement at the final stage (slide-level), and the per-nucleus
        residual table from which each cell prints the median displacement of
        the nuclei INSIDE its ROI.
    <arm>/csv/registered.csv
        which slide is which round (its channel set) and the reference.

The Before column is the left panel of the first arm's composite (every arm
draws the same Before: same native slide, same reference). Rounds are matched
across arms by their channel set, inherited from the shared preprocessing run.
The ASHLAR external arm writes the same composite, checkpoint and QC files
(benchmarks/run_ashlar_arm.sh), so it is a column like any other.

Per cell the corner reads ``Dice = <slide dice_matched>  Δ = <median residual of
the nuclei in this ROI, µm>``; when fewer than ``--min-nuclei`` nuclei fall in
the ROI (or no *_reg_residuals.csv was published, as for a Nextflow arm) the
slide-level displacement is printed instead, marked with ``*``. The Before cell
reads ``Dice = <the first arm's native-stage dice_matched>``: the scorer's own
number for the untransformed pair. A scale bar with its length in µm sits in the
top-left cell (``--scalebar-where``).

Outputs in OUTDIR:
    <patient>_mosaic.png/.pdf   the figure (PDF keeps native pixels, fonts editable)
    <patient>_locator.png/.pdf  low-res reference with numbered ROI boxes
    <patient>_rois.json         ROIs (reference frame, full-res px), files,
                                stretch limits and every cell's numbers; pass
                                back with --rois-json to reuse identical ROIs
    <patient>_patches/          every cell as a PNG at native resolution

Recipes:
    python -m benchmarks.reg_mosaic arm_results/valis_high_micro2 arm_results/tiled_high_gate1 \\
        --rows 6 --patient 5456 -o figs/mosaic
    python -m benchmarks.reg_mosaic arm_results/tiled_high_gate1 arm_results/tiled_high_gate1_solver_robust \\
        arm_results/ashlar_t1024_s30 --rows 4 --kinds overlay,checker -o figs/solver

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
from skimage.registration import phase_cross_correlation

log = logging.getLogger("reg_mosaic")

NUCLEAR_RE = re.compile(r"DAPI|HOECHST|CELLTOX", re.I)
REGISTERED_CSV = Path("csv") / "registered.csv"
QC_SUBDIR = Path("qc") / "registration"
COMPOSITE_SUBDIR = "qc"  # a Nextflow arm publishes the composites under <QC_SUBDIR>/qc/
FULLRES_SUFFIX = "_QC_RGB_fullres.tif"
PREVIEW_SUFFIX = "_QC_RGB.tif"
SEG_QC_SUFFIX = "_seg_qc.json"
RESIDUALS_SUFFIX = "_reg_residuals.csv"
BEFORE_LABEL = "Before"
NATIVE_STAGE = "native"  # warp_seg_qc.py scores the untransformed pair under this stage
CH_MOVING, CH_REFERENCE, CH_SEPARATOR = 0, 1, 2  # red, green, blue in the composite

# how the two colours of the composite are shown; the composite itself is red/green
PALETTES = {
    "magenta-cyan": (
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
    ),  # overlap -> white (the default)
    "magenta-green": ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0)),  # overlap -> white
    "red-green": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ),  # overlap -> yellow (as the QC file)
    "cyan-magenta": ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0)),
    "cyan-red": ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
}
PALETTE_LEGEND = {
    "magenta-cyan": "magenta = moving, cyan = reference, white = overlap",
    "magenta-green": "magenta = moving, green = reference, white = overlap",
    "red-green": "red = moving, green = reference, yellow = overlap",
    "cyan-magenta": "cyan = moving, magenta = reference, white = overlap",
    "cyan-red": "cyan = moving, red = reference, white = overlap",
}
FOOTER_SCORER = (
    "Dice = matched-nucleus Dice, Δ = median nucleus displacement in the ROI "
    "(* slide-level), from the pipeline's reg_qc=2 scorer"
)
FOOTER_IMAGE = (
    "Dice = overlap of Otsu nuclear masks in the crop, Δ = residual shift by phase "
    "correlation; computed from the QC image, no segmentation"
)
# above 1 mm only whole millimetres (1, 2, 5, 10, 20): cleaner on an overview than 2.5 mm
NICE_BARS_UM = (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 20000)


# --- checkpoints --------------------------------------------------------------
@dataclass
class Slide:
    """One row of a mirage checkpoint CSV."""

    patient: str
    slide_id: str
    image: Path
    is_reference: bool
    channels: list[str]
    pixel_size: float | None
    index: int = 0

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

    @property
    def stem(self) -> str:
        """The registered file's stem the way generate_registration_qc.py names outputs."""
        name = self.image.name
        for suf in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
            if name.lower().endswith(suf):
                return name[: -len(suf)]
        return self.image.stem

    @property
    def names(self) -> set[str]:
        """Every name the QC scorer may have used for this slide (VALIS: file stem;
        STARE: the channel list joined by '_', optionally patient-prefixed)."""
        joined = "_".join(self.channels)
        stem = self.stem
        return {
            stem,
            stem.removesuffix("_registered"),
            joined,
            f"{self.patient}_{joined}",
            self.slide_id,
        }


def _truthy(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes", "y", "t")


def _float_or_none(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_checkpoint(
    path: Path, image_col: str = "registered_image"
) -> dict[str, list[Slide]]:
    """A mirage checkpoint (lib/Checkpoint.groovy columns) as patient -> slides."""
    if not path.is_file():
        raise SystemExit(
            f"{path}: not found -- is this an arm directory that reached registration?"
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


# --- the QC composite -----------------------------------------------------------
class TiffSource:
    """TIFF / OME-TIFF (pyramidal or not); crops decode only the tiles they touch."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise SystemExit(f"{self.path}: not found")
        self.tf = tifffile.TiffFile(str(self.path))
        s = self.tf.series[0]
        self.series = s
        self.axes = s.axes
        self.levels = list(s.levels) if s.is_pyramidal else [s]
        self.nchannels = s.shape[self.axes.index("C")] if "C" in self.axes else 1
        self.shape = self._yx(s.shape, self.axes)
        self.px = self._pixel_size()
        self.channel_names = (
            re.findall(r'<Channel[^>]*\bName="([^"]*)"', self.tf.ome_metadata)
            if self.tf.ome_metadata
            else []
        )

    def nuclear_index(self) -> int | None:
        """The DAPI/Hoechst/CellTox channel, by the file's OME channel names; None if unnamed."""
        if self.nchannels == 1:
            return 0
        for i, name in enumerate(self.channel_names):
            if NUCLEAR_RE.search(name):
                return i
        return None

    @staticmethod
    def _yx(shape, axes):
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])

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
                continue
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

    def read(self, ci, ys, xs) -> np.ndarray:
        Hl, Wl = self.shape
        y0, y1, sy = ys.indices(Hl)
        x0, x1, sx = xs.indices(Wl)
        return self._read_region(0, ci, y0, y1, x0, x1)[::sy, ::sx]

    def read_patch(
        self,
        ci: int,
        y: int,
        x: int,
        h: int,
        w: int,
        x_offset: int = 0,
        x_limit: int | None = None,
        step: int = 1,
    ) -> np.ndarray:
        """Crop with the pad-or-crop rule: zero outside [0, x_limit) x [0, H), never resampled.
        ``x_offset`` shifts the read into the file (the After panel starts after the separator).
        ``step`` > 1 returns every step-th pixel (== the full crop[::step, ::step]), read in
        bands so a whole-slide-sized field never sits in memory at full resolution."""
        if step > 1:
            return self._read_patch_strided(ci, y, x, h, w, x_offset, x_limit, step)
        H, W = self.shape
        W = W if x_limit is None else min(W, x_offset + x_limit)
        y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + h, H), min(x + w, W - x_offset)
        if y1 <= y0 or x1 <= x0:
            return np.zeros((h, w), self.series.dtype)
        sub = self.read(ci, slice(y0, y1), slice(x0 + x_offset, x1 + x_offset))
        if (y0, x0, y1, x1) == (y, x, y + h, x + w):
            return sub
        out = np.zeros((h, w), sub.dtype)
        out[y0 - y : y1 - y, x0 - x : x1 - x] = sub
        return out

    def _read_patch_strided(
        self, ci, y, x, h, w, x_offset, x_limit, step, band_rows=4096
    ):
        H, W = self.shape
        W = W if x_limit is None else min(W, x_offset + x_limit)
        ys, xs = np.arange(y, y + h, step), np.arange(x, x + w, step)
        out = np.zeros((len(ys), len(xs)), self.series.dtype)
        iy = np.flatnonzero((ys >= 0) & (ys < H))
        ix = np.flatnonzero((xs >= 0) & (xs < W - x_offset))
        if not iy.size or not ix.size:
            return out
        xa, xb = int(xs[ix[0]]) + x_offset, int(xs[ix[-1]]) + 1 + x_offset
        per_band = max(1, band_rows // step)
        for k in range(0, iy.size, per_band):
            rows = iy[k : k + per_band]
            ya, yb = int(ys[rows[0]]), int(ys[rows[-1]]) + 1
            region = self._read_region(0, ci, ya, yb, xa, xb)
            out[np.ix_(rows, ix)] = region[::step, ::step]
        return out

    def overview(self, ci: int, max_px: int) -> tuple[np.ndarray, float]:
        """The whole plane at about ``max_px`` on its long side, plus full-res px per output px.

        Read from the smallest pyramid level still at least ``max_px`` long (a whole-slide
        pyramid has one), then strided -- never the full-resolution plane in memory.
        """
        H, W = self.shape
        level = 0
        for lv in range(len(self.levels)):
            if max(self.level_shape(lv)) >= max_px:
                level = lv
        h, w = self.level_shape(level)
        # rounded, not ceiled: a pyramid level of 3250 px and max_px 2400 would otherwise
        # halve to 1625; this keeps the overview within ~1.5x of max_px either way
        step = max(1, round(max(h, w) / max_px))
        ys = range(0, h, step)
        rows, per_band = [], max(1, 4096 // step)
        ys = list(ys)
        for k in range(0, len(ys), per_band):
            ya, yb = ys[k], ys[min(k + per_band, len(ys)) - 1] + 1
            rows.append(self._read_region(level, ci, ya, yb, 0, w)[::step, ::step])
        low = np.concatenate(rows, axis=0)
        return low, W / low.shape[1]

    def close(self):
        self.tf.close()


class Composite:
    """One registration-QC composite: the Before panel, a separator, the After panel."""

    def __init__(self, path: Path, preview: Path | None = None):
        self.src = TiffSource(path)
        if self.src.nchannels != 3:
            raise SystemExit(
                f"{path.name}: expected a 3-channel RGB composite, found {self.src.nchannels} channel(s)"
            )
        H, Wtot = self.src.shape
        self.height = H
        self.before_width, self.after_offset, self.after_width = self._split(Wtot)
        self.preview = TiffSource(preview) if preview and preview.is_file() else None
        self.px = self.src.px

    def _split(self, wtot: int) -> tuple[int, int, int]:
        """Panel geometry from the blue separator (a band of B=255 with no red/green).
        A single-panel composite (no native panel) has no Before."""
        y = self.height // 2
        b = self.src.read(CH_SEPARATOR, slice(y, y + 1), slice(None))[0]
        r = self.src.read(CH_MOVING, slice(y, y + 1), slice(None))[0]
        g = self.src.read(CH_REFERENCE, slice(y, y + 1), slice(None))[0]
        sep = np.flatnonzero((b == 255) & (r == 0) & (g == 0))
        if sep.size == 0:
            return 0, 0, wtot
        # the separator is one contiguous band; anything else blue is not one
        start, end = int(sep[0]), int(sep[-1]) + 1
        if end - start != sep.size:
            raise SystemExit(
                f"{self.src.path.name}: cannot find one contiguous blue separator"
            )
        return start, end, wtot - end

    @property
    def has_before(self) -> bool:
        return self.before_width > 0

    @property
    def canvas(self) -> tuple[int, int]:
        return self.height, self.after_width

    def crop(
        self, panel: str, y: int, x: int, h: int, w: int, step: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """(reference, moving) uint8 crops of the Before or After panel."""
        if panel == "before":
            if not self.has_before:
                raise SystemExit(f"{self.src.path.name} carries no Before panel")
            off, lim = 0, self.before_width
        else:
            off, lim = self.after_offset, self.after_width
        ref = self.src.read_patch(CH_REFERENCE, y, x, h, w, off, lim, step)
        mov = self.src.read_patch(CH_MOVING, y, x, h, w, off, lim, step)
        return ref, mov

    def lowres_reference(self, factor_hint: int) -> tuple[np.ndarray, float]:
        """The reference (green) plane of the After panel, downsampled: from the preview
        when present (factor from its width), else by striding the full-res file."""
        if self.preview is not None:
            H, Wtot = self.preview.shape
            pb, po, pw = Composite._split_static(self.preview, H, Wtot)
            g = self.preview.read(CH_REFERENCE, slice(None), slice(po, po + pw))
            return g, self.after_width / pw
        step = max(1, factor_hint)
        g = self.src.read(
            CH_REFERENCE,
            slice(None, None, step),
            slice(self.after_offset, self.after_offset + self.after_width, step),
        )
        return g, float(step)

    @staticmethod
    def _split_static(src: TiffSource, H: int, wtot: int) -> tuple[int, int, int]:
        y = H // 2
        b = src.read(CH_SEPARATOR, slice(y, y + 1), slice(None))[0]
        r = src.read(CH_MOVING, slice(y, y + 1), slice(None))[0]
        g = src.read(CH_REFERENCE, slice(y, y + 1), slice(None))[0]
        sep = np.flatnonzero((b == 255) & (r == 0) & (g == 0))
        if sep.size == 0:
            return 0, 0, wtot
        return int(sep[0]), int(sep[-1]) + 1, wtot - int(sep[-1]) - 1

    def close(self):
        self.src.close()
        if self.preview is not None:
            self.preview.close()


# --- the seg QC numbers ---------------------------------------------------------
@dataclass
class SegQC:
    stage: str
    dice: float | None
    displacement_um: float | None
    displacement_px: float | None
    n_pairs: int | None
    native_dice: float | None = (
        None  # the "native" stage: the pair before any transform
    )
    residuals: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )  # ref_x, ref_y, residual_px

    def local_displacement_px(
        self, y: int, x: int, h: int, w: int, min_nuclei: int
    ) -> tuple[float | None, int]:
        if self.residuals.size == 0:
            return None, 0
        rx, ry, res = self.residuals.T
        inside = (rx >= x) & (rx < x + w) & (ry >= y) & (ry < y + h)
        n = int(inside.sum())
        if n < min_nuclei:
            return None, n
        return float(np.median(res[inside])), n


def _read_seg_qc(path: Path) -> tuple[str, dict]:
    d = json.loads(path.read_text())
    stages = d.get("stages") or {}
    order = d.get("stage_order") or list(stages)
    final = order[-1] if order else None
    return d.get("moving", ""), {
        "stage": final,
        "final": stages.get(final) or {},
        "native": stages.get(NATIVE_STAGE) or {},
    }


def load_seg_qc(qc_dir: Path, slide: Slide) -> SegQC | None:
    """The scorer's final-stage numbers for ``slide``, matched by the name it recorded."""
    if not qc_dir.is_dir():
        return None
    for js in sorted(qc_dir.glob(f"*{SEG_QC_SUFFIX}")):
        moving, info = _read_seg_qc(js)
        if moving not in slide.names and not any(
            moving.endswith(n) for n in slide.names
        ):
            continue
        final = info["final"]
        stage = info["stage"]
        residuals = np.zeros((0, 3))
        csv_path = js.with_name(js.name[: -len(SEG_QC_SUFFIX)] + RESIDUALS_SUFFIX)
        if csv_path.is_file():
            rows = []
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if stage and row.get("stage") not in (None, "", stage):
                        continue
                    try:
                        rows.append(
                            (
                                float(row["ref_x"]),
                                float(row["ref_y"]),
                                float(row["residual_px"]),
                            )
                        )
                    except (KeyError, ValueError):
                        continue
            if rows:
                residuals = np.asarray(rows, dtype=float)
        return SegQC(
            stage=stage or "",
            dice=_float_or_none(final.get("dice_matched")),
            displacement_um=_float_or_none(final.get("displacement_um_p50")),
            displacement_px=_float_or_none(final.get("displacement_px_p50")),
            n_pairs=final.get("n_pairs"),
            native_dice=_float_or_none(info["native"].get("dice_matched")),
            residuals=residuals,
        )
    return None


# --- one arm = one column -------------------------------------------------------
class Arm:
    """One registration output directory: one column of the mosaic.

    PIXELS COME FROM THE ORIGINAL 16-BIT SLIDES by default (``source='auto'``): the
    registered moving slide (csv/registered.csv), the native moving slide as it entered
    registration (csv/preprocessed.csv of this run or of a sibling preprocess_shared/), and
    the reference. The 8-bit QC composite is only the fallback. It is min-max scaled over
    the WHOLE slide, so a handful of extreme pixels in a registered slide pushes all the
    real tissue into 2-3 grey levels, and re-stretching a crop of that gives a flat,
    speckled field with a hard contour edge -- observed on 5 of 9 rounds of a real mosaic
    (2026-09-17), identical for VALIS and STARE, while the native slide was fine.
    """

    def __init__(
        self,
        root: Path,
        label: str | None = None,
        source: str = "auto",
        native_csv: Path | None = None,
    ):
        self.root = Path(root)
        self.name = label or self.root.name
        self.per = read_checkpoint(self.root / REGISTERED_CSV)
        self.ref: Slide | None = None
        self.moving: dict[str, Slide] = {}
        self._comp: dict[str, Composite] = {}
        self._qc: dict[str, SegQC | None] = {}
        self._src: dict[Path, TiffSource] = {}
        self._orig: dict[str, tuple | None] = {}
        self.source = source
        self.patient = ""
        self.native_csv = native_csv or next(
            (
                c
                for c in (
                    self.root / "csv" / "preprocessed.csv",
                    self.root.parent / "preprocess_shared" / "csv" / "preprocessed.csv",
                )
                if c.is_file()
            ),
            None,
        )
        self.native_per = (
            read_checkpoint(self.native_csv, image_col="preprocessed_image")
            if self.native_csv
            else {}
        )

    def patients(self) -> list[str]:
        return list(self.per)

    def open(self, patient: str) -> None:
        rows = self.per.get(patient)
        if not rows:
            raise SystemExit(
                f"[{self.name}] patient {patient!r} not in {self.root / REGISTERED_CSV}"
            )
        refs = [r for r in rows if r.is_reference]
        if len(refs) != 1:
            raise SystemExit(
                f"[{self.name}] {patient}: expected one reference row, found {len(refs)}"
            )
        self.patient = patient
        self.ref = refs[0]
        self.moving = {r.key: r for r in rows if not r.is_reference}
        self._comp, self._qc, self._orig = {}, {}, {}

    @property
    def qc_dir(self) -> Path:
        return self.root / self.patient / QC_SUBDIR

    def slide(self, key: str) -> Slide:
        if key not in self.moving:
            raise SystemExit(
                f"[{self.name}] {self.patient}: no registered slide for round {key!r}; this arm has "
                f"{sorted(self.moving)} -- did it finish for this patient?"
            )
        return self.moving[key]

    def composite_path(self, key: str) -> Path | None:
        """The slide's full-res composite, in whichever of the two layouts holds it.

        A Nextflow arm publishes it one level down, <qc_dir>/qc/: GENERATE_REGISTRATION_QC's
        publishDir pattern is "qc/*_QC_RGB_fullres.tif" and a pattern keeps its relative
        path. run_ashlar_arm.sh writes it flat into <qc_dir>. The preview sits beside it.
        """
        name = f"{self.slide(key).stem}{FULLRES_SUFFIX}"
        for d in (self.qc_dir, self.qc_dir / COMPOSITE_SUBDIR):
            if (d / name).is_file():
                return d / name
        return None

    def composite(self, key: str) -> Composite:
        if key not in self._comp:
            full = self.composite_path(key)
            if full is None:
                name = f"{self.slide(key).stem}{FULLRES_SUFFIX}"
                raise SystemExit(
                    f"[{self.name}] {self.patient}: no registration QC composite {name} in {self.qc_dir} "
                    f"or {self.qc_dir / COMPOSITE_SUBDIR} -- the arm's GENERATE_REGISTRATION_QC did not run for this slide"
                )
            preview = full.with_name(full.name[: -len(FULLRES_SUFFIX)] + PREVIEW_SUFFIX)
            self._comp[key] = Composite(full, preview)
        return self._comp[key]

    def _open(self, path: Path) -> TiffSource:
        if path not in self._src:
            self._src[path] = TiffSource(path)
        return self._src[path]

    def originals(self, key: str):
        """(reference, ref channel, registered, reg channel, native, native channel) for a
        round, or None -- with the reason logged -- when any of them cannot be used."""
        if key in self._orig:
            return self._orig[key]
        sl, ref = self.slide(key), self.ref
        natives = {
            r.key: r
            for r in self.native_per.get(self.patient, [])
            if not r.is_reference
        }
        why, got = None, None
        if key not in natives:
            why = f"round {key!r} not in {self.native_csv or 'any preprocessed.csv'}"
        else:
            paths = (ref.image, sl.image, natives[key].image)
            missing = [str(p) for p in paths if not Path(p).is_file()]
            if missing:
                why = f"missing {missing}"
            else:
                srcs = [self._open(Path(p)) for p in paths]
                idx = [src.nuclear_index() for src in srcs]
                if None in idx:
                    why = (
                        "no DAPI/Hoechst/CellTox among the OME channel names of "
                        + ", ".join(
                            src.path.name for src, i in zip(srcs, idx) if i is None
                        )
                    )
                else:
                    got = (srcs[0], idx[0], srcs[1], idx[1], srcs[2], idx[2])
        if got is None:
            log.warning(
                "[%s] %s %s: original slides unusable (%s); using the 8-bit QC composite",
                self.name,
                self.patient,
                key,
                why,
            )
        self._orig[key] = got
        return got

    def crop(self, key: str, panel: str, y: int, x: int, h: int, w: int, step: int = 1):
        """(reference, moving, source) for one panel on the reference canvas.

        before = the native moving slide pad-or-cropped at the origin (bin/utils/qc.py's
        compose_on_reference_canvas); after = the registered slide, already on the canvas.
        """
        orig = None if self.source == "composite" else self.originals(key)
        if orig is None:
            if self.source == "originals":
                raise SystemExit(
                    f"[{self.name}] {self.patient} {key}: --source originals unusable"
                )
            ref, mov = self.composite(key).crop(panel, y, x, h, w, step)
            return ref, mov, "composite"
        rsrc, ri, gsrc, gi, nsrc, ni = orig
        try:
            ref = rsrc.read_patch(ri, y, x, h, w, step=step)
            mov = (
                gsrc.read_patch(gi, y, x, h, w, step=step)
                if panel == "after"
                else nsrc.read_patch(ni, y, x, h, w, step=step)
            )
        except (ValueError, OSError, RuntimeError) as exc:
            # typically a compression this environment cannot decode: the pipeline's slides
            # are LZW, which tifffile decodes only with imagecodecs (job 6844142 ran in an
            # image without it). One warning per round, then the composite.
            if self.source == "originals":
                raise SystemExit(f"[{self.name}] {self.patient} {key}: {exc}") from exc
            log.warning(
                "[%s] %s %s: cannot decode the original slides (%s); using the 8-bit QC "
                "composite. Run in an image with imagecodecs (bolt3x/mirage-quantify).",
                self.name,
                self.patient,
                key,
                exc,
            )
            self._orig[key] = None
            return self.crop(key, panel, y, x, h, w, step)
        return ref, mov, "originals"

    def seg_qc(self, key: str, warn: bool = True) -> SegQC | None:
        if key not in self._qc:
            self._qc[key] = load_seg_qc(self.qc_dir, self.slide(key))
            if self._qc[key] is None and warn:
                log.warning(
                    "[%s] %s: no *_seg_qc.json names round %s in %s -- cells carry no numbers",
                    self.name,
                    self.patient,
                    key,
                    self.qc_dir,
                )
        return self._qc[key]

    @property
    def px(self) -> float | None:
        assert self.ref is not None
        return self.ref.pixel_size

    def files(self, keys) -> dict[str, str]:
        return {
            k: str(
                self.composite_path(k)
                or self.qc_dir / f"{self.moving[k].stem}{FULLRES_SUFFIX}"
            )
            for k in keys
            if k in self.moving
        }

    def close(self):
        for c in self._comp.values():
            c.close()
        for src in self._src.values():
            src.close()
        self._comp, self._src, self._orig = {}, {}, {}


# --- rows -----------------------------------------------------------------------
def plan_rows(round_keys: list[str], n_rois: int, n_rows: int) -> list[tuple[str, int]]:
    """Exactly ``n_rows`` (round, roi) pairs, ROI-major.

    ROI-major so a truncated plan still shows every round at the first ROI
    before it shows any round at a second one: ``--rows 4`` over three rounds
    gives (ROI 1 x all three rounds) + (ROI 2 x the first round).
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


def image_metrics(
    ref: np.ndarray, mov: np.ndarray, sigma: float = 1.0
) -> tuple[float | None, float | None]:
    """(Dice, residual shift in px) of one crop pair, from the pixels alone.

    The fast stand-in for the reg_qc=2 scorer when WARP_SEG_QC did not run. Dice is the
    overlap of the two Otsu nuclear masks (a PIXEL Dice, not the scorer's matched-nucleus
    Dice, so the two are not interchangeable in a table); the shift is the translation
    phase correlation still finds between the crops -- 0 for a perfect registration.
    """
    r = ndimage.gaussian_filter(np.asarray(ref, np.float32), sigma)
    m = ndimage.gaussian_filter(np.asarray(mov, np.float32), sigma)
    if r.max() == r.min() or m.max() == m.min():
        return None, None
    a, b = r > threshold_otsu(r), m > threshold_otsu(m)
    den = int(a.sum() + b.sum())
    dice = 2.0 * float((a & b).sum()) / den if den else None
    # Plain (not phase-normalised) cross-correlation of mean-free, Hann-windowed crops. The
    # phase-normalised default locks onto the crop's own hard edges -- a small window is not
    # periodic -- and reports ~0 for a 15 px misalignment (measured on the test fixture).
    win = np.outer(np.hanning(r.shape[0]), np.hanning(r.shape[1]))
    shift, _err, _phase = phase_cross_correlation(
        (r - r.mean()) * win,
        (m - m.mean()) * win,
        upsample_factor=10,
        normalization=None,
    )
    return dice, float(np.hypot(*shift))


def image_note(ref, mov, px: float | None) -> tuple[str, dict]:
    """``Dice = 0.81  Δ = 0.4 µm`` computed from the crop (see image_metrics)."""
    dice, shift_px = image_metrics(ref, mov)
    vals = {"source": "image", "dice_pixel": dice, "shift_px": shift_px}
    parts = []
    if dice is not None:
        parts.append(f"Dice = {dice:.2f}")
    if shift_px is not None:
        vals["shift_um"] = shift_px * px if px else None
        parts.append(f"Δ = {shift_px * px:.1f} µm" if px else f"Δ = {shift_px:.1f} px")
    return "  ".join(parts), vals


def select_rois(
    low: np.ndarray,
    factor: float,
    patch_px: int,
    n: int,
    min_sep: float,
    spread: float,
    full_shape,
    exclude=(),
) -> list[tuple[int, int]]:
    """Pick `n` patch positions on a low-res nuclear image.

    Score = tissue coverage (window mostly inside tissue, Otsu on the log image)
          x local texture (std of log intensity within the tissue pixels)
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
    den = np.maximum(fg_frac, 1e-3)
    m1 = _box(f * fg, win) / den
    m2 = _box(f * f * fg, win) / den
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    sat = _box((img >= np.percentile(img, 99.98)).astype(np.float32), win)
    cover = np.clip((fg_frac - 0.5) / 0.4, 0.0, 1.0)
    norm = float(np.percentile(std[fg > 0], 95)) if fg.any() else float(std.max())
    texture = np.clip(std / (norm + 1e-6), 0.0, 1.0)
    score = cover * texture * np.clip(1.0 - 25.0 * sat, 0.0, 1.0)
    half = win // 2
    ok = np.zeros(score.shape, bool)
    ok[half : score.shape[0] - half, half : score.shape[1] - half] = True
    score[~ok] = 0.0
    # exclude = [(y, x, size_px), ...] full-res boxes (e.g. another figure's ROIs): no
    # window centred here may overlap one, so the two figures show different tissue.
    for ey, ex, esize in exclude:
        cy, cx = (ey + esize / 2) / factor, (ex + esize / 2) / factor
        hy = (patch_px + esize) / 2 / factor
        y0, y1 = max(int(cy - hy), 0), min(int(np.ceil(cy + hy)) + 1, score.shape[0])
        x0, x1 = max(int(cx - hy), 0), min(int(np.ceil(cx + hy)) + 1, score.shape[1])
        score[y0:y1, x0:x1] = 0.0
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
    """Per-channel MAXIMUM of the two coloured planes, not their sum.

    magenta (1,0,1) and cyan (0,1,1) share blue. Added, blue clips before red and green, so
    equal overlap renders lavender and the background purple (a real overlay measured mean
    blue 0.53 against 0.30 red/green). With the maximum, equal overlap is neutral grey/white
    and each colour alone is unchanged; for palettes sharing no channel (red-green) the two
    rules are identical.
    """
    cm, cr = (np.asarray(c, np.float32) for c in PALETTES[palette])
    return np.clip(np.maximum(mov01[..., None] * cm, ref01[..., None] * cr), 0.0, 1.0)


def checkerboard(mov01, ref01, tiles) -> np.ndarray:
    h, w = ref01.shape
    ty, tx = max(1, -(-h // tiles)), max(1, -(-w // tiles))
    yy, xx = np.indices((h, w))
    g = np.where(((yy // ty + xx // tx) % 2) == 0, ref01, mov01)
    return np.repeat(g[..., None], 3, axis=-1)


def scalebar_label(um: float) -> str:
    """``500 µm`` below a millimetre, ``1 mm`` / ``2.5 mm`` from one millimetre up.

    The bar is about a quarter of the field, so fields from ~4 mm get a mm label -- the unit
    of whole-slide overviews -- while a 200-500 µm crop keeps µm.
    """
    return f"{um / 1000:g} mm" if um >= 1000 else f"{um:g} µm"


def draw_legend(ax, entries, font, x=0.97, y=0.03, spacing=1.25):
    """Channel names in their own colours, stacked and right-aligned in the lower right
    (the first entry on top), each with a thin dark outline for legibility."""
    texts = []
    n = len(entries)
    for i, (name, color) in enumerate(entries):
        t = ax.text(
            x,
            y
            + (n - 1 - i)
            * spacing
            * font
            / 72.0
            / ax.figure.get_size_inches()[1]
            / max(ax.get_position().height, 1e-6),
            name,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=font,
            color=color,
        )
        _outline(t)
        texts.append(t)
    return texts


def auto_scalebar_um(patch_um: float) -> float:
    target = patch_um / 4.0 * 1.05
    return float(max([b for b in NICE_BARS_UM if b <= target] or [NICE_BARS_UM[0]]))


def cell_note(
    qc: SegQC | None, y: int, x: int, size: int, px: float | None, min_nuclei: int
) -> tuple[str, dict]:
    """``Dice = 0.87  Δ = 1.3 µm`` (+ ``*`` when the slide-level displacement stands in)."""
    if qc is None:
        return "", {}
    parts, vals = (
        [],
        {
            "stage": qc.stage,
            "dice_matched": qc.dice,
            "slide_displacement_um": qc.displacement_um,
        },
    )
    if qc.dice is not None:
        parts.append(f"Dice = {qc.dice:.2f}")
    local_px, n = qc.local_displacement_px(y, x, size, size, min_nuclei)
    vals["n_nuclei_in_roi"] = n
    if local_px is not None:
        vals["roi_displacement_px"] = local_px
        vals["roi_displacement_um"] = local_px * px if px else None
        parts.append(f"Δ = {local_px * px:.1f} µm" if px else f"Δ = {local_px:.1f} px")
    elif qc.displacement_um is not None:
        parts.append(f"Δ = {qc.displacement_um:.1f} µm*")
    elif qc.displacement_px is not None:
        parts.append(f"Δ = {qc.displacement_px:.1f} px*")
    return "  ".join(parts), vals


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


def draw_scalebar(ax, h, w, bar_px, label, font, thick=0.014, color="white"):
    from matplotlib.patches import Rectangle

    x0, y0 = 0.05 * w, 0.92 * h
    ax.add_patch(Rectangle((x0, y0), bar_px, max(1.0, thick * h), color=color, lw=0))
    _outline(
        ax.text(
            x0 + bar_px / 2,
            y0 - 0.012 * h,
            label,
            color=color,
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
    footer: str = "",
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
                    # top-right: the bottom-left corner holds the scale bar
                    ax.text(
                        0.97,
                        0.96,
                        notes[r][c],
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=font * 0.8,
                        color="white",
                    )
                )
            if (
                scalebar
                and c == 0
                and (
                    scalebar_where == "all"
                    or (scalebar_where == "first" and r == 0)
                    or (scalebar_where == "last" and r == nr - 1)
                )
            ):
                h, w = grid[r][c].shape[:2]
                draw_scalebar(ax, h, w, scalebar[0], scalebar[1], font * 0.8)
    if footer:
        fig.text(0.0, -0.004, footer, ha="left", va="top", fontsize=font * 0.72)
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
            ax, h, w, bar_um / (px * factor), scalebar_label(bar_um), 7, thick=0.006
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
    palette: str = "magenta-cyan"
    numbers: str = "auto"
    orient: str = "rounds-as-columns"
    stretch: str = "patch"
    pmin: float = 1.0
    pmax: float = 99.8
    gamma: float = 1.0
    checker_tiles: int = 6
    annotate_where: str = "all"
    min_nuclei: int = 5
    scalebar_um: float | None = None
    scalebar_where: str = "first"
    cell_in: float | None = None
    source: str = "auto"
    native_csv: Path | None = None
    dpi: int = 300
    formats: str = "png,pdf"
    pixel_size_um: float | None = None
    lowres_um: float = 5.0


def process_patient(pid: str, arms: list[Arm], opt: Options) -> dict:
    for arm in arms:
        arm.open(pid)
    first = arms[0]
    assert first.ref is not None
    moving = list(first.moving.values())
    if opt.rounds:
        moving = [sl for sl in moving if round_matches(sl, opt.rounds)]
    if not moving:
        raise SystemExit(f"{pid}: no moving round matches {opt.rounds}")
    for arm in arms:
        for sl in moving:
            arm.slide(sl.key)  # names the missing round, if any
    keys = [sl.key for sl in moving]
    log.info(
        "== %s: reference %s; %d moving round(s): %s",
        pid,
        first.ref.image.name,
        len(moving),
        ", ".join(keys),
    )

    # the frame and the Before panel come from the first arm's composite of the first round
    lead = first.composite(keys[0])
    if not lead.has_before:
        raise SystemExit(
            f"[{first.name}] {pid}: its composite has no Before panel; put an arm whose QC ran with --native first"
        )
    H, W = lead.canvas
    px = opt.pixel_size_um or lead.px or first.px
    if opt.patch_px:
        patch_px = opt.patch_px
    elif px:
        patch_px = int(round(opt.patch_um / px))
    else:
        raise SystemExit("pixel size unknown; pass --pixel-size-um or --patch-px")
    patch_um = patch_px * px if px else None
    log.info(
        "reference canvas %dx%d px, %s µm/px; patch %d px%s",
        H,
        W,
        px,
        patch_px,
        f" = {patch_um:.0f} µm" if patch_um else "",
    )
    for arm in arms:
        for k in keys:
            if (
                arm is not first
                and arm.source != "composite"
                and arm.composite_path(k) is None
            ):
                # no QC composite (e.g. an external arm whose QC step failed): fine as long as
                # its original slides are usable, which crop() checks and falls back from
                continue
            c = arm.composite(k)
            if c.canvas != (H, W):
                log.warning(
                    "[%s] %s: composite canvas %s differs from %s; ROI coordinates assume one reference canvas",
                    arm.name,
                    k,
                    c.canvas,
                    (H, W),
                )

    factor_hint = max(1, int(round(opt.lowres_um / px))) if px else 16
    low, f = lead.lowres_reference(factor_hint)

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
    plan = plan_rows(keys, len(rois), opt.rows)
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
        f"{pid} — {first.ref.label} (reference)",
        opt.dpi,
    )

    def limits(col: str, chan: str, key: str, crop) -> tuple[float, float]:
        if opt.stretch == "patch":
            return percentile_limits(crop, opt.pmin, opt.pmax)
        # no re-stretch: the composite's own 8-bit scaling, or the originals' full dtype range
        return (
            0.0,
            float(np.iinfo(crop.dtype).max if crop.dtype.kind in "ui" else 1.0),
        )

    kinds = [k.strip() for k in opt.kinds.split(",") if k.strip()]
    col_names = [BEFORE_LABEL] + [a.name for a in arms]
    col_labels = [
        c if k == "overlay" else f"{c} (checker)" for k in kinds for c in col_names
    ]
    pdir = outdir / f"{pid}_patches"
    pdir.mkdir(exist_ok=True)

    grid, notes, row_labels, rows_meta = [], [], [], []
    sources: set[str] = set()
    for key, ri in plan:
        sl = by_key[key]
        y, x = rois[ri]
        crops, used = {}, {}
        *crops[BEFORE_LABEL], used[BEFORE_LABEL] = first.crop(
            key, "before", y, x, patch_px, patch_px
        )
        for arm in arms:
            *crops[arm.name], used[arm.name] = arm.crop(
                key, "after", y, x, patch_px, patch_px
            )
        lim = {
            n: (limits(n, "ref", key, rc), limits(n, "mov", key, mc))
            for n, (rc, mc) in crops.items()
        }
        st = {
            n: (stretch(rc, lim[n][0], opt.gamma), stretch(mc, lim[n][1], opt.gamma))
            for n, (rc, mc) in crops.items()
        }
        cells_meta: dict = {}
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
                    rc, mc = crops[n]
                    arm = (
                        first
                        if n == BEFORE_LABEL
                        else next(a for a in arms if a.name == n)
                    )
                    qc = (
                        None
                        if opt.numbers in ("image", "none")
                        else arm.seg_qc(key, warn=opt.numbers == "scorer")
                    )
                    meta = {
                        "ref_limits": lim[n][0],
                        "mov_limits": lim[n][1],
                        "pixels": used[n],
                    }
                    if opt.numbers == "none":
                        note, vals = "", {}
                    elif qc is None and opt.numbers != "scorer":
                        # no scorer output (reg_qc < 2) or --numbers image: from the crop
                        note, vals = image_note(rc, mc, px)
                        sources.add("image")
                    elif n == BEFORE_LABEL:
                        # The Before image is the first arm's composite, so its number is the
                        # first arm's scorer on the untransformed pair (the native stage). It
                        # can differ by a hair between arms: each pairs nuclei at its own
                        # anchor stage.
                        native = qc.native_dice if qc else None
                        vals = (
                            {
                                "source": "scorer",
                                "stage": NATIVE_STAGE,
                                "dice_matched": native,
                            }
                            if native is not None
                            else {}
                        )
                        note = f"Dice = {native:.2f}" if native is not None else ""
                        sources.add("scorer")
                    else:
                        note, vals = cell_note(qc, y, x, patch_px, px, opt.min_nuclei)
                        vals = {"source": "scorer", **vals} if vals else vals
                        sources.add("scorer")
                    cells_meta[n] = {**meta, **vals}
                    if n == BEFORE_LABEL and opt.annotate_where == "after":
                        note = ""
                row_grid.append(img)
                row_notes.append(note)
                write_png(pdir / f"r{sl.index:02d}_{key}_roi{ri + 1}_{n}_{k}.png", img)
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
        scalebar = (bar_um / px, scalebar_label(bar_um), bar_um)
    if opt.orient == "rounds-as-columns":
        # One column per (round, ROI), one row per Before + arm: reading DOWN a column
        # compares the methods on the same tissue. Transposing the finished grid keeps every
        # cell, note and patch PNG identical; only the titles swap sides.
        grid = [list(col) for col in zip(*grid)]
        notes = [list(col) for col in zip(*notes)]
        row_labels, col_labels = (
            col_labels,
            [lab.replace("\n", "  ") for lab in row_labels],
        )
    assemble_figure(
        grid,
        notes,
        row_labels,
        col_labels,
        outdir / f"{pid}_mosaic",
        formats,
        opt.cell_in or patch_px / opt.dpi,
        opt.dpi,
        scalebar,
        opt.scalebar_where,
        footer="\n".join(
            [PALETTE_LEGEND.get(opt.palette, "")]
            + [FOOTER_SCORER] * ("scorer" in sources)
            + [FOOTER_IMAGE] * ("image" in sources)
        ),
    )

    manifest = {
        "patient": pid,
        "reference": str(first.ref.image),
        "pixel_size_um": px,
        "patch_px": patch_px,
        "patch_um": patch_um,
        "rows": opt.rows,
        "palette": opt.palette,
        "numbers": opt.numbers,
        "orient": opt.orient,
        "number_sources": sorted(sources),
        "kinds": kinds,
        "stretch": opt.stretch,
        "pmin": opt.pmin,
        "pmax": opt.pmax,
        "gamma": opt.gamma,
        "source": opt.source,
        "pixels": sorted(
            {c.get("pixels") for r in rows_meta for c in r["cells"].values()} - {None}
        ),
        "min_nuclei": opt.min_nuclei,
        "scalebar_um": scalebar and float(scalebar[2]),
        "rois": [{"id": i, "y": y, "x": x} for i, (y, x) in enumerate(rois, 1)],
        "columns": {
            BEFORE_LABEL: {
                "dir": str(first.root),
                "files": first.files(keys),
                "panel": "before",
            },
            **{
                a.name: {"dir": str(a.root), "files": a.files(keys), "panel": "after"}
                for a in arms
            },
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
    for arm in arms:
        arm.close()
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
        help="one or more <arm_results>/<arm> directories (one column each; the first one also supplies the Before column)",
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
    g.add_argument(
        "--orient",
        choices=("rounds-as-columns", "rounds-as-rows"),
        default="rounds-as-columns",
        help="rounds-as-columns: one column per (round, ROI), rows Before + one per arm "
        "(landscape; compare methods down a column); rounds-as-rows: the transpose",
    )
    g.add_argument("--kinds", default="overlay", help="comma list of: overlay, checker")
    g.add_argument(
        "--palette",
        choices=list(PALETTES),
        default="magenta-cyan",
        help="moving/reference colours (red-green = as the QC file)",
    )
    g.add_argument(
        "--numbers",
        choices=("auto", "scorer", "image", "none"),
        default="auto",
        help="where Dice/Δ come from: scorer = the reg_qc=2 *_seg_qc.json (WARP_SEG_QC); "
        "image = computed from the crop (Otsu-mask Dice, phase-correlation shift), for a run "
        "without WARP_SEG_QC; auto = scorer when its JSON exists, else image; none = no "
        "numbers in the cells (a prototype whose numbers come later from reg_qc=2)",
    )
    g.add_argument(
        "--stretch",
        choices=("patch", "global"),
        default="patch",
        help="re-stretch each crop per channel by percentiles, or keep the QC file's global scaling",
    )
    g.add_argument("--pmin", type=float, default=1.0)
    g.add_argument("--pmax", type=float, default=99.8)
    g.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="display gamma (<1 lifts the background)",
    )
    g.add_argument("--checker-tiles", type=int, default=6)
    g.add_argument(
        "--annotate-where",
        choices=("after", "all"),
        default="all",
        help="all = every cell, Before included (its scorer's native-stage Dice); "
        "after = the arm columns only",
    )
    g.add_argument(
        "--min-nuclei",
        type=int,
        default=5,
        help="nuclei an ROI needs for a local displacement; below it the slide-level value is printed with *",
    )
    g.add_argument(
        "--scalebar-um",
        type=float,
        default=None,
        help="scale bar length (default auto ≈ patch/4; 0 = none)",
    )
    g.add_argument(
        "--scalebar-where",
        choices=("first", "last", "all"),
        default="first",
        help="first = the top-left cell only (IF panel convention); last = the bottom-left one; "
        "all = every row's Before cell",
    )
    g.add_argument(
        "--cell-in",
        type=float,
        default=None,
        help="cell size in inches; default = patch px / dpi, i.e. one image pixel per output "
        "pixel (a smaller cell downsamples every crop)",
    )
    g.add_argument("--dpi", type=int, default=300)
    g.add_argument("--formats", default="png,pdf")
    g = ap.add_argument_group("inputs")
    g.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="override the pixel size (default: the QC TIFF's tag, then csv/registered.csv)",
    )
    g.add_argument(
        "--lowres-um",
        type=float,
        default=5.0,
        help="pixel size of the low-res image used for ROI selection when no preview TIFF exists",
    )
    g.add_argument(
        "--source",
        choices=("auto", "originals", "composite"),
        default="auto",
        help="pixels from the original 16-bit slides (registered, native, reference) or the "
        "8-bit QC composite; auto = originals, falling back per round to the composite",
    )
    g.add_argument(
        "--native-csv",
        type=Path,
        default=None,
        help="preprocessed.csv naming the native slides (default: <arm>/csv/ or "
        "<arm>/../preprocess_shared/csv/)",
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
    arms = [Arm(d, labels.get(d.name), args.source, args.native_csv) for d in args.arm]
    names = [a.name for a in arms]
    if len(set(names)) != len(names) or BEFORE_LABEL in names:
        raise SystemExit(
            f"column titles must be distinct and not {BEFORE_LABEL!r}: {names}; pass --label ARM=Title"
        )

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
        numbers=args.numbers,
        orient=args.orient,
        stretch=args.stretch,
        pmin=args.pmin,
        pmax=args.pmax,
        gamma=args.gamma,
        checker_tiles=args.checker_tiles,
        annotate_where=args.annotate_where,
        min_nuclei=args.min_nuclei,
        scalebar_um=args.scalebar_um,
        scalebar_where=args.scalebar_where,
        cell_in=args.cell_in,
        source=args.source,
        native_csv=args.native_csv,
        dpi=args.dpi,
        formats=args.formats,
        pixel_size_um=args.pixel_size_um,
        lowres_um=args.lowres_um,
    )
    patients = args.patient or arms[0].patients()
    for pid in patients:
        process_patient(pid, arms, opt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

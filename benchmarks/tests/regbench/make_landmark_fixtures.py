"""Write the committed ANHIR and HyReCo fixtures (``fixtures/anhir/``, ``fixtures/hyreco/``).

Tiny SYNTHETIC images and landmarks in each dataset's real layout: no pixel and no
landmark row of either dataset is in them (ANHIR is CC BY-NC-SA). Each pair's moving
landmarks are its reference landmarks moved by a known translation (``SHIFT_PX``), and
the moving image is the reference image moved by the same translation, so a method that
recovers it scores (almost) zero. Where every format detail comes from is in each
folder's ``README.md``. Run from the repository root:

    python -m benchmarks.tests.regbench.make_landmark_fixtures

It rewrites both folders but their ``README.md``; the images and landmarks are
deterministic (fixed seeds).
"""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "fixtures"
ANHIR_DIR, HYRECO_DIR = HERE / "anhir", HERE / "hyreco"
SHIFT_PX = (4.0, -3.0)  # moving = reference + this, in pixels (x, y)
ANHIR_HEADER = [
    "",
    "Image diagonal [pixels]",
    "Image size [pixels]",
    "Source image",
    "Source landmarks",
    "Target image",
    "Target landmarks",
    "status",
    "Warped target landmarks",
    "Warped source landmarks",
    "Execution time [minutes]",
]
# (table index, set, scale folder, source stem, target stem, extension, status)
ANHIR_ROWS = (
    (0, "COAD_01", "scale-25pc", "S2", "S1", "jpg", "evaluation"),
    (4, "COAD_01", "scale-25pc", "S3", "S1", "jpg", "training"),
    (12, "COAD_02", "scale-25pc", "S5", "S1", "jpg", "training"),
    (
        370,
        "lung-lobes_1",
        "scale-100pc",
        "29-039-U-35W-Izd1-2-cd31",
        "29-039-U-35W-Izd1-1-HE",
        "png",
        "training",
    ),
)
ANHIR_HW = (60, 80)  # (H, W) of every synthetic image
HYRECO_PX_UM = 0.24
HYRECO_HW = (96, 128)
# (folder, id, how the TIFF states its pixel size)
HYRECO_SLIDES = (
    ("HyReCo", "29", "resolution"),
    ("HyReCo", "533", "mpp"),
    ("HyReCo-Additional", "8", "none"),
    ("HyReCo-Additional", "29", "resolution"),
)


def _clear(folder):
    """Remove everything of a fixture folder but its ``README.md`` (written by hand)."""
    folder.mkdir(parents=True, exist_ok=True)
    for child in folder.iterdir():
        if child.name == "README.md":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _texture(hw, seed, pad=8):
    """``(H + 2 pad, W + 2 pad, 3)`` uint8: smooth blobs on a light background (a stain
    look-alike), so a shifted crop of it is the moving image."""
    rng = np.random.default_rng(seed)
    h, w = hw[0] + 2 * pad, hw[1] + 2 * pad
    yy, xx = np.mgrid[0:h, 0:w]
    dark = np.zeros((h, w))
    for _ in range(18):
        cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(2.5, 6.0)
        dark += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    dark = np.clip(dark, 0, 1)
    tint = np.array([90.0, 40.0, 120.0])
    rgb = 235.0 - dark[..., None] * (235.0 - tint)
    return rgb.round().clip(0, 255).astype(np.uint8)


def _pair_images(hw, seed, pad=8):
    """``(reference, moving)``: two crops of one texture, the moving one taken
    ``SHIFT_PX`` further, so a reference point p is the moving point p + SHIFT_PX."""
    big = _texture(hw, seed, pad)
    dx, dy = (int(v) for v in SHIFT_PX)
    ref = big[pad : pad + hw[0], pad : pad + hw[1]]
    mov = big[pad - dy : pad - dy + hw[0], pad - dx : pad - dx + hw[1]]
    return np.ascontiguousarray(ref), np.ascontiguousarray(mov)


def _points(hw, seed, n=6, margin=6):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(margin, hw[1] - margin, n).round(1)
    ys = rng.uniform(margin, hw[0] - margin, n).round(1)
    return np.stack([xs, ys], axis=1)


def _write_anhir_landmarks(path, xy):
    """``,X,Y`` with an index column: the archive's own layout (README)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["", "X", "Y"])
        for i, (x, y) in enumerate(xy):
            w.writerow([i, f"{x:.1f}", f"{y:.1f}"])


def make_anhir():
    from PIL import Image

    _clear(ANHIR_DIR)
    rows = []
    written = set()
    for k, (index, set_name, scale, src, tgt, ext, status) in enumerate(ANHIR_ROWS):
        base = f"{set_name}/{scale}"
        ref, mov = _pair_images(ANHIR_HW, 100 + k)
        for stem, img in ((tgt, ref), (src, mov)):
            path = ANHIR_DIR / "images" / base / f"{stem}.{ext}"
            if path in written:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if ext == "jpg":
                Image.fromarray(img).save(path, quality=95)
            else:
                Image.fromarray(img).save(path)
            written.add(path)
        ref_xy = _points(ANHIR_HW, 200 + k)
        mov_xy = ref_xy + np.asarray(SHIFT_PX)
        if status == "training":
            _write_anhir_landmarks(
                ANHIR_DIR / "landmarks" / base / f"{tgt}.csv", ref_xy
            )
            _write_anhir_landmarks(
                ANHIR_DIR / "landmarks" / base / f"{src}.csv", mov_xy
            )
        else:  # an evaluation pair: only the source's landmarks are public
            _write_anhir_landmarks(
                ANHIR_DIR / "landmarks" / base / f"{src}.csv", mov_xy
            )
        h, w = ANHIR_HW
        rows.append(
            [
                index,
                f"{math.hypot(h, w):.1f}",
                f"({w}, {h})",
                f"{base}/{src}.{ext}",
                f"{base}/{src}.csv",
                f"{base}/{tgt}.{ext}",
                f"{base}/{tgt}.csv",
                status,
                "",
                "",
                "",
            ]
        )
    with open(ANHIR_DIR / "dataset_medium.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(ANHIR_HEADER)
        w.writerows(rows)


def _write_tiff(path, rgb, how):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"photometric": "rgb", "tile": (32, 32), "compression": "zlib"}
    if how == "resolution":
        per_cm = 10000.0 / HYRECO_PX_UM
        kw.update(resolution=(per_cm, per_cm), resolutionunit="CENTIMETER")
    elif how == "mpp":
        kw["description"] = f"Synthetic fixture|MPP = {HYRECO_PX_UM}"
    tifffile.imwrite(str(path), rgb, **kw)


def _write_hyreco_landmarks(path, xy_px):
    """``x,y,z`` in millimetres, no header (README)."""
    mm = xy_px * HYRECO_PX_UM / 1000.0
    with open(path, "w", newline="") as fh:
        for x, y in mm:
            fh.write(f"{x:.6f},{y:.6f},0\n")


def make_hyreco():
    _clear(HYRECO_DIR)
    for k, (folder, cid, how) in enumerate(HYRECO_SLIDES):
        ref, mov = _pair_images(HYRECO_HW, 300 + k)
        ref_xy = _points(HYRECO_HW, 400 + k, n=5)
        mov_xy = ref_xy + np.asarray(SHIFT_PX)
        for stain, img, xy in (("HE", ref, ref_xy), ("PHH3", mov, mov_xy)):
            tif = HYRECO_DIR / folder / stain / f"{cid}.tif"
            _write_tiff(tif, img, how)
            _write_hyreco_landmarks(tif.with_suffix(".csv"), xy)


def main():
    make_anhir()
    make_hyreco()
    for path in sorted(HERE.glob("anhir/**/*")) + sorted(HERE.glob("hyreco/**/*")):
        if path.is_file():
            print(path.relative_to(HERE))


if __name__ == "__main__":
    main()

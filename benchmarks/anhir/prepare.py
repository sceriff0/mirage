"""Turn the ANHIR download into pipeline input.

Three steps, each idempotent:

``join``     ANHIR ships the image archive split into 2 GB parts
             (``dataset_medium.z01 .. z05`` + ``dataset_medium.zip``). Python's
             zipfile cannot read a split archive, so this shells out to Info-ZIP's
             ``zip -s 0`` to rejoin it, then unzips into ``<data-root>/images/``.
``convert``  The images are brightfield JPEGs; the pipeline registers OME-TIFFs
             and estimates its transform on a nuclear channel. Each distinct image
             is written ONCE as a two-channel uint8 OME-TIFF: channel ``DAPI`` is
             the inverted luminance (nuclei bright, as in fluorescence -- what
             both backends anchor on) and channel ``<STEM>`` (``HE``, ``S1``,
             ``PAS``...) is the plain luminance. Two channels rather than one
             because the pipeline claims each channel name once per patient and
             a moving slide whose only channel is nuclear would have nothing
             left to keep. Per-case files are symlinks named
             ``<patient>_<stem>.ome.tif`` so no two rows share a basename.
``convert``  also writes ``samplesheet.csv`` -- a ``--start registration``
             checkpoint (``patient_id,preprocessed_image,is_reference,channels``)
             with one patient per case, the target as the reference -- and
             ``pairs_manifest.csv``, which ``warp.py`` and ``evaluate.py`` read.

The landmark archive is a separate ANHIR download with the same tree; unpack it
to ``<data-root>/landmarks/``. Landmark files are never rewritten -- the
pipeline registers at the archive's own scale, so the coordinates already match.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset as ds
from .warp import PAIRS_COLUMNS

REPO = Path(__file__).resolve().parents[2]
BIN_UTILS = REPO / "bin" / "utils"

NUCLEAR_CHANNEL = "DAPI"
SAMPLESHEET_COLUMNS = ["patient_id", "preprocessed_image", "is_reference", "channels"]
IMAGES_DIRNAME = "images"
LANDMARKS_DIRNAME = "landmarks"
DEFAULT_PIXEL_SIZE_UM = (
    1.0  # nominal: the JPEGs carry none, and rTRE is diagonal-relative
)

_UNSAFE = re.compile(r"[^A-Za-z0-9]+")


def patient_id_for(case: ds.Case) -> str:
    return f"anhir{case.case_id}"


def channel_for_stem(stem: str) -> str:
    """A channel name the samplesheet accepts: alphanumeric, upper-case, never nuclear."""
    name = _UNSAFE.sub("_", stem).strip("_").upper() or "STAIN"
    if name == NUCLEAR_CHANNEL:
        name = f"{name}_STAIN"
    return name


def channels_for(stem: str) -> str:
    return f"{NUCLEAR_CHANNEL}|{channel_for_stem(stem)}"


# ── join ──────────────────────────────────────────────────────────────────────
def join_archive(data_root, archive_name: str = "dataset_medium") -> Path:
    """Rejoin the split zip and extract it under ``<data_root>/images``."""
    data_root = Path(data_root)
    last = data_root / f"{archive_name}.zip"
    joined = data_root / f"{archive_name}_joined.zip"
    images = data_root / IMAGES_DIRNAME
    if not last.exists():
        raise FileNotFoundError(f"no archive at {last}")
    if not joined.exists():
        parts = sorted(data_root.glob(f"{archive_name}.z[0-9][0-9]"))
        if not parts:
            raise FileNotFoundError(
                f"no split parts {archive_name}.z01.. next to {last}"
            )
        subprocess.run(["zip", "-s", "0", str(last), "--out", str(joined)], check=True)
    images.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-n", str(joined), "-d", str(images)], check=True)
    return images


# ── convert ───────────────────────────────────────────────────────────────────
def read_luminance_uint8(jpeg_path) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = (
        None  # ANHIR's 100 % scale exceeds PIL's decompression-bomb cap
    )
    with Image.open(jpeg_path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def convert_image(jpeg_path, out_path, stem: str, pixel_size_um: float) -> Path:
    """Write the two-channel OME-TIFF described in the module docstring."""
    if str(BIN_UTILS) not in sys.path:
        sys.path.insert(0, str(BIN_UTILS))
    from ome_io import write_ome_tiff  # the pipeline's own writer

    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    lum = read_luminance_uint8(jpeg_path)
    data = np.stack([255 - lum, lum], axis=0)  # (C, Y, X)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_ome_tiff(
        out_path,
        data,
        channels=[NUCLEAR_CHANNEL, channel_for_stem(stem)],
        pixel_size_um=pixel_size_um,
    )
    return out_path


def _ome_name(rel_image: str) -> str:
    return str(Path(rel_image).with_suffix(".ome.tif"))


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(os.path.relpath(target, link.parent), link)


def build_inputs(
    cases,
    data_root,
    work_dir,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    convert: bool = True,
) -> tuple:
    """Convert every distinct image once, link per case, write the two CSVs.

    Returns ``(samplesheet_df, pairs_df)``. With ``convert=False`` nothing is
    written to disk but the CSVs are still built (a dry run of the layout).
    """
    data_root = Path(data_root)
    work_dir = Path(work_dir)
    images_root = data_root / IMAGES_DIRNAME
    landmarks_root = data_root / LANDMARKS_DIRNAME
    converted_root = work_dir / IMAGES_DIRNAME
    cases_root = work_dir / "cases"

    converted: dict = {}

    def ensure(rel_image: str, stem: str) -> Path:
        if rel_image not in converted:
            out = converted_root / _ome_name(rel_image)
            if convert:
                src = images_root / rel_image
                if not src.exists():
                    raise FileNotFoundError(
                        f"{src} is missing -- run `prepare.py join` first, or check --data-root"
                    )
                convert_image(src, out, stem, pixel_size_um)
            converted[rel_image] = out
        return converted[rel_image]

    sheet_rows, pair_rows = [], []
    for c in cases:
        pid = patient_id_for(c)
        ref_ome = ensure(c.target_image, c.target_stem)
        mov_ome = ensure(c.source_image, c.source_stem)
        ref_link = cases_root / pid / f"{pid}_{c.target_stem}.ome.tif"
        mov_link = cases_root / pid / f"{pid}_{c.source_stem}.ome.tif"
        if convert:
            _link(ref_ome, ref_link)
            _link(mov_ome, mov_link)
        ref_channels = channels_for(c.target_stem)
        mov_channels = channels_for(c.source_stem)
        # absolute but NOT resolved: resolving would follow the symlink back to the
        # shared converted image and hand the pipeline duplicate basenames again
        sheet_rows.append([pid, os.path.abspath(ref_link), "true", ref_channels])
        sheet_rows.append([pid, os.path.abspath(mov_link), "false", mov_channels])
        pair_rows.append(
            {
                "case_id": c.case_id,
                "patient_id": pid,
                "status": c.status,
                "tissue": c.tissue,
                "scale": c.scale,
                "source_image": c.source_image,
                "target_image": c.target_image,
                "source_landmarks": str(
                    ds.resolve_landmarks(c, landmarks_root, "source")
                ),
                "target_landmarks": str(
                    ds.resolve_landmarks(c, landmarks_root, "target")
                ),
                "diagonal": c.diagonal,
                "moving_stem": mov_link.name.split(".")[
                    0
                ],  # VALIS names slides by file stem
                "reference_stem": ref_link.name.split(".")[0],
                "moving_channels": mov_channels.replace(
                    "|", "_"
                ),  # STARE: channels joined by _
                "reference_channels": ref_channels.replace("|", "_"),
            }
        )
    sheet = pd.DataFrame(sheet_rows, columns=SAMPLESHEET_COLUMNS)
    pairs = pd.DataFrame(pair_rows, columns=PAIRS_COLUMNS)
    if convert:
        work_dir.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(work_dir / "samplesheet.csv", index=False)
        pairs.to_csv(work_dir / "pairs_manifest.csv", index=False)
    return sheet, pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Prepare the ANHIR download for the pipeline."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("join", help="rejoin the split image archive and unzip it")
    j.add_argument(
        "--data-root", required=True, help="folder holding dataset_medium.z01..zip"
    )
    j.add_argument("--archive", default="dataset_medium")

    c = sub.add_parser("convert", help="OME-TIFFs + samplesheet + pairs manifest")
    c.add_argument(
        "--data-root", required=True, help="folder with images/ and landmarks/"
    )
    c.add_argument(
        "--dataset", help="cover CSV (default <data-root>/dataset_medium.csv)"
    )
    c.add_argument("--work", required=True, help="where the OME-TIFFs and CSVs go")
    c.add_argument("--status", default="all", choices=("all",) + ds.STATUSES)
    c.add_argument("--tissue", action="append")
    c.add_argument(
        "--case", action="append", type=int, help="specific case ids; repeatable"
    )
    c.add_argument("--limit", type=int)
    c.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    c.add_argument(
        "--dry-run", action="store_true", help="print the plan, write nothing"
    )

    a = ap.parse_args(argv)
    if a.cmd == "join":
        images = join_archive(a.data_root, a.archive)
        print(f"images extracted under {images}")
        return 0

    dataset = a.dataset or str(Path(a.data_root) / "dataset_medium.csv")
    cases = ds.select_cases(
        ds.load_cases(dataset),
        status=a.status,
        tissues=a.tissue,
        case_ids=a.case,
        limit=a.limit,
    )
    if not cases:
        raise SystemExit("no cases selected")
    sheet, pairs = build_inputs(
        cases, a.data_root, a.work, pixel_size_um=a.pixel_size_um, convert=not a.dry_run
    )
    n_images = sheet["preprocessed_image"].nunique()
    verb = "would write" if a.dry_run else "wrote"
    print(f"{verb} {len(pairs)} cases ({n_images} slide rows) under {a.work}")
    if not a.dry_run:
        print(f"  samplesheet:    {Path(a.work) / 'samplesheet.csv'}")
        print(f"  pairs manifest: {Path(a.work) / 'pairs_manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The arms cohort as a ``prepare multiplex`` manifest: one row per moving round.

    python -m benchmarks.regbench.arms_manifest --root <arm results> --out rounds.csv

Reads two things the arm benchmark (``benchmarks/run_arms.sh``) leaves under its results
folder, and writes nothing there:

``<root>/preprocess_shared/csv/preprocessed.csv``
    the pipeline's preprocessing checkpoint (``lib/Checkpoint.groovy``: ``patient_id, id,
    preprocessed_image, is_reference, channels, pixel_size``; RFC 4180 quoting; ``channels``
    is ``|``-separated). One row per NATIVE slide, the one every arm registers.
``<root>/<from_arm>/<patient>/qc/registration/geojson/<slide>.geojson``
    the QC outlines that arm published (``conf/modules.config``, ``SEG_QC_GEOJSON``): each
    native slide segmented BEFORE registration, in full-resolution pixels of that slide.
    They are the CELL mask of the arm's segmenter (``subworkflows/local/seg_qc.nf:140``),
    which for this arm (``arms.yaml``: ``seg_method: stardist``) is the StarDist nuclei of
    channel 0, the nuclear marker, each grown by ``seg_expand_distance`` = 10 px
    (``bin/segment.py:223-225``): nuclei-derived outlines, larger than a nucleus, and not
    cells found on other channels. The benchmark calls them nuclei, as the pipeline's QC
    does; README, "What the cohort's nuclei are".
    ``<slide>`` is the file name less ``.ome.tif`` / ``.ome.tiff``, then less ``.tif`` /
    ``.tiff`` (``subworkflows/local/seg_qc.nf``, ``slide_of``).

A row is a (reference, moving round) pair of one patient. What cannot become a row is
dropped and listed in ``problems`` with its reason and the path at fault: a patient without
exactly one reference, a slide that is not on disk, a slide whose channel list does not name
exactly one nuclear channel, a round whose nuclei (or whose reference's nuclei) are missing.

**The nuclear channel** follows the pipeline's rule, a case-insensitive substring match of
``DAPI`` or ``CELLTOX`` (``lib/MarkerUtils.groovy``, ``nuclearIndex``), with one difference:
the pipeline resolves two matching channels by marker preference, this refuses the slide and
says why. Pass ``--nuclear`` with one marker to settle it.

**Pixel size** is the checkpoint's ``pixel_size`` column; where a row has none it is
``FALLBACK_PIXEL_SIZE_UM``, the cohort's 0.325 um, and ``pixel_size_source`` says which.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

PREPROCESS_ARM = "preprocess_shared"
FROM_ARM = "valis_high_micro2"
NUCLEAR_MARKERS = ("DAPI", "CELLTOX")  # nextflow.config: params.nuclear_markers
FALLBACK_PIXEL_SIZE_UM = 0.325  # benchmarks/configs/benchmark.config: pixel_size
CHANNEL_SEP = "|"
REQUIRED = ("patient_id", "preprocessed_image", "is_reference", "channels")
COLUMNS = (
    "case_id",
    "reference",
    "moving",
    "group",
    "ref_channel",
    "mov_channel",
    "channels_ref",
    "channels_mov",
    "pixel_size_um",
    "pixel_size_source",
    "ref_cells",
    "mov_cells",
)


def slide_name(path):
    """The pipeline's slide name of an image: ``P1.r2.ome.tif`` -> ``P1.r2``."""
    name = Path(str(path)).name
    return re.sub(r"\.tiff?$", "", re.sub(r"\.ome\.tiff?$", "", name))


def _markers(nuclear):
    raw = [nuclear] if isinstance(nuclear, str) else list(nuclear or [])
    out = [m.strip().upper() for part in raw for m in re.split(r"[,\s]+", str(part))]
    out = [m for m in out if m]
    if not out:
        raise ValueError("no nuclear marker given")
    return out


def nuclear_index(channels, nuclear=NUCLEAR_MARKERS):
    """Index of the one channel whose name contains a nuclear marker; ``ValueError`` with
    the reason when none or several do."""
    markers = _markers(nuclear)
    names = [str(c).strip().upper() for c in channels]
    hits = [i for i, n in enumerate(names) if any(m in n for m in markers)]
    shown = CHANNEL_SEP.join(str(c) for c in channels)
    if not hits:
        raise ValueError(f"no channel of [{shown}] contains {' or '.join(markers)}")
    if len(hits) > 1:
        which = ", ".join(repr(str(channels[i])) for i in hits)
        raise ValueError(
            f"{len(hits)} channels of [{shown}] contain {' or '.join(markers)} "
            f"({which}): the nuclear channel is ambiguous"
        )
    return hits[0]


def _truthy(s):
    return str(s).strip().lower() in ("true", "1", "yes", "y", "t")


def _published(path):
    """The slide a checkpoint row names, or its copy under ``converted/``, or None.

    Rows written before 2026-09-17 name ``<pid>/preprocessed/`` for a slide the pipeline
    published under ``<pid>/converted/`` (``benchmarks/reg_mosaic.py``, ``published_file``).
    """
    path = Path(path)
    if path.is_file():
        return path
    if path.parent.name == "preprocessed":
        converted = path.parent.parent / "converted" / path.name
        if converted.is_file():
            return converted
    return None


def _pixel_size(cell):
    try:
        px = float(cell)
    except (TypeError, ValueError):
        return None
    return px if math.isfinite(px) and px > 0 else None


def read_checkpoint(path):
    """``{patient: [slide rows]}`` of a ``preprocessed.csv``, in file order."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"{path}: not found. --root must be the arm results folder that holds "
            f"{PREPROCESS_ARM}/"
        )
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing column(s) {missing}")
        per = {}
        for row in reader:
            patient = (row["patient_id"] or "").strip()
            if not patient:
                continue
            image = (row["preprocessed_image"] or "").strip()
            per.setdefault(patient, []).append(
                {
                    "image": image,
                    "slide": slide_name(image),
                    "is_reference": _truthy(row["is_reference"]),
                    "channels": [
                        c.strip()
                        for c in (row["channels"] or "").split(CHANNEL_SEP)
                        if c.strip()
                    ],
                    "pixel_size": _pixel_size(row.get("pixel_size")),
                }
            )
    return per


def build(root, from_arm=FROM_ARM, nuclear=NUCLEAR_MARKERS):
    """``(rows, problems)``: the manifest rows, and what was dropped and why.

    A problem is ``{"patient", "slide", "reason", "path"}``; ``slide`` is empty when the
    whole patient is dropped.
    """
    root = Path(root)
    checkpoint = root / PREPROCESS_ARM / "csv" / "preprocessed.csv"
    nuclei_root = root / from_arm
    rows, problems = [], []

    def drop(patient, slide, reason, path):
        problems.append(
            {"patient": patient, "slide": slide, "reason": reason, "path": str(path)}
        )

    def resolve(patient, s, blame):
        """Image, nuclear index and nuclei of one slide, or None after noting why not."""
        image = _published(s["image"])
        if image is None:
            drop(patient, blame, f"slide {s['slide']} is not on disk", s["image"])
            return None
        try:
            channel = nuclear_index(s["channels"], nuclear)
        except ValueError as exc:
            drop(patient, blame, f"slide {s['slide']}: {exc}", s["image"])
            return None
        cells = nuclei_root / patient / "qc" / "registration" / "geojson"
        cells = cells / f"{s['slide']}.geojson"
        if not cells.is_file():
            drop(patient, blame, f"no QC nuclei for slide {s['slide']}", cells)
            return None
        return image, channel, cells

    for patient, slides in sorted(read_checkpoint(checkpoint).items()):
        refs = [s for s in slides if s["is_reference"]]
        if len(refs) != 1:
            n = "no" if not refs else str(len(refs))
            drop(patient, "", f"{n} reference row{'s' * (len(refs) != 1)}", checkpoint)
            continue
        ref = refs[0]
        moving = sorted(
            (s for s in slides if not s["is_reference"]), key=lambda s: s["slide"]
        )
        if not moving:
            drop(patient, "", "no moving round", checkpoint)
            continue
        seen = set()
        for mov in moving:
            if mov["slide"] in seen or mov["slide"] == ref["slide"]:
                drop(patient, mov["slide"], "slide name is not unique", mov["image"])
                continue
            seen.add(mov["slide"])
            # The reference is resolved once per round so that each lost round is listed.
            got_ref = resolve(patient, ref, mov["slide"])
            got_mov = got_ref and resolve(patient, mov, mov["slide"])
            if not got_mov:
                continue
            px_ref, px_mov = ref["pixel_size"], mov["pixel_size"]
            if px_ref and px_mov and not math.isclose(px_ref, px_mov, rel_tol=1e-6):
                drop(
                    patient,
                    mov["slide"],
                    f"pixel size {px_mov} differs from the reference's {px_ref}",
                    mov["image"],
                )
                continue
            px = px_ref or px_mov
            rows.append(
                {
                    "case_id": f"{patient}_{mov['slide']}",
                    "reference": str(got_ref[0]),
                    "moving": str(got_mov[0]),
                    "group": patient,
                    "ref_channel": got_ref[1],
                    "mov_channel": got_mov[1],
                    "channels_ref": CHANNEL_SEP.join(ref["channels"]),
                    "channels_mov": CHANNEL_SEP.join(mov["channels"]),
                    "pixel_size_um": px or FALLBACK_PIXEL_SIZE_UM,
                    "pixel_size_source": "preprocessed.csv" if px else "fallback",
                    "ref_cells": str(got_ref[2]),
                    "mov_cells": str(got_mov[2]),
                }
            )
    return rows, problems


def write(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.regbench.arms_manifest",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("--root", required=True, help="the arm results folder")
    ap.add_argument("--out", required=True, help="the manifest CSV to write")
    ap.add_argument(
        "--from-arm",
        default=FROM_ARM,
        help="the arm whose QC nuclei score every method",
    )
    ap.add_argument(
        "--nuclear",
        action="append",
        default=None,
        metavar="MARKER",
        help="a nuclear marker, matched as a substring of the channel name; "
        f"repeatable (default: {' '.join(NUCLEAR_MARKERS)})",
    )
    a = ap.parse_args(argv)
    rows, problems = build(a.root, a.from_arm, a.nuclear or NUCLEAR_MARKERS)
    for p in problems:
        who = "/".join(v for v in (p["patient"], p["slide"]) if v)
        print(f"DROPPED {who}: {p['reason']}: {p['path']}")
    fallback = sum(r["pixel_size_source"] == "fallback" for r in rows)
    if fallback:
        print(
            f"{fallback} round(s) have no pixel_size in the checkpoint: "
            f"{FALLBACK_PIXEL_SIZE_UM} um is used"
        )
    patients = len({r["group"] for r in rows})
    if not rows:
        print(f"no round is left ({len(problems)} dropped); nothing written")
        return 1
    write(rows, a.out)
    print(
        f"{len(rows)} round(s) of {patients} patient(s) -> {a.out} "
        f"({len(problems)} dropped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

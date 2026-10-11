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

**Bare nuclei that already exist** (columns ``ref_nuclei`` / ``mov_nuclei``, with
``*_nuclei_from``; empty by default). The arms publish only the grown outlines, so a
label image of a slide's nuclei exists only where something kept one:

``--nuclei-glob PATTERN`` (repeatable)
    masks of the QC segmentation kept in a Nextflow work directory. ``{patient}``,
    ``{slide}`` and ``{stem}`` are replaced per slide; ``{stem}`` is the image's file name
    less its last suffix, which is how ``bin/segment.py`` names its masks
    (``P1.r2.ome.tif`` -> ``P1.r2.ome_nuclei_mask.tif``). Of several matches the newest
    is taken and the count is recorded.
``--seg-arm ARM``
    the ``nuclei_mask`` a segmentation arm lists for the patient in
    ``<root>/<ARM>/csv/segmented.csv``, for the REFERENCE slide only: that mask is made
    on the registered image, whose reference round is LIKELY the native reference slide
    unchanged. That is not verified here; ``regbench nuclei`` / ``prepare multiplex``
    accept the mask only if it has the slide's shape and sits in its published outlines.

This module only FINDS the files (it reads no pixel); whether one may be used is decided
where it is read.

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
    "ref_nuclei",
    "mov_nuclei",
    "ref_nuclei_from",
    "mov_nuclei_from",
)
# the columns of the BARE nuclei (README, "Bare nuclei"): a label image of each native
# slide where one already exists, and how it was found. Empty where there is none.
NUCLEI_COLUMNS = COLUMNS[-4:]
SEG_CHECKPOINT = ("csv", "segmented.csv")  # lib/Checkpoint.groovy: `segmented`


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


def stem_of(path):
    """A slide's file name less its last suffix: what ``bin/segment.py`` prefixes its
    masks with (``Path(image).stem``)."""
    return Path(str(path)).stem


def find_by_glob(patterns, patient, image):
    """``(path, how)`` of the newest file a pattern matches for one slide, else
    ``("", "")``. The first pattern with a match wins."""
    import glob

    names = {"patient": patient, "slide": slide_name(image), "stem": stem_of(image)}
    for pattern in patterns or ():
        hits = [p for p in glob.glob(pattern.format(**names)) if Path(p).is_file()]
        if hits:
            newest = max(hits, key=lambda p: (Path(p).stat().st_mtime_ns, p))
            how = "glob" if len(hits) == 1 else f"glob (newest of {len(hits)})"
            return str(newest), how
    return "", ""


def seg_arm_masks(root, arm):
    """``{patient: nuclei mask}`` of a segmentation arm's ``csv/segmented.csv``: the
    reference row's ``nuclei_mask`` (any row's when none is marked), where it is on disk."""
    path = Path(root).joinpath(arm, *SEG_CHECKPOINT)
    if not path.is_file():
        raise SystemExit(
            f"--seg-arm {arm}: {path} not found (a segmentation arm's segmented.csv)"
        )
    per = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if "nuclei_mask" not in (reader.fieldnames or []):
            raise SystemExit(f"{path}: no nuclei_mask column")
        for row in reader:
            patient = (row.get("patient_id") or "").strip()
            mask = (row.get("nuclei_mask") or "").strip()
            if not patient or not mask or not Path(mask).is_file():
                continue
            if _truthy(row.get("is_reference")) or patient not in per:
                per[patient] = mask
    return per


def build(
    root, from_arm=FROM_ARM, nuclear=NUCLEAR_MARKERS, nuclei_globs=(), seg_arm=None
):
    """``(rows, problems)``: the manifest rows, and what was dropped and why.

    A problem is ``{"patient", "slide", "reason", "path"}``; ``slide`` is empty when the
    whole patient is dropped. ``nuclei_globs`` and ``seg_arm`` fill the bare-nuclei
    columns where a label image already exists (module docstring).
    """
    root = Path(root)
    from_seg = seg_arm_masks(root, seg_arm) if seg_arm else {}

    def nuclei_of(patient, image, is_reference):
        path, how = find_by_glob(nuclei_globs, patient, image)
        if not path and is_reference and patient in from_seg:
            path, how = from_seg[patient], f"seg_arm:{seg_arm}"
        return path, how

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
            ref_nuc, ref_how = nuclei_of(patient, got_ref[0], True)
            mov_nuc, mov_how = nuclei_of(patient, got_mov[0], False)
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
                    "ref_nuclei": ref_nuc,
                    "mov_nuclei": mov_nuc,
                    "ref_nuclei_from": ref_how,
                    "mov_nuclei_from": mov_how,
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
    ap.add_argument(
        "--nuclei-glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help="where label images of bare nuclei were kept, e.g. "
        "'<work>/*/*/{stem}_nuclei_mask.tif' ({patient}, {slide}, {stem}); repeatable",
    )
    ap.add_argument(
        "--seg-arm",
        default=None,
        metavar="ARM",
        help="a segmentation arm whose per-patient nuclei_mask (csv/segmented.csv) is "
        "offered for each patient's REFERENCE slide",
    )
    a = ap.parse_args(argv)
    rows, problems = build(
        a.root,
        a.from_arm,
        a.nuclear or NUCLEAR_MARKERS,
        a.nuclei_glob or (),
        a.seg_arm,
    )
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
    slides = {}
    for r in rows:
        slides[(r["group"], r["reference"])] = r["ref_nuclei"]
        slides[(r["group"], r["moving"])] = r["mov_nuclei"]
    have = sum(bool(v) for v in slides.values())
    print(
        f"bare nuclei: a label image was found for {have} of {len(slides)} slide(s); "
        "`python -m benchmarks.regbench nuclei --list` says which may be used, the "
        "rest need the StarDist re-run"
    )
    print(
        f"{len(rows)} round(s) of {patients} patient(s) -> {a.out} "
        f"({len(problems)} dropped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

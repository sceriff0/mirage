"""The cohort as a ``prepare multiplex`` manifest: one row per moving round.

    python -m benchmarks.regbench.manifest --preprocessed <preprocessed.csv> \
        --outlines <folder> --out rounds.csv

Reads one file, regbench's OWN input step (``slurm/inputs.sbatch``, submitted by
``launch.sbatch --inputs``: the pipeline's preprocessing, ``--start preprocessing --stop
preprocessing``), and writes nothing but ``--out``:

``--preprocessed``
    the pipeline's preprocessing checkpoint, ``<inputs>/preprocess/results/csv/
    preprocessed.csv`` (``lib/Checkpoint.groovy``: ``patient_id, id, preprocessed_image,
    is_reference, channels, pixel_size``; RFC 4180 quoting; ``channels`` is
    ``|``-separated). One row per NATIVE slide.
``--outlines``
    the folder the nuclei step fills (``regbench nuclei``, ``slurm/nuclei.sbatch``): the
    outlines of each slide are ``<folder>/<patient>/<slide>.geojson``, the trace of the
    cell mask of regbench's own StarDist run of that slide (``outlines.py``), defined to
    be what the arms' QC published (``subworkflows/local/seg_qc.nf``: StarDist nuclei of
    the nuclear channel, each grown by ``seg_expand_distance`` = 10 px, traced by
    ``bin/mask_to_geojson.py`` at ``simplify_tolerance``; README, "What the cohort's
    nuclei are"). The file is NOT there when the manifest is written: the round is a row
    all the same, and ``prepare multiplex`` drops a round whose outlines were not traced.
    ``<slide>`` is the file name less ``.ome.tif`` / ``.ome.tiff``, then less ``.tif`` /
    ``.tiff`` (``subworkflows/local/seg_qc.nf``, ``slide_of``).

A row is a (reference, moving round) pair of one patient. What cannot become a row is
dropped and listed in ``problems`` with its reason and the path at fault: a patient without
exactly one reference, a slide that is not on disk, a slide whose channel list does not name
exactly one nuclear channel.

**The nuclear channel** follows the pipeline's rule, a case-insensitive substring match of
``DAPI`` or ``CELLTOX`` (``lib/MarkerUtils.groovy``, ``nuclearIndex``), with one difference:
the pipeline resolves two matching channels by marker preference, this refuses the slide and
says why. Pass ``--nuclear`` with one marker to settle it.

**The bare-nuclei columns** (``ref_nuclei`` / ``mov_nuclei`` and ``*_nuclei_from``) stay in
the schema (``NUCLEI_COLUMNS``) and are written empty: the nuclei of every slide are the
StarDist run of the nuclei step.

**Pixel size** is the checkpoint's ``pixel_size`` column; where a row has none it is
``FALLBACK_PIXEL_SIZE_UM``, the cohort's 0.325 um (the input step pins it, as the arms
did), and ``pixel_size_source`` says which.

A ``rounds.csv`` written before 2026-10-16 from the arms' results
(``arms_manifest --root``) has the same columns, its ``*_cells`` naming the arm's
published GeoJSON; ``prepare multiplex`` reads it as before (README, "Bare nuclei").
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

NUCLEAR_MARKERS = ("DAPI", "CELLTOX")  # nextflow.config: params.nuclear_markers
FALLBACK_PIXEL_SIZE_UM = 0.325  # slurm/inputs.sbatch pins pixel_size, as the arms did
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
# the columns of the BARE nuclei (README, "Bare nuclei"): kept in the schema, written
# empty (the nuclei step's StarDist run is every slide's nuclei)
NUCLEI_COLUMNS = COLUMNS[-4:]


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
            f"{path}: not found. It is the checkpoint of regbench's input step "
            "(<inputs>/preprocess/results/csv/preprocessed.csv): run "
            "launch.sbatch --inputs first"
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


def outline_path(outlines_dir, patient, slide):
    """Where the nuclei step traces the outlines of one slide."""
    return Path(outlines_dir) / patient / f"{slide}.geojson"


def build(preprocessed_csv, outlines_dir, nuclear=NUCLEAR_MARKERS):
    """``(rows, problems)``: the manifest rows, and what was dropped and why.

    A problem is ``{"patient", "slide", "reason", "path"}``; ``slide`` is empty when the
    whole patient is dropped. ``*_cells`` name where the nuclei step traces each slide's
    outlines (:func:`outline_path`); they need not exist yet.
    """
    checkpoint = Path(preprocessed_csv)
    rows, problems = [], []

    def drop(patient, slide, reason, path):
        problems.append(
            {"patient": patient, "slide": slide, "reason": reason, "path": str(path)}
        )

    def resolve(patient, s, blame):
        """Image, nuclear index and outlines of one slide, or None after noting why not."""
        image = _published(s["image"])
        if image is None:
            drop(patient, blame, f"slide {s['slide']} is not on disk", s["image"])
            return None
        try:
            channel = nuclear_index(s["channels"], nuclear)
        except ValueError as exc:
            drop(patient, blame, f"slide {s['slide']}: {exc}", s["image"])
            return None
        return image, channel, outline_path(outlines_dir, patient, s["slide"])

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
                    **{c: "" for c in NUCLEI_COLUMNS},
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
        prog="python -m benchmarks.regbench.manifest",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument(
        "--preprocessed",
        required=True,
        help="the input step's checkpoint, <inputs>/preprocess/results/csv/"
        "preprocessed.csv",
    )
    ap.add_argument(
        "--outlines",
        required=True,
        help="the folder the nuclei step traces each slide's outlines into "
        "(<folder>/<patient>/<slide>.geojson)",
    )
    ap.add_argument("--out", required=True, help="the manifest CSV to write")
    ap.add_argument(
        "--nuclear",
        action="append",
        default=None,
        metavar="MARKER",
        help="a nuclear marker, matched as a substring of the channel name; "
        f"repeatable (default: {' '.join(NUCLEAR_MARKERS)})",
    )
    a = ap.parse_args(argv)
    rows, problems = build(a.preprocessed, a.outlines, a.nuclear or NUCLEAR_MARKERS)
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
        f"({len(problems)} dropped); their outlines are traced by the nuclei step "
        f"into {a.outlines}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

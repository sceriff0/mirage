"""ANHIR, an external landmark set: the challenge's 230 TRAINING pairs as benchmark cases.

ANHIR (Automatic Non-rigid Histological Image Registration, ISBI 2019;
https://anhir.grand-challenge.org) registers differently stained CONSECUTIVE sections:
481 pairs in 49 tissue sets, at about 1 to 9 um per pixel. Landmarks of both images are
public for the 230 pairs marked ``training`` in the cover table and only those are
prepared here: the 251 ``evaluation`` pairs have no public target landmarks and cannot be
scored outside the challenge's server. Nothing is exported for submission.

**It is not in the default submission** and none of its numbers enters the cohort's tables
(``regbench.challenge`` scores it). Licence: CC BY-NC-SA (``LICENCE``).

**Two datasets from one prepare.** ``prepare anhir --source DIR`` writes

``anhir``       all 230 training pairs, each marked ``extra["side"]`` = ``tune`` or
                ``report``;
``anhir_tune``  the tuning-side pairs again, alone: the only cases the preprocessing
                candidates of the ``stare_prep`` row are run on (``regbench.prep``).

**The tune / report split** (``anhir_split.json``, tracked, hashed) is by TISSUE SET, never
by pair: images are shared between the pairs of a set, and most sets also hold hidden
evaluation pairs. Nobody picked the sets: inside each tissue the sets are ordered by
``sha256("regbench-anhir-split-v1:" + set name)`` and the first ``ceil(n / 4)`` are the
tuning side; a tissue with a single set is wholly on the report side. ``prepare``
recomputes the split from the cover table and refuses when it differs from the frozen
file; the scorer refuses cases of another split. Headline numbers are the report side's;
the tuning side is shown beside them, labelled. BOTH sides are training pairs of a public
challenge: neither is blind.

**Roles** (the cover table's, BIRL ``birl/benchmark.py:135-149`` ``[PARTIAL: see
regbench/challenge.py]``): the ``Source image`` is the moving image, the ``Target image``
the reference. ``truth_mov`` / ``truth_ref`` are their landmarks, paired by row. Where
the two files differ in length the common prefix is used, as the official evaluation does
(``birl/utilities/evaluate.py:17-35``, ``compute_tre``); both counts are recorded. Each
pair is its own group: a set has several targets, so no per-patient reference exists.

**Landmark files** (Data page ``[FULL: the saved copy of
https://anhir.grand-challenge.org/Data/ fetched 2026-10-10]``): CSV with a header
``,X,Y``, an index column, pixel coordinates of the image at that scale, "the origin [0,
0] is the top left pixel of the image plane".

**Pixel size** (``PIXEL_UM``). The Data page's summary table gives one resolution per
tissue, of the ORIGINAL image; the medium-scale archive holds each set at the percentage
its folder names (``scale-25pc``). Micrometres per pixel here are that resolution divided
by the folder's fraction: derived arithmetic, labelled so (``pixel_size_source``). One
tissue is NOT established and is reported in pixels only (``pixel_size_um`` None, every
micrometre number NaN): ``mammary-gland``, for which the page states 2.294 um/pixel and,
in the same paragraph, that the images "were acquired with the same microscope and set of
acquisition parameters as the mice lung lobes" (1.274 um/pixel, same 10x objective). The
two statements cannot both hold and no other official source read settles it.

**What a method reads.** ``ref_image`` / ``mov_image`` are the ORIGINAL RGB images (JPEG,
or PNG for the lung lobes): what VALIS and DeeperHistReg are given. ``ref_nuclear`` /
``mov_nuclear`` are a one-channel proxy of each, written here once per image, for a
method that takes one channel (STARE): **inverted luminance, ``255 - (0.299 R + 0.587 G +
0.114 B)``, 8 bit, full resolution** (``regbench.prep.grey_plane``). It is benchmark-side
preprocessing, part of this benchmark's input, and recorded as such. The ``stare_prep``
row replaces it by the frozen candidate of ``regbench.prep``.

**The source folder** (``--source``) holds the cover table ``dataset_medium.csv``, the
unpacked image archive under ``images/`` and the unpacked landmark archive under
``landmarks/``, each with the table's relative paths (``--table``, ``--images``,
``--landmarks`` name them one by one). It is only read.

Checked on the copy on the laptop this was written on (cover table sha256
``2b78b484...9f34``, 160 landmark files, 57 of the 245 images unpacked): 230 training
pairs in 49 sets, every pair inside one set, landmark files as described. Not checked:
the images of the other sets (not unpacked there).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .. import ANHIR, ANHIR_TUNE, prep
from ..cases import Case, case_dir, write_case

TABLE = "dataset_medium.csv"
COL_SOURCE, COL_TARGET = "Source image", "Target image"
COL_SOURCE_LND, COL_TARGET_LND = "Source landmarks", "Target landmarks"
COL_STATUS, COL_DIAGONAL = "status", "Image diagonal [pixels]"
TRAINING = "training"
N_TRAINING = 230
SIDES = ("tune", "report")
SPLIT_PATH = Path(__file__).resolve().parent.parent / "anhir_split.json"
SPLIT_SALT = "regbench-anhir-split-v1"
SPLIT_RULE = (
    "inside each tissue, the sets with training pairs ordered by sha256(salt + ':' + "
    "set name); the first ceil(n / 4) are the tuning side; a tissue with one set is "
    "wholly on the report side"
)
DATA_PAGE = (
    "https://anhir.grand-challenge.org/Data/ [FULL: the saved copy fetched 2026-10-10]"
)
# tissue -> (um per pixel of the ORIGINAL image, established?), Data page summary table
PIXEL_UM = {
    "lung-lesion": (0.174, True),
    "lung-lobes": (1.274, True),
    "mammary-gland": (2.294, False),  # contradicted in the page's own text
    "mice-kidney": (0.227, True),
    "COAD": (0.468, True),
    "gastric": (0.2528, True),
    "breast": (0.2528, True),
    "kidney": (0.2528, True),
}
NOT_ESTABLISHED = (
    "not established: the Data page states 2.294 um/pixel for the mammary glands and, "
    "in the same paragraph, the acquisition parameters of the lung lobes (1.274); "
    "errors are reported in pixels only"
)
GREY = "inverted luminance, 255 - (0.299 R + 0.587 G + 0.114 B), uint8, full resolution"
LICENCE = "CC BY-NC-SA (ANHIR Data page, Data Usage Agreement)"
CITATION = (
    "Borovec et al. 2020, ANHIR: Automatic Non-Rigid Histological Image Registration "
    "Challenge, IEEE TMI 39(10), doi:10.1109/TMI.2020.2986331; Borovec, Munoz-Barrutia, "
    "Kybic 2018, ICIP, doi:10.1109/icip.2018.8451040; and the tissue providers listed "
    "on the challenge's Data page"
)


class Refused(ValueError):
    """A pair this module will not prepare, with the reason."""


# ── the cover table ───────────────────────────────────────────────────────────
def set_of(rel):
    return Path(str(rel)).parts[0]


def tissue_of(rel):
    """``COAD_01/scale-25pc/S1.jpg`` -> ``COAD`` (BIRL ``evaluate_submission.py:371-376``
    takes the text before the first underscore; the same for every set name here)."""
    return re.sub(r"_\d+$", "", set_of(rel))


def scale_of(rel):
    """The percentage a set's folder names: ``scale-25pc`` -> 25."""
    m = re.fullmatch(r"scale-(\d+)pc", Path(str(rel)).parts[1])
    if not m:
        raise Refused(f"{rel}: no scale-<N>pc folder")
    return int(m.group(1))


def read_table(path):
    """Every row of the cover table, in file order: ``{"index", "source", "target",
    "source_landmarks", "target_landmarks", "status", "table_diagonal"}``."""
    path = Path(path)
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        raise SystemExit(f"{path}: the cover table is not readable ({exc})")
    need = (COL_SOURCE, COL_TARGET, COL_SOURCE_LND, COL_TARGET_LND, COL_STATUS)
    if not rows or any(c not in rows[0] for c in need):
        raise SystemExit(f"{path}: not an ANHIR cover table (columns {need})")
    out = []
    for r in rows:
        try:
            diagonal = float(r.get(COL_DIAGONAL) or "nan")
        except ValueError:
            diagonal = math.nan
        out.append(
            {
                "index": int(r[""])
                if (r.get("") or "").strip().isdigit()
                else len(out),
                "source": r[COL_SOURCE],
                "target": r[COL_TARGET],
                "source_landmarks": r[COL_SOURCE_LND],
                "target_landmarks": r[COL_TARGET_LND],
                "status": r[COL_STATUS].strip(),
                "table_diagonal": diagonal,
            }
        )
    return out


def training(rows):
    """The training pairs, in table order. A pair across two sets is refused."""
    out = [r for r in rows if r["status"] == TRAINING]
    for r in out:
        if set_of(r["source"]) != set_of(r["target"]):
            raise SystemExit(
                f"pair {r['index']}: {r['source']} and {r['target']} are in two sets; "
                "the split is by set"
            )
    return out


def case_id(row):
    return f"{int(row['index']):03d}"


# ── the split ─────────────────────────────────────────────────────────────────
def _order(name):
    return hashlib.sha256(f"{SPLIT_SALT}:{name}".encode()).hexdigest()


def make_split(rows):
    """The split of the training pairs of ``rows`` by the rule, as the frozen record."""
    pairs = {}
    for r in training(rows):
        pairs[set_of(r["source"])] = pairs.get(set_of(r["source"]), 0) + 1
    by_tissue = {}
    for name in pairs:
        by_tissue.setdefault(tissue_of(name), []).append(name)
    tune = []
    for names in by_tissue.values():
        names = sorted(names, key=_order)
        tune += names[: 0 if len(names) == 1 else math.ceil(len(names) / 4)]
    tune = sorted(tune)
    report = sorted(set(pairs) - set(tune))
    record = {
        "version": 1,
        "rule": SPLIT_RULE,
        "salt": SPLIT_SALT,
        "tune": tune,
        "report": report,
        "pairs": {
            "tune": sum(pairs[s] for s in tune),
            "report": sum(pairs[s] for s in report),
        },
        "pairs_per_set": dict(sorted(pairs.items())),
    }
    record["sha256"] = split_hash(record)
    return record


def split_hash(record):
    body = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_split(path=None):
    """The frozen split, checked against its hash and its own rule (no set on both
    sides). SystemExit otherwise."""
    path = Path(path or SPLIT_PATH)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path}: the frozen ANHIR split is unreadable ({exc})")
    if record.get("sha256") != split_hash(record):
        raise SystemExit(f"{path}: its content does not match its sha256")
    both = set(record["tune"]) & set(record["report"])
    if both:
        raise SystemExit(f"{path}: {sorted(both)} are on both sides of the split")
    return record


def side_of(split, set_name):
    if set_name in split["tune"]:
        return "tune"
    if set_name in split["report"]:
        return "report"
    raise SystemExit(f"the set {set_name!r} is on neither side of the frozen split")


def check_split(rows, split):
    """Refuse a cover table whose split by the rule is not the frozen one."""
    now = make_split(rows)
    if now["sha256"] != split["sha256"]:
        raise SystemExit(
            f"the cover table gives the split {now['sha256'][:12]} "
            f"({now['pairs']['tune']} tuning and {now['pairs']['report']} report "
            f"pairs), the frozen one is {split['sha256'][:12]} "
            f"({split['pairs']['tune']} and {split['pairs']['report']}): this is "
            "another release of the table. A new split is a new benchmark: nothing "
            "was prepared."
        )


# ── pixel size ────────────────────────────────────────────────────────────────
def pixel_size(tissue, scale_pc):
    """``(um per pixel or None, source text, established)`` at a set's scale."""
    if tissue not in PIXEL_UM:
        return None, f"no resolution on the Data page for the tissue {tissue!r}", False
    full, ok = PIXEL_UM[tissue]
    if not ok:
        return None, NOT_ESTABLISHED, False
    return (
        full * 100.0 / float(scale_pc),
        f"derived: {full} um/pixel of the original image (Data page summary table) "
        f"divided by the folder's scale, {scale_pc} %",
        True,
    )


# ── landmarks and images ──────────────────────────────────────────────────────
def read_landmarks(path):
    """``(N, 2)`` x, y in pixels of one landmark file (header ``,X,Y``, an index
    column). A file of another shape, or with a value that is not a finite number, is
    refused with the line."""
    path = Path(path)
    try:
        with open(path, newline="") as fh:
            lines = list(csv.reader(fh))
    except OSError as exc:
        raise Refused(f"{path}: landmark file not readable ({exc})") from None
    lines = [ln for ln in lines if any(c.strip() for c in ln)]
    if not lines or [c.strip() for c in lines[0]][-2:] != ["X", "Y"]:
        raise Refused(f"{path}: expected the header `,X,Y`")
    out = []
    for n, line in enumerate(lines[1:], 2):
        try:
            x, y = float(line[-2]), float(line[-1])
        except (ValueError, IndexError):
            raise Refused(f"{path}:{n}: not `index,X,Y` ({line!r})") from None
        if len(line) != len(lines[0]) or not (math.isfinite(x) and math.isfinite(y)):
            raise Refused(f"{path}:{n}: not `index,X,Y` with finite numbers ({line!r})")
        out.append((x, y))
    if not out:
        raise Refused(f"{path}: no landmarks")
    return np.array(out, dtype=float).reshape(-1, 2)


def image_hw(path):
    """``(H, W)`` from the image's header."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(str(path)) as im:
            w, h = im.size
    except OSError as exc:
        raise Refused(f"{path}: image not readable ({exc})") from None
    return int(h), int(w)


def grey_path(cases_root, rel):
    rel = Path(str(rel))
    return Path(cases_root) / "_cache" / ANHIR / rel.parent / f"{rel.stem}.grey.ome.tif"


def write_grey(image, out, pixel_size_um, force=False):
    """The plain row's one-channel input of one image (``GREY``), once per image."""
    out = Path(out)
    if out.exists() and not force:
        return out
    prep.write(prep.array_reader(prep.read_rgb(image)), "grey", out, pixel_size_um)
    return out


def _outside(xy, hw):
    h, w = hw
    return int(
        np.sum(
            (xy[:, 0] < -0.5)
            | (xy[:, 1] < -0.5)
            | (xy[:, 0] > w - 0.5)
            | (xy[:, 1] > h - 0.5)
        )
    )


def prepare_one(cases_root, row, images, landmarks, split, force=False, complete=True):
    """One training pair as a case of ``anhir`` (and of ``anhir_tune`` when its set is
    on the tuning side). Returns the case directories written."""
    cid, set_name = case_id(row), set_of(row["source"])
    side = side_of(split, set_name)
    wanted = [ANHIR] + ([ANHIR_TUNE] if side == "tune" else [])
    dirs = [case_dir(cases_root, ds, cid) for ds in wanted]
    if all((d / "case.json").exists() for d in dirs) and not force:
        return dirs
    tissue, scale = tissue_of(row["source"]), scale_of(row["source"])
    if scale_of(row["target"]) != scale:
        raise Refused(f"pair {cid}: source and target are at different scales")
    mov_img, ref_img = Path(images) / row["source"], Path(images) / row["target"]
    mov_csv = Path(landmarks) / row["source_landmarks"]
    ref_csv = Path(landmarks) / row["target_landmarks"]
    for p in (mov_img, ref_img, mov_csv, ref_csv):
        if not p.is_file():
            raise Refused(f"{p}: missing")
    mov_all, ref_all = read_landmarks(mov_csv), read_landmarks(ref_csv)
    n = min(len(mov_all), len(ref_all))  # the official rule: the common prefix
    mov_xy, ref_xy = mov_all[:n], ref_all[:n]
    ref_hw, mov_hw = image_hw(ref_img), image_hw(mov_img)
    px, px_source, established = pixel_size(tissue, scale)
    ref_n = write_grey(ref_img, grey_path(cases_root, row["target"]), px, force)
    mov_n = write_grey(mov_img, grey_path(cases_root, row["source"]), px, force)
    diagonal = math.hypot(*ref_hw)
    table = row.get("table_diagonal", math.nan)
    extra = {
        "challenge": "ANHIR",
        "status": row["status"],
        "table_index": int(row["index"]),
        "side": side,
        "split_sha256": split["sha256"],
        "set": set_name,
        "tissue": tissue,
        "scale_pc": scale,
        "stain_ref": Path(row["target"]).stem,
        "stain_mov": Path(row["source"]).stem,
        "n_landmarks": int(n),
        "n_landmarks_file": {"ref": int(len(ref_all)), "mov": int(len(mov_all))},
        "landmark_pairing": "by row; the common prefix where the two files differ in "
        "length (the official evaluation's rule)",
        "landmarks_outside_image": {
            "ref": _outside(ref_xy, ref_hw),
            "mov": _outside(mov_xy, mov_hw),
        },
        "landmark_files": {"ref": str(ref_csv), "mov": str(mov_csv)},
        "pixel_size_source": px_source,
        "pixel_size_established": bool(established),
        "diagonal_of": "the target (reference) image as read, px",
        "table_diagonal": None if not math.isfinite(table) else float(table),
        "table_diagonal_agrees": bool(
            math.isfinite(table) and abs(table - diagonal) <= 1.0
        ),
        # STARE's one-channel input, read by methods/stare.py
        "settings": {"proxy": "grey (inverted luminance)", "reduce": 1},
        "grey": GREY,
        "subset_complete": bool(complete),
        "truth": "manual landmarks (ANHIR training pairs; not blind)",
        "licence": LICENCE,
        "citation": CITATION,
    }
    for ds, d in zip(wanted, dirs):
        case = Case(
            dataset=ds,
            case_id=cid,
            group=cid,  # each pair alone: a set has several targets
            ref_nuclear=str(ref_n),
            mov_nuclear=str(mov_n),
            ref_image=str(ref_img),
            mov_image=str(mov_img),
            modality="brightfield",
            diagonal=diagonal,
            pixel_size_um=px,
            ref_hw=list(ref_hw),
            mov_hw=list(mov_hw),
            extra=dict(extra),
        )
        write_case(cases_root, case, truth_mov=mov_xy, truth_ref=ref_xy)
    return dirs


def _paths(a):
    source = Path(a.source) if a.source else None

    def pick(given, default):
        if given:
            return Path(given)
        if source is None:
            raise SystemExit(
                "give --source (a folder holding dataset_medium.csv, images/ and "
                "landmarks/) or --table, --images and --landmarks"
            )
        return source / default

    return (
        pick(a.table, TABLE),
        pick(a.images, "images"),
        pick(a.landmarks, "landmarks"),
    )


def prepare(a):
    table, images, landmarks = _paths(a)
    rows = read_table(table)
    split = load_split()
    check_split(rows, split)
    pairs = training(rows)
    if getattr(a, "count", False):
        print(len(pairs))
        return 0
    for folder in (images, landmarks):
        if not folder.is_dir():
            raise SystemExit(f"{folder}: no such folder")
    todo = pairs if a.index is None else pairs[a.index : a.index + 1]
    there = [
        r
        for r in pairs
        if (images / r["source"]).is_file() and (images / r["target"]).is_file()
    ]
    complete = len(there) == len(pairs)
    if not complete and not a.allow_incomplete:
        raise SystemExit(
            f"{images}: the images of {len(pairs) - len(there)} of {len(pairs)} "
            "training pairs are missing (the archive is not fully unpacked?). "
            "--allow-incomplete prepares what is there and marks every case so."
        )
    done = bad = skipped = 0
    for row in todo:
        if row not in there:
            skipped += 1
            continue
        try:
            dirs = prepare_one(
                a.cases, row, images, landmarks, split, a.force, complete
            )
            done += 1
            print(f"{ANHIR}/{case_id(row)} -> {', '.join(str(d) for d in dirs)}")
        except Refused as exc:
            bad += 1
            print(f"{ANHIR}/{case_id(row)}: REFUSED {exc}")
    print(
        f"{ANHIR}: {done}/{len(todo)} pair(s) prepared under {a.cases}"
        + (f", {skipped} without images" if skipped else "")
        + f"; split {split['sha256'][:12]} ({split['pairs']['tune']} tuning, "
        f"{split['pairs']['report']} report pairs); the tuning side also as "
        f"{ANHIR_TUNE}"
    )
    return 1 if bad else 0

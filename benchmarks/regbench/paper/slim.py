"""``regbench slim``: keep the CSVs, free the space taken by images and work folders.

    python -m benchmarks.regbench slim --root ROOT --level images|csv [--collected DIR]
        [--apply]

``root`` is a regbench results folder (``out/``: ``<row>/<dataset>/<case>/run.json``), an
arms results root (``<arm>/csv/registered.csv`` or ``<arm>/<patient>/qc/registration/
*_seg_qc.json``) or a sweep root (``<run_id>/trace/trace.txt``), told apart by that layout
(``layout``, in that order). Anything else is refused.

**Levels.**

``images``  the written images: every image file (``IMAGE_SUFFIXES``: registered / warped
            OME-TIFFs, pyramids, stage images, a method's ``native/`` copies, QC PNGs),
            ``*.zarr`` folders, a method's ``work/`` folder (beside a ``run.json``) and the
            Nextflow ``work/`` folders of an arms or sweep root. KEPT: the maps
            (``warped.npz``), ``run.json``, stage and case records, traces, every CSV and
            JSON, and the two kinds of image a re-score reads (``KEPT_IMAGES``): a result's
            reduced plane ``tissue/moving_eval.tif`` (``tissue._variant_plane``: without it
            the tissue block would have to re-reduce the written image, which is gone) and
            the scorer's cached planes ``cache/tissue/<dataset>/<case>/*_eval.tif``
            (``tissue.reference_plane`` / ``initial_plane``). Re-scoring still works; the
            overlay zooms of those runs no longer can (their written images are gone).
``csv``     everything but ``*.csv`` / ``*.tsv``, ``run.json``, ``*_seg_qc.json``, the
            traces (``trace*.txt``, a ``trace/`` or ``.trace/`` folder), ``size_logs/`` and
            ``SLIMMED.txt``: the maps and records go too, so re-scoring and reuse are no
            longer possible (the launcher runs such a unit again: ``methods.record_gaps``
            finds no ``warped.npz``). REFUSED unless ``--collected`` names a collection
            (``paper/collect.py``, its ``README.txt``) or a ``tables/`` folder that is newer
            than every run of the root and holds more than 0 rows of it (``_check_collected``).

A regbench root's ``paper/`` folder (the paper tables and figures) is never touched.

**Safety.** The default is a DRY RUN: each path class, its file count and the bytes it
would free, and the total; only ``--apply`` deletes. A symlink is never followed and never
deleted (file or folder; ``os.walk(followlinks=False)``), so nothing outside ``root`` is
reached. Refused: ``/``, ``$HOME`` or a folder holding it, a root of no recognised layout,
and any root inside the repository checkout or holding it. ``--apply`` writes
``<root>/SLIMMED.txt`` (level, date, bytes freed, what can no longer be done), which
``score``, ``reuse``, ``tissue`` and the launcher read (``regbench.slimmed``).
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import shutil
from pathlib import Path
from typing import NamedTuple

from .. import DATASETS, slimmed

SLIMMED = slimmed.NAME
LEVELS = slimmed.LEVELS
REPO = Path(__file__).resolve().parents[3]
IMAGE_SUFFIXES = (".tif", ".tiff", ".btf", ".png", ".jpg", ".jpeg", ".jp2", ".svs", ".ndpi",
                  ".czi", ".vsi", ".qptiff", ".scn", ".mrxs")
KEPT_IMAGES = ("<row>/<dataset>/<case>/tissue/moving_eval.tif",
               "cache/tissue/<dataset>/<case>/*_eval.tif")
CSV_KEEP_SUFFIXES = (".csv", ".tsv")
COLLECTION_README = "README.txt"
COLLECTION_FIRST_LINE = "interim only"
NO_LONGER = {
    "images": "the overlay zooms of these runs (their written images are removed); "
    "re-scoring still works (warped.npz, run.json, tissue/moving_eval.tif, cache/tissue "
    "and the case records are kept)",
    "csv": "re-scoring (the maps warped.npz and the case records are removed), the launcher's "
    "reuse of these runs (every unit runs again), and the overlay zooms; only the CSVs, "
    "run.json, *_seg_qc.json, traces and size_logs/ are left",
}


class Removal(NamedTuple):
    path: str
    kind: str
    bytes: int
    is_dir: bool


# ── refusals and layout ──────────────────────────────────────────────────────
def _inside(a, b):
    """Is ``a`` equal to or inside ``b`` (both resolved)?"""
    return a == b or b in a.parents


def _refuse_root(root):
    raw = Path(root)
    if raw.is_symlink():
        raise SystemExit(f"slim: refusing {raw}: the root is a symlink (never followed)")
    root = raw.resolve()
    if root == Path(root.anchor):
        raise SystemExit(f"slim: refusing {root}: the filesystem root")
    home = Path(os.path.expanduser("~")).resolve()
    if root == home or root in home.parents:
        raise SystemExit(f"slim: refusing {root}: it is $HOME ({home}) or holds it")
    repo = REPO.resolve()
    if _inside(root, repo) or _inside(repo, root):
        raise SystemExit(
            f"slim: refusing {root}: it is inside the regbench repository checkout ({repo}) "
            "or holds it")
    if not root.is_dir():
        raise SystemExit(f"slim: refusing {root}: not a folder")
    return root


def _any(root, pattern):
    return next(iter(Path(root).glob(pattern)), None) is not None


def layout(root):
    """``"regbench"``, ``"arms"``, ``"sweep"`` or None."""
    root = Path(root)
    if any(_any(root, f"*/{ds}/*/run.json") for ds in DATASETS) or (
            root / "cache" / "scores").is_dir():
        return "regbench"
    if _any(root, "*/csv/registered.csv") or _any(root, "*/*/qc/registration/*_seg_qc.json"):
        return "arms"
    if _any(root, "*/trace/trace.txt"):
        return "sweep"
    return None


# ── classes ──────────────────────────────────────────────────────────────────
def _is_image(name):
    return name.lower().endswith(IMAGE_SUFFIXES)


def _kept_image(rel, kind):
    parts = rel.parts
    if kind != "regbench":
        return False
    if len(parts) >= 5 and parts[-2:] == ("tissue", "moving_eval.tif"):
        return True
    return len(parts) >= 4 and parts[:2] == ("cache", "tissue") and rel.name.endswith("_eval.tif")


def _image_kind(rel, kind):
    parts = set(rel.parts[:-1])
    if kind == "regbench":
        if "native" in parts:
            return "native images"
        if "stages" in parts:
            return "stage images"
        if parts & {"tissue"} or rel.parts[:2] == ("cache", "tissue"):
            return "reduced planes"
        return "other images"
    if "pyramid" in parts:
        return "pyramids"
    if "registered" in parts:
        return "registered images"
    return "other images"


def _csv_kept(rel):
    name = rel.name
    if rel.parts[-1] == SLIMMED and len(rel.parts) == 1:
        return True
    if name.lower().endswith(CSV_KEEP_SUFFIXES) or name == "run.json" or name.endswith(
            "_seg_qc.json"):
        return True
    if name.startswith("trace") and name.endswith(".txt"):
        return True
    return bool({"trace", ".trace", "size_logs"} & set(rel.parts[:-1]))


def _file_kind(rel, name):
    low = name.lower()
    if low.endswith((".npz", ".npy")):
        return "maps"
    if low.endswith((".rec", ".pkl", ".pickle", ".json")) or rel.parts[:2] == ("cache", "scores"):
        return "records"
    return "other files"


def _dir_kind(path, rel, kind):
    """The class of a folder removed whole, or None (descend into it)."""
    if path.name.endswith(".zarr"):
        return "zarr images"
    if path.name == "work":
        if kind == "regbench":
            return "method work" if (path.parent / "run.json").exists() else None
        return "Nextflow work"
    return None


def _size(path):
    total = 0
    for dirpath, dirnames, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                st = os.lstat(os.path.join(dirpath, f))
            except OSError:
                continue
            total += st.st_size
    return total


def _walk(root, kind, level):
    removals = []
    for dirpath, dirnames, files in os.walk(root, followlinks=False):
        here = Path(dirpath)
        rel_here = here.relative_to(root)
        if kind == "regbench" and rel_here.parts[:1] == ("paper",):
            dirnames[:] = []
            continue
        keep_dirs = []
        for d in sorted(dirnames):
            p = here / d
            if p.is_symlink():
                continue  # never followed, never deleted
            dk = _dir_kind(p, rel_here / d, kind)
            if dk is not None:
                removals.append(Removal(str(p), dk, _size(p), True))
            elif not (kind == "regbench" and not rel_here.parts and d == "paper"):
                keep_dirs.append(d)
        dirnames[:] = keep_dirs
        for f in sorted(files):
            p = here / f
            if p.is_symlink():
                continue
            rel = rel_here / f
            if not rel_here.parts and f == SLIMMED:
                continue
            if _is_image(f):
                if level == "images" and _kept_image(rel, kind):
                    continue
                k = _image_kind(rel, kind)
            elif level == "csv" and not _csv_kept(rel):
                k = _file_kind(rel, f)
            else:
                continue
            removals.append(Removal(str(p), k, os.lstat(p).st_size, False))
    return removals


# ── the csv rule ─────────────────────────────────────────────────────────────
def _readme_inputs(folder):
    """``{input: path}`` of a collection's README.txt ``Read:`` block, or None."""
    path = Path(folder) / COLLECTION_README
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    if not lines or not lines[0].startswith(COLLECTION_FIRST_LINE):
        return None
    out, inside = {}, False
    for ln in lines:
        if ln.strip() == "Read:":
            inside = True
            continue
        if inside:
            if not ln.startswith("  "):
                break
            key, _, value = ln.strip().partition(": ")
            out[key] = value
    return out


def is_collection(folder):
    return _readme_inputs(folder) is not None


def _csvs(folder):
    return sorted(p for p in Path(folder).glob("*.csv") if p.is_file())


def _rows_of_root(folder, root, kind):
    """The rows of ``folder`` that are of ``root``."""
    inputs = _readme_inputs(folder)
    if inputs is None:  # a tables/ folder: only the root's own
        own = {(root / "tables").resolve(), (root / "paper").resolve()}
        if kind != "regbench" or Path(folder).resolve() not in own:
            raise SystemExit(
                f"slim --level csv: refused: --collected {folder} is neither a collection "
                f"(`regbench collect`, its {COLLECTION_README}) nor this root's own tables/ "
                f"or paper/ folder ({root / 'tables'})")
        n = 0
        for p in _csvs(folder):
            with open(p, newline="") as fh:
                n += sum(1 for _ in csv.DictReader(fh))
        return n

    def same(value):
        try:
            return value not in (None, "-") and Path(value).resolve() == root
        except OSError:
            return False

    if kind == "regbench":
        mine = [f"regbench-interim:{v.strip()}" for v in (inputs.get("regbench") or "").split(",")
                if same(v.strip())]
        match = lambda s: s in mine  # noqa: E731
    else:
        if not same(inputs.get(kind)):
            return 0
        match = lambda s: s.startswith(f"{kind}:")  # noqa: E731
    n = 0
    for p in _csvs(folder):
        with open(p, newline="") as fh:
            n += sum(1 for r in csv.DictReader(fh) if match(r.get("source") or ""))
    return n


def _run_times(root, kind):
    pats = {"regbench": ["*/*/*/run.json"],
            "arms": ["*/*/qc/registration/*_seg_qc.json", "*/trace/trace.txt",
                     "*/csv/registered.csv"],
            "sweep": ["*/trace/trace.txt"]}[kind]
    return [p.stat().st_mtime for pat in pats for p in root.glob(pat) if p.is_file()]


def _check_collected(root, kind, collected):
    rule = (
        "slim --level csv removes the maps and records, so it is refused unless "
        "--collected names a collection (`regbench collect`) or a tables/ folder that is "
        "newer than every run of the root and holds more than 0 rows of it"
    )
    if collected is None:
        raise SystemExit(f"slim: refused: {rule} (no --collected given)")
    collected = Path(collected)
    files = _csvs(collected)
    if not files:
        raise SystemExit(f"slim: refused: {rule} ({collected} holds no CSV)")
    n = _rows_of_root(collected, root, kind)
    if n <= 0:
        raise SystemExit(f"slim: refused: {rule} ({collected} has 0 rows of {root})")
    made = min(p.stat().st_mtime for p in files)
    runs = _run_times(root, kind)
    if runs and not made > max(runs):
        raise SystemExit(
            f"slim: refused: {rule} ({collected} is not newer than every run it covers: "
            "collect again)")
    return n


# ── plan and run ─────────────────────────────────────────────────────────────
def plan(root, level, collected=None):
    """``[Removal]`` of ``level`` under ``root`` (nothing is deleted)."""
    if level not in LEVELS:
        raise SystemExit(f"slim: no level {level!r}; known: {', '.join(LEVELS)}")
    root = _refuse_root(root)
    kind = layout(root)
    if kind is None:
        raise SystemExit(
            f"slim: refusing {root}: no recognised layout (a regbench results folder, an "
            "arms root or a sweep root)")
    if level == "csv":
        _check_collected(root, kind, collected)
    return _walk(root, kind, level)


def _by_kind(removals):
    out = {}
    for r in removals:
        n, b = out.get(r.kind, (0, 0))
        out[r.kind] = (n + 1, b + r.bytes)
    return out


def _delete(r):
    p = Path(r.path)
    if p.is_symlink():
        return 0
    if r.is_dir:
        shutil.rmtree(p)  # never follows a symlink inside it
    else:
        p.unlink()
    return r.bytes


def _write_slimmed(root, level, freed, kind, collected):
    held = slimmed.level(root)
    lv = max(level, held or level, key=LEVELS.index)
    lines = [
        f"level: {lv}",
        f"date: {_dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"bytes freed: {freed}",
        f"layout: {kind}",
        f"collected: {collected if collected is not None else '-'}",
        f"no longer possible: {NO_LONGER[lv]}",
    ]
    if held and held != level:
        lines.append(f"earlier level: {held}")
    (root / SLIMMED).write_text("\n".join(lines) + "\n")


def run(root, level, apply=False, collected=None):
    """Plan ``level`` and, with ``apply``, delete; ``{"applied", "bytes", "kinds", ...}``."""
    removals = plan(root, level, collected)
    root = Path(root).resolve()
    kinds = _by_kind(removals)
    total = sum(r.bytes for r in removals)
    freed = 0
    if apply:
        for r in removals:
            freed += _delete(r)
        _write_slimmed(root, level, freed, layout(root) or "-", collected)
    return {"applied": bool(apply), "bytes": total, "freed": freed, "kinds": kinds,
            "removals": removals, "root": str(root), "level": level}


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n} B"


def main(a):
    if a.status:
        said = slimmed.said(a.root)
        print(said or "slimmed: no")
        return 0
    if not a.level:
        raise SystemExit("slim: --level images|csv is required (or --status)")
    got = run(Path(a.root), a.level, a.apply, Path(a.collected) if a.collected else None)
    print(f"# slim --level {a.level} {got['root']} ({layout(got['root'])})")
    for kind, (n, b) in sorted(got["kinds"].items()):
        print(f"{kind}\t{n} path(s)\t{b} bytes ({_human(b)})")
    print(f"total\t{len(got['removals'])} path(s)\t{got['bytes']} bytes ({_human(got['bytes'])})")
    if got["applied"]:
        print(f"applied: {got['freed']} bytes freed; {Path(got['root']) / SLIMMED} written")
    else:
        print("dry run: nothing deleted (--apply deletes)")
    return 0

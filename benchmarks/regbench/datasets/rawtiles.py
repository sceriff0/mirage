"""Raw-tile cyclic-IF datasets -- ``mcmicro``, ``rashid``, ``phenocycler``, ``codex`` --
prepared once per STITCHING FRONT-END.

    prepare rawtiles --dataset <name> --source <path> --front-end <fe|raw> --cases <dir>
                     [--patients ...] [--max-shift-um 15] [--export-only] [--image]

**Units.** A unit is what one patient is everywhere else: one tile grid imaged over
several cycles, the group a per-patient method registers. ``phenocycler``: a source tree
holding ``MICROSCOPE_OUTPUT/cycNNN_regRRR`` (one unit per region; the folder's name, with
``_regRRR`` when it holds more than one region). ``codex``: a sample folder holding
``Experiment.json`` and ``CycN_regR.zip`` (or the folders they hold), one unit per region
``<sample>_regR``. ``mcmicro``: a folder of ``*.ome.tif(f)``, one per cycle (in ``raw/``
or the folder itself). ``rashid``: every folder of ``*.rcpnl`` files under the source (one
per cycle). ``--source`` is one unit's folder, or a folder of them; ``--patients`` picks
units by name. Cycles are in natural order of their names; cycle 1 is the reference.

**Tiles** (``regbench.readers``): each cycle is exported ONCE to the tile series
(``regbench.tiles``) under ``<cases>/_cache/series/<dataset>/<unit>/<cycle>/``, which
every front-end then reads. ``mcmicro`` and ``rashid`` are read through ASHLAR's own
Bio-Formats reader, so their export runs in the ASHLAR image: ``--export-only`` there
first, then any front-end anywhere. A cycle's flat-/dark-field profiles are looked for
beside its file, as ``<stem>-ffp.tif`` / ``<stem>-dfp.tif``, there or in an
``illumination/`` folder beside it or one level up (MCMICRO's layout [MEMORY]: not
checked against a listing; a cycle without them is recorded as uncorrected).

**Nuclei** (``regbench.stitching``): segmented once per RAW tile (``tile_nuclei``,
cached per tile under ``<cases>/_cache/tiles/<dataset>/<unit>/<cycle>/``: a second
front-end does not segment again), one copy of each kept by ``owned`` -- decided ONCE per
cycle on the stage layout -- and carried into each front-end's mosaic by
``mosaic_nuclei(..., owned_polys=...)``. Every front-end therefore has the SAME nucleus
population (``stitching.csv``'s ``n_nuclei`` is the check); a front-end's misplacement
shows as position error. The case's outlines ARE these nuclei (no growth: growing a
polygon needs its neighbours' pixels).

**Case sets** ``<cases>/<dataset>@<front end>/``:

* ``stage`` | ``ashlar_edge`` | ``stare_mosaic``: every cycle stitched by that front-end
  (``frontends.stitch``, streamed) into ``_stitched/<unit>/<cycle>/mosaic.ome.tif`` (the
  nuclear channel: the tile series holds no other), one multiplex-style case per
  (cycle 1, cycle k): ``ref_nuclear`` / ``mov_nuclear`` the two mosaics, the mosaic
  nuclei as both polygon sets, ``extra.front_end``, ``extra.stitching`` (the front-end's
  report of each cycle, ``benchmark_side`` included). Every method runs on them unchanged.
  ``stitching.csv``: per unit and cycle, the stitching error between neighbour tiles
  (``seam_error``: disp p50/p90 in um, pairs, matches) and ``n_nuclei``.
* ``raw``: the tiles themselves (``extra.tiles``: the two cycles' tile-series folders),
  which ASHLAR reads natively (``methods/ashlar.py``, ``_register_raw``). The nuclei are
  the SAME owned set, placed in each cycle's STAGE layout at whole pixels (zero origin);
  ASHLAR's map moves a moving nucleus into its mosaic by its tile's (ASHLAR position -
  stage origin) and back by the reference tile drawn there, so the reference nuclei are
  compared where ASHLAR's cycle-1 positions put them (``extra.nuclei``). ``initial`` on
  ``@raw`` is the stage layout, as on ``@stage``.

**Images** (the in-image mechanism of ``slurm/prepare.sbatch``,
``REGBENCH_PREPARE_IMAGE``): ``ashlar_edge`` needs ASHLAR (``REGBENCH_SIF_ASHLAR``),
``stare_mosaic`` STARE >= 1.3.0 (``REGBENCH_SIF_STARE_REGBENCH``); ``--image`` prints the
one a command needs (``image_for``).

Memory (user ruling 2026-10-14): one tile at a time through the readers, the segmenter,
the front-end and its streaming writer; only polygons are held across tiles.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from .. import RAW, RAW_FRONT_ENDS, RAW_TILE_DATASETS, frontends, stitching, tiles
from ..cases import Case, case_dir, write_case, write_json

SERIES = "series"  # <cases>/_cache/series/<dataset>/<unit>/<cycle>/: the tile series
NUCLEI_CACHE = "tiles"  # <cases>/_cache/tiles/<dataset>/<unit>/<cycle>/: tile_nuclei's
STITCHED = "_stitched"  # <cases>/<dataset>@<fe>/_stitched/<unit>/<cycle>/
STITCHING_CSV = "stitching.csv"
UNIT_STITCHING = "stitching.json"
STITCHING_COLUMNS = (
    "front_end",
    "unit",
    "cycle",
    "n_tiles",
    "n_pairs",
    "n_matched",
    "n_candidates",
    "disp_um_p50",
    "disp_um_p90",
    "n_nuclei",
    "fallback_stage_positions",
)
BIOFORMATS = ("mcmicro", "rashid")
ASHLAR_IMAGE, STARE_IMAGE = "ASHLAR", "STARE_REGBENCH"
SEP = "__"  # case id = <unit>__<moving cycle>

NUCLEI_MOSAIC = (
    "segmented once per raw tile (stitching.tile_nuclei), one copy of each kept by the "
    "stage-layout ownership rule (stitching.owned: the same set for every front-end), "
    "each moved by the whole-pixel origin this front-end's mosaic pastes its tile at "
    "(stitching.mosaic_nuclei). The outlines are the nuclei themselves"
)
NUCLEI_RAW = (
    "the same owned set as on the mosaic case sets (stitching.owned), placed in each "
    "cycle's STAGE layout at whole pixels, zero origin (frontends.placed_origins of "
    "frontends.stage_positions). A method's map (ASHLAR's, methods/ashlar.py) moves a "
    "moving nucleus into its mosaic by its tile's (position - stage origin) and back by "
    "the reference tile drawn there: the comparison is made where ASHLAR's cycle-1 "
    "positions put the reference nuclei, with the population of every other front-end. "
    "The outlines are the nuclei themselves"
)


class Source(NamedTuple):
    cycle: str  # the cycle's folder name in the tile series
    export: Callable  # dest -> tiles.Cycle


def case_set_name(dataset, front_end):
    return f"{dataset}@{front_end}"


def image_for(dataset, front_end, export_only=False):
    """The image (``REGBENCH_SIF_<image>``) a ``prepare rawtiles`` command runs in."""
    if export_only:
        return ASHLAR_IMAGE if dataset in BIOFORMATS else STARE_IMAGE
    return ASHLAR_IMAGE if front_end == "ashlar_edge" else STARE_IMAGE


def _natural(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


# ── what the source holds, per dataset ───────────────────────────────────────
def _pheno_units(source):
    from ..readers import phenocycler

    roots = (
        [source]
        if (source / "MICROSCOPE_OUTPUT").is_dir() or (source / "experiment.json").is_file()
        else sorted(d for d in source.iterdir() if (d / "MICROSCOPE_OUTPUT").is_dir())
    )
    out = {}
    for root in roots:
        found, exp_path = phenocycler.cycle_sources(root)
        regions = sorted({reg for _, _, reg, _ in found})
        for reg in regions:
            unit = root.name if len(regions) == 1 else f"{root.name}_reg{reg:03d}"
            out[unit] = [
                Source(
                    cdir.name,
                    lambda dest, c=cdir, n=names, e=exp_path: phenocycler.export_cycle(
                        c, e, n, dest
                    ),
                )
                for cdir, cyc, r, names in sorted(found, key=lambda t: t[1])
                if r == reg
            ]
    return out


CODEX_MEMBER = re.compile(r"^Cyc(\d+)_reg(\d+)(\.zip)?$", re.IGNORECASE)


def _codex_units(source):
    from ..readers import codex_keyence

    samples = (
        [source]
        if (source / "Experiment.json").is_file()
        else sorted(d for d in source.iterdir() if (d / "Experiment.json").is_file())
    )
    out = {}
    for sample in samples:
        by_reg = {}
        for p in sorted(sample.iterdir()):
            m = CODEX_MEMBER.match(p.name)
            if m and (p.is_dir() or m[3]):
                by_reg.setdefault(int(m[2]), []).append((int(m[1]), p))
        exp = sample / "Experiment.json"
        for reg, members in sorted(by_reg.items()):
            out[f"{sample.name}_reg{reg}"] = [
                Source(
                    f"Cyc{cyc}_reg{reg}",
                    lambda dest, p=p, e=exp: codex_keyence.export(p, e, dest),
                )
                for cyc, p in sorted(members)
            ]
    return out


def _profiles(path):
    """``(ffp, dfp)`` of one Bio-Formats cycle file, each None when not found."""
    stem = re.sub(r"\.ome\.tiff?$|\.[^.]+$", "", path.name, flags=re.IGNORECASE)
    found = []
    for kind in ("ffp", "dfp"):
        hit = None
        for d in (path.parent, path.parent / "illumination", path.parent.parent / "illumination"):
            for ext in (".tif", ".tiff"):
                if (d / f"{stem}-{kind}{ext}").is_file():
                    hit = d / f"{stem}-{kind}{ext}"
                    break
            if hit:
                break
        found.append(hit)
    return tuple(found)


def _bioformats_sources(files, nuclear_channel):
    from ..readers import bioformats_tiles

    out = []
    for f in sorted(files, key=lambda p: _natural(p.name)):
        ffp, dfp = _profiles(f)
        name = re.sub(r"\.ome\.tiff?$|\.[^.]+$", "", f.name, flags=re.IGNORECASE)
        out.append(
            Source(
                name,
                lambda dest, f=f, ffp=ffp, dfp=dfp: bioformats_tiles.export(
                    f, dest, nuclear_channel, ffp=ffp, dfp=dfp
                ),
            )
        )
    return out


def _ome_files(d):
    return [p for p in d.iterdir() if p.is_file() and re.search(r"\.ome\.tiff?$", p.name, re.I)]


def _mcmicro_units(source, nuclear_channel):
    def files_of(d):
        return _ome_files(d / "raw") if (d / "raw").is_dir() else _ome_files(d)

    if files_of(source):
        return {source.name: _bioformats_sources(files_of(source), nuclear_channel)}
    return {
        d.name: _bioformats_sources(files_of(d), nuclear_channel)
        for d in sorted(source.iterdir())
        if d.is_dir() and files_of(d)
    }


def _rashid_units(source, nuclear_channel):
    by_dir = {}
    for p in sorted(source.rglob("*.rcpnl")):
        by_dir.setdefault(p.parent, []).append(p)
    out = {}
    for d, files in by_dir.items():
        rel = [q for q in d.relative_to(source).parts if q.upper() != "RCPNL"]
        out["_".join([source.name, *rel]) if rel else source.name] = _bioformats_sources(
            files, nuclear_channel
        )
    return out


def units(dataset, source, nuclear_channel=0):
    """``{unit: [Source, ...]}`` (cycles in order) of a dataset's source folder."""
    source = Path(source)
    if not source.is_dir():
        raise SystemExit(f"--source {source}: not a folder")
    if dataset == "phenocycler":
        found = _pheno_units(source)
    elif dataset == "codex":
        found = _codex_units(source)
    elif dataset == "mcmicro":
        found = _mcmicro_units(source, nuclear_channel)
    elif dataset == "rashid":
        found = _rashid_units(source, nuclear_channel)
    else:
        raise SystemExit(f"{dataset!r} is not a raw-tile dataset {RAW_TILE_DATASETS}")
    for unit in found:
        if SEP in unit or "|" in unit or "/" in unit:
            raise SystemExit(f"unit name {unit!r} holds {SEP!r}, '|' or '/'")
    return found


# ── one unit ──────────────────────────────────────────────────────────────────
def series_dir(cases_root, dataset, unit, cycle):
    return Path(cases_root) / "_cache" / SERIES / dataset / unit / cycle


def nuclei_cache(cases_root, dataset, unit, cycle):
    """``<cases>/_cache/tiles/<dataset>/<unit>/<cycle>/``: ``tile_nuclei``'s cache."""
    return Path(cases_root) / "_cache" / NUCLEI_CACHE / dataset / unit / cycle


def ensure_series(cases_root, dataset, unit, src: Source):
    """The cycle's tile series, exported once (a folder without ``cycle.json`` is an
    interrupted export and is written again)."""
    dest = series_dir(cases_root, dataset, unit, src.cycle)
    if (dest / tiles.CYCLE_JSON).is_file() and (dest / tiles.TILES_CSV).is_file():
        return tiles.read_cycle(dest)
    try:
        return src.export(dest)
    except ImportError as e:
        raise RuntimeError(
            f"{dataset}/{unit}/{src.cycle}: the tiles are not exported yet and their "
            f"reader cannot run here ({e}). Export them first in the {ASHLAR_IMAGE} "
            "image: `prepare rawtiles --export-only` (REGBENCH_PREPARE_IMAGE="
            f"{ASHLAR_IMAGE})"
        ) from None


def make_segmenter(cycle, work):
    """The nuclei's segmenter: StarDist when ``REGBENCH_STARDIST_MODEL`` is set, else the
    built-in one (``stitching.default_segmenter``)."""
    return stitching.default_segmenter(cycle, work)


def _extent(cycle, positions):
    o = frontends.placed_origins(cycle.tiles, positions)
    h = max(int(o[k, 1]) + r.h for k, r in enumerate(cycle.tiles))
    w = max(int(o[k, 0]) + r.w for k, r in enumerate(cycle.tiles))
    return [h, w]


def _nuclear_name(cycle):
    ch = list(cycle.channels)
    nuc = int(cycle.nuclear_channel)
    return [ch[nuc] if nuc < len(ch) else f"C{nuc}"]


def _placed(cases_root, dataset, unit, cycle, front_end, segmenter, set_dir, max_shift_um,
            out_tile):
    """One cycle under one front-end: ``(nuclei, report, image, hw, stitching row)``."""
    tile_polys = stitching.tile_nuclei(
        cycle, segmenter, cache_dir=nuclei_cache(cases_root, dataset, unit, cycle.dir.name)
    )
    own = stitching.owned(tile_polys, cycle.tiles)  # once per cycle, the stage layout's
    if front_end == RAW:
        positions = frontends.stage_positions(cycle)
        nuc = stitching.mosaic_nuclei(tile_polys, positions, cycle.tiles, owned_polys=own)
        return nuc, None, None, _extent(cycle, positions), None
    st = frontends.stitch(
        cycle,
        front_end,
        set_dir / STITCHED / unit / cycle.dir.name,
        max_shift_um=max_shift_um,
        out_tile=out_tile,
    )
    nuc = stitching.mosaic_nuclei(tile_polys, st.positions, cycle.tiles, owned_polys=own)
    seam = stitching.seam_error(
        tile_polys, st.positions, cycle.tiles, cycle.pixel_size_um
    )
    row = {
        "front_end": front_end,
        "unit": unit,
        "cycle": cycle.dir.name,
        "n_tiles": len(cycle.tiles),
        "n_pairs": seam["n_pairs"],
        "n_matched": seam["n_matched"],
        "n_candidates": seam["n_candidates"],
        "disp_um_p50": seam["disp_um_p50"],
        "disp_um_p90": seam["disp_um_p90"],
        "n_nuclei": len(nuc),
        "fallback_stage_positions": bool(st.report.get("fallback_stage_positions", False)),
    }
    report = {**st.report, "seam_error": {k: v for k, v in seam.items() if k != "per_edge"}}
    return nuc, report, str(st.mosaic), list(st.report["mosaic_shape"]), row


def _cell_sets(segmenter):
    tag = str(getattr(segmenter, "tag", "") or "")
    source = "stardist_rerun" if tag.startswith("stardist") else "builtin"
    side = {
        "outlines": {"source": source, "note": "the nuclei themselves (no growth)"},
        "nuclei": {"source": source, "exact": True, "segmenter": tag},
    }
    return {"ref": dict(side), "mov": dict(side)}


def prepare_unit(cases_root, dataset, unit, sources, front_end, *, force=False,
                 segmenter_factory=None, max_shift_um=frontends.DEFAULT_MAX_SHIFT_UM,
                 out_tile=frontends.OUT_TILE):
    """Prepare the cases of one unit under one front-end; returns the case folders.
    Writes the unit's stitching rows to ``_stitched/<unit>/stitching.json``."""
    if front_end not in RAW_FRONT_ENDS:
        raise ValueError(f"front end {front_end!r}: one of {RAW_FRONT_ENDS}")
    set_name = case_set_name(dataset, front_end)
    set_dir = Path(cases_root) / set_name
    if len(sources) < 2:
        raise ValueError(f"{dataset}/{unit}: {len(sources)} cycle(s); a case needs two")
    ids = [f"{unit}{SEP}{s.cycle}" for s in sources[1:]]
    dirs = [case_dir(cases_root, set_name, cid) for cid in ids]
    if not force and all((d / "case.json").is_file() for d in dirs):
        return dirs
    cycles = [ensure_series(cases_root, dataset, unit, s) for s in sources]
    px = float(cycles[0].pixel_size_um)
    for c in cycles[1:]:
        if not math.isclose(float(c.pixel_size_um), px, rel_tol=1e-6):
            raise ValueError(
                f"{dataset}/{unit}: cycle {c.dir.name} has {c.pixel_size_um} um/px, the "
                f"reference {px}: one unit is one microscope setting"
            )
    work = Path(cases_root) / "_cache" / "segment" / dataset / unit
    factory = segmenter_factory or make_segmenter
    segmenter = factory(cycles[0], work)
    args = (cases_root, dataset, unit)
    rest = (front_end, segmenter, set_dir, max_shift_um, out_tile)
    ref_nuc, ref_report, ref_img, ref_hw, ref_row = _placed(*args, cycles[0], *rest)
    rows = [ref_row] if ref_row else []
    ref_tiles = str(cycles[0].dir / tiles.TILES_CSV)
    out = []
    for cid, cyc in zip(ids, cycles[1:]):
        mov_nuc, mov_report, mov_img, mov_hw, mov_row = _placed(*args, cyc, *rest)
        if mov_row:
            rows.append(mov_row)
        mov_tiles = str(cyc.dir / tiles.TILES_CSV)
        extra = {
            "raw_dataset": dataset,
            "front_end": front_end,
            "unit": unit,
            "ref_cycle": cycles[0].dir.name,
            "mov_cycle": cyc.dir.name,
            "cycles": [c.dir.name for c in cycles],
            "reference": ref_img or ref_tiles,
            "moving": mov_img or mov_tiles,
            "ref_channel": 0,
            "mov_channel": 0,
            "channels_ref": _nuclear_name(cycles[0]),
            "channels_mov": _nuclear_name(cyc),
            "channel_names": _nuclear_name(cyc),
            "pixel_size_source": "tile series (cycle.json)",
            "tile_series": {"ref": str(cycles[0].dir), "mov": str(cyc.dir)},
            "nuclei": NUCLEI_RAW if front_end == RAW else NUCLEI_MOSAIC,
            "segmenter": str(getattr(segmenter, "tag", "")),
            "n_nuclei": {"ref": len(ref_nuc), "mov": len(mov_nuc)},
            "cells": "builtin",
            "cell_sets": _cell_sets(segmenter),
        }
        if front_end == RAW:
            extra["tiles"] = {"ref": str(cycles[0].dir), "mov": str(cyc.dir)}
            extra["stitching"] = {
                "by": "ASHLAR, inside the method's run, as published (methods/ashlar.py)",
                "benchmark_side": [],
            }
        else:
            extra["stitching"] = {"ref": ref_report, "mov": mov_report}
        case = Case(
            dataset=set_name,
            case_id=cid,
            group=unit,
            ref_nuclear=ref_img or ref_tiles,
            mov_nuclear=mov_img or mov_tiles,
            ref_image=ref_img or ref_tiles,
            mov_image=mov_img or mov_tiles,
            modality="fluorescence",
            diagonal=math.hypot(*ref_hw),
            pixel_size_um=px,
            ref_hw=list(ref_hw),
            mov_hw=list(mov_hw),
            extra=extra,
        )
        out.append(
            write_case(
                cases_root,
                case,
                cells_mov=mov_nuc,
                cells_ref=ref_nuc,
                nuclei_mov=mov_nuc,
                nuclei_ref=ref_nuc,
                # nucleus k is outline k: a nucleus inherits its outline's seam flag
                nuclei_ref_outline=np.arange(len(ref_nuc), dtype=np.int64),
                nuclei_mov_outline=np.arange(len(mov_nuc), dtype=np.int64),
            )
        )
        del mov_nuc
    if front_end != RAW:
        write_json(set_dir / STITCHED / unit / UNIT_STITCHING, rows)
    return out


def stitching_rows(cases_root, set_name):
    """Every unit's stitching rows of one case set (``[]`` for ``@raw``)."""
    rows = []
    for p in sorted((Path(cases_root) / set_name / STITCHED).glob(f"*/{UNIT_STITCHING}")):
        try:
            rows += json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    return rows


def write_stitching_csv(cases_root, set_name):
    rows = stitching_rows(cases_root, set_name)
    if not rows:
        return None
    path = Path(cases_root) / set_name / STITCHING_CSV
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(STITCHING_COLUMNS), restval="",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


# ── the command ───────────────────────────────────────────────────────────────
def prepare(a):
    dataset = a.raw_dataset
    if a.image:
        print(image_for(dataset, a.front_end, a.export_only))
        return 0
    if not a.source:
        raise SystemExit("prepare rawtiles: --source is required")
    if not a.export_only and not a.front_end:
        raise SystemExit("prepare rawtiles: --front-end is required (or --export-only)")
    found = units(dataset, a.source, getattr(a, "nuclear_channel", 0) or 0)
    if a.patients:
        unknown = sorted(set(a.patients) - set(found))
        if unknown:
            raise SystemExit(
                f"--patients {unknown}: not units of {a.source} ({sorted(found)})"
            )
        found = {u: s for u, s in found.items() if u in a.patients}
    names = sorted(found, key=_natural)
    if a.index is not None:
        names = names[a.index : a.index + 1]
    if not names:
        raise SystemExit(f"{dataset}: no unit under {a.source}")
    bad = 0
    for unit in names:
        try:
            if a.export_only:
                for src in found[unit]:
                    cyc = ensure_series(a.cases, dataset, unit, src)
                    print(f"{dataset}/{unit}/{src.cycle} -> {cyc.dir}")
                continue
            for d in prepare_unit(
                a.cases,
                dataset,
                unit,
                found[unit],
                a.front_end,
                force=a.force,
                max_shift_um=a.max_shift_um,
            ):
                print(f"{case_set_name(dataset, a.front_end)}/{Path(d).name} -> {d}")
        except Exception as exc:
            bad += 1
            print(f"{dataset}/{unit}: FAILED {type(exc).__name__}: {exc}")
    if not a.export_only and a.front_end != RAW:
        write_stitching_csv(a.cases, case_set_name(dataset, a.front_end))
    return 1 if bad == len(names) else 0

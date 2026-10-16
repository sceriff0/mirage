"""Your own multiplex-IF rounds as benchmark cases, from a manifest CSV.

One row is one (reference round, moving round) pair::

    case_id,reference,moving[,group,ref_channel,mov_channel,channels_ref,channels_mov,
                              pixel_size_um,ref_cells,mov_cells,ref_nuclei,mov_nuclei]

``reference`` / ``moving``   OME-TIFFs of the two rounds, BEFORE registration
``group``                    the patient; default: ``case_id``
``ref_channel``/``mov_channel``  index of the nuclear channel in each file (default 0)
``channels_ref``/``channels_mov``  each file's channel names, ``|``-separated; default: the
                             names in the file's OME header
``pixel_size_um``            default: the reference's OME header
``ref_cells`` / ``mov_cells``  optional nuclear segmentation of each NATIVE image, as a cell
                             GeoJSON or an integer label TIFF. Give the masks your pipeline
                             ships (e.g. mirage's SEG_QC output) to score its cells; leave
                             empty to use the built-in threshold + watershed segmenter.

``ref_nuclei`` / ``mov_nuclei``  kept in the schema and not read: the bare nuclei of a
                             slide are the nuclei step's StarDist run (``--nuclei-dir``)

For the cohort the manifest is written by ``benchmarks.regbench.manifest`` from
regbench's own preprocessing; its ``ref_cells`` / ``mov_cells`` name where the nuclei step
traces each slide's outlines. **A round whose outline GeoJSON is not there is DROPPED**
with the reason (``outlines not traced yet: run the nuclei step (launch.sbatch
--prepare)``), never read as an empty set. A manifest written from the arms' results
before 2026-10-16 names the arm's published GeoJSON and is read as before.

**Two polygon sets per slide** (``cases.SETS``). ``ref_cells`` / ``mov_cells`` are the
OUTLINES (on the cohort: each StarDist nucleus grown by 10 px). The bare NUCLEI of a
slide come from the first of these that applies, and ``case.json: extra.cell_sets`` records
which, per side:

1. ``--nuclei-dir``: the record ``python -m benchmarks.regbench nuclei`` left for that
   slide (its StarDist run);
2. ``--nuclei-fallback shrink``, only when asked: the outline with the 10 px rim taken off
   (``shrunk_outlines``, ``exact: false``; too small where cells touch);
3. nothing: the case is still prepared and is scored on its outlines; the reason is kept
   (``extra.cell_sets.<side>.nuclei.reason``) and the scorer lists it.

A case that is already prepared is kept (``--force`` prepares it again) only when it was
prepared from the slides and outlines the row names: one prepared from other inputs (a
manifest of the arms' results, then one of regbench's own) is refused with the reason,
never silently kept or replaced.

Without ``ref_cells`` both sets come from the built-in segmenter, as on the toy
cases: its objects are the nuclei, and the outlines are those labels grown by
``cells.EXPAND_PX``.

A case points a method at the ORIGINAL multichannel slides (``ref_image`` / ``mov_image``,
with ``extra["ref_channel"]``, ``["mov_channel"]``, ``["channels_ref"]``, ``["channels_mov"]``
and ``["channel_names"]``, the moving slide's names) and also at the nuclear channel alone
(``ref_nuclear`` / ``mov_nuclear``), for a method that must be given one channel.

A GeoJSON is read by the pipeline's own ``cell_pairs.from_feature_collection``, holes and
multi-part cells included, and must fit its slide: a file that is empty or has a vertex
outside ``[-1, W] x [-1, H]`` refuses that case with the file's path, and the other cases
are still prepared. ``--max-cells`` does not apply to GeoJSON nuclei.

There is no ground truth here, so there are no truth probes: these cases are scored by matched
Dice and centroid displacement of the nuclei only.

The nuclear channel of each slide is extracted once and its nuclei cached under
``<cases>/_cache/multiplex`` keyed by file and channel, so a reference shared by many rounds
is read once.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

from .. import cells, nuclei
from ..cases import Case, case_dir, polys, polyset_arrays, write_case, write_npz
from ..imageio import channel_names, extract_channel, pixel_size_um, shape_of
from ..manifest import slide_name

CHANNEL_SEP = "|"

REQUIRED = ("case_id", "reference", "moving")


def read_manifest(path):
    with open(path, newline="") as fh:
        rows = [
            {k.strip(): (v or "").strip() for k, v in r.items() if k}
            for r in csv.DictReader(fh)
        ]
    if not rows:
        raise SystemExit(f"{path}: no rows")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing}")
    ids = [r["case_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: case_id is not unique")
    base = Path(path).resolve().parent
    for r in rows:
        for k in (
            "reference",
            "moving",
            "ref_cells",
            "mov_cells",
            "ref_nuclei",
            "mov_nuclei",
        ):
            if r.get(k):
                p = Path(r[k]).expanduser()
                r[k] = str(p if p.is_absolute() else base / p)
    return rows


def _key(path, channel):
    st = Path(path).stat()
    raw = f"{Path(path).resolve()}|{channel}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _geojson_cells(cache, key, image, cells_file, force):
    """The nuclei of a GeoJSON as a ``PolySet``, checked against their slide; cached."""
    st = Path(cells_file).stat() if Path(cells_file).is_file() else None
    raw = f"{cells_file}|{st.st_size if st else 0}|{st.st_mtime_ns if st else 0}"
    tag = hashlib.sha1(raw.encode()).hexdigest()[:8]
    npz = cache / f"{key}.cells.{tag}.polyset.npz"
    if npz.exists() and not force:
        with np.load(npz) as z:
            return polys({k: z[k] for k in z.files}, "ref")
    ps = cells.cells_from_geojson(cells_file, hw=shape_of(image)[1:])
    # stored under the `ref` array names whichever side this slide is on in a case
    write_npz(npz, **polyset_arrays(ps, "ref"))
    return ps


NO_NUCLEI = (
    "no StarDist run of this slide "
    "(python -m benchmarks.regbench nuclei; README, 'Bare nuclei')"
)
NOT_TRACED = "outlines not traced yet: run the nuclei step (launch.sbatch --prepare)"


class Dropped(Exception):
    """A round that cannot be a case yet, with the reason; not a failure."""


SHRUNK = "shrunk_outlines"


def _slide(cache, image, channel, cells_file, px, a):
    """``(nuclear path, outlines, nuclei | None, how)`` for one slide, all cached.

    ``how`` is ``"arm_geojson"`` (a GeoJSON the manifest names: regbench's own trace or,
    in a manifest of the arms' results, the arm's), ``"label_image:<path>"`` or
    ``"builtin"``: where the
    outlines came from. Only the built-in segmenter gives nuclei here (its objects; the
    outlines are then those labels grown); for given outlines they are found by
    ``_nuclei``."""
    key = _key(image, channel)
    # A GeoJSON is read BEFORE the nuclear channel is extracted: a nuclei file that does
    # not fit refuses the case without copying a whole-slide plane first.
    given = (
        _geojson_cells(cache, key, image, cells_file, a.force)
        if cells_file and cells.is_geojson(cells_file)
        else None
    )
    nuclear = extract_channel(image, channel, cache / f"{key}.ome.tif", px)
    if given is not None:
        return nuclear, given, None, "arm_geojson"
    src = cells_file or "builtin"
    tag = hashlib.sha1(f"{src}|{a.diameter}".encode()).hexdigest()[:8]
    if cells_file:
        npz = cache / f"{key}.cells.{tag}.npz"
        if npz.exists() and not a.force:
            with np.load(npz) as z:
                ps = cells.Polys(z["xy"].astype(float), z["off"])
        else:
            ps = cells.load_cells(cells_file, a.diameter, a.tile, a.workers)
            write_npz(npz, xy=ps.xy.astype(np.float32), off=ps.off)
        return nuclear, ps, None, f"label_image:{cells_file}"
    npz = cache / f"{key}.sets.{tag}.npz"
    if npz.exists() and not a.force:
        with np.load(npz) as z:
            return (
                nuclear,
                cells.Polys(z["out_xy"].astype(float), z["out_off"]),
                cells.Polys(z["xy"].astype(float), z["off"]),
                "builtin",
            )
    nuc, out = cells.segment_slide_sets(nuclear, 0, a.diameter, a.tile, a.workers)
    write_npz(
        npz,
        xy=nuc.xy.astype(np.float32),
        off=nuc.off,
        out_xy=out.xy.astype(np.float32),
        out_off=out.off,
    )
    return nuclear, out, nuc, "builtin"


def _nuclei(cache, row, side, image, channel, outlines, a):
    """``(nuclei | None, what case.json records, parent | None)`` for a slide whose
    outlines were GIVEN. ``parent`` (the outline of each nucleus) exists only for the
    shrunk fallback, where it is known by construction."""
    group = row.get("group") or row["case_id"]
    slide = {
        "key": f"{group}/{slide_name(image)}",
        "group": group,
        "slide": slide_name(image),
        "image": image,
        "channel": channel,
        "outlines": row.get(f"{side}_cells") or "",
    }
    rec, store = None, None
    if getattr(a, "nuclei_dir", None):
        store = nuclei.store_of(a.nuclei_dir, slide)
        rec = nuclei.load(store)
        other = nuclei.other_image(rec, slide)
        if other:
            # a record of another file is nobody's nuclei: not used, and said
            rec = {
                "usable": False,
                "reason": f"the nuclei kept under {store} are not used: {other}. "
                "Run the nuclei step again (python -m benchmarks.regbench nuclei)",
            }
    if rec is not None and rec.get("usable"):
        info = {"source": rec["source"], "exact": bool(rec["exact"])}
        for k in ("agreement", "origin", "command", "outlines_from", "tolerance"):
            if rec.get(k):
                info[k] = rec[k]
        return nuclei.read_polys(store), info, None
    reason = (rec or {}).get("reason") or NO_NUCLEI
    if getattr(a, "nuclei_fallback", "") == "shrink":
        ps, parent = cells.shrink(outlines, cells.EXPAND_PX)
        if len(ps):
            return (
                ps,
                {
                    "source": SHRUNK,
                    "exact": False,
                    "shrunk_by_px": cells.EXPAND_PX,
                    "instead_of": reason,
                },
                parent,
            )
        reason = f"{reason}; and no outline survives shrinking by {cells.EXPAND_PX} px"
    return None, {"source": "", "reason": reason}, None


def _two_sets(cache, row, side, px, a):
    """One slide of a row: ``(nuclear path, outlines, nuclei | None, cell_sets entry,
    parent | None)``."""
    image = row["reference" if side == "ref" else "moving"]
    channel = int(row.get(f"{side}_channel") or 0)
    nuclear, outlines, nuc, how = _slide(
        cache, image, channel, row.get(f"{side}_cells"), px, a
    )
    if how == "builtin":
        info = {
            "outlines": {"source": "builtin", "expand_px": cells.EXPAND_PX},
            "nuclei": {"source": "builtin", "exact": True},
        }
        return nuclear, outlines, nuc, info, np.arange(len(nuc), dtype=np.int64)
    nuc, nuc_info, parent = _nuclei(cache, row, side, image, channel, outlines, a)
    info = {"outlines": {"source": how}, "nuclei": nuc_info}
    return nuclear, outlines, nuc, info, parent


def _names(row, key, image):
    """A slide's channel names: the manifest's, else the OME header's."""
    given = [c.strip() for c in (row.get(key) or "").split(CHANNEL_SEP) if c.strip()]
    return given or channel_names(image)


INPUT_KEYS = ("reference", "moving", "ref_cells", "mov_cells")


NOT_REPLACED = (
    "It is not replaced unasked (results made on it would no longer match it): "
    "prepare again with --force (REGBENCH_PREPARE_FORCE=1), or into another cases "
    "folder"
)


def input_fingerprints(row):
    """What each input of a row IS, not only where it is: the two slides'
    ``file_fingerprint``, and for each outlines file the fingerprint of the cell mask it
    was traced from (its ``.traced.json``; a re-trace of the same run keeps it), else of
    the file itself. A slide rewritten in place (``REGBENCH_INPUTS_FORCE=1`` re-runs the
    preprocessing into the same paths) is another input under the same name."""
    from ..cases import file_fingerprint

    out = {}
    for k in INPUT_KEYS:
        path = row.get(k) or ""
        if not path:
            out[k] = None
            continue
        made = nuclei.traced_record(path) if k.endswith("_cells") else None
        if made and made.get("cell_mask_fingerprint"):
            out[k] = {"cell_mask": made["cell_mask_fingerprint"]}
        else:
            out[k] = file_fingerprint(path) if Path(path).is_file() else None
    return out


def _prepared_from_other_inputs(d, row):
    """``""``, or why the case under ``d`` is not of this row: it was prepared from
    other slides or outlines, by path or, where the case recorded them, by content
    (``case.json: extra``, ``input_fingerprints``)."""
    import json

    try:
        extra = json.loads((d / "case.json").read_text()).get("extra") or {}
    except (OSError, ValueError):
        return ""
    for k in INPUT_KEYS:
        if k in extra and (extra[k] or "") != (row.get(k) or ""):
            return (
                f"the case under {d} was prepared from {k} {extra[k] or '(none)'} and "
                f"the manifest names {row.get(k) or '(none)'}. {NOT_REPLACED}"
            )
    had = extra.get("input_fingerprints")
    if had:
        now = input_fingerprints(row)
        for k in INPUT_KEYS:
            if k in had and had[k] != now.get(k):
                return (
                    f"the case under {d} was prepared from another file than "
                    f"{row.get(k) or '(none)'} ({k}) is now: it was rewritten in "
                    f"place since. {NOT_REPLACED}"
                )
    return ""


def _require_outlines(row):
    """A round is a case only with the outlines of both slides: not traced yet, or
    traced from another file than the slide that is there now, it is DROPPED."""
    from ..cases import file_fingerprint

    for side, image_col in (("ref", "reference"), ("mov", "moving")):
        path = row.get(f"{side}_cells") or ""
        if not path or not cells.is_geojson(path):
            continue
        if not Path(path).is_file():
            raise Dropped(f"{NOT_TRACED}: {path}")
        made = nuclei.traced_record(path)
        if made and made.get("image_fingerprint") != file_fingerprint(row[image_col]):
            raise Dropped(
                f"the outlines {path} were traced from another file than the slide "
                f"{row[image_col]} is now: run the nuclei step again "
                "(launch.sbatch --prepare)"
            )


def prepare_one(cases_root, row, a):
    d = case_dir(cases_root, "multiplex", row["case_id"])
    if (d / "case.json").exists() and not a.force:
        other = _prepared_from_other_inputs(d, row)
        if other:
            raise ValueError(other)
        return d
    _require_outlines(row)
    fingerprints = input_fingerprints(row)  # before anything is read
    if a.max_cells and cells.is_geojson(row.get("mov_cells") or ""):
        raise ValueError(
            f"{row['mov_cells']}: --max-cells does not apply to nuclei given as GeoJSON"
        )
    cache = Path(cases_root) / "_cache" / "multiplex"
    cache.mkdir(parents=True, exist_ok=True)
    if row.get("pixel_size_um"):
        px = float(row["pixel_size_um"])
        px_source = row.get("pixel_size_source") or "manifest"
    else:
        px, px_source = pixel_size_um(row["reference"]), "OME header"
    ref_c, mov_c = int(row.get("ref_channel") or 0), int(row.get("mov_channel") or 0)
    ref_n, ref_cells, ref_nuc, ref_sets, ref_parent = _two_sets(
        cache, row, "ref", px, a
    )
    mov_n, mov_cells, mov_nuc, mov_sets, mov_parent = _two_sets(
        cache, row, "mov", px, a
    )
    if a.max_cells and len(mov_cells) > a.max_cells:
        # Only the moving side is thinned: each sampled cell still finds its partner among
        # ALL reference cells, so the pairs are a sample of the full pairing.
        keep = np.sort(
            np.random.default_rng(0).choice(len(mov_cells), a.max_cells, replace=False)
        )
        mov_cells = mov_cells.take(keep)
        if mov_nuc is not None and mov_sets["nuclei"]["source"] == "builtin":
            # nucleus k is outline k: the same sample of both sets
            mov_nuc, mov_parent = mov_nuc.take(keep), np.arange(len(keep))
        elif mov_nuc is not None:
            mov_nuc, mov_parent = None, None
            mov_sets["nuclei"] = {
                "source": "",
                "reason": "--max-cells sampled the outlines; nuclei that are not "
                "the outlines' own objects cannot be sampled alike",
            }
    _, h, w = shape_of(ref_n)
    names_ref = _names(row, "channels_ref", row["reference"])
    names_mov = _names(row, "channels_mov", row["moving"])
    case = Case(
        dataset="multiplex",
        case_id=row["case_id"],
        group=row.get("group") or row["case_id"],
        ref_nuclear=str(ref_n),
        mov_nuclear=str(mov_n),
        # the slides as the cohort has them, every channel: what a method reads itself
        ref_image=row["reference"],
        mov_image=row["moving"],
        modality="fluorescence",
        diagonal=math.hypot(h, w),
        pixel_size_um=px,
        ref_hw=[h, w],
        mov_hw=list(shape_of(mov_n)[1:]),
        extra={
            "reference": row["reference"],
            "moving": row["moving"],
            "ref_channel": ref_c,
            "mov_channel": mov_c,
            "channels_ref": names_ref,
            "channels_mov": names_mov,
            # as `selfcheck.pair_case` names them: the channels of the slide that is warped
            "channel_names": names_mov,
            "pixel_size_source": px_source if px else "",
            "ref_cells": row.get("ref_cells") or "",
            "mov_cells": row.get("mov_cells") or "",
            "cells": "provided" if row.get("ref_cells") else "builtin",
            "diameter": a.diameter,
            "cell_sets": {"ref": ref_sets, "mov": mov_sets},
            "input_fingerprints": fingerprints,
        },
    )
    more = {}
    if ref_nuc is not None and ref_parent is not None:
        more["nuclei_ref_outline"] = ref_parent
    if mov_nuc is not None and mov_parent is not None:
        more["nuclei_mov_outline"] = mov_parent
    return write_case(
        cases_root,
        case,
        cells_mov=mov_cells,
        cells_ref=ref_cells,
        nuclei_mov=mov_nuc,
        nuclei_ref=ref_nuc,
        **more,
    )


def prepare(a):
    rows = read_manifest(a.manifest)
    if a.index is not None:
        rows = rows[a.index : a.index + 1]
    bad = 0
    for row in rows:
        try:
            d = prepare_one(a.cases, row, a)
            print(f"multiplex/{row['case_id']} -> {d}")
        except Dropped as exc:
            bad += 1
            print(f"multiplex/{row['case_id']}: DROPPED {exc}")
        except Exception as exc:
            bad += 1
            print(f"multiplex/{row['case_id']}: FAILED {type(exc).__name__}: {exc}")
    return 1 if rows and bad == len(rows) else 0

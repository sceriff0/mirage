"""Which channel each method registered on: the table ``registration_channels`` (T29, WD6).

SCORE SIDE ONLY: no method imports this module and no method's behaviour changes (the
method files are frozen). It reads what each round's ``run.json`` already records:

- the VALIS rows (``valis_mirage``, ``valis120``): ``info.nuclear_channel``, ``{slide name:
  {index, name}}``, the channel VALIS's default fluorescence processor read on each slide
  (a ``dapi`` regular expression on the channel names, channel 0 on no match or several:
  a CELLTOX round can fall to channel 0). A slide is matched to the case's reference or
  moving file by its name without extensions; one that matches neither is said in ``note``;
- STARE: ``info.nuclear_index``, the one ``--nuclear-index`` of both slides;
- ASHLAR: ``native.nuclear_channel``, its ``align_channel``, the same index in every cycle;
- PALOM and DeeperHistReg: ``native.nuclear_channel``, the moving slide's channel (the
  reference's is not recorded).

Against the case's nuclear channel (``case.extra``: ``ref_channel``, ``mov_channel``),
``registration_channel_is_nuclear`` is False when any recorded side differs, True when
every recorded side matches, and ``not recorded`` when no side is recorded (no run.json,
or none of these keys). One row per (run, case), from the first readable record of the
run's result names; ``initial`` registers nothing and has none.
"""

from __future__ import annotations

import json

from . import variants
from .cases import result_dir

NOT_RECORDED = "not recorded"
COLUMNS = (
    "dataset",
    "patient_id",
    "case_id",
    "run",
    "method",
    "nuclear_channel_ref",
    "nuclear_channel_mov",
    "nuclear_name_ref",
    "nuclear_name_mov",
    "registered_on_ref",
    "registered_on_mov",
    "registered_name_ref",
    "registered_name_mov",
    "recorded_from",
    "registration_channel_is_nuclear",
    "note",
)
_SUFFIXES = (".ome.tiff", ".ome.tif", ".ome.zarr", ".tiff", ".tif", ".zarr", ".svs",
             ".ndpi", ".czi", ".qptiff", ".vsi", ".mrxs", ".scn")  # fmt: skip


def _stem(path):
    """A slide's name as VALIS gives it: the file name without its extension(s)."""
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    low = name.lower()
    for suffix in _SUFFIXES:
        if low.endswith(suffix):
            return name[: -len(suffix)]
    return name.rsplit(".", 1)[0] if "." in name else name


def _name(case, side, index):
    names = (case.extra or {}).get(f"channels_{side}") or []
    if isinstance(index, int) and 0 <= index < len(names):
        return str(names[index])
    return NOT_RECORDED


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def registered_on(rec, case):
    """``(ref index, mov index, ref name, mov name, recorded_from, note)`` of one run.json;
    an index nothing recorded is None."""
    info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
    native = rec.get("native") if isinstance(rec.get("native"), dict) else {}
    used = info.get("nuclear_channel")
    if isinstance(used, dict) and used:
        side_of = {_stem(case.ref_image): "ref", _stem(case.mov_image): "mov"}
        got = {"ref": (None, None), "mov": (None, None)}
        unmatched = []
        for slide, entry in used.items():
            side = side_of.get(str(slide))
            if side is None or not isinstance(entry, dict):
                unmatched.append(str(slide))
                continue
            got[side] = (_int(entry.get("index")), entry.get("name"))
        note = f"slide(s) matched to neither file: {', '.join(unmatched)}" if unmatched else ""
        return (got["ref"][0], got["mov"][0], got["ref"][1], got["mov"][1],
                "run.json info.nuclear_channel", note)
    if _int(info.get("nuclear_index")) is not None:
        k = _int(info["nuclear_index"])
        return k, k, None, None, "run.json info.nuclear_index (both slides)", ""
    k = _int(native.get("nuclear_channel"))
    if k is not None:
        if rec.get("method") == "ashlar":
            return k, k, None, None, "run.json native.nuclear_channel (every cycle)", ""
        return None, k, None, None, "run.json native.nuclear_channel (the moving slide)", ""
    return None, None, None, None, "", ""


def _run_of(name):
    try:
        return variants.run_name(variants.run_of(name))
    except (KeyError, SystemExit, ValueError):
        return name


def _record(out, names, case):
    for name in names:
        path = result_dir(out, name, case) / "run.json"
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict):
            return rec
    return None


def rows(out, case, names):
    """One row per run among the result names ``names`` of one case (``COLUMNS``)."""
    extra = case.extra or {}
    want = {"ref": _int(extra.get("ref_channel", 0)), "mov": _int(extra.get("mov_channel", 0))}
    runs = {}
    for name in names:
        if name in ("initial",):
            continue
        runs.setdefault(_run_of(name), []).append(name)
    out_rows = []
    for run, mine in runs.items():
        rec = _record(out, mine, case)
        if rec is None:
            got = (None, None, None, None, "", "no readable run.json for this case")
        else:
            got = registered_on(rec, case)
        ref, mov, ref_name, mov_name, source, note = got
        checked = [(ref, want["ref"]), (mov, want["mov"])]
        checked = [(a, b) for a, b in checked if a is not None]
        flag = all(a == b for a, b in checked) if checked else NOT_RECORDED
        out_rows.append({
            "dataset": case.dataset,
            "patient_id": case.group,
            "case_id": case.case_id,
            "run": run,
            "method": (rec or {}).get("method") or NOT_RECORDED,
            "nuclear_channel_ref": want["ref"],
            "nuclear_channel_mov": want["mov"],
            "nuclear_name_ref": _name(case, "ref", want["ref"]),
            "nuclear_name_mov": _name(case, "mov", want["mov"]),
            "registered_on_ref": NOT_RECORDED if ref is None else ref,
            "registered_on_mov": NOT_RECORDED if mov is None else mov,
            "registered_name_ref": ref_name or (
                NOT_RECORDED if ref is None else _name(case, "ref", ref)),
            "registered_name_mov": mov_name or (
                NOT_RECORDED if mov is None else _name(case, "mov", mov)),
            "recorded_from": source or NOT_RECORDED,
            "registration_channel_is_nuclear": flag,
            "note": note,
        })
    return out_rows

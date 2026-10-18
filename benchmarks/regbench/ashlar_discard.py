"""ASHLAR's replaced tiles per round and cell: the table ``ashlar_discard``.

SCORE SIDE ONLY. ``methods/ashlar.py`` is frozen and does not import this module; it
writes ``discard_*`` counts into each round's ``run.json: info``, and this module reads them.
One row per (case, ASHLAR result name) whose result folder holds a ``run.json``; a name that
never ran has no row. A round whose record lacks a key has None for it, never 0.

``rigid_offset_um`` is the size of the case's rigid offset, set beside the cell's
``max_shift_um`` (15, 60, 240): |M0(c) - c| x pixel size, c the reference centre taken as a
moving point. M0 is STARE's anchor, read by the adapter's own ``read_anchor`` (the ``stare``
result of the round, or the folder the row's ``affine_from`` / ``m0_from`` option names), so
the table and the method cannot disagree about where it came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import variants
from .cases import result_dir

FAMILIES = ("ashlar", "affine_ashlar")


def rigid_offset_um(m0, ref_hw, pixel_size_um):
    """``|M0(c) - c|`` x pixel size, c = (W/2, H/2) of the reference as a moving point.
    ``m0`` is 2 x 3 or 3 x 3 (moving px -> reference px); None without a pixel size."""
    if pixel_size_um is None or m0 is None or ref_hw is None:
        return None
    m = np.asarray(m0, dtype=float)
    h, w = (float(v) for v in ref_hw)
    c = np.array([w / 2.0, h / 2.0])
    moved = m[:2, :2] @ c + m[:2, 2]
    return float(np.linalg.norm(moved - c) * float(pixel_size_um))


def _anchor(opts, case, out):
    from .methods import ashlar

    root = Path(opts["out"]) if opts.get("out") else Path(out)
    m0, where = ashlar.read_anchor(opts, case, root)
    return (None, None) if m0 is None else (m0, where)


def rows(out, cases):
    from .score import panel_of

    names = [n for f in FAMILIES for n in variants.members(f)]
    found = []
    for case in cases:
        for name in names:
            path = result_dir(out, name, case) / "run.json"
            if not path.is_file():
                continue
            try:
                run = json.loads(path.read_text())
            except ValueError:
                run = {}
            run = run if isinstance(run, dict) else {}
            info, opts = run.get("info") or {}, run.get("opts") or {}
            m0, where = _anchor(opts, case, out)
            found.append(
                {
                    "dataset": case.dataset,
                    "patient_id": case.group,
                    "panel": panel_of(case),
                    "case_id": case.case_id,
                    "row": name,
                    "affine": opts.get("affine"),
                    "noise_frac": opts.get("noise_frac"),
                    "max_shift_um": opts.get("max_shift_um"),
                    "n_tiles": info.get("n_tiles"),
                    "discard_frac": info.get("discard_frac"),
                    "discard_infinite_error": info.get("discard_infinite_error"),
                    "discard_zero_offset": info.get("discard_zero_offset"),
                    "discard_over_budget": info.get("discard_over_budget"),
                    "cycle_fully_modelled": info.get("cycle_fully_modelled"),
                    "rigid_offset_um": rigid_offset_um(
                        m0, case.ref_hw, case.pixel_size_um
                    ),
                    "rigid_offset_from": where,
                    "ok": run.get("ok"),
                }
            )
    return found

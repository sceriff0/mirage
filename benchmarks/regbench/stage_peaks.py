"""STARE's peak memory per stage, per round: the table ``stare_stage_peaks``.

SCORE SIDE ONLY. ``methods/stare.py`` does not import this module; it reads files the run
already wrote. A round of STARE is four kinds of job (COARSE, one REG_TILE per tile, SOLVE,
STITCH). Each job's record (``stages.record_path``) keeps every attempt it made
(``stages.attempts_of``, the last one last); the peak counted is the one of the attempt the
job ENDED OK with, because a failed attempt is a smaller or larger job that the round did not
use. A job whose last attempt did not end ok has no peak (None). The tiles are summarised by
their median and their maximum; n_tiles is how many of them ended ok. STITCH is not read from
a record: its peak is ``run.json: native.stitch_job.peak_rss_mb`` (the attempt that wrote the
image, written by the run itself).

The read box is STARE's: ``stare.vector_grid.read_box`` reads a tile's core plus ``3 * stride``
each side [PARTIAL: read vector_grid.read_box, lines 87-99 of the installed STARE 1.3.0 source;
rest of the module not read]. For an interior tile that is ``tile + 2 * 3 * stride`` on a side;
``read_box_mpx`` is that squared. ``pipeline_window_px`` is the window ``conf/modules.config``
sizes a tile job for: ``tile + 2 * max(halo, 3 * stride)``. ``tile`` and ``halo`` are the run's
tier (``methods.stare.TIERS``) with the changes the run recorded
(``info.implementation.changed_from_tier``).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import stages, variants
from .cases import result_dir

MB_PER_GB = 1024.0


def _peak_gb(rec):
    """The peak (GB) of the attempt this job ended ok with; None when it did not."""
    if not rec:
        return None
    last = stages.attempts_of(rec)[-1]
    peak = last.get("peak_rss_mb")
    if last.get("ok") is True and isinstance(peak, (int, float)):
        return peak / MB_PER_GB
    return None


def _tile_rows(stage_dir):
    jobs = Path(stage_dir) / "jobs"
    prefix = "tile_"  # stages.record_path(stage_dir, "tile", row)
    rows = []
    for path in sorted(jobs.glob(f"{prefix}*.json")) if jobs.is_dir() else []:
        tail = path.stem[len(prefix) :]
        if tail.isdigit():
            rows.append(int(tail))
    return rows


def _sizes(info):
    """``(tier, stride, tile, halo)`` of a run; None for what it does not say."""
    impl = info.get("implementation") or {}
    tier, stride = impl.get("tier"), info.get("stride")
    tile = halo = None
    if tier:
        from .methods.stare import TIERS

        row = TIERS.get(tier) or {}
        changed = impl.get("changed_from_tier") or {}
        tile = int(changed.get("tile", row.get("tile"))) if "tile" in row else None
        halo = int(changed.get("halo", row.get("halo"))) if "halo" in row else None
    return tier, (int(stride) if stride is not None else None), tile, halo


def round_row(run_json, stage_dir, case):
    """The ``stare_stage_peaks`` columns (every one but ``source``) of one round."""
    from .score import panel_of

    info = (run_json or {}).get("info") or {}
    tier, stride, tile, halo = _sizes(info)
    peaks = [
        p
        for i in _tile_rows(stage_dir)
        if (p := _peak_gb(stages.read_record(stage_dir, "tile", i))) is not None
    ]
    stitch = ((run_json or {}).get("native") or {}).get("stitch_job") or {}
    stitch_mb = stitch.get("peak_rss_mb")
    box = None if tile is None or stride is None else tile + 2 * 3 * stride
    return {
        "dataset": case.dataset,
        "patient_id": case.group,
        "panel": panel_of(case),
        "case_id": case.case_id,
        "row": None,
        "tier": tier,
        "stride": stride,
        "coarse_peak_gb": _peak_gb(stages.read_record(stage_dir, "coarse")),
        "tile_peak_gb_p50": statistics.median(peaks) if peaks else None,
        "tile_peak_gb_max": max(peaks) if peaks else None,
        "n_tiles": len(peaks),
        "solve_peak_gb": _peak_gb(stages.read_record(stage_dir, "solve")),
        "stitch_peak_gb": (
            stitch_mb / MB_PER_GB if isinstance(stitch_mb, (int, float)) else None
        ),
        "read_box_px": box,
        "read_box_mpx": None if box is None else box**2 / 1e6,
        "pipeline_window_px": (
            None
            if tile is None or stride is None or halo is None
            else tile + 2 * max(halo, 3 * stride)
        ),
    }


def rows(out, cases, names):
    """One row per round of every STARE result name present under ``out``. A result is a
    round when it kept stage records, or has a ``run.json`` and is not the rigid sibling
    (``<name>_rigid``, written by the same registration, which keeps none)."""
    found = []
    for case in cases:
        for name in names:
            if variants.family(name) != "stare":
                continue
            d = result_dir(out, name, case)
            has_run = (d / "run.json").is_file()
            if not (d / "stages").is_dir() and not (
                has_run and not name.endswith("_rigid")
            ):
                continue
            try:
                run = json.loads((d / "run.json").read_text()) if has_run else {}
            except ValueError:
                run = {}
            found.append({**round_row(run, d / "stages", case), "row": name})
    return found

"""Cost from the cohort's files to a merged image, step by step (secondary).

The headline cost of a method is the time it takes to ESTIMATE its transform
(``resources.csv``, ``summary_resources.csv``). This module adds a second view,
``cost_end_to_end.csv``: one row per patient and result that wrote an image, with the
steps apart.

==================  =================================================================
column              what it is, and where it was measured
==================  =================================================================
``method_convert``  a conversion the METHOD needs before it can read the slides (the
                    single-page TIFFs DeeperHistReg is given, the tiles ASHLAR is
                    given). Measured here, clock paused. ``none`` when the method reads
                    the cohort's files as they are; ``not separable`` when its own
                    conversion happens inside its registration call.
``estimate``        the registration clock (``register_s``). Measured here.
``write``           the method's own writer. Measured here, clock paused. Where an
                    adapter times the parts of its write step apart
                    (``native.write_parts_s``), this is the package's writer alone: a
                    conversion that writer needed goes to ``method_convert``, and the
                    benchmark's own copying (``benchmark_copy``: the split of ASHLAR's
                    mosaic per round, the stacking of DeeperHistReg's per-channel
                    files) is shown and is in NO total, because it is not the method's
                    work and its output is described as the package left it.
``pipeline_convert``  the PIPELINE's conversion of the patient's slides to the files
                    every method reads. IMPORTED from a Nextflow trace of an earlier
                    pipeline run; nothing is run for it.
``pipeline_merge``  the PIPELINE's merge-and-pyramid step, IMPORTED the same way, and
                    added only where the method's own written image is not already a
                    pyramidal image that holds every channel (``not needed`` otherwise).
==================  =================================================================

Methods are NOT given a common reader or writer (ruling R3.13): each row is what that
package does, and ``written`` says what it leaves on disk (channels, pyramid or not,
compression, size), so that unlike outputs are not read as like ones.

**The imported numbers** come from tasks that ran with OTHER resources than this
benchmark's 4 CPUs, so each carries its source and its allocation
(``pipeline_source``, ``pipeline_convert_alloc``, ``pipeline_merge_alloc``). They are
matched to a patient by the task's ``tag``, which is the patient in both processes
(``modules/local/convert_image.nf``, ``merge_and_pyramid.nf``). A patient the trace
does not have gets no ``total_s``. Without a trace the pipeline columns say ``not
available`` and everything measured here is still reported (``measured_here_s``).

**The trace** is the file Nextflow writes (``trace.txt``, tab-separated). Its column
ORDER is not fixed (``nextflow.config`` names the fields, a site can change them), so it
is read by header name, never by position.

``CLOCK`` says, per method, what the registration clock includes today. It is the table
in the README; ``test_cost.py`` holds the two together.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

NOT_AVAILABLE = "not available"
NOT_SEPARABLE = "not separable"
NOT_NEEDED = "not needed"
NONE = "none"

# The pipeline's processes, by the last part of the trace's `process` column (a
# qualified name such as MIRAGE:PREPROCESS:CONVERT_IMAGE ends in it).
PIPELINE_CONVERT = ("CONVERT_IMAGE",)
PIPELINE_MERGE = ("MERGE_AND_PYRAMID",)
# A cached task's row repeats the figures of the run that produced it, so it is used,
# and counted, so that the reader can see it was not measured in the named run.
USED_STATUS = ("COMPLETED", "CACHED")

# What each adapter's clock includes, as built (benchmarks/regbench/methods/<name>.py).
# `convert`: how the method's own conversion is reported (a run.json info key, NONE, or
# NOT_SEPARABLE). `per`: whether that number is one per round or one per patient launch.
#
# `writes` is what each package's writer leaves, from its source. It is a description
# for the reader; the table itself never trusts it: whether a merge-and-pyramid step is
# added is decided from the file each run wrote (`native.written`, read by tifffile).
#   STARE     stare-registration 1.2.1, `stages/stitch.py:185-232` `[PARTIAL: those
#             lines]`: `ome_tiff_writer(..., bigtiff=True, ome=True)`, tiled, no
#             compression argument, no SubIFDs.
#   VALIS     PyPI sdist valis-wsi 1.0.0, `registration.py:921`, `:4454`,
#             `slide_io.py:2826`, `:2962` (`compression="lzw"`, `pyramid=True,
#             subifd=True, bigtiff=True`); 1.2.0, `registration.py:117`, `:1004-1011`,
#             `:5107-5111`, `slide_io.py:47`, `:3537`, `:3688`
#             (`DEFAULT_COMPRESSION = ...DEFLATE`, `pyramid=True`) `[PARTIAL: those
#             lines of each]`.
#   ASHLAR    labsyspharm/ashlar v1.20.0, `scripts/ashlar.py:165-172`, `:292` (an output
#             name ending in .ome.tif selects `PyramidWriter`), `reg.py:1290-1303`
#             (`bigtiff=True`, `subifds`, `compression="adobe_deflate"`) `[PARTIAL:
#             those lines]`.
#   DeeperHistReg  MWod/DeeperHistReg main @ 42e7c9d, `dhr_input_output/dhr_savers/
#             tiff_saver.py:24-29` (`'compression': 'lzw'`, `'pyramid': True`)
#             `[PARTIAL: those lines]`; one file per channel is the adapter's call.
#   PALOM     as `methods/palom.py` records it from labsyspharm/palom v2026.4.1
#             `pyramid.py` (not re-read for this table) `[MEMORY]`.
CLOCK = {
    "stare": {
        "reading": "inside the clock: STARE's own reader on the cohort's OME-TIFFs",
        "convert": NONE,
        "estimate": "`stare register` up to SOLVE's manifest (COARSE, REG-TILE, SOLVE)",
        "write": "STITCH, deferred until the clock has stopped, timed on its own",
        "writes": "one OME-TIFF per round, every channel, BigTIFF, uncompressed, no "
        "pyramid",
    },
    "valis_mirage": {
        "reading": "inside the clock: VALIS's readers, its conversion of every slide "
        "to its working images, and the JVM start",
        "convert": NOT_SEPARABLE,
        "estimate": "the pipeline's `valis_registration` up to the writer (rigid, "
        "non-rigid, micro)",
        "write": "`Slide.warp_and_save_slide`, clock and peak paused, timed per slide",
        "writes": "one pyramidal OME-TIFF per round, every channel (libvips, LZW)",
    },
    "valis": {
        "reading": "inside the clock: VALIS's readers, its conversion of every slide "
        "to its working images, and the JVM start",
        "convert": NOT_SEPARABLE,
        "estimate": "`Valis.register()` and `register_micro()`",
        "write": "`warp_and_save_slides`, clock and peak paused; one call writes the "
        "patient, a round's share is the call over the slides written",
        "writes": "one pyramidal OME-TIFF per round, every channel (libvips, deflate)",
    },
    "deeperhistreg": {
        "reading": "the package's own loading is inside the clock (its log's `Loading "
        "time` is kept in `stage_s`)",
        "convert": "convert_s",
        "per": "round",
        "estimate": "`run_registration` with the final saving off (its diagnostic "
        "images and intermediate fields included, as shipped)",
        "write": "`apply_deformation_pyvips` once per channel; the conversion of each "
        "further channel before it and the benchmark's copy of those files into one "
        "after it are timed apart (`native.write_parts_s`)",
        "writes": "one single-channel pyramidal TIFF per channel (the package's); the "
        "benchmark stacks their first pages into one OME-TIFF without a pyramid",
    },
    "ashlar": {
        "reading": "inside the clock: ASHLAR's reading of the tiles it is given",
        "convert": "retile_s",
        "per": "patient",
        "estimate": "the command up to the last aligner; a round's share is its "
        "`layer_s` plus the reference stitch over the rounds",
        "write": "ASHLAR's writer, inside the same command, timed from the last "
        "aligner; then the benchmark's split of the mosaic per round, timed apart "
        "(`native.write_parts_s`)",
        "writes": "one pyramidal OME-TIFF for the patient, every cycle's channels "
        "(BigTIFF, deflate); the benchmark's per-round copies have no pyramid",
    },
    "palom": {
        "reading": "inside the clock: PALOM's reader, lazily, on the cohort's "
        "OME-TIFFs",
        "convert": NONE,
        "estimate": "`Aligner.coarse_register_affine`, `compute_shifts`, "
        "`constrain_shifts`",
        "write": "`block_affine_transformed_moving_img` and `write_pyramid`, timed on "
        "their own",
        "writes": "one pyramidal OME-TIFF per round, every channel (BigTIFF, zlib)",
    },
}


def _nan(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


# ── the trace ─────────────────────────────────────────────────────────────────
_BYTES = {"B": 1.0, "KB": 2.0**10, "MB": 2.0**20, "GB": 2.0**30, "TB": 2.0**40}


def parse_gb(text):
    """Nextflow's ``3.2 GB`` / ``512 MB`` / a bare byte count, in GiB; NaN for ``-``."""
    s = str(text if text is not None else "").strip()
    if not s or s == "-":
        return math.nan
    m = re.match(r"^([\d.]+)\s*([KMGT]?B)$", s, re.IGNORECASE)
    try:
        if m:
            return float(m.group(1)) * _BYTES[m.group(2).upper()] / _BYTES["GB"]
        return float(s) / _BYTES["GB"]
    except ValueError:
        return math.nan


def parse_seconds(text):
    """Nextflow's ``1h 2m 3s`` / ``120ms`` / ``1.5s`` / a bare number of milliseconds
    (its raw trace format), in seconds; NaN for ``-``."""
    s = str(text if text is not None else "").strip()
    if not s or s == "-":
        return math.nan
    found = re.findall(r"([\d.]+)\s*(ms|d|h|m|s)", s)
    if found:
        unit = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
        return sum(float(v) * unit[u] for v, u in found)
    try:
        return float(s) / 1000.0
    except ValueError:
        return math.nan


def read_trace(path):
    """The trace's rows, read BY HEADER NAME: ``process`` (its last part), ``tag``,
    ``status``, ``realtime_s``, ``peak_rss_gb``, ``cpus``, ``memory_gb``.

    SystemExit when the file is missing or lacks a column this view needs: a trace that
    cannot be read is said so, never turned into a table of blanks.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"--pipeline-trace: no such file: {path}")
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        need = ("process", "tag", "status", "realtime", "peak_rss")
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"--pipeline-trace: {path} has no column {', '.join(missing)} (found: "
                f"{', '.join(reader.fieldnames or []) or 'nothing'}); it must be a "
                "Nextflow trace.txt with its header line"
            )
        rows = []
        for r in reader:
            cpus = _nan(r.get("cpus"))
            rows.append(
                {
                    "process": str(r.get("process") or "").strip().split(":")[-1],
                    "tag": str(r.get("tag") or "").strip(),
                    "status": str(r.get("status") or "").strip().upper(),
                    "realtime_s": parse_seconds(r.get("realtime")),
                    "peak_rss_gb": parse_gb(r.get("peak_rss")),
                    "cpus": cpus,
                    "memory_gb": parse_gb(r.get("memory")),
                }
            )
    return rows


def _step(rows):
    """One patient's tasks of one step, as a cost: the time is the sum over the tasks
    (a patient has one conversion per slide), the peak and the allocation the largest."""
    rows = [r for r in rows if r["status"] in USED_STATUS]
    times = [r["realtime_s"] for r in rows if r["realtime_s"] == r["realtime_s"]]
    if not rows or not times:
        return None

    def most(key):
        values = [r[key] for r in rows if r[key] == r[key]]
        return max(values) if values else math.nan

    return {
        "s": float(sum(times)),
        "peak_gb": most("peak_rss_gb"),
        "cpus": most("cpus"),
        "memory_gb": most("memory_gb"),
        "tasks": len(rows),
        "cached": sum(r["status"] == "CACHED" for r in rows),
    }


def pipeline_steps(trace_rows, convert=PIPELINE_CONVERT, merge=PIPELINE_MERGE):
    """``{patient: {"convert": step | None, "merge": step | None}}`` from a trace."""
    out = {}
    for tag in dict.fromkeys(r["tag"] for r in trace_rows):
        mine = [r for r in trace_rows if r["tag"] == tag]
        steps = {
            "convert": _step([r for r in mine if r["process"] in convert]),
            "merge": _step([r for r in mine if r["process"] in merge]),
        }
        if steps["convert"] or steps["merge"]:
            out[tag] = steps
    return out


def _alloc(step):
    """What a task was given, in words: these are not this benchmark's 4 CPUs."""
    parts = []
    if step["cpus"] == step["cpus"]:
        parts.append(f"{step['cpus']:g} CPUs")
    if step["memory_gb"] == step["memory_gb"]:
        parts.append(f"{step['memory_gb']:g} GB")
    text = ", ".join(parts) or "allocation not in the trace"
    text += f"; {step['tasks']} task(s)"
    if step["cached"]:
        text += f", {step['cached']} cached (figures of the run that produced them)"
    return text


# ── what was written ──────────────────────────────────────────────────────────
def written_text(written):
    """``native.written`` in a few words; ``""`` when the run recorded none."""
    if not isinstance(written, dict) or written.get("unread"):
        return ""
    parts = []
    files = int(written.get("files") or 1)
    channels = written.get("channels")
    if channels is not None:
        each = " each" if files > 1 else ""
        parts.append(f"{channels} channel(s){each}")
    if files > 1:
        parts.append(f"{files} files")
    levels = written.get("levels")
    parts.append(
        f"pyramid ({levels} levels)" if written.get("pyramid") else "no pyramid"
    )
    parts.append(f"compression {str(written.get('compression') or '?').lower()}")
    size = written.get("bytes")
    if size is not None:
        parts.append(f"{float(size) / 2**30:.3g} GB")
    if written.get("of"):
        parts.append(str(written["of"]))
    return "; ".join(parts)


def complete_image(written):
    """True / False: is the method's own written image a pyramidal image holding every
    channel in one file (then no merge-and-pyramid step is added)? None when the run
    recorded no description."""
    if not isinstance(written, dict) or written.get("unread"):
        return None
    if "pyramid" not in written:
        return None
    return bool(written["pyramid"]) and int(written.get("files") or 1) == 1


# ── cores ─────────────────────────────────────────────────────────────────────
def cores_confined(res_rows):
    """One line for the head of every cost table: was each run held to its CPUs?

    Read from what every run recorded (``run.json``): ``cpus_requested`` (what the
    scheduler was asked for) and ``cpus_affinity`` (the cores the process could run
    on). A run is confined when the second is no larger than the first. It matters
    because STARE's coarse FFT and PALOM's dask pool take every core they can see.
    """
    ran = [r for r in res_rows if r.get("cause") != "not_run"]
    seen = [
        (r.get("cpus_asked"), r.get("cpus_affinity"), r.get("cpus_node")) for r in ran
    ]
    known = [(q, a, n) for q, a, n in seen if _is_number(q) and _is_number(a)]
    if not known:
        return (
            "cores confined: unknown (no run recorded both the CPUs it asked for and "
            "the cores it could run on; outside a scheduler nothing is asked for)"
        )
    yes = sum(a <= q for q, a, _ in known)
    asked = sorted({int(q) for q, _, _ in known})
    saw = sorted({int(a) for _, a, _ in known})
    node = sorted({int(n) for _, _, n in known if _is_number(n)})
    verdict = "yes" if yes == len(known) else "no" if yes == 0 else "partly"
    pools = {}
    for r in ran:
        if r.get("adapter") and str(r.get("workers_seen", "")) != "":
            pools.setdefault(r["adapter"], set()).add(str(r["workers_seen"]))
    held = "; ".join(f"{a} {'/'.join(sorted(v))}" for a, v in pools.items())
    return (
        f"cores confined: {verdict} ({yes} of {len(known)} runs could use no more "
        f"cores than the {', '.join(map(str, asked))} they asked for; cores usable "
        f"{', '.join(map(str, saw))}"
        + (f", cores on the node {', '.join(map(str, node))}" if node else "")
        + "). Where a run is not confined, a method that starts one thread per "
        "visible core used more than its allocation: read `cpu_efficiency`."
        + (f" Workers each method's own pool had: {held}." if held else "")
    )


# ── the table ─────────────────────────────────────────────────────────────────
COLUMNS = (
    "dataset",
    "group",
    "method",
    "device",
    "gpu_model",
    "n_rounds",
    "method_convert_s",
    "estimate_s",
    "write_s",
    "measured_here_s",
    "benchmark_copy_s",
    "pipeline_convert_s",
    "pipeline_merge_s",
    "total_s",
    "estimate_peak_gb",
    "job_peak_gb",
    "pipeline_convert_peak_gb",
    "pipeline_merge_peak_gb",
    "written",
    "written_from",
    "estimate_includes",
    "pipeline_convert_alloc",
    "pipeline_merge_alloc",
    "pipeline_source",
    "measured_here",
    "imported",
)


def _sum(values):
    values = [v for v in values if _is_number(v)]
    return float(sum(values)) if values else math.nan


def _max(values):
    values = [v for v in values if _is_number(v)]
    return float(max(values)) if values else math.nan


def end_to_end(res_rows, steps=None, source=""):
    """One row per (dataset, patient, result that wrote an image).

    ``res_rows`` are the scorer's resource rows (``score.resource_row`` plus dataset,
    group, method). ``steps`` is ``pipeline_steps(read_trace(...))`` or None when no
    trace was given; ``source`` names the run the trace is of.
    """
    out = []
    keys = dict.fromkeys((r["dataset"], r["group"], r["method"]) for r in res_rows)
    for ds, group, method in keys:
        mine = [
            r
            for r in res_rows
            if (r["dataset"], r["group"], r["method"]) == (ds, group, method)
        ]
        ok = [r for r in mine if r["ok"] and r.get("written_file")]
        if not ok:
            continue  # this result wrote no image: it has no end-to-end cost
        patient = any(r["unit"] == "patient" for r in ok)
        adapter = next((r.get("adapter") for r in ok if r.get("adapter")), "")
        clock = CLOCK.get(adapter, {})
        how = clock.get("convert", NONE)
        if how in (NONE, NOT_SEPARABLE):
            convert = how
        else:
            values = [r.get("method_convert_s") for r in ok]
            # one number per patient launch is repeated on every round of it
            convert = _max(values) if clock.get("per") == "patient" else _sum(values)
            if not _is_number(convert):
                convert = NOT_AVAILABLE
            else:  # and the conversion its writer needed, where that is timed apart
                convert = _sum([convert, _sum(r.get("write_convert_s") for r in ok)])
        estimate = (_max if patient else _sum)(r["register_s"] for r in ok)
        # the package's own writer where the adapter times it apart, else the step
        write = _sum(
            r["write_writer_s"] if _is_number(r.get("write_writer_s")) else r["write_s"]
            for r in ok
        )
        copy = _sum(r.get("write_benchmark_s") for r in ok)
        here = _sum([convert, estimate, write])

        described = [r.get("written") for r in ok if r.get("written")]
        complete = {r.get("written_complete") for r in ok} - {None, ""}
        if len(complete) == 1:
            needs_merge, basis = (not complete.pop()), "the file, read after the run"
        elif complete:
            needs_merge, basis = True, "the files differ between rounds"
        else:
            needs_merge, basis = None, "not recorded by this run"

        step = (steps or {}).get(str(group)) if steps is not None else None
        conv = (step or {}).get("convert")
        merge = (step or {}).get("merge")
        if steps is None:
            p_convert = p_merge = NOT_AVAILABLE
            why = "no pipeline trace was given"
        else:
            why = "" if step else f"patient {group} is not in the trace"
            p_convert = conv["s"] if conv else NOT_AVAILABLE
            p_merge = merge["s"] if merge else NOT_AVAILABLE
        if needs_merge is False:
            p_merge = NOT_NEEDED
        elif needs_merge is None and _is_number(p_merge):
            p_merge = NOT_AVAILABLE  # never added on a guess
        total = math.nan
        if _is_number(p_convert) and (_is_number(p_merge) or p_merge == NOT_NEEDED):
            total = _sum([here, p_convert, p_merge])

        imported = []
        if _is_number(p_convert):
            imported.append("pipeline_convert")
        if _is_number(p_merge):
            imported.append("pipeline_merge")
        out.append(
            {
                "dataset": ds,
                "group": group,
                "method": method,
                "device": "; ".join(
                    sorted({str(r.get("device") or "cpu") for r in ok})
                ),
                "gpu_model": "; ".join(
                    sorted({r.get("gpu_model") or "" for r in ok} - {""})
                ),
                "n_rounds": len(ok),
                "method_convert_s": convert,
                "estimate_s": estimate,
                "write_s": write,
                "measured_here_s": here,
                "benchmark_copy_s": copy,
                "pipeline_convert_s": p_convert,
                "pipeline_merge_s": p_merge,
                "total_s": total,
                "estimate_peak_gb": _max(r["peak_rss_gb"] for r in ok),
                "job_peak_gb": _max(r["cgroup_peak_gb"] for r in ok),
                "pipeline_convert_peak_gb": conv["peak_gb"] if conv else math.nan,
                "pipeline_merge_peak_gb": merge["peak_gb"]
                if merge and _is_number(p_merge)
                else math.nan,
                "written": described[0] if described else NOT_AVAILABLE,
                "written_from": basis,
                "estimate_includes": clock.get("reading", ""),
                "pipeline_convert_alloc": _alloc(conv) if conv else "",
                "pipeline_merge_alloc": _alloc(merge)
                if merge and _is_number(p_merge)
                else "",
                "pipeline_source": (source if step else why)
                if steps is not None
                else why,
                "measured_here": "method_convert, estimate, write"
                if _is_number(convert)
                else "estimate, write",
                "imported": ", ".join(imported) or NONE,
            }
        )
    return out


def summarise(rows):
    """Per (dataset, result): the median over patients of every step, how many patients
    have each, and the texts the patients share."""
    out = []
    for ds, method in dict.fromkeys((r["dataset"], r["method"]) for r in rows):
        mine = [r for r in rows if (r["dataset"], r["method"]) == (ds, method)]
        rec = {
            "dataset": ds,
            "method": method,
            "device": "; ".join(sorted({r["device"] for r in mine})),
            "gpu_model": "; ".join(sorted({r["gpu_model"] for r in mine} - {""})),
            "n_patients": len(mine),
        }
        for key in (
            "method_convert_s",
            "estimate_s",
            "write_s",
            "pipeline_convert_s",
            "pipeline_merge_s",
            "total_s",
        ):
            numbers = sorted(r[key] for r in mine if _is_number(r[key]))
            words = sorted({r[key] for r in mine if isinstance(r[key], str)})
            if numbers:
                mid = len(numbers) // 2
                rec[key] = (
                    numbers[mid]
                    if len(numbers) % 2
                    else (numbers[mid - 1] + numbers[mid]) / 2.0
                )
            else:
                rec[key] = "; ".join(words) if words else math.nan
            if key in ("pipeline_convert_s", "pipeline_merge_s", "total_s"):
                rec[f"n_{key[:-2]}"] = len(numbers)
        rec["written"] = "; ".join(dict.fromkeys(r["written"] for r in mine))
        rec["pipeline_source"] = "; ".join(
            dict.fromkeys(r["pipeline_source"] for r in mine if r["pipeline_source"])
        )
        out.append(rec)
    return out


SUMMARY_COLUMNS = (
    "dataset",
    "method",
    "device",
    "gpu_model",
    "n_patients",
    "method_convert_s",
    "estimate_s",
    "write_s",
    "pipeline_convert_s",
    "pipeline_merge_s",
    "total_s",
    "n_total",
    "written",
)


def markdown(rows, summary, source, table):
    """The block of ``summary.md``; ``table(rows, columns)`` renders one table."""
    if not rows:
        return []
    allocs = list(
        dict.fromkeys(
            f"convert: {r['pipeline_convert_alloc']}"
            for r in rows
            if r["pipeline_convert_alloc"]
        )
    ) + list(
        dict.fromkeys(
            f"merge and pyramid: {r['pipeline_merge_alloc']}"
            for r in rows
            if r["pipeline_merge_alloc"]
        )
    )
    parts = [
        "## Cost from the cohort's files to a merged image (secondary)\n",
        "The headline cost above is the time to ESTIMATE the transform. This block "
        "puts the other steps beside it, each in its own column, per result that wrote "
        "an image (median over patients; seconds). Methods keep their own readers and "
        "writers, so the outputs differ: `written` says what each leaves on disk. "
        f"`{NOT_SEPARABLE}` means the package does that step inside its registration "
        "call (it is in `estimate_s`); no number is guessed. `method_convert_s`, "
        "`estimate_s` and `write_s` were measured in this benchmark at 4 CPUs.\n",
    ]
    if source:
        parts.append(
            "`pipeline_convert_s` and `pipeline_merge_s` were NOT measured here: they "
            f"are read from the trace of an earlier pipeline run ({source}), whose "
            "tasks were given other resources ("
            + ("; ".join(allocs) or "none matched a patient")
            + "). The merge-and-pyramid step is added only where the method's own "
            f"written image is not already a pyramid holding every channel "
            f"(`{NOT_NEEDED}` otherwise). `total_s` is the sum of the columns and "
            "exists only for the `n_total` patients the trace has.\n"
        )
    else:
        parts.append(
            f"No pipeline trace was given (`score --pipeline-trace`): the pipeline's "
            f"conversion and merge-and-pyramid steps are `{NOT_AVAILABLE}` and there "
            "is no total.\n"
        )
    parts += [table(summary, list(SUMMARY_COLUMNS)), ""]
    return parts


def clock_markdown():
    """The table of what each adapter's clock includes (README, hand-back)."""
    head = (
        "| method (adapter) | reading and conversion | its own conversion, reported "
        "as | the registration clock | the writer | what it writes |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [
        f"| `{name}` | {c['reading']} | "
        + (
            f"`{c['convert']}`"
            if c["convert"] not in (NONE, NOT_SEPARABLE)
            else c["convert"]
        )
        + f" | {c['estimate']} | {c['write']} | {c['writes']} |"
        for name, c in CLOCK.items()
    ]
    return "\n".join([head, *lines])

"""Cost by step, from the cohort's files to the image each method writes (secondary).

The headline cost of a method is the CPU-hours of ESTIMATING its transform
(``regbench.billing``), with the times and the memory peaks beside them
(``resources.csv``, ``summary_resources.csv``, ``cost_summary.csv``). This module adds a
second view,
``cost_end_to_end.csv``: one row per patient and result that wrote an image, with the
steps apart. Every number in it was measured by this benchmark, in the method's own job.

==================  =================================================================
column              what it is
==================  =================================================================
``method_convert``  a conversion the METHOD needs before it can read the slides (the
                    single-page TIFFs DeeperHistReg is given, the tiles ASHLAR is
                    given). Clock paused. ``none`` when the method reads the cohort's
                    files as they are; ``not separable`` when its own conversion
                    happens inside its registration call.
``estimate``        the registration clock (``register_s``).
``write``           the method's own writer. Clock paused. Where an adapter times the
                    parts of its write step apart (``native.write_parts_s``), this is
                    the package's writer alone: a conversion that writer needed goes to
                    ``method_convert``, and the benchmark's own copying
                    (``benchmark_copy``: the split of ASHLAR's mosaic per round, the
                    stacking of DeeperHistReg's per-channel files) is shown and is in
                    NO total, because it is not the method's work and its output is
                    described as the package left it.
``measured_here``   the sum of the four: the total measured time of that result.
==================  =================================================================

Methods are NOT given a common reader or writer (ruling R3.13): each row is what that
package does, and ``written`` says what it leaves on disk (channels, pyramid or not,
compression, size), so that unlike outputs are not read as like ones.

**Where the total stops.** At the image each method writes itself. Those images are
not the same deliverable: ``merged_image`` says, from the file each run wrote, whether
it is already ONE pyramidal file for the PATIENT holding every round's channels
(ASHLAR's mosaic) or one file per round (the others), which a pipeline would still have
to merge and pyramid. Neither that step nor the conversion that made the cohort's files
is measured here, and none is imported from another run (the owner's ruling of
2026-10-12: no pipeline trace is read, and no total depends on one). Until that date
``score --pipeline-trace`` read both from a Nextflow trace into ``pipeline_convert_s``,
``pipeline_merge_s`` and a ``total_s``; those columns are gone.

``CLOCK`` says, per method, what the registration clock includes today. It is the table
in the README; ``test_cost.py`` holds the two together.
"""

from __future__ import annotations

import math

NOT_AVAILABLE = "not available"
NOT_SEPARABLE = "not separable"
NOT_RECORDED = "not recorded"
NONE = "none"

# What each adapter's clock includes, as built (benchmarks/regbench/methods/<name>.py).
# `convert`: how the method's own conversion is reported (a run.json info key, NONE, or
# NOT_SEPARABLE). `per`: whether that number is one per round or one per patient launch.
#
# `writes` is what each package's writer leaves, from its source. It is a description
# for the reader; the table itself never trusts it: `written` and `merged_image` are
# read from the file each run wrote (`native.written`, read by tifffile).
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
        "estimate": "COARSE, one REG-TILE job per tile and SOLVE, each a scheduler job "
        "of its own with the pipeline's allocation: COARSE + the longest tile + SOLVE, "
        "each the attempt its job ended ok with (`wall_unlimited_parallel_s`; a failed "
        "attempt is `wasted_*`; queue waits are in `elapsed_with_queue_s`)",
        "write": "STITCH, a process of its own inside the round's last job once the "
        "transform exists, timed on its own (the attempt that wrote the image)",
        "writes": "one OME-TIFF per round, every channel, BigTIFF, uncompressed, no "
        "pyramid",
    },
    "valis_mirage": {
        "reading": "inside the clock: VALIS's readers, its conversion of every slide "
        "to its working images, and the JVM start. Outside it: the benchmark placing "
        "the slides in the folder VALIS reads (`info.place_s`: a hard link, or a copy)",
        "convert": NOT_SEPARABLE,
        "estimate": "the pipeline's `valis_registration` up to the writer (rigid, "
        "non-rigid, micro)",
        "write": "`Slide.warp_and_save_slide`, clock and peak paused, timed per slide",
        "writes": "one pyramidal OME-TIFF per round, every channel (libvips, LZW)",
    },
    "valis": {
        "reading": "inside the clock: VALIS's readers, its conversion of every slide "
        "to its working images, and the JVM start. Outside it: the benchmark placing "
        "the slides in the folder VALIS reads (`info.place_s`: a hard link, or a copy)",
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


# The same work, stated per adapter (P11, R15): whether the method's own conversion is
# inside its registration clock (`yes`), timed apart and outside it (`no`), or there is
# none (`none`); the size it registers at; and the channels it reads. Read from each
# adapter's docstring and options (benchmarks/regbench/methods/<name>.py) `[FULL: the
# adapters' module docstrings and the lines cited there]`.
WORK = {
    "stare": {
        "convert_inside_clock": "none",
        "working_px": "full resolution, tile by tile (a STARE_PREP row reads the "
        "reduced nuclear slide, f x f blocks)",
        "channels_read": "the nuclear channel (--nuclear-index)",
    },
    "valis_mirage": {
        "convert_inside_clock": "yes",
        "working_px": "max_image_dim_px 4000 for the rigid step; the tier's "
        "max_processed_image_dim_px for the non-rigid and micro steps",
        "channels_read": "the DAPI channel (get_channel_index('dapi'), else channel 0)",
    },
    "valis": {
        "convert_inside_clock": "yes",
        "working_px": "VALIS's defaults (the _hires rows: upstream's high-resolution "
        "example)",
        "channels_read": "the DAPI channel (a fuzzy match on 'dapi')",
    },
    "deeperhistreg": {
        "convert_inside_clock": "no",
        "working_px": "4096 px on the short side (default_initial_nonrigid), 8192 px "
        "for the _hr rows",
        "channels_read": "the nuclear channel, written as one single-page TIFF per slide",
    },
    "ashlar": {
        "convert_inside_clock": "no",
        "working_px": "full resolution, the tiles the benchmark cut (retile_s, timed "
        "apart)",
        "channels_read": "the nuclear channel (-c), the same index in every cycle",
    },
    "palom": {
        "convert_inside_clock": "none",
        "working_px": "an affine from 2000 px thumbnails, then block shifts at level 0",
        "channels_read": "the nuclear channel of each slide (read_level_channels)",
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
    """True / False: is the method's own written image ONE pyramidal file for the
    PATIENT holding every round's channels, which is what a pipeline's
    merge-and-pyramid step produces? None when the run recorded no description, or not
    enough of one to tell.

    Decided from the file (ruling R3.14): it has a pyramid, it is one file, and it
    holds at least as many channels as the patient's slides hold together
    (``written.patient_channels``, which the driver records beside the description:
    the reference's channels plus every moving round's of that launch). A pyramidal
    file per ROUND, however complete for its round, is not the patient's image."""
    if not isinstance(written, dict) or written.get("unread"):
        return None
    if "pyramid" not in written:
        return None
    if not written["pyramid"] or int(written.get("files") or 1) != 1:
        return False
    needed, held = written.get("patient_channels"), written.get("channels")
    if needed is None or held is None:
        return None
    return int(held) >= int(needed)


# ── the lines above every cost table ──────────────────────────────────────────
STARE_LADDER = (
    "STARE's memory: its stage jobs climb the PIPELINE's own per-process ladder "
    "(COARSE, every tile, SOLVE at their requests x attempt), not the benchmark's "
    "16/64/240 GB rungs every other method climbs, and one tile job needs at most "
    "16 GB; its `mem_requested_gb` and `attempt` are those of the round's last job "
    "(the benchmark's own), not of the registration."
)
INSTRUMENTS_DIFFER = (
    "instruments differ by path: a registration in one process is read by its job "
    "cgroup's CPU counter, a stage job by `os.wait4` of its command, and the driver's "
    "sampler of the process tree costs a few percent of a core [MEMORY: the overhead "
    "is the sampler's design figure, unverified on this cluster]; the CPU-hours of the two "
    "paths are of the same kind, the peaks are not (`rss_method`)."
)
GB_H_POLICY = (
    "`gb_h_reserved` measures the REQUEST policy, not the method: STARE's stage jobs "
    "at the pipeline's per-process requests, every other method at the ladder's "
    "rung (16 GB first). `gb_h_used` (each attempt's own peak x its duration) is the "
    "memory the method held; `gb_h_reserved_scheduler` is what the cluster held "
    "(sacct's ReqMem x ElapsedRaw)."
)
JVM_RSS = (
    "JVM methods (VALIS): a JVM's resident memory follows its maximum heap (-Xmx, "
    "`jvm_max_heap_gb`), which the adapter sets from the rung, so its peak grows with "
    "the rung it ran at and is not the method's need; `jvm_peak_heap_gb` (the heap "
    "the JVM used) is NA: it is not measured."
)
def header_lines(res_rows):
    """What a reader of any cost table must know first, from what the runs recorded:
    how many had no out-of-memory watcher, which thread variables a method's image
    set, how many results were reused from an earlier submission, and the time limit
    each kind of unit ran under."""
    ran = [r for r in res_rows if r.get("cause") != "not_run"]
    lines = []
    told = [r for r in ran if r.get("oom_watch") in ("yes", "no")]
    without = [r for r in told if r["oom_watch"] == "no"]
    if not told:
        lines.append(
            "out-of-memory watcher: unknown (no run recorded whether its cgroup "
            "counter was readable)"
        )
    else:
        by = {}
        for r in without:
            by[r.get("adapter") or "?"] = by.get(r.get("adapter") or "?", 0) + 1
        lines.append(
            f"out-of-memory watcher: {len(without)} of {len(told)} results ran "
            "WITHOUT one (the cgroup's oom_kill counter was not readable from inside "
            "the job)"
            + (
                ": " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())) + ". A "
                "worker the kernel killed there could leave the run waiting until "
                "its time limit, so a `timeout` of these runs may be a memory death."
                if without
                else "."
            )
        )
    threads = {}
    for r in ran:
        if r.get("threads_env"):
            threads.setdefault(r.get("adapter") or "?", set()).add(r["threads_env"])
    lines.append(
        "thread variables: the benchmark sets none. "
        + (
            "Found set inside the job, so by the method's own image: "
            + "; ".join(
                f"{a} {', '.join(sorted(v))}" for a, v in sorted(threads.items())
            )
            + ". Such a method ran as its distribution configures it, on fewer "
            "threads than the 4 CPUs it was given and is billed for: read its "
            "`cpu_efficiency`."
            if threads
            else "No run found one set in its own process."
        )
    )
    split = [r for r in ran if r.get("ok") and _is_number(r.get("peak_cpus_at_once"))]
    if split:
        names = ", ".join(sorted({str(r.get("adapter") or "?") for r in split}))
        most = int(max(r["peak_cpus_at_once"] for r in split))
        tiles = [r["n_tiles"] for r in split if _is_number(r.get("n_tiles"))]
        from .stages import ALLOCATION as alloc

        lines.append(
            f"several jobs per round ({names}): its registration ran as separate "
            "scheduler jobs with the pipeline's allocation for each (COARSE "
            f"{alloc['coarse']['cpus']} CPUs, every tile {alloc['tile']['cpus']} CPUs, "
            f"SOLVE {alloc['solve']['cpus']} CPU), so its wall time used more than 4 "
            f"CPUs at once, up to {most} for one round; its wall time is NOT comparable "
            "with the 4-CPU methods', its CPU-hours (`cpu_h_used`, `cpu_h_reserved`) are. "
            "For these rows `register_s` is `wall_unlimited_parallel_s` (COARSE + the "
            "longest tile + SOLVE, each the attempt that succeeded, no queue; "
            "`elapsed_with_queue_s` has the waits), `total_to_image_s` adds STITCH's "
            "own process, `peak_rss_gb` is the peak of the largest single stage job "
            "(`largest_job`), and `attempt`, `mem_requested_gb`, `cpus_requested` and "
            "`cgroup_peak_gb` describe the round's LAST job (the benchmark's "
            "allocation: its own work and STITCH), not the registration. A failed "
            "stage attempt is in `wasted_*`; `unmeasured_attempts` counts attempts "
            "with no recorded duration, and where it is not 0 every time and CPU "
            "figure of that row is a lower bound"
            + (f"; {int(_median(tiles))} tiles per round (median)." if tiles else ".")
        )
    if split:
        lines.append(STARE_LADDER)  # R22
    lines.append(INSTRUMENTS_DIFFER)  # R23
    lines.append(GB_H_POLICY)  # R8
    if any(r.get("jvm_max_heap_gb") not in ("", None) for r in ran):
        lines.append(JVM_RSS)  # R16
    rusage = [r for r in ran if str(r.get("rss_method") or "").startswith("rusage")]
    failures = [
        r for r in ran if _is_number(r.get("cpu_read_failures")) and r["cpu_read_failures"]
    ]
    methods = sorted({str(r.get("rss_method")) for r in ran if r.get("rss_method")})
    lines.append(  # R19
        "memory instrument (`rss_method`): "
        + (", ".join(f"`{m}`" for m in methods) if methods else "none recorded")
        + f"; {len(rusage)} result(s) read their peak by `rusage` alone (a LOWER bound: "
        "`peak_rss_comparable = False`, in no peak comparison and counted where a peak is "
        f"compared); {len(failures)} result(s) had CPU counter reads that failed "
        "(`cpu_read_failures`)."
    )
    from . import telemetry

    lines.append(telemetry.contention_line(ran))  # R14
    reused = sum(bool(r.get("reused")) for r in ran)
    lines.append(
        f"reused: {reused} of {len(ran)} results were already ok for the same "
        "configuration and were not run again by this submission (their cost is the "
        "earlier launch's)."
    )
    limits = {}
    for r in ran:
        if r.get("time_limit"):
            limits.setdefault(r.get("unit") or "?", set()).add(str(r["time_limit"]))
    if limits:
        lines.append(
            "time limit per task: "
            + "; ".join(
                f"{unit} {', '.join(sorted(v))}" for unit, v in sorted(limits.items())
            )
            + ". One rule for both kinds of unit, one REGBENCH_TIME per round: a "
            "pair task gets it once, a patient task gets it times the rounds of the "
            "LARGEST patient of its array (one array has one limit), never above 7 "
            "days. So a patient with fewer rounds has more time per round than a "
            "pair has, and above 7 rounds less; `timeout` is never retried."
        )
    return lines


# ── cores ─────────────────────────────────────────────────────────────────────
EFFICIENCY_ABOVE = 1.05
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
    # R9: the stage jobs of a registration in several jobs, from their own records
    s_known = sum(int(r["stage_jobs_known"]) for r in ran if _is_number(r.get("stage_jobs_known")))
    s_yes = sum(
        int(r["stage_jobs_confined"]) for r in ran if _is_number(r.get("stage_jobs_confined"))
    )
    eff = {}
    for r in ran:
        for key in ("cpu_efficiency", "stage_cpu_efficiency_max"):
            if _is_number(r.get(key)):
                a = r.get("adapter") or "?"
                eff[a] = max(eff.get(a, -math.inf), float(r[key]))
    above = sum(
        _is_number(r.get("cpu_efficiency")) and r["cpu_efficiency"] > EFFICIENCY_ABOVE
        for r in ran
    )
    tail = (
        (
            f" Stage jobs: {s_yes} of {s_known} attempts held to their CPUs."
            if s_known
            else ""
        )
        + (
            " Largest cpu_efficiency per method: "
            + ", ".join(f"{a} {v:.2f}" for a, v in sorted(eff.items()))
            + f"; {above} row(s) above {EFFICIENCY_ABOVE:g} (more CPU than allocated)."
            if eff
            else ""
        )
    )
    if not known:
        return (
            "cores confined: unknown (no run recorded both the CPUs it asked for and "
            "the cores it could run on; outside a scheduler nothing is asked for)"
        ) + tail
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
        + tail
    )


# ── the table ─────────────────────────────────────────────────────────────────
COLUMNS = (
    "dataset",
    "group",
    "method",
    "device",
    "gpu_model",
    "n_rounds",
    "complete",
    "method_convert_s",
    "estimate_s",
    "write_s",
    "measured_here_s",
    "benchmark_copy_s",
    "estimate_peak_gb",
    "job_peak_gb",
    "written",
    "merged_image",
    "written_from",
    "estimate_includes",
    "measured_here",
)


def _sum(values):
    values = [v for v in values if _is_number(v)]
    return float(sum(values)) if values else math.nan


def _max(values):
    values = [v for v in values if _is_number(v)]
    return float(max(values)) if values else math.nan


def end_to_end(res_rows):
    """One row per (dataset, patient, result that wrote an image).

    ``res_rows`` are the scorer's resource rows (``score.resource_row`` plus dataset,
    group, method). Every number is this benchmark's own measurement.
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
            continue  # this result wrote no image: it has no cost to its image
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
        # the package's own writer where the adapter times it apart, else the step:
        # `score.resource_row` puts exactly that in the row's `write_s`, so this
        # column and the headline `write_s` are one number
        write = _sum(r["write_s"] for r in ok)
        copy = _sum(r.get("write_benchmark_s") for r in ok)
        here = _sum([convert, estimate, write])

        described = [r.get("written") for r in ok if r.get("written")]
        complete = {r.get("written_complete") for r in ok} - {None, ""}
        if len(complete) == 1:
            merged = "yes" if complete.pop() else "no"
            basis = "the file, read after the run"
        elif complete:
            merged, basis = "no", "the files differ between rounds"
        else:
            merged, basis = NOT_RECORDED, "not recorded by this run"
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
                # every round of the patient wrote its image: only then is the sum
                # over the rounds the patient's cost (`summarise` reads these alone,
                # the rule of the per-method block, `score.summarise_cost`)
                "complete": len(ok) == len(mine),
                "method_convert_s": convert,
                "estimate_s": estimate,
                "write_s": write,
                "measured_here_s": here,
                "benchmark_copy_s": copy,
                "estimate_peak_gb": _max(r["peak_rss_gb"] for r in ok),
                "job_peak_gb": _max(r["cgroup_peak_gb"] for r in ok),
                "written": described[0] if described else NOT_AVAILABLE,
                # is that image already the patient's one merged pyramid (module
                # docstring, "Where the total stops")
                "merged_image": merged,
                "written_from": basis,
                "estimate_includes": clock.get("reading", ""),
                "measured_here": ", ".join(
                    [
                        *(["method_convert"] if _is_number(convert) else []),
                        "estimate",
                        "write",
                    ]
                ),
            }
        )
    return out


def _median(numbers):
    numbers = sorted(numbers)
    if not numbers:
        return math.nan
    mid = len(numbers) // 2
    return numbers[mid] if len(numbers) % 2 else (numbers[mid - 1] + numbers[mid]) / 2.0


def summarise(rows, common=None, cohort=None, rule=None):
    """Per (dataset, result): the median over patients of every step, the largest
    memory peak, how many patients have each, and the texts the patients share.

    The medians are over the patients for which this result WROTE AN IMAGE
    (``n_patients``, of the ``n_patients_cohort`` it was run on; ``cohort`` maps
    (dataset, result) to that count): survivors, so a result that fails on the largest
    patients looks cheaper. ``common`` maps a dataset to the patients every compared
    row completed (``score.common_patients``); ``*_common`` are the same medians on
    those patients only (``n_patients_common``), the same ones for every result. A
    dataset's entry may be ``{result: patients}`` (the per-method block, B2); ``rule``
    names the block in column ``common_block``."""
    out = []
    for ds, method in dict.fromkeys((r["dataset"], r["method"]) for r in rows):
        every = [r for r in rows if (r["dataset"], r["method"]) == (ds, method)]
        # a patient some of whose rounds wrote no image has no cost of its own: the
        # sum over the rounds it has is not the patient's
        mine = [r for r in every if r.get("complete", True)]
        rec = {
            "dataset": ds,
            "method": method,
            "device": "; ".join(sorted({r["device"] for r in every})),
            "gpu_model": "; ".join(sorted({r["gpu_model"] for r in every} - {""})),
            "n_patients": len(mine),
            "n_patients_partial": len(every) - len(mine),
            "n_patients_cohort": (cohort or {}).get((ds, method), len(mine)),
        }
        if rule is not None:
            rec["common_block"] = rule
        block = (common or {}).get(ds, ())
        if isinstance(block, dict):
            block = block.get(method, ())
        shared = [r for r in mine if r["group"] in block] if common is not None else []
        rec["n_patients_common"] = len(shared) if common is not None else math.nan
        for key in ("estimate_s", "write_s", "measured_here_s"):
            rec[f"{key}_common"] = _median(
                [r[key] for r in shared if _is_number(r[key])]
            )
        for key in (
            "method_convert_s",
            "estimate_s",
            "write_s",
            "measured_here_s",
        ):
            numbers = sorted(r[key] for r in mine if _is_number(r[key]))
            words = sorted({r[key] for r in mine if isinstance(r[key], str)})
            if numbers:
                rec[key] = _median(numbers)
            else:
                rec[key] = "; ".join(words) if words else math.nan
        # the largest over the patients: what the result needed at most
        rec["estimate_peak_gb_max"] = _max(r["estimate_peak_gb"] for r in mine)
        rec["job_peak_gb_max"] = _max(r["job_peak_gb"] for r in mine)
        rec["written"] = "; ".join(dict.fromkeys(r["written"] for r in mine))
        rec["merged_image"] = "; ".join(dict.fromkeys(r["merged_image"] for r in mine))
        out.append(rec)
    return out


SUMMARY_COLUMNS = (
    "dataset",
    "method",
    "device",
    "gpu_model",
    "n_patients",
    "n_patients_partial",
    "n_patients_cohort",
    "n_patients_common",
    "method_convert_s",
    "estimate_s",
    "estimate_s_common",
    "write_s",
    "write_s_common",
    "measured_here_s",
    "measured_here_s_common",
    "estimate_peak_gb_max",
    "job_peak_gb_max",
    "written",
    "merged_image",
)


def markdown(rows, summary, table):
    """The block of ``summary.md``; ``table(rows, columns)`` renders one table."""
    if not rows:
        return []
    return [
        "## Cost by step, as measured here (secondary)\n",
        "The headline cost above is that of ESTIMATING the transform, in CPU-hours "
        "first. This block is in seconds and "
        "puts the other steps beside the estimate, each in its own column, per result that wrote "
        "an image (median over patients; seconds). Every number was measured in this "
        "benchmark, in the method's own job at 4 CPUs (DeeperHistReg's GPU rows also "
        "had one GPU: column `device`), EXCEPT a result whose registration ran as "
        "several scheduler jobs per round (STARE): its `estimate_s` is COARSE + the "
        "longest tile + SOLVE, jobs of 2, 2 and 1 CPUs, many tiles at once, and is not "
        "a 4-CPU time; nothing is taken from another run. "
        "`measured_here_s` is the sum of `method_convert_s`, "
        "`estimate_s` and `write_s`: the total measured time from the cohort's files "
        "to the image that result wrote. The two peaks are the largest over the "
        "patients: `estimate_peak_gb_max` the process tree while registering (for a "
        "result in several jobs: its largest single stage job), `job_peak_gb_max` the "
        "whole job, writer included (for such a result: its round's LAST job, which "
        "holds the benchmark's own work and the writer, not the registration). "
        "`write_s` is the method's own writer and is the headline's `write_s`: the "
        "benchmark's own copying is in neither. Methods keep their own "
        "readers and writers, so the outputs differ: `written` says what each leaves "
        "on disk, and `merged_image` whether that is already one pyramidal file for "
        "the patient with every round's channels (`yes`) or a file per round (`no`), "
        "which a pipeline would still merge and pyramid. That step, and the conversion "
        "that made the cohort's files, are not measured here and are in no number. "
        f"`{NOT_SEPARABLE}` means the package does that step inside its registration "
        "call (it is in `estimate_s`); no number is guessed. Each median is over the "
        "`n_patients` for which that result wrote an image for EVERY round, of the "
        "`n_patients_cohort` it was run on (a patient with some rounds missing is "
        "counted in `n_patients_partial` and is in no median): survivors only, so a "
        "result that fails on the largest patients looks cheaper. The `*_common` "
        "columns are the same "
        "medians on the `n_patients_common` patients every compared row completed: "
        "only those are on the same patients for every result.\n",
        table(summary, list(SUMMARY_COLUMNS)),
        "",
    ]


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

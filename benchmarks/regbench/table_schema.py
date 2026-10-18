"""The columns of every paper table: one owner, read by the scorer, the paper writers, the
collector, the plots and the manuscript-sources check. A missing value is the literal ``NA``.

``source`` (the first column of every table) is one of ``regbench``, ``arms:<arm>``,
``sweep:<run_id>``, ``regbench-interim:<folder>`` or ``expected`` (a row the collector
knows should exist but found no result for).

Every column carries a one-line ``what``; where a table gives none, ``_WHAT`` supplies the
definition shared by that column name in every table."""
from __future__ import annotations

import csv
import math
from typing import NamedTuple

__all__ = ["NA", "Col", "Table", "TABLES", "columns", "empty_row", "write", "read"]

NA = "NA"


class Col(NamedTuple):
    name: str
    unit: str
    what: str


class Table(NamedTuple):
    name: str
    key: tuple
    cols: tuple
    what: str

_WHAT = {
    "arm": "the retired arm the number comes from",
    "arm_stage": "the arm's registration stage the number was read at",
    "coarse_max_dim": "longest side of the coarse (low-resolution) registration image",
    "coarse_peak_gb": "peak memory of the coarse-stage job",
    "cpu_h_used": "CPU-hours actually used (trace cpu time)",
    "cpu_h_used_med": "median over patients of the CPU-hours used",
    "cpu_model": "CPU model string as the run recorded it",
    "cpus": "CPUs the run was given",
    "cycle_fully_modelled": "every tile of the cycle was replaced by the stage model",
    "d_dice_vs_stare_q25": "25th percentile of the per-case Dice difference against STARE",
    "d_dice_vs_stare_q75": "75th percentile of the per-case Dice difference against STARE",
    "d_final_um": "median centroid distance after the full registration",
    "d_initial_um": "median centroid distance before any registration",
    "d_rigid_um": "median centroid distance after the rigid stage",
    "dataset": "the dataset the row belongs to",
    "default": "the parameter's default value in nextflow.config",
    "description": "the parameter's description in nextflow_schema.json",
    "device": "cpu or gpu",
    "dice_arm": "matched-pair Dice of the retired arm",
    "dice_matched": "matched-pair Dice over all pairs of the case",
    "dice_mean": "mean matched-pair Dice over cases",
    "dice_med": "median matched-pair Dice over cases",
    "dice_q25": "25th percentile of matched-pair Dice over cases",
    "dice_q75": "75th percentile of matched-pair Dice over cases",
    "dice_regbench": "matched-pair Dice of the regbench result",
    "discard_infinite_error": "tiles discarded because their offset error was infinite",
    "discard_over_budget": "tiles discarded because the shift exceeded the budget",
    "discard_zero_offset": "tiles discarded because their offset was exactly zero",
    "disp_um_arm": "median centroid displacement of the retired arm",
    "disp_um_med": "median over cases of the median centroid displacement",
    "disp_um_q25": "25th percentile of the per-case median displacement",
    "disp_um_q75": "75th percentile of the per-case median displacement",
    "disp_um_regbench": "median centroid displacement of the regbench result",
    "family": "the method the row is shown under",
    "fit_ok": "the fit had enough points and distinct x to be a regression",
    "frame": "the frame the two sets were compared in",
    "gpu_h_med": "median GPU-hours per patient",
    "gpu_model": "GPU model as the run recorded it",
    "intercept": "fitted intercept",
    "key": "the parameter name",
    "mem_total_gb": "total memory of the machine",
    "method": "the registration method",
    "n": "points the fit used",
    "n_cores_logical": "logical cores of the machine",
    "n_cores_physical": "physical cores of the machine",
    "n_distinct_x": "distinct predictor values among the points",
    "n_failed": "pairs whose registration failed",
    "n_pairs": "number of pairs",
    "n_pairs_max": "largest matched-nuclei count over the case's pairs",
    "n_pairs_med": "median matched-nuclei count over the case's pairs",
    "n_patients": "patients with at least one result",
    "n_readable": "cases whose displacement is readable",
    "n_results": "result records behind the row",
    "n_tiles": "tiles registered",
    "name": "the parameter name",
    "noise_frac": "fraction of the tiles' content replaced by noise in the run",
    "note": "free-text remark, e.g. why a value is NA",
    "ok": "the round finished and produced the record",
    "os": "operating system of the machine",
    "out_root": "the results root the row was read from",
    "out_tile": "side of the tile the stage writes",
    "panel": "the moving slide's channel set, joined by '_'",
    "patient_id": "the patient",
    "peak_rss_gb": "peak resident memory",
    "peak_rss_gb_max": "largest peak resident memory over patients",
    "process": "the pipeline process name",
    "python": "Python version of the run",
    "r2": "coefficient of determination of the fit",
    "read_box_px_s64": "read box side for an interior tile at stride 64",
    "robustness_mean": "mean robustness over pairs (share of landmarks that improved)",
    "roi_px": "side of the zoom field",
    "roi_x": "left edge of the zoom field",
    "roi_y": "top edge of the zoom field",
    "row": "the result name (a variant of a run)",
    "rtre_median_mean": "mean over pairs of the median relative TRE",
    "rtre_median_med": "median over pairs of the median relative TRE",
    "rule": "the rule that chose the field or row, in words",
    "sigma": "residual standard deviation of the fit",
    "slope": "fitted slope",
    "solve_peak_gb": "peak memory of the solve job",
    "sprep": "the STARE preprocessing variant",
    "stare_version": "stare-registration version of the run",
    "stitch_peak_gb": "peak memory of the stitch job",
    "stride": "STARE tile stride",
    "tier": "the STARE tier (high, medium or low)",
    "tile": "STARE tile side",
    "tile_peak_gb_max": "largest peak memory over the round's tile jobs",
    "tile_peak_gb_p50": "median peak memory over the round's tile jobs",
    "tool": "the software package that ran",
    "tool_version": "the package version the run recorded",
    "tre_um_med": "median landmark error in micrometres",
    "type": "the parameter's schema type",
    "unit_id": "the round or patient the point comes from",
    "value": "the recorded value",
    "wall_s": "wall-clock seconds",
    "wall_s_med": "median wall-clock seconds",
}


def _col(name, unit, what):
    return Col(name, unit, what or _WHAT[name])


def _t(name, key, what, *cols):
    head = (Col("source", "", "where the row comes from: regbench, arms:<arm>, "
                "sweep:<run_id>, regbench-interim:<folder> or expected"),)
    return name, Table(name, tuple(key), head + tuple(_col(*c) for c in cols), what)


_ID = (
    ("dataset", "", "the dataset (multiplex = the cohort)"),
    ("patient_id", "", "the patient (the case's group)"),
    ("panel", "", "the moving slide's channel set, joined by '_' (e.g. DAPI_CD8_CD3)"),
    ("case_id", "", "the case: one moving panel against its patient's reference"),
)
_TABLE_LIST = [
    _t("per_pair", ("dataset", "patient_id", "panel", "row", "set"),
       "one row per (case, result, polygon set): the tidy per-pair numbers",
       *_ID,
       ("row", "", "the result name (a variant of a run)"),
       ("family", "", "the method the row is shown under (variants.family)"),
       ("set", "", "outlines (headline) or nuclei"),
       ("stage", "", "initial, rigid, micro_rigid, nonrigid, micro or final (R18: micro_rigid "
                     "is VALIS depth >= 1's _rigid row; micro exists at depth 2 only)"),
       ("tool_version", "", "the package version the run recorded (run.json info.version)"),
       ("frame", "", "reference or moving: where the two sets were compared"),
       ("dice_matched", "1", "matched-pair Dice over all pairs of the case"),
       ("iou_p50", "1", "median IoU of the pairs"),
       ("pair_fraction", "1", "pairs over the smaller native count"),
       ("pair_fraction_null_max", "1", "the null's largest pair fraction"),
       ("displacement_um_p50", "um", "median centroid displacement of the pairs"),
       ("displacement_um_p90", "um", "90th percentile centroid displacement"),
       ("displacement_readable", "bool", "pair fraction exceeds the null max by 0.10"),
       ("within_1um", "1", "share of nuclei with a partner centroid within 1 um"),
       ("n_pairs", "count", "matched pairs"),
       ("n_ref", "count", "reference objects"), ("n_mov", "count", "moving objects"),
       ("imputed_initial", "bool", "scored at the unregistered pose (a failure)"),
       ("excluded_from_summaries", "bool", "lost for a cause that is not the method's"),
       ("cause", "", "run.json cause"), ("selfcheck_status", "", "reported, never a gate")),
    _t("method_best", ("dataset", "family"),
       "the best variant of each method (family), by the rule in `rule`",
       ("dataset", "", ""), ("family", "", ""), ("best_row", "", "the chosen result name"),
       ("n_candidates", "count", "terminal rows of the family considered"),
       ("criterion", "", "median over patients of the patient-median dice_matched (outlines)"),
       ("criterion_value", "1", "the chosen row's value"),
       ("n_cases_common", "count", "cases every candidate has a non-excluded row on"),
       ("tie_break", "", "lower cpu_h_used median over patients, then the name"),
       ("in_sample_selection", "bool", "chosen on this cohort with this metric"),
       ("rule", "", "the rule, in words")),
    _t("method_summary", ("dataset", "family", "row", "which"),
       "per method: accuracy over panel pairs, the paired difference against STARE, cost",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""),
       ("which", "", "best or primary"), ("tool_version", "", ""),
       ("n_patients", "count", ""), ("n_pairs", "count", "panel pairs (cases)"),
       ("n_ok", "count", "cases registered"),
       ("dice_med", "1", ""), ("dice_q25", "1", ""), ("dice_q75", "1", ""), ("dice_mean", "1", ""),
       ("disp_um_med", "um", "median over readable cases of displacement_um_p50"),
       ("disp_um_q25", "um", ""), ("disp_um_q75", "um", ""), ("n_readable", "count", ""),
       ("d_dice_vs_stare_med", "1", "row minus STARE's best, per case, median"),
       ("d_dice_vs_stare_q25", "1", ""), ("d_dice_vs_stare_q75", "1", ""),
       ("n_pairs_vs_stare", "count", "cases both have"),
       ("n_pairs_stare_ge", "count", "cases where STARE's Dice >= this row's"),
       ("peak_rss_gb_med", "GB", "median over patients"), ("peak_rss_gb_max", "GB", ""),
       ("peak_rss_unit", "", "per task (largest job), per patient launch, or per round"),
       ("fold_peak_vs_stare", "1", "peak_rss_gb_med over STARE's"),
       ("cpu_h_used_med", "CPU-h", "per patient, median"), ("wall_s_med", "s", ""),
       ("wall_kind", "", "register_s, or time_unlimited_parallel_tiles_s for STARE jobs"),
       ("gpu_h_med", "GPU-h", ""), ("device", "", "cpu or gpu"), ("gpu_model", "", ""),
       ("cpu_model", "", "the node's CPU model as the runs recorded it")),
    _t("valis_stages", ("dataset", "row", "stage"),
       "Dice by VALIS stage, per VALIS run (tier x micro)",
       ("dataset", "", ""), ("row", "", "the run's final result name"),
       ("memory_mode", "", "high, medium or low"), ("micro_reg", "", "0, 1 or 2"),
       ("stage", "", "initial, rigid (depth 0), micro_rigid (the _rigid row at depth 1 and 2: "
                     "affine + micro-rigid), nonrigid, or micro (depth 2 only); R18"),
       ("stage_row", "", "result name of the stage; NA when the run has no such stage"),
       ("dice_med", "1", ""), ("dice_q25", "1", ""), ("dice_q75", "1", ""),
       ("n_pairs", "count", ""), ("disp_um_med", "um", "")),
    _t("stare_tiers", ("dataset", "row"), "Dice and cost per STARE run (tier, stride, prep)",
       ("dataset", "", ""), ("row", "", ""), ("tier", "", ""), ("stride", "px", ""),
       ("sprep", "", ""), ("dice_med", "1", ""), ("dice_q25", "1", ""), ("dice_q75", "1", ""),
       ("n_pairs", "count", ""), ("peak_task_gb_max", "GB", "largest single job"),
       ("cpu_h_used_med", "CPU-h", "")),
    _t("stare_stage_peaks", ("dataset", "case_id", "row"),
       "per STARE round: the peak of each stage job (the attempt it ended ok with)",
       *_ID, ("row", "", ""), ("tier", "", ""), ("stride", "px", ""),
       ("coarse_peak_gb", "GB", ""), ("tile_peak_gb_p50", "GB", ""), ("tile_peak_gb_max", "GB", ""),
       ("n_tiles", "count", ""), ("solve_peak_gb", "GB", ""), ("stitch_peak_gb", "GB", ""),
       ("read_box_px", "px", "STARE's read box side for an interior tile: core + 2 x 3 x stride"),
       ("read_box_mpx", "Mpx", "read_box_px squared / 1e6"),
       ("pipeline_window_px", "px", "conf/modules.config's sizing window: tile + 2 x max(halo, 3 x stride)")),
    _t("ashlar_discard", ("dataset", "case_id", "row"),
       "per ASHLAR round and cell: tiles replaced by ASHLAR's stage model",
       *_ID, ("row", "", ""), ("affine", "", "none or stare"), ("noise_frac", "1", ""),
       ("max_shift_um", "um", "15, 60 or 240"), ("n_tiles", "count", ""),
       ("discard_frac", "1", "share of the cycle's tiles replaced by the linear model"),
       ("discard_infinite_error", "count", ""), ("discard_zero_offset", "count", ""),
       ("discard_over_budget", "count", ""), ("cycle_fully_modelled", "bool", ""),
       ("rigid_offset_um", "um", "|M0(c) - c| x pixel size at the reference centre, STARE's anchor"),
       ("rigid_offset_from", "", "the run.json the anchor was read from"), ("ok", "bool", "")),
    _t("cost_points", ("dataset", "row", "unit_id"),
       "one point per unit (round for a pair method, patient for a per-patient one)",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""),
       ("unit_kind", "", "round or patient"), ("unit_id", "", ""),
       ("input_gb", "GB", "bytes of the slide files the unit read, 2^30"),
       ("input_gpx", "Gpx", "moving gigapixels"), ("wall_s", "s", ""), ("peak_rss_gb", "GB", ""),
       ("cpu_h_used", "CPU-h", ""), ("device", "", "")),
    _t("cost_fits", ("dataset", "row", "target", "predictor"),
       "target ~ predictor, least squares, per row; flat below 3 points or 2 distinct x",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""), ("target", "", "wall_s or peak_rss_gb"),
       ("predictor", "", "input_gb or input_gpx"), ("slope", "", ""), ("intercept", "", ""),
       ("r2", "1", ""), ("sigma", "", "residual s.d."), ("n", "count", ""),
       ("n_distinct_x", "count", ""), ("fit_ok", "bool", "")),
    _t("pipeline_cost_fits", ("backend", "process", "target"),
       "compute-all: per process, target ~ input_gb over its traced tasks",
       ("backend", "", "valis or tiled"), ("process", "", ""), ("target", "", "realtime_s or peak_rss_gb"),
       ("predictor", "", "input_gb"), ("slope", "", ""), ("intercept", "", ""), ("r2", "1", ""),
       ("sigma", "", ""), ("n", "count", ""), ("n_distinct_x", "count", ""), ("fit_ok", "bool", "")),
    _t("cpu_scaling", ("row", "cpus", "patient_id", "execution"),
       "time and memory against CPUs, primary rows, a few patients",
       ("family", "", ""), ("row", "", ""), ("cpus", "count", ""), ("patient_id", "", ""),
       ("execution", "", "one process, or local for STARE's stages in one job"),
       ("wall_s", "s", ""), ("cpu_h_used", "CPU-h", ""), ("peak_rss_gb", "GB", ""),
       ("speedup_vs_4", "1", "wall at 4 CPUs over wall at this count"),
       ("efficiency", "1", "speedup_vs_4 / (cpus / 4)"), ("out_root", "", "")),
    _t("laptop", ("case_id", "row"), "STARE on one cohort slide on a laptop",
       *_ID, ("row", "", ""), ("wall_s", "s", ""), ("peak_rss_gb", "GB", ""),
       ("cpu_h_used", "CPU-h", ""), ("cpu_model", "", ""), ("n_cores_logical", "count", ""),
       ("n_cores_physical", "count", ""), ("mem_total_gb", "GB", ""), ("os", "", ""),
       ("python", "", ""), ("stare_version", "", ""), ("dice_matched", "1", "")),
    _t("versions", ("row",), "what ran, per run",
       ("row", "", ""), ("method", "", ""), ("tool", "", ""), ("tool_version", "", ""),
       ("versions_seen", "", "every distinct version in the run's records, '|'-joined"),
       ("image", "", "REGBENCH_SIF_<image>"), ("n_results", "count", "")),
    _t("run_parameters", ("row", "key"), "the parameters each run recorded (Table S6)",
       ("row", "", ""), ("method", "", ""), ("key", "", ""), ("value", "", ""),
       ("from", "", "run.json info.implementation.parameters, or the run's options")),
    _t("hosts", ("row",), "where the runs ran",
       ("row", "", ""), ("cpu_model", "", ""), ("cpus_requested", "", "distinct, '|'-joined"),
       ("mem_requested_gb", "", "distinct rungs"), ("n_results", "count", "")),
    _t("stare_tier_table", ("tier",), "STARE's tiers (Table S3), from lib/RegPresets.groovy",
       ("tier", "", ""), ("tile", "px", ""), ("halo", "px", "the displacement bound"),
       ("out_tile", "px", ""), ("coarse_max_dim", "px", ""), ("read_box_px_s64", "px", ""),
       ("reg_tile_mem_gb", "GB", "attempt 1"), ("stitch_mem_gb", "GB", "attempt 1")),
    _t("pipeline_params", ("name",), "every pipeline parameter with its default (Table S1)",
       ("name", "", ""), ("default", "", ""), ("type", "", ""), ("description", "", "")),
    _t("cohort_counts", ("dataset", "patient_id", "row"), "panel pairs per case, nuclei per pair",
       ("dataset", "", ""), ("patient_id", "", ""), ("row", "", ""), ("n_panels", "count", "moving panels"),
       ("n_pairs_min", "count", "matched nuclei per pair, min"), ("n_pairs_med", "count", "matched nuclei per pair, median over the row's pairs in the patient"),
       ("n_pairs_max", "count", "matched nuclei per pair, max over the row's pairs in the patient")),
    _t("crosscheck_arms", ("patient_id", "panel", "row"),
       "regbench against the retired arms on the same (patient, panel)",
       ("patient_id", "", ""), ("panel", "", ""), ("row", "", ""), ("arm", "", ""),
       ("dice_regbench", "1", ""), ("dice_arm", "1", ""), ("dice_diff", "1", "regbench - arm"),
       ("disp_um_regbench", "um", ""), ("disp_um_arm", "um", ""), ("arm_stage", "", ""),
       ("note", "", "")),
    _t("overlay_rois", ("dataset", "case_id", "row"), "the zoom field of each overlay and why",
       *_ID, ("family", "", ""), ("row", "", ""), ("roi_x", "px", ""), ("roi_y", "px", ""),
       ("roi_px", "px", ""), ("d_initial_um", "um", ""), ("d_rigid_um", "um", ""),
       ("d_final_um", "um", ""), ("monotone", "bool", "d_initial > d_rigid > d_final"),
       ("rule", "", "")),
    _t("public_landmarks", ("dataset", "row"), "landmark error per public dataset and row",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""), ("n_pairs", "count", ""),
       ("n_failed", "count", ""), ("rtre_median_med", "1", ""), ("rtre_median_mean", "1", ""),
       ("tre_um_med", "um", ""), ("robustness_mean", "1", ""), ("frame", "", ""),
       ("aggregation", "", "the dataset's own definition")),
    _t("public_tissue", ("dataset", "row"), "tissue-level agreement (secondary), public datasets",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""), ("n_pairs", "count", ""),
       ("score_med", "1", "tissue.py's score, median over pairs"), ("note", "", "")),
]


def _unique(pairs):
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("duplicate table name in the schema")
    return out


TABLES = _unique(_TABLE_LIST)


def columns(name):
    return [c.name for c in TABLES[name].cols]


def empty_row(name, **values):
    cols = columns(name)
    unknown = set(values) - set(cols)
    if unknown:
        raise KeyError(f"{name}: no column {sorted(unknown)}")
    return {c: values.get(c, NA) for c in cols}


def _cell(v):
    if v is None or v == "":
        return NA
    if isinstance(v, float) and math.isnan(v):
        return NA
    return v


def write(path, name, rows):
    cols = columns(name)
    rows = list(rows)
    for r in rows:
        extra = set(r) - set(cols)
        if extra:
            raise KeyError(f"{name}: no column {sorted(extra)}")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _cell(r.get(c)) for c in cols})


def read(path):
    with open(path, newline="") as fh:
        return [{k: (None if v == NA else v) for k, v in r.items()} for r in csv.DictReader(fh)]

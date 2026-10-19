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
    "cpu_h_used": "CPU-hours actually used: the job cgroup's CPU counter for a "
    "one-process registration, os.wait4 of each stage job's command for STARE's stage "
    "jobs (resources.csv: cpu_instrument)",
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
# Task 21b: beside every headline Dice, never ranked and never in a selection (best_of,
# the paired difference): the full pairings and how much of the headline was the
# pipeline's fallback (regbench.full_pairs; score.FULL_PAIRINGS)
_FULL = (
    ("dice_full_lsa_med", "1", "dice_matched of the full LSA (the exact assignment in "
     "every component: no mutual-NN fallback, no work cap; regbench.full_pairs), median "
     "over patients of the patient medians; a case without one (not certified, failed: "
     "losses.csv) is left out, never 0. Beside dice_med, ranks nothing"),
    ("dice_full_mnn_med", "1", "dice_matched of mutual nearest neighbour on the whole case "
     "(dice_matched_mnn), median over patients of the patient medians. Ranks nothing"),
    ("fallback_frac_med", "1", "share of a case's cells (both sides) in a component the "
     "pipeline's pairing left to mutual nearest neighbour (more than cell_pairs' "
     "DEFAULT_MAX_COMPONENT_CELLS cells), median over patients of the patient medians"),
    ("fallback_frac_max", "1", "the largest fallback_frac of a case"),
    ("n_full_lsa", "count", "cases with full-LSA numbers"),
)


# T29 (the fair comparison): every accuracy table of the paper has rows per polygon set
# (outlines: the headline; nuclei beside) and per pooling (user 2026-10-18 "pool various
# ways"; ruling: patient pooling = the median of the per-patient medians)
_SET = ("set", "", "outlines (the headline: each nucleus grown by 10 px) or nuclei (the "
        "bare nuclei, beside)")
_POOLING = ("pooling", "", "how the cases are pooled: case (every case one value), patient "
            "(per-patient median first, then the median over patients) or common (patient "
            "pooling on the cross-method common cases: every compared row, the primary and "
            "best rows, has a kept row there; the headline)")
_N_UNITS = ("n_units", "count", "values the medians are over: cases (pooling=case) or patients")
_DICE_RULES = (
    ("dice_zero_med", "1", "dice_med said explicitly: a case with no matched pair counts "
     "as 0 (a failure scores 0; the headline rule)"),
    ("dice_skip_med", "1", "the same median with a case with no matched pair left out "
     "(finite values only; the scorer's rule before T29)"),
)
_PAIR_FRACTIONS = (
    ("pair_fraction_med", "1", "pairs over the smaller native count (the headline pair "
     "fraction), pooled as `pooling` says"),
    ("pair_fraction_max_med", "1", "pairs over the LARGER native count: a loss on the larger "
     "side lowers it"),
    ("pair_fraction_sym_med", "1", "2 x pairs over (n_ref + n_mov)"),
)


def _cost_summary_cols():
    """method_summary's paper.costs columns (``costs.summary_column_specs``)."""
    from .paper import costs

    return tuple(costs.summary_column_specs())


def _point_stem_cols():
    from .paper import costs

    return tuple(
        (stem, unit, what) for stem, (_, unit, what) in costs.STEMS.items()
    )


_COST_SUMMARY = _cost_summary_cols()
_POINT_STEMS = _point_stem_cols()
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
       ("pair_fraction_max", "1", "pairs over the larger native count (score.add_pair_fractions)"),
       ("pair_fraction_max_null_max", "1", "the null's largest pair_fraction_max"),
       ("pair_fraction_sym", "1", "2 x pairs over (n_ref + n_mov)"),
       ("pair_fraction_sym_null_max", "1", "the null's largest pair_fraction_sym"),
       ("displacement_um_p50", "um", "median centroid displacement of the pairs"),
       ("displacement_um_p90", "um", "90th percentile centroid displacement"),
       ("displacement_readable", "bool", "pair fraction exceeds the null max by 0.10"),
       ("within_1um", "1", "share of nuclei with a partner centroid within 1 um"),
       ("n_pairs", "count", "matched pairs"),
       ("n_ref", "count", "reference objects"), ("n_mov", "count", "moving objects"),
       ("imputed_initial", "bool", "scored at the unregistered pose (a failure)"),
       ("excluded_from_summaries", "bool", "lost for a cause that is not the method's"),
       ("cause", "", "run.json cause"), ("selfcheck_status", "", "reported, never a gate"),
       ("dice_matched_full_lsa", "1", "dice_matched of the full LSA (no fallback, no cap; "
        "regbench.full_pairs): beside the headline, ranks nothing"),
       ("dice_matched_full_mnn", "1", "dice_matched of mutual NN on the whole case "
        "(dice_matched_mnn): beside the headline, ranks nothing"),
       ("pair_fraction_full_lsa", "1", "pair_fraction of the full LSA"),
       ("pair_fraction_full_mnn", "1", "pair_fraction of mutual NN on the whole case "
        "(pair_fraction_mnn)"),
       ("fallback_frac", "1", "share of the case's cells (both sides) the pipeline's pairing "
        "left to mutual nearest neighbour"),
       ("full_lsa_status", "", "full_lsa_from_exact (the record's *_exact, no solve), "
        "full_lsa_solved, not_certified or failed (both: NA numbers, a losses.csv row)")),
    _t("method_best", ("dataset", "family", "set"),
       "the best variant of each method (family), by the rule in `rule`",
       ("dataset", "", ""), ("family", "", ""), ("best_row", "", "the chosen result name"),
       ("n_candidates", "count", "terminal rows of the family considered"),
       ("criterion", "", "median over patients of the patient-median dice_matched (outlines)"),
       ("criterion_value", "1", "the chosen row's value"),
       ("criterion_value_skip", "1", "the chosen row's value with a case with no matched pair "
        "left out (finite values only), beside the zero-filled criterion"),
       _SET,
       ("unit", "", "patient: the criterion is the median over patients of the patient "
        "medians (best_of and the paired tables pool on patients)"),
       ("n_cases_common", "count", "cases every candidate has a non-excluded row on, less those "
        "where no candidate has a matched pair"),
       ("n_dice_zeroed", "count", "the chosen row's common cases with no matched pair, whose "
        "missing dice_matched counted as 0"),
       ("tie_break", "", "the name (device-blind: no cost decides a tie)"),
       ("in_sample_selection", "bool", "chosen on this cohort with this metric"),
       ("rule", "", "the rule, in words"),
       ("common_block", "", "strict (the headline) or per_method (B2)"),
       ("excluded_runs", "", "the family's runs left out of this dataset (excluded_runs.csv), "
        "'run: reason' joined by ' | '; a family with no run left is a row of NA (T29)")),
    _t("method_best_per_method", ("dataset", "family", "set"),
       "method_best on the per-method common set (B2): beside the strict one, never the headline",
       ("dataset", "", ""), ("family", "", ""), ("best_row", "", "the chosen result name"),
       ("n_candidates", "count", "terminal rows of the family considered"),
       ("criterion", "", "median over patients of the patient-median dice_matched (outlines)"),
       ("criterion_value", "1", "the chosen row's value"),
       ("criterion_value_skip", "1", "the chosen row's value with a case with no matched pair "
        "left out (finite values only), beside the zero-filled criterion"),
       _SET,
       ("unit", "", "patient: the criterion is the median over patients of the patient "
        "medians (best_of and the paired tables pool on patients)"),
       ("n_cases_common", "count", "cases every candidate has a non-excluded row on, less those "
        "where no candidate has a matched pair"),
       ("n_dice_zeroed", "count", "the chosen row's common cases with no matched pair, whose "
        "missing dice_matched counted as 0"),
       ("tie_break", "", "the name (device-blind: no cost decides a tie)"),
       ("in_sample_selection", "bool", "chosen on this cohort with this metric"),
       ("rule", "", "the rule, in words"),
       ("common_block", "", "strict (the headline) or per_method (B2)"),
       ("excluded_runs", "", "the family's runs left out of this dataset (excluded_runs.csv), "
        "'run: reason' joined by ' | '; a family with no run left is a row of NA (T29)")),
    _t("method_summary", ("dataset", "family", "row", "which", "set", "pooling"),
       "per method: accuracy over panel pairs, the paired difference against STARE, cost "
       "(paper.costs: per patient, the common patients first, clock and unit named)",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""),
       ("which", "", "DEPRECATED (one release): the old name of row_kind; `excluded` for "
        "a run left out of this dataset (a row of NA, excluded_reason)"),
       ("excluded_reason", "", "why the run is not on this dataset (excluded_runs.csv); NA "
        "for a run that is"),
       ("row_kind", "", "primary (drawn filled), best (the best-Dice row: drawn hollow, "
        "'best (ceiling)'), local_4cpu (STARE in one 4-CPU job) or cpu_twin (a GPU row's "
        "CPU twin)"),
       _SET, _POOLING, _N_UNITS,
       ("selection", "", "how the row was chosen: a best row is the best of n_candidates "
        "by Dice on this dataset (an in-sample ceiling); a primary row is fixed before "
        "scoring (variants.PRIMARY), except DeeperHistReg's (primary = best, user "
        "2026-10-18: in-sample)"),
       ("n_candidates", "count", "a best row: the family's terminal rows that competed "
        "(method_best); NA on a row that was not selected"),
       ("in_sample_selection", "bool", "the row was chosen on this dataset with this metric"),
       ("tool_version", "", ""),
       ("n_patients", "count", ""), ("n_pairs", "count", "panel pairs (cases)"),
       ("n_ok", "count", "cases registered"),
       ("n_dice_zeroed", "count", "cases with no matched pair, whose missing dice_matched "
        "counted as 0 in the Dice statistics"),
       ("dice_med", "1", ""), ("dice_q25", "1", ""), ("dice_q75", "1", ""), ("dice_mean", "1", ""),
       *_DICE_RULES,
       ("dice_skip_q25", "1", "25th percentile of the units' Dice, a missing one left out"),
       ("dice_skip_q75", "1", "75th percentile of the units' Dice, a missing one left out"),
       ("dice_skip_mean", "1", "mean of the units' Dice, a missing one left out"),
       *_PAIR_FRACTIONS,
       ("disp_um_med", "um", "median over readable cases of displacement_um_p50"),
       ("disp_um_q25", "um", ""), ("disp_um_q75", "um", ""), ("n_readable", "count", ""),
       ("disp_um_common_med", "um", "the same on the cross-method common READABLE cases "
        "(displacement_readable for every compared row): the displacement that may be set "
        "beside another row's (WD11)"),
       ("disp_um_common_q25", "um", "25th percentile of disp_um_common's units"),
       ("disp_um_common_q75", "um", "75th percentile of disp_um_common's units"),
       ("n_readable_common", "count", "this row's cases among the common readable ones"),
       *_FULL,
       ("d_dice_vs_stare_med", "1", "row minus STARE's best, per unit (paired_unit: a case, "
        "or a patient's median over the cases both have), median over units"),
       ("d_dice_vs_stare_q25", "1", ""), ("d_dice_vs_stare_q75", "1", ""),
       ("d_dice_skip_vs_stare_med", "1", "the same with a missing Dice left out, not 0"),
       ("paired_unit", "", "case (pooling=case) or patient"),
       ("n_pairs_vs_stare", "count", "cases both have"),
       ("n_pairs_stare_ge", "count", "DEPRECATED (= n_units_stare_ge on pooling=case): cases "
        "where STARE's Dice >= this row's"),
       ("n_units_vs_stare", "count", "units (paired_unit) both have"),
       ("n_units_stare_ge", "count", "units where STARE's Dice >= this row's (a tie counts "
        "for STARE)"),
       ("n_patients_complete", "count", "patients whose every round registered (the "
        "deprecated *_med columns read these: the survivors)"),
       ("peak_rss_gb_med", "GB", "DEPRECATED: median over this row's complete patients "
        "(peak_rss_gb_common_med is the headline)"), ("peak_rss_gb_max", "GB", ""),
       ("peak_rss_unit", "", "per task (largest job), per patient launch, or per round"),
       ("cpu_h_used_med", "CPU-h", "DEPRECATED: per patient, median over this row's complete "
        "patients (cpu_h_used_common_med is the headline)"),
       ("wall_s_med", "s", "DEPRECATED: wall_s_survivors_med, on the clock in `clock`"),
       ("wall_kind", "", "DEPRECATED: register_s, or time_unlimited_parallel_tiles_s for "
        "STARE jobs; `clock` generalises it"),
       ("gpu_h_med", "GPU-h", "DEPRECATED: gpu_h_survivors_med"),
       ("device", "", "cpu or gpu"), ("gpu_model", "", ""),
       ("cpu_model", "", "the node's CPU model as the runs recorded it"),
       *_COST_SUMMARY,
       ("total_cpu_h_used_med", "CPU-h", "good + wasted attempts' CPU time, median over the "
        "COMMON patients (R10)"),
       ("cpu_h_reserved_med", "CPU-h", "allocated CPUs x duration of the good attempts, "
        "median over the common patients"),
       ("total_cpu_h_reserved_med", "CPU-h", "good + wasted reserved, median over the common "
        "patients (fig. 3f's total marker)"),
       ("rung_gb_med", "GB", "the memory rung the patient ended on, median over the common "
        "patients"),
       ("wasted_attempts", "count", "failed attempts over the common patients (rungs, retried "
        "stage jobs)"),
       ("fold_peak_vs_stare_local4", "1", "the row's median peak_rss_gb over stare_local4's, "
        "both over the SAME patients (n_patients_fold: common, complete for both, a comparable "
        "peak), both read by the same resources.Monitor in one 4-CPU job; NaN for STARE's "
        "stage-job row (another instrument; fold_peak_vs_stare is gone)"),
       ("n_patients_fold", "count", "the patients fold_peak_vs_stare_local4 is over"),
       ("fold_note", "", "why the fold is blank, or what it is of (a GPU row: its host RSS "
        "only)"),
       ("peak_largest_job_gb", "GB", "STARE's stage jobs: the largest single attempt's peak, "
        "what a node must give ONE job (median over the common patients)"),
       ("peak_concurrent_gb", "GB", "STARE's stage jobs: the most the jobs of one round held "
        "at once, the sum of the peaks of overlapping attempts (median, common patients)"),
       ("gpu_peak_gb", "GB", "torch's peak allocated CUDA memory, median over the common patients"),
       ("convert_s_med", "s", "the method's own conversion outside its clock, median over the "
        "common patients"),
       ("convert_inside_clock", "", "yes (the conversion is inside the registration clock), "
        "no (timed apart: convert_s_med), none (the method converts nothing)"),
       ("working_px", "", "the size the method registers at (cost.WORK)"),
       ("channels_read", "", "the channels the method reads (cost.WORK)"),
       ("jvm_max_heap_gb", "GB", "a JVM method: the heap the JVM reported (Runtime.maxMemory); "
        "its RSS follows it and grows with the rung"),
       ("jvm_peak_heap_gb", "GB", "the JVM's own peak heap use: NA, not measured (the reason)"),
       ("prior_cpu_h", "CPU-h", "an affine-anchored ASHLAR row: what the row it reads cost up "
        "to that stage (STARE's COARSE, or the valis_mirage _rigid row), median, common patients"),
       ("with_prior_cpu_h", "CPU-h", "the row's own cpu_h_used plus its prior, summed per "
        "common patient, then the median (never a median plus a median)"),
       ("n_prior_rounds_missing", "count", "rounds of the common patients with no "
        "stare_stage_peaks line: their prior is a lower bound")),
    _t("valis_stages", ("dataset", "row", "stage", "set", "pooling"),
       "Dice by VALIS stage, per VALIS run (tier x micro)",
       ("dataset", "", ""), ("row", "", "the run's final result name"),
       ("memory_mode", "", "high, medium or low"), ("micro_reg", "", "0, 1 or 2"),
       ("stage", "", "initial, rigid (depth 0), micro_rigid (the _rigid row at depth 1 and 2: "
                     "affine + micro-rigid), nonrigid, or micro (depth 2 only); R18"),
       ("stage_row", "", "result name of the stage; NA when the run has no such stage"),
       _SET, _POOLING, _N_UNITS,
       ("dice_med", "1", ""), ("dice_q25", "1", ""), ("dice_q75", "1", ""),
       *_DICE_RULES,
       ("n_dice_zeroed", "count", "cases with no matched pair (0 in dice_med, left out of "
        "dice_skip_med)"),
       ("n_pairs", "count", ""), ("disp_um_med", "um", ""), *_FULL),
    _t("stare_tiers", ("dataset", "row", "set", "pooling"),
       "Dice and cost per STARE run (tier, stride, prep)",
       ("dataset", "", ""), ("row", "", ""), ("tier", "", ""), ("stride", "px", ""),
       ("sprep", "", ""), _SET, _POOLING, _N_UNITS,
       ("dice_med", "1", ""), ("dice_q25", "1", ""), ("dice_q75", "1", ""),
       *_DICE_RULES,
       ("n_dice_zeroed", "count", "cases with no matched pair (0 in dice_med, left out of "
        "dice_skip_med)"),
       ("n_pairs", "count", ""), *_FULL,
       ("n_patients_common", "count", "common patients this row completed: the cost columns "
        "are over these (rule 2, method_summary's)"),
       ("n_patients_survivors", "count", "patients this row completed"),
       ("n_peak_not_comparable", "count", "of the common patients, peaks left out (a rusage "
        "lower bound)"),
       ("clock", "", "the clock of the row's times (critical_path_unlimited for stage jobs)"),
       ("unit", "", "patient: one value per patient"),
       ("cpu_h_used_common_med", "CPU-h", "CPU time the good attempts used, median over the "
        "common patients"),
       ("cpu_h_used_common_ci_low", "CPU-h", "bootstrap 95% interval of that median, low"),
       ("cpu_h_used_common_ci_high", "CPU-h", "bootstrap 95% interval of that median, high"),
       ("peak_rss_gb_common_med", "GB", "the memory one job needs (STARE's largest stage job), "
        "median over the common patients"),
       ("peak_rss_gb_common_ci_low", "GB", "bootstrap 95% interval of that median, low"),
       ("peak_rss_gb_common_ci_high", "GB", "bootstrap 95% interval of that median, high")),
    _t("stare_stage_peaks", ("dataset", "case_id", "row"),
       "per STARE round: the peak of each stage job (the attempt it ended ok with)",
       *_ID, ("row", "", ""), ("tier", "", ""), ("stride", "px", ""),
       ("clock", "", "stage_job: every figure is of one stage job's own good attempt"),
       ("unit", "", "round: one row per round"),
       ("coarse_peak_gb", "GB", ""), ("tile_peak_gb_p50", "GB", ""), ("tile_peak_gb_max", "GB", ""),
       ("n_tiles", "count", ""), ("solve_peak_gb", "GB", ""), ("stitch_peak_gb", "GB", ""),
       ("read_box_px", "px", "STARE's read box side for an interior tile: core + 2 x 3 x stride"),
       ("read_box_mpx", "Mpx", "read_box_px squared / 1e6"),
       ("pipeline_window_px", "px", "conf/modules.config's sizing window: tile + 2 x max(halo, 3 x stride)"),
       ("coarse_cpu_efficiency", "1", "COARSE's cpu_s / (cpus x wall_s), the attempt it ended ok with "
        "(above 1: its threads were not held to its CPUs)"),
       ("tile_cpu_efficiency", "1", "the same over the round's tiles' good attempts"),
       ("solve_cpu_efficiency", "1", "the same for SOLVE"),
       ("coarse_cpu_h", "CPU-h", "COARSE's measured CPU time (the prior an affine=stare ASHLAR row reads)"),
       ("coarse_s", "s", "COARSE's wall time, the attempt it ended ok with")),
    _t("ashlar_discard", ("dataset", "case_id", "row"),
       "per ASHLAR round and cell: tiles replaced by ASHLAR's stage model",
       *_ID, ("row", "", ""), ("affine", "", "none or stare"), ("noise_frac", "1", ""),
       ("max_shift_um", "um", "15, 60 or 240"), ("n_tiles", "count", ""),
       ("discard_frac", "1", "share of the cycle's tiles replaced by the linear model"),
       ("discard_infinite_error", "count", ""), ("discard_zero_offset", "count", ""),
       ("discard_over_budget", "count", ""), ("cycle_fully_modelled", "bool", ""),
       ("rigid_offset_um", "um", "|M0(c) - c| x pixel size at the reference centre, STARE's anchor"),
       ("rigid_offset_from", "", "the run.json the anchor was read from"), ("ok", "bool", "")),
    _t("cost_points", ("dataset", "row", "unit_kind", "unit_id"),
       "one point per PATIENT of every row (whatever became of it: `status`), and fig. 2c's "
       "per-round points (unit_kind round), by paper.costs's rules",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""),
       ("row_kind", "", "primary, best, local_4cpu, cpu_twin or other"),
       ("is_primary", "bool", "the family's primary row (drawn filled)"),
       ("is_best", "bool", "the family's best-Dice row (drawn hollow: best (ceiling))"),
       ("unit_kind", "", "patient, or round (fig. 2c's per-round facet)"),
       ("unit", "", "the unit of every number of the point: patient or round"),
       ("unit_id", "", "the patient, or the case"),
       ("status", "", "complete, capped, failed, excluded or not_run: only complete points "
        "enter a median or a fit"),
       ("in_common", "bool", "the patient is in the dataset's common set"),
       ("clock", "", "whole_process, local_4cpu, critical_path_unlimited or with_queue"),
       ("input_gb", "GB", "bytes of the slide files the unit read, 2^30"),
       ("input_gpx", "Gpx", "moving gigapixels"),
       ("lower_bound", "", "why the times and CPU-hours are lower bounds ('|'-joined), or NA"),
       ("peak_instrument", "", "rss_method of the unit's rounds"),
       ("peak_rss_comparable", "bool", "False for a rusage peak: in no peak comparison"),
       ("n_imputed_costed", "count", "rounds that registered, were scored as imputed failures "
        "and are costed"),
       ("wasted_attempts", "count", "the patient's failed attempts (rungs, retried stage jobs)"),
       ("device", "", ""), ("gpu_model", "", ""), ("cpu_model", "", ""),
       *_POINT_STEMS),
    _t("cost_fits", ("dataset", "row", "target", "predictor"),
       "log10(target) ~ log10(predictor), least squares, per row over its complete patients; "
       "no fit below 5 points or 2 distinct x",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""),
       ("target", "", "wall_s, peak_rss_gb or cpu_h_used"),
       ("predictor", "", "input_gb or input_gpx"),
       ("unit", "", "patient"), ("patient_set", "", "common: the common patients the row completed (every compared row "
        "completed them; the set fig. 2c draws)"),
       ("scale", "", "log-log"), ("min_n", "count", "points a fit needs (5)"),
       ("slope", "", "the exponent: target grows as predictor^slope"),
       ("slope_ci_low", "", "95% interval of the slope (Student's t, n - 2 d.f.), low"),
       ("slope_ci_high", "", "95% interval of the slope, high"),
       ("intercept", "", "log10 of the target at predictor 1"),
       ("r2", "1", ""), ("sigma", "", "residual s.d. in log10 units (ddof = 1)"),
       ("n", "count", ""), ("n_distinct_x", "count", ""), ("fit_ok", "bool", ""),
       ("n_dropped", "count", "points left out: no finite target or predictor, or <= 0")),
    _t("pipeline_cost_fits", ("backend", "process", "target"),
       "compute-all: per process, target ~ input_gb over its traced tasks",
       ("backend", "", "valis or tiled"), ("process", "", ""), ("target", "", "realtime_s or peak_rss_gb"),
       ("predictor", "", "input_gb"), ("slope", "", ""), ("intercept", "", ""), ("r2", "1", ""),
       ("sigma", "", ""), ("n", "count", ""), ("n_distinct_x", "count", ""), ("fit_ok", "bool", ""),
       ("n_dropped", "count", "completed tasks left out: no input size or no finite target")),
    _t("cpu_scaling", ("row", "cpus", "patient_id", "execution", "repeat"),
       "the CPU cross: each family's best-Dice CPU row at 4 and 8 CPUs, per patient and repeat",
       ("family", "", ""), ("row", "", ""), ("cpus", "count", ""), ("patient_id", "", ""),
       ("execution", "", "one process, or local for STARE's stages in one job"),
       ("repeat", "count", "the launcher's rep<k> (REGBENCH_REPEAT); 1 without"),
       ("wall_s", "s", "on the clock in wall_clock (shared nodes)"),
       ("wall_clock", "", "whole_process, local_4cpu or critical_path_unlimited"),
       ("clock", "", "the clock of every figure of the point (wall_s, cpu_h_used, "
        "peak_rss_gb): wall_clock's value"),
       ("unit", "", "patient: one value per patient, count and repeat"),
       ("cpu_h_used", "CPU-h", ""), ("peak_rss_gb", "GB", ""),
       ("speedup_vs_4", "1", "wall at 4 CPUs over wall at this count, the same repeat"),
       ("efficiency", "1", "speedup_vs_4 / (cpus / 4)"),
       ("cpus_checked", "", "yes when every record asked for this count (cpus_requested), "
        "else no: <what they asked>"),
       ("cpu_model", "", "the CPU model(s) of the point's records"),
       ("cpu_model_matches", "", "yes when every point of the patient ran on one CPU model"),
       ("n_patients_incomplete", "count", "patients of the row not complete at every count: "
        "left out of it"),
       ("out_root", "", "")),
    _t("end_to_end", ("dataset", "family", "row", "row_kind"),
       "per method_summary row: conversion, registration, writer and the prior, median over "
       "the common patients (figS_end_to_end)",
       ("dataset", "", ""), ("family", "", ""), ("row", "", ""),
       ("row_kind", "", "as in method_summary"),
       ("clock", "", "the registration's clock"),
       ("unit", "", "patient: one value per patient"),
       ("n_patients", "count", "common patients this row completed, with a line in "
        "cost_end_to_end.csv"),
       ("convert_s_med", "s", "the method's own conversion, timed apart"),
       ("convert_s_ci_low", "s", "bootstrap 95% interval of that median, low"),
       ("convert_s_ci_high", "s", "bootstrap 95% interval of that median, high"),
       ("estimate_s_med", "s", "the registration (estimate_s of cost_end_to_end.csv)"),
       ("estimate_s_ci_low", "s", "bootstrap 95% interval of that median, low"),
       ("estimate_s_ci_high", "s", "bootstrap 95% interval of that median, high"),
       ("write_s_med", "s", "the method's own writer"),
       ("write_s_ci_low", "s", "bootstrap 95% interval of that median, low"),
       ("write_s_ci_high", "s", "bootstrap 95% interval of that median, high"),
       ("prior_cpu_h", "CPU-h", "an affine-anchored ASHLAR row's prior (method_summary)"),
       ("prior_s_med", "s", "that prior's wall time"),
       ("prior_s_ci_low", "s", "bootstrap 95% interval of that median, low"),
       ("prior_s_ci_high", "s", "bootstrap 95% interval of that median, high"),
       ("convert_inside_clock", "", "yes, no or none")),
    _t("laptop", ("case_id", "row"), "STARE on one cohort slide on a laptop",
       *_ID, ("row", "", ""),
       ("clock", "", "whole_process: register + write, to the written image, on one laptop "
        "(not shared)"),
       ("unit", "", "case: one value per round"),
       ("wall_s", "s", ""), ("peak_rss_gb", "GB", ""),
       ("cpu_h_used", "CPU-h", ""), ("cpu_model", "", ""), ("n_cores_logical", "count", ""),
       ("n_cores_physical", "count", ""), ("mem_total_gb", "GB", ""), ("os", "", ""),
       ("python", "", ""), ("stare_version", "", ""), ("dice_matched", "1", "")),
    _t("versions", ("row",), "what ran, per run",
       ("row", "", ""), ("method", "", ""), ("tool", "", ""), ("tool_version", "", ""),
       ("versions_seen", "", "every distinct version in the run's records, '|'-joined"),
       ("image", "", "REGBENCH_SIF_<image>"),
       ("image_tag", "", "the image the launcher pins for it (slurm/launch.sbatch IMAGES); "
        "'not recorded' for a row of no run"),
       ("image_digest", "", "as the run's records give it (image_digest); 'not recorded' "
        "where none does"),
       ("python", "", "the Python of the method's process, as its records give it; "
        "'not recorded' where none does"),
       ("stardist_image", "", "the image the nuclei (outlines) were segmented with "
        "(nuclei.PIPELINE_IMAGE)"),
       ("stardist_version", "", "the StarDist package version, as a record gives it; 'not "
        "recorded' where none does"),
       ("n_results", "count", ""),
       ("n_unreadable", "count", "run.json files of the run that could not be read"),
       ("excluded_reason", "", "why the run is not on this dataset (excluded_runs.csv); NA "
        "for a run that is")),
    _t("run_parameters", ("row", "key"), "the parameters each run recorded (Table S6)",
       ("row", "", ""), ("method", "", ""), ("key", "", ""), ("value", "", ""),
       ("from", "", "run.json info.implementation.parameters, or the run's options"),
       ("n_unreadable", "count", "run.json files of the final row that could not be read")),
    _t("hosts", ("row",), "where the runs ran",
       ("row", "", ""), ("cpu_model", "", ""), ("cpus_requested", "", "distinct, '|'-joined"),
       ("mem_requested_gb", "", "distinct rungs"), ("n_results", "count", "")),
    _t("stare_tier_table", ("tier",), "STARE's tiers (Table S3), from lib/RegPresets.groovy",
       ("tier", "", ""), ("tile", "px", ""), ("halo", "px", "the displacement bound"),
       ("out_tile", "px", ""), ("coarse_max_dim", "px", ""), ("read_box_px_s64", "px", ""),
       ("reg_tile_mem_gb", "GB", "attempt 1"), ("stitch_mem_gb", "GB", "attempt 1")),
    _t("pipeline_params", ("name",), "every pipeline parameter with its default (Table S1)",
       ("name", "", ""), ("default", "", ""), ("type", "", ""), ("description", "", "")),
    _t("cohort_counts", ("dataset", "patient_id", "row", "set"),
       "panel pairs per case, nuclei per pair",
       ("dataset", "", ""), ("patient_id", "", ""), ("row", "", ""), _SET, ("n_panels", "count", "moving panels"),
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
    _t("public_landmarks", ("dataset", "set", "row"),
       "landmark error per public landmark dataset, set of pairs and row (the terminal "
       "rows and initial; regbench.landmarks)",
       ("dataset", "", ""), ("set", "", "all_pairs, subset_a, subset_b (HyReCo)"),
       ("family", "", ""), ("row", "", ""),
       ("n_pairs", "count", "pairs in the statistics (not_run pairs left out)"),
       ("n_failed", "count", "pairs scored at their unregistered landmarks"),
       ("n_not_run", "count", "pairs never registered for a cause not the method's"),
       ("n_flagged", "count", "pairs whose landmarks failed a sanity check (kept)"),
       ("rtre_median_med", "1", "median over pairs of the per-pair median rTRE"),
       ("rtre_median_mean", "1", "mean over pairs of the per-pair median rTRE"),
       ("rtre_max_med", "1", "median over pairs of the per-pair maximum rTRE"),
       ("rtre_max_mean", "1", "mean over pairs of the per-pair maximum rTRE"),
       ("n_robust", "count", "pairs with robustness above 0.5 (BIRL's robust subset)"),
       ("rtre_median_robust_med", "1", "rtre_median_med over the robust pairs"),
       ("rtre_median_robust_mean", "1", "rtre_median_mean over the robust pairs"),
       ("rtre_max_robust_mean", "1", "rtre_max_mean over the robust pairs"),
       ("robustness_med", "1", "median over pairs of the share of landmarks improved"),
       ("robustness_mean", "1", "mean over pairs of the share of landmarks improved"),
       ("tre_um_med", "um", "median over pairs of the per-pair median TRE"),
       ("tre_um_pooled_med", "um", "HyReCo: one median over all landmarks of all pairs "
        "(Lotz et al. 2023's MTRE [PARTIAL]); NA elsewhere"),
       ("best_of_family", "bool", "the family's best row on this set by rtre_median_med "
        "(every terminal row of the family a candidate)"),
       ("in_sample_selection", "bool", "the best row is chosen on these same pairs"),
       ("frame", "", ""),
       ("aggregation", "", "the dataset's own definition"),
       ("excluded_reason", "", "why the run is not on this dataset (excluded_runs.csv): a "
        "row of NA (T29); NA for a run that is")),
    _t("public_landmark_pairs", ("dataset", "case_id", "row"),
       "one row per landmark pair and row (the terminal rows and initial) of a public "
       "landmark dataset",
       ("dataset", "", ""), ("case_id", "", "the pair (the scorer's case id)"),
       ("set", "", "HyReCo: the subset (A, B)"),
       ("family", "", ""), ("row", "", ""), ("frame", "", ""),
       ("rtre_median", "1", "the median landmark error of the pair over the target image's diagonal"), ("rtre_max", "1", "the largest landmark error of the pair over the target image's diagonal"), ("robustness", "1", "the share of the pair's landmarks whose error is below their initial error"),
       ("tre_um_p50", "um", "the median landmark error of the pair, micrometres (NA without a pixel size)"), ("tre_um_p90", "um", "the 90th percentile landmark error of the pair, micrometres"),
       ("imputed_initial", "bool", "scored at the unregistered landmarks"),
       ("excluded_from_summaries", "bool", "not run for a cause not the method's"),
       ("flagged", "bool", "a landmark sanity check failed (kept)"),
       ("flags", "", "which checks"),
       ("shared_id", "bool", "HyReCo: an id both subsets list; NA elsewhere"),
       ("cause", "", "why the pair did not register, when it did not")),
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

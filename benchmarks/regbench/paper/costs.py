"""The paper's cost: one rule for every method and every figure (Task 31).

Every cost number of the paper tables and figures is made here, from the scorer's
``summary_resources.csv`` and ``resources.csv``, by these rules:

1. **One unit: the patient** (``UNITS``). Every method has one point per patient: a pair
   method (STARE, DeeperHistReg, PALOM) the sums over the patient's rounds that the
   scorer made (``score.summarise_resources``: CPU-hours, GB-hours, GPU-hours, wall), its
   peak the largest of its rounds; a per-patient method its launch. Per-round points
   exist only for fig. 2c (``round_points``), and there every method is per round or is
   on a facet of its own.
2. **Common patients first, survivors second** (``common_set``). The common set of a
   dataset is computed ONCE, over every row the paper compares (``compared_rows``: the
   primary rows and the best rows), by the scorer's own rule
   (``score.common_patients``, strict). Every cost column has ``<stem>_common_*`` (the
   headline) and ``<stem>_survivors_*`` (this row's own complete patients) beside it,
   and every row the counts ``n_patients_common``, ``n_patients_survivors``,
   ``n_capped``, ``n_failed`` and ``n_excluded``.
3. **Rows** (``ROW_KINDS``): the PRIMARY row of each family is the cost the paper
   draws (filled); its best-Dice row is the ceiling (hollow, "best (ceiling)").
   ``local_4cpu`` is STARE at 4 CPUs in one job (``variants.STARE_LOCAL4``), and
   ``cpu_twin`` a GPU row's CPU twin (the same configuration on the CPU).
4. **Clock and unit are columns** (``CLOCKS``): ``whole_process`` (one process in one
   4-CPU job: every method but STARE's stage jobs), ``local_4cpu`` (STARE's stages
   inside one 4-CPU job), ``critical_path_unlimited`` (STARE's stage jobs: COARSE + the
   longest tile + SOLVE, every tile at once, no queue) and ``with_queue`` (STARE's stage
   jobs from the first COARSE start to the end of SOLVE, queue waits in it).
5. **Uncertainty**: every median carries the bootstrap 95% interval over patients of
   ``score.bootstrap_interval`` (2000 resamples, fixed seed), the accuracy tables' own.
6. **Nothing silent**: a patient whose figure is missing is counted
   (``n_<stem>_missing``), a lower bound is flagged (``lower_bound``), and a peak the
   process could only read as ``rusage`` (a lower bound, ``resources.py``) is flagged
   ``peak_rss_comparable = False``, counted and left out of every peak comparison; a
   round that registered and was scored as an imputed failure is costed (it used the
   resources) and counted (``n_imputed_costed``).
"""

from __future__ import annotations

import math
import statistics

WHOLE, LOCAL4, CRITICAL, QUEUE = CLOCKS = (
    "whole_process",
    "local_4cpu",
    "critical_path_unlimited",
    "with_queue",
)
PATIENT, ROUND = UNITS = ("patient", "round")
PRIMARY, BEST, LOCAL_4CPU, CPU_TWIN, OTHER = ROW_KINDS = (
    "primary",
    "best",
    "local_4cpu",
    "cpu_twin",
    "other",
)
COMPLETE, CAPPED, FAILED, EXCLUDED, NOT_RUN = STATUSES = (
    "complete",
    "capped",
    "failed",
    "excluded",
    "not_run",
)
RUSAGE = "rusage"  # resources.py: a peak from getrusage alone, a LOWER bound
POLLED = "polled (may undercount)"  # resources.POLLED
NA = "NA"

# The cost of one patient, per stem: the column of ``summary_resources.csv`` it is read
# from (None: made here), its unit, and what it is. The wall stems name their clock.
STEMS = {
    "cpu_h_used": ("cpu_h_used", "CPU-h", "CPU time the good attempts used"),
    "cpu_h_reserved": ("cpu_h_reserved", "CPU-h", "allocated CPUs x duration, good attempts"),
    "total_cpu_h_used": ("total_cpu_h_used", "CPU-h", "good + wasted attempts, used"),
    "total_cpu_h_reserved": ("total_cpu_h_reserved", "CPU-h", "good + wasted, reserved"),
    "wasted_cpu_h_reserved": ("wasted_cpu_h_reserved", "CPU-h", "the wasted attempts' reservation"),
    "gb_h_used": ("gb_h_used", "GB-h", "peak x duration of the good attempts"),
    "gb_h_reserved": ("gb_h_reserved", "GB-h", "the memory REQUEST x duration (the request policy: "
                      "STARE at the pipeline's per-process requests, the others on the 16 GB-floor ladder)"),
    "wasted_gb_h_reserved": ("wasted_gb_h_reserved", "GB-h", "the wasted attempts' request"),
    "total_gb_h_reserved": ("total_gb_h_reserved", "GB-h", "good + wasted request"),
    "gpu_h": ("gpu_h", "GPU-h", "GPU-hours"),
    "cpu_h_reserved_scheduler": ("cpu_h_reserved_scheduler", "CPU-h", "what the cluster held: "
                                 "sacct AllocCPUS x ElapsedRaw of every job of the good attempts"),
    "gb_h_reserved_scheduler": ("gb_h_reserved_scheduler", "GB-h", "what the cluster held: "
                                "sacct ReqMem x ElapsedRaw of the same jobs"),
    "wall_s": (None, "s", "the registration's wall time on the row's clock (shared nodes)"),
    "elapsed_with_queue_s": ("elapsed_with_queue_s", "s", "STARE's stage jobs, queue waits in"),
    "peak_rss_gb": ("peak_rss_gb", "GB", "the memory one job needs: the launch's peak, the "
                    "largest round's, or STARE's largest stage job"),
    "peak_hwm_gb": ("peak_hwm_gb", "GB", "sum of VmHWM over the job's processes (telemetry)"),
    "gpu_peak_gb": ("gpu_peak_gb", "GB", "the GPU's peak memory"),
    "peak_largest_job_gb": ("peak_largest_job_gb", "GB", "STARE's stage jobs: the largest "
                            "single good attempt (what a node must give one job)"),
    "peak_concurrent_gb": ("peak_concurrent_gb", "GB", "STARE's stage jobs: the most memory "
                           "the jobs of one round held at once (sum of the peaks of "
                           "overlapping attempts)"),
    "rung_gb": ("mem_requested_gb", "GB", "the memory rung the patient ended on"),
    "convert_s": (None, "s", "the method's own conversion outside its clock"),
}
# a wall stem's clock is the row's own unless named here
STEM_CLOCK = {"elapsed_with_queue_s": QUEUE}
# the stems that are a peak: compared only where the instrument is comparable
PEAKS = ("peak_rss_gb", "peak_hwm_gb", "peak_largest_job_gb", "peak_concurrent_gb")


# ── reading ──────────────────────────────────────────────────────────────────
def num(v):
    if v is None or isinstance(v, bool):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def flag(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def missing(v):
    return v is None or v == "" or v == NA or (isinstance(v, float) and math.isnan(v))


def finite(values):
    return [f for f in (num(v) for v in values) if math.isfinite(f)]


def median(values):
    vals = finite(values)
    return float(statistics.median(vals)) if vals else math.nan


def interval(values):
    """The bootstrap 95% interval of the median over patients (``score.bootstrap_interval``,
    the accuracy tables' helper: 2000 resamples, fixed seed). NaN with fewer than two."""
    from .. import score

    return score.bootstrap_interval(finite(values))


def staged(r):
    """A row of STARE's stage jobs (``n_jobs`` present)."""
    return not missing(r.get("n_jobs"))


def clock_of(row_name, r):
    """The clock of a row's wall time (rule 4)."""
    from .. import variants

    if row_name in getattr(variants, "LOCAL_4CPU", ()):
        return LOCAL4
    return CRITICAL if staged(r) else WHOLE


def wall_of(r):
    """The wall time of a summary or resource row on its own clock."""
    return num(r.get("wall_unlimited_parallel_s") if staged(r) else r.get("register_s"))


# ── the rows the paper compares, and the common patients ─────────────────────
def compared_rows(best_rows, primary):
    """The rows the common set is computed over (rule 2): every family's primary row,
    then every best row, once each, in that order."""
    out = [p for p in primary if not missing(p)]
    for r in best_rows:
        b = r.get("best_row")
        if not missing(b) and b not in out:
            out.append(b)
    return out


def typed(cost_rows):
    """``summary_resources.csv`` rows typed as ``score.common_patients`` reads them."""
    out = []
    for r in cost_rows:
        n_rounds = num(r.get("n_rounds"))
        out.append({
            "dataset": r.get("dataset"),
            "group": r.get("group"),
            "method": r.get("method"),
            "launched": flag(r.get("launched")) if not missing(r.get("launched")) else True,
            "complete": flag(r.get("complete")),
            "n_ok": int(num(r.get("n_ok"))) if math.isfinite(num(r.get("n_ok"))) else 0,
            "n_rounds": int(n_rounds) if math.isfinite(n_rounds) else 0,
            "n_lost_not_method": int(num(r.get("n_lost_not_method")))
            if math.isfinite(num(r.get("n_lost_not_method")))
            else 0,
        })
    return out


def common_set(cost_rows, compared, dataset):
    """The patients every compared row completed (``score.common_patients``, strict):
    ONE set per dataset, the same for every row."""
    from .. import score

    rows = [r for r in typed(cost_rows) if r["dataset"] == dataset]
    if not rows or not compared:
        return set()
    got = score.common_patients(rows, list(compared), score.STRICT)
    return set(got.get(dataset, (set(), []))[0])


def status_of(r, res_rows):
    """``complete``, ``capped`` (a round the ladder could not climb for: the method's),
    ``excluded`` (rounds lost for a cause that is not the method's and no failure of its
    own), ``not_run`` or ``failed``."""
    if flag(r.get("complete")):
        return COMPLETE
    if not missing(r.get("launched")) and not flag(r.get("launched")):
        return NOT_RUN
    mine = [x for x in res_rows if x.get("method") == r.get("method") and x.get("group") == r.get("group")]
    if any(x.get("cause") == "capped" for x in mine):
        return CAPPED
    n_ok, lost, n = (num(r.get(k)) for k in ("n_ok", "n_lost_not_method", "n_rounds"))
    if math.isfinite(lost) and lost > 0 and n_ok + lost == n:
        return EXCLUDED
    return FAILED


# ── one patient ──────────────────────────────────────────────────────────────
def _convert(rounds, adapter):
    from .. import cost

    clock = cost.CLOCK.get(adapter, {})
    if clock.get("convert", cost.NONE) in (cost.NONE, cost.NOT_SEPARABLE):
        return math.nan
    vals = finite(x.get("method_convert_s") for x in rounds)
    if not vals:
        return math.nan
    return max(vals) if clock.get("per") == "patient" else float(sum(vals))


def patient_values(r, rounds):
    """``{stem: value}`` of one patient (a ``summary_resources.csv`` row) and its rounds
    (``resources.csv``): rule 1."""
    adapter = next((x.get("adapter") for x in rounds if x.get("adapter")), "")
    got = {}
    for stem, (col, _, _) in STEMS.items():
        if stem == "wall_s":
            got[stem] = wall_of(r)
        elif stem == "convert_s":
            got[stem] = _convert([x for x in rounds if flag(x.get("ok"))], adapter)
        else:
            got[stem] = num(r.get(col))
    if not staged(r):
        got["elapsed_with_queue_s"] = math.nan
        got["peak_concurrent_gb"] = math.nan
    return got


def lower_bound(r, rounds):
    """Why a patient's times and CPU-hours are a lower bound, or ``""``."""
    why = []
    if num(r.get("unmeasured_attempts")) > 0:
        why.append("unmeasured_attempts")
    if num(r.get("unmeasured_records")) > 0:
        why.append("unmeasured_records")
    if POLLED in str(r.get("cpu_instrument") or ""):
        why.append("polled_cpu")
    if num(r.get("wasted_unknown")) > 0:
        why.append("wasted_unknown")
    return "|".join(why)


def peak_instrument(rounds):
    """The peak's instruments over a patient's ok rounds (``rss_method``), '|'-joined."""
    seen = sorted({str(x.get("rss_method") or "") for x in rounds if flag(x.get("ok"))} - {""})
    return "|".join(seen)


# ── the summary columns of one row ───────────────────────────────────────────
def summary_columns(points, row, common, clock):
    """The cost columns of ``method_summary`` for one row, from its patient points
    (``points``: the ``cost_points`` rows of unit ``patient`` of this row): per stem the
    median and its interval on the COMMON patients and on the row's own complete ones,
    with the clock and unit, plus the counts of rule 2 and rule 6."""
    mine = [p for p in points if p["row"] == row and p["unit_kind"] == PATIENT]
    done = [p for p in mine if p["status"] == COMPLETE]
    shared = [p for p in done if p["unit_id"] in common]
    out = {
        "n_patients_common": len(shared),
        "n_patients_survivors": len(done),
        "n_patients_all": len(mine),
        "n_capped": sum(p["status"] == CAPPED for p in mine),
        "n_failed": sum(p["status"] == FAILED for p in mine),
        "n_excluded": sum(p["status"] == EXCLUDED for p in mine),
        "n_lower_bound": sum(bool(p.get("lower_bound")) for p in shared),
        "n_peak_not_comparable": sum(p.get("peak_rss_comparable") is False for p in shared),
        "n_imputed_costed": int(sum(num(p.get("n_imputed_costed")) or 0 for p in shared
                                    if math.isfinite(num(p.get("n_imputed_costed"))))),
        "clock": clock,
        "unit": PATIENT,
    }
    for stem in STEMS:
        for which, rows in (("common", shared), ("survivors", done)):
            vals = [p.get(stem) for p in rows
                    if not (stem in PEAKS and p.get("peak_rss_comparable") is False)]
            low, high = interval(vals)
            out[f"{stem}_{which}_med"] = median(vals)
            out[f"{stem}_{which}_ci_low"] = low
            out[f"{stem}_{which}_ci_high"] = high
            out[f"n_{stem}_{which}_missing"] = len(rows) - len(finite(vals))
        out[f"{stem}_clock"] = STEM_CLOCK.get(stem, clock)
        out[f"{stem}_unit"] = PATIENT
    return out


def summary_column_specs():
    """``(name, unit, what)`` of every column ``summary_columns`` makes (the schema)."""
    out = [
        ("n_patients_common", "count", "patients every compared row completed (rule 2): "
         "the headline cost columns are over these"),
        ("n_patients_survivors", "count", "patients THIS row completed: the *_survivors columns"),
        ("n_patients_all", "count", "patients this row has a summary line for"),
        ("n_capped", "count", "patients with a round the memory ladder could not climb for (the method's)"),
        ("n_failed", "count", "patients with a round that failed for the method's own cause"),
        ("n_excluded", "count", "patients with rounds lost for a cause that is not the method's"),
        ("n_lower_bound", "count", "common patients whose times and CPU-hours are lower bounds"),
        ("n_peak_not_comparable", "count", "common patients whose peak is a rusage lower bound "
         "(left out of every peak median)"),
        ("n_imputed_costed", "count", "rounds of the common patients that registered, were "
         "scored as imputed failures, and are costed (they used the resources)"),
        ("clock", "", "the clock of the row's times: " + ", ".join(CLOCKS)),
        ("unit", "", "patient: one value per patient"),
    ]
    for stem, (_, unit, what) in STEMS.items():
        for which, who in (("common", "the common patients (the headline)"),
                           ("survivors", "this row's own complete patients")):
            out += [
                (f"{stem}_{which}_med", unit, f"{what}: median over {who}"),
                (f"{stem}_{which}_ci_low", unit, "bootstrap 95% interval of that median, low"),
                (f"{stem}_{which}_ci_high", unit, "bootstrap 95% interval of that median, high"),
                (f"n_{stem}_{which}_missing", "count", "of those patients, how many have no figure"),
            ]
        out += [(f"{stem}_clock", "", "the clock of the stem (rule 4)"),
                (f"{stem}_unit", "", "the unit of the stem (patient)")]
    return out


# ── the points: one per patient for every row, and per round for fig. 2c ──────
POINT_STEMS = tuple(STEMS)


def row_kinds(row, primary, best, local4=(), twins=()):
    """``(row_kind, is_primary, is_best)`` of a result name (rule 3)."""
    is_p, is_b = row in primary, row in best
    kind = (PRIMARY if is_p else BEST if is_b else LOCAL_4CPU if row in local4
            else CPU_TWIN if row in twins else OTHER)
    return kind, is_p, is_b


def patient_points(row, family, cost_rows, res_rows, common, kinds, imputed, gb_of=None):
    """One ``cost_points`` row per patient of ``row`` in ``summary_resources.csv``,
    whatever became of it (``status``): only ``complete`` patients enter a median or a
    fit, the others are counted. ``imputed`` is ``{(row, patient): rounds that
    registered and were scored as imputed failures}``; ``gb_of(rounds)`` the input
    size in GB, or None."""
    out = []
    kind, is_p, is_b = kinds
    for r in cost_rows:
        if r.get("method") != row:
            continue
        patient = r.get("group")
        rounds = [x for x in res_rows if x.get("method") == row and x.get("group") == patient]
        vals = patient_values(r, rounds)
        status = status_of(r, res_rows)
        inst = peak_instrument(rounds)
        rec = {
            "family": family,
            "row": row,
            "row_kind": kind,
            "is_primary": is_p,
            "is_best": is_b,
            "unit_kind": PATIENT,
            "unit": PATIENT,
            "unit_id": patient,
            "status": status,
            "in_common": patient in common,
            "clock": clock_of(row, r),
            "input_gb": gb_of(rounds) if gb_of else None,
            "input_gpx": num(r.get("moving_gpx")),
            "lower_bound": lower_bound(r, rounds),
            "peak_instrument": inst,
            "peak_rss_comparable": RUSAGE not in inst.split("|")
            and not (r.get("peak_rss_comparable") is not None
                     and not missing(r.get("peak_rss_comparable"))
                     and not flag(r.get("peak_rss_comparable"))),
            "n_imputed_costed": imputed.get((row, patient), 0),
            "wasted_attempts": num(r.get("wasted_attempts")),
            "device": r.get("device") or None,
            "gpu_model": r.get("gpu_model") or None,
            "cpu_model": "|".join(sorted({str(x.get("cpu_model")) for x in rounds
                                          if not missing(x.get("cpu_model"))})) or None,
            **vals,
        }
        out.append(rec)
    return out


def round_points(row, family, res_rows, excluded, kinds, gb_of=None):
    """fig. 2c's per-round points: every ok round of a pair method; a per-patient
    method's round only where its launch timed each round (``register_s_per_round_from
    = info``: ``info.register_s_round``), with its CPU-hours split in proportion to its
    time and no peak (one launch has one peak). ``(points, rounds of a per-patient
    method with no per-round figure)``."""
    out, without = [], 0
    kind, is_p, is_b = kinds
    for x in res_rows:
        if x.get("method") != row or not flag(x.get("ok")) or (x.get("case_id"), row) in excluded:
            continue
        group = x.get("unit") == PATIENT
        if group and x.get("register_s_per_round_from") != "info":
            without += 1
            continue
        if group:
            launch_s, round_s = num(x.get("register_s")), num(x.get("register_s_per_round"))
            share = round_s / launch_s if launch_s > 0 else math.nan
            wall, cpu, peak = round_s, num(x.get("cpu_h_used")) * share, math.nan
        else:
            wall, cpu, peak = wall_of(x), num(x.get("cpu_h_used")), num(x.get("peak_rss_gb"))
        out.append({
            "family": family, "row": row, "row_kind": kind, "is_primary": is_p,
            "is_best": is_b, "unit_kind": ROUND, "unit": ROUND,
            "unit_id": x.get("case_id"), "status": COMPLETE, "in_common": None,
            "clock": clock_of(row, x),
            "input_gb": gb_of([x]) if gb_of else None,
            "input_gpx": num(x.get("moving_gpx")),
            "wall_s": wall, "cpu_h_used": cpu, "peak_rss_gb": peak,
            "peak_instrument": str(x.get("rss_method") or "") or None,
            "peak_rss_comparable": x.get("rss_method") != RUSAGE,
            "device": x.get("device") or None, "gpu_model": x.get("gpu_model") or None,
            "cpu_model": x.get("cpu_model") or None,
        })
    return out, without

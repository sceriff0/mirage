#!/usr/bin/env bash
# Drive the REAL-SAMPLE arm benchmark: one pipeline launch per arm_plan.csv row.
#
# Usage:
#   benchmarks/run_arms.sh <arm_plan.csv> <real_input.csv> <results_root> [extra nextflow args...]
#
# Example (cluster):
#   ARMS_PROFILE="singularity,ieo" ARMS_CONCURRENCY=3 \
#     benchmarks/run_arms.sh arm_plan.csv real_input.csv arm_results
#
# The sibling of run_sweep.sh, and deliberately simpler in one way: the
# samplesheet is GIVEN (your real slides) rather than synthesized from a matrix,
# so there is no cell resolution and no channel-name construction here.
#
# WHAT IT WRITES — exactly the tree ihc_method/code/registration_arms.R reads:
#
#     <results_root>/<arm>/<patient>/qc/registration/*_seg_qc.json
#     <results_root>/<arm>/<patient>/registered/summary/*.csv     (VALIS arms)
#     <results_root>/arms.csv                                     (build_arm_plan.py)
#
# That is just `--outdir <results_root>/<arm>`; Layout.patientDir() is already
# `<outdir>/<patient_id>/<kind>`. Nothing is repackaged, so the consumer reads
# run_qc.R's readers unchanged.

PLAN="${1:?arm_plan.csv}"
INPUT="${2:?real input.csv}"
ROOT="${3:?results root}"
shift 3
EXTRA=("$@")

mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
START_DIR="$PWD"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$INPUT" = /* ]] || INPUT="$START_DIR/$INPUT"

# Pin Nextflow to the supported 25.04 line, as run_sweep.sh does. NF 26.x dropped
# the automatic lib/*.groovy class loading this pipeline relies on
# (ParamUtils/CsvUtils) and rejects the CLI boolean forms used below.
export NXF_VER="${NXF_VER:-25.04.7}"

PROFILE="${ARMS_PROFILE:-docker}"
CONCURRENCY="${ARMS_CONCURRENCY:-1}"

# benchmark.config turns on enable_trace + enable_size_logs, which is what makes
# every arm contribute cost rows to measurements.csv alongside its QC. It costs
# nothing on the QC arms and is the whole point of the compute arm.
BENCH_CONF="$PIPELINE_DIR/benchmarks/configs/benchmark.config"

# Absolutize any -c/-config/-params-file in the pass-through args: each run is
# launched from its own directory (isolated .nextflow/ so parallel Nextflow heads
# do not fight over .nextflow/history on NFS/beegfs), which breaks relative paths.
EXTRA_ABS=(); _i=0
while (( _i < ${#EXTRA[@]} )); do
  _a="${EXTRA[$_i]}"; EXTRA_ABS+=("$_a")
  case "$_a" in
    -c|-C|-config|-params-file)
      _i=$((_i + 1)); _v="${EXTRA[$_i]:-}"
      [[ -n "$_v" && "$_v" != /* && -e "$_v" ]] && _v="$(cd "$(dirname "$_v")" && pwd)/$(basename "$_v")"
      EXTRA_ABS+=("$_v") ;;
  esac
  _i=$((_i + 1))
done

header=$(head -n1 "$PLAN" | tr -d '\r')
IFS=',' read -r -a cols <<< "$header"

col_index() {
  local name="$1" i
  for i in "${!cols[@]}"; do
    [[ "${cols[$i]}" == "$name" ]] && echo "$i" && return
  done
  echo -1
}

# col_val <name> <vals...>. build_arm_plan.py guarantees no value contains a comma
# (labels live only in arms.csv, which R reads), so this naive IFS split is safe —
# the same contract run_sweep.sh relies on. benchmarks/tests/test_build_arm_plan.py
# asserts it, because a comma sneaking into a value would shift every later column
# on that row and silently launch the wrong configuration.
col_val() {
  local name="$1"; shift
  local idx; idx=$(col_index "$name")
  [[ "$idx" -lt 0 ]] && { echo ""; return; }
  local vals=("$@")
  echo "${vals[$idx]:-}"
}

# add_param <name> <value>: record `name=value` ONLY when value is non-empty.
#
# The blank check is load-bearing, not defensive. A tiled/STARE arm has NO
# memory_mode and NO reg_micro_reg — those are VALIS-only params — so those cells
# are legitimately blank. Passing an empty value would set the param to an empty
# string rather than leave it at its default, and schema validation would reject
# it (or worse, accept it and register at a configuration nobody chose).
#
# These are collected as name=value pairs and written to a -params-file, NOT
# passed as `--name value`. Nextflow 26 delivers every CLI param as a String and
# nextflow_schema.json declares real types, so the command-line form fails
# validateParameters() before a single task runs:
#     * --reg_qc (1): Value is [string] but should be [integer]
# 25.04.7 accepts it and 26.04.6 does not. JSON carries types; the coercion is
# benchmarks/params_json.py, which reads each param's declared type out of the
# schema rather than guessing from the text.
PAIRS=()
add_param() { if [[ -n "${2:-}" ]]; then PAIRS+=("$1=$2"); fi; }

# RESUME_RUN (env, optional): the run_id of a BASE arm whose Nextflow session this launch
# resumes. Set for arm_kind=registration_qc rows -- the QC instrument crosses -- which
# differ from their base arm only in params.seg_method or params.seg_qc_pairing. Both
# reach only the QC chain (SEG_QC_SEGMENT via SegBackends, WARP_SEG_QC via ext.args), so
# with the base arm's work dir and session, REGISTER and everything before it are served
# from the cache and only the QC tasks run again -- publishing into the cross arm's OWN
# --outdir. The launch happens INSIDE the base arm's launch directory, because that is
# where `.nextflow/history` and `.nextflow/cache/<session>` live and -resume looks them
# up relative to the launch directory; the session id is read out of the history by run
# name, so a later run in the same directory cannot be picked up by accident. A resume
# that misses is correct and merely slow (it re-registers), never wrong.
launch() {                       # launch <run_id> <arm> <input> <outdir> <name=value...>
  local run_id="$1" arm="$2" in_csv="$3" outdir="$4"; shift 4
  local run_pairs=("$@")
  local rundir="$ROOT/.launch/$run_id"
  local resume_args=()
  if [[ -n "${RESUME_RUN:-}" ]]; then
    rundir="$ROOT/.launch/$RESUME_RUN"
    if [[ ! -d "$rundir/work" ]]; then
      echo "[$run_id] SKIP: base arm '$RESUME_RUN' has no work dir at $rundir/work" >&2
      return 1
    fi
    local sid=""
    if [[ -f "$rundir/.nextflow/history" ]]; then
      # history columns: timestamp, duration, run name, status, revision, SESSION ID, command
      sid=$(awk -F'\t' -v n="arms-$RESUME_RUN" '$3 == n { s = $6 } END { print s }' "$rundir/.nextflow/history")
    fi
    if [[ -n "$sid" ]]; then resume_args=(-resume "$sid"); else resume_args=(-resume); fi
    echo "[$run_id] resumes base arm $RESUME_RUN (session ${sid:-latest}); only the QC chain should run"
  fi
  mkdir -p "$rundir" "$outdir" "$outdir/trace"
  # Typed params as JSON — see the add_param comment above for why this cannot be
  # a list of --name value flags on Nextflow 26. Named per run_id: a resumed cross arm
  # shares its base arm's launch directory and must not overwrite the base's file.
  local run_params="$rundir/params.${run_id}.json"
  if ! (cd "$PIPELINE_DIR" && python3 -m benchmarks.params_json --out "$run_params" \
          ${run_pairs[@]+"${run_pairs[@]}"}); then
    echo "[$run_id] SKIP: could not type its parameters against nextflow_schema.json" >&2
    return 1
  fi
  echo "[$run_id] arm=$arm -> $outdir"
  (
    cd "$rundir"
    nextflow -q run "$PIPELINE_DIR" \
      -profile "$PROFILE" \
      -c "$BENCH_CONF" \
      -work-dir "$rundir/work" \
      -name "arms-$run_id" \
      "${resume_args[@]+"${resume_args[@]}"}" \
      -params-file "$run_params" \
      --input "$in_csv" \
      --outdir "$outdir" \
      --trace_dir "$outdir/trace" \
      "${EXTRA_ABS[@]+"${EXTRA_ABS[@]}"}" \
      > "$outdir/nextflow.stdout.log" 2> "$outdir/nextflow.stderr.log" \
      || {
        # Nextflow writes most run-level errors (validation, missing input, a failed
        # task's .command.err excerpt) to STDOUT, not stderr -- a message naming only
        # the stderr log sends you to a file containing just the version banner.
        # Show both, and name the .nextflow.log, which has the rest.
        echo "[$run_id] FAILED" >&2
        echo "----- last 25 lines of stdout ($outdir/nextflow.stdout.log) -----" >&2
        tail -n 25 "$outdir/nextflow.stdout.log" >&2 2>/dev/null
        echo "----- last 15 lines of stderr ($outdir/nextflow.stderr.log) -----" >&2
        tail -n 15 "$outdir/nextflow.stderr.log" >&2 2>/dev/null
        echo "----- full log: $rundir/.nextflow.log -----" >&2
      }
  )
}

# filtered_sheet <patient> <dest>: the real samplesheet restricted to one patient.
# There is no --only_patient param; restricting the cohort means writing a smaller
# samplesheet. Header + every row whose first field matches.
filtered_sheet() {
  local pat="$1" dest="$2"
  head -n1 "$INPUT" > "$dest"
  awk -F',' -v p="$pat" 'NR>1 {gsub(/\r/,"")} NR>1 && $1==p' "$INPUT" >> "$dest"
  if [[ $(wc -l < "$dest") -le 1 ]]; then
    echo "ERROR: no rows for patient '$pat' in $INPUT" >&2; return 1
  fi
}

# ---------------------------------------------------------------------------
# PASS ORDER IS A DEPENDENCY, NOT A PREFERENCE.
# A segmentation arm resumes from <root>/<from_arm>/csv/registered.csv, which
# does not exist until that registration arm has finished. Running the passes in
# plan order (or all concurrently) would race: the segmentation arms would fail
# on a missing checkpoint, and because conf/modules.config's errorStrategy has an
# 'ignore' branch, a partial tree can still look like a completed run.
# ---------------------------------------------------------------------------
pids=()
reap() {                          # block until under the concurrency cap
  local rc
  while (( ${#pids[@]} >= CONCURRENCY )); do
    # The status is RECORDED, not discarded. Every launched child already prints its
    # own `[run_id] FAILED`, so reaping a non-zero pid must not abort the pass -- but
    # `|| true` here meant the reap itself said nothing, and a run that died before
    # printing anything (an sbatch that never started, an OOM-killed subshell) reaped
    # in silence. tests/test_no_swallowed_failures.py forbids the `|| true` form.
    rc=0
    wait "${pids[0]}" || rc=$?
    (( rc == 0 )) || echo "[reap] pid ${pids[0]} exited $rc" >&2
    pids=("${pids[@]:1}")
  done
}

# launch_row <csv row values...>: assemble one plan row's flags and launch it, in the
# FOREGROUND. The pass functions below decide what runs concurrently with what.
launch_row() {
    local vals=("$@")
    local kind; kind=$(col_val arm_kind "${vals[@]}")
    local run_id arm start stop from_arm from_csv only_patient
    run_id=$(col_val run_id "${vals[@]}")
    arm=$(col_val arm "${vals[@]}")
    start=$(col_val start "${vals[@]}")
    stop=$(col_val stop "${vals[@]}")
    from_arm=$(col_val from_arm "${vals[@]}")
    from_csv=$(col_val from_csv "${vals[@]}")
    only_patient=$(col_val only_patient "${vals[@]}")

    # EXTERNAL ARMS ARE NOT NEXTFLOW LAUNCHES. Dispatched here, before any --flag is
    # assembled, because nothing in these rows is a pipeline param -- see
    # build_arm_plan.py::_external_arms. They run in their own pass (see the pass list at
    # the bottom), after registration, because they reuse a registration arm's published
    # QC nuclei.
    if [[ "$kind" == "external" ]]; then
      local ext_tool ext_from_arm ext_tile ext_overlap ext_shift preproc_csv
      ext_tool=$(col_val ext_tool "${vals[@]}")
      ext_from_arm=$(col_val ext_from_arm "${vals[@]}")
      ext_tile=$(col_val ext_tile_size "${vals[@]}")
      ext_overlap=$(col_val ext_overlap "${vals[@]}")
      ext_shift=$(col_val ext_max_shift_um "${vals[@]}")
      if [[ "$ext_tool" != "ashlar" ]]; then
        echo "[$run_id] SKIP: unknown ext_tool '$ext_tool'" >&2; return 1
      fi
      preproc_csv="$ROOT/$from_arm/csv/${from_csv}.csv"
      if [[ ! -f "$preproc_csv" ]]; then
        echo "[$run_id] SKIP: $preproc_csv missing — arm '$from_arm' did not complete" >&2
        return 1
      fi
      if [[ ! -d "$ROOT/$ext_from_arm" ]]; then
        echo "[$run_id] SKIP: $ROOT/$ext_from_arm missing — no QC nuclei to score against" >&2
        return 1
      fi
      mkdir -p "$ROOT/$arm"
      if ! "$PIPELINE_DIR/benchmarks/run_ashlar_arm.sh" "$ROOT" "$arm" "$ext_from_arm" "$preproc_csv" \
            "$ext_tile" "$ext_overlap" "$ext_shift" \
            > "$ROOT/$arm/ashlar.stdout.log" 2> "$ROOT/$arm/ashlar.stderr.log"; then
        echo "[$run_id] FAILED" >&2
        tail -n 25 "$ROOT/$arm/ashlar.stderr.log" >&2 2>/dev/null
        return 1
      fi
      return 0
    fi

    PAIRS=()
    add_param start                "$start"
    add_param stop                 "$stop"
    add_param seg_method           "$(col_val seg_method "${vals[@]}")"
    add_param reg_qc               "$(col_val reg_qc "${vals[@]}")"
    add_param registration_method  "$(col_val registration_method "${vals[@]}")"
    add_param memory_mode          "$(col_val memory_mode "${vals[@]}")"
    add_param reg_micro_reg        "$(col_val reg_micro_reg "${vals[@]}")"
    # tiled-only, blank on every other arm -- the add_param blank-guard is what keeps a
    # VALIS arm from ever receiving --reg_tiled_mode "", which the schema enum rejects.
    add_param reg_tiled_mode       "$(col_val reg_tiled_mode "${vals[@]}")"
    add_param reg_tiled_gate_tre   "$(col_val reg_tiled_gate_tre "${vals[@]}")"
    add_param seg_qc_pairing       "$(col_val seg_qc_pairing "${vals[@]}")"
    # THE THREE reg_ashlar_* FLAGS ARE GONE. ashlar stopped being a pipeline backend at
    # :fire: 6a54479, so nextflow.config declares none of them and the schema would reject
    # all three. The external ashlar baseline is an arm_kind='external' row instead, and
    # never reaches this flag block at all -- it is dispatched below, before in_csv is even
    # resolved. Note that every column it carries is ext_-prefixed for exactly that reason:
    # no ext_* name can collide with one of the add_param calls above.

    local in_csv="$INPUT"
    if [[ -n "$from_arm" ]]; then
      # Which checkpoint depends on WHERE this arm resumes: a registration arm
      # picks up csv/preprocessed.csv from the one shared preprocessing run, a
      # segmentation arm picks up csv/registered.csv from the arm it was told to
      # follow. Layout.checkpointCsvName() derives both names from the step
      # vocabulary, which is why they are the step name minus the "-ing".
      in_csv="$ROOT/$from_arm/csv/${from_csv}.csv"
      if [[ ! -f "$in_csv" ]]; then
        echo "[$run_id] SKIP: $in_csv missing — arm '$from_arm' did not complete" >&2
        return 1
      fi
    elif [[ -n "$only_patient" ]]; then
      in_csv="$ROOT/.launch/$run_id.samplesheet.csv"
      mkdir -p "$(dirname "$in_csv")"
      filtered_sheet "$only_patient" "$in_csv" || return 1
    fi

    RESUME_RUN="$(col_val resume_run "${vals[@]}")" \
      launch "$run_id" "$arm" "$in_csv" "$ROOT/$arm" ${PAIRS[@]+"${PAIRS[@]}"}
}

# run_pass <arm_kind>: every row of that kind, up to CONCURRENCY at once.
run_pass() {
  local want_kind="$1"
  if [[ "$want_kind" == "registration_qc" ]]; then run_qc_pass; return; fi
  while IFS=',' read -r -a vals; do
    [[ "$(col_val arm_kind "${vals[@]}")" == "$want_kind" ]] || continue
    reap
    launch_row "${vals[@]}" &
    pids+=($!)
  done < <(tail -n +2 "$PLAN" | tr -d '\r')
}

# run_qc_pass: the QC instrument crosses, ONE CHAIN PER BASE ARM.
#
# Every registration_qc row resumes its base arm's Nextflow session (see launch()). Two
# runs resuming the SAME session at once fight over .nextflow/cache/<session>/db/LOCK and
# one of them dies ("Unable to acquire lock on session") -- measured on this pipeline when
# the documented-command launch leg was first parallelised. So the rows are grouped by
# resume_run and each group runs SEQUENTIALLY in one background chain; the chains of
# different base arms run concurrently, still capped by CONCURRENCY. A failed link is
# reported and the chain continues: the next cross arm resumes the same base session
# and is independent of the one that failed.
run_qc_pass() {
  local kind_col resume_col sorted
  kind_col=$(( $(col_index arm_kind) + 1 ))
  resume_col=$(( $(col_index resume_run) + 1 ))
  if (( resume_col == 0 )); then
    echo "[registration_qc] plan has no resume_run column — nothing to run in this pass"
    return
  fi
  mkdir -p "$ROOT/.launch"
  sorted="$ROOT/.launch/_registration_qc.rows"
  tail -n +2 "$PLAN" | tr -d '\r' \
    | awk -F, -v k="$kind_col" '$k == "registration_qc"' \
    | sort -t, -k"$resume_col,$resume_col" -s > "$sorted"

  local base="" line b
  local rows=()
  start_chain() {
    reap
    (
      for line in "${rows[@]}"; do
        IFS=',' read -r -a v <<< "$line"
        if ! launch_row "${v[@]}"; then
          echo "[chain $(col_val resume_run "${v[@]}")] $(col_val run_id "${v[@]}") failed; continuing with the next cross arm" >&2
        fi
      done
    ) &
    pids+=($!)
  }
  while IFS= read -r line; do
    IFS=',' read -r -a v <<< "$line"
    b=$(col_val resume_run "${v[@]}")
    if [[ "$b" != "$base" && ${#rows[@]} -gt 0 ]]; then
      start_chain
      rows=()
    fi
    base="$b"
    rows+=("$line")
  done < "$sorted"
  if (( ${#rows[@]} > 0 )); then start_chain; fi
}

# preprocess FIRST: every registration arm resumes from its csv/preprocessed.csv.
# external AFTER registration (it reuses a registration arm's published QC nuclei) and
# BEFORE compute (which is being timed and must not contend for nodes).
# registration_qc AFTER registration: each QC cross arm resumes its base arm's session,
# so the base must have finished. The barrier below is what guarantees that.
for kind in preprocess registration registration_qc external segmentation compute; do
  echo "=== pass: $kind ==="
  run_pass "$kind"
  # Barrier between passes: segmentation needs registration's checkpoint, and the
  # compute arm should not contend with the QC arms for nodes while it is being
  # timed — a cost measurement taken under self-inflicted contention is not the
  # cost of the pipeline.
  # Same as reap: report the status rather than discarding it, so a child that
  # died without printing its own failure is still visible at the barrier.
  for p in "${pids[@]+"${pids[@]}"}"; do
    rc=0
    wait "$p" || rc=$?
    (( rc == 0 )) || echo "[barrier] pid $p exited $rc" >&2
  done
  pids=()
done

echo
echo "All arms finished. Results under $ROOT"
echo "Label manifest: $ROOT/arms.csv (written by build_arm_plan.py --results-root)"
echo
echo "Next: pull the artifacts into ihc_method —"
echo "  benchmarks/pull_to_ihc_method.sh $ROOT ../ihc_method"

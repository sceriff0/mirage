#!/usr/bin/env bash
# Drive a benchmark sweep: one pipeline launch per run_plan.csv row.
# Usage: benchmarks/run_sweep.sh <run_plan.csv> <matrix_manifest.csv> <results_root> [extra nextflow args...]
# Requires bash 4+ (associative arrays). On macOS the system /bin/bash is 3.x;
# install a modern bash (brew install bash) and invoke explicitly if needed.

RUN_PLAN="${1:?run_plan.csv}"
MANIFEST="${2:?matrix_manifest.csv}"
ROOT="${3:?results root}"
shift 3
EXTRA=("$@")

mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"                                   # absolute: each run cd's into $ROOT/<run_id>
START_DIR="$PWD"                                              # relative matrix/config paths resolve here
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pin Nextflow to the supported 25.04 line. NF 26.x dropped the automatic lib/*.groovy class loading
# this pipeline relies on (ParamUtils/CsvUtils), so it fails with "Missing process or function ...".
# Override by exporting NXF_VER yourself.
export NXF_VER="${NXF_VER:-25.04.7}"

# Each concurrent run is launched from its OWN dir (isolated .nextflow/ so parallel heads don't fight
# over .nextflow/history — beegfs/NFS file locks choke otherwise). That `cd` breaks relative paths in
# the pass-through args, so absolutize any -c/-config/-params-file file in EXTRA up front.
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

# tr -d '\r': tolerate CRLF-terminated CSVs (else the last column carries a trailing \r).
header=$(head -n1 "$RUN_PLAN" | tr -d '\r')
IFS=',' read -r -a cols <<< "$header"

# col_index <name>: print the 0-based index of column <name> in $cols[].
col_index() {
  local name="$1" i
  for i in "${!cols[@]}"; do
    [[ "${cols[$i]}" == "$name" ]] && echo "$i" && return
  done
  echo -1
}

# col_val <name> <vals_array_elements...>: extract value for named column.
# Usage: col_val run_id "${vals[@]}"
col_val() {
  local name="$1"; shift
  local vals=("$@")
  local idx
  idx=$(col_index "$name")
  if [[ "$idx" -ge 0 ]]; then echo "${vals[$idx]}"; else echo ""; fi
}

  # Pre-compute manifest column indices (done once, outside the loop).
manifest_header=$(head -n1 "$MANIFEST" | tr -d '\r')
IFS=',' read -r -a mcols <<< "$manifest_header"
manifest_col_index() {
  local name="$1" i
  for i in "${!mcols[@]}"; do
    [[ "${mcols[$i]}" == "$name" ]] && echo "$i" && return
  done
  echo -1
}
# 1-based awk field for the path column (always present).
path_col_idx=$(manifest_col_index "path")
path_awk_field=$((path_col_idx + 1))
moving_col_idx=$(manifest_col_index "moving_paths")
moving_awk_field=$((moving_col_idx + 1))

# SWEEP_CONCURRENCY: how many pipeline runs to launch AT ONCE (default 1 = serial). Each concurrent
# run is its own `nextflow run` orchestrator (isolated -work-dir/-log/-name) that submits its OWN
# process jobs to SLURM, so measurements stay clean (SLURM gives each process a dedicated allocation).
# Set it >1 to parallelise the sweep across the cluster; the launching host just holds N Nextflow JVMs.
CONCURRENCY="${SWEEP_CONCURRENCY:-1}"
pids=()
# One timestamp per launch, so every run this invocation replaces lands under the same
# <root>/.replaced/<timestamp>/ (see the SWEEP_REPLACE block in the loop).
REPLACE_TS="$(date +%Y%m%d-%H%M%S)"

# SWEEP_PROFILE: Nextflow profile(s) applied to EVERY run (default docker). On a cluster set
# SWEEP_PROFILE="singularity,ieo". It is passed as a SINGLE -profile — Nextflow rejects a second
# -profile ("Can only specify option -profile once"), so do NOT also put -profile in the trailing
# args. Extra `-c <config>` in the trailing args IS fine (Nextflow merges multiple -c).
# `docker,local` not `docker`: max_cpus/max_memory have no pipeline default and nf-schema
# refuses a run supplying neither, so an engine profile alone no longer launches. `local`
# is the sizing profile (4 CPU / 16 GB); the cluster sets SWEEP_PROFILE="singularity,ieo".
PROFILE="${SWEEP_PROFILE:-docker,local}"

while IFS=',' read -r -a vals; do
  run_id=$(col_val run_id "${vals[@]}")
  target_px=$(col_val target_px "${vals[@]}")
  n_channels=$(col_val n_channels "${vals[@]}")
  varied_axis=$(col_val varied_axis "${vals[@]}")
  cell_id="px${target_px}_ch${n_channels}"

  # Resolve this cell's reference image from the matrix manifest.
  # gsub(/\r/,""): tolerate CRLF manifests so the last field has no trailing \r.
  img=$(awk -F',' -v c="$cell_id" -v f="$path_awk_field" 'NR>1 {gsub(/\r/,"")} $1==c {print $f}' "$MANIFEST")
  if [[ -z "$img" ]]; then echo "WARN: no matrix cell $cell_id; skipping $run_id" >&2; continue; fi
  # Absolutize (the samplesheet is consumed from inside the per-run cwd).
  [[ "$img" = /* ]] || img="$START_DIR/$img"

  # Resolve moving images: a ';'-separated list (empty when column absent or value empty).
  mov_list=""
  if [[ "$moving_col_idx" -ge 0 ]]; then
    mov_list=$(awk -F',' -v c="$cell_id" -v f="$moving_awk_field" 'NR>1 {gsub(/\r/,"")} $1==c {print $f}' "$MANIFEST")
  fi

  # Number of registration images for this run: 1 reference + (n_reg-1) moving panels.
  # Defaults to 2 (ref + 1 moving) when the axis is absent — the classic pair.
  n_reg=$(col_val n_register_images "${vals[@]}")
  [[ -z "$n_reg" ]] && n_reg=2

  nch="${vals[$(col_index n_channels)]}"
  # Build reference channel string: DAPI|ch1|...|ch{nch-1}
  ref_chans="DAPI"
  for ((i=1; i<nch; i++)); do ref_chans+="|ch${i}"; done

  run_dir="$ROOT/$run_id"
  # A run name is fixed per run_id (-name bench_<run_id>) and Nextflow refuses one its
  # launch dir's history already holds, so a relaunch into an existing root would die
  # per run behind a log dump. Refuse it here by name -- or, with SWEEP_REPLACE=1 (env,
  # opt-in; the sweep's counterpart of run_arms.sh's ARMS_REPLACE), MOVE the previous
  # run dir aside to <root>/.replaced/<timestamp>/<run_id>, never delete it: that is
  # what a subset re-run (build_run_plan.py --only-method / --only) compares against.
  # Sweep runs are independent launches (no resume between them), so there is no
  # cross-of-a-base refusal to mirror here.
  if [[ -f "$run_dir/.nextflow/history" ]] &&
     awk -F'\t' -v n="bench_${run_id}" '$3 == n { f = 1 } END { exit !f }' "$run_dir/.nextflow/history"; then
    if [[ "${SWEEP_REPLACE:-0}" == "1" ]]; then
      replaced="$ROOT/.replaced/$REPLACE_TS"
      mkdir -p "$replaced" && mv "$run_dir" "$replaced/$run_id"
      echo "[$run_id] replaced: previous run dir moved to $replaced/$run_id (kept, not deleted)"
    else
      echo "SKIP: $run_id already ran in $run_dir (bench_${run_id} is in its .nextflow/history);" \
           "relaunch with SWEEP_REPLACE=1 to move it aside to $ROOT/.replaced/<timestamp>/$run_id, or rm -rf $run_dir" >&2
      continue
    fi
  fi
  mkdir -p "$run_dir"
  sheet="$run_dir/samplesheet.csv"
  printf 'patient_id,path_to_file,is_reference,channels\n' > "$sheet"
  printf '%s,%s,true,%s\n' "$cell_id" "$img" "$ref_chans" >> "$sheet"

  # Emit up to (n_reg-1) moving panels, each with a distinct channel set
  # (DAPI|m{panel}_1|...) so the pipeline's duplicate-channel guard accepts them.
  want=$(( n_reg - 1 )); emitted=0
  if [[ -n "$mov_list" ]]; then
    IFS=';' read -r -a movings <<< "$mov_list"
    for ((j=0; j<${#movings[@]} && emitted<want; j++)); do
      panel=$(( j + 1 ))
      mov_chans="DAPI"
      for ((i=1; i<nch; i++)); do mov_chans+="|m${panel}_${i}"; done
      mv="${movings[$j]}"; [[ "$mv" = /* ]] || mv="$START_DIR/$mv"
      printf '%s,%s,false,%s\n' "$cell_id" "$mv" "$mov_chans" >> "$sheet"
      emitted=$(( emitted + 1 ))
    done
  fi
  # Hard guard (paired cells only): if the matrix can't supply the panels this run
  # needs, registration would silently downgrade to fewer images and quietly corrupt
  # the n_register_images scaling data. Skip the run loudly instead — regenerate the
  # matrix from THIS sweep (generate_matrix --sweep is plan-aware and sizes each
  # cell's moving panels to the runs that use it).
  if [[ "$nch" -ge 2 && "$emitted" -lt "$want" ]]; then
    echo "ERROR: $run_id needs $want moving panel(s) for $cell_id but the matrix has only $emitted;" \
         "regenerate the matrix from this sweep — SKIPPING (no silent pair-registration downgrade)." >&2
    continue
  fi

  # Map non-structural columns to --<param>. Skipped columns fall in two groups:
  #   * matrix/registration shape (target_px/n_channels/n_register_images) — consumed here to build the
  #     samplesheet, not passed to nextflow as pipeline params;
  #   * run-plan bookkeeping (run_id/varied_axis/config_id/rep) — build_run_plan.py's own labels, which
  #     are not pipeline params at all. config_id/rep were previously passed through as
  #     `--config_id cfg000 --rep 0`; nextflow_schema.json sets additionalProperties:true so nf-schema
  #     tolerated them, but they are still undeclared params and would become hard validation errors the
  #     moment the schema is tightened.
  #
  # These become a -params-file, NOT `--<name> <value>` flags. Nextflow 26 delivers
  # every CLI param as a String and nextflow_schema.json declares real types, so the
  # command-line form fails validateParameters() before a single task runs:
  #     * --reg_qc (1): Value is [string] but should be [integer]
  # 25.04.7 accepts it and 26.04.6 does not, so the whole sweep was unlaunchable on
  # the engine this pipeline is verified on -- and it would have failed at LAUNCH,
  # on the cluster, after the matrix had already been generated. JSON carries types.
  # benchmarks/params_json.py does the coercion, reading each param's declared type
  # out of the schema rather than guessing from the text; it is unit-tested in
  # benchmarks/tests/test_params_json.py.
  pairs=()
  for i in "${!cols[@]}"; do
    k="${cols[$i]}"
    case "$k" in run_id|varied_axis|config_id|rep|target_px|n_channels|n_register_images) continue;; esac
    # Skip EMPTY values (a null/blank cell, e.g. reg_jvm_heap_gb: null) — an empty
    # value would override the pipeline default with an empty string. Empty means
    # "use the pipeline default".
    [[ -z "${vals[$i]:-}" ]] && continue
    pairs+=("${k}=${vals[$i]}")
  done
  run_params="$run_dir/params.json"
  if ! (cd "$PIPELINE_DIR" && python3 -m benchmarks.params_json --out "$run_params" \
          ${pairs[@]+"${pairs[@]}"}); then
    echo "ERROR: $run_id — could not type its parameters against nextflow_schema.json; SKIPPING" >&2
    continue
  fi

  echo ">>> $run_id (varied=${varied_axis}, cell=$cell_id)"
  # Launch in the background FROM the per-run dir, so each run's .nextflow/ (history+cache), work dir,
  # log and session name are isolated — parallel heads never collide (the beegfs history-lock killer).
  # cwd = $run_dir, so pipeline + configs are absolute and outputs are run-dir-relative.
  (
    cd "$run_dir" && nextflow -log nextflow.log run "$PIPELINE_DIR" -profile "$PROFILE" \
      -c "$PIPELINE_DIR/benchmarks/configs/benchmark.config" \
      -work-dir work -name "bench_${run_id}" \
      -params-file "$run_params" \
      --input "$sheet" --outdir out --trace_dir trace \
      ${EXTRA_ABS[@]+"${EXTRA_ABS[@]}"} \
      && echo "RUN OK: $run_id" || echo "RUN FAILED: $run_id" >&2
  ) &
  pids+=("$!")
  # Throttle to CONCURRENCY in-flight runs. Wait on the OLDEST pid (FIFO) rather than `wait -n`
  # (bash 4.3+ only) so this works on the older bash found on many clusters. This never exceeds
  # CONCURRENCY runs at once (the property that matters). A run that exited non-zero has
  # already logged RUN FAILED and must not abort the sweep here -- but its status is
  # RECORDED rather than discarded, so a child that died before logging anything (an
  # OOM-killed subshell) is still visible. tests/test_no_swallowed_failures.py forbids
  # the `|| true` form this line used to carry.
  if (( ${#pids[@]} >= CONCURRENCY )); then
    reap_rc=0
    wait "${pids[0]}" || reap_rc=$?
    (( reap_rc == 0 )) || echo "[reap] pid ${pids[0]} exited $reap_rc" >&2
    pids=("${pids[@]:1}")
  fi
done < <(tail -n +2 "$RUN_PLAN" | tr -d '\r')

# Wait for the last in-flight runs to finish before returning (so the analysis step sees all outputs).
wait

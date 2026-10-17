#!/bin/bash
#SBATCH --job-name=regbench-inputs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=regbench-inputs-%j.log
# The benchmark's INPUT STEP: the pipeline's own preprocessing of the cohort, run by
# regbench (it no longer reads the arms' results). One Nextflow head job; the pipeline
# submits its own process jobs from it. Submitted by `launch.sbatch --inputs`
# (submit_cohort.sh --inputs: 2 CPU, 8 G, REGBENCH_INPUTS_TIME, default 2-00:00:00).
#
#   nextflow run <repo> --start preprocessing --stop preprocessing   (via -params-file)
#
# at the SHIPPED DEFAULTS but one: pixel_size is pinned to 0.325 um, as the arms ran it
# (user ruling 2026-10-15), so these inputs are comparable with results made on theirs.
# cleanup_level stays 'none' (so csv/preprocessed.csv is written), cleanup_work stays
# on (what is read is the published tree), skip_preprocess_qc stays off. Every typed
# param is in the params file: Nextflow 26 delivers a command-line param as a String.
#
# Reads REGBENCH_REPO, REGBENCH_INPUT (the cohort's samplesheet), REGBENCH_INPUTS (the
# input folder), REGBENCH_NF_PROFILE (default singularity,ieo, the arms' profile),
# REGBENCH_SITE_CONFIG (passed with -c; `none` passes none) and REGBENCH_SUBMISSION
# (the run's name). Writes, under $REGBENCH_INPUTS/preprocess (the launch directory):
#   params.json, results/csv/preprocessed.csv, results/<patient>/..., work/, trace/
#
# It refuses to start when results/csv/preprocessed.csv is there already, unless
# REGBENCH_INPUTS_FORCE=1, which moves the old preprocess/ to
# preprocess.replaced/<time>/ (never deleted). Its last line:
#   REGBENCH_END kind=inputs status=<ok|failed> cause=... exit=... dir=... job=...
set -uo pipefail
# the operator's start and end lines (env.sh's format; this job does not source env.sh)
t0="$(date +%s)"
commit="$( [ -n "${REGBENCH_REPO:-}" ] && git -C "$REGBENCH_REPO" rev-parse --short=12 HEAD 2>/dev/null)" || commit=unknown
echo "[regbench] start kind=inputs job=${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:--} run=- label=- dataset=- unit=- rung_gb=- attempt=- out=${REGBENCH_INPUTS:--} commit=${commit:-unknown} host=$(hostname 2>/dev/null || echo -) at=$(date +%FT%T)"
end() {  # end <status> <cause> <exit>
    local outcome="$1"
    [ "$1" = ok ] || outcome="$1/$2"
    echo "[regbench] end kind=inputs job=${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:--} outcome=$outcome wall=$(( $(date +%s) - t0 ))s peak_rss=- record=${d:+$d/results/csv/preprocessed.csv} exit=$3"
    echo "REGBENCH_END kind=inputs status=$1 cause=$2 exit=$3 dir=${d:--} job=${SLURM_JOB_ID:--}"
    exit "$3"
}
refuse() {  # refuse <cause> <message>
    echo "regbench-inputs: $2" >&2
    end failed "$1" 2
}
[ -n "${REGBENCH_REPO:-}" ] || refuse no_repo "REGBENCH_REPO is not set (this repository)."
[ -n "${REGBENCH_INPUT:-}" ] || refuse no_samplesheet "REGBENCH_INPUT is not set: the cohort's samplesheet (the pipeline's --input)."
[ -f "$REGBENCH_INPUT" ] || refuse no_samplesheet "REGBENCH_INPUT=$REGBENCH_INPUT: no such file."
[ -n "${REGBENCH_INPUTS:-}" ] || refuse no_inputs_folder "REGBENCH_INPUTS is not set: the folder the input step writes into."
: "${REGBENCH_NF_PROFILE:=singularity,ieo}"
: "${REGBENCH_SUBMISSION:=inputs-$(date +%Y%m%dT%H%M%S)}"
site="${REGBENCH_SITE_CONFIG:-$REGBENCH_REPO/conf/ieo.config}"
case "$site" in none|-) site="" ;; esac
[ -z "$site" ] || [ -f "$site" ] || refuse no_site_config "the site config $site is not there (REGBENCH_SITE_CONFIG; \`none\` for no -c)."

d="$REGBENCH_INPUTS/preprocess"
if [ -e "$d/results/csv/preprocessed.csv" ]; then
    if [ "${REGBENCH_INPUTS_FORCE:-0}" != 1 ]; then
        refuse already_there "$d/results/csv/preprocessed.csv already exists: the input step has run. REGBENCH_INPUTS_FORCE=1 moves $d aside (to $REGBENCH_INPUTS/preprocess.replaced/) and runs it again."
    fi
    aside="$REGBENCH_INPUTS/preprocess.replaced/$(date +%Y%m%dT%H%M%S)-${SLURM_JOB_ID:-$$}"
    mkdir -p "$(dirname "$aside")" && mv "$d" "$aside" \
        || refuse move_failed "could not move $d to $aside; nothing was deleted."
    echo "regbench-inputs: REGBENCH_INPUTS_FORCE=1: the earlier input step moved to $aside"
fi
mkdir -p "$d" || refuse no_inputs_folder "cannot create $d."
cd "$d" || refuse no_inputs_folder "cannot enter $d."

# the params file: typed against nextflow_schema.json (stdlib only, the head's python3)
if ! PYTHONPATH="$REGBENCH_REPO${PYTHONPATH:+:$PYTHONPATH}" python3 -m benchmarks.regbench.params_json \
        --out "$d/params.json" start=preprocessing stop=preprocessing pixel_size=0.325; then
    refuse params_file "could not type the parameters against nextflow_schema.json."
fi

# A run name Nextflow takes (a letter first; letters and digits, a single - or _ between
# them; at most 80), made unique by the job's id: a submission id used twice never hits
# "run name already used" in this launch directory's history.
sub="$(printf '%s' "$REGBENCH_SUBMISSION" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' \
    | tr -s '-' | sed -E 's/^-+//; s/-+$//')"
job="${SLURM_JOB_ID:-$$}"
name="rb-inputs-${sub:+$sub-}"
name="${name:0:$((79 - ${#job}))}"
name="$(printf '%s' "$name" | sed -E 's/-+$//')-$job"
export NXF_VER="${NXF_VER:-25.04.7}"
# one head JVM in an 8 G job: a bounded heap (the JVM default is a quarter of the node)
export NXF_OPTS="${NXF_OPTS:--Xms256m -Xmx4g}"
# Apptainer's shared image cache raced concurrent pulls ("unexpected end of JSON
# input"): each pull unpacks into its own folder instead
export APPTAINER_DISABLE_CACHE="${APPTAINER_DISABLE_CACHE:-true}"
export SINGULARITY_DISABLE_CACHE="${SINGULARITY_DISABLE_CACHE:-$APPTAINER_DISABLE_CACHE}"
echo "regbench-inputs: preprocessing $REGBENCH_INPUT into $d/results (profile $REGBENCH_NF_PROFILE, site config ${site:-none}, run $name, Nextflow $NXF_VER)"
echo "regbench-inputs: params $(tr -d '\n ' < "$d/params.json")"

# no -resume: the step is run once, from the start (a forced re-run starts over).
# Without PYTHONPATH: the launcher's env.sh exports the repository root on it (and the job
# inherits it, --export=ALL); it must not reach the pipeline's tasks, which SLURM starts
# with the head's environment and Singularity passes into their containers
env -u PYTHONPATH nextflow -q run "$REGBENCH_REPO" \
    -profile "$REGBENCH_NF_PROFILE" \
    ${site:+-c "$site"} \
    -work-dir "$d/work" \
    -name "$name" \
    -params-file "$d/params.json" \
    --input "$REGBENCH_INPUT" \
    --outdir "$d/results" \
    --trace_dir "$d/trace"
code=$?
if [ "$code" -ne 0 ]; then
    echo "regbench-inputs: Nextflow exited $code; its log is $d/.nextflow.log" >&2
    end failed preprocessing_failed "$code"
fi
if [ ! -s "$d/results/csv/preprocessed.csv" ]; then
    echo "regbench-inputs: Nextflow ended without writing $d/results/csv/preprocessed.csv (cleanup_level must be 'none')" >&2
    end failed no_checkpoint 1
fi
echo "regbench-inputs: $d/results/csv/preprocessed.csv is written. Next: the --prepare step (launch.sbatch)"
end ok - 0

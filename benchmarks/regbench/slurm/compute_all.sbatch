#!/bin/bash
#SBATCH --job-name=regbench-compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=regbench-compute-%j.log
# compute-all: the WHOLE Mirage pipeline (no --start/--stop) for ONE patient with ONE
# registration backend, traced end to end. One Nextflow head job; every process runs as
# its own scheduler job through the site profile. Submitted by `submit_cohort.sh
# --compute-all` (and `--compute-only`) after everything else, on its own lanes
# (2 CPU, 8 G, REGBENCH_COMPUTE_TIME, default 7-00:00:00):
#
#   compute_all.sbatch <valis|tiled> <patient>
#
# Its launch folder is $REGBENCH_OUT/compute/<backend>/<patient>, with its own
# samplesheet.csv (the cohort's header and that patient's rows), params.json, work/,
# trace/ and results/; input_gb.txt (the patient's slide files, GB) beside them. A folder
# that is there already is MOVED to $REGBENCH_OUT/compute/.replaced/<time>-<job>/
# (with its end line), never deleted and never reused: there is never a -resume, whose
# cached tasks would cost nothing.
#
# The params file (`regbench compute-all params`, typed by nextflow_schema.json) holds
# registration_method, enable_trace (restated at its shipped default) and
# seg_method=stardist and pixel_size=0.325 (the rulings of 2026-10-15: StarDist's nuclei
# and the input step's scale); everything else is the shipped
# default and the site profile's. On the command line only paths: --input, --outdir,
# --trace_dir (Nextflow 26 delivers a command-line param as a String).
#
# Reads REGBENCH_REPO, REGBENCH_OUT, REGBENCH_INPUT (the cohort's samplesheet),
# REGBENCH_NF_PROFILE (default singularity,ieo), REGBENCH_SITE_CONFIG (passed with -c;
# `none` passes none) and REGBENCH_SUBMISSION. Its last line, also written to
# $REGBENCH_OUT/logs/compute/<backend>/<patient>.end:
#   REGBENCH_END kind=compute backend=... patient=... status=<ok|failed> exit=... log=...
set -uo pipefail
backend="${1:-}" patient="${2:-}"
d="" endfile=""
end() {  # end <status> <exit>
    local line="REGBENCH_END kind=compute backend=${backend:--} patient=${patient:--} status=$1 exit=$2 log=${d:+$d/.nextflow.log} job=${SLURM_JOB_ID:--}"
    echo "$line"
    [ -z "$endfile" ] || { mkdir -p "$(dirname "$endfile")" && echo "$line" > "$endfile"; }
    exit "$2"
}
refuse() {  # refuse <message>
    echo "regbench-compute: $1" >&2
    end failed 2
}
case "$backend" in valis|tiled) ;; *) refuse "the backend '$backend': valis or tiled." ;; esac
[ -n "$patient" ] || refuse "no patient named (the arguments: <backend> <patient>)."
case "$patient" in */*|.*) refuse "the patient '$patient' is not a folder name." ;; esac
[ -n "${REGBENCH_REPO:-}" ] || refuse "REGBENCH_REPO is not set (this repository)."
[ -n "${REGBENCH_OUT:-}" ] || refuse "REGBENCH_OUT is not set (the results folder)."
[ -n "${REGBENCH_INPUT:-}" ] || refuse "REGBENCH_INPUT is not set: the cohort's samplesheet (the pipeline's --input)."
[ -f "$REGBENCH_INPUT" ] || refuse "REGBENCH_INPUT=$REGBENCH_INPUT: no such file."
# shellcheck source=benchmarks/regbench/slurm/env.sh
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
: "${REGBENCH_NF_PROFILE:=singularity,ieo}"
: "${REGBENCH_SUBMISSION:=compute-$(date +%Y%m%dT%H%M%S)}"
site="${REGBENCH_SITE_CONFIG:-$REGBENCH_REPO/conf/ieo.config}"
case "$site" in none|-) site="" ;; esac
[ -z "$site" ] || [ -f "$site" ] || refuse "the site config $site is not there (REGBENCH_SITE_CONFIG; \`none\` for no -c)."

endfile="$REGBENCH_OUT/logs/compute/$backend/$patient.end"
d="$REGBENCH_OUT/compute/$backend/$patient"
# a head killed at its time limit still writes its end line
trap 'end failed 143' TERM
# never reused: an earlier run of the same backend and patient is moved aside whole,
# with its end line
if [ -e "$d" ] || [ -e "$endfile" ]; then
    aside="$REGBENCH_OUT/compute/.replaced/$(date +%Y%m%dT%H%M%S)-${SLURM_JOB_ID:-$$}"
    mkdir -p "$aside/$backend" || refuse "cannot create $aside; nothing was moved."
    if [ -e "$d" ]; then
        mv "$d" "$aside/$backend/$patient" || refuse "could not move $d to $aside; nothing was deleted."
    fi
    if [ -e "$endfile" ]; then
        mv "$endfile" "$aside/$backend/$patient.end" || refuse "could not move $endfile to $aside."
    fi
    echo "regbench-compute: the earlier run of $backend on $patient moved to $aside"
fi
mkdir -p "$d" || refuse "cannot create $d."

# the patient's samplesheet and the params file, in the scorer's image
regbench_py "${REGBENCH_HEAD_IMAGE:-STARE_REGBENCH}" compute-all sheet --input "$REGBENCH_INPUT" \
    --patient "$patient" --out "$d/samplesheet.csv" --sizes "$d/input_gb.txt" >/dev/null \
    || refuse "no samplesheet for $patient from $REGBENCH_INPUT (see above)."
regbench_py "${REGBENCH_HEAD_IMAGE:-STARE_REGBENCH}" compute-all params --backend "$backend" \
    --out "$d/params.json" \
    || refuse "could not type the parameters against nextflow_schema.json."

# A run name Nextflow takes (a letter first; letters and digits, a single - or _ between
# them; at most 80), unique by the submission and the job's id
raw="rb-compute-$backend-$patient-$REGBENCH_SUBMISSION"
base="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' \
    | tr -s '-' | sed -E 's/^-+//; s/-+$//')"
job="${SLURM_JOB_ID:-$$}"
name="${base:0:$((79 - ${#job}))}"
name="$(printf '%s' "$name" | sed -E 's/-+$//')-$job"
export NXF_VER="${NXF_VER:-25.04.7}"
# one head JVM in an 8 G job: a bounded heap (the JVM default is a quarter of the node)
export NXF_OPTS="${NXF_OPTS:--Xms256m -Xmx4g}"
# Apptainer's shared image cache raced concurrent pulls ("unexpected end of JSON
# input"): each pull unpacks into its own folder instead
export APPTAINER_DISABLE_CACHE="${APPTAINER_DISABLE_CACHE:-true}"
export SINGULARITY_DISABLE_CACHE="${SINGULARITY_DISABLE_CACHE:-$APPTAINER_DISABLE_CACHE}"
echo "regbench-compute: $backend on $patient: the whole pipeline into $d/results (profile $REGBENCH_NF_PROFILE, site config ${site:-none}, run $name, Nextflow $NXF_VER)"
echo "regbench-compute: params $(tr -d '\n ' < "$d/params.json")"

cd "$d" || refuse "cannot enter $d."
# no -resume, ever. Without PYTHONPATH: env.sh exports the repository root on it, and it
# must not reach the pipeline's tasks (SLURM starts them with the head's environment)
env -u PYTHONPATH nextflow -q run "$REGBENCH_REPO" -profile "$REGBENCH_NF_PROFILE" ${site:+-c "$site"} -work-dir "$d/work" -name "$name" -params-file "$d/params.json" --input "$d/samplesheet.csv" --outdir "$d/results" --trace_dir "$d/trace" > "$d/nextflow.stdout.log" 2> "$d/nextflow.stderr.log"
code=$?
if [ "$code" -ne 0 ]; then
    {
        echo "regbench-compute: Nextflow exited $code ($backend on $patient)"
        echo "----- last 25 lines of stdout ($d/nextflow.stdout.log) -----"
        tail -n 25 "$d/nextflow.stdout.log" 2>/dev/null
        echo "----- last 15 lines of stderr ($d/nextflow.stderr.log) -----"
        tail -n 15 "$d/nextflow.stderr.log" 2>/dev/null
        echo "----- full log: $d/.nextflow.log -----"
    } >&2
    end failed "$code"
fi
if [ ! -s "$d/trace/trace.txt" ]; then
    echo "regbench-compute: Nextflow ended without writing $d/trace/trace.txt (enable_trace is in the params file)" >&2
    end failed 1
fi
end ok 0

#!/bin/bash
#SBATCH --job-name=regbench-launch
#SBATCH --export=ALL
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=regbench-launch-%j.log
# The launch as ONE job: submit_cohort.sh run on a compute node instead of a login node,
# with the caller's whole environment (--export=ALL), so every REGBENCH_* variable set
# before the call reaches the launcher and, through it, every job. From the repository:
#
#   sbatch [site options] benchmarks/regbench/slurm/launch.sbatch [submit_cohort.sh arguments]
#
#   sbatch -p <partition> -A <account> benchmarks/regbench/slurm/launch.sbatch -- -p <partition> -A <account>
#   sbatch ... benchmarks/regbench/slurm/launch.sbatch --prepare -- ...
#   sbatch ... benchmarks/regbench/slurm/launch.sbatch --dry-run -- ...
#
# The site options are named twice on purpose: before the script they are this job's,
# after `--` they are the ones the launcher gives every job it submits.
#
# This job only starts the images once (the preflight) and submits; it ends when the
# last job is in the queue, and its log (regbench-launch-<job id>.log, where it was
# submitted from) is what the launcher prints. It needs what the tile submitter needs
# of a compute node: the scheduler's commands and the container runtime. A node has
# often no network: the pinned STARE source is downloaded once by a launch, so the first
# call (--preflight is enough) is made from a login node, and this job finds it there.
#
# The scheduler runs a COPY of this file, so the repository is not where the file is:
# it is REGBENCH_REPO, else the folder the job was submitted from.
set -uo pipefail
repo="${REGBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
launcher="$repo/benchmarks/regbench/slurm/submit_cohort.sh"
if [ ! -f "$launcher" ]; then
    echo "regbench-launch: $launcher is not there. Submit from the repository, or export REGBENCH_REPO. Nothing was submitted." >&2
    exit 2
fi
export REGBENCH_REPO="$repo"
cd "$repo" || exit 2
exec bash "$launcher" "$@"

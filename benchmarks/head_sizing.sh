#!/usr/bin/env bash
# Head-job sizing for the benchmark submitters: sourced by submit_arms.sh and
# submit_sweep.sh, so the two cannot drift on how "N Nextflow heads fit in the
# allocation" is decided.
#
# A benchmark head job runs CONCURRENCY Nextflow JVMs at once. Each is one
# pipeline launch that submits its own SLURM process jobs; the JVMs themselves
# only poll SLURM and hold the DAG. Their memory is the ONLY thing raising
# concurrency can OOM: the process jobs are sized per task in conf/modules.config
# and scheduled by SLURM against node memory, so more heads means more of them in
# the queue, never more memory per job. Hence the two checks here:
#
#   check_head_memory  CONCURRENCY x (-Xmx + JVM/OS overhead) must fit --mem.
#                      Refuses the launch, naming the two knobs, when it does not.
#   derive_queue_size  keep the TOTAL in-flight SLURM jobs at a target while the
#                      head count changes: per-head queueSize = target / heads,
#                      floored at max_forks so one head can still fill its own
#                      per-process clamps (REGISTER 10, TILED_* 20 -- see
#                      nextflow.config's Concurrency block).
#
# Guarded by benchmarks/tests/test_submit_head_sizing.py, which runs these
# functions with a fake SLURM_MEM_PER_NODE and parses both submitters' defaults.

# Per-head memory above -Xmx: JVM metaspace, code cache, threads, plus the
# Nextflow launcher's own footprint. Measured heads sit 0.4-0.7 GB above heap.
HEAD_OVERHEAD_GB="${HEAD_OVERHEAD_GB:-0.75}"

# head_heap_gb "<NXF_OPTS>": the -Xmx in GB (accepts 2g / 2048m / 2G). 0 if absent.
head_heap_gb() {
  local opts="$1" xmx unit num
  xmx=$(printf '%s' "$opts" | grep -oE -- '-Xmx[0-9]+[gGmM]' | tail -n1)
  [[ -n "$xmx" ]] || { echo 0; return; }
  num="${xmx#-Xmx}"; unit="${num: -1}"; num="${num%?}"
  case "$unit" in
    g|G) echo "$num" ;;
    m|M) awk -v m="$num" 'BEGIN { printf "%.2f", m / 1024 }' ;;
  esac
}

# allocation_gb: the head job's memory allocation in GB, from what SLURM exports
# (--mem sets SLURM_MEM_PER_NODE in MB; --mem-per-cpu sets SLURM_MEM_PER_CPU).
# Prints nothing outside SLURM.
allocation_gb() {
  if [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
    awk -v mb="$SLURM_MEM_PER_NODE" 'BEGIN { printf "%.2f", mb / 1024 }'
  elif [[ -n "${SLURM_MEM_PER_CPU:-}" && -n "${SLURM_CPUS_ON_NODE:-}" ]]; then
    awk -v mb="$SLURM_MEM_PER_CPU" -v c="$SLURM_CPUS_ON_NODE" 'BEGIN { printf "%.2f", mb * c / 1024 }'
  fi
}

# check_head_memory <concurrency> "<NXF_OPTS>" [overhead_gb]
# Exit 0 when CONCURRENCY heads fit the allocation (or no allocation is exported,
# i.e. not under SLURM -- prints a note); exit 1 with the arithmetic otherwise.
check_head_memory() {
  local heads="$1" opts="$2" overhead="${3:-$HEAD_OVERHEAD_GB}"
  local heap alloc need
  heap=$(head_heap_gb "$opts")
  if [[ "$heap" == "0" ]]; then
    echo "ERROR: NXF_OPTS='$opts' sets no -Xmx; each head would take the JVM default (1/4 of node RAM)" >&2
    echo "       and $heads of them would exceed any allocation. Set NXF_OPTS='-Xms256m -Xmx2g'." >&2
    return 1
  fi
  alloc=$(allocation_gb)
  need=$(awk -v h="$heads" -v x="$heap" -v o="$overhead" 'BEGIN { printf "%.2f", h * (x + o) }')
  if [[ -z "$alloc" ]]; then
    echo "head sizing: $heads heads x (${heap} GB heap + ${overhead} GB overhead) = ${need} GB; no SLURM allocation exported, not checked"
    return 0
  fi
  if awk -v n="$need" -v a="$alloc" 'BEGIN { exit !(n > a) }'; then
    echo "ERROR: $heads heads x (${heap} GB -Xmx + ${overhead} GB overhead) = ${need} GB exceeds the head job's ${alloc} GB." >&2
    echo "       Either raise '#SBATCH --mem' to at least $(awk -v n="$need" 'BEGIN { printf "%d", n + 1 }')G, or lower the" >&2
    echo "       head count (ARMS_CONCURRENCY / SWEEP_CONCURRENCY) or the per-head heap (NXF_OPTS -Xmx)." >&2
    echo "       The process jobs are unaffected by this: they are sized per task in conf/modules.config." >&2
    return 1
  fi
  echo "head sizing: $heads heads x (${heap} GB heap + ${overhead} GB overhead) = ${need} GB of ${alloc} GB -- fits"
}

# derive_queue_size <concurrency> <max_forks> <peak_jobs_target>
# Per-head executor.queueSize so that heads x queueSize ~= peak target, never
# below max_forks (a head must be able to fill its own per-process clamps).
derive_queue_size() {
  local heads="$1" max_forks="$2" target="$3" q
  q=$(( target / heads ))
  (( q < max_forks )) && q="$max_forks"
  echo "$q"
}

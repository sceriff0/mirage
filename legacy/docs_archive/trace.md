# Plan: Integrating Input Size Tracking with Nextflow Trace Data

## Goal

Enable per-process analysis of runtime and memory usage as a function of input file size and allocated resources. This requires combining Nextflow's built-in trace output with custom input size metadata collected during pipeline execution.

---

## Phase 1: Enable and Configure Trace Output

Add the following to `nextflow.config`:

```groovy
trace {
    enabled   = true
    file      = "${params.outdir}/trace/trace.txt"
    fields    = 'task_id,process,tag,name,status,exit,submit,start,complete,duration,realtime,%cpu,cpus,memory,peak_rss,peak_vmem,rchar,wchar'
    overwrite = true
}
```

Keep `-with-report` and `-with-timeline` as optional extras for quick visual checks, but the trace file is the primary data source going forward.

---

## Phase 2: Collect Input Size Metadata

### Approach A — Lightweight (shell-based, per task)

Add input size logging to each process that you want to analyze. Each process emits a single CSV row to a standardized file.

```groovy
process EXAMPLE_PROCESS {
    tag "${sample_id}"
    publishDir "${params.outdir}/size_logs", mode: 'copy', pattern: '*.size.csv'

    input:
    tuple val(sample_id), path(input_file)

    output:
    tuple val(sample_id), path("result.bam"),   emit: results
    path("${sample_id}.size.csv"),               emit: size_log

    script:
    """
    # Log input size
    input_bytes=\$(stat --printf="%s" ${input_file})
    echo "${task.process},${sample_id},${input_file},\${input_bytes}" > ${sample_id}.size.csv

    # Actual process command
    my_tool --input ${input_file} --output result.bam
    """
}
```

Repeat this pattern for every process you want to track.

### Approach B — Centralized (Nextflow-native, no shell changes)

Use a dedicated process that runs before or alongside your pipeline to compute sizes, or use a channel operator to attach size info:

```groovy
// Compute input sizes in the channel itself
input_ch = Channel
    .fromFilePairs("${params.reads}/*_{1,2}.fastq.gz")
    .map { sample_id, files ->
        def total_bytes = files.collect { it.size() }.sum()
        tuple(sample_id, files, total_bytes)
    }
```

Then pass `total_bytes` through the pipeline and emit it alongside results. This avoids modifying process scripts but requires threading the value through every relevant channel.

### Recommendation

Use **Approach A** for most pipelines — it's explicit, self-contained per process, and doesn't add coupling between processes. Use Approach B only if you want a single pre-computed size value attached to a sample without touching process scripts.

---

## Phase 3: Aggregate Size Logs

Add a final process that collects all per-task size logs into one file:

```groovy
process AGGREGATE_SIZE_LOGS {
    publishDir "${params.outdir}/trace", mode: 'copy'

    input:
    path(size_csvs)

    output:
    path("input_sizes.csv")

    script:
    """
    echo "process,sample_id,filename,bytes" > input_sizes.csv
    cat ${size_csvs} >> input_sizes.csv
    """
}
```

Wire it up by collecting the `size_log` outputs from all processes:

```groovy
AGGREGATE_SIZE_LOGS(
    EXAMPLE_PROCESS.out.size_log
        .mix(OTHER_PROCESS.out.size_log)
        .collect()
)
```

---

## Phase 4: Join Trace and Size Data

After the pipeline completes, join the two files on process name and sample ID. A minimal Python script:

```python
#!/usr/bin/env python3
"""join_trace_sizes.py — Merge Nextflow trace with input size metadata."""

import pandas as pd
import sys

trace = pd.read_csv(sys.argv[1], sep='\t')
sizes = pd.read_csv(sys.argv[2])

# Extract sample_id from the trace 'tag' column (assumes tag = sample_id)
trace['sample_id'] = trace['tag']

merged = trace.merge(sizes, on=['process', 'sample_id'], how='left')

# Convert memory strings (e.g. "3.2 GB") to numeric bytes
def parse_mem(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    units = {'B': 1, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9, 'TB': 1e12}
    for unit, factor in units.items():
        if val.upper().endswith(unit):
            return float(val[:len(val)-len(unit)].strip()) * factor
    return float(val)

for col in ['peak_rss', 'peak_vmem', 'memory']:
    if col in merged.columns:
        merged[f'{col}_bytes'] = merged[col].apply(parse_mem)

merged.to_csv(sys.argv[3], index=False)
print(f"Merged {len(merged)} rows -> {sys.argv[3]}")
```

Usage:

```bash
python join_trace_sizes.py results/trace/trace.txt results/trace/input_sizes.csv results/trace/merged.csv
```

---

## Phase 5: Analysis and Visualization

With `merged.csv` you can now answer the core questions. Example analysis script:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/trace/merged.csv')

processes = df['process'].unique()

fig, axes = plt.subplots(len(processes), 2, figsize=(14, 5 * len(processes)))

for i, proc in enumerate(processes):
    subset = df[df['process'] == proc]

    # Runtime vs input size
    axes[i][0].scatter(subset['bytes'] / 1e9, subset['realtime'] / 1000)
    axes[i][0].set_xlabel('Input Size (GB)')
    axes[i][0].set_ylabel('Runtime (s)')
    axes[i][0].set_title(f'{proc} — Runtime vs Input Size')

    # Peak RSS vs input size
    axes[i][1].scatter(subset['bytes'] / 1e9, subset['peak_rss_bytes'] / 1e9)
    axes[i][1].set_xlabel('Input Size (GB)')
    axes[i][1].set_ylabel('Peak RSS (GB)')
    axes[i][1].set_title(f'{proc} — Memory vs Input Size')

plt.tight_layout()
plt.savefig('results/trace/scaling_analysis.png', dpi=150)
```

### Key questions this enables

- Which processes scale linearly with input size? Which are superlinear?
- Are any processes over-provisioned (allocated >> used)?
- At what input size does a process start running out of memory?
- What is the optimal `cpus` / `memory` request per process for your data sizes?

---

## File Structure After Integration

```
results/
├── trace/
│   ├── trace.txt              # Nextflow trace output
│   ├── input_sizes.csv        # Aggregated size metadata
│   ├── merged.csv             # Joined dataset
│   └── scaling_analysis.png   # Visualization
├── size_logs/
│   ├── sample_A.size.csv      # Per-task size logs
│   ├── sample_B.size.csv
│   └── ...
└── ...                        # Normal pipeline outputs
```

---

## Checklist

- [ ] Add `trace` block to `nextflow.config`
- [ ] Add size logging to each target process
- [ ] Add `AGGREGATE_SIZE_LOGS` process and wire it up
- [ ] Add `join_trace_sizes.py` to `bin/`
- [ ] Run pipeline on a test dataset and verify `merged.csv`
- [ ] Run analysis script and review plots
- [ ] Tune resource requests in `nextflow.config` based on findings
# SLURM Execution Guide

This guide explains how to run the ATEIA WSI processing pipeline on SLURM clusters.

## Quick Start

### 1. Basic SLURM Submission

Create a submission script `submit_pipeline.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ateia_wsi
#SBATCH --output=logs/nextflow_%j.out
#SBATCH --error=logs/nextflow_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=normal

# Load required modules (adjust for your cluster)
module load nextflow/23.10.0
module load singularity/3.8.0

# Create logs directory
mkdir -p logs

# Run Nextflow pipeline
nextflow run main.nf \
  -profile slurm \
  --input "data/*.nd2" \
  --outdir results \
  -resume
```

Submit with:
```bash
sbatch submit_pipeline.sh
```

### 2. Interactive Session (for testing)

```bash
# Request interactive session
srun --cpus-per-task=4 --mem=16G --time=2:00:00 --pty bash

# Load modules
module load nextflow singularity

# Run pipeline
nextflow run main.nf -profile slurm --input "data/*.nd2" --outdir results
```

## Configuration Options

### A. Set SLURM Account/Partition

Edit `nextflow.config` or use command-line parameters:

```bash
# Via config file
params {
    slurm_partition = "gpu"      # Your partition name
    slurm_account   = "proj123"  # Your SLURM account
    slurm_qos       = "normal"   # QOS if required
}
```

Or via command line:
```bash
nextflow run main.nf \
  -profile slurm \
  --slurm_partition gpu \
  --slurm_account proj123 \
  --input "data/*.nd2"
```

### B. GPU Configuration

The pipeline automatically requests GPUs based on parameters:

```bash
nextflow run main.nf \
  -profile slurm \
  --seg_gpu true \        # SEGMENT process gets GPU
  --input "data/*.nd2"
```

SLURM options added automatically:
- `SEGMENT`: `--gres=gpu:1` if `seg_gpu=true`

### C. Customize Resources per Process

Edit `conf/modules.config`:

```groovy
process {
    withName: 'SEGMENT' {
        cpus   = 16          // Increase CPUs
        memory = '64.GB'     // Increase memory
        time   = '12.h'      // Extend time limit
        clusterOptions = '--gres=gpu:2'  // Request 2 GPUs
    }
}
```

## Resource Allocation

### Default Resources

| Process | CPUs | Memory | Time | GPU |
|---------|------|--------|------|-----|
| PREPROCESS | 4 | 16 GB | 2h | No |
| REGISTER | 8 | 32 GB | 4h | No |
| SEGMENT | 8 | 32 GB | 6h | Optional |
| QUANTIFY | 4 | 16 GB | 4h | Optional |
| PHENOTYPE | 4 | 16 GB | 2h | No |

### Resource Labels

Processes use labels that map to resources:

```groovy
withLabel: 'process_low' {     // 2 CPUs, 4 GB, 1h
withLabel: 'process_medium' {  // 4 CPUs, 16 GB, 4h
withLabel: 'process_high' {    // 8 CPUs, 32 GB, 8h
withLabel: 'gpu' {             // 8 CPUs, 64 GB, 12h, --gres=gpu:1
```

## Advanced SLURM Submission

### 1. Submission Script with Email Notifications

```bash
#!/bin/bash
#SBATCH --job-name=ateia_wsi
#SBATCH --output=logs/nextflow_%j.out
#SBATCH --error=logs/nextflow_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=normal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.com

module load nextflow singularity

mkdir -p logs work

# Run with resume capability
nextflow run main.nf \
  -profile slurm \
  --input "data/*.nd2" \
  --outdir results \
  --seg_gpu true \
  -resume \
  -with-report reports/report_${SLURM_JOB_ID}.html \
  -with-timeline reports/timeline_${SLURM_JOB_ID}.html \
  -with-dag reports/dag_${SLURM_JOB_ID}.html
```

### 2. Batch Processing Multiple Samples

Create `run_samples.sh`:

```bash
#!/bin/bash

for sample in sample1 sample2 sample3; do
  sbatch --job-name=ateia_${sample} << EOF
#!/bin/bash
#SBATCH --output=logs/${sample}_%j.out
#SBATCH --error=logs/${sample}_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

module load nextflow singularity

nextflow run main.nf \
  -profile slurm \
  --input "data/${sample}/*.nd2" \
  --outdir results/${sample} \
  -resume
EOF
done
```

### 3. Custom Partition for Different Processes

```groovy
// In conf/modules.config
process {
    // CPU-only processes on standard partition
    withName: 'PREPROCESS|REGISTER|PHENOTYPE' {
        queue = 'standard'
    }

    // GPU processes on GPU partition
    withName: 'SEGMENT|QUANTIFY' {
        queue = 'gpu'
        clusterOptions = '--gres=gpu:1'
    }
}
```

## Monitoring

### Check Job Status

```bash
# View all jobs
squeue -u $USER

# View specific pipeline jobs
squeue -u $USER | grep ateia

# Check job details
scontrol show job <JOB_ID>
```

### Monitor Nextflow

```bash
# Watch Nextflow log
tail -f .nextflow.log

# View running processes
nextflow log <RUN_NAME> -f name,status,duration,realtime,%cpu,rss
```

### Access Reports

Nextflow generates useful reports:

```bash
nextflow run main.nf \
  -with-report report.html \
  -with-timeline timeline.html \
  -with-dag dag.html
```

View in browser or copy to local machine:
```bash
scp user@cluster:path/to/report.html .
```

## Troubleshooting

### Issue: Jobs Not Starting

**Check partition availability:**
```bash
sinfo -p <partition_name>
squeue -p <partition_name>
```

**Verify account:**
```bash
sacctmgr show user $USER
```

### Issue: GPU Not Available

**Check GPU partitions:**
```bash
sinfo -o "%20P %5D %6t %10G"
```

**Test GPU allocation:**
```bash
srun --gres=gpu:1 --pty bash
nvidia-smi  # Should show GPU
```

### Issue: Out of Memory

**Increase memory in `conf/modules.config`:**
```groovy
withName: 'SEGMENT' {
    memory = '64.GB'  // Increase from 32GB
}
```

**Or use command line:**
```bash
nextflow run main.nf \
  -profile slurm \
  --input "data/*.nd2" \
  -c <(echo "process { withName: 'SEGMENT' { memory = '64.GB' } }")
```

### Issue: Time Limit Exceeded

**Increase time limits:**
```groovy
withName: 'SEGMENT' {
    time = '24.h'  // Extend from 6h
}
```

**Enable automatic retry with more time:**
```groovy
process {
    errorStrategy = { task.exitStatus == 140 ? 'retry' : 'finish' }
    maxRetries = 2
    time = { 6.h * task.attempt }  // Double time on retry
}
```

## Best Practices

### 1. Use `-resume`

Always use `-resume` flag to continue from last successful step:
```bash
nextflow run main.nf -profile slurm --input "data/*.nd2" -resume
```

### 2. Test with Small Dataset First

```bash
nextflow run main.nf \
  -profile test \
  --input "data/test_sample.nd2" \
  --outdir test_results
```

### 3. Monitor Resource Usage

After pipeline completes, check resource usage:
```bash
# View job statistics
sacct -j <JOB_ID> --format=JobID,JobName,MaxRSS,Elapsed,State

# Detailed resource usage
seff <JOB_ID>
```

Adjust resource allocations based on actual usage.

### 4. Clean Work Directory

Nextflow stores intermediate files in `work/`:
```bash
# After successful completion, clean up
nextflow clean -f

# Or manually remove work directory
rm -rf work/
```

### 5. Use Singularity Caching

```bash
# Set singularity cache directory
export NXF_SINGULARITY_CACHEDIR=/path/to/cache

# Pull containers before running pipeline
singularity pull docker://alech00/attend_image_analysis:v2.1
```

## Example Configurations

### Configuration for IEO HPC Cluster

```groovy
// custom_slurm.config
params {
    slurm_partition = 'P_DIMA_ATTEND'
    slurm_account   = 'P_DIMA_ATTEND'
}

process {
    withName: 'SEGMENT|QUANTIFY' {
        queue = 'gpu_partition'
        clusterOptions = '--gres=gpu:1 --constraint=v100'
    }
}

singularity {
    enabled = true
    autoMounts = true
    cacheDir = '/hpcnfs/scratch/singularity_cache'
}
```

Use with:
```bash
nextflow run main.nf -profile slurm -c custom_slurm.config --input "data/*.nd2"
```

## Summary

### Quick Command Reference

| Task | Command |
|------|---------|
| **Submit job** | `sbatch submit_pipeline.sh` |
| **Interactive run** | `srun --cpus-per-task=4 --mem=16G --pty bash` |
| **Check status** | `squeue -u $USER` |
| **Cancel job** | `scancel <JOB_ID>` |
| **View log** | `tail -f logs/nextflow_<JOB_ID>.out` |
| **Resume pipeline** | Add `-resume` flag |
| **Clean work** | `nextflow clean -f` |

### Typical Workflow

1. Create submission script
2. Test with small dataset: `sbatch submit_pipeline.sh`
3. Monitor: `tail -f logs/nextflow_*.out`
4. If fails, fix issue and resubmit with `-resume`
5. Review reports when complete
6. Clean up work directory

For more information, see [Nextflow SLURM documentation](https://www.nextflow.io/docs/latest/executor.html#slurm).

# Building a Restartable Nextflow Pipeline (Without the Work Directory)

## How nf-core/sarek Implements Restart Capability

### The Core Concept

Sarek uses a **checkpoint-based restart system** that is completely independent of Nextflow's built-in `-resume` functionality. The key insight is:

> **Instead of relying on the `work/` directory cache, publish intermediate results as "checkpoint files" along with auto-generated CSV samplesheets that can be used as inputs to restart from any step.**

### The Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   mapping    │────▶│ markduplicates│────▶│  recalibrate │────▶│variant_calling│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
   mapped.csv        markduplicates.csv   recalibrated.csv    variantcalled.csv
   + BAM/CRAM        + CRAM files         + CRAM files        + VCF files
```

Each step:
1. **Publishes its output files** to a predictable location in `results/`
2. **Generates a CSV samplesheet** that describes those outputs
3. The next step can start fresh using that CSV as input with `--step <stepname>`

---

## The `--step` Parameter

Sarek defines valid steps as entry points:

```groovy
// In nextflow_schema.json or params definition
params.step = 'mapping'  // default

// Valid steps:
// - mapping (default - start from FASTQ)
// - markduplicates (start from mapped BAM/CRAM)
// - prepare_recalibration
// - recalibrate (start from duplicate-marked + recal table)
// - variant_calling (start from recalibrated CRAM)
// - annotation (start from VCF files)
```

---

## How CSV Samplesheets Are Generated

In Sarek's `workflows/sarek/main.nf`, there are include statements like:

```groovy
// Create samplesheets to restart from different steps
include { CHANNEL_VARIANT_CALLING_CREATE_CSV } from '../../subworkflows/local/channel_variant_calling_create_csv/main'
include { CHANNEL_MARKDUPLICATES_CREATE_CSV  } from '../../subworkflows/local/channel_markduplicates_create_csv/main'
include { CHANNEL_RECALIBRATE_CREATE_CSV     } from '../../subworkflows/local/channel_recalibrate_create_csv/main'
```

Each of these subworkflows:
1. Takes a channel of output files from the current step
2. Writes a CSV file with the paths to those files plus metadata (patient, sample, sex, status)
3. Publishes the CSV to `results/csv/`

### Example: What `markduplicates.csv` looks like:

```csv
patient,sex,status,sample,cram,crai,table
patient1,XX,0,normal_sample,/path/to/results/preprocessing/markduplicates/normal_sample/normal_sample.md.cram,/path/to/results/preprocessing/markduplicates/normal_sample/normal_sample.md.cram.crai,/path/to/results/preprocessing/recal_table/normal_sample/normal_sample.recal.table
patient1,XX,1,tumor_sample,/path/to/results/preprocessing/markduplicates/tumor_sample/tumor_sample.md.cram,/path/to/results/preprocessing/markduplicates/tumor_sample/tumor_sample.md.cram.crai,/path/to/results/preprocessing/recal_table/tumor_sample/tumor_sample.recal.table
```

---

## How to Use It (User Perspective)

### Initial Run (Full Pipeline):
```bash
nextflow run nf-core/sarek \
  --input samplesheet.csv \
  --outdir results \
  --tools haplotypecaller
```

### Restart from Variant Calling (after deleting work/):
```bash
rm -rf work/  # Delete the work directory

nextflow run nf-core/sarek \
  --input results/csv/recalibrated.csv \  # Use generated CSV!
  --step variant_calling \                 # Start from this step
  --outdir results \
  --tools strelka  # Maybe try a different tool
```

---

## Implementing This in Your Own Pipeline

### Step 1: Define Your Pipeline Steps

```groovy
// params.nf or nextflow.config
params.step = 'preprocessing'  // default

def valid_steps = ['preprocessing', 'alignment', 'markdup', 'variant_calling', 'annotation']

// Validate step parameter
if (!valid_steps.contains(params.step)) {
    error "Invalid step: ${params.step}. Valid steps: ${valid_steps.join(', ')}"
}
```

### Step 2: Create a CSV Writer Process

```groovy
process WRITE_CSV {
    publishDir "${params.outdir}/csv", mode: 'copy'
    
    input:
    val(csv_name)
    val(header)
    val(rows)  // List of lists, each inner list is a row
    
    output:
    path("${csv_name}.csv")
    
    script:
    def content = header + '\n' + rows.collect { it.join(',') }.join('\n')
    """
    echo "${content}" > ${csv_name}.csv
    """
}
```

Or as a Groovy function in a subworkflow:

```groovy
workflow CREATE_CHECKPOINT_CSV {
    take:
    ch_files     // Channel of [meta, file1, file2, ...]
    csv_name     // String: name of output CSV
    columns      // List of column names
    
    main:
    // Collect all entries and write CSV
    ch_files
        .map { meta, files -> 
            // Build row from meta and file paths
            def row = [
                meta.patient,
                meta.sex ?: 'NA',
                meta.status ?: '0',
                meta.sample
            ]
            // Add file paths
            files.each { row << it.toString() }
            row
        }
        .collect()
        .map { rows ->
            def header = columns.join(',')
            def content = rows.collect { it.join(',') }.join('\n')
            [csv_name, header + '\n' + content]
        }
        .set { ch_csv_content }
    
    // Write to file
    ch_csv_content
        .map { name, content ->
            def csv_file = file("${params.outdir}/csv/${name}.csv")
            csv_file.text = content
            csv_file
        }
        .set { ch_csv }
    
    emit:
    csv = ch_csv
}
```

### Step 3: Conditional Workflow Execution

```groovy
workflow MY_PIPELINE {
    
    // Parse input based on step
    if (params.step == 'preprocessing') {
        ch_input = Channel.fromPath(params.input)
            .splitCsv(header: true)
            .map { row -> 
                [[patient: row.patient, sample: row.sample], 
                 file(row.fastq_1), file(row.fastq_2)]
            }
        
        // Run preprocessing
        PREPROCESS(ch_input)
        ch_preprocessed = PREPROCESS.out.preprocessed
        
        // Create checkpoint CSV
        CREATE_CHECKPOINT_CSV(
            ch_preprocessed,
            'preprocessed',
            ['patient', 'sample', 'preprocessed_fastq_1', 'preprocessed_fastq_2']
        )
        
    } else {
        // Load from checkpoint
        ch_preprocessed = Channel.fromPath(params.input)
            .splitCsv(header: true)
            .map { row -> 
                [[patient: row.patient, sample: row.sample],
                 file(row.preprocessed_fastq_1), file(row.preprocessed_fastq_2)]
            }
    }
    
    if (params.step in ['preprocessing', 'alignment']) {
        // Run alignment
        ALIGN(ch_preprocessed)
        ch_aligned = ALIGN.out.bam
        
        // Create checkpoint CSV
        CREATE_CHECKPOINT_CSV(
            ch_aligned,
            'aligned',
            ['patient', 'sample', 'bam', 'bai']
        )
        
    } else if (params.step in ['markdup', 'variant_calling', 'annotation']) {
        // Load from alignment checkpoint
        ch_aligned = Channel.fromPath(params.input)
            .splitCsv(header: true)
            .map { row -> 
                [[patient: row.patient, sample: row.sample],
                 file(row.bam), file(row.bai)]
            }
    }
    
    // Continue pattern for remaining steps...
}
```

### Step 4: Complete Example Workflow

```groovy
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// ============================================================================
// PROCESSES
// ============================================================================

process TRIM_READS {
    tag "${meta.sample}"
    publishDir "${params.outdir}/trimmed/${meta.sample}", mode: 'copy'
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.trimmed.fq.gz"), emit: trimmed
    
    script:
    """
    fastp -i ${reads[0]} -I ${reads[1]} \
          -o ${meta.sample}_R1.trimmed.fq.gz \
          -O ${meta.sample}_R2.trimmed.fq.gz
    """
}

process ALIGN {
    tag "${meta.sample}"
    publishDir "${params.outdir}/aligned/${meta.sample}", mode: 'copy'
    
    input:
    tuple val(meta), path(reads)
    path(reference)
    
    output:
    tuple val(meta), path("*.bam"), path("*.bai"), emit: bam
    
    script:
    """
    bwa mem -t ${task.cpus} ${reference} ${reads} | \
        samtools sort -o ${meta.sample}.bam -
    samtools index ${meta.sample}.bam
    """
}

process MARK_DUPLICATES {
    tag "${meta.sample}"
    publishDir "${params.outdir}/markdup/${meta.sample}", mode: 'copy'
    
    input:
    tuple val(meta), path(bam), path(bai)
    
    output:
    tuple val(meta), path("*.md.bam"), path("*.md.bam.bai"), emit: bam
    
    script:
    """
    gatk MarkDuplicates \
        -I ${bam} \
        -O ${meta.sample}.md.bam \
        -M ${meta.sample}.metrics.txt
    samtools index ${meta.sample}.md.bam
    """
}

process CALL_VARIANTS {
    tag "${meta.sample}"
    publishDir "${params.outdir}/variants/${meta.sample}", mode: 'copy'
    
    input:
    tuple val(meta), path(bam), path(bai)
    path(reference)
    
    output:
    tuple val(meta), path("*.vcf.gz"), path("*.vcf.gz.tbi"), emit: vcf
    
    script:
    """
    bcftools mpileup -f ${reference} ${bam} | \
        bcftools call -mv -Oz -o ${meta.sample}.vcf.gz
    bcftools index -t ${meta.sample}.vcf.gz
    """
}

// ============================================================================
// CSV CREATION WORKFLOW
// ============================================================================

workflow WRITE_CHECKPOINT_CSV {
    take:
    ch_data      // Channel with data to write
    csv_name     // Name for the CSV file
    
    main:
    ch_data
        .collect()
        .map { items ->
            def outdir = params.outdir
            def csv_file = file("${outdir}/csv/${csv_name}.csv")
            csv_file.parent.mkdirs()
            
            // Build CSV content based on data structure
            def lines = []
            items.each { item ->
                def meta = item[0]
                def files = item[1..-1]
                def row = [meta.patient, meta.sex ?: 'NA', meta.status ?: '0', meta.sample]
                files.each { f -> row << f.toString() }
                lines << row.join(',')
            }
            
            csv_file
        }
        .set { ch_csv }
    
    emit:
    csv = ch_csv
}

// ============================================================================
// MAIN WORKFLOW
// ============================================================================

workflow {
    
    // Define valid steps
    def steps = ['trimming', 'alignment', 'markdup', 'variant_calling']
    def step_index = steps.indexOf(params.step)
    
    if (step_index == -1) {
        error "Invalid step '${params.step}'. Valid: ${steps.join(', ')}"
    }
    
    // Reference files
    ch_reference = Channel.fromPath(params.reference)
    
    // ========================================================================
    // STEP: TRIMMING
    // ========================================================================
    if (params.step == 'trimming') {
        // Parse FASTQ input
        ch_input = Channel.fromPath(params.input)
            .splitCsv(header: true)
            .map { row -> 
                def meta = [patient: row.patient, sample: row.sample, 
                           sex: row.sex, status: row.status]
                [meta, [file(row.fastq_1), file(row.fastq_2)]]
            }
        
        TRIM_READS(ch_input)
        ch_trimmed = TRIM_READS.out.trimmed
        
    } else {
        // Skip - will load from later checkpoint
        ch_trimmed = Channel.empty()
    }
    
    // ========================================================================
    // STEP: ALIGNMENT
    // ========================================================================
    if (params.step == 'trimming') {
        // Continue from trimming output
        ch_to_align = ch_trimmed
        
    } else if (params.step == 'alignment') {
        // Load from trimmed checkpoint CSV
        ch_to_align = Channel.fromPath(params.input)
            .splitCsv(header: true)
            .map { row ->
                def meta = [patient: row.patient, sample: row.sample,
                           sex: row.sex, status: row.status]
                [meta, [file(row.trimmed_r1), file(row.trimmed_r2)]]
            }
    }
    
    if (step_index <= steps.indexOf('alignment')) {
        ALIGN(ch_to_align, ch_reference.collect())
        ch_aligned = ALIGN.out.bam
        
        // Create checkpoint CSV for alignment output
        ch_aligned
            .map { meta, bam, bai ->
                "${meta.patient},${meta.sex ?: 'NA'},${meta.status ?: '0'},${meta.sample},${bam},${bai}"
            }
            .collect()
            .map { rows ->
                def header = "patient,sex,status,sample,bam,bai"
                def csv = file("${params.outdir}/csv/aligned.csv")
                csv.parent.mkdirs()
                csv.text = header + '\n' + rows.join('\n')
            }
            
    } else {
        ch_aligned = Channel.empty()
    }
    
    // ========================================================================
    // STEP: MARK DUPLICATES
    // ========================================================================
    if (step_index <= steps.indexOf('alignment')) {
        ch_to_markdup = ch_aligned
        
    } else if (params.step == 'markdup') {
        ch_to_markdup = Channel.fromPath(params.input)
            .splitCsv(header: true)
            .map { row ->
                def meta = [patient: row.patient, sample: row.sample,
                           sex: row.sex, status: row.status]
                [meta, file(row.bam), file(row.bai)]
            }
    }
    
    if (step_index <= steps.indexOf('markdup')) {
        MARK_DUPLICATES(ch_to_markdup)
        ch_markdup = MARK_DUPLICATES.out.bam
        
        // Create checkpoint CSV
        ch_markdup
            .map { meta, bam, bai ->
                "${meta.patient},${meta.sex ?: 'NA'},${meta.status ?: '0'},${meta.sample},${bam},${bai}"
            }
            .collect()
            .map { rows ->
                def header = "patient,sex,status,sample,bam,bai"
                def csv = file("${params.outdir}/csv/markdup.csv")
                csv.parent.mkdirs()
                csv.text = header + '\n' + rows.join('\n')
            }
            
    } else {
        ch_markdup = Channel.empty()
    }
    
    // ========================================================================
    // STEP: VARIANT CALLING
    // ========================================================================
    if (step_index <= steps.indexOf('markdup')) {
        ch_to_call = ch_markdup
        
    } else if (params.step == 'variant_calling') {
        ch_to_call = Channel.fromPath(params.input)
            .splitCsv(header: true)
            .map { row ->
                def meta = [patient: row.patient, sample: row.sample,
                           sex: row.sex, status: row.status]
                [meta, file(row.bam), file(row.bai)]
            }
    }
    
    if (step_index <= steps.indexOf('variant_calling')) {
        CALL_VARIANTS(ch_to_call, ch_reference.collect())
        ch_vcf = CALL_VARIANTS.out.vcf
        
        // Create checkpoint CSV
        ch_vcf
            .map { meta, vcf, tbi ->
                "${meta.patient},${meta.sex ?: 'NA'},${meta.status ?: '0'},${meta.sample},${vcf},${tbi}"
            }
            .collect()
            .map { rows ->
                def header = "patient,sex,status,sample,vcf,tbi"
                def csv = file("${params.outdir}/csv/variants.csv")
                csv.parent.mkdirs()
                csv.text = header + '\n' + rows.join('\n')
            }
    }
}
```

---

## Key Design Principles

### 1. **Publish Everything Needed for Restart**
Don't just publish final results. Publish intermediate files that would be needed to restart:
```groovy
publishDir "${params.outdir}/preprocessing/markduplicates/${meta.sample}", mode: 'copy'
```

### 2. **Use Absolute Paths in CSV**
The generated CSVs should use absolute paths so they work regardless of where you run from:
```groovy
csv.text = rows.collect { row ->
    row.files.collect { it.toAbsolutePath().toString() }.join(',')
}.join('\n')
```

### 3. **Preserve Metadata**
CSVs should contain all metadata needed to reconstruct the sample information:
- patient/subject ID
- sample ID
- sex (for sex-chromosome aware processing)
- status (normal=0, tumor=1)
- Any other relevant metadata

### 4. **Validate Inputs Based on Step**
Each step should validate that the required columns exist in the input CSV:
```groovy
if (params.step == 'variant_calling') {
    def required = ['patient', 'sample', 'bam', 'bai']
    def header = file(params.input).readLines()[0].split(',')
    def missing = required - header
    if (missing) {
        error "Missing columns for step ${params.step}: ${missing.join(', ')}"
    }
}
```

### 5. **Document the CSV Formats**
Create clear documentation for each CSV format:
```
## aligned.csv
| Column  | Required | Description                    |
|---------|----------|--------------------------------|
| patient | Yes      | Patient/subject identifier     |
| sample  | Yes      | Sample identifier              |
| sex     | No       | XX, XY, etc. Default: NA       |
| status  | No       | 0=normal, 1=tumor. Default: 0  |
| bam     | Yes      | Path to aligned BAM file       |
| bai     | Yes      | Path to BAM index              |
```

---

## Summary

The nf-core/sarek restart mechanism works by:

1. **Publishing intermediate files** at each major step
2. **Auto-generating CSV samplesheets** pointing to those files
3. **Using `--step` parameter** to control entry point
4. **Conditional workflow logic** that loads from checkpoints when needed

This approach is **completely independent of Nextflow's `-resume`** and works even after deleting the `work/` directory, moving to a different machine, or starting a fresh Nextflow session.

The trade-off is increased storage usage (you're keeping intermediate files), but the benefit is robust restartability that doesn't depend on cache integrity.
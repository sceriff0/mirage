# Nextflow Pipeline Refactoring Guide for AI Agents

This document provides structured guidance for AI agents tasked with evaluating, modifying, or building Nextflow pipelines according to modern best practices and nf-core standards.

---

## Table of Contents

1. [Initial Assessment Protocol](#1-initial-assessment-protocol)
2. [DSL Version Migration](#2-dsl-version-migration)
3. [Project Structure Refactoring](#3-project-structure-refactoring)
4. [Process Refactoring Patterns](#4-process-refactoring-patterns)
5. [Channel Design Patterns](#5-channel-design-patterns)
6. [Configuration Modernization](#6-configuration-modernization)
7. [Container and Reproducibility](#7-container-and-reproducibility)
8. [Resource Management](#8-resource-management)
9. [Input Validation](#9-input-validation)
10. [Output and Publishing](#10-output-and-publishing)
11. [Error Handling](#11-error-handling)
12. [Testing Infrastructure](#12-testing-infrastructure)
13. [Documentation Requirements](#13-documentation-requirements)
14. [Anti-Patterns to Fix](#14-anti-patterns-to-fix)
15. [Refactoring Checklist](#15-refactoring-checklist)

---

## 1. Initial Assessment Protocol

Before making changes, assess the pipeline's current state by answering these questions:

### 1.1 DSL Version Check

```groovy
// Look for this at the top of main.nf
nextflow.enable.dsl=2  // Modern - keep
// OR
nextflow.enable.dsl=1  // Legacy - requires migration
// OR no declaration      // Defaults to DSL1 in older Nextflow - requires migration
```

**Action**: If DSL1 or no declaration, prioritize DSL2 migration (see Section 2).

### 1.2 Structure Assessment

Check for the presence of these directories and files:

| Path | Required | Purpose |
|------|----------|---------|
| `main.nf` | ✅ | Entry point |
| `nextflow.config` | ✅ | Main configuration |
| `modules/` | ✅ | Process definitions |
| `subworkflows/` | Recommended | Workflow components |
| `conf/` | ✅ | Configuration profiles |
| `bin/` | If scripts exist | Executable scripts |
| `assets/` | Recommended | Static assets |
| `lib/` | Optional | Groovy utilities |
| `tests/` | Recommended | Test definitions |

### 1.3 Quick Health Indicators

Scan for these red flags:

```groovy
// 🔴 RED FLAGS - Must fix
process.executor = 'local'           // Hardcoded executor
params.outdir = '/absolute/path'     // Hardcoded absolute paths
"source activate myenv"              // Conda activation in script
container = 'ubuntu:latest'          // Mutable container tag
file("/hardcoded/path/reference.fa") // Hardcoded file paths

// 🟡 WARNINGS - Should fix
publishDir "results"                 // Missing params reference
cpus = 8                             // Static resource allocation
errorStrategy 'terminate'            // No retry logic
```

---

## 2. DSL Version Migration

### 2.1 DSL1 to DSL2 Migration Steps

**Step 1**: Add DSL2 declaration

```groovy
// Add at the very top of main.nf
nextflow.enable.dsl=2
```

**Step 2**: Convert process outputs from `into` to `emit`

```groovy
// ❌ DSL1 Pattern
process FASTQC {
    input:
    set sample_id, file(reads) from reads_ch
    
    output:
    file("*.html") into fastqc_html
    file("*.zip") into fastqc_zip
    
    script:
    """
    fastqc $reads
    """
}

// ✅ DSL2 Pattern
process FASTQC {
    input:
    tuple val(sample_id), path(reads)
    
    output:
    path("*.html"), emit: html
    path("*.zip"), emit: zip
    
    script:
    """
    fastqc $reads
    """
}
```

**Step 3**: Convert channel connections to workflow blocks

```groovy
// ❌ DSL1 - Implicit channel flow
reads_ch = Channel.fromFilePairs('data/*_{1,2}.fq.gz')
FASTQC(reads_ch)
MULTIQC(fastqc_html.collect())

// ✅ DSL2 - Explicit workflow block
workflow {
    reads_ch = Channel.fromFilePairs('data/*_{1,2}.fq.gz')
    
    FASTQC(reads_ch)
    MULTIQC(FASTQC.out.html.collect())
}
```

**Step 4**: Convert `set` to `tuple`, `file` to `path`

```groovy
// ❌ DSL1 keywords
set sample_id, file(reads) from reads_ch
file reference from ref_ch

// ✅ DSL2 keywords
tuple val(sample_id), path(reads)
path(reference)
```

**Step 5**: Extract processes into modules

```groovy
// modules/local/fastqc.nf
process FASTQC {
    tag "$meta.id"
    label 'process_medium'
    
    container 'quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0'
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.zip"), emit: zip
    
    script:
    """
    fastqc $reads --threads $task.cpus
    """
}

// main.nf
include { FASTQC } from './modules/local/fastqc'
```

---

## 3. Project Structure Refactoring

### 3.1 Target Directory Structure

Transform any pipeline to this structure:

```
pipeline/
├── main.nf                      # Minimal entry point
├── nextflow.config              # Main config (imports others)
├── nextflow_schema.json         # Parameter validation schema
├── README.md
├── CHANGELOG.md
├── LICENSE
│
├── workflows/
│   └── my_pipeline.nf           # Main workflow logic
│
├── subworkflows/
│   ├── local/
│   │   ├── input_check.nf       # Samplesheet parsing
│   │   ├── prepare_genome.nf    # Reference preparation
│   │   └── bam_qc.nf            # BAM quality control
│   └── nf-core/                 # Imported nf-core subworkflows
│
├── modules/
│   ├── local/
│   │   └── custom_tool.nf       # Pipeline-specific modules
│   └── nf-core/                 # Imported nf-core modules
│       ├── fastqc/
│       ├── multiqc/
│       └── samtools/
│
├── conf/
│   ├── base.config              # Default process resources
│   ├── modules.config           # Module-specific settings
│   ├── test.config              # Test profile
│   ├── test_full.config         # Full test profile
│   └── igenomes.config          # Reference genome paths
│
├── assets/
│   ├── multiqc_config.yml
│   ├── schema_input.json        # Samplesheet schema
│   └── email_template.html
│
├── bin/
│   └── check_samplesheet.py     # Custom scripts
│
├── lib/
│   ├── WorkflowMain.groovy
│   └── NfcoreTemplate.groovy
│
└── tests/
    ├── main.nf.test
    └── modules/
        └── fastqc.nf.test
```

### 3.2 Minimal main.nf Template

```groovy
#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES/SUBWORKFLOWS
========================================================================================
*/

include { MY_PIPELINE } from './workflows/my_pipeline'
include { validateParameters; paramsSummaryLog } from 'plugin/nf-validation'

/*
========================================================================================
    NAMED WORKFLOWS
========================================================================================
*/

workflow {
    // Validate parameters
    validateParameters()
    
    // Print parameter summary
    log.info paramsSummaryLog(workflow)
    
    // Run main workflow
    MY_PIPELINE()
}

/*
========================================================================================
    COMPLETION HANDLERS
========================================================================================
*/

workflow.onComplete {
    if (workflow.success) {
        log.info "Pipeline completed successfully!"
    } else {
        log.error "Pipeline failed"
    }
}
```

### 3.3 Moving Inline Processes to Modules

**Before** (monolithic main.nf):

```groovy
// main.nf - 500+ lines with embedded processes
process ALIGN {
    ...
}
process SORT {
    ...
}
process CALL_VARIANTS {
    ...
}

workflow {
    ALIGN(reads)
    SORT(ALIGN.out)
    CALL_VARIANTS(SORT.out)
}
```

**After** (modular structure):

```groovy
// modules/local/align.nf
process ALIGN { ... }

// modules/local/sort.nf  
process SORT { ... }

// modules/local/call_variants.nf
process CALL_VARIANTS { ... }

// workflows/my_pipeline.nf
include { ALIGN } from '../modules/local/align'
include { SORT } from '../modules/local/sort'
include { CALL_VARIANTS } from '../modules/local/call_variants'

workflow MY_PIPELINE {
    take:
    reads
    
    main:
    ALIGN(reads)
    SORT(ALIGN.out.bam)
    CALL_VARIANTS(SORT.out.sorted_bam)
    
    emit:
    variants = CALL_VARIANTS.out.vcf
}

// main.nf
include { MY_PIPELINE } from './workflows/my_pipeline'

workflow {
    reads_ch = Channel.fromPath(params.input)
    MY_PIPELINE(reads_ch)
}
```

---

## 4. Process Refactoring Patterns

### 4.1 Standard Process Template

Every process should follow this template:

```groovy
process TOOL_ACTION {
    tag "$meta.id"
    label 'process_medium'
    
    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/tool:1.0.0--h1234567_0' :
        'quay.io/biocontainers/tool:1.0.0--h1234567_0' }"
    
    input:
    tuple val(meta), path(input_file)
    path(reference)
    
    output:
    tuple val(meta), path("*.output"), emit: results
    path "versions.yml"             , emit: versions
    
    when:
    task.ext.when == null || task.ext.when
    
    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    tool_command \\
        $args \\
        --threads $task.cpus \\
        --input $input_file \\
        --reference $reference \\
        --output ${prefix}.output
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        tool: \$(tool_command --version | head -n1)
    END_VERSIONS
    """
    
    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.output
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        tool: 1.0.0
    END_VERSIONS
    """
}
```

### 4.2 The Meta Map Pattern

**Always use meta maps** to carry sample information:

```groovy
// ❌ Bad - Losing metadata
process ALIGN {
    input:
    tuple val(sample_id), path(reads)
    
    output:
    path("*.bam")  // Lost sample association!
}

// ✅ Good - Preserving metadata
process ALIGN {
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.bam"), emit: bam
}
```

**Standard meta map structure**:

```groovy
def meta = [
    id: 'sample1',           // Required: unique identifier
    single_end: false,       // Required for reads
    strandedness: 'auto',    // Optional: analysis parameters
    patient: 'patient1',     // Optional: grouping info
    condition: 'treatment'   // Optional: experimental info
]
```

### 4.3 Splitting Monolithic Processes

**Before** - Process doing too much:

```groovy
process ALIGN_SORT_INDEX {
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.sorted.bam"), path("*.bai")
    
    script:
    """
    bwa mem ref.fa $reads | samtools sort -o ${meta.id}.sorted.bam
    samtools index ${meta.id}.sorted.bam
    """
}
```

**After** - Single responsibility:

```groovy
process BWA_MEM {
    input:
    tuple val(meta), path(reads)
    path(reference)
    
    output:
    tuple val(meta), path("*.bam"), emit: bam
    
    script:
    """
    bwa mem -t $task.cpus $reference $reads > ${meta.id}.bam
    """
}

process SAMTOOLS_SORT {
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.sorted.bam"), emit: sorted_bam
    
    script:
    """
    samtools sort -@ $task.cpus -o ${meta.id}.sorted.bam $bam
    """
}

process SAMTOOLS_INDEX {
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.bai"), emit: bai
    
    script:
    """
    samtools index $bam
    """
}
```

### 4.4 Adding Version Tracking

Every process must emit software versions:

```groovy
process MY_TOOL {
    output:
    tuple val(meta), path("*.out"), emit: results
    path "versions.yml"          , emit: versions
    
    script:
    """
    my_tool --input $input --output output.out
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mytool: \$(my_tool --version 2>&1 | sed 's/.*version //')
        python: \$(python --version | sed 's/Python //')
    END_VERSIONS
    """
}
```

---

## 5. Channel Design Patterns

### 5.1 Input Channel Creation

**From samplesheet (recommended)**:

```groovy
// subworkflows/local/input_check.nf
workflow INPUT_CHECK {
    take:
    samplesheet
    
    main:
    Channel.fromPath(samplesheet)
        .splitCsv(header: true, sep: ',')
        .map { row -> 
            def meta = [
                id: row.sample,
                single_end: row.fastq_2 ? false : true
            ]
            def reads = row.fastq_2 ? 
                [ file(row.fastq_1), file(row.fastq_2) ] : 
                [ file(row.fastq_1) ]
            [ meta, reads ]
        }
        .set { reads }
    
    emit:
    reads
}
```

**From file patterns**:

```groovy
// Paired-end reads
Channel
    .fromFilePairs(params.reads, checkIfExists: true)
    .map { id, files -> 
        def meta = [id: id, single_end: false]
        [meta, files]
    }
    .set { reads_ch }

// Single files with metadata extraction
Channel
    .fromPath(params.bams, checkIfExists: true)
    .map { bam ->
        def meta = [id: bam.baseName.replaceAll(/\.sorted$/, '')]
        [meta, bam]
    }
    .set { bam_ch }
```

### 5.2 Channel Joining Patterns

**Join by meta.id**:

```groovy
// When you need to combine outputs from different processes
bam_ch      // [ [id: 'sample1'], bam ]
    .join(bai_ch)  // [ [id: 'sample1'], bai ]
    .set { bam_bai_ch }  // [ [id: 'sample1'], bam, bai ]

// With explicit key
bam_ch
    .map { meta, bam -> [meta.id, meta, bam] }
    .join(
        bai_ch.map { meta, bai -> [meta.id, bai] }
    )
    .map { id, meta, bam, bai -> [meta, bam, bai] }
    .set { bam_bai_ch }
```

**Combine channels**:

```groovy
// Add reference to every sample
reads_ch
    .combine(reference_ch)
    .set { reads_with_ref_ch }
// Result: [ meta, reads, reference ]
```

### 5.3 Branching Logic

```groovy
reads_ch
    .branch {
        single: it[0].single_end
        paired: !it[0].single_end
    }
    .set { branched }

ALIGN_SE(branched.single)
ALIGN_PE(branched.paired)

// Merge back together
ALIGN_SE.out.bam
    .mix(ALIGN_PE.out.bam)
    .set { all_bams }
```

### 5.4 Collecting and Grouping

```groovy
// Collect all items (for MultiQC, etc.)
fastqc_ch
    .map { meta, files -> files }
    .collect()
    .set { all_fastqc }

// Group by patient/condition
bam_ch
    .map { meta, bam -> [meta.patient, meta, bam] }
    .groupTuple(by: 0)
    .map { patient, metas, bams -> [metas[0], bams] }
    .set { grouped_bams }
```

### 5.5 Handling Optional Inputs

```groovy
// Create empty channel if parameter not provided
reference_ch = params.reference ? 
    Channel.fromPath(params.reference, checkIfExists: true) : 
    Channel.empty()

// Handle optional process inputs
process ANNOTATE {
    input:
    tuple val(meta), path(vcf)
    path(db)  // Can be empty
    
    script:
    def db_arg = db ? "--db $db" : ''
    """
    annotate $vcf $db_arg
    """
}
```

---

## 6. Configuration Modernization

### 6.1 Main nextflow.config Structure

```groovy
/*
========================================================================================
    DEFAULT PARAMETERS
========================================================================================
*/

params {
    // Input options
    input                      = null
    outdir                     = './results'
    
    // Reference options
    genome                     = null
    igenomes_base              = 's3://ngi-igenomes/igenomes'
    igenomes_ignore            = false
    
    // Pipeline options
    skip_fastqc                = false
    skip_multiqc               = false
    save_intermediates         = false
    
    // Resource limits
    max_memory                 = '128.GB'
    max_cpus                   = 16
    max_time                   = '240.h'
    
    // Boilerplate options
    publish_dir_mode           = 'copy'
    email                      = null
    email_on_fail              = null
    help                       = false
    version                    = false
    
    // Config options
    config_profile_name        = null
    config_profile_description = null
}

/*
========================================================================================
    LOAD CONFIGS
========================================================================================
*/

// Load base.config by default
includeConfig 'conf/base.config'

// Load modules.config
includeConfig 'conf/modules.config'

// Load nf-core custom profiles
try {
    includeConfig "${params.custom_config_base}/nfcore_custom.config"
} catch (Exception e) {
    System.err.println("WARNING: Could not load custom config")
}

// Load igenomes config if applicable
if (!params.igenomes_ignore) {
    includeConfig 'conf/igenomes.config'
}

/*
========================================================================================
    PROFILES
========================================================================================
*/

profiles {
    debug {
        dumpHashes             = true
        process.beforeScript   = 'echo $HOSTNAME'
        cleanup                = false
    }
    docker {
        docker.enabled         = true
        docker.userEmulation   = true
        conda.enabled          = false
        singularity.enabled    = false
        podman.enabled         = false
    }
    singularity {
        singularity.enabled    = true
        singularity.autoMounts = true
        conda.enabled          = false
        docker.enabled         = false
        podman.enabled         = false
    }
    conda {
        conda.enabled          = true
        docker.enabled         = false
        singularity.enabled    = false
    }
    test {
        includeConfig 'conf/test.config'
    }
    test_full {
        includeConfig 'conf/test_full.config'
    }
}

/*
========================================================================================
    RESOURCE LIMITS
========================================================================================
*/

// Function to ensure resources don't exceed limits
def check_max(obj, type) {
    if (type == 'memory') {
        try {
            if (obj.compareTo(params.max_memory as nextflow.util.MemoryUnit) == 1)
                return params.max_memory as nextflow.util.MemoryUnit
            else
                return obj
        } catch (all) {
            println "WARNING: Max memory '${params.max_memory}' is not valid"
            return obj
        }
    } else if (type == 'time') {
        try {
            if (obj.compareTo(params.max_time as nextflow.util.Duration) == 1)
                return params.max_time as nextflow.util.Duration
            else
                return obj
        } catch (all) {
            println "WARNING: Max time '${params.max_time}' is not valid"
            return obj
        }
    } else if (type == 'cpus') {
        try {
            return Math.min(obj, params.max_cpus as int)
        } catch (all) {
            println "WARNING: Max cpus '${params.max_cpus}' is not valid"
            return obj
        }
    }
}

/*
========================================================================================
    MANIFEST
========================================================================================
*/

manifest {
    name            = 'org/pipeline'
    author          = 'Your Name'
    homePage        = 'https://github.com/org/pipeline'
    description     = 'Pipeline description'
    mainScript      = 'main.nf'
    nextflowVersion = '!>=23.04.0'
    version         = '1.0.0'
}
```

### 6.2 Base Config (conf/base.config)

```groovy
/*
========================================================================================
    BASE CONFIG - Default process requirements
========================================================================================
*/

process {
    // Default error strategy
    errorStrategy = { task.exitStatus in ((130..145) + 104) ? 'retry' : 'finish' }
    maxRetries    = 1
    maxErrors     = '-1'

    // Default resources
    cpus   = { check_max( 1    * task.attempt, 'cpus'   ) }
    memory = { check_max( 6.GB * task.attempt, 'memory' ) }
    time   = { check_max( 4.h  * task.attempt, 'time'   ) }

    // Process labels
    withLabel: 'process_single' {
        cpus   = { check_max( 1                  , 'cpus'   ) }
        memory = { check_max( 6.GB * task.attempt, 'memory' ) }
        time   = { check_max( 4.h  * task.attempt, 'time'   ) }
    }
    withLabel: 'process_low' {
        cpus   = { check_max( 2     * task.attempt, 'cpus'   ) }
        memory = { check_max( 12.GB * task.attempt, 'memory' ) }
        time   = { check_max( 4.h   * task.attempt, 'time'   ) }
    }
    withLabel: 'process_medium' {
        cpus   = { check_max( 6     * task.attempt, 'cpus'   ) }
        memory = { check_max( 36.GB * task.attempt, 'memory' ) }
        time   = { check_max( 8.h   * task.attempt, 'time'   ) }
    }
    withLabel: 'process_high' {
        cpus   = { check_max( 12    * task.attempt, 'cpus'   ) }
        memory = { check_max( 72.GB * task.attempt, 'memory' ) }
        time   = { check_max( 16.h  * task.attempt, 'time'   ) }
    }
    withLabel: 'process_long' {
        time   = { check_max( 20.h  * task.attempt, 'time'   ) }
    }
    withLabel: 'process_high_memory' {
        memory = { check_max( 200.GB * task.attempt, 'memory' ) }
    }
    withLabel: 'error_ignore' {
        errorStrategy = 'ignore'
    }
    withLabel: 'error_retry' {
        errorStrategy = 'retry'
        maxRetries    = 2
    }
}
```

### 6.3 Modules Config (conf/modules.config)

```groovy
/*
========================================================================================
    MODULES CONFIG - Tool-specific settings
========================================================================================
*/

process {
    // Default publish settings
    publishDir = [
        path: { "${params.outdir}/${task.process.tokenize(':')[-1].toLowerCase()}" },
        mode: params.publish_dir_mode,
        saveAs: { filename -> filename.equals('versions.yml') ? null : filename }
    ]

    // FastQC
    withName: 'FASTQC' {
        ext.args = '--quiet'
        publishDir = [
            path: { "${params.outdir}/fastqc" },
            mode: params.publish_dir_mode,
            pattern: "*.{html,zip}"
        ]
    }

    // MultiQC
    withName: 'MULTIQC' {
        ext.args = { params.multiqc_title ? "--title \"$params.multiqc_title\"" : '' }
        publishDir = [
            path: { "${params.outdir}/multiqc" },
            mode: params.publish_dir_mode
        ]
    }

    // BWA MEM
    withName: 'BWA_MEM' {
        ext.args = { "-K 100000000 -Y" }
        publishDir = [
            [
                path: { "${params.outdir}/alignment" },
                mode: params.publish_dir_mode,
                pattern: "*.bam",
                enabled: params.save_intermediates
            ],
            [
                path: { "${params.outdir}/alignment/logs" },
                mode: params.publish_dir_mode,
                pattern: "*.log"
            ]
        ]
    }

    // Variant calling
    withName: 'GATK_HAPLOTYPECALLER' {
        ext.args = '--dont-use-soft-clipped-bases'
        memory = { check_max( 16.GB * task.attempt, 'memory' ) }
    }
}
```

---

## 7. Container and Reproducibility

### 7.1 Container Requirements

Every process MUST have a container defined:

```groovy
process MY_TOOL {
    // Support both Docker and Singularity
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/tool:1.2.3--h1234567_0' :
        'quay.io/biocontainers/tool:1.2.3--h1234567_0' }"
    
    // OR for custom containers
    container 'ghcr.io/org/custom-container:v1.0.0'
}
```

### 7.2 Container Version Rules

```groovy
// ❌ Never use mutable tags
container 'ubuntu:latest'
container 'tool:stable'

// ✅ Always use immutable versions
container 'ubuntu:22.04'
container 'quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0'
container 'ghcr.io/org/tool:v1.2.3@sha256:abc123...'
```

### 7.3 Conda Environment Files

If supporting conda, create `environment.yml` in each module directory:

```yaml
# modules/local/my_tool/environment.yml
name: my_tool
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - my_tool=1.2.3
  - python=3.11
  - pandas=2.0
```

---

## 8. Resource Management

### 8.1 Dynamic Resource Allocation

```groovy
process MEMORY_SCALING {
    label 'process_high'
    
    // Scale memory based on input size
    memory { check_max( input_file.size() * 3 + 4.GB, 'memory' ) }
    
    // Or scale with retries
    memory { check_max( 32.GB * Math.pow(2, task.attempt - 1), 'memory' ) }
    
    script:
    def avail_mem = (task.memory.toGiga() - 1)
    """
    tool --memory ${avail_mem}G $input_file
    """
}
```

### 8.2 Appropriate Label Selection

| Task Type | Label | Typical Resources |
|-----------|-------|-------------------|
| Simple file operations | `process_single` | 1 CPU, 2-6 GB |
| Read QC, trimming | `process_low` | 2-4 CPU, 8-12 GB |
| Alignment, quantification | `process_medium` | 6-8 CPU, 24-36 GB |
| Variant calling, assembly | `process_high` | 12+ CPU, 48-72 GB |
| Genome indexing | `process_high_memory` | 4 CPU, 100+ GB |
| Long-running jobs | `process_long` | Variable, 24+ hours |

### 8.3 Exit Code Handling

```groovy
process {
    // Retry on resource-related failures
    errorStrategy = { 
        task.exitStatus in [
            104,  // SIGBUS
            134,  // SIGABRT  
            137,  // SIGKILL (OOM)
            139,  // SIGSEGV
            140,  // SIGTERM
            143   // SIGTERM
        ] ? 'retry' : 'finish' 
    }
    maxRetries = 2
}
```

---

## 9. Input Validation

### 9.1 Parameter Schema (nextflow_schema.json)

```json
{
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "Pipeline Parameters",
    "type": "object",
    "definitions": {
        "input_output_options": {
            "title": "Input/output options",
            "type": "object",
            "required": ["input", "outdir"],
            "properties": {
                "input": {
                    "type": "string",
                    "format": "file-path",
                    "exists": true,
                    "pattern": "^\\S+\\.csv$",
                    "description": "Path to samplesheet CSV"
                },
                "outdir": {
                    "type": "string",
                    "format": "directory-path",
                    "description": "Output directory"
                }
            }
        },
        "reference_genome_options": {
            "title": "Reference genome options",
            "type": "object",
            "properties": {
                "genome": {
                    "type": "string",
                    "description": "Reference genome ID",
                    "enum": ["GRCh37", "GRCh38", "GRCm38", "GRCm39"]
                },
                "fasta": {
                    "type": "string",
                    "format": "file-path",
                    "exists": true,
                    "pattern": "^\\S+\\.fn?a(sta)?(\\.gz)?$"
                }
            }
        }
    },
    "allOf": [
        { "$ref": "#/definitions/input_output_options" },
        { "$ref": "#/definitions/reference_genome_options" }
    ]
}
```

### 9.2 Samplesheet Schema (assets/schema_input.json)

```json
{
    "$schema": "http://json-schema.org/draft-07/schema",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["sample", "fastq_1"],
        "properties": {
            "sample": {
                "type": "string",
                "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$",
                "errorMessage": "Sample name must start with letter, contain only alphanumeric, underscore, or hyphen"
            },
            "fastq_1": {
                "type": "string",
                "format": "file-path",
                "exists": true,
                "pattern": "^\\S+\\.f(ast)?q(\\.gz)?$"
            },
            "fastq_2": {
                "type": "string",
                "format": "file-path",
                "exists": true,
                "pattern": "^\\S+\\.f(ast)?q(\\.gz)?$"
            },
            "strandedness": {
                "type": "string",
                "enum": ["auto", "forward", "reverse", "unstranded"],
                "default": "auto"
            }
        }
    }
}
```

### 9.3 Using nf-validation Plugin

```groovy
// main.nf
plugins {
    id 'nf-validation@1.1.3'
}

include { validateParameters; paramsHelp; paramsSummaryLog; samplesheetToList } from 'plugin/nf-validation'

// Show help if requested
if (params.help) {
    log.info paramsHelp("nextflow run main.nf --input samplesheet.csv")
    exit 0
}

// Validate parameters
validateParameters()

// Convert samplesheet to channel
Channel
    .fromList(samplesheetToList(params.input, "assets/schema_input.json"))
    .map { meta, fastq_1, fastq_2 ->
        def reads = fastq_2 ? [fastq_1, fastq_2] : [fastq_1]
        [meta, reads]
    }
    .set { reads_ch }
```

---

## 10. Output and Publishing

### 10.1 Organized Output Structure

```
results/
├── fastqc/
│   ├── raw/
│   └── trimmed/
├── trimming/
│   └── logs/
├── alignment/
│   ├── bam/
│   ├── stats/
│   └── logs/
├── variants/
│   ├── vcf/
│   ├── filtered/
│   └── annotations/
├── multiqc/
│   ├── multiqc_report.html
│   └── multiqc_data/
└── pipeline_info/
    ├── software_versions.yml
    ├── execution_report.html
    └── pipeline_dag.svg
```

### 10.2 Multi-Path Publishing

```groovy
process VARIANT_CALLING {
    publishDir = [
        [
            path: { "${params.outdir}/variants/vcf" },
            mode: params.publish_dir_mode,
            pattern: "*.vcf.gz{,.tbi}"
        ],
        [
            path: { "${params.outdir}/variants/stats" },
            mode: params.publish_dir_mode,
            pattern: "*.stats"
        ],
        [
            path: { "${params.outdir}/variants/logs" },
            mode: params.publish_dir_mode,
            pattern: "*.log",
            enabled: params.save_intermediates
        ]
    ]
}
```

### 10.3 Version Collection

```groovy
// Collect all versions
workflow {
    // ... process calls ...
    
    // Collect all version outputs
    ch_versions = Channel.empty()
    ch_versions = ch_versions.mix(FASTQC.out.versions.first())
    ch_versions = ch_versions.mix(TRIM_GALORE.out.versions.first())
    ch_versions = ch_versions.mix(BWA_MEM.out.versions.first())
    
    CUSTOM_DUMPSOFTWAREVERSIONS(ch_versions.unique().collectFile(name: 'collated_versions.yml'))
}
```

---

## 11. Error Handling

### 11.1 Error Strategy Matrix

| Scenario | Strategy | Use Case |
|----------|----------|----------|
| Resource exhaustion | `retry` | OOM, timeouts |
| Tool failure | `finish` | Continue other samples |
| Optional step | `ignore` | Non-critical QC |
| Critical failure | `terminate` | Reference issues |

### 11.2 Conditional Execution

```groovy
// Skip based on parameter
workflow {
    if (!params.skip_qc) {
        FASTQC(reads_ch)
        ch_fastqc = FASTQC.out.zip
    } else {
        ch_fastqc = Channel.empty()
    }
}

// Skip based on input
process ANNOTATE {
    when:
    task.ext.when == null || task.ext.when
    
    // In modules.config:
    // withName: 'ANNOTATE' { ext.when = { !params.skip_annotation } }
}
```

### 11.3 Optional Outputs

```groovy
process MAY_FAIL_PARTIALLY {
    errorStrategy 'ignore'
    
    output:
    tuple val(meta), path("*.primary.out")  , emit: primary
    tuple val(meta), path("*.secondary.out"), emit: secondary, optional: true
    
    script:
    """
    primary_tool $input > ${meta.id}.primary.out
    secondary_tool $input > ${meta.id}.secondary.out || true
    """
}
```

---

## 12. Testing Infrastructure

### 12.1 Test Profile (conf/test.config)

```groovy
/*
========================================================================================
    TEST CONFIG - Minimal test dataset
========================================================================================
*/

params {
    config_profile_name        = 'Test profile'
    config_profile_description = 'Minimal test dataset to check pipeline function'
    
    // Limit resources
    max_cpus   = 2
    max_memory = '6.GB'
    max_time   = '6.h'
    
    // Test input data
    input  = 'https://raw.githubusercontent.com/org/test-datasets/main/samplesheet.csv'
    
    // Genome references
    genome = null
    fasta  = 'https://raw.githubusercontent.com/org/test-datasets/main/reference/genome.fa'
    gtf    = 'https://raw.githubusercontent.com/org/test-datasets/main/reference/genes.gtf'
}
```

### 12.2 nf-test Setup

```groovy
// tests/main.nf.test
nextflow_pipeline {
    name "Test full pipeline"
    script "main.nf"
    
    test("Should run with test profile") {
        when {
            params {
                outdir = "$outputDir"
            }
        }
        
        then {
            assert workflow.success
            assert path("$outputDir/multiqc/multiqc_report.html").exists()
            assert path("$outputDir/pipeline_info/software_versions.yml").exists()
        }
    }
}

// tests/modules/fastqc.nf.test
nextflow_process {
    name "Test FASTQC"
    script "modules/local/fastqc.nf"
    process "FASTQC"
    
    test("Should run on paired-end reads") {
        when {
            process {
                """
                input[0] = [
                    [ id: 'test', single_end: false ],
                    [ file(params.test_data['sarscov2']['illumina']['test_1_fastq_gz'], checkIfExists: true),
                      file(params.test_data['sarscov2']['illumina']['test_2_fastq_gz'], checkIfExists: true) ]
                ]
                """
            }
        }
        
        then {
            assert process.success
            assert snapshot(process.out).match()
        }
    }
    
    test("Should run on single-end reads") {
        when {
            process {
                """
                input[0] = [
                    [ id: 'test', single_end: true ],
                    [ file(params.test_data['sarscov2']['illumina']['test_1_fastq_gz'], checkIfExists: true) ]
                ]
                """
            }
        }
        
        then {
            assert process.success
            assert snapshot(process.out).match()
        }
    }
}
```

### 12.3 GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: nf-core CI
on:
  push:
    branches: [main, dev]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        NXF_VER:
          - "23.04.0"
          - "latest-everything"
    steps:
      - uses: actions/checkout@v4
      
      - uses: nf-core/setup-nextflow@v2
        with:
          version: ${{ matrix.NXF_VER }}
      
      - name: Run pipeline with test profile
        run: |
          nextflow run ${GITHUB_WORKSPACE} -profile test,docker --outdir ./results
      
      - name: Check outputs
        run: |
          test -f ./results/multiqc/multiqc_report.html
```

---

## 13. Documentation Requirements

### 13.1 README.md Template

```markdown
# Pipeline Name

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A523.04.0-23aa62.svg)](https://www.nextflow.io/)
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed?logo=docker)](https://www.docker.com/)
[![run with singularity](https://img.shields.io/badge/run%20with-singularity-1d355c.svg)](https://sylabs.io/docs/)

## Introduction

Brief description of what the pipeline does.

## Pipeline Summary

1. Read QC ([FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/))
2. Read trimming ([Trim Galore!](https://github.com/FelixKrueger/TrimGalore))
3. Alignment ([BWA-MEM](https://github.com/lh3/bwa))
4. ... additional steps

## Quick Start

1. Install [`Nextflow`](https://www.nextflow.io/docs/latest/getstarted.html#installation) (`>=23.04.0`)

2. Install [`Docker`](https://docs.docker.com/get-docker/) or [`Singularity`](https://sylabs.io/guides/latest/user-guide/)

3. Download the pipeline and run:

   ```bash
   nextflow run org/pipeline -profile docker --input samplesheet.csv --outdir results
   ```

## Documentation

- [Usage](docs/usage.md)
- [Output](docs/output.md)
- [Parameters](docs/parameters.md)

## Credits

Developed by [Author Name](https://github.com/author).

## Citations

If you use this pipeline, please cite:

> Author et al. (2024). Pipeline Name: Description. Journal. DOI: xxx
```

### 13.2 Usage Documentation (docs/usage.md)

Document every parameter with examples, typical values, and edge cases.

### 13.3 Output Documentation (docs/output.md)

Document every output file, its format, and how to interpret it.

---

## 14. Anti-Patterns to Fix

### 14.1 Common Anti-Patterns and Solutions

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Hardcoded paths | Not portable | Use `params.` or `projectDir` |
| `latest` container tags | Not reproducible | Pin exact versions |
| No error handling | Pipeline stops unexpectedly | Add `errorStrategy` |
| Monolithic processes | Hard to debug/reuse | Split into single-purpose |
| Missing `tag` | Hard to track samples | Add `tag "$meta.id"` |
| No version tracking | Not reproducible | Emit `versions.yml` |
| Static resources | Wastes cluster resources | Use labels and scaling |
| `file()` in process | Stage issues | Use `path()` input |
| Missing `emit:` names | Hard to reference outputs | Name all outputs |
| No `stub:` block | Slow testing | Add stub commands |

### 14.2 Code Smell Detection

```groovy
// 🔴 CRITICAL - Fix immediately
"/absolute/path"                 // Hardcoded path
"source activate"                // Conda in script
container = null                 // Missing container
file("/path/to/file")           // file() instead of input

// 🟡 WARNING - Should fix
publishDir "results"            // Missing params.outdir
cpus = 8                        // Static, not label-based
errorStrategy 'terminate'       // No retry
output: file("*.txt")           // Missing emit name

// 🔵 SUGGESTION - Consider fixing
// No tag directive
// No stub block
// Long script blocks (>50 lines)
// Repeated code patterns
```

---

## 15. Refactoring Checklist

Use this checklist when refactoring a pipeline:

### Phase 1: Assessment
- [ ] Identify DSL version (migrate to DSL2 if needed)
- [ ] Map current structure against target structure
- [ ] List all processes and their dependencies
- [ ] Identify hardcoded values and paths
- [ ] Check container specifications

### Phase 2: Structure
- [ ] Create standard directory structure
- [ ] Move processes to `modules/local/`
- [ ] Create subworkflows for related processes
- [ ] Create minimal `main.nf`
- [ ] Split configuration into `conf/` files

### Phase 3: Processes
- [ ] Add `tag` directive to all processes
- [ ] Add `label` for resource management
- [ ] Ensure all processes emit `versions.yml`
- [ ] Use meta maps for sample tracking
- [ ] Add `stub:` blocks for testing
- [ ] Use `task.ext.args` for tool arguments
- [ ] Add `when:` directive for conditional execution

### Phase 4: Configuration
- [ ] Create `nextflow_schema.json`
- [ ] Create samplesheet schema
- [ ] Set up `base.config` with resource labels
- [ ] Create `modules.config` for process settings
- [ ] Add `test.config` with minimal dataset
- [ ] Create profiles for docker/singularity/conda

### Phase 5: Containers
- [ ] Add container to every process
- [ ] Use immutable version tags
- [ ] Support both Docker and Singularity
- [ ] Add conda `environment.yml` if needed

### Phase 6: Testing
- [ ] Create test profile with minimal data
- [ ] Write nf-test for main workflow
- [ ] Write nf-test for critical modules
- [ ] Set up GitHub Actions CI
- [ ] Test on local, HPC, and cloud

### Phase 7: Documentation
- [ ] Write comprehensive README
- [ ] Document all parameters
- [ ] Document all outputs
- [ ] Add CHANGELOG
- [ ] Add usage examples

### Phase 8: Validation
- [ ] Run `nextflow run . -profile test,docker`
- [ ] Run `nf-core lint` (if applicable)
- [ ] Verify all outputs are published correctly
- [ ] Check resource usage is appropriate
- [ ] Validate version collection works

---

## Appendix A: Quick Reference Commands

```bash
# Lint pipeline (nf-core)
nf-core lint

# Run tests
nextflow run . -profile test,docker

# Run nf-test
nf-test test

# Generate schema from params
nf-core schema build

# Create module from template
nf-core modules create tool/subtool --author "@username"

# Check for updates
nf-core modules update --all
```

---

## Appendix B: Useful Groovy Snippets

```groovy
// Get file basename without extension
file.baseName

// Get file extension
file.extension

// Check if file exists
file.exists()

// Get file size
file.size()

// String interpolation
"${params.outdir}/${meta.id}.bam"

// Conditional expression
def value = condition ? 'yes' : 'no'

// Safe navigation (null-safe)
meta?.id ?: 'unknown'

// List operations
[1, 2, 3].collect { it * 2 }  // [2, 4, 6]
[1, 2, 3].findAll { it > 1 }  // [2, 3]
[1, 2, 3].find { it > 1 }     // 2

// Map operations
[a: 1, b: 2].collect { k, v -> "$k=$v" }  // ['a=1', 'b=2']
```

---

*This guide is designed for AI agents to systematically improve Nextflow pipelines. Follow the checklist in order and apply patterns incrementally for best results.*
# Registration System Architecture

This document describes the registration system architecture using the **Adapter Pattern** to support different registration methods while maintaining clean separation of concerns.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  REGISTRATION Subworkflow (Orchestrator)                    │
│  - Common padding logic                                     │
│  - Common grouping logic                                    │
│  - Adapter dispatch                                         │
│  - Common checkpoint logic                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼──────────────┐
        │             │              │
        ▼             ▼              ▼
┌──────────────┐ ┌─────────┐ ┌─────────┐
│ VALIS_ADAPTER│ │GPU_ADAPT│ │CPU_ADAPT│
│              │ │ER       │ │ER       │
│ Batch        │ │Pairwise │ │Pairwise │
│ Processing   │ │Processing│ │Processing│
└──────┬───────┘ └────┬────┘ └────┬────┘
       │              │            │
       ▼              ▼            ▼
 ┌──────────┐  ┌──────────┐ ┌──────────┐
 │ REGISTER │  │GPU_      │ │CPU_      │
 │ (VALIS)  │  │REGISTER  │ │REGISTER  │
 └──────────┘  └──────────┘ └──────────┘

All adapters output: [meta, file] (Standard Format)
```

**Key Principle**:
- **Adapters** handle method-specific transformations
- **Processes** focus on registration algorithm
- **Subworkflow** orchestrates common logic

---

## Adapter Contract

All registration adapters MUST implement this interface:

### Input
```groovy
take:
    ch_images         // Channel of [meta, file] tuples
    ch_grouped_meta   // Channel of [patient_id, ref_item, all_items]
```

### Output
```groovy
emit:
    registered        // Channel of [meta, file] tuples
    qc                // Channel of QC outputs
```

### Adapter Responsibilities
1. Convert standard input to method-specific format
2. Invoke registration process
3. Convert process output back to standard [meta, file] format
4. Handle reference images appropriately

---

## Registration Process Contracts

Processes can have method-specific interfaces, but adapters normalize them.

### For Batch Methods (e.g., VALIS)

**Use when**: Registration algorithm requires all images simultaneously (e.g., global optimization)

```groovy
process REGISTER {
    input:
    tuple val(patient_id), path(reference), path(all_files), val(all_metas)

    output:
    tuple val(patient_id), path("*_registered.ome.tiff"), val(all_metas), emit: registered
    path "qc/*"                                                          , emit: qc, optional: true
    path "versions.yml"                                                  , emit: versions
}
```

**Adapter converts**: `[meta, file]` → `[patient_id, ref, [files], [metas]]` → `[meta, file]`

---

### For Pairwise Methods (e.g., GPU, CPU)

**Use when**: Registration algorithm processes one moving image against reference

```groovy
process GPU_REGISTER {
    input:
    tuple val(meta), path(reference), path(moving)

    output:
    tuple val(meta), path("*_registered.ome.tiff"), emit: registered
    path "qc/*"                                    , emit: qc, optional: true
    path "versions.yml"                           , emit: versions
}
```

**Adapter converts**: `[meta, file]` → `[meta, ref, moving]` → `[meta, file]` (+ add references back)

**CRITICAL**: `meta` in output MUST be identical to `meta` in input (no modifications)

---

## Metadata Schema

### Required Fields in `meta` Map

```groovy
meta = [
    patient_id: 'P001',           // String - Patient identifier
    id: 'P001_CD3',               // String - Unique sample identifier
    is_reference: false,          // Boolean - Reference image flag
    channels: ['CD3']             // List<String> - Channel names
]
```

**Additional fields** can be added as needed (e.g., `strandedness`, `condition`, etc.)

---

## Adding New Registration Methods

### Step 1: Decide on Processing Model

**Question**: Does your registration method need all images at once, or can it process pairwise?

- **All at once** → Create batch adapter (like VALIS)
- **Pairwise** → Create pairwise adapter (like GPU/CPU)

### Step 2: Create Registration Process

**For pairwise method**:
```groovy
// modules/local/elastix_register.nf
process ELASTIX_REGISTER {
    tag "${meta.id}"
    label 'process_high'

    container 'my-elastix:1.0.0'

    input:
    tuple val(meta), path(reference), path(moving)

    output:
    tuple val(meta), path("*_registered.ome.tiff"), emit: registered
    path "qc/*"                                    , emit: qc, optional: true
    path "versions.yml"                           , emit: versions

    script:
    """
    mkdir -p qc
    elastix -f ${reference} -m ${moving} -out ${moving.simpleName}_registered.ome.tiff
    """
}
```

### Step 3: Create Adapter

**For pairwise method** (copy from [gpu_adapter.nf](cci:1:///Users/valer/Downloads/ateia/subworkflows/local/adapters/gpu_adapter.nf:0:0-0:0)):

```groovy
// subworkflows/local/adapters/elastix_adapter.nf
include { ELASTIX_REGISTER } from '../../../modules/local/elastix_register'

workflow ELASTIX_ADAPTER {
    take:
    ch_images
    ch_grouped_meta

    main:
    ch_pairs = ch_grouped_meta
        .flatMap { patient_id, ref_item, all_items ->
            def ref_file = ref_item[1]
            all_items
                .findAll { item -> !item[0].is_reference }
                .collect { moving_item -> tuple(moving_item[0], ref_file, moving_item[1]) }
        }

    ELASTIX_REGISTER(ch_pairs)

    ch_references = ch_grouped_meta.map { pid, ref, items -> ref }
    ch_all = ch_references.concat(ELASTIX_REGISTER.out.registered)

    emit:
    registered = ch_all
    qc = ELASTIX_REGISTER.out.qc
}
```

### Step 4: Add to Main Subworkflow

In [registration_refactored.nf](cci:1:///Users/valer/Downloads/ateia/subworkflows/local/registration_refactored.nf:0:0-0:0):

```groovy
include { ELASTIX_ADAPTER } from './adapters/elastix_adapter'

// In switch statement (around line 115):
case 'elastix':
    ELASTIX_ADAPTER(ch_images, ch_grouped)
    ch_registered = ELASTIX_ADAPTER.out.registered
    ch_qc = ELASTIX_ADAPTER.out.qc
    break
```

### Done!

**Total effort**:
- Process: 30 lines
- Adapter: 30 lines (mostly copy-paste)
- Integration: 4 lines

**Total: ~60 lines, low complexity**

---

## Error Handling Philosophy

### ✅ Fail-Fast (Recommended)

```groovy
if (!matched_meta) {
    error "Could not match metadata for file ${reg_file.name}. " +
          "Expected: ${basename}. Available: ${filename_to_meta.keySet()}"
}
```

**Benefits**:
- Immediate feedback
- Clear error messages
- Prevents silent data corruption

### ❌ Silent Fallbacks (Anti-Pattern)

```groovy
if (!matched_meta) {
    log.warn "Could not find metadata, using first"
    matched_meta = metas[0]  // WRONG DATA!
}
```

**Problems**:
- Pipeline continues with wrong metadata
- Errors discovered late (or never)
- Hard to debug

---

## Current Adapters

| Adapter | Processing Model | Lines of Code | Status |
|---------|-----------------|---------------|--------|
| [valis_adapter.nf](cci:1:///Users/valer/Downloads/ateia/subworkflows/local/adapters/valis_adapter.nf:0:0-0:0) | Batch (all images) | 73 | ✅ Implemented |
| [gpu_adapter.nf](cci:1:///Users/valer/Downloads/ateia/subworkflows/local/adapters/gpu_adapter.nf:0:0-0:0) | Pairwise | 57 | ✅ Implemented |
| [cpu_adapter.nf](cci:1:///Users/valer/Downloads/ateia/subworkflows/local/adapters/cpu_adapter.nf:0:0-0:0) | Pairwise | 57 | ✅ Implemented |

---

## Benefits of Adapter Pattern

### KISS (Keep It Simple, Stupid)
- Each adapter is focused and simple
- Main subworkflow is clean orchestration
- No complex branching logic

### DRY (Don't Repeat Yourself)
- Transformation logic appears once per adapter
- No duplicate input prep / output reconstruction
- Pairwise adapters share similar structure

### Extensibility
- Adding methods is trivial (copy template)
- No impact on existing methods
- Clear pattern to follow

### Robustness
- Fail-fast error handling
- Type-safe channel contracts
- Isolated testing per adapter

/*
 * REGISTER - VALIS whole-slide registration
 *
 * Performs multi-modal image registration using VALIS with SuperPoint/SuperGlue
 * feature detection. Supports rigid, non-rigid, and micro-registration stages.
 *
 * Input: Reference image path, preprocessed images, and metadata
 * Output: Registered OME-TIFF files aligned to reference coordinate space
 */
process REGISTER {
    // Uses patient_id (not meta.patient_id) because this is a fan-in process:
    // multiple per-slide metas are grouped into a single patient-level invocation.
    // patient_id is the grouping key; all_metas carries the per-slide metadata list.
    tag "${patient_id}"

    // VALIS uses the maintained upstream image (linux/amd64); we do not rebuild it (its
    // from-source libvips build is heavy and not vendored). See containers/README.md.
    // DIGEST-PINNED (ruling R6), digest only and no tag: Apptainer documents an image
    // reference as NAME[:TAG|@DIGEST] and has known bugs with the combined form, and the
    // cluster pulls through Singularity. The digest below is cdgatenbee/valis-wsi:1.0.0
    // as resolved 2026-09-02, and it is the SAME image containers/merge builds FROM.
    container 'cdgatenbee/valis-wsi@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6'

    input:
    // Use stageAs to avoid filename collision when reference is included in preproc_files
    // meta carries patient_id for publishDir consistency across all processes
    tuple val(meta), val(patient_id), path(reference, stageAs: 'ref/*'), path(preproc_files, stageAs: 'input_?/*'), val(all_metas)

    output:
    tuple val(patient_id), path("registered_slides/*_registered.ome.tiff"), val(all_metas), path("channels_manifest.json"), emit: registered
    path "versions.yml"                                                                    , emit: versions
    path("*.size.csv")                                                                     , emit: size_log
    path("preprocessed/data/*.csv")                                                        , emit: summary, optional: true
    tuple val(patient_id), path("preprocessed/data/*_registrar.pickle")                    , emit: registrar, optional: true
    // Pre-micro displacement fields for staged registration QC (reg_qc >= 2). Optional
    // because it is only requested at that level, and because a failure to write it must
    // never fail a registration — see bin/utils/stage_checkpoint.py.
    tuple val(patient_id), path("reg_stage_checkpoint")                                    , emit: stage_checkpoint, optional: true

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    // Extract reference filename from the staged path (ref/filename.tif). The pipeline
    // always identifies the reference slide (meta.is_reference) and
    // stages it here, so --reference is always passed; register.py's standalone
    // marker-based fallback is not used from the pipeline.
    def ref_filename = reference ? reference.name : ''
    def ref_arg = ref_filename ? "--reference ${ref_filename}" : ''
    // Memory mode controls feature detector, matcher, and dimension settings
    def memory_mode = params.memory_mode
    def micro_reg_fraction = params.reg_micro_reg_fraction
    def max_image_dim = params.reg_max_image_dim
    // Tier-owned VALIS knobs. Rendered ONLY when set, so an unset knob leaves register.py to take
    // it from --memory-mode's preset row -- passing an explicit 'null' would override the preset
    // with nothing. ParamUtils has already rejected these under any tier other than 'custom'.
    def valis_override_flags = [
        (params.reg_valis_max_processed_dim != null ? "--max-processed-dim ${params.reg_valis_max_processed_dim}" : null),
        (params.reg_valis_max_non_rigid_dim != null ? "--max-non-rigid-dim ${params.reg_valis_max_non_rigid_dim}" : null),
    ].findAll { it != null }
    // `?: ['']` is load-bearing, not defensive: with no overrides the interpolation below sits at
    // column 0 between two backslash-continued lines, and an EMPTY line there ends the shell
    // command early -- everything after it would run as separate commands. A single empty-string
    // entry renders a harmless whitespace-only continuation instead. Same guard, same reason, as
    // modules/local/warp_seg_qc.nf's backend_flags.
    def valis_overrides = (valis_override_flags ?: ['']).collect { "        ${it} \\" }.join('\n')
    // Micro-registration depth (0/1/2), resolved via the single-source ParamUtils helper.
    def micro_reg = ParamUtils.microRegLevelOf(params.reg_micro_reg)
    // reg_qc >= 2 scores each registration stage separately, which needs the post-non-rigid,
    // pre-micro displacement fields. VALIS composes micro into the same attribute, so nothing
    // downstream can recover them — REGISTER is the only place they can be captured.
    // ParamUtils.regQcLevel is the single source of truth (null reg_qc -> 2), shared with
    // registration.nf/add_cycle.nf so this process can't drift from the rest of the pipeline.
    def reg_qc_level = ParamUtils.regQcLevelOf(params.skip_registration_qc, params.reg_qc)
    def stage_ckpt = reg_qc_level >= 2 ? '--stage-checkpoint-dir reg_stage_checkpoint' : ''
    // JVM heap scales with retry attempts: base 32GB + 16GB per attempt.
    // Explicit null test, NOT `?:` -- Elvis treats 0 as unset, so an explicit
    // `--reg_jvm_heap_gb 0` would silently become the derived ramp instead of being honoured
    // or refused. The schema's `minimum: 1` refuses it up front; this keeps the code correct
    // without depending on that. Guarded by tests/test_nullable_numeric_params_no_elvis.py.
    def heap_request = params.reg_jvm_heap_gb != null ? params.reg_jvm_heap_gb : (32 + 16 * task.attempt)
    def jvm_heap_gb = Math.min(heap_request, task.memory.toGiga() - 4)

    """
    ${ProcessEnvelope.sizeLog(task.process, patient_id, ['ref/*', 'input_*/*'], "${patient_id}.REGISTER.size.csv")}

    mkdir -p registered_slides preprocessed

    # === PRINT REGISTRATION SETTINGS ===
    echo "========================================================================"
    echo "VALIS Registration - Attempt ${task.attempt}"
    echo "========================================================================"
    echo "Settings:"
    echo "  - memory_mode: ${memory_mode}"
    echo "  - valis overrides: ${valis_override_flags ? valis_override_flags.join(' ') : '(none - every knob from the preset)'}"
    echo "  - max_image_dim: ${max_image_dim}"
    echo "  - micro_reg: ${micro_reg} (0=none, 1=micro-rigid, 2=+micro non-rigid)"
    echo "========================================================================"

    # === STAGE INPUT FILES ===
    # Copy files (dereferencing symlinks) into preprocessed/ because VALIS
    # loses track of src_f when working with Nextflow symlinks.
    echo "=== Copying input files to preprocessed/ ==="

    # Collect all OME-TIFF files from staged ref/ and input_*/ directories
    find -L ref input_* -maxdepth 1 -type f \\( -name "*.ome.tif" -o -name "*.ome.tiff" \\) 2>/dev/null > files_to_copy.txt || true

    echo "Files to copy:"
    cat files_to_copy.txt

    # Parallel hard-link copy; cp -Ln skips duplicates (reference may also be in input files)
    cat files_to_copy.txt | xargs -P ${task.cpus} -I {} sh -c '
        dest="preprocessed/\$(basename "{}")"
        cp -Ln "{}" "\$dest" 2>/dev/null && echo "Copied: {}" || echo "Skipped (already exists): {}"
    '

    echo "=== Contents of preprocessed/ ==="
    ls -lh preprocessed/

    # Verify we have actual files (not empty directory)
    file_count=\$(find preprocessed -type f -name '*.ome.tif*' | wc -l)
    echo "Total files copied: \$file_count"

    if [ "\$file_count" -eq 0 ]; then
        echo "ERROR: No .ome.tif files were copied to preprocessed/"
        echo "Available directories and contents:"
        ls -lR
        exit 1
    fi

    # === RUN VALIS REGISTRATION ===
    echo "=== Running registration ==="
    echo "Command: register.py --input-dir preprocessed --out registered_slides ${ref_arg}"

    register.py \\
        --input-dir preprocessed \\
        --out registered_slides \\
        ${ref_arg} \\
        --memory-mode ${memory_mode} \\
        --micro-reg-fraction ${micro_reg_fraction} \\
        --max-image-dim ${max_image_dim} \\
${valis_overrides}
        --micro-reg ${micro_reg} \\
        --jvm-heap-gb ${jvm_heap_gb} \\
        --cpus ${task.cpus} \\
        ${stage_ckpt} \\
        ${args}

    # === VALIDATE OUTPUTS ===
    echo "=== Contents of registered_slides/ ==="
    ls -lh registered_slides/ || echo "Directory is empty or doesn't exist"

    # Check that registration produced output files
    output_count=\$(find registered_slides -type f -name '*_registered.ome.tiff' 2>/dev/null | wc -l)
    echo "Total registered files created: \$output_count"

    if [ "\$output_count" -eq 0 ]; then
        echo "ERROR: No registered output files (*_registered.ome.tiff) were created"
        echo "Registration may have failed. Check the logs above."
        exit 1
    fi

    # Strict check: every input slide must produce a corresponding registered output
    if [ "\$output_count" -ne "\$file_count" ]; then
        echo "ERROR: Output count mismatch — expected \$file_count registered files but got \$output_count"
        echo "VALIS failed to warp some slides. Check the logs above for details."
        exit 1
    fi

    # === GENERATE CHANNELS MANIFEST ===
    # Extract channel names from OME-XML metadata of registered files.
    # convert_image guarantees OME-XML metadata is always present in pipeline images.
    echo "=== Creating channels manifest from OME metadata ==="
    create_channels_manifest.py \\
        --input-dir registered_slides \\
        --output channels_manifest.json

    # === RENAME SUMMARY CSVs WITH PATIENT ID ===
    # Prefix CSV names with patient_id to avoid collisions when aggregated in QC report
    for csv in preprocessed/data/*.csv; do
        if [ -f "\$csv" ]; then
            dir=\$(dirname "\$csv")
            base=\$(basename "\$csv" .csv)
            mv "\$csv" "\${dir}/${patient_id}_\${base}.csv"
        fi
    done

    # === CLEANUP INTERMEDIATES ===
    # Remove working copies to reclaim disk space. Do NOT delete staged inputs
    # (input_*, ref) — Nextflow needs them intact for retries.
    echo "=== Cleaning up intermediate files to save disk space ==="
    find preprocessed -maxdepth 1 -type f -delete
    rm -rf preprocessed/deformation_fields preprocessed/masks preprocessed/overlaps \
           preprocessed/rigid_registration preprocessed/non_rigid_registration preprocessed/processed

    ${ProcessEnvelope.versions(task.process, ['valis'], task.container)}
    """

    stub:
    // Generate output files matching input count with proper naming pattern
    def output_files = all_metas.collect { m ->
        def markers = m.channels.join('_')
        "${patient_id}_${markers}_registered.ome.tiff"
    }
    def touch_commands = output_files.collect { "touch registered_slides/${it}" }.join('\n    ')
    // Build stub manifest JSON mapping filenames to their channel names
    def manifest_map = [all_metas, output_files].transpose().collectEntries { m, fname ->
        [(fname): m.channels]
    }
    def manifest_json = groovy.json.JsonOutput.toJson(manifest_map)
    // Mirror the real path: the stage checkpoint exists only at reg_qc >= 2, so stub runs
    // exercise the same channel wiring the real run does (including its absence at reg_qc<2).
    // Uses the same ParamUtils.regQcLevel default (null -> 2) as the script block above.
    def stub_qc_level = ParamUtils.regQcLevelOf(params.skip_registration_qc, params.reg_qc)
    def stub_micro_reg = ParamUtils.microRegLevelOf(params.reg_micro_reg)
    def stub_ckpt = stub_qc_level < 2 ? '' : """
    mkdir -p reg_stage_checkpoint
    echo '{"version": 1, "stage": "post_non_rigid_pre_micro", "micro_registration": ${stub_micro_reg >= 2}, "slides": {}, "errors": []}' > reg_stage_checkpoint/stage_checkpoint.json
    """
    """
    mkdir -p registered_slides preprocessed/data
    ${touch_commands}
    touch preprocessed/data/${patient_id}_registrar.pickle
    echo '${manifest_json}' > channels_manifest.json
    ${ProcessEnvelope.sizeLogStub(task.process, patient_id, "${patient_id}.REGISTER.size.csv")}
    ${stub_ckpt}

    ${ProcessEnvelope.versionsStub(task.process, ['valis'], task.container)}
    """
}

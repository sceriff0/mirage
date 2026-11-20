/*
 * REGISTER - VALIS whole-slide registration: valis-wsi 1.2.0, its default or its hires example
 *
 * 'default': register() at VALIS's defaults (rigid 512 px, non-rigid 2048 px,
 * align_to_reference False, no micro-registration), every slide warped in the reference
 * frame. The only non-default is the registration channel, passed by index (nuclear,
 * index 0) instead of VALIS's name match on "dapi".
 * 'hires' (params.reg_valis_mode): valis-wsi's own high-resolution example,
 * MicroRigidRegistrar then register_micro() at 0.25 x the smallest slide's longest side.
 * Details in bin/register.py's header.
 */
process REGISTER {
    // Fan-in: one task per patient; all_metas carries the per-slide metas.
    tag "${patient_id}"

    // Upstream valis-wsi:1.2.0 plus the fixes for running as non-root under Apptainer
    // (containers/valis). Needs --writable-tmpfs under Singularity. WarpBackends' valis row
    // must use the same image so the registrar pickle loads in the VALIS that wrote it.
    container 'bolt3x/mirage-valis:1.0.0'

    input:
    // stageAs avoids a name collision: the reference is also in preproc_files
    tuple val(meta), val(patient_id), path(reference, stageAs: 'ref/*'), path(preproc_files, stageAs: 'input_?/*'), val(all_metas)

    output:
    tuple val(patient_id), path("registered_slides/*_registered.ome.tiff"), val(all_metas), path("channels_manifest.json"), emit: registered
    path "versions.yml"                                                                    , emit: versions
    path("*.size.csv")                                                                     , emit: size_log
    path("preprocessed/data/*.csv")                                                        , emit: summary, optional: true
    tuple val(patient_id), path("preprocessed/data/*_registrar.pickle")                    , emit: registrar, optional: true
    // Displacement fields for staged QC (reg_qc >= 2); optional, and failing to write
    // it never fails registration.
    tuple val(patient_id), path("reg_stage_checkpoint")                                    , emit: stage_checkpoint, optional: true

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    // The reference is always staged, so --reference is always passed.
    def ref_filename = reference ? reference.name : ''
    def ref_arg = ref_filename ? "--reference ${ref_filename}" : ''
    // reg_qc >= 2: write the checkpoint WARP_SEG_QC needs to separate stages.
    // ParamUtils.regQcLevel owns the default (null -> 2).
    def reg_qc_level = ParamUtils.regQcLevelOf(params.reg_qc)
    def stage_ckpt = reg_qc_level >= 2 ? '--stage-checkpoint-dir reg_stage_checkpoint' : ''
    // 'default' renders nothing, so register.py runs its own default. The flag shares the
    // stage-checkpoint line to avoid an empty continuation line.
    def hires      = params.reg_valis_mode == 'hires'
    def mode_flags = [hires ? '--valis-mode hires' : '', stage_ckpt].findAll { it }.join(' ')
    // JVM heap: fixed 10 GB (VALIS's own value) unless reg_jvm_heap_gb is set.
    // Explicit null test, not `?:`, which would treat 0 as unset.
    def heap_request = params.reg_jvm_heap_gb != null ? params.reg_jvm_heap_gb : 10
    def jvm_heap_gb = Math.min(heap_request, task.memory.toGiga() - 4)

    """
    ${ProcessEnvelope.sizeLog(task.process, patient_id, ['ref/*', 'input_*/*'], "${patient_id}.REGISTER.size.csv")}

    mkdir -p registered_slides preprocessed

    # === PRINT REGISTRATION SETTINGS ===
    echo "========================================================================"
    echo "VALIS Registration - Attempt ${task.attempt}"
    echo "========================================================================"
    echo "Settings: ${hires ? 'valis-wsi high-resolution example (hires)' : 'valis-wsi defaults'}; registration channel index 0; JVM heap ${jvm_heap_gb} GB"
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
        --jvm-heap-gb ${jvm_heap_gb} \\
        ${mode_flags} \\
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
    def output_files = all_metas.collect { m ->
        def markers = m.channels.join('_')
        "${patient_id}_${markers}_registered.ome.tiff"
    }
    def touch_commands = output_files.collect { "touch registered_slides/${it}" }.join('\n    ')
    // stub manifest: filename -> channel names
    def manifest_map = [all_metas, output_files].transpose().collectEntries { m, fname ->
        [(fname): m.channels]
    }
    def manifest_json = groovy.json.JsonOutput.toJson(manifest_map)
    // As in script:, the checkpoint exists only at reg_qc >= 2; at hires it records
    // micro_registration true.
    def stub_qc_level = ParamUtils.regQcLevelOf(params.reg_qc)
    def stub_micro    = params.reg_valis_mode == 'hires'
    def stub_ckpt = stub_qc_level < 2 ? '' : """
    mkdir -p reg_stage_checkpoint
    echo '{"version": 1, "stage": "post_non_rigid_pre_micro", "micro_registration": ${stub_micro}, "slides": {}, "errors": []}' > reg_stage_checkpoint/stage_checkpoint.json
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

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
    tag "${patient_id}"
    label 'process_high'

    container 'docker://cdgatenbee/valis-wsi:1.0.0'

    input:
    // Use stageAs to avoid filename collision when reference is included in preproc_files
    tuple val(patient_id), path(reference, stageAs: 'ref/*'), path(preproc_files, stageAs: 'input_?/*'), val(all_metas)

    output:
    tuple val(patient_id), path("registered_slides/*_registered.ome.tiff"), val(all_metas), emit: registered
    path "versions.yml"                                                                    , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    // Extract reference filename from the staged path (ref/filename.tif)
    def ref_filename = reference ? reference.name.replaceAll(/^ref\//, '') : ''
    def ref_arg = ref_filename ? "--reference ${ref_filename}" :
                  params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''
    // Reduced defaults for large 25GB+ OME-TIFF files to prevent memory issues
    // Lower resolution for initial rigid registration reduces RAM requirements significantly
    def max_processed_dim = params.reg_max_processed_dim ?: 512
    def max_non_rigid_dim = params.reg_max_non_rigid_dim ?: 2048
    def micro_reg_fraction = params.reg_micro_reg_fraction ?: 0.125
    def num_features = params.reg_num_features ?: 5000
    def max_image_dim = params.reg_max_image_dim ?: 4000
    // Smart retry: skip micro-registration after multiple failures
    // - Exit 1 (general failure): would benefit from skipping on attempt 2
    // - Exit 137 (OOM): give it 2 attempts before skipping micro-reg
    // Strategy: skip micro-registration on attempt 3+ (allows 2 attempts with micro-reg)
    def skip_micro = (task.attempt > 1 || params.skip_micro_registration) ? '--skip-micro-registration' : ''
    // Performance options
    def parallel_warping = params.reg_parallel_warping ? '--parallel-warping' : ''
    def n_workers = params.reg_n_workers ?: 4
    // Advanced registration options
    def use_tiled = params.reg_use_tiled_registration ? '--use-tiled-registration' : ''
    def tile_size = params.reg_tile_size ?: 2048

    """
    mkdir -p registered_slides preprocessed

    echo "========================================================================"
    echo "VALIS Registration - Attempt ${task.attempt}"
    echo "========================================================================"
    echo "Settings:"
    echo "  - max_processed_dim: ${max_processed_dim}"
    echo "  - max_non_rigid_dim: ${max_non_rigid_dim}"
    echo "  - max_image_dim: ${max_image_dim}"
    echo "  - skip_micro_registration: ${skip_micro ? 'YES' : 'NO'}"
    if [ ${task.attempt} -gt 2 ]; then
        echo ""
        echo "  ⚠️  RETRY MODE (attempt 3+): Micro-registration disabled to reduce memory usage"
    fi
    echo "========================================================================"

    # Copy files (dereferencing symlinks) to preprocessed/ to avoid VALIS symlink path resolution issues
    # VALIS loses track of src_f when working with symlinks
    # Using parallel copy to speed up large file transfers
    echo "=== Copying input files to preprocessed/ ==="

    # Collect all ome.tif files from ref and input_* directories
    find -L ref input_* -maxdepth 1 -type f \\( -name "*.ome.tif" -o -name "*.ome.tiff" \\) 2>/dev/null > /tmp/files_to_copy.txt || true

    echo "Files to copy:"
    cat /tmp/files_to_copy.txt

    # Use xargs with multiple parallel processes for faster copying
    # Use cp -n (no-clobber) to skip files that already exist (handles case where reference is also in input files)
    cat /tmp/files_to_copy.txt | xargs -P ${task.cpus} -I {} sh -c '
        dest="preprocessed/\$(basename "{}")"
        cp -Ln "{}" "\$dest" 2>/dev/null && echo "Copied: {}" || echo "Skipped (already exists): {}"
    '

    echo "=== Contents of preprocessed/ ==="
    ls -lh preprocessed/

    # Verify we have actual files (not symlinks)
    file_count=\$(find preprocessed -type f -name '*.ome.tif*' | wc -l)
    echo "Total files copied: \$file_count"

    if [ "\$file_count" -eq 0 ]; then
        echo "ERROR: No .ome.tif files were copied to preprocessed/"
        echo "Available directories and contents:"
        ls -lR
        exit 1
    fi

    echo "=== Running registration ==="
    echo "Command: register.py --input-dir preprocessed --out registered_slides ${ref_arg}"

    register.py \\
        --input-dir preprocessed \\
        --out registered_slides \\
        ${ref_arg} \\
        --max-processed-dim ${max_processed_dim} \\
        --max-non-rigid-dim ${max_non_rigid_dim} \\
        --micro-reg-fraction ${micro_reg_fraction} \\
        --num-features ${num_features} \\
        --max-image-dim ${max_image_dim} \\
        ${skip_micro} \\
        ${parallel_warping} \\
        --n-workers ${n_workers} \\
        ${use_tiled} \\
        --tile-size ${tile_size} \\
        ${args}

    echo "=== Contents of registered_slides/ ==="
    ls -lh registered_slides/ || echo "Directory is empty or doesn't exist"

    # Verify outputs were created
    output_count=\$(find registered_slides -type f -name '*_registered.ome.tiff' 2>/dev/null | wc -l)
    echo "Total registered files created: \$output_count"

    if [ "\$output_count" -eq 0 ]; then
        echo "ERROR: No registered output files (*_registered.ome.tiff) were created"
        echo "Registration may have failed. Check the logs above."
        exit 1
    fi

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        valis: \$(python -c "import valis; print(valis.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    // Generate output files matching input count with proper naming pattern
    // VALIS adapter expects: {patient_id}_{markers}_corrected_registered.ome.tiff
    def output_files = all_metas.collect { meta ->
        def markers = meta.channels.join('_')
        "${patient_id}_${markers}_corrected_registered.ome.tiff"
    }
    def touch_commands = output_files.collect { "touch registered_slides/${it}" }.join('\n    ')
    """
    mkdir -p registered_slides
    ${touch_commands}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        valis: stub
    END_VERSIONS
    """
}

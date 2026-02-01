/*
 * REGISTER_VALIS_PAIRS - VALIS pairwise registration
 *
 * Performs pairwise registration of a single moving image to a reference
 * using VALIS. Used when processing images individually rather than as a batch.
 *
 * Input: Reference image and moving image paths
 * Output: Registered moving image aligned to reference coordinate space
 */
process REGISTER_VALIS_PAIRS {
    tag "${meta.patient_id}"
    label 'process_high'

    container 'docker://cdgatenbee/valis-wsi:1.0.0'

    input:
    tuple val(meta), path(reference), path(moving)

    output:
    tuple val(meta), path("${moving.simpleName}_registered.ome.tiff"), emit: registered
    path "versions.yml"                                               , emit: versions
    path("*.size.csv")                                                , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    // Get reference filename for --reference argument
    def ref_filename = reference.name
    def max_processed_dim = params.reg_max_processed_dim ?: 1800
    def max_non_rigid_dim = params.reg_max_non_rigid_dim ?: 3500
    def micro_reg_fraction = params.reg_micro_reg_fraction ?: 0.5
    def num_features = params.reg_num_features ?: 5000
    def max_image_dim = params.reg_max_image_dim ?: 6000
    def skip_micro = params.skip_micro_registration ? '--skip-micro-registration' : ''

    """
    # Log input sizes for tracing (sum of reference + moving, -L follows symlinks)
    ref_bytes=\$(stat -L --printf="%s" ${reference})
    mov_bytes=\$(stat -L --printf="%s" ${moving})
    total_bytes=\$((ref_bytes + mov_bytes))
    echo "${task.process},${meta.patient_id},${reference.name}+${moving.name},\${total_bytes}" > ${meta.patient_id}_${moving.simpleName}.REGISTER_VALIS_PAIRS.size.csv

    mkdir -p registered_slides preprocessed

    echo "=== VALIS Pairwise Registration ==="
    echo "Reference: ${reference}"
    echo "Moving:    ${moving}"
    echo "Pair:      ${meta.patient_id}"
    echo ""

    # Copy both files to preprocessed directory
    echo "=== Copying input files to preprocessed/ ==="
    cp -L ${reference} preprocessed/
    cp -L ${moving} preprocessed/

    echo "=== Contents of preprocessed/ ==="
    ls -lh preprocessed/

    # Verify we have exactly 2 files
    file_count=\$(find preprocessed -type f \\( -name '*.ome.tif' -o -name '*.ome.tiff' \\) | wc -l)
    echo "Total files: \$file_count"

    if [ "\$file_count" -ne 2 ]; then
        echo "ERROR: Expected 2 files (reference + moving), got \$file_count"
        echo "Available files:"
        ls -lR preprocessed/
        exit 1
    fi

    echo "=== Running VALIS pairwise registration ==="
    echo "Command: register.py --input-dir preprocessed --out registered_slides --reference ${ref_filename}"

    register.py \\
        --input-dir preprocessed \\
        --out registered_slides \\
        --reference ${ref_filename} \\
        --max-processed-dim ${max_processed_dim} \\
        --max-non-rigid-dim ${max_non_rigid_dim} \\
        --micro-reg-fraction ${micro_reg_fraction} \\
        --num-features ${num_features} \\
        --max-image-dim ${max_image_dim} \\
        ${skip_micro} \\
        ${args}

    echo "=== Contents of registered_slides/ ==="
    ls -lh registered_slides/ || echo "Directory is empty or doesn't exist"

    # Verify outputs were created - VALIS creates registered versions of BOTH images
    output_count=\$(find registered_slides -type f -name '*_registered.ome.tiff' 2>/dev/null | wc -l)
    echo "Total registered files created: \$output_count"

    if [ "\$output_count" -eq 0 ]; then
        echo "ERROR: No registered output files (*_registered.ome.tiff) were created"
        echo "Registration may have failed. Check the logs above."
        exit 1
    fi

    # CRITICAL: VALIS registers ALL images including reference
    # We only want the registered MOVING image, not the reference
    # The moving image filename is: ${moving.simpleName}_registered.ome.tiff
    echo "=== Selecting registered moving image ==="
    echo "Looking for: ${moving.simpleName}_registered.ome.tiff"

    if [ -f "registered_slides/${moving.simpleName}_registered.ome.tiff" ]; then
        mv registered_slides/${moving.simpleName}_registered.ome.tiff .
        echo "✓ Found and moved registered moving image"
    else
        echo "ERROR: Could not find registered moving image: ${moving.simpleName}_registered.ome.tiff"
        echo "Available files:"
        ls -lh registered_slides/
        exit 1
    fi

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        valis: \$(python -c "import valis; print(valis.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    touch ${moving.simpleName}_registered.ome.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${moving.simpleName}.REGISTER_VALIS_PAIRS.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        valis: unknown
    END_VERSIONS
    """
}

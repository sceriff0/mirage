nextflow.enable.dsl = 2

process SPLIT_CHANNELS {
    tag "${meta.patient_id}"
    label 'process_medium'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:preprocess' :
        'docker://bolt3x/attend_image_analysis:preprocess' }"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/channels", mode: 'copy'

    input:
    tuple val(meta), path(registered_image), val(is_reference)

    output:
    tuple val(meta), path("*.tiff"), emit: channels
    path "channel_names.txt"        , emit: channel_manifest, optional: true
    path "versions.yml"             , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def ref_flag = is_reference ? "--is-reference" : ""
    // FIX BUG #5: Add defensive null check for meta.channels
    // Pass channel names from metadata if available and valid
    def channel_args = (meta.channels && meta.channels instanceof List && !meta.channels.isEmpty()) ?
        "--channels ${meta.channels.join(' ')}" : ""
    """
    echo "Sample: ${meta.patient_id}"
    echo "Channels: ${(meta.channels && meta.channels instanceof List) ? meta.channels.join(', ') : 'Will read from OME metadata'}"

    split_multichannel.py \\
        ${registered_image} \\
        . \\
        ${ref_flag} \\
        ${channel_args} \\
        ${args}

    # FIX BUG #1: Create manifest of generated channel files for validation
    # List all generated TIFF files (excluding work directory artifacts)
    ls -1 *.tiff | sort > channel_names.txt || echo "No TIFF files generated" > channel_names.txt

    # Validate expected channel count matches actual output
    EXPECTED_CHANNELS=${meta.channels ? meta.channels.size() : 0}
    ACTUAL_CHANNELS=\$(ls -1 *.tiff 2>/dev/null | wc -l | tr -d ' ')

    echo "Expected channels: \$EXPECTED_CHANNELS"
    echo "Actual TIFF files: \$ACTUAL_CHANNELS"

    if [ "\$EXPECTED_CHANNELS" -gt 0 ]; then
        # Adjust expected count if DAPI was skipped (non-reference)
        if [ "${is_reference}" = "false" ]; then
            EXPECTED_ADJUSTED=\$((EXPECTED_CHANNELS - 1))
            echo "Adjusted for DAPI skip (non-reference): \$EXPECTED_ADJUSTED"
        else
            EXPECTED_ADJUSTED=\$EXPECTED_CHANNELS
        fi

        if [ "\$ACTUAL_CHANNELS" -ne "\$EXPECTED_ADJUSTED" ]; then
            echo "❌ ERROR: Channel count mismatch!"
            echo "   Expected \$EXPECTED_ADJUSTED channels, got \$ACTUAL_CHANNELS TIFF files"
            echo "   Generated files:"
            ls -1 *.tiff 2>/dev/null || echo "   (none)"
            exit 1
        fi
    fi

    echo "✅ Channel validation passed"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    # Create stub channel files based on metadata
    touch DAPI.tiff
    ${meta.channels && meta.channels.size() > 1 ?
        meta.channels.drop(1).collect { "touch ${it}.tiff" }.join('\n    ') :
        'touch Marker1.tiff'}

    ls -1 *.tiff | sort > channel_names.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        tifffile: stub
        numpy: stub
    END_VERSIONS
    """
}

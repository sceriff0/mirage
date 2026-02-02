/*
 * PREPROCESS - BaSiC illumination correction
 *
 * Applies BaSiC illumination correction with FOV tiling for flat-field and
 * dark-field estimation. Corrects shading artifacts in multi-channel images.
 *
 * Input: Raw OME-TIFF image with channel metadata
 * Output: Illumination-corrected OME-TIFF
 */
process PREPROCESS {
    tag "${meta.patient_id}"

    container 'docker://bolt3x/attend_image_analysis:preprocess'

    input:
    tuple val(meta), path(ome_tiff)

    output:
    tuple val(meta), path("*_corrected.ome.tif"), emit: preprocessed
    path "versions.yml"                         , emit: versions
    path("*.size.csv")                          , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def skip_dapi_flag = params.preproc_skip_dapi ? '--skip_dapi' : ''
    def autotune_flag = params.preproc_autotune ? '--autotune' : ''
    def no_darkfield_flag = params.preproc_no_darkfield ? '--no_darkfield' : ''
    def overlap = params.preproc_overlap ?: 0
    def channels = meta.channels.join(' ')
    """
    # Log input size for tracing (-L follows symlinks)
    input_bytes=\$(stat -L --printf="%s" ${ome_tiff})
    echo "${task.process},${meta.patient_id},${ome_tiff.name},\${input_bytes}" > ${meta.patient_id}_${ome_tiff.simpleName}.PREPROCESS.size.csv

    preprocess.py \\
        --image ${ome_tiff} \\
        --output_dir . \\
        --channels ${channels} \\
        --fov_size ${params.preproc_tile_size} \\
        --n_workers ${params.preproc_pool_workers} \\
        --n_iter ${params.preproc_n_iter} \\
        --overlap ${overlap} \\
        ${skip_dapi_flag} \\
        ${autotune_flag} \\
        ${no_darkfield_flag} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch ${ome_tiff.simpleName}_corrected.ome.tif
    touch ${ome_tiff.simpleName}_dims.txt
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${ome_tiff.simpleName}.PREPROCESS.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: unknown
    END_VERSIONS
    """
}
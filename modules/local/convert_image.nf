process CONVERT_IMAGE {
    tag "${meta.patient_id}"
    label 'process_medium'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:convert_bioformats' :
        'docker://bolt3x/attend_image_analysis:convert_bioformats' }"

    input:
    tuple val(meta), path(image_file)

    output:
    tuple val(meta), path("*.ome.tif"), path("*_channels.txt"), emit: ome_tiff
    path "versions.yml"                                       , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def pixel_size = params.pixel_size ?: '0.325'
    def channels = meta.channels.join(',')
    """
    convert_image.py \\
        --input_file ${image_file} \\
        --output_dir . \\
        --patient_id ${prefix} \\
        --channels ${channels} \\
        --pixel_size ${pixel_size} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
        aicsimageio: \$(python -c "import aicsimageio; print(aicsimageio.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch ${prefix}.ome.tif
    touch ${prefix}_channels.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        tifffile: unknown
        aicsimageio: unknown
    END_VERSIONS
    """
}

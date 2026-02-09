process CONVERT_IMAGE {
    tag "${meta.patient_id}"
    label 'process_medium'
    
    container 'docker://bolt3x/attend_image_analysis:convert_bioformats_2'

    input:
    tuple val(meta), path(image_file)

    output:
    tuple val(meta), path("*.ome.tif"), path("*_channels.txt"), emit: ome_tiff
    path "versions.yml"                                       , emit: versions
    path("*.size.csv")                                        , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def pixel_size = params.pixel_size ?: '0.325'
    def channels = meta.channels.join(',')
    """
    # Log input size for tracing (-L follows symlinks)
    input_bytes=\$(stat -L --printf="%s" ${image_file})
    echo "${task.process},${meta.patient_id},${image_file.name},\${input_bytes}" > ${meta.patient_id}_${image_file.simpleName}.CONVERT_IMAGE.size.csv

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
    def channels = meta.channels.join(',')
    """
    touch ${prefix}.ome.tif
    echo "${channels}" > ${prefix}_channels.txt
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${image_file.simpleName}.CONVERT_IMAGE.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        tifffile: stub
        aicsimageio: stub
    END_VERSIONS
    """
}

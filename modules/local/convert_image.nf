process CONVERT_IMAGE {
    tag "${meta.patient_id}"
    label 'process_medium'

    conda "${moduleDir}/convert_image/environment.yml"
    container null // Use conda environment

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
    # Set JAVA_HOME for bioio-bioformats (scyjava)
    export JAVA_HOME=\$CONDA_PREFIX

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
        bioio: \$(python -c "import bioio; print(bioio.__version__)" 2>/dev/null || echo "unknown")
        bioio_bioformats: \$(python -c "import bioio_bioformats; print(bioio_bioformats.__version__)" 2>/dev/null || echo "not installed")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def channels = meta.channels.join(',')
    """
    touch ${prefix}.ome.tif
    echo "${channels}" > ${prefix}_channels.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        bioio: stub
        bioio_bioformats: stub
        tifffile: stub
    END_VERSIONS
    """
}

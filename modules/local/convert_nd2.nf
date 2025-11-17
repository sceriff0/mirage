process CONVERT_ND2 {
    tag "$meta.id"
    label 'process_medium'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://ghcr.io/valearna/ateia:latest':
        'ghcr.io/valearna/ateia:latest' }"

    input:
    tuple val(meta), path(nd2_file)

    output:
    tuple val(meta), path("*.ome.tif"), emit: ome_tiff
    path "versions.yml"                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def pixel_size = task.ext.pixel_size ?: '0.325'
    def verify = task.ext.verify ? '--verify' : ''

    """
    convert_nd2.py \\
        --nd2_file ${nd2_file} \\
        --output_dir . \\
        --pixel_size ${pixel_size} \\
        ${verify} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
        nd2: \$(python -c "import nd2; print(nd2.__version__)")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
    END_VERSIONS
    """

    stub:
    def prefix = nd2_file.baseName
    """
    touch ${prefix}.ome.tif

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
        nd2: \$(python -c "import nd2; print(nd2.__version__)")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
    END_VERSIONS
    """
}

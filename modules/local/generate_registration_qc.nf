process GENERATE_REGISTRATION_QC {
    tag "${meta.patient_id}"
    label 'process_medium'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:debug_diffeo' :
        'docker://bolt3x/attend_image_analysis:debug_diffeo' }"

    input:
    tuple val(meta), path(registered), path(reference)

    output:
    tuple val(meta), path("qc/*_QC_RGB.{png,tif}"), emit: qc
    path "versions.yml"                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def scale_factor = params.qc_scale_factor ?: 0.25
    """
    mkdir -p qc

    generate_registration_qc.py \\
        --reference ${reference} \\
        --registered ${registered} \\
        --output qc \\
        --scale-factor ${scale_factor} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    mkdir -p qc
    touch qc/${registered.simpleName}_QC_RGB.png
    touch qc/${registered.simpleName}_QC_RGB_fullres.tif

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
        tifffile: stub
    END_VERSIONS
    """
}

process GENERATE_REGISTRATION_QC {
    tag "${meta.patient_id}"
    label 'process_high'

    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    input:
    tuple val(meta), path(registered), path(reference)

    output:
    tuple val(meta), path("qc/*_QC_RGB.{png,tif}"), emit: qc
    tuple val(meta), path("qc/*_QC_RGB_fullres.tif"), emit: qc_fullres
    path "versions.yml"                           , emit: versions
    path("*.size.csv")                            , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def scale_factor = params.qc_scale_factor ?: 0.25
    """
    # Log input sizes for tracing (sum of registered + reference, -L follows symlinks)
    reg_bytes=\$(stat -L --printf="%s" ${registered})
    ref_bytes=\$(stat -L --printf="%s" ${reference})
    total_bytes=\$((reg_bytes + ref_bytes))
    echo "${task.process},${meta.patient_id},${registered.name}+${reference.name},\${total_bytes}" > ${meta.patient_id}_${registered.simpleName}.GENERATE_REGISTRATION_QC.size.csv

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
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${registered.simpleName}.GENERATE_REGISTRATION_QC.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
        tifffile: stub
    END_VERSIONS
    """
}

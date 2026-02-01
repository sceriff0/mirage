process GENERATE_PREPROCESS_QC {
    tag "${meta.patient_id}"
    label 'process_low'

    container 'docker://bolt3x/attend_image_analysis:preprocess'

    input:
    tuple val(meta), path(preprocessed)

    output:
    tuple val(meta), path("qc/*.png"), emit: qc
    path "versions.yml"              , emit: versions
    path("*.size.csv")               , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def scale_factor = params.preprocess_qc_scale_factor ?: 0.25
    def channels = meta.channels.join(' ')
    """
    # Log input size for tracing
    input_bytes=\$(stat --printf="%s" ${preprocessed})
    echo "${task.process},${meta.patient_id},${preprocessed.name},\${input_bytes}" > ${meta.patient_id}_${preprocessed.simpleName}.GENERATE_PREPROCESS_QC.size.csv

    mkdir -p qc

    generate_preprocess_qc.py \\
        --image ${preprocessed} \\
        --output qc \\
        --channels ${channels} \\
        --scale-factor ${scale_factor} \\
        --prefix ${prefix} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
        scikit-image: \$(python -c "import skimage; print(skimage.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    mkdir -p qc
    touch qc/${prefix}_DAPI.png
    touch qc/${prefix}_channel1.png
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${preprocessed.simpleName}.GENERATE_PREPROCESS_QC.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
        tifffile: stub
        scikit-image: stub
    END_VERSIONS
    """
}

nextflow.enable.dsl = 2

process ESTIMATE_FEATURE_DISTANCES {
    tag "${meta.patient_id}_${meta.channels.join('_')}"
    label 'process_medium'
    container "${params.container.registration}"

    publishDir "${params.outdir}/${meta.patient_id}/${params.registration_method}/feature_distances", mode: 'copy', pattern: "*.{json,png}"

    // Measures feature distances BEFORE and AFTER registration for a single image
    // Detects and matches features in (ref vs moving), then (ref vs registered)
    // Computes pixel-level distances between matched features as TRE metric
    // Complements segmentation-based overlap metrics with sparse feature-based quality assessment

    input:
    tuple val(meta), path(reference), path(moving), path(registered)

    output:
    tuple val(meta), path("*_feature_distances.json"), emit: distance_metrics
    tuple val(meta), path("*_distance_histogram.png"), emit: distance_plots
    path "versions.yml"                               , emit: versions
    path("*.size.csv")                                , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def detector = params.feature_detector ?: 'superpoint'
    def max_dim = params.feature_max_dim ?: 1024
    def n_features = params.feature_n_features ?: 5000
    def prefix = meta.channels.join('_')
    """
    # Log input sizes for tracing (sum of reference + moving + registered, -L follows symlinks)
    ref_bytes=\$(stat -L --printf="%s" ${reference})
    mov_bytes=\$(stat -L --printf="%s" ${moving})
    reg_bytes=\$(stat -L --printf="%s" ${registered})
    total_bytes=\$((ref_bytes + mov_bytes + reg_bytes))
    echo "${task.process},${meta.patient_id},${reference.name}+${moving.name}+${registered.name},\${total_bytes}" > ${meta.patient_id}_${registered.simpleName}.ESTIMATE_FEATURE_DISTANCES.size.csv

    echo "=================================================================="
    echo "Feature Distance Estimation (Before vs After Registration)"
    echo "=================================================================="
    echo "Reference:         ${reference.simpleName}"
    echo "Moving (before):   ${moving.simpleName}"
    echo "Registered (after): ${registered.simpleName}"
    echo "Detector:          ${detector}"
    echo "Max dimension:     ${max_dim}px"
    echo "Number of features: ${n_features}"
    echo "=================================================================="
    echo ""
    echo "Purpose: Measure registration quality via feature-based TRE"
    echo "         Compares feature distances before and after registration"
    echo ""

    estimate_feature_distances.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --registered ${registered} \\
        --output-prefix ${prefix} \\
        --detector ${detector} \\
        --max-dim ${max_dim} \\
        --n-features ${n_features} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
        valis: \$(python -c "import valis; print(valis.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = meta.channels.join('_')
    """
    touch ${prefix}_feature_distances.json
    touch ${prefix}_distance_histogram.png
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${registered.simpleName}.ESTIMATE_FEATURE_DISTANCES.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
        valis: stub
    END_VERSIONS
    """
}

nextflow.enable.dsl = 2

process COMPUTE_FEATURES {
    tag "${moving.simpleName}"
    label 'process_medium'
    container "${params.container.registration}"

    // Resource allocation for feature detection
    memory '32.GB'
    cpus 4
    time '1.h'

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/features", mode: 'copy', pattern: "*.json"

    input:
    tuple path(reference), path(moving)

    output:
    path "${moving.simpleName}_features.json", emit: features
    path "versions.yml"                       , emit: versions

    script:
    def detector = params.feature_detector ?: 'superpoint'
    def max_dim = params.feature_max_dim ?: 2048
    def n_features = params.feature_n_features ?: 5000
    """
    echo "=================================================================="
    echo "Pre-Registration Feature Detection and Matching"
    echo "=================================================================="
    echo "Reference:         ${reference.simpleName}"
    echo "Moving:            ${moving.simpleName}"
    echo "Detector:          ${detector}"
    echo "Max dimension:     ${max_dim}px"
    echo "Number of features: ${n_features}"
    echo "=================================================================="
    echo ""
    echo "Purpose: Establish ground-truth feature correspondences"
    echo "         for VALIS-style TRE measurement"
    echo ""

    compute_features.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output-dir . \\
        --detector ${detector} \\
        --max-dim ${max_dim} \\
        --n-features ${n_features}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    touch ${moving.simpleName}_features.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
    END_VERSIONS
    """
}

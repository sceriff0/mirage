nextflow.enable.dsl = 2

process ESTIMATE_REG_ERROR {
    tag "${registered.simpleName}"
    label 'process_medium'
    container "${params.container.registration}"

    // Resource allocation for error estimation
    memory '32.GB'
    cpus 4
    time '1.h'

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registration_errors", mode: 'copy', pattern: "*.{json,png}"

    // Note: This process implements VALIS-style TRE measurement
    // It tracks the same matched keypoints through registration to compute true error

    input:
    tuple path(reference), path(registered), path(pre_features)

    output:
    path "${registered.simpleName}_registration_error.json", emit: error_metrics
    path "${registered.simpleName}_tre_histogram.png"      , emit: error_plot, optional: true
    path "versions.yml"                                    , emit: versions

    script:
    def detector = params.feature_detector ?: 'superpoint'
    def max_dim = params.feature_max_dim ?: 2048
    def n_features = params.feature_n_features ?: 5000
    """
    echo "=================================================================="
    echo "Registration Error Estimation (VALIS Method)"
    echo "=================================================================="
    echo "Reference:                 ${reference.simpleName}"
    echo "Registered:                ${registered.simpleName}"
    echo "Pre-registration features: ${pre_features}"
    echo "Detector:                  ${detector}"
    echo "=================================================================="
    echo ""
    echo "Method: Tracks same matched keypoints through registration"
    echo "  1. Load pre-registration matched keypoints"
    echo "  2. Estimate global transform (ref → registered)"
    echo "  3. Apply transform to original moving keypoints"
    echo "  4. Compute TRE as distance from expected positions"
    echo ""

    estimate_registration_error.py \\
        --reference ${reference} \\
        --registered ${registered} \\
        --pre-features ${pre_features} \\
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
    touch ${registered.simpleName}_registration_error.json
    touch ${registered.simpleName}_tre_histogram.png

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
    END_VERSIONS
    """
}

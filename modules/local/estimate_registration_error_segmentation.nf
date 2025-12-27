nextflow.enable.dsl = 2

process ESTIMATE_REG_ERROR_SEGMENTATION {
    tag "${registered.simpleName}"
    label 'gpu'
    container "${params.container.segmentation}"

    // Resource allocation for segmentation-based error estimation
    memory '16.GB'
    cpus 2
    time '30.m'

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registration_errors_segmentation", mode: 'copy', pattern: "*.{json,png}"

    // This process uses segmentation-based metrics (IoU/Dice) for robust error estimation
    // Complements feature-based TRE with dense, biologically meaningful measurements

    input:
    tuple path(reference), path(registered)

    output:
    path "${registered.simpleName}_segmentation_error.json", emit: error_metrics
    path "${registered.simpleName}_segmentation_overlay.png", emit: overlay_plot
    path "versions.yml"                                      , emit: versions

    script:
    def max_dim = params.feature_max_dim ?: 2048
    def min_nucleus_size = params.min_nucleus_size ?: 100
    def max_nucleus_size = params.max_nucleus_size ?: 5000
    """
    echo "=================================================================="
    echo "Registration Error Estimation (Segmentation-Based)"
    echo "=================================================================="
    echo "Reference:          ${reference.simpleName}"
    echo "Registered:         ${registered.simpleName}"
    echo "Max dimension:      ${max_dim}px"
    echo "Min nucleus size:   ${min_nucleus_size}px"
    echo "Max nucleus size:   ${max_nucleus_size}px"
    echo "=================================================================="
    echo ""
    echo "Method: Nucleus segmentation + IoU/Dice overlap metrics"
    echo "  1. Segment DAPI nuclei in reference and registered images"
    echo "  2. Compute IoU, Dice, and coverage metrics"
    echo "  3. Analyze spatial error distribution"
    echo "  4. Generate overlay visualization"
    echo ""

    estimate_registration_error_segmentation.py \\
        --reference ${reference} \\
        --registered ${registered} \\
        --output-dir . \\
        --max-dim ${max_dim} \\
        --min-nucleus-size ${min_nucleus_size} \\
        --max-nucleus-size ${max_nucleus_size}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        deepcell: \$(python -c "import deepcell; print(deepcell.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    touch ${registered.simpleName}_segmentation_error.json
    touch ${registered.simpleName}_segmentation_overlay.png

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        deepcell: stub
    END_VERSIONS
    """
}

nextflow.enable.dsl = 2

process ESTIMATE_SEGMENTATION_OVERLAP {
    tag "${meta.patient_id}_${meta.channels.join('_')}"
    label 'gpu'
    container "${params.container.segmentation}"

    memory '16.GB'
    cpus 2
    time '30.m'

    publishDir "${params.outdir}/${meta.patient_id}/${params.registration_method}/segmentation_overlap", mode: 'copy', pattern: "*.{json,png}"

    // Measures nucleus segmentation overlap (IoU/Dice) between reference and registered
    // Dense, biologically meaningful quality metrics complementing feature-based distances

    input:
    tuple val(meta), path(reference), path(registered)

    output:
    tuple val(meta), path("*_segmentation_overlap.json"), emit: overlap_metrics
    tuple val(meta), path("*_segmentation_overlay.png") , emit: overlay_plots
    path "versions.yml"                                  , emit: versions

    script:
    def max_dim = params.feature_max_dim ?: 2048
    def min_nucleus_size = params.min_nucleus_size ?: 100
    def max_nucleus_size = params.max_nucleus_size ?: 5000
    def prefix = meta.channels.join('_')
    """
    estimate_segmentation_overlap.py \\
        --reference ${reference} \\
        --registered ${registered} \\
        --output-prefix ${prefix} \\
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
    def prefix = meta.channels.join('_')
    """
    touch ${prefix}_segmentation_overlap.json
    touch ${prefix}_segmentation_overlay.png

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        deepcell: stub
    END_VERSIONS
    """
}

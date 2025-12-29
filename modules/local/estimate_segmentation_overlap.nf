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
    def prefix = meta.channels.join('_')
    def pmin = params.seg_pmin ?: 1.0
    def pmax = params.seg_pmax ?: 99.8
    def n_tiles_y = params.seg_n_tiles_y ?: 24
    def n_tiles_x = params.seg_n_tiles_x ?: 24
    def max_nucleus_distance = params.max_nucleus_distance ?: 50.0
    """
    estimate_segmentation_overlap.py \\
        --reference ${reference} \\
        --registered ${registered} \\
        --output-prefix ${prefix} \\
        --model-dir ${params.segmentation_model_dir} \\
        --model-name ${params.segmentation_model} \\
        --max-dim ${max_dim} \\
        --n-tiles ${n_tiles_y} ${n_tiles_x} \\
        --pmin ${pmin} \\
        --pmax ${pmax} \\
        --max-nucleus-distance ${max_nucleus_distance}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        stardist: \$(python -c "import stardist; print(stardist.__version__)" 2>/dev/null || echo "unknown")
        tensorflow: \$(python -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null || echo "unknown")
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
        stardist: stub
        tensorflow: stub
    END_VERSIONS
    """
}

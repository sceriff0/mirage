process QUANTIFY {
    tag "quantify"
    label "${params.quant_gpu ? 'gpu' : 'process_high'}"
    container "${params.container.quantification}"

    input:
    path merged_ome
    path seg_mask

    output:
    path "quant/merged_quant.csv", emit: csv
    path "quant/quant.log"       , emit: log, optional: true
    path "versions.yml"          , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    mkdir -p quant
    python3 scripts/quantify.py \\
        --mode ${params.quant_gpu ? 'gpu' : 'cpu'} \\
        --mask_file ${seg_mask} \\
        --indir \$(dirname ${merged_ome}) \\
        --outdir quant \\
        --output_file quant/merged_quant.csv \\
        --log_file quant/quant.log \\
        ${args} || true

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //g')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p quant
    touch quant/merged_quant.csv
    touch quant/quant.log
    touch versions.yml
    """
}

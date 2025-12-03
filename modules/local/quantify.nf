nextflow.enable.dsl = 2

process QUANTIFY {
    tag "quantify"
    label "${params.quant_gpu ? 'gpu' : 'process_high'}"
    container "${params.container.quantification}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/quantification", mode: 'copy'

    input:
    path merged_ome
    path seg_mask

    output:
    path "quant/merged_quant.csv", emit: csv
    path "quant/quant.log"       , emit: log, optional: true

    script:
    """
    mkdir -p quant
    quantify_gpu.py \\
        --mode ${params.quant_gpu ? 'gpu' : 'cpu'} \\
        --mask_file ${seg_mask} \\
        --indir \$(dirname ${merged_ome}) \\
        --outdir quant \\
        --output_file quant/merged_quant.csv \\
        --min_area ${params.quant_min_area} \\
        --log_file quant/quant.log \\
        || true
    """
}

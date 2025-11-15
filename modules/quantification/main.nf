nextflow.enable.dsl=2

/*
 * Quantification module
 * - Inputs: single merged WSI and a single segmentation mask
 * - Outputs: quant/merged_quant.csv
 */

process QUANT_PROC {
    tag "quantify"
    container params.container.quantification
    label "${params.quant_gpu ? 'gpu' : 'standard'}"

    input:
    path merged_ome
    path seg_mask

    output:
    path "quant/merged_quant.csv"

    script:
    '''
    mkdir -p quant
    python3 scripts/quantify.py \
        --mode ${params.quant_gpu ? 'gpu' : 'cpu'} \
        --mask_file ${seg_mask} \
        --indir $(dirname ${merged_ome}) \
        --outdir quant \
        --output_file quant/merged_quant.csv \
        --log_file quant/quant.log || true
    '''
}

workflow QUANTIFY_CELLS {
    take:
    merged_ch
    seg_ch

    main:
    quant_ch = QUANT_PROC(merged_ch, seg_ch)

    emit:
    quant_ch
}

nextflow.enable.dsl=2

/* Phenotyping module
 * - Inputs: quant CSV and segmentation mask
 * - Outputs: pheno/merged_pheno.csv
 */

process PHENO_PROC {
    tag "phenotype"

    input:
    path quant_csv
    path seg_mask

    output:
    path "pheno/merged_pheno.csv"

    script:
    '''
    mkdir -p pheno
    python3 scripts/phenotype.py \
        --cell_data ${quant_csv} \
        --segmentation_mask ${seg_mask} \
        -o pheno || true
    '''
}

workflow PHENOTYPE_CELLS {
    take:
    quant_ch
    seg_ch

    main:
    pheno_ch = PHENO_PROC(quant_ch, seg_ch)

    emit:
    pheno_ch
}

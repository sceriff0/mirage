process PHENOTYPE {
    tag "phenotype"
    label 'process_medium'
    container "${params.container.phenotyping}"

    input:
    path quant_csv
    path seg_mask

    output:
    path "pheno/merged_pheno.csv", emit: csv

    script:
    def markers_arg = params.pheno_markers ? "--markers ${params.pheno_markers.join(' ')}" : ''
    def cutoffs_arg = params.pheno_cutoffs ? "--cutoffs ${params.pheno_cutoffs.join(' ')}" : ''
    """
    mkdir -p pheno
    python3 scripts/phenotype.py \\
        --cell_data ${quant_csv} \\
        --segmentation_mask ${seg_mask} \\
        -o pheno \\
        ${markers_arg} \\
        ${cutoffs_arg} \\
        --quality_percentile ${params.pheno_quality_percentile} \\
        --noise_percentile ${params.pheno_noise_percentile} \\
        || true
    """
}

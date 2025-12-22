process PHENOTYPE {
    tag "phenotype"
    label 'process_medium'
    // container "${params.container.phenotyping}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/phenotype", mode: 'copy'

    input:
    path quant_csv
    path seg_mask

    output:
    path "pheno/phenotypes_data.csv", emit: csv
    path "pheno/phenotypes_mask.tiff", emit: mask
    path "pheno/phenotype_mapping.json", emit: mapping

    script:
    def markers_arg = params.pheno_markers ? "--markers ${params.pheno_markers.join(' ')}" : ''
    def cutoffs_arg = params.pheno_cutoffs ? "--cutoffs ${params.pheno_cutoffs.join(' ')}" : ''
    """
    mkdir -p pheno
    phenotype.py \\
        --cell_data ${quant_csv} \\
        --segmentation_mask ${seg_mask} \\
        -o pheno \\
        ${markers_arg} \\
        ${cutoffs_arg} \\
        --quality_percentile ${params.pheno_quality_percentile} \\
        --noise_percentile ${params.pheno_noise_percentile} \\
    """
}

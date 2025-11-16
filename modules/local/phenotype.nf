process PHENOTYPE {
    tag "phenotype"
    label 'process_medium'
    container "${params.container.phenotyping}"

    input:
    path quant_csv
    path seg_mask

    output:
    path "pheno/merged_pheno.csv", emit: csv
    path "versions.yml"          , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    mkdir -p pheno
    python3 scripts/phenotype.py \\
        --cell_data ${quant_csv} \\
        --segmentation_mask ${seg_mask} \\
        -o pheno \\
        ${args} || true

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //g')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p pheno
    touch pheno/merged_pheno.csv
    touch versions.yml
    """
}

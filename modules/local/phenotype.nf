process PHENOTYPE {
    tag "${meta.patient_id}"
    label 'process_medium'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:quantification_gpu' :
        'docker://bolt3x/attend_image_analysis:quantification_gpu' }"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/phenotype", mode: 'copy'

    input:
    tuple val(meta), path(quant_csv), path(seg_mask)

    output:
    tuple val(meta), path("pheno/phenotypes_data.csv")   , emit: csv
    tuple val(meta), path("pheno/phenotypes_mask.tiff")  , emit: mask
    tuple val(meta), path("pheno/phenotype_mapping.json"), emit: mapping
    path "versions.yml"                                   , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def markers_arg = params.pheno_markers ? "--markers ${params.pheno_markers.join(' ')}" : ''
    def cutoffs_arg = params.pheno_cutoffs ? "--cutoffs ${params.pheno_cutoffs.join(' ')}" : ''
    """
    echo "Sample: ${meta.patient_id}"

    mkdir -p pheno
    phenotype.py \\
        --cell_data ${quant_csv} \\
        --segmentation_mask ${seg_mask} \\
        -o pheno \\
        ${markers_arg} \\
        ${cutoffs_arg} \\
        --quality_percentile ${params.pheno_quality_percentile} \\
        --noise_percentile ${params.pheno_noise_percentile} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        pandas: \$(python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "unknown")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    mkdir -p pheno
    touch pheno/phenotypes_data.csv
    touch pheno/phenotypes_mask.tiff
    touch pheno/phenotype_mapping.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        pandas: stub
        numpy: stub
    END_VERSIONS
    """
}

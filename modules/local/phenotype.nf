/*
 * PHENOTYPE - Cell phenotype classification
 *
 * Classifies cells into phenotypes based on marker expression using z-score
 * thresholding. Generates QuPath-compatible GeoJSON and classification outputs.
 *
 * Input: Merged quantification CSV with per-cell marker intensities
 * Output: Phenotype data CSV, GeoJSON annotations, and classification mappings
 */
process PHENOTYPE {
    tag "${meta.patient_id}"
    label 'process_medium'

    container 'docker://bolt3x/attend_image_analysis:quantification_gpu'

    publishDir "${params.outdir}/${meta.patient_id}/phenotype", mode: 'copy'

    input:
    tuple val(meta), path(quant_csv)

    output:
    tuple val(meta), path("pheno/phenotypes_data.csv")            , emit: csv
    tuple val(meta), path("pheno/phenotypes.geojson")             , emit: geojson
    tuple val(meta), path("pheno/phenotypes.classifications.json"), emit: classifications
    tuple val(meta), path("pheno/phenotypes_mapping.json")        , emit: mapping
    path "versions.yml"                                            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def markers_arg = params.pheno_markers ? "--markers ${params.pheno_markers.join(' ')}" : ''
    def cutoffs_arg = params.pheno_cutoffs ? "--cutoffs ${params.pheno_cutoffs.join(' ')}" : ''
    def pixel_size = params.pixel_size ?: 0.325
    """
    echo "Sample: ${meta.patient_id}"

    mkdir -p pheno
    phenotype.py \\
        --cell_data ${quant_csv} \\
        -o pheno \\
        --pixel_size ${pixel_size} \\
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
    touch pheno/phenotypes.geojson
    touch pheno/phenotypes.classifications.json
    touch pheno/phenotypes_mapping.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        pandas: stub
        numpy: stub
    END_VERSIONS
    """
}

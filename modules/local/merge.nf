nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MERGE MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Merges individually registered slides into a single multi-channel OME-TIFF.
    Keeps all channels from all slides, but for DAPI only retains it from the reference image.
    Appends segmentation and phenotype masks as additional channels.
    Phenotype mask includes distinct colors for each phenotype for visualization.
----------------------------------------------------------------------------------------
*/

process MERGE {
    tag "${meta.patient_id}"
    label 'process_high'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:merge' :
        'docker://bolt3x/attend_image_analysis:merge' }"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/merged", mode: 'copy'

    input:
    tuple val(meta), path(registered_slides), path(seg_mask), path(pheno_mask), path(pheno_mapping)

    output:
    tuple val(meta), path("merged_all.ome.tiff"), emit: merged
    path "versions.yml"                          , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def ref_markers = params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''
    """
    echo "Sample: ${meta.patient_id}"

    merge_registered.py \\
        --input-dir . \\
        --output merged_all.ome.tiff \\
        --segmentation-mask ${seg_mask} \\
        --phenotype-mask ${pheno_mask} \\
        --phenotype-mapping ${pheno_mapping} \\
        ${ref_markers} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch merged_all.ome.tiff

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
        tifffile: stub
    END_VERSIONS
    """
}

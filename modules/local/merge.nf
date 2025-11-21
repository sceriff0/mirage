nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MERGE MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Merges individually registered slides into a single multi-channel OME-TIFF,
    skipping duplicate channels.
----------------------------------------------------------------------------------------
*/

process MERGE {
    tag "merge_registered"
    label 'process_medium'
    container "${params.container.registration}"

    publishDir "${params.outdir}/merged", mode: 'copy'

    input:
    path registered_slides

    output:
    path "merged_all.ome.tiff", emit: merged

    script:
    """
    merge_registered.py \\
        --input-dir . \\
        --output merged_all.ome.tiff
    """
}

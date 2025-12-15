nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MERGE MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Merges individually registered slides into a single multi-channel OME-TIFF.
    Keeps all channels from all slides, but for DAPI only retains it from the reference image.
----------------------------------------------------------------------------------------
*/

process MERGE {
    tag "merge_registered"
    label 'process_high'
    container "${params.container.merge}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/merged", mode: 'copy'

    input:
    path registered_slides
    path seg_mask

    output:
    path "merged_all.ome.tiff", emit: merged

    script:
    def ref_markers = params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''
    """
    merge_registered.py \\
        --input-dir . \\
        --output merged_all.ome.tiff \\
        --segmentation-mask ${seg_mask} \\
        ${ref_markers}
    """
}

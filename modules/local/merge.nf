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
    tag "merge_registered"
    label 'process_high'
    container "${params.container.merge}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/merged", mode: 'copy'

    input:
    path registered_slides
    path seg_mask
    path pheno_mask
    path pheno_mapping

    output:
    path "merged_all.ome.tiff", emit: merged

    script:
    def ref_markers = params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''
    """
    merge_registered.py \\
        --input-dir . \\
        --output merged_all.ome.tiff \\
        --segmentation-mask ${seg_mask} \\
        --phenotype-mask ${pheno_mask} \\
        --phenotype-mapping ${pheno_mapping} \\
        ${ref_markers}
    """
}

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CONVERSION MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Creates a pyramidal OME-TIFF combining registered images with segmentation
    and phenotype masks for efficient visualization.
----------------------------------------------------------------------------------------
*/

process CONVERSION {
    tag "create_pyramid"
    label 'process_high'

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/pyramid", mode: 'copy'

    input:
    path merged_image

    output:
    path "pyramid.ome.tiff", emit: pyramid

    script:
    """
    bfconvert \
        -noflat \
        -bigtiff \
        -tilex ${params.tilex} \
        -tiley ${params.tiley} \
        -pyramid-resolutions ${params.pyramid_resolutions} \
        -pyramid-scale ${params.pyramid_scale} \
        "${merged_image}" \
        "pyramid.ome.tiff"
    """
}

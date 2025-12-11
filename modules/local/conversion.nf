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
    container "${params.container.conversion}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/pyramid", mode: 'copy'

    input:
    path merged_image
    path seg_mask
    path phenotype_mask

    output:
    path "pyramid.ome.tiff", emit: pyramid

    script:
    def pyramid_resolutions = params.pyramid_resolutions ?: 3
    def pyramid_scale = params.pyramid_scale ?: 2
    def tilex = params.tilex ?: 256
    def tiley = params.tiley ?: 256
    """
    # Use vips command-line tool for maximum speed and minimum memory
    # First, combine merged image with masks
    vips bandjoin "${merged_image} ${seg_mask} ${phenotype_mask}" combined_temp.tif

    # Create pyramid using vips tiffsave with pyramid options
    vips tiffsave combined_temp.tif pyramid.ome.tiff \\
        --compression lzw \\
        --tile \\
        --tile-width ${tilex} \\
        --tile-height ${tiley} \\
        --pyramid \\
        --bigtiff

    # Clean up
    rm -f combined_temp.tif
    """
}

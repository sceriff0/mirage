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
    # Normalize TIFFs to fix metadata issues before joining
    vips copy ${merged_image} merged_normalized.tif
    vips copy ${seg_mask} seg_normalized.tif
    vips copy ${phenotype_mask} pheno_normalized.tif

    # Combine normalized images
    vips bandjoin "merged_normalized.tif seg_normalized.tif pheno_normalized.tif" combined_temp.tif

    # Create pyramid using vips tiffsave with pyramid options
    vips tiffsave combined_temp.tif pyramid.ome.tiff \\
        --compression lzw \\
        --tile \\
        --tile-width ${tilex} \\
        --tile-height ${tiley} \\
        --pyramid \\
        --bigtiff

    # Clean up
    rm -f combined_temp.tif merged_normalized.tif seg_normalized.tif pheno_normalized.tif
    """
}

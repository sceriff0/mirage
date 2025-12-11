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
    #!/usr/bin/env python3
    import pyvips
    import sys

    try:
        # Load images with pyvips (handles metadata issues better than CLI)
        print("Loading merged image...")
        merged = pyvips.Image.new_from_file("${merged_image}", access='sequential')

        print("Loading segmentation mask...")
        seg = pyvips.Image.new_from_file("${seg_mask}", access='sequential')

        print("Loading phenotype mask...")
        pheno = pyvips.Image.new_from_file("${phenotype_mask}", access='sequential')

        # Combine images using bandjoin
        print("Combining images...")
        combined = merged.bandjoin([seg, pheno])

        # Save as pyramidal TIFF
        print("Creating pyramidal TIFF...")
        combined.tiffsave(
            "pyramid.ome.tiff",
            compression="lzw",
            tile=True,
            tile_width=${tilex},
            tile_height=${tiley},
            pyramid=True,
            bigtiff=True
        )

        print("✓ Pyramid created successfully")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    """
}

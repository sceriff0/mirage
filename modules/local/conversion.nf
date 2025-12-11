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
    import tifffile
    import numpy as np
    import sys

    try:
        # Load TIFFs with tifffile first to normalize them
        print("Loading merged image with tifffile...")
        merged_array = tifffile.imread("${merged_image}")
        print(f"  Shape: {merged_array.shape}, dtype: {merged_array.dtype}")

        print("Loading segmentation mask with tifffile...")
        seg_array = tifffile.imread("${seg_mask}")
        print(f"  Shape: {seg_array.shape}, dtype: {seg_array.dtype}")

        print("Loading phenotype mask with tifffile...")
        pheno_array = tifffile.imread("${phenotype_mask}")
        print(f"  Shape: {pheno_array.shape}, dtype: {pheno_array.dtype}")

        # Save normalized versions
        print("Saving normalized TIFFs...")
        tifffile.imwrite("merged_norm.tif", merged_array, bigtiff=True, compression='lzw')
        tifffile.imwrite("seg_norm.tif", seg_array, bigtiff=True, compression='lzw')
        tifffile.imwrite("pheno_norm.tif", pheno_array, bigtiff=True, compression='lzw')

        # Now load with pyvips and create pyramid
        print("Loading normalized images with pyvips...")
        merged = pyvips.Image.new_from_file("merged_norm.tif", access='sequential')
        seg = pyvips.Image.new_from_file("seg_norm.tif", access='sequential')
        pheno = pyvips.Image.new_from_file("pheno_norm.tif", access='sequential')

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
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    """
}

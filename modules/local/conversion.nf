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
        # Load TIFFs with tifffile first to normalize them and extract metadata
        print("Loading merged image with tifffile...")
        with tifffile.TiffFile("${merged_image}") as tif:
            merged_array = tif.asarray()
            # Try to extract channel names from OME-XML metadata
            channel_names = []
            try:
                if tif.ome_metadata:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(tif.ome_metadata)
                    # Parse OME-XML namespace
                    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    for channel in root.findall('.//ome:Channel', ns):
                        name = channel.get('Name', channel.get('ID', ''))
                        if name:
                            channel_names.append(name)
                    print(f"  Found {len(channel_names)} channel names: {channel_names}")
            except Exception as e:
                print(f"  Could not extract channel names: {e}")

        print(f"  Shape: {merged_array.shape}, dtype: {merged_array.dtype}")

        print("Loading segmentation mask with tifffile...")
        seg_array = tifffile.imread("${seg_mask}")
        print(f"  Shape: {seg_array.shape}, dtype: {seg_array.dtype}")

        print("Loading phenotype mask with tifffile...")
        pheno_array = tifffile.imread("${phenotype_mask}")
        print(f"  Shape: {pheno_array.shape}, dtype: {pheno_array.dtype}")

        # Convert int64 to int32 for pyvips compatibility (it doesn't support int64)
        # Also convert to uint8 or uint16 for categorical display
        if pheno_array.dtype == np.int64:
            print("  Converting phenotype mask from int64...")
            pheno_min = pheno_array.min()
            pheno_max = pheno_array.max()
            print(f"    Phenotype range: {pheno_min} to {pheno_max}")

            # Shift negative values to start from 0 for categorical LUT
            if pheno_min < 0:
                print(f"    Shifting values by {-pheno_min} to make non-negative")
                pheno_array = pheno_array - pheno_min
                pheno_max = pheno_max - pheno_min

            # Convert to smallest compatible unsigned type
            if pheno_max <= 255:
                pheno_array = pheno_array.astype(np.uint8)
                print(f"    Converted to uint8 for categorical display")
            elif pheno_max <= 65535:
                pheno_array = pheno_array.astype(np.uint16)
                print(f"    Converted to uint16 for categorical display")
            else:
                pheno_array = pheno_array.astype(np.int32)
                print(f"    Converted to int32 (too many categories for uint16)")

        # Convert uint32 to uint16 if values fit, otherwise keep uint32
        if seg_array.dtype == np.uint32:
            max_val = seg_array.max()
            if max_val <= 65535:
                print(f"  Converting segmentation mask from uint32 to uint16 (max value: {max_val})...")
                seg_array = seg_array.astype(np.uint16)

        # Save normalized versions with compatible dtypes
        print("Saving normalized TIFFs...")
        tifffile.imwrite("merged_norm.tif", merged_array, bigtiff=True, compression='lzw')
        tifffile.imwrite("seg_norm.tif", seg_array, bigtiff=True, compression='lzw')

        # Create categorical colormap for phenotype mask
        n_phenotypes = int(pheno_array.max() + 1)
        print(f"  Creating categorical LUT for {n_phenotypes} phenotypes...")

        # Distinctive colors for categorical display
        base_colors = [
            [0, 0, 0],         # 0: Background (black)
            [255, 0, 0],       # 1: Red
            [0, 255, 0],       # 2: Green
            [0, 0, 255],       # 3: Blue
            [255, 255, 0],     # 4: Yellow
            [255, 0, 255],     # 5: Magenta
            [0, 255, 255],     # 6: Cyan
            [255, 128, 0],     # 7: Orange
            [128, 0, 255],     # 8: Purple
            [0, 255, 128],     # 9: Spring green
            [255, 0, 128],     # 10: Rose
            [128, 255, 0],     # 11: Lime
            [0, 128, 255],     # 12: Sky blue
            [255, 128, 128],   # 13: Light red
            [128, 255, 128],   # 14: Light green
            [128, 128, 255],   # 15: Light blue
            [192, 192, 0],     # 16: Olive
            [192, 0, 192],     # 17: Dark magenta
            [0, 192, 192],     # 18: Teal
            [255, 192, 128],   # 19: Peach
        ]

        # Extend with random colors if needed
        import random
        colors = base_colors.copy()
        for i in range(len(colors), n_phenotypes):
            random.seed(i)
            colors.append([random.randint(50, 255) for _ in range(3)])

        pheno_lut = np.array(colors[:n_phenotypes], dtype=np.uint8)

        # Save phenotype with colormap for categorical display
        tifffile.imwrite("pheno_norm.tif", pheno_array,
                        bigtiff=True,
                        compression='lzw',
                        photometric='palette',
                        colormap=pheno_lut)

        # Now load with pyvips and create pyramid
        print("Loading normalized images with pyvips...")
        merged = pyvips.Image.new_from_file("merged_norm.tif", access='sequential')
        seg = pyvips.Image.new_from_file("seg_norm.tif", access='sequential')
        pheno = pyvips.Image.new_from_file("pheno_norm.tif", access='sequential')

        # Combine images using bandjoin
        print("Combining images...")
        combined = merged.bandjoin([seg, pheno])

        # Add channel names to metadata if available
        if channel_names:
            # Add segmentation and phenotype names
            all_channel_names = channel_names + ['Segmentation', 'Phenotype']
            print(f"  Setting channel names: {all_channel_names}")
            # Store as ImageDescription metadata
            channel_desc = "Channels: " + ", ".join(all_channel_names)
            combined = combined.copy()
            combined.set_type(pyvips.GValue.gstr_type, 'image-description', channel_desc)

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

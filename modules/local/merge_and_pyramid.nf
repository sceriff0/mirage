nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MERGE AND PYRAMID MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Merges single-channel TIFFs into a pyramidal OME-TIFF with proper metadata
    for QuPath visualization. This replaces the previous two-step approach of
    merge_channels.py followed by bfconvert.

    Features:
    - Generates pyramidal OME-TIFF directly (no bfconvert needed)
    - Preserves channel names, colors, and pixel sizes in OME-XML
    - Supports segmentation and phenotype mask overlays
    - Memory-efficient processing for large images
    - Full QuPath compatibility
----------------------------------------------------------------------------------------
*/

process MERGE_AND_PYRAMID {
    tag "${meta.patient_id}"
    label 'process_high'

    container 'docker://bolt3x/attend_image_analysis:merge'

    // Publish both the pyramid and colormap files
    publishDir "${params.outdir}/${meta.patient_id}/pyramid", mode: 'copy'

    input:
    tuple val(meta), path(split_channels, stageAs: 'channels/*'), path(seg_mask)

    output:
    tuple val(meta), path("pyramid.ome.tiff"), emit: pyramid
    path "versions.yml", emit: versions
    path("*.size.csv") , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"

    // Get pixel size from params (use pixel_size which is already defined in config)
    def pixel_size_x = params.pixel_size ?: 0.325
    def pixel_size_y = params.pixel_size ?: 0.325
    def pyramid_resolutions = params.pyramid_resolutions ?: 5
    def pyramid_scale = params.pyramid_scale ?: 2
    def tile_size = params.tilex ?: 256
    def compression = params.compression ?: 'lzw'

    """
    # Log input size for tracing (sum of channels/ dir + seg_mask)
    channels_bytes=\$(du -sb channels/ | cut -f1)
    mask_bytes=\$(stat --printf="%s" ${seg_mask})
    total_bytes=\$((channels_bytes + mask_bytes))
    echo "${task.process},${meta.patient_id},channels/+${seg_mask.name},\${total_bytes}" > ${meta.patient_id}.size.csv

    echo "Sample: ${meta.patient_id}"
    echo "Input directory: channels/"
    ls -lh channels/

    merge_channels_pyramid.py \\
        --input-dir channels \\
        --output pyramid.ome.tiff \\
        --physical-size-x ${pixel_size_x} \\
        --physical-size-y ${pixel_size_y} \\
        --pyramid-resolutions ${pyramid_resolutions} \\
        --pyramid-scale ${pyramid_scale} \\
        --tile-size ${tile_size} \\
        --compression ${compression} \\
        --segmentation-mask ${seg_mask} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        tifffile: \$(python3 -c "import tifffile; print(tifffile.__version__)")
        numpy: \$(python3 -c "import numpy; print(numpy.__version__)")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch pyramid.ome.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        tifffile: stub
        numpy: stub
    END_VERSIONS
    """
}

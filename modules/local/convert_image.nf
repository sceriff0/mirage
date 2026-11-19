/*
 * CONVERT_IMAGE - convert an input slide to standardized OME-TIFF via Bio-Formats.
 */
process CONVERT_IMAGE {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-convert:1.0.0"

    input:
    tuple val(meta), path(image_file)

    output:
    tuple val(meta), path("*.ome.tif"), path("*_channels.txt"), emit: ome_tiff
    path "versions.yml"                                       , emit: versions
    path("*.size.csv")                                        , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: (meta.id ?: meta.patient_id)
    def pixel_size = params.pixel_size
    def channels = meta.channels.join(',')
    // Via MarkerUtils: from the CLI the param is a String, and "X".join(' ') silently
    // returns '' (Java's static String.join), leaving a bare --nuclear-markers.
    def nuclear_markers = MarkerUtils.markerList(params.nuclear_markers).join(' ')
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${image_file}"], "${meta.patient_id}_${image_file.simpleName}.CONVERT_IMAGE.size.csv")}

    convert_image.py \\
        --input_file ${image_file} \\
        --output_dir . \\
        --patient_id ${prefix} \\
        --channels ${channels} \\
        --pixel_size ${pixel_size} \\
        --nuclear-markers ${nuclear_markers} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['tifffile', 'bioio', 'h5py'], task.container)}
    """

    stub:
    def prefix = task.ext.prefix ?: (meta.id ?: meta.patient_id)
    def channels = meta.channels.join(',')
    """
    touch ${prefix}.ome.tif
    echo "${channels}" > ${prefix}_channels.txt
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}_${image_file.simpleName}.CONVERT_IMAGE.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['tifffile', 'bioio', 'h5py'], task.container)}
    """
}

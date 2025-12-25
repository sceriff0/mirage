process CONVERT_IMAGE {
    tag "${meta.patient_id}"
    label 'process_medium'

    publishDir "${params.outdir}/${meta.patient_id}/${params.registration_method}/converted", mode: 'copy'

    input:
    tuple val(meta), path(image_file)

    output:
    tuple val(meta), path("*.ome.tif"), path("*_channels.txt"), emit: ome_tiff

    script:
    def pixel_size = params.pixel_size ?: '0.325'
    def channels = meta.channels.join(',')
    """
    convert_image.py \\
        --input_file ${image_file} \\
        --output_dir . \\
        --patient_id ${meta.patient_id} \\
        --channels ${channels} \\
        --pixel_size ${pixel_size}
    """
}

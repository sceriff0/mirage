process CONVERT_ND2 {
    tag "${nd2_file.simpleName}"
    label 'process_medium'

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/converted", mode: 'copy'

    input:
    path nd2_file

    output:
    path "*.ome.tif", emit: ome_tiff

    script:
    def pixel_size = params.pixel_size ?: '0.325'
    """
    convert_nd2.py \\
        --nd2_file ${nd2_file} \\
        --output_dir . \\
        --pixel_size ${pixel_size}
    """
}

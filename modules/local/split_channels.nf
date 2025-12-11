nextflow.enable.dsl = 2

process SPLIT_CHANNELS {
    tag "${registered_image.simpleName}"
    label 'process_low'
    container "${params.container.quantification}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/channels", mode: 'copy'

    input:
    tuple path(registered_image), val(is_reference)

    output:
    path "*.tiff", emit: channels

    script:
    def ref_flag = is_reference ? "--is-reference" : ""
    """
    split_multichannel.py \\
        ${registered_image} \\
        . \\
        ${ref_flag}
    """
}

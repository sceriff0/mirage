nextflow.enable.dsl = 2

process REGISTER {
    tag "register_all"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registered", mode: 'copy'

    input:
    path preproc_dir 

    output:
    path "merged/merged_all.ome.tif", emit: merged
    path "merged_qc"          , emit: qc, optional: true

    script:
    def ref_markers = params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''

    """
    mkdir -p merged merged_qc

    register.py \\
        --input-dir ${preproc_dir} \\
        --out merged/merged_all.ome.tif \\
        --qc-dir merged_qc \\
        ${ref_markers}
    """
}

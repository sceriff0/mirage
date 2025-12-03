nextflow.enable.dsl = 2

process GET_PREPROCESS_DIR {
    tag "get_preprocess_dir"

    input:
    val files 

    output:
    path "${params.outdir}/${params.id}/${params.registration_method}/preprocessed", emit: preprocess_dir

    script:
    """
    # This directory exists because all PREPROCESS tasks published to it
    echo "Preprocessed directory is ready: ${params.outdir}/preprocessed"
    """
}
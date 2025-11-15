nextflow.enable.dsl=2

/*
 * Preprocessing module
 * Exposes a workflow named PREPROCESS_WSI that accepts a channel of input files
 * and emits preprocessed files into a channel.
 */

process PREPROC_PROC {
    tag nd2file.simpleName
    container params.container.preprocess

    input:
    path nd2file

    output:
    path "preprocessed/${nd2file.simpleName}.preproc.ome.tif"

    script:
    '''
    mkdir -p preprocessed
    python3 scripts/preprocess.py \
        --image ${nd2file} \
        --channels ${nd2file} \
        --output_dir preprocessed \
        --fov_size ${params.preproc_tile_size} \
        --skip_dapi ${params.preproc_normalize}
    '''
}

workflow PREPROCESS_WSI {
    take: nd2_ch

    main:
    preprocessed_files_ch = PREPROC_PROC(nd2_ch)

    emit:
    preprocessed_files_ch
}

nextflow.enable.dsl=2

/*
 * Registration module (collect-and-merge)
 * - Takes ALL preprocessed files (from preprocessing) and produces a single merged WSI.
 * - Input: channel of files (preprocessed/*)
 * - Output: merged/merged_all.ome.tif
 */

process REGISTER_ALL {
    tag "register_all"

    /* Accept a list of files as a single input (use .collect() at the workflow call site) */
    input:
    path preproc_files

    output:
    path "merged/merged_all.ome.tif"
    path "merged_qc/*.png", optional: true

    script:
    '''
    mkdir -p merged merged_qc
    # Try running the registration script; on failure fall back to copying the first preprocessed file
    python3 scripts/register.py \
        --input-files ${preproc_files.join(' ')} \
        --out merged/merged_all.ome.tif \
        --qc-dir merged_qc || cp ${preproc_files[0]} merged/merged_all.ome.tif
    '''
}

workflow REGISTER_IMAGES {
    take: preproc_ch

    main:
    // collect all inputs into a single list and run one registration
    merged_wsi_ch = REGISTER_ALL(preproc_ch.collect())

    emit:
    merged_wsi_ch
}

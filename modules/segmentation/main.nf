nextflow.enable.dsl=2

/* Segmentation module
 * Inputs: merged WSI file (single image)
 * Outputs: segmentation/<basename>_segmentation.tif
 * This process uses label 'gpu' and will run on GPU nodes when available. If params.seg_gpu is false,
 * it will fall back to a CPU implementation.
 */

process SEGMENT_PROC {
    tag merged_file

        label "${params.seg_gpu ? 'gpu' : 'standard'}"
        container params.container.segmentation

    input:
    path merged_file

    output:
    path "segmentation/${merged_file.simpleName}_segmentation.tif"

    script:
    '''
    mkdir -p segmentation
    python3 scripts/segment.py \
        --input ${merged_file} \
        --out segmentation/${merged_file.simpleName}_segmentation.tif \
        --use-gpu ${params.seg_gpu} \
        --model ${params.seg_model}
    '''
}

workflow SEGMENT_DAPI {
    take: merged_ch

    main:
    seg_mask_ch = SEGMENT_PROC(merged_ch)

    emit:
    seg_mask_ch
}

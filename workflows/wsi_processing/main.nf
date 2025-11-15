nextflow.enable.dsl=2
include { PREPROCESS_WSI } from '../../modules/preprocessing/main.nf'
include { REGISTER_IMAGES } from '../../modules/registration/main.nf'
include { SEGMENT_DAPI } from '../../modules/segmentation/main.nf'
include { QUANTIFY_CELLS } from '../../modules/quantification/main.nf'
include { PHENOTYPE_CELLS } from '../../modules/phenotyping/main.nf'

workflow WSI_PROCESSING {
    /*
     * Simplified linear per-file pipeline.
     * Input: glob pattern (e.g. params.input -> './data/*.nd2')
     * For each input file we run: PREPROCESS_WSI -> REGISTER_IMAGES -> SEGMENT_DAPI -> QUANTIFY_CELLS -> PHENOTYPE_CELLS
     * Outputs are written into stage-specific subdirectories under the working directory (preprocessed/, merged/, segmentation/, quant/ , pheno/).
     */

    take: input_pattern

    main:
    // Create a channel of input files from the provided glob pattern
    nd2_files_ch = channel.fromPath(input_pattern)

    // 1) Preprocess each input file -> produces preprocessed/<basename>.preproc.ome.tif
    preprocessed_files_ch = PREPROCESS_WSI(nd2_files_ch)

    // 2) Register (or pass-through) preprocessed files -> produces merged/<basename>_merged.ome.tif
    merged_wsi_ch = REGISTER_IMAGES(preprocessed_files_ch)

    // 3) Segment the merged WSI -> produces segmentation/<basename>_segmentation.tif
    seg_mask_ch = SEGMENT_DAPI(merged_wsi_ch)

    // 4) Quantify cells using merged image + segmentation mask -> produces quant/<basename>_quant.csv
    quant_ch = QUANTIFY_CELLS(merged_wsi_ch, seg_mask_ch)

    // 5) Phenotype cells using quant CSV + segmentation mask -> produces pheno/<basename>_pheno.csv
    pheno_ch = PHENOTYPE_CELLS(quant_ch, seg_mask_ch)

    emit:
    merged_wsi_ch
    seg_mask_ch
    quant_ch
    pheno_ch
}

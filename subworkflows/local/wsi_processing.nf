/*
 * WSI Processing Subworkflow
 * Processes whole slide images through preprocessing, registration,
 * segmentation, quantification, and phenotyping stages
 */

include { PREPROCESS } from '../../modules/local/preprocess'
include { REGISTER   } from '../../modules/local/register'
include { SEGMENT    } from '../../modules/local/segment'
include { QUANTIFY   } from '../../modules/local/quantify'
include { PHENOTYPE  } from '../../modules/local/phenotype'

workflow WSI_PROCESSING {
    take:
    nd2_files_ch    // channel: [ path(nd2_file) ]

    main:
    ch_versions = Channel.empty()

    //
    // MODULE: Preprocess each input file
    //
    PREPROCESS ( nd2_files_ch )
    ch_versions = ch_versions.mix(PREPROCESS.out.versions.first())

    //
    // MODULE: Register/merge all preprocessed files
    //
    REGISTER ( PREPROCESS.out.preprocessed.collect() )
    ch_versions = ch_versions.mix(REGISTER.out.versions)

    //
    // MODULE: Segment the merged WSI
    //
    SEGMENT ( REGISTER.out.merged )
    ch_versions = ch_versions.mix(SEGMENT.out.versions)

    //
    // MODULE: Quantify cells
    //
    QUANTIFY (
        REGISTER.out.merged,
        SEGMENT.out.mask
    )
    ch_versions = ch_versions.mix(QUANTIFY.out.versions)

    //
    // MODULE: Phenotype cells
    //
    PHENOTYPE (
        QUANTIFY.out.csv,
        SEGMENT.out.mask
    )
    ch_versions = ch_versions.mix(PHENOTYPE.out.versions)

    emit:
    merged_wsi  = REGISTER.out.merged       // channel: [ path(merged.ome.tif) ]
    qc_images   = REGISTER.out.qc           // channel: [ path(*.png) ]
    seg_mask    = SEGMENT.out.mask          // channel: [ path(segmentation.tif) ]
    quant_csv   = QUANTIFY.out.csv          // channel: [ path(quant.csv) ]
    quant_log   = QUANTIFY.out.log          // channel: [ path(quant.log) ]
    pheno_csv   = PHENOTYPE.out.csv         // channel: [ path(pheno.csv) ]
    versions    = ch_versions               // channel: [ path(versions.yml) ]
}

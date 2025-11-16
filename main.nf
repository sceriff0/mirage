#!/usr/bin/env nextflow
/*
========================================================================================
    ATEIA WSI Processing Pipeline
========================================================================================
    Whole Slide Image Processing Pipeline
    Github: https://github.com/ateia/wsi-processing
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { PREPROCESS } from './modules/local/preprocess'
include { REGISTER   } from './modules/local/register'
include { SEGMENT    } from './modules/local/segment'
include { QUANTIFY   } from './modules/local/quantify'
include { PHENOTYPE  } from './modules/local/phenotype'

/*
========================================================================================
    PRINT PARAMETER SUMMARY
========================================================================================
*/

log.info """\
    ATEIA WSI PROCESSING PIPELINE
    =============================
    input        : ${params.input}
    outdir       : ${params.outdir}
    GPU enabled  : ${params.use_gpu}
    """
    .stripIndent()

/*
========================================================================================
    RUN MAIN WORKFLOW
========================================================================================
*/

workflow {
    // Validate input parameters
    if (!params.input) {
        error "Please provide an input glob pattern with --input"
    }

    // Create input channel from glob pattern
    ch_input = Channel.fromPath(params.input, checkIfExists: true)

    // MODULE: Preprocess each input file
    PREPROCESS ( ch_input )

    // MODULE: Register/merge all preprocessed files
    REGISTER ( PREPROCESS.out.preprocessed.collect() )

    // MODULE: Segment the merged WSI
    SEGMENT ( REGISTER.out.merged )

    // MODULE: Quantify cells
    QUANTIFY (
        REGISTER.out.merged,
        SEGMENT.out.mask
    )

    // MODULE: Phenotype cells
    PHENOTYPE (
        QUANTIFY.out.csv,
        SEGMENT.out.mask
    )
}

/*
========================================================================================
    THE END
========================================================================================
*/

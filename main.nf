nextflow.enable.dsl=2
include { WSI_PROCESSING } from './workflows/wsi_processing/main.nf'

workflow {
    WSI_PROCESSING(params.input)
}

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { MERGE      } from '../../modules/local/merge'
include { CONVERSION } from '../../modules/local/conversion'
include { SAVE  } from '../../modules/local/save'

/*
========================================================================================
    SUBWORKFLOW: RESULTS
========================================================================================
    Description:
        Merges registered images, creates pyramidal OME-TIFF with masks,
        and saves final results to archive location.

    Input:
        ch_registered: Channel of registered multichannel images
        cell_mask: Segmentation mask
        phenotype_mask: Phenotype mask
        merged_csv: Merged quantification CSV
        phenotype_csv: Phenotyped cell data CSV
        savedir: Final archive directory path

    Output:
        merged: Merged multichannel OME-TIFF
        pyramid: Pyramidal OME-TIFF with masks
========================================================================================
*/

workflow RESULTS {
    take:
    ch_registered    // Channel of registered images
    cell_mask        // Cell segmentation mask
    phenotype_mask   // Phenotype mask
    merged_csv       // Merged quantification CSV
    phenotype_csv    // Phenotype CSV
    savedir          // Archive directory

    main:
    // Merge all registered images into single multichannel OME-TIFF
    MERGE ( ch_registered.collect() )

    // Create pyramidal OME-TIFF with masks (CONVERSION process)
    CONVERSION (
        MERGE.out.merged,
        cell_mask,
        phenotype_mask
    )

    // Collect only the outputs to save
    ch_to_save = channel.empty()
        .mix(
            ch_registered,          // QC: registered images
            merged_csv,             // Quantification results
            phenotype_csv,          // Phenotype results
            CONVERSION.out.pyramid  // Pyramidal visualization
        )
        .collect()
        .map { _files ->
            // All files are published under the same parent directory
            return file("${params.outdir}")
        }

    // Save to final archive location
    SAVE (
        ch_to_save,
        savedir
    )
}

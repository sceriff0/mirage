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
        Merges registered images with segmentation and phenotype masks,
        creates pyramidal OME-TIFF, and saves final results to archive location.

    Input:
        ch_registered: Channel of registered multichannel images
        ch_qc: Channel of QC RGB images
        cell_mask: Cell segmentation mask
        phenotype_mask: Phenotype mask with different colors for each phenotype
        phenotype_mapping: Phenotype number to name mapping (JSON)
        merged_csv: Merged quantification CSV
        phenotype_csv: Phenotyped cell data CSV
        savedir: Final archive directory path

    Output:
        merged: Merged multichannel OME-TIFF with segmentation and phenotype masks
        pyramid: Pyramidal OME-TIFF with masks
========================================================================================
*/

workflow RESULTS {
    take:
    ch_registered      // Channel of registered images
    ch_qc              // Channel of QC RGB images
    cell_mask          // Cell segmentation mask
    phenotype_mask     // Phenotype mask
    phenotype_mapping  // Phenotype number to name mapping JSON
    merged_csv         // Merged quantification CSV
    phenotype_csv      // Phenotype CSV
    savedir            // Archive directory

    main:
    // Merge all registered images into single multichannel OME-TIFF with masks
    MERGE (
        ch_registered.collect(),
        cell_mask,
        phenotype_mask,
        phenotype_mapping
    )

    // Create pyramidal OME-TIFF (CONVERSION process)
    CONVERSION ( MERGE.out.merged )

    // Collect only the outputs to save
    ch_to_save = channel.empty()
        .mix(
            ch_qc,                  // QC: RGB visualization images
            merged_csv,             // Quantification results
            phenotype_csv,          // Phenotype results
            CONVERSION.out.pyramid  // Pyramidal visualization
        )
        .collect()

    // Save to final archive location
    SAVE (
        ch_to_save,
        savedir
    )
}

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { SEGMENT          } from '../../modules/local/segment'
include { SPLIT_CHANNELS   } from '../../modules/local/split_channels'
include { QUANTIFY         } from '../../modules/local/quantify'
include { MERGE_QUANT_CSVS } from '../../modules/local/quantify'
include { PHENOTYPE        } from '../../modules/local/phenotype'

/*
========================================================================================
    SUBWORKFLOW:POSTPROCESSING
========================================================================================
    Description:
        Segments reference image, splits multichannel images to single channels,
        quantifies marker intensities per cell, merges results, and assigns phenotypes.

    Input:
        ch_registered: Channel of registered multichannel images
        reference_markers: List of markers to identify reference image

    Output:
        phenotype_csv: Phenotyped cell data CSV
        phenotype_mask: Phenotype mask image
        merged_csv: Merged quantification CSV
        cell_mask: Cell segmentation mask
========================================================================================
*/

workflow POSTPROCESSING {
    take:
    ch_registered       // Channel of registered images
    reference_markers   // List of markers for reference identification

    main:
    // Extract reference image from registered channel
    ch_reference_for_seg = ch_registered
        .filter { file ->
            def filename = file.name.toUpperCase()
            reference_markers.every { marker ->
                filename.contains(marker.toUpperCase())
            }
        }
        .first()  // Take only the reference image

    // Segment the reference image only (no merge needed)
    SEGMENT ( ch_reference_for_seg )

    // Split multichannel images into single-channel TIFFs
    // (DAPI only from reference)
    ch_for_split = ch_registered
        .map { registered_file ->
            def filename = registered_file.name.toUpperCase()
            def is_reference = reference_markers.every { marker ->
                filename.contains(marker.toUpperCase())
            }
            return tuple(registered_file, is_reference)
        }

    SPLIT_CHANNELS ( ch_for_split )

    // Quantify using individual channel TIFFs
    // Combine all single-channel TIFFs with the segmentation mask
    ch_for_quant = SPLIT_CHANNELS.out.channels
        .flatten()
        .combine(SEGMENT.out.cell_mask)

    QUANTIFY ( ch_for_quant )

    // Merge individual quantification CSVs
    // Collect all individual CSVs and merge them
    ch_individual_csvs = QUANTIFY.out.individual_csv
        .collect()

    MERGE_QUANT_CSVS ( ch_individual_csvs )

    // Phenotype cells based on predefined rules
    PHENOTYPE (
        MERGE_QUANT_CSVS.out.merged_csv,
        SEGMENT.out.cell_mask
    )

    emit:
    phenotype_csv = PHENOTYPE.out.csv
    phenotype_mask = PHENOTYPE.out.mask
    phenotype_mapping = PHENOTYPE.out.mapping
    merged_csv = MERGE_QUANT_CSVS.out.merged_csv
    cell_mask = SEGMENT.out.cell_mask
    individual_csvs = QUANTIFY.out.individual_csv
}

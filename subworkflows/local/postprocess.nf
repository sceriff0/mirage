nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { SEGMENT               } from '../../modules/local/segment'
include { SPLIT_CHANNELS        } from '../../modules/local/split_channels'
include { QUANTIFY              } from '../../modules/local/quantify'
include { MERGE_QUANT_CSVS      } from '../../modules/local/quantify'
include { PHENOTYPE             } from '../../modules/local/phenotype'
include { WRITE_CHECKPOINT_CSV  } from '../../modules/local/write_checkpoint_csv'

/*
========================================================================================
    SUBWORKFLOW:POSTPROCESSING
========================================================================================
    Description:
        Segments reference image, splits multichannel images to single channels,
        quantifies marker intensities per cell, merges results, and assigns phenotypes.

    Input:
        ch_registered: Channel of [meta, file] tuples for registered images
        reference_markers: List of markers to identify reference image (not used, uses is_reference metadata)

    Output:
        phenotype_csv: Phenotyped cell data CSV
        phenotype_mask: Phenotype mask image
        merged_csv: Merged quantification CSV
        cell_mask: Cell segmentation mask
========================================================================================
*/

workflow POSTPROCESSING {
    take:
    ch_registered       // Channel of [meta, file] tuples

    main:
    // Extract reference image using is_reference metadata
    ch_reference_for_seg = ch_registered
        .filter { meta, file -> meta.is_reference }
        .map { meta, file -> file }
        .first()  // Take only the reference image

    // Segment the reference image only (no merge needed)
    SEGMENT ( ch_reference_for_seg )

    // Split multichannel images into single-channel TIFFs
    // (DAPI only from reference)
    ch_for_split = ch_registered
        .map { meta, file ->
            return tuple(file, meta.is_reference)
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

    // Generate checkpoint CSV for restart from postprocessing step
    // Extract reference metadata for checkpoint (postprocessing works on all images together)
    ch_reference_meta = ch_registered
        .filter { meta, file -> meta.is_reference }
        .map { meta, file -> meta }
        .first()

    ch_checkpoint_data = PHENOTYPE.out.csv
        .combine(PHENOTYPE.out.mask)
        .combine(PHENOTYPE.out.mapping)
        .combine(MERGE_QUANT_CSVS.out.merged_csv)
        .combine(SEGMENT.out.cell_mask)
        .combine(ch_reference_meta)
        .map { pheno_csv, pheno_mask, pheno_map, merged_csv, cell_mask, ref_meta ->
            [
                ref_meta.patient_id,
                true,  // is_reference
                pheno_csv.toString(),
                pheno_mask.toString(),
                pheno_map.toString(),
                merged_csv.toString(),
                cell_mask.toString()
            ]
        }
        .collect()

    WRITE_CHECKPOINT_CSV(
        'postprocessed',
        'patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask',
        ch_checkpoint_data
    )

    emit:
    phenotype_csv = PHENOTYPE.out.csv
    phenotype_mask = PHENOTYPE.out.mask
    phenotype_mapping = PHENOTYPE.out.mapping
    merged_csv = MERGE_QUANT_CSVS.out.merged_csv
    cell_mask = SEGMENT.out.cell_mask
    individual_csvs = QUANTIFY.out.individual_csv
    checkpoint_csv = WRITE_CHECKPOINT_CSV.out.csv
}

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
    // SEGMENT now accepts [meta, file] tuple
    ch_reference_for_seg = ch_registered
        .filter { meta, file -> meta.is_reference }
        .first()  // Keep [meta, file] tuple

    // Segment the reference image only (no merge needed)
    SEGMENT ( ch_reference_for_seg )

    // Split multichannel images into single-channel TIFFs
    // SPLIT_CHANNELS now accepts [meta, file] tuple
    SPLIT_CHANNELS ( ch_registered )

    // Quantify using individual channel TIFFs
    // SPLIT_CHANNELS now outputs [meta, channels] - combine with cell_mask
    // SEGMENT now outputs [meta, mask]
    ch_for_quant = SPLIT_CHANNELS.out.channels
        .transpose()  // Flatten channel list while keeping meta
        .combine(SEGMENT.out.cell_mask.map { meta, mask -> mask })

    QUANTIFY ( ch_for_quant )

    // Merge individual quantification CSVs
    // QUANTIFY now outputs [meta, csv] - extract meta and collect CSVs
    ch_reference_meta = ch_reference_for_seg.map { meta, file -> meta }
    ch_individual_csvs = QUANTIFY.out.individual_csv
        .map { meta, csv -> csv }
        .collect()
        .combine(ch_reference_meta)
        .map { csvs, meta -> [meta, csvs] }

    MERGE_QUANT_CSVS ( ch_individual_csvs )

    // Phenotype cells based on predefined rules
    // PHENOTYPE now accepts [meta, csv, mask]
    ch_for_phenotype = MERGE_QUANT_CSVS.out.merged_csv
        .combine(SEGMENT.out.cell_mask.map { meta, mask -> mask })

    PHENOTYPE ( ch_for_phenotype )

    // Generate checkpoint CSV for restart from postprocessing step
    // All modules now output [meta, file] tuples - extract metadata and files
    ch_checkpoint_data = PHENOTYPE.out.csv
        .combine(PHENOTYPE.out.mask.map { meta, mask -> mask })
        .combine(PHENOTYPE.out.mapping.map { meta, mapping -> mapping })
        .combine(MERGE_QUANT_CSVS.out.merged_csv.map { meta, csv -> csv })
        .combine(SEGMENT.out.cell_mask.map { meta, mask -> mask })
        .map { meta, pheno_csv, pheno_mask, pheno_map, merged_csv, cell_mask ->
            [
                meta.patient_id,
                meta.is_reference,
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

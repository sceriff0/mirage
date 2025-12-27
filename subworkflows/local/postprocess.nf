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
    // ========================================================================
    // SEGMENTATION - Process reference images only
    // ========================================================================
    ch_references = ch_registered
        .filter { meta, file -> meta.is_reference }

    SEGMENT(ch_references)

    // ========================================================================
    // CHANNEL SPLITTING - Split all multichannel images
    // ========================================================================
    SPLIT_CHANNELS(
        ch_registered.map { meta, file -> [meta, file, meta.is_reference] }
    )

    // ========================================================================
    // QUANTIFICATION - Join channels with their patient's mask
    // ========================================================================
    ch_for_quant = SPLIT_CHANNELS.out.channels
        .transpose()                                                    // [meta, single_tiff]
        .map { meta, tiff -> [meta.patient_id, meta, tiff] }
        .join(
            SEGMENT.out.cell_mask.map { meta, mask -> [meta.patient_id, mask] },
            by: 0
        )
        .map { _patient_id, meta, tiff, mask -> [meta, tiff, mask] }

    QUANTIFY(ch_for_quant)

    // ========================================================================
    // MERGE - Group CSVs by patient_id
    // ========================================================================
    ch_grouped_csvs = QUANTIFY.out.individual_csv
        .map { meta, csv -> [meta.patient_id, meta, csv] }
        .groupTuple(by: 0)
        .map { patient_id, metas, csvs ->
            def meta = metas[0].clone()
            meta.id = patient_id
            [meta, csvs]
        }

    MERGE_QUANT_CSVS(ch_grouped_csvs)

    // ========================================================================
    // PHENOTYPING - Join merged CSV with patient's mask
    // ========================================================================
    ch_for_phenotype = MERGE_QUANT_CSVS.out.merged_csv
        .map { meta, csv -> [meta.patient_id, meta, csv] }
        .join(
            SEGMENT.out.cell_mask.map { meta, mask -> [meta.patient_id, mask] },
            by: 0
        )
        .map { _patient_id, meta, csv, mask -> [meta, csv, mask] }

    PHENOTYPE(ch_for_phenotype)

    // ========================================================================
    // CHECKPOINT - Collect all outputs by patient
    // ========================================================================
    ch_checkpoint_data = PHENOTYPE.out.csv
        .map { meta, csv -> [meta.patient_id, csv] }
        .join(PHENOTYPE.out.mask.map { meta, mask -> [meta.patient_id, mask] })
        .join(PHENOTYPE.out.mapping.map { meta, map -> [meta.patient_id, map] })
        .join(MERGE_QUANT_CSVS.out.merged_csv.map { meta, csv -> [meta.patient_id, csv] })
        .join(SEGMENT.out.cell_mask.map { meta, mask -> [meta.patient_id, mask] })
        .map { patient_id, pheno_csv, pheno_mask, pheno_map, merged_csv, cell_mask ->
            [
                patient_id,
                true,  // All checkpoint rows are for reference (one per patient)
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

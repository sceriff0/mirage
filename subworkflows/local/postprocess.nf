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
include { MERGE                 } from '../../modules/local/merge'
include { CONVERSION            } from '../../modules/local/conversion'
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
    ch_split_output = SPLIT_CHANNELS.out.channels
        .view { meta, tiffs -> "SPLIT_CHANNELS output: meta.patient_id=${meta.patient_id}, tiffs=${tiffs*.name}" }

    ch_flatmapped = ch_split_output
        .flatMap { meta, tiffs ->
            // Ensure tiffs is always a list (handle both single file and multiple files)
            def tiff_list = tiffs instanceof List ? tiffs : [tiffs]

            // Create unique meta map for each channel file
            tiff_list.collect { tiff ->
                def channel_meta = meta.clone()
                channel_meta.id = "${meta.patient_id}_${tiff.baseName}"
                channel_meta.channel_name = tiff.baseName
                [channel_meta, tiff]
            }
        }
        .view { meta, tiff -> "After flatMap: meta.id=${meta.id}, channel=${meta.channel_name}, tiff=${tiff.name}" }

    ch_for_combine = ch_flatmapped
        .map { meta, tiff -> [meta.patient_id, meta, tiff] }
        .view { patient_id, _meta, _tiff -> "Before combine: key=${patient_id}, channel=${_meta.channel_name}" }

    ch_mask = SEGMENT.out.cell_mask
        .map { meta, mask -> [meta.patient_id, mask] }
        .view { patient_id, _mask -> "Mask available: key=${patient_id}, mask=${_mask.name}" }

    ch_for_quant = ch_for_combine
        .combine(ch_mask, by: 0)
        .view { patient_id, _meta, _tiff, _mask -> "After combine: patient=${patient_id}, channel=${_meta.channel_name}" }
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
    // MERGE - Combine split channel TIFFs with masks (per patient)
    // ========================================================================
    // Group split channel TIFFs by patient for merging
    // SPLIT_CHANNELS already handles DAPI filtering correctly
    ch_split_grouped = SPLIT_CHANNELS.out.channels
        .map { meta, tiffs -> [meta.patient_id, meta, tiffs] }
        .groupTuple(by: 0)
        .map { patient_id, _metas, tiff_lists ->
            // Flatten all TIFF files from all slides into one list
            def all_tiffs = tiff_lists.flatten()
            // Create patient-level metadata
            def patient_meta = [
                patient_id: patient_id,
                is_reference: false  // Not relevant at patient level
            ]
            [patient_meta, all_tiffs]
        }

    // Join split channels with all masks for MERGE
    ch_for_merge = ch_split_grouped
        .map { meta, tiffs -> [meta.patient_id, meta, tiffs] }
        .join(
            SEGMENT.out.cell_mask.map { meta, mask -> [meta.patient_id, mask] },
            by: 0
        )
        .join(
            PHENOTYPE.out.mask.map { meta, mask -> [meta.patient_id, mask] },
            by: 0
        )
        .join(
            PHENOTYPE.out.mapping.map { meta, mapping -> [meta.patient_id, mapping] },
            by: 0
        )
        .map { _patient_id, meta, split_tiffs, cell_mask, pheno_mask, pheno_mapping ->
            [meta, split_tiffs, cell_mask, pheno_mask, pheno_mapping]
        }

    MERGE(ch_for_merge)

    // ========================================================================
    // CONVERSION - Create pyramidal OME-TIFF (per patient)
    // ========================================================================
    CONVERSION(MERGE.out.merged)

    // ========================================================================
    // CHECKPOINT - Collect all outputs by patient
    // ========================================================================
    ch_checkpoint_data = PHENOTYPE.out.csv
        .map { meta, csv ->
            def published_path = "${params.outdir}/${meta.patient_id}/phenotype/${csv.name}"
            [meta.patient_id, published_path]
        }
        .join(PHENOTYPE.out.mask.map { meta, mask ->
            def published_path = "${params.outdir}/${meta.patient_id}/phenotype/${mask.name}"
            [meta.patient_id, published_path]
        })
        .join(PHENOTYPE.out.mapping.map { meta, map ->
            def published_path = "${params.outdir}/${meta.patient_id}/phenotype/${map.name}"
            [meta.patient_id, published_path]
        })
        .join(MERGE_QUANT_CSVS.out.merged_csv.map { meta, csv ->
            def published_path = "${params.outdir}/${meta.patient_id}/quantification/${csv.name}"
            [meta.patient_id, published_path]
        })
        .join(SEGMENT.out.cell_mask.map { meta, mask ->
            def published_path = "${params.outdir}/${meta.patient_id}/segmentation/${mask.name}"
            [meta.patient_id, published_path]
        })
        .join(MERGE.out.merged.map { meta, merged ->
            def published_path = "${params.outdir}/${meta.patient_id}/merged/${merged.name}"
            [meta.patient_id, published_path]
        })
        .join(CONVERSION.out.pyramid.map { meta, pyramid ->
            def published_path = "${params.outdir}/${meta.patient_id}/pyramid/${pyramid.name}"
            [meta.patient_id, published_path]
        })
        .map { patient_id, pheno_csv, pheno_mask, pheno_map, merged_csv, cell_mask, merged_image, pyramid ->
            [
                patient_id,
                pheno_csv,
                pheno_mask,
                pheno_map,
                merged_csv,
                cell_mask,
                merged_image,
                pyramid
            ]
        }
        .toList()
        .view { data -> "Checkpoint data: $data" }

    WRITE_CHECKPOINT_CSV(
        'postprocessed',
        'patient_id,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask,merged_image,pyramid',
        ch_checkpoint_data
    )

    emit:
    checkpoint_csv = WRITE_CHECKPOINT_CSV.out.csv
}

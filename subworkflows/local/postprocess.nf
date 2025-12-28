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
    // VALIDATION - Ensure metadata has required fields
    // ========================================================================
    // FIX BUG #3: Add defensive checks for meta.channels before use
    // Validate metadata early to prevent cryptic downstream errors

    ch_validated = ch_registered
        .map { meta, file ->
            // Validate required metadata fields exist and are valid
            if (!meta.containsKey('channels')) {
                throw new Exception("""
                ❌ Missing 'channels' field in metadata for patient ${meta.patient_id}
                Available fields: ${meta.keySet().join(', ')}
                💡 This usually means the metadata was not properly constructed from CSV
                """.stripIndent())
            }

            if (!(meta.channels instanceof List) || meta.channels.isEmpty()) {
                throw new Exception("""
                ❌ Invalid 'channels' field for patient ${meta.patient_id}
                Type: ${meta.channels?.class}
                Value: ${meta.channels}
                💡 channels must be a non-empty List
                """.stripIndent())
            }

            if (meta.channels[0]?.toUpperCase() != 'DAPI') {
                throw new Exception("""
                ❌ DAPI must be first channel for patient ${meta.patient_id}
                Current order: ${meta.channels}
                💡 Check your registration checkpoint CSV has correct channel order
                """.stripIndent())
            }

            [meta, file]
        }

    // ========================================================================
    // SEGMENTATION - Process reference images only
    // ========================================================================
    ch_references = ch_validated
        .filter { meta, file -> meta.is_reference }

    SEGMENT(ch_references)

    // ========================================================================
    // CHANNEL SPLITTING - Split all multichannel images
    // ========================================================================
    SPLIT_CHANNELS(
        ch_validated.map { meta, file -> [meta, file, meta.is_reference] }
    )

    // ========================================================================
    // QUANTIFICATION - Join channels with their patient's mask
    // ========================================================================
    // FIX ISSUE #11: Validate SPLIT_CHANNELS output cardinality
    ch_for_quant = SPLIT_CHANNELS.out.channels
        .map { meta, tiffs ->
            // Validate we got the expected number of channels
            def expected_channels = meta.channels.size()
            def actual_channels = tiffs.size()
            if (actual_channels != expected_channels) {
                throw new Exception("""
                ❌ CRITICAL: Channel count mismatch for ${meta.patient_id}!

                Expected ${expected_channels} channels (from metadata): ${meta.channels}
                Got ${actual_channels} channel files from SPLIT_CHANNELS

                💡 This indicates SPLIT_CHANNELS may have failed or produced corrupted output.
                   Check SPLIT_CHANNELS logs for patient ${meta.patient_id}
                """.stripIndent())
            }
            // Also validate no empty files
            tiffs.each { tiff ->
                if (tiff.size() == 0) {
                    throw new Exception("❌ CRITICAL: Empty channel file detected: ${tiff} for patient ${meta.patient_id}")
                }
            }
            return [meta, tiffs]
        }
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
    // FIX CRITICAL BUG: Include registered images in checkpoint for pyramid regeneration
    // Group registered images by patient to get all images (reference + moving)
    ch_registered_grouped = ch_registered
        .map { meta, file ->
            // Construct the published path for registered images
            def published_path = "${params.outdir}/${meta.patient_id}/registered/${file.name}"
            [meta.patient_id, published_path]
        }
        .groupTuple(by: 0)
        .map { patient_id, files ->
            // Join all file paths with pipe delimiter (consistent with channels format)
            [patient_id, files.join('|')]
        }

    ch_checkpoint_data = PHENOTYPE.out.csv
        .map { meta, csv ->
            def published_path = "${params.outdir}/phenotype/${csv.name}"
            [meta.patient_id, published_path]
        }
        .join(PHENOTYPE.out.mask.map { meta, mask ->
            def published_path = "${params.outdir}/phenotype/${mask.name}"
            [meta.patient_id, published_path]
        })
        .join(PHENOTYPE.out.mapping.map { meta, map ->
            def published_path = "${params.outdir}/phenotype/${map.name}"
            [meta.patient_id, published_path]
        })
        .join(MERGE_QUANT_CSVS.out.merged_csv.map { meta, csv ->
            def published_path = "${params.outdir}/merge_quant_csvs/${csv.name}"
            [meta.patient_id, published_path]
        })
        .join(SEGMENT.out.cell_mask.map { meta, mask ->
            def published_path = "${params.outdir}/segment/${mask.name}"
            [meta.patient_id, published_path]
        })
        .join(ch_registered_grouped, by: 0)
        .map { patient_id, pheno_csv, pheno_mask, pheno_map, merged_csv, cell_mask, registered_images ->
            [
                patient_id,
                true,  // All checkpoint rows are for reference (one per patient)
                pheno_csv,
                pheno_mask,
                pheno_map,
                merged_csv,
                cell_mask,
                registered_images  // Pipe-delimited list of registered image paths
            ]
        }
        .collect()

    WRITE_CHECKPOINT_CSV(
        'postprocessed',
        'patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask,registered_images',
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

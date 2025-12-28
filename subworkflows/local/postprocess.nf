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
    ch_for_quant = SPLIT_CHANNELS.out.channels
        .map { meta, tiffs ->
            // Validate we got the expected number of channels
            // Account for DAPI being skipped on non-reference images
            def has_dapi = meta.channels.any { ch -> ch.toUpperCase() == 'DAPI' }
            def expected_channels = meta.is_reference ?
                meta.channels.size() :
                (has_dapi ? meta.channels.size() - 1 : meta.channels.size())

            def actual_channels = tiffs.size()
            if (actual_channels != expected_channels) {
                def ref_status = meta.is_reference ? "reference" : "non-reference"
                def dapi_note = has_dapi && !meta.is_reference ? " (DAPI skipped)" : ""
                throw new Exception("""
                Channel count mismatch for ${meta.patient_id} (${ref_status})!

                Expected ${expected_channels} channels${dapi_note}: ${meta.channels}
                Got ${actual_channels} channel files from SPLIT_CHANNELS

                This indicates SPLIT_CHANNELS may have failed or produced corrupted output.
                Check SPLIT_CHANNELS logs for patient ${meta.patient_id}
                """.stripIndent())
            }
            // Also validate no empty files
            tiffs.each { tiff ->
                if (tiff.size() == 0) {
                    throw new Exception("Empty channel file detected: ${tiff} for patient ${meta.patient_id}")
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
    // MERGE - Combine registered images with masks (per patient)
    // ========================================================================
    // Group registered images by patient for merging
    ch_registered_grouped = ch_registered
        .map { meta, file -> [meta.patient_id, meta, file] }
        .groupTuple(by: 0)
        .map { patient_id, _metas, files ->
            // Create patient-level metadata
            def patient_meta = [
                patient_id: patient_id,
                is_reference: false  // Not relevant at patient level
            ]
            [patient_meta, files]
        }

    // Join registered images with all masks for MERGE
    ch_for_merge = ch_registered_grouped
        .map { meta, files -> [meta.patient_id, meta, files] }
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
        .map { _patient_id, meta, registered_files, cell_mask, pheno_mask, pheno_mapping ->
            [meta, registered_files, cell_mask, pheno_mask, pheno_mapping]
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

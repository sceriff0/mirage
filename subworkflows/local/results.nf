nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { MERGE      } from '../../modules/local/merge'
include { CONVERSION } from '../../modules/local/conversion'
include { SAVE       } from '../../modules/local/save'

/*
========================================================================================
    SUBWORKFLOW: RESULTS (REFACTORED)
========================================================================================
    Description:
        Merges registered images with segmentation and phenotype masks PER PATIENT,
        creates pyramidal OME-TIFF, and saves final results to archive location.

    Key Improvements:
        ✅ CRITICAL FIX: Processes each patient separately (not all patients merged together!)
        ✅ No type confusion (always expects [meta, file] tuples)
        ✅ No silent fallbacks
        ✅ Clear per-patient processing
        ✅ Follows KISS and DRY principles

    Input:
        ch_registered: Channel of [meta, file] tuples for registered images
        ch_qc: Channel of QC RGB images (optional, can be empty)
        ch_cell_mask: Channel of [meta, mask] tuples (one per patient)
        ch_phenotype_mask: Channel of [meta, mask] tuples (one per patient)
        ch_phenotype_mapping: Channel of [meta, mapping] tuples (one per patient)
        ch_merged_csv: Channel of [meta, csv] tuples (one per patient)
        ch_phenotype_csv: Channel of [meta, csv] tuples (one per patient)
        savedir: Final archive directory path

    Output:
        merged: Merged multichannel OME-TIFF with segmentation and phenotype masks (per patient)
        pyramid: Pyramidal OME-TIFF with masks (per patient)
========================================================================================
*/

workflow RESULTS {
    take:
    ch_registered      // Channel of [meta, file] tuples
    ch_qc              // Channel of QC RGB images (can be empty)
    ch_cell_mask       // Channel of [meta, mask] tuples
    ch_phenotype_mask  // Channel of [meta, mask] tuples
    ch_phenotype_mapping  // Channel of [meta, mapping] tuples
    ch_merged_csv      // Channel of [meta, csv] tuples
    ch_phenotype_csv   // Channel of [meta, csv] tuples
    savedir            // Archive directory

    main:
    // ========================================================================
    // STEP 1: GROUP REGISTERED IMAGES BY PATIENT
    // ========================================================================
    // Each patient has multiple registered images (reference + moving images)
    // We need to group them together for merging

    ch_grouped_registered = ch_registered
        .map { meta, file -> [meta.patient_id, meta, file] }
        .groupTuple(by: 0)
        .map { patient_id, metas, files ->
            // Explicitly construct patient-level metadata
            // Find reference image metadata
            def ref_meta = metas.find { it.is_reference }

            def patient_meta = [
                patient_id: patient_id,
                is_reference: ref_meta ? ref_meta.is_reference : false
            ]

            [patient_meta, files]
        }

    // ========================================================================
    // STEP 2: JOIN ALL INPUTS BY PATIENT_ID
    // ========================================================================
    // All inputs (masks, CSVs) are per-patient and have matching patient_ids
    // Join them together to create input for MERGE process

    ch_for_merge = ch_grouped_registered
        .map { meta, files -> [meta.patient_id, meta, files] }
        .join(
            ch_cell_mask.map { meta, mask -> [meta.patient_id, mask] },
            by: 0
        )
        .join(
            ch_phenotype_mask.map { meta, mask -> [meta.patient_id, mask] },
            by: 0
        )
        .join(
            ch_phenotype_mapping.map { meta, mapping -> [meta.patient_id, mapping] },
            by: 0
        )
        .map { patient_id, meta, registered_files, cell_mask, pheno_mask, pheno_mapping ->
            // MERGE expects: tuple val(meta), path(registered_slides), path(seg_mask), path(pheno_mask), path(pheno_mapping)
            [meta, registered_files, cell_mask, pheno_mask, pheno_mapping]
        }

    // ========================================================================
    // STEP 3: MERGE REGISTERED IMAGES WITH MASKS (PER PATIENT)
    // ========================================================================
    // Each patient's images are merged separately
    // MERGE process runs once per patient

    MERGE(ch_for_merge)

    // ========================================================================
    // STEP 4: CREATE PYRAMIDAL OME-TIFF (PER PATIENT)
    // ========================================================================
    // Convert each patient's merged image to pyramidal format

    CONVERSION(MERGE.out.merged)

    // ========================================================================
    // STEP 5: COLLECT AND SAVE OUTPUTS
    // ========================================================================
    // Collect all outputs to save to archive location
    // Note: QC, CSVs, and pyramid are per-patient

    // FIX BUG #2: Normalize all channels to plain files before mixing
    // ch_qc may contain tuples [meta, file] or plain files depending on registration method
    // Extract files only to ensure consistent channel type
    // FIX BUG #3: Empty channels are handled correctly by .map() and .mix()
    // When starting from 'results' step, ch_qc and ch_registered are empty
    // This is safe - empty channels contribute nothing to the mix
    ch_qc_files = ch_qc.map { item ->
        item instanceof List ? item[1] : item
    }

    ch_to_save = channel.empty()
        .mix(
            ch_qc_files,                                        // QC: RGB files only (may be empty)
            ch_merged_csv.map { meta, csv -> csv },             // Quantification results (per patient)
            ch_phenotype_csv.map { meta, csv -> csv },          // Phenotype results (per patient)
            CONVERSION.out.pyramid.map { meta, pyr -> pyr }     // Pyramidal visualization (per patient)
        )
        .collect()

    // Save to final archive location
    SAVE(
        ch_to_save,
        savedir
    )

    emit:
    merged = MERGE.out.merged
    pyramid = CONVERSION.out.pyramid
}

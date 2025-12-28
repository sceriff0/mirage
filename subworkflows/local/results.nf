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

    // FIX ISSUE #5: Add validation to detect missing patients in joins
    // Track which patients we expect vs what we get after joins

    // Collect expected patients from registered images
    ch_expected_patients = ch_grouped_registered
        .map { meta, files -> meta.patient_id }
        .collect()
        .map { it.toSet() }

    // Perform joins
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

    // Validate all expected patients made it through the joins
    ch_for_merge
        .map { meta, files, cell_mask, pheno_mask, pheno_mapping -> meta.patient_id }
        .collect()
        .map { it.toSet() }
        .combine(ch_expected_patients)
        .subscribe { processed_patients, expected_patients ->
            def missing = expected_patients - processed_patients

            if (!missing.isEmpty()) {
                log.error """
                ❌ CRITICAL: Some patients were dropped during join operations!

                Expected patients: ${expected_patients.size()}
                Processed patients: ${processed_patients.size()}
                Missing patients: ${missing}

                💡 This usually means:
                   1. SEGMENT failed for these patients (no cell_mask)
                   2. PHENOTYPE failed for these patients (no phenotype outputs)
                   3. Mismatched patient_id values between channels

                💡 Check earlier process logs for failures in:
                   - SEGMENT
                   - QUANTIFY
                   - MERGE_QUANT_CSVS
                   - PHENOTYPE

                ⚠️  Pipeline will continue but these patients will NOT be in final results!
                """.stripIndent()
            } else {
                log.info "✅ All ${expected_patients.size()} patients successfully joined for MERGE"
            }
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

    // FIX BUG #2: Extract QC files from tuples
    // GENERATE_REGISTRATION_QC emits: tuple val(meta), path("qc/*_QC_RGB.{png,tif}")
    // The path() glob returns a LIST of files [png, tif]
    // We need to flatten this to get individual files
    ch_qc_files = ch_qc.flatMap { meta, files ->
        // files is a list containing [png_file, tif_file]
        // Return the list of files (without meta)
        files instanceof List ? files : [files]
    }

    ch_to_save = channel.empty()
        .mix(
            ch_qc_files,                                        // QC: PNG and TIFF files (may be empty)
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

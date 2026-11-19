/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { GENERATE_REGISTRATION_QC          } from '../../modules/local/generate_registration_qc'

include { SEG_QC                            } from './seg_qc'

// Passthrough, backend dispatch and checkpoint live in REGISTER_PATIENT (shared with add_cycle).
include { REGISTER_PATIENT                  } from './register_patient'


/*
========================================================================================
    SUBWORKFLOW: REGISTRATION
========================================================================================
    Configuration:
        - params.skip_registration_qc: true | false (skip QC generation)
        - params.qc_scale_factor: float (QC downsampling factor, default 0.25)
        - params.registration_method: 'valis' | 'tiled' -- read once here and passed
          as an argument to REGISTER_PATIENT and SEG_QC, which never read the global.

    Input:
        ch_preprocessed: Channel of [meta, file] tuples

    Output:
        registered: Channel of [meta, file] tuples (standard format)
        qc: Channel of QC outputs (PNG and TIFF)
        checkpoint_csv: Checkpoint CSV file

    QC (params.reg_qc):
        - 1: GENERATE_REGISTRATION_QC, registered vs reference DAPI (TIFF + PNG)
        - 2: also SEG_QC, segmentation overlap before vs after registration
        - the method's own TRE estimate is passed through as registration_tre
========================================================================================
*/

workflow REGISTRATION {
    take:
    ch_preprocessed

    main:
    // Both backends handle inputs of differing sizes, so no padding step is needed.
    ch_images = ch_preprocessed

    // ========================================================================
    // STEP 2: GROUP BY PATIENT AND IDENTIFY REFERENCES
    // ========================================================================
    // Output: [patient_id, reference_item, all_items]
    //   where reference_item = [meta, file]
    //   and all_items = [[meta1, file1], [meta2, file2], ...]
    // groupKey(images_count) emits each patient as soon as its slides are in.

    ch_grouped = ch_images
        .map { meta, file ->
            def key = meta.images_count ? groupKey(meta.patient_id, meta.images_count) : meta.patient_id
            [key, meta, file]
        }
        .groupTuple()
        .map { group_key, metas, files ->
            // Unwrap the groupKey to a plain String: GroupKey.equals() is asymmetric, so a
            // later combine keyed on String vs GroupKey can silently miss
            // (tests/test_group_key_unwrapped.py).
            def patient_id = group_key.toString()

            // groupTuple emits in arrival order and REGISTER hashes its lists positionally,
            // so sort for -resume. Pair first, then sort: groupTuple(sort:) sorts each list
            // on its own and re-pairs metas with the wrong files.
            def items = [metas, files].transpose().toSorted { it[0].id }

            // The reference was resolved in INPUT_CHECK; this only guards it. Nothing infers one.
            def ref = items.find { item -> item[0].is_reference }
            if (!ref) {
                throw new Exception("""
                No reference image found for patient ${patient_id}
                Fix: set is_reference=true on exactly one of that patient's images
                in your input CSV. There is no auto-promotion.
                """.stripIndent())
            }

            [patient_id, ref, items]
        }

    // ========================================================================
    // STEP 3: REGISTER EACH PATIENT GROUP
    // ========================================================================
    // The one read of params.registration_method on this path, passed to both consumers.
    def registration_method = params.registration_method

    REGISTER_PATIENT(ch_grouped, registration_method)

    ch_registered         = REGISTER_PATIENT.out.registered
    ch_images_multi       = REGISTER_PATIENT.out.images_multi
    // Per-patient and per-slide transforms; the one a method lacks is Channel.empty().
    ch_transform          = REGISTER_PATIENT.out.transform
    ch_transform_by_slide = REGISTER_PATIENT.out.transform_by_slide
    ch_stage_checkpoint   = REGISTER_PATIENT.out.stage_checkpoint
    ch_adapter_logs       = REGISTER_PATIENT.out.size_logs
    ch_adapter_versions   = REGISTER_PATIENT.out.versions
    // The method's own TRE estimate (VALIS CSV, STARE JSON); may be empty.
    ch_registration_tre   = REGISTER_PATIENT.out.intrinsic_tre

    // reg_qc: 0 = none, 1 = DAPI overlay, 2 = + segmentation overlap
    // (skip_registration_qc=true forces 0).
    def reg_qc_level = ParamUtils.regQcLevel(params)

    // Level 2 warps polygons through the method's transform (lib/WarpBackends.groovy picks
    // the warper); segmentation and scoring are shared.
    def do_seg_qc = reg_qc_level >= 2

    // ========================================================================
    // STEP 3b: GENERATE QC (method-independent)
    // ========================================================================
    // Input for QC: [meta, registered_file, native_file, reference_file]
    ch_qc_input = ch_registered
        .branch {
            reference: it[0].is_reference
            moving: !it[0].is_reference
        }

    ch_references_for_qc = ch_qc_input.reference
        .map { meta, file -> [meta.patient_id, file] }

    // The native (pre-registration) image of each slide, for the "before" panel.
    // Keyed on meta.id, not the meta map: a map key breaks on any `meta + [k: v]`, and
    // join() drops unmatched items silently.
    ch_natives_for_qc = ch_images_multi
        .map { meta, file -> [meta.id, file] }

    // Each moving image with its native counterpart (1:1) and its patient's reference (1:N).
    ch_for_qc = ch_qc_input.moving
        .map { meta, file -> [meta.id, meta, file] }
        .join(ch_natives_for_qc, by: 0)
        .map { _id, meta, registered_file, native_file ->
            [meta.patient_id, meta, registered_file, native_file]
        }
        .combine(ch_references_for_qc, by: 0)
        .map { _patient_id, meta, registered_file, native_file, reference_file ->
            [meta, registered_file, native_file, reference_file]
        }

    // Level >= 1: DAPI overlay image QC for all non-reference images
    if (reg_qc_level >= 1) {
        GENERATE_REGISTRATION_QC(ch_for_qc)
        ch_qc = GENERATE_REGISTRATION_QC.out.qc
    } else {
        ch_qc = Channel.empty()
    }

    // Level >= 2: segment each native slide, warp the cell polygons through the transform and
    // score overlap (IoU, dice_matched, centroid residuals) before vs after.
    ch_seg_qc = Channel.empty()
    // Per-cell registration residuals (final stage), for the SpatialData export.
    ch_seg_residuals = Channel.empty()
    ch_seg_qc_size_log = Channel.empty()
    ch_seg_qc_versions = Channel.empty()
    // SEG_QC is shared with add_cycle.nf; the transform the method lacks arrives empty.
    if (do_seg_qc) {
        SEG_QC(ch_images_multi, ch_transform, ch_stage_checkpoint, ch_transform_by_slide,
               registration_method)
        ch_seg_qc          = SEG_QC.out.metrics
        ch_seg_residuals   = SEG_QC.out.per_cell
        ch_seg_qc_size_log = SEG_QC.out.size_log
        ch_seg_qc_versions = SEG_QC.out.versions
    }

    // ========================================================================
    // STEP 4: CHECKPOINT
    // ========================================================================
    // Written by REGISTER_PATIENT, so add_cycle produces the same manifest.
    ch_checkpoint_csv = REGISTER_PATIENT.out.checkpoint_csv

    ch_size_logs = Channel.empty()
    ch_size_logs = ch_size_logs.mix(ch_adapter_logs)

    if (reg_qc_level >= 1) {
        ch_size_logs = ch_size_logs.mix(GENERATE_REGISTRATION_QC.out.size_log)
    }
    if (do_seg_qc) {
        ch_size_logs = ch_size_logs.mix(ch_seg_qc_size_log)
    }

    ch_versions = Channel.empty()

    ch_versions = ch_versions.mix(ch_adapter_versions)

    if (reg_qc_level >= 1) {
        ch_versions = ch_versions.mix(GENERATE_REGISTRATION_QC.out.versions.first())
    }
    // SEG_QC.out.versions is already de-duplicated.
    if (do_seg_qc) {
        ch_versions = ch_versions.mix(ch_seg_qc_versions)
    }

    emit:
    registered       = ch_registered
    checkpoint_csv   = ch_checkpoint_csv
    qc               = ch_qc
    seg_qc           = ch_seg_qc
    seg_residuals    = ch_seg_residuals
    registration_tre = ch_registration_tre
    size_logs        = ch_size_logs
    versions         = ch_versions
}

/*
========================================================================================
    VALIS REGISTRATION ADAPTER
========================================================================================
    Converts patient-grouped data to VALIS batch format and back. VALIS needs all of a
    patient's images at once to build its registration graph.

    Input:  ch_grouped_meta - Channel of [patient_id, reference_item, all_items]
            where reference_item = [meta, file] for the reference image
            and all_items = [[meta1, file1], [meta2, file2], ...] for all images
    Output: Channel of [meta, file] tuples (standard format)

    Adapter contract (shared with adapters/tiled_adapter.nf; any new adapter must follow it):

        registered          [meta, file]                registered slides (+ passthroughs)
        transform           [patient_id, transform]     one transform object per patient
        transform_by_slide  [meta, transform]           one transform per moving slide
        stage_checkpoint    [patient_id, dir]           intermediate-stage fields
        intrinsic_tre       file                        the method's own TRE estimate, in its
                                                        own format (the reader handles formats)
        size_logs / versions

    An emit the method cannot produce is `Channel.empty()`, never missing, so consumers
    need not know which method ran.
========================================================================================
*/

include { REGISTER } from '../../../modules/local/register'

workflow VALIS_ADAPTER {
    take:
    ch_grouped_meta   // Channel of [patient_id, reference_item, all_items] from grouping

    main:
    // ========================================================================
    // CONVERT TO VALIS BATCH FORMAT
    // ========================================================================
    // VALIS needs: [patient_id, reference_file, [all_files], [all_metas]]

    ch_valis_input = ch_grouped_meta
        .map { patient_id, ref_item, all_items ->
            def ref_file = ref_item[1]

            // The reference goes both separately (--reference) and in all_files;
            // REGISTER's stageAs avoids the filename collision.
            def all_files = all_items.collect { item -> item[1] }
            def all_metas = all_items.collect { item -> item[0] }

            // Not a sample meta (so not built through lib/Meta.groovy): REGISTER's control
            // tuple, carrying patient_id for publishDir. The per-slide metas are all_metas.
            // Exempted in tests/test_meta_module.py's ALLOWED set.
            def meta = [patient_id: patient_id]
            tuple(meta, patient_id, ref_file, all_files, all_metas)
        }

    // ========================================================================
    // RUN VALIS BATCH REGISTRATION
    // ========================================================================

    REGISTER(ch_valis_input)

    // ========================================================================
    // CONVERT BACK TO STANDARD FORMAT
    // ========================================================================
    // [patient_id, [registered_files], [metas], manifest_file] -> [meta, file], matched by
    // OME channel names (VALIS renames its outputs). The rule is lib/RegisteredMatch.groovy.

    ch_registered = REGISTER.out.registered
        .flatMap { patient_id, reg_files, metas, manifest_file ->
            // Normalize to lists — Nextflow unwraps single-element globs/vals
            // to bare Path/Map objects, breaking .size() and .collect()
            def files_list = reg_files instanceof List ? reg_files : [reg_files]
            def metas_list = metas instanceof List ? metas : [metas]

            // filename -> [channel names], written by bin/create_channels_manifest.py in REGISTER.
            def manifest = new groovy.json.JsonSlurper().parseText(manifest_file.text)

            try {
                return RegisteredMatch.pair(metas_list, files_list, manifest)
            }
            catch (RuntimeException e) {
                // Catches IllegalArgumentException (null channels) as well as
                // IllegalStateException, so every error names the patient.
                throw new IllegalStateException(
                    "VALIS adapter, patient ${patient_id}: ${e.message}", e)
            }
        }

    ch_size_logs = REGISTER.out.size_log

    emit:
    registered = ch_registered
    // [patient_id, registrar.pickle]: one group-wide transform per patient, used by the seg-QC warper.
    transform  = REGISTER.out.registrar
    // VALIS has no per-slide transform.
    transform_by_slide = Channel.empty()
    // [patient_id, reg_stage_checkpoint/]: pre-micro displacement fields (reg_qc >= 2), so
    // WARP_SEG_QC can score non_rigid apart from micro; VALIS composes the two.
    stage_checkpoint = REGISTER.out.stage_checkpoint
    size_logs  = ch_size_logs
    versions   = REGISTER.out.versions.first()
    // VALIS's intrinsic TRE: error_df (*_summary.csv), feature distances on its own keypoints.
    intrinsic_tre = REGISTER.out.summary
}

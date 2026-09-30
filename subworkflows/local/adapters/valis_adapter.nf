/*
========================================================================================
    VALIS REGISTRATION ADAPTER
========================================================================================
    Adapter that converts patient-grouped data to VALIS batch format and back.

    VALIS requires all images for a patient at once to build optimal transformation graph.
    This adapter handles the batch conversion while maintaining the standard interface.

    Input:  ch_grouped_meta - Channel of [patient_id, reference_item, all_items]
            where reference_item = [meta, file] for the reference image
            and all_items = [[meta1, file1], [meta2, file2], ...] for all images
    Output: Channel of [meta, file] tuples (standard format)

    THE ADAPTER CONTRACT (identical in adapters/tiled_adapter.nf, and binding on any third
    adapter). Every registration adapter takes [patient_id, reference_item, all_items]
    and emits EXACTLY these names:

        registered          [meta, file]                registered slides (+ passthroughs)
        transform           [patient_id, transform]     ONE transform object per patient
        transform_by_slide  [meta, transform]           one transform per MOVING slide
        stage_checkpoint    [patient_id, dir]           intermediate-stage fields
        intrinsic_tre       file                        the method's OWN target-registration
                                                        -error estimate, whatever its format
        size_logs / versions

    `intrinsic_tre` is deliberately NOT named after any one method. Both shipped backends
    estimate a TRE from their own registration -- VALIS a feature-distance CSV, STARE a
    *_tre.json -- and the seam used to call the slot `summary` and then re-emit it as
    `valis_summary`, which pinned one method's name into the artifact vocabulary all the way
    out to the QC report. Formats are NOT normalised here; that is the reader's job.

    A method that produces no artifact for one of these emits `Channel.empty()` for it --
    a NULL OBJECT, never a missing emit and never an error. That is what lets
    REGISTER_PATIENT wire both backends with one short branch, and it is why nothing
    downstream of that branch has to know which method ran. A future adapter inherits the
    rule: declare every name; empty the ones your method cannot produce.
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

            // VALIS requires all slides, including the reference, for graph optimization.
            // We pass reference both separately (for --reference flag) AND in all_files
            // The REGISTER process uses stageAs to avoid filename collision
            def all_files = all_items.collect { item -> item[1] }
            def all_metas = all_items.collect { item -> item[0] }

            // NOT a sample meta, and deliberately not built through Meta (lib/Meta.groovy):
            // this is REGISTER's process-input control tuple (modules/local/register.nf
            // -- "meta carries patient_id for publishDir consistency across all
            // processes"), consumed only as tuple(meta, ...)'s first field for a
            // FAN-IN process call. It carries none of Meta.REQUIRED_KEYS and never
            // reaches an `emit:` or a [meta, file] sample stream -- the real per-slide
            // metas are `all_metas`, already built (via Meta.fromSamplesheetRow /
            // Meta.fromCheckpointRow) by whichever upstream reader produced them.
            // Documented, permanent exemption in tests/test_meta_module.py's ALLOWED
            // set: forcing this through Meta.fromSamplesheetRow/fromCheckpointRow would
            // mean inventing values for five keys this call site has no data to supply
            // truthfully.
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
    // VALIS outputs: [patient_id, [registered_files], [metas], manifest_file]
    // Need to convert to: [meta, file]
    //
    // Match registered outputs back to metadata using OME channel names
    // from the channels manifest (no filename parsing — VALIS renames its
    // outputs, and name parsing broke on any patient id containing '_').
    // The rule is lib/RegisteredMatch.groovy; this is wiring only.

    ch_registered = REGISTER.out.registered
        .flatMap { patient_id, reg_files, metas, manifest_file ->
            // Normalize to lists — Nextflow unwraps single-element globs/vals
            // to bare Path/Map objects, breaking .size() and .collect()
            def files_list = reg_files instanceof List ? reg_files : [reg_files]
            def metas_list = metas instanceof List ? metas : [metas]

            // filename -> [channel names], from the OME-XML of each registered output.
            // Written by bin/create_channels_manifest.py inside REGISTER.
            def manifest = new groovy.json.JsonSlurper().parseText(manifest_file.text)

            // The matching rule itself is lib/RegisteredMatch.groovy, where it is
            // unit-tested (tests/lib_probe.nf) and reachable by a second backend.
            // The patient id is added here because it is wiring context this
            // subworkflow has and that class deliberately does not.
            try {
                return RegisteredMatch.pair(metas_list, files_list, manifest)
            }
            catch (RuntimeException e) {
                // RegisteredMatch.pair throws IllegalStateException for its own three
                // failure modes, but signature() throws IllegalArgumentException when
                // a meta carries null channels -- catch both, or the patient prefix
                // below is silently dropped for that case.
                throw new IllegalStateException(
                    "VALIS adapter, patient ${patient_id}: ${e.message}", e)
            }
        }

    // Collect size logs
    ch_size_logs = REGISTER.out.size_log

    emit:
    registered = ch_registered
    // [patient_id, registrar.pickle] — VALIS's transform is ONE object per patient: the graph
    // it optimised over the whole group. Consumed by the GeoJSON seg-QC warper.
    transform  = REGISTER.out.registrar
    // VALIS has no per-slide transform: the registrar is a single group-wide object and is not
    // decomposable into one transform per moving slide. This is the null object the contract
    // above requires — not an omission, and not an error for any consumer.
    transform_by_slide = Channel.empty()
    // [patient_id, reg_stage_checkpoint/] — pre-micro displacement fields, emitted only at
    // reg_qc >= 2. Lets WARP_SEG_QC score the non_rigid stage apart from micro; VALIS composes
    // the two into one field, so REGISTER is the only place they can be told apart.
    stage_checkpoint = REGISTER.out.stage_checkpoint
    size_logs  = ch_size_logs
    versions   = REGISTER.out.versions.first()
    // VALIS's intrinsic TRE: `error_df` written as preprocessed/data/*_summary.csv, the
    // feature distances it measures on its OWN SuperPoint/SuperGlue keypoints.
    intrinsic_tre = REGISTER.out.summary
}

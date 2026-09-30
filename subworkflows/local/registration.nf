/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { GENERATE_REGISTRATION_QC          } from '../../modules/local/generate_registration_qc'

include { SEG_QC                            } from './seg_qc'

// The passthrough branch, the VALIS/tiled dispatch and the checkpoint manifest are
// REGISTER_PATIENT's, shared with add_cycle.nf.
include { REGISTER_PATIENT                  } from './register_patient'


/*
========================================================================================
    SUBWORKFLOW: REGISTRATION
========================================================================================
    Configuration:
        - params.skip_registration_qc: true | false (skip QC generation)
        - params.qc_scale_factor: float (QC downsampling factor, default 0.25)
        - params.registration_method: 'valis' | 'tiled' -- read ONCE here, the single
          decision site on this path, and handed as an ARGUMENT to both REGISTER_PATIENT
          (which owns the adapter dispatch) and SEG_QC (which owns the warp dispatch).
          Neither reads the global, so neither can be what lifts add_cycle's VALIS-only
          guarantee.

    Input:
        ch_preprocessed: Channel of [meta, file] tuples

    Output:
        registered: Channel of [meta, file] tuples (standard format)
        qc: Channel of QC outputs (PNG and TIFF)
        checkpoint_csv: Checkpoint CSV file

    QC Generation:
        - Decoupled from registration methods
        - Uses unified GENERATE_REGISTRATION_QC module
        - Compares registered vs reference DAPI channels
        - Outputs: full-res TIFF + downsampled PNG

    Error Estimation (Optional):
        - BEFORE registration: COMPUTE_FEATURES extracts matched keypoints
        - AFTER registration: Two complementary methods:
          1. Feature-based TRE (fast, sparse measurements)
          2. Segmentation-based IoU/Dice (dense, biologically meaningful)
========================================================================================
*/

workflow REGISTRATION {
    take:
    ch_preprocessed

    main:
    // ========================================================================
    // STEP 1: Images enter registration as-is. Both backends align inputs of
    // differing sizes natively (VALIS resolves them into a shared space; the
    // tiled/STARE backend warps each moving slide into the reference's shape),
    // so no common-canvas padding step is needed.
    // ========================================================================
    ch_images = ch_preprocessed

    // ========================================================================
    // STEP 2: GROUP BY PATIENT AND IDENTIFY REFERENCES
    // ========================================================================
    // This is common preparation needed by all methods
    // Output: [patient_id, reference_item, all_items]
    //   where reference_item = [meta, file]
    //   and all_items = [[meta1, file1], [meta2, file2], ...]
    //
    // Use groupKey for streaming - emits as soon as images_count items collected
    // This enables patient-level parallelism (Patient A processes while Patient B preprocesses)

    ch_grouped = ch_images
        .map { meta, file ->
            def key = meta.images_count ? groupKey(meta.patient_id, meta.images_count) : meta.patient_id
            [key, meta, file]
        }
        .groupTuple()
        .map { group_key, metas, files ->
            // UNWRAP THE groupKey. `groupKey(id, size)` above is a STREAMING SIZE HINT,
            // nothing more -- but groupTuple() emits the nextflow.extension.GroupKey
            // OBJECT itself as element 0, and it used to travel on from here as the
            // patient id: into REGISTER's `val(patient_id)` and back out on
            // REGISTER.out.registrar / .stage_checkpoint. GroupKey delegates hashCode()
            // to the id it wraps but its equals() is ASYMMETRIC --
            // `groupKey('P001',2).equals('P001')` is true, `'P001'.equals(groupKey(...))`
            // is false -- so a HashMap lookup matches in one direction only.
            //
            // That is a silent, ~1-run-in-2-under-load data loss, not a cosmetic issue.
            // seg_qc.nf keys its GeoJSON tuples off the plain String meta.patient_id and
            // combines them with the transform, whose key was this GroupKey.
            // CombineOp.emit() stores each side in a HashMap under the `by` values and
            // crosses with what the other side has already stored, so WHICHEVER SIDE
            // ARRIVES FIRST decides the stored key type: GeoJSON first -> stored under
            // String, and the later GroupKey lookup hits; transform first -> stored under
            // GroupKey, and the later String lookup MISSES and the patient's whole
            // registration QC is dropped with the run still exiting 0.
            //
            // The id is a plain String everywhere else it is used (meta.patient_id, from
            // the samplesheet), so it must be a plain String here too. Unwrapped the
            // moment the group exists -- the hint has done its job by then.
            // tests/test_group_key_unwrapped.py fails if any groupKey escapes its
            // grouping again.
            def patient_id = group_key.toString()

            // CANONICAL ORDER. groupTuple() emits in ARRIVAL order -- whichever slide
            // finished preprocessing first -- and everything below carries that order
            // into REGISTER as `preproc_files` / `all_metas`, which Nextflow hashes
            // POSITIONALLY. So an identical rerun produced a different task hash and
            // -resume missed on REGISTER, cascading a re-run of the entire downstream
            // pipeline. Sorting by meta.id makes the order a function of the data.
            //
            // SORT AFTER transpose(), NEVER `groupTuple(sort:)`. groupTuple's own sort
            // orders each grouped list INDEPENDENTLY, so metas and files are sorted by
            // their own natural orders and silently re-paired:
            //     in:  ['k','zeta','a.txt'], ['k','alpha','z.txt']
            //     groupTuple(sort:true) -> [k, [alpha, zeta], [a.txt, z.txt]]
            //                                   ^ alpha now pairs with a.txt -- WRONG
            // Pairing first and sorting the pairs is the only safe form.
            // subworkflows/local/add_cycle.nf:334 is the existing precedent.
            def items = [metas, files].transpose().toSorted { it[0].id }

            // The reference was RESOLVED at samplesheet-read time (INPUT_CHECK, via
            // CsvUtils.resolveReferenceRows) and travels in meta.is_reference. This is
            // a guard on that invariant, not a fallback: it used to promote items[0]
            // here, which meant the reference was whichever slide arrived first AND was
            // decided downstream of the preprocessing checkpoint writer. Reaching this
            // error now means the samplesheet named no reference for this patient.
            // NOTHING infers one -- not here, not at samplesheet-read time. The
            // reference is the slide every other slide is warped onto; a pipeline that
            // picks it silently produces a run nobody can reproduce from the sheet.
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
    // The single-slide passthrough branch, the method dispatch (VALIS / tiled) and the
    // checkpoint manifest all live in REGISTER_PATIENT, which add_cycle.nf also calls.
    //
    // The GROUPING above stays here: the linear path's sized groupKey has no
    // counterpart on the add_cycle path, whose reference is the frozen prior one.
    // The ONE read of params.registration_method on this path. Bound once and passed as an
    // argument to both consumers, so there is a single decision site instead of two reads
    // that could drift apart.
    def registration_method = params.registration_method

    REGISTER_PATIENT(ch_grouped, registration_method)

    ch_registered         = REGISTER_PATIENT.out.registered
    ch_images_multi       = REGISTER_PATIENT.out.images_multi
    // The method's transform, per patient (VALIS registrar pickle / STARE manifest) and --
    // where the method has one -- per moving slide. The unfilled one is Channel.empty().
    ch_transform          = REGISTER_PATIENT.out.transform
    ch_transform_by_slide = REGISTER_PATIENT.out.transform_by_slide
    ch_stage_checkpoint   = REGISTER_PATIENT.out.stage_checkpoint
    ch_adapter_logs       = REGISTER_PATIENT.out.size_logs
    ch_adapter_versions   = REGISTER_PATIENT.out.versions
    // The method's OWN target-registration-error estimate. Both backends produce one
    // (VALIS a feature-distance CSV, STARE a *_tre.json); a backend that produced none
    // would emit Channel.empty() here and every consumer must tolerate that.
    ch_registration_tre   = REGISTER_PATIENT.out.intrinsic_tre

    // reg_qc: 0 = none, 1 = DAPI overlay, 2 = + segmentation overlap (legacy
    // skip_registration_qc=true forces 0). ParamUtils.regQcLevel is the single definition.
    def reg_qc_level = ParamUtils.regQcLevel(params)

    // Level 2 warps polygons through the registrar the method produced — the VALIS pickle (via the
    // BioFormats JVM) or the STARE manifest (JVM-free). The scorer is identical; only the warper
    // differs (WARP_SEG_QC's two backends, keyed by lib/WarpBackends.groovy). Segmentation of
    // the native slides is shared.
    def do_seg_qc = reg_qc_level >= 2

    // ========================================================================
    // STEP 3b: GENERATE QC (Method-independent)
    // ========================================================================
    // For each registered image, create QC comparing it to its reference
    // This is now decoupled from the registration method

    // Prepare input for QC: [meta, registered_file, native_file, reference_file]
    ch_qc_input = ch_registered
        .branch {
            reference: it[0].is_reference
            moving: !it[0].is_reference
        }

    // Extract references by patient
    ch_references_for_qc = ch_qc_input.reference
        .map { meta, file -> [meta.patient_id, file] }

    // The NATIVE (pre-registration) image of every slide, keyed by meta.id.
    // REGISTER_PATIENT.out.images_multi is the multi-slide patients' input stream --
    // exactly what entered registration -- so the "before" panel needs no new process
    // and no second read of the samplesheet.
    //
    // Keyed on meta.id, not on the meta MAP. Both channels do carry the same meta
    // object for a given slide today, but a map key is one `meta + [k: v]` on either
    // branch away from never matching again, and join() drops an unmatched pair
    // silently with the run still exiting 0. meta.id is a plain String, set once at
    // samplesheet-read time, unique per slide -- which is also what join() requires.
    ch_natives_for_qc = ch_images_multi
        .map { meta, file -> [meta.id, file] }

    // Combine moving images with their patient's reference (1 reference to N moving
    // images) and with their own native counterpart (1:1 on meta.id).
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

    // Level >= 2: GeoJSON segmentation-overlap QC (per-pair IoU, dice_matched, centroid
    // residuals) BEFORE vs AFTER.
    // Segment each slide's DAPI on its NATIVE (pre-registration) image -> cell GeoJSON,
    // then warp the polygons through the registrar and score overlap. Classic VALIS only
    // (needs the registrar pickle; distributed produces none).
    ch_seg_qc = Channel.empty()
    // Per-cell registration residuals (final stage), for the SpatialData export.
    ch_seg_residuals = Channel.empty()
    // Captured from whichever WARP process the method dispatches to, so the version/size-log
    // aggregation below never references a process that was not invoked in this run.
    ch_seg_qc_size_log = Channel.empty()
    ch_seg_qc_versions = Channel.empty()
    // The whole block (segment natives -> branch -> checkpoint totalisation -> method-specific
    // warp) lives in subworkflows/local/seg_qc.nf, shared with add_cycle.nf. Both transform
    // channels are handed over under the adapters' shared names; whichever one the method does
    // not produce is Channel.empty() (the adapters' null-object contract) and is never read.
    // The method reaches SEG_QC as an ARGUMENT — the same value REGISTER_PATIENT got above.
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
    // Written by REGISTER_PATIENT (via CHECKPOINT_WRITER), so the add_cycle path
    // — which does not come through here — produces the identical manifest.
    ch_checkpoint_csv = REGISTER_PATIENT.out.checkpoint_csv

    // Collect size logs from all registration processes
    ch_size_logs = Channel.empty()

    // Add size logs from the registration adapter
    ch_size_logs = ch_size_logs.mix(ch_adapter_logs)

    // Add size logs from QC processes (if enabled)
    if (reg_qc_level >= 1) {
        ch_size_logs = ch_size_logs.mix(GENERATE_REGISTRATION_QC.out.size_log)
    }
    // SEG_QC.out.size_log already carries SEG_QC_GEOJSON's log plus the warp process's.
    if (do_seg_qc) {
        ch_size_logs = ch_size_logs.mix(ch_seg_qc_size_log)
    }

    // Collect versions from all registration processes
    ch_versions = Channel.empty()

    ch_versions = ch_versions.mix(ch_adapter_versions)

    if (reg_qc_level >= 1) {
        ch_versions = ch_versions.mix(GENERATE_REGISTRATION_QC.out.versions.first())
    }
    // SEG_QC.out.versions is already de-duplicated per process (.first() applied inside).
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

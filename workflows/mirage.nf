/*
================================================================================
    MIRAGE WSI Processing Pipeline — Main Workflow
================================================================================
    Validates the launch parameters and routes the run to the right subworkflow.
    Channel shaping and aggregation belong to the subworkflows:

      samplesheet -> sample channel        subworkflows/local/input_check.nf
      prior-run asset reconstruction       subworkflows/local/add_cycle.nf
      QC report + size-log aggregation     subworkflows/local/final_qc.nf
================================================================================
*/

include { validateParameters  } from 'plugin/nf-schema'

include { INPUT_CHECK         } from '../subworkflows/local/input_check'
include { PREPROCESSING       } from '../subworkflows/local/preprocess'
include { REGISTRATION        } from '../subworkflows/local/registration'
include { SEGMENTATION; READ_SEGMENTED_CHECKPOINT } from '../subworkflows/local/segmentation'
include { POSTPROCESSING      } from '../subworkflows/local/postprocess'
include { FINAL_QC            } from '../subworkflows/local/final_qc'

// Unknown columns are accepted (checkpoint CSVs carry extra columns) but warned about,
// so a typo is not silently read as null. Script-level so `log` resolves; shared by
// the add_cycle and linear paths.
def warnUnknownColumns(String csv, String step) {
    def unknown_columns = CsvUtils.unknownColumns(csv, step)
    if (unknown_columns)
        log.warn "Samplesheet column(s) this pipeline does not read, and will ignore: " +
                 "${unknown_columns.join(', ')} (in ${csv})"
}

workflow MIRAGE {

    /* -------------------- PARAMETER VALIDATION -------------------- */

    // Types, enums and ranges come from nextflow_schema.json (nf-schema); cross-parameter
    // and filesystem rules follow in Groovy. Retired params are checked first: they pass
    // nf-schema via validation.ignoreParams so this gives the specific message.
    def retiredWarning = ParamUtils.checkRetiredParams(params)
    if (retiredWarning) log.warn(retiredWarning)
    validateParameters()

    ParamUtils.validateOutdir(params.outdir)
    // Cross-parameter checks, all before any process is instantiated.
    ParamUtils.validatePixelSize(params)
    ParamUtils.validateStareKnobs(params)
    // add_cycle must produce a re-enterable tree, so a cleaning level is refused.
    ParamUtils.validateCleanup(params)
    // --seg_qc_nuclei_dir: reg_qc=2 only, and an existing directory.
    ParamUtils.validateSegQcNucleiDir(params)
    // Refuse a run whose params disagree with the scalars nextflow.config froze before
    // `-c` files were merged. The command line exempts Nextflow's own -with-trace etc.
    ParamUtils.validateFrozenConfig(params, workflow.session.config, workflow.commandLine)

    // cleanup_work empties work/ after success, so the next -resume caches nothing.
    // Warn rather than refuse: the cost is a re-run, not lost data.
    if (params.cleanup_work && workflow.resume) {
        log.warn "--cleanup_work is true (the default) AND -resume was passed. The work " +
                 "directory is emptied after a successful run, so the NEXT -resume will " +
                 "find nothing cached and re-run every task. Set cleanup_work to false in " +
                 "a -params-file or a profile (Nextflow 26 rejects a boolean value on the " +
                 "command line) for the iterate-with--resume loop docs/usage.md describes."
    }

    // --start past preprocessing at a cleaning level: this run reads the prior tree fine,
    // but its own output cannot be re-entered the same way.
    if (params.cleanup_level != 'none' && !ParamUtils.isEntryPoint(params, 'preprocessing')) {
        log.warn "--start ${params.start} re-enters from intermediates that " +
                 "--cleanup_level=${params.cleanup_level} does not publish. This run reads " +
                 "the prior output fine, but its OWN output cannot be re-entered the same " +
                 "way. Pass --cleanup_level none to keep that open."
    }

    /* -------------------- STEP GATE -------------------- */

    // Validate and resolve --stop (default: last step). Before the mode branch so
    // add_cycle shares the --start/--stop validation.
    if (params.stop) {
        ParamUtils.validateStop(params.stop, params.start)
    }
    def effective_stop = params.stop ?: ParamUtils.STEP_ORDER.last()

    // Which steps this run covers, computed once. add_cycle does not use these.
    boolean run_preprocessing  = ParamUtils.shouldRun('preprocessing', params.start, effective_stop)
    boolean run_registration   = ParamUtils.shouldRun('registration', params.start, effective_stop)
    boolean run_segmentation   = ParamUtils.shouldRun('segmentation', params.start, effective_stop)
    boolean run_postprocessing = ParamUtils.shouldRun('postprocessing', params.start, effective_stop)

    // MERGE_AND_PYRAMID's memory uses an unmeasured compression ratio
    // (conf/modules.config). Gated on run_postprocessing; add_cycle reaches the same
    // process without --start/--stop.
    if (run_postprocessing) {
        log.warn "MERGE_AND_PYRAMID memory is estimated from an unmeasured ratio. " +
                 "On slides larger than ~40 GB, measure r = (H*W*2)/file_size on your " +
                 "own data and set --max_memory accordingly (see docs/resources.md)."
    }

    // Compartment flags resolved once and passed down as an argument. Nothing below may
    // read the three params directly (tests/test_compartment_mode_routing.py).
    def compartment_mode = ParamUtils.compartmentMode(params)

    if (run_postprocessing) {
        ParamUtils.validateCompartmentQuant(compartment_mode)
    }

    if (!params.input) {
        error "Please provide --input for start '${params.start}'"
    }
    CsvUtils.validateInputCSV(
        params.input,
        ParamUtils.requiredColumnsForStep(params.start)
    )
    warnUnknownColumns(params.input, params.start)

    // Fail-fast semantic validation (row format, reference counts, file existence), also
    // run by --dry_run. false: on the linear path every patient must declare its
    // reference. Last argument: when REGISTER will run with a backend that pairs outputs
    // by channel signature, refuse two slides of one patient with the same channel set.
    CsvUtils.validateInputSemantics(
        params.input,
        params.start,
        false,
        params.nuclear_markers,
        ParamUtils.shouldRun('registration', params.start, effective_stop) &&
            RegBackends.of(params.registration_method).pairsOutputsBySignature
    )

    if (params.dry_run) {
        log.info "DRY RUN: all validations passed (start=${params.start}, stop=${effective_stop})"
        return
    }

    /* -------------------- INPUT -------------------- */

    // INPUT_CHECK reads the samplesheet once. --start picks the column holding the image
    // to carry forward. At --start postprocessing, READ_SEGMENTED_CHECKPOINT also reads
    // the mask/contour columns that [meta, one_file] cannot carry.
    def entry_column = ParamUtils.entryColumnForStep(params.start)

    // INPUT_CHECK resolves the reference once into meta.is_reference.
    INPUT_CHECK(params.input, entry_column)

    /* -------------------- PREPROCESSING -------------------- */

    if (run_preprocessing) {
        PREPROCESSING(INPUT_CHECK.out.samples)
    }

    /* -------------------- REGISTRATION -------------------- */

    if (run_registration) {
        REGISTRATION(
            ParamUtils.isEntryPoint(params, 'registration')
                ? INPUT_CHECK.out.samples
                : PREPROCESSING.out.preprocessed  // Direct channel - enables patient-level parallelism!
        )
    }

    /* -------------------- SEGMENTATION -------------------- */

    if (run_segmentation) {
        SEGMENTATION(
            ParamUtils.isEntryPoint(params, 'segmentation')
                ? INPUT_CHECK.out.samples
                : REGISTRATION.out.registered,  // Direct channel - enables patient-level parallelism!
            compartment_mode
        )
    }

    /* -------------------- POSTPROCESSING -------------------- */

    // Versions/size logs from READ_SEGMENTED_CHECKPOINT (--start postprocessing only);
    // declared here so FINAL QC can read them. Empty at every other entry point.
    def ch_seg_reader_versions  = Channel.empty()
    def ch_seg_reader_size_logs = Channel.empty()

    if (run_postprocessing) {

        // Registration QC for the SpatialData export exists only if REGISTRATION ran in
        // this session (entry could be 'segmentation').
        def ch_reg_qc_for_post        = run_registration ? REGISTRATION.out.seg_qc        : Channel.empty()
        def ch_reg_residuals_for_post = run_registration ? REGISTRATION.out.seg_residuals : Channel.empty()

        def ch_for_postprocessing
        def ch_cell_mask_for_post
        def ch_nuclei_mask_for_post
        def ch_contours_for_post
        def ch_nucleus_contours_for_post
        def ch_morphology_for_post

        if (ParamUtils.isEntryPoint(params, 'postprocessing')) {
            // segmented.csv carries mask/contour columns beyond one image path.
            READ_SEGMENTED_CHECKPOINT(params.input, compartment_mode)
            ch_for_postprocessing       = READ_SEGMENTED_CHECKPOINT.out.samples
            ch_cell_mask_for_post       = READ_SEGMENTED_CHECKPOINT.out.cell_mask
            ch_nuclei_mask_for_post     = READ_SEGMENTED_CHECKPOINT.out.nuclei_mask
            ch_contours_for_post        = READ_SEGMENTED_CHECKPOINT.out.contours
            ch_nucleus_contours_for_post = READ_SEGMENTED_CHECKPOINT.out.nucleus_contours
            ch_morphology_for_post      = READ_SEGMENTED_CHECKPOINT.out.morphology
            ch_seg_reader_versions      = READ_SEGMENTED_CHECKPOINT.out.versions
            ch_seg_reader_size_logs     = READ_SEGMENTED_CHECKPOINT.out.size_logs
        } else {
            // postprocessing is not the entry, so segmentation ran this session
            // (the only other way to reach postprocessing in the linear gate).
            ch_for_postprocessing = ParamUtils.isEntryPoint(params, 'segmentation')
                ? INPUT_CHECK.out.samples
                : REGISTRATION.out.registered  // Direct channel - enables patient-level parallelism!
            ch_cell_mask_for_post        = SEGMENTATION.out.cell_mask
            ch_nuclei_mask_for_post      = SEGMENTATION.out.nuclei_mask
            ch_contours_for_post         = SEGMENTATION.out.contours
            ch_nucleus_contours_for_post = SEGMENTATION.out.nucleus_contours
            ch_morphology_for_post       = SEGMENTATION.out.morphology
        }

        POSTPROCESSING(
            ch_for_postprocessing,
            ch_cell_mask_for_post,
            ch_nuclei_mask_for_post,
            ch_contours_for_post,
            ch_nucleus_contours_for_post,
            ch_morphology_for_post,
            ch_reg_qc_for_post,
            ch_reg_residuals_for_post,
            compartment_mode
        )
    }

    /* -------------------- FINAL QC + TRACE -------------------- */

    // One tagged stream of everything the run produced for reporting. FINAL_QC owns the
    // report and trace gates; steps that did not run contribute nothing. INPUT_CHECK
    // always runs (PREFLIGHT_SCALE), so its artifacts are always included.
    def ch_qc_artifacts = Channel.empty()
        .mix(INPUT_CHECK.out.versions.map  { f -> ['versions', f] })
        .mix(INPUT_CHECK.out.size_logs.map { f -> ['size_log', f] })

    if (run_preprocessing) {
        ch_qc_artifacts = ch_qc_artifacts
            .mix(PREPROCESSING.out.preprocess_qc.map { f -> ['preprocess_qc', f] })
            .mix(PREPROCESSING.out.versions.map      { f -> ['versions', f] })
            .mix(PREPROCESSING.out.size_logs.map     { f -> ['size_log', f] })
    }
    if (run_registration) {
        ch_qc_artifacts = ch_qc_artifacts
            .mix(REGISTRATION.out.qc.map               { _meta, files -> ['registration_qc', files] })
            .mix(REGISTRATION.out.seg_qc.map           { _meta, files -> ['seg_qc', files] })
            .mix(REGISTRATION.out.seg_residuals.map    { _meta, files -> ['seg_residuals', files] })
            .mix(REGISTRATION.out.registration_tre.map { f -> ['registration_tre', f] })
            .mix(REGISTRATION.out.versions.map         { f -> ['versions', f] })
            .mix(REGISTRATION.out.size_logs.map        { f -> ['size_log', f] })
    }
    if (run_segmentation) {
        // Segmentation has no QC-image kind, only versions and size logs.
        ch_qc_artifacts = ch_qc_artifacts
            .mix(SEGMENTATION.out.versions.map  { f -> ['versions', f] })
            .mix(SEGMENTATION.out.size_logs.map { f -> ['size_log', f] })
    }
    if (run_postprocessing) {
        ch_qc_artifacts = ch_qc_artifacts
            .mix(POSTPROCESSING.out.postprocess_qc.map { f -> ['postprocess_qc', f] })
            .mix(POSTPROCESSING.out.versions.map       { f -> ['versions', f] })
            .mix(POSTPROCESSING.out.size_logs.map      { f -> ['size_log', f] })
            // From READ_SEGMENTED_CHECKPOINT at --start postprocessing; empty otherwise.
            .mix(ch_seg_reader_versions.map  { f -> ['versions', f] })
            .mix(ch_seg_reader_size_logs.map { f -> ['size_log', f] })
    }

    FINAL_QC(
        ch_qc_artifacts,
        INPUT_CHECK.out.counts.map { counts -> counts + [stop: effective_stop] }
    )
}

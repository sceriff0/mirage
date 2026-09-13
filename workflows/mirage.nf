/*
================================================================================
    MIRAGE WSI Processing Pipeline — Main Workflow
================================================================================
    This file has ONE job: validate the launch parameters and route the run to
    the right subworkflow. Anything that shapes a channel, assembles a
    subworkflow's inputs, or aggregates its outputs belongs to that subworkflow:

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
include { ADD_CYCLE           } from '../subworkflows/local/add_cycle'
include { FINAL_QC            } from '../subworkflows/local/final_qc'

// Unknown columns are ACCEPTED -- a checkpoint CSV legitimately carries columns
// the entry step does not read -- but never silently: a mistyped 'channles'
// beside a missing 'channels' otherwise surfaces much later, as a null,
// somewhere else. CsvUtils.unknownColumns derives the known set from
// ParamUtils.STEPS + Checkpoint.STEPS, so a legitimate checkpoint header warns
// about nothing.
//
// ONE implementation, called from BOTH the add_cycle branch (step is always the
// literal 'preprocessing' there -- add_cycle has no --start/--stop choice) and
// the linear start/stop path (step is params.start) below, rather than two
// copies of the same log.warn. Defined at SCRIPT level, not inside workflow
// MIRAGE{} or in lib/: `log` is unbound inside lib/*.groovy and inside
// conf/*.config (see CLAUDE.md), but resolves in a workflow script -- which is
// why this lives here, where both call sites can see it, and nowhere else.
def warnUnknownColumns(String csv, String step) {
    def unknown_columns = CsvUtils.unknownColumns(csv, step)
    if (unknown_columns)
        log.warn "Samplesheet column(s) this pipeline does not read, and will ignore: " +
                 "${unknown_columns.join(', ')} (in ${csv})"
}

workflow MIRAGE {

    /* -------------------- PARAMETER VALIDATION -------------------- */

    // Types, enums and ranges come from nextflow_schema.json (nf-schema). This
    // covers every parameter in one place, including the ones the hand-rolled
    // Groovy never checked, and it rejects a -params-file that delivers a
    // boolean as the STRING "false" instead of silently treating it as true.
    // Cross-parameter and filesystem rules that no JSON Schema can express
    // (--stop ordering, samplesheet semantics, add_cycle prerequisites) stay
    // below in Groovy.
    validateParameters()

    ParamUtils.validateOutdir(params.outdir)
    // Tier vs per-knob-override consistency for BOTH registration backends. Cross-parameter,
    // so it belongs here rather than in the schema; runs before any process is instantiated.
    ParamUtils.validatePixelSize(params)
    ParamUtils.validateRegPresets(params)
    // --cleanup_level against --mode. add_cycle must PRODUCE a re-enterable tree, so a
    // cleaning level is refused outright rather than discovered by the next cycle.
    ParamUtils.validateCleanup(params)
    // Settings nextflow.config derives from params as SCALARS (cleanup, trace.*,
    // executor.queueSize, process.maxForks) froze when that file was parsed -- before
    // any `-c` file was merged. Refuse a run whose final params disagree with them,
    // naming the route, rather than let a site.config pin be silently ignored. The
    // command line is passed so Nextflow's own -with-trace/-report/-timeline overrides
    // (nf-test passes -with-trace on every run) are not mistaken for a frozen scalar.
    ParamUtils.validateFrozenConfig(params, workflow.session.config, workflow.commandLine)

    // cleanup_work is mutually exclusive with -resume: the work directory's task files
    // are removed at teardown of a SUCCESSFUL run, so the NEXT -resume finds nothing
    // cached and re-runs everything. It defaults to true now, so this warns rather than
    // refuses -- which of the two the user meant is genuinely ambiguous, the cost is a
    // re-run rather than lost data, and refusing would break every documented
    // `--start <step> ... -resume` invocation in docs/usage.md.
    //
    // `log` IS bound here. It is NOT bound in conf/*.config, where the same call aborts
    // the run under the v1 config parser -- see conf/modules.config's errorStrategy
    // comments and tests/test_error_strategy_policy.py.
    if (params.cleanup_work && workflow.resume) {
        log.warn "--cleanup_work is true (the default) AND -resume was passed. The work " +
                 "directory is emptied after a successful run, so the NEXT -resume will " +
                 "find nothing cached and re-run every task. Set cleanup_work to false in " +
                 "a -params-file or a profile (Nextflow 26 rejects a boolean value on the " +
                 "command line) for the iterate-with--resume loop docs/usage.md describes."
    }

    // --start past preprocessing re-enters from artifacts a cleaning level does not
    // publish. THIS run reads the prior output fine -- it is a different run's tree --
    // so it is a warning rather than a refusal; what it cannot do is be re-entered the
    // same way itself.
    if (params.cleanup_level != 'none' && !ParamUtils.isEntryPoint(params, 'preprocessing')) {
        log.warn "--start ${params.start} re-enters from intermediates that " +
                 "--cleanup_level=${params.cleanup_level} does not publish. This run reads " +
                 "the prior output fine, but its OWN output cannot be re-entered the same " +
                 "way. Pass --cleanup_level none to keep that open."
    }

    /* -------------------- STEP GATE -------------------- */

    // Validate and resolve --stop: default to last step if not provided. Computed
    // HERE, before the mode branch, so add_cycle shares it with the standard path
    // instead of bypassing it: add_cycle used to fall straight into its own
    // validation block and never look at --start/--stop at all, so a contradictory
    // pair (e.g. --start postprocessing --stop preprocessing) silently passed
    // --dry_run instead of erroring the way the standard path always has.
    if (params.stop) {
        ParamUtils.validateStop(params.stop, params.start)
    }
    def effective_stop = params.stop ?: ParamUtils.STEP_ORDER.last()

    // Which steps this run covers. Computed once: the gate is a pure function of
    // (start, stop), so re-asking it at every site only creates opportunities for
    // the three arguments to disagree. (A closure would read better still, but the
    // strict Nextflow parser cannot invoke a closure-typed local as a function.)
    // add_cycle does not consume these booleans below -- it has its own fixed
    // recompute-registration-and-quantification flow, gated by validateAddCycle's
    // own prerequisites, not by --start/--stop -- but it shares the validation
    // above them, which is the point of moving this block up.
    boolean run_preprocessing  = ParamUtils.shouldRun('preprocessing', params.start, effective_stop)
    boolean run_registration   = ParamUtils.shouldRun('registration', params.start, effective_stop)
    boolean run_segmentation   = ParamUtils.shouldRun('segmentation', params.start, effective_stop)
    boolean run_postprocessing = ParamUtils.shouldRun('postprocessing', params.start, effective_stop)

    // MERGE_AND_PYRAMID's memory model uses an UNMEASURED bytes-per-file-byte
    // ratio (conf/modules.config, the `plane * 3.25d` term). On real slides it
    // has been observed to under-reserve into a hard node-memory cliff. The
    // retry ramp is the only safety net, and it is x4 at most. Gated on
    // run_postprocessing -- the real "does this run reach MERGE_AND_PYRAMID"
    // condition (ParamUtils.shouldRun against the STEPS table above), not a
    // dedicated skip param, because add_cycle mode reaches the same process via
    // ASSEMBLE_EXPORT without ever setting --start/--stop.
    if (run_postprocessing) {
        log.warn "MERGE_AND_PYRAMID memory is estimated from an unmeasured ratio. " +
                 "On slides larger than ~40 GB, measure r = (H*W*2)/file_size on your " +
                 "own data and set --max_memory accordingly (see docs/resources.md)."
    }

    // --quantify_compartments / --expanded_quantification / --embed_masks, resolved
    // ONCE here (the single decision site on every path, standard and add_cycle
    // alike) and threaded down as an argument -- the same seam
    // --registration_method has in subworkflows/local/registration.nf. Nothing
    // below this line should read params.quantify_compartments /
    // params.expanded_quantification / params.embed_masks directly;
    // tests/test_compartment_mode_routing.py enforces that.
    def compartment_mode = ParamUtils.compartmentMode(params)

    /* -------------------- MODE: ADD_CYCLE -------------------- */
    if (params.mode == 'add_cycle') {
        // add_cycle has a FIXED path (no --start/--stop choice), so a caller who
        // passes either is rejected here rather than accepted-and-ignored: the
        // earlier behaviour let --stop registration run the ENTIRE path through
        // export while run_summary.json claimed the run stopped after
        // registration — an accuracy bug at the label's source, not the label.
        ParamUtils.validateAddCycleStepFlags(params)
        ParamUtils.validateAddCycle(params.outdir, params.prior_outdir)
        ParamUtils.validateCompartmentQuant(compartment_mode)
        // add_cycle re-registers the new cycle through whichever adapters declare they
        // support that mode — today VALIS alone; add_cycle.nf hands REGISTER_PATIENT the
        // literal 'valis'. Reject anything else loudly rather than registering with VALIS
        // under another method's name.
        //
        // ALLOWLIST, NOT DENYLIST, and now one the backend TABLE owns. This used to name
        // 'tiled' explicitly (so any method the enum gained afterwards passed the check
        // and was silently registered with VALIS), was then narrowed to `!= 'valis'`
        // (correct, but a second place to update when a backend gains add_cycle support),
        // and is now RegBackends.supportsMode — the same field lib_probe asserts and the
        // same table register_patient.nf dispatches from.
        if (!RegBackends.supportsMode(params.registration_method, 'add_cycle')) {
            def supported = RegBackends.methods().findAll {
                RegBackends.supportsMode(it, 'add_cycle')
            }
            error "mode='add_cycle' does not support --registration_method " +
                  "${params.registration_method}; supported: ${supported.join(', ')}."
        }

        if (!params.input) error "mode='add_cycle' requires --input (the new cycle samplesheet)"
        CsvUtils.validateInputCSV(params.input, ParamUtils.requiredColumnsForStep('preprocessing'))
        // add_cycle has no --start/--stop choice, so the step is always the
        // literal 'preprocessing' here -- the same column set INPUT_CHECK reads
        // below via 'path_to_file'.
        warnUnknownColumns(params.input, 'preprocessing')

        // The new-cycle samplesheet intentionally has NO reference row: the
        // registration reference is the frozen prior-run reference (external to
        // this CSV). Pass allow-no-reference=true so validation does not reject
        // the by-design zero-reference sheet. (Registration still uses the prior
        // reference — ADD_CYCLE forces the new cycle to is_reference=false.)
        CsvUtils.validateInputSemantics(params.input, 'preprocessing', true, params.nuclear_markers)

        // Fast-fail: every new-cycle patient must exist in the prior run's
        // postprocessed checkpoint, else its masks/base-table can't be sourced.
        def newPatients   = CsvUtils.countImagesPerPatient(params.input).keySet()
        def priorPostCsv  = Layout.checkpointCsv(params.prior_outdir, Layout.POSTPROCESSED)
        def priorPatients = CsvUtils.countImagesPerPatient(priorPostCsv).keySet()
        def orphans = newPatients - priorPatients
        if (orphans) {
            error "mode='add_cycle': new-cycle patient(s) ${orphans} have no entry in ${priorPostCsv}. " +
                  "Each new-cycle patient_id must match a patient from the prior completed run."
        }

        if (params.dry_run) {
            log.info "DRY RUN (add_cycle): validations passed for --input=${params.input}, --prior_outdir=${params.prior_outdir}; mask extraction will run against ${priorPostCsv}'s pyramid column."
            return
        }

        // add_cycle registers against the PRIOR run's reference, which is never a row
        // in this sheet, so no row here keeps its nuclear channel and the new cycle's
        // markers are the declared channels minus the nuclear one. INPUT_CHECK no
        // longer takes a flag for this: it never promotes a reference on any path.
        INPUT_CHECK(params.input, 'path_to_file')

        // ADD_CYCLE rebuilds the prior run's reusable assets itself, from
        // --prior_outdir's checkpoint CSVs.
        ADD_CYCLE(INPUT_CHECK.out.samples, compartment_mode)

        // ADD_CYCLE has no preprocess_qc / registration_tre / postprocess_qc of its own
        // (it calls PREPROCESSING internally without re-exposing its QC pngs, and has no
        // POSTPROCESSING step at all — masks are reused, not re-segmented). Those kinds
        // are simply not contributed; FINAL_QC defaults them to empty. seg_residuals IS
        // now contributed: ADD_CYCLE captures SEG_QC.out.per_cell (previously dropped).
        FINAL_QC(
            Channel.empty()
                .mix(INPUT_CHECK.out.versions.map    { f -> ['versions', f] })
                .mix(INPUT_CHECK.out.size_logs.map   { f -> ['size_log', f] })
                .mix(ADD_CYCLE.out.qc.map            { _meta, files -> ['registration_qc', files] })
                .mix(ADD_CYCLE.out.seg_qc.map        { _meta, files -> ['seg_qc', files] })
                .mix(ADD_CYCLE.out.seg_residuals.map { _meta, files -> ['seg_residuals', files] })
                .mix(ADD_CYCLE.out.versions.map      { f -> ['versions', f] })
                .mix(ADD_CYCLE.out.size_logs.map     { f -> ['size_log', f] }),
            // ParamUtils.STEP_ORDER.last() ('postprocessing'), NOT effective_stop and NOT
            // the literal 'add_cycle' this used to smuggle in. Neither of those was safe:
            // 'add_cycle' is not a member of STEP_ORDER, so a consumer indexing it would
            // get -1; effective_stop reflects whatever --stop the caller passed, but
            // validateAddCycleStepFlags (above) now REJECTS a non-default --start/--stop
            // in this mode, so the only value that can ever reach here honestly is "ran
            // the whole fixed path through export" — the last step, unconditionally. Before
            // that rejection existed, an accepted-and-ignored --stop registration ran the
            // FULL path through export while still labelling itself "registration" here.
            INPUT_CHECK.out.counts.map { counts -> counts + [stop: ParamUtils.STEP_ORDER.last()] }
        )

        return   // do NOT fall through to the standard start/stop flow
    }

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

    // Fail-fast semantic validation (per-row format + per-patient reference
    // counts + file existence). Runs here so it is also exercised by --dry_run.
    // false: on the linear path a patient MUST declare its reference. There is no
    // auto-promotion any more -- `--allow_auto_reference` and the rule that promoted a
    // patient's first samplesheet row are both gone. Which slide the others are warped
    // onto is not something the pipeline may choose on the operator's behalf, and at
    // every entry point after preprocessing the sheet is a checkpoint this pipeline
    // wrote, so a missing reference there means a corrupt or hand-edited file.
    CsvUtils.validateInputSemantics(
        params.input,
        params.start,
        false,
        params.nuclear_markers
    )

    if (params.dry_run) {
        log.info "DRY RUN: all validations passed (start=${params.start}, stop=${effective_stop})"
        return
    }

    /* -------------------- INPUT -------------------- */

    // INPUT_CHECK reads the samplesheet once here, at the entry step, for every
    // sample/count-derived need downstream (PREPROCESSING/REGISTRATION/SEGMENTATION's
    // inputs, FINAL_QC's manifest). Which column holds the image to carry forward is
    // fixed by --start, because each step's checkpoint CSV names the file that step
    // produced. postprocessing's entry is the one exception: segmented.csv carries
    // four columns INPUT_CHECK's [meta, one_file] shape cannot express, so
    // READ_SEGMENTED_CHECKPOINT (subworkflows/local/segmentation.nf) reads it a
    // second time, below, for the extra mask/contour columns specifically.
    def entry_column = ParamUtils.entryColumnForStep(params.start)

    // INPUT_CHECK resolves the reference from this sheet (CsvUtils.resolveReferenceRows)
    // and stamps it into meta.is_reference, so the decision is made once, here, and every
    // step downstream -- including the checkpoint writers -- carries it rather than
    // re-deriving it. registration.nf used to derive it instead, from arrival order.
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

    // READ_SEGMENTED_CHECKPOINT re-runs EXTRACT_CELL_PROPERTIES (and, under
    // --quantify_compartments, EXTRACT_NUCLEI_PROPERTIES) to recover contours from a
    // reused mask at --start postprocessing -- the one entry point where
    // run_segmentation is false, so ch_qc_artifacts below cannot get their
    // versions/size_log from SEGMENTATION.out. Declared here (not inside the
    // isEntryPoint branch below) so the FINAL QC section can read it regardless;
    // stays Channel.empty() at every other entry point.
    def ch_seg_reader_versions  = Channel.empty()
    def ch_seg_reader_size_logs = Channel.empty()

    if (run_postprocessing) {

        // Registration QC feeds the SpatialData export's `uns`/`obsm`. It only exists
        // when REGISTRATION actually ran IN THIS SESSION. Gated on run_registration
        // itself (not on isEntryPoint('postprocessing')): with the segmentation step
        // now sitting between registration and postprocessing, "postprocessing is not
        // the entry point" no longer implies registration ran this session — entry
        // could be 'segmentation' instead, in which case REGISTRATION was never
        // invoked and referencing REGISTRATION.out would fail outright.
        def ch_reg_qc_for_post        = run_registration ? REGISTRATION.out.seg_qc        : Channel.empty()
        def ch_reg_residuals_for_post = run_registration ? REGISTRATION.out.seg_residuals : Channel.empty()

        def ch_for_postprocessing
        def ch_cell_mask_for_post
        def ch_nuclei_mask_for_post
        def ch_contours_for_post
        def ch_nucleus_contours_for_post
        def ch_morphology_for_post

        if (ParamUtils.isEntryPoint(params, 'postprocessing')) {
            // The ONE place INPUT_CHECK's [meta, one_file] shape is not enough:
            // postprocessing's entry checkpoint (segmented.csv) carries four more
            // columns beyond a single image path. READ_SEGMENTED_CHECKPOINT
            // (subworkflows/local/segmentation.nf) is the dedicated reader —
            // mirrors add_cycle.nf's own splitCsv checkpoint readers.
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

    // One tagged stream of everything the run produced for reporting. FINAL_QC
    // owns both end-of-run aggregations and both of their param gates
    // (--skip_final_qc_report, --enable_trace), so there is nothing to branch on
    // here — steps that did not run simply contribute nothing.
    //
    // INPUT_CHECK runs unconditionally above (every entry point reads the samplesheet
    // through it), and now runs a real process -- PREFLIGHT_SCALE -- so its versions/
    // size_log are contributed here unconditionally too, same as every linear step.
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
        // No dedicated QC-image kind: ParamUtils.STEPS' 'segmentation' entry declares
        // qcKinds: [] on purpose (see its comment) — SEGMENT and the two property
        // extractors only ever contribute the two UNIVERSAL_QC_KINDS.
        ch_qc_artifacts = ch_qc_artifacts
            .mix(SEGMENTATION.out.versions.map  { f -> ['versions', f] })
            .mix(SEGMENTATION.out.size_logs.map { f -> ['size_log', f] })
    }
    if (run_postprocessing) {
        ch_qc_artifacts = ch_qc_artifacts
            .mix(POSTPROCESSING.out.postprocess_qc.map { f -> ['postprocess_qc', f] })
            .mix(POSTPROCESSING.out.versions.map       { f -> ['versions', f] })
            .mix(POSTPROCESSING.out.size_logs.map      { f -> ['size_log', f] })
            // READ_SEGMENTED_CHECKPOINT's re-run EXTRACT_CELL_PROPERTIES (+
            // EXTRACT_NUCLEI_PROPERTIES) versions/size_log, at --start postprocessing
            // only -- Channel.empty() (declared above) at every other entry point.
            .mix(ch_seg_reader_versions.map  { f -> ['versions', f] })
            .mix(ch_seg_reader_size_logs.map { f -> ['size_log', f] })
    }

    FINAL_QC(
        ch_qc_artifacts,
        INPUT_CHECK.out.counts.map { counts -> counts + [stop: effective_stop] }
    )
}

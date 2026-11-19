/*
 * ParamUtils - validation helpers for the pipeline's step-routing parameters.
 *
 * Single-parameter checks (types, enums, ranges) live in nextflow_schema.json and run
 * via validateParameters() in workflows/mirage.nf. This holds what a schema cannot
 * express: cross-parameter rules, mode-conditional rules (add_cycle), filesystem
 * prerequisites, and step-routing helpers.
 */
class ParamUtils {

    /*
     * What a step is. STEP_ORDER, requiredColumnsForStep, entryColumnForStep and
     * knownArtifactKinds() derive from it; the schema's start/stop enums are checked
     * against it (tests/test_step_vocabulary_consistency.py).
     *
     *   name            - the step's canonical identifier; also the value
     *                      --start/--stop accept.
     *   requiredColumns - samplesheet columns CsvUtils must find when this step
     *                      is the run's entry point (its own input, not a
     *                      downstream checkpoint column).
     *   entryColumn     - the checkpoint-CSV column INPUT_CHECK reads when this
     *                      step is the entry point.
     *   qcKinds         - the FINAL_QC artifact tags this step alone contributes
     *                      ('versions'/'size_log' are in UNIVERSAL_QC_KINDS).
     */
    static final List STEPS = [
        [
            name           : 'preprocessing',
            requiredColumns: ['patient_id', 'path_to_file', 'is_reference', 'channels'],
            entryColumn    : 'path_to_file',
            qcKinds        : ['preprocess_qc'],
        ],
        [
            name           : 'registration',
            requiredColumns: ['patient_id', 'preprocessed_image', 'is_reference', 'channels'],
            entryColumn    : 'preprocessed_image',
            qcKinds        : ['registration_qc', 'registration_tre', 'seg_qc', 'seg_residuals'],
        ],
        [
            name           : 'segmentation',
            requiredColumns: ['patient_id', 'registered_image', 'is_reference', 'channels'],
            entryColumn    : 'registered_image',
            // empty: segmentation emits only universal kinds
            qcKinds        : [],
        ],
        [
            name           : 'postprocessing',
            // cell_mask, nuclei_mask and id are required so a plain registered.csv, or a
            // checkpoint predating the id column, fails at validation (and --dry_run)
            // rather than deep inside READ_SEGMENTED_CHECKPOINT.
            requiredColumns: ['patient_id', 'id', 'registered_image', 'is_reference', 'channels', 'cell_mask', 'nuclei_mask'],
            entryColumn    : 'registered_image',
            qcKinds        : ['postprocess_qc'],
        ],
    ]

    // artifact tags every step emits
    static final List UNIVERSAL_QC_KINDS = ['versions', 'size_log']

    /**
     * The complete FINAL_QC artifact vocabulary: every step's own qcKinds plus the
     * two kinds every step emits.
     *
     * Here rather than in final_qc.nf because Nextflow 26's strict parser allows no
     * top-level assignment there, and a script-level `def` is invisible to the file's
     * functions. Returns a fresh list each call. It is the check that mirage.nf's
     * hand-written FINAL_QC tags agree.
     */
    static List knownArtifactKinds() {
        return STEPS.collectMany { it.qcKinds } + UNIVERSAL_QC_KINDS
    }

    static final List STEP_ORDER = STEPS.collect { it.name }

    /**
     * Require a non-blank --outdir.
     *
     * Otherwise everything publishes, successfully, under a literal 'null/' path.
     *
     * @throws IllegalArgumentException when outdir is null, empty or whitespace.
     */
    static void validateOutdir(String outdir) {
        if (!outdir?.trim()) {
            throw new IllegalArgumentException(
                "Please provide --outdir (the pipeline's output directory). " +
                "Without it every published file lands under a literal 'null/' path.")
        }
    }

    /**
     * Ordering-only check: --stop must not name an earlier step than --start.
     * That both are valid step names is asserted by the schema before this runs.
     */
    static void validateStop(String stop, String start) {
        if (STEP_ORDER.indexOf(stop) < STEP_ORDER.indexOf(start)) {
            throw new IllegalArgumentException("--stop '${stop}' cannot come before --start '${start}'. Pipeline order: ${STEP_ORDER.join(' → ')}")
        }
    }

    /**
     * Check whether a given pipeline step should run, based on --start and --stop.
     */
    static boolean shouldRun(String targetStep, String start, String stop) {
        def idx = STEP_ORDER.indexOf(targetStep)
        if (idx == -1) throw new IllegalArgumentException("Unknown step: '${targetStep}'. Valid: ${STEP_ORDER}")
        return idx >= STEP_ORDER.indexOf(start) && idx <= STEP_ORDER.indexOf(stop)
    }

    /**
     * Effective registration-QC depth: 0 = none, 1 = DAPI overlay, 2 = + segmentation
     * overlap. skip_registration_qc=true forces 0; null reg_qc means 2.
     *
     * Call regQcLevelOf (scalars) from a `script:` block: a bare `params` there puts
     * the whole map in the task hash. It is a distinct name, not an overload, because
     * a null argument would dispatch to the Map form and NPE.
     */
    static int regQcLevelOf(def skipRegistrationQc, def regQc) {
        return skipRegistrationQc ? 0 : (regQc == null ? 2 : (regQc as int))
    }

    /** Map form for workflow code; delegates to regQcLevelOf. */
    static int regQcLevel(Map params) {
        return regQcLevelOf(params.skip_registration_qc, params.reg_qc)
    }

    /**
     * --seg_qc_nuclei_dir: reuse another run's reg_qc=2 QC nuclei. Only meaningful at
     * reg_qc=2 (the QC that reads them), and the directory must exist now -- a typo would
     * otherwise surface only after registration, when SEG_QC looks for the first slide.
     * Per-slide presence is checked in subworkflows/local/seg_qc.nf, where the slide names
     * are known.
     */
    static void validateSegQcNucleiDir(Map params) {
        def dir = params.seg_qc_nuclei_dir
        if (!dir) return
        if (regQcLevel(params) != 2)
            throw new IllegalArgumentException(
                "--seg_qc_nuclei_dir is set but reg_qc is ${regQcLevel(params)}: only reg_qc=2 " +
                "reads QC nuclei. Unset it, or run at reg_qc=2.")
        if (!new File(dir.toString()).isDirectory())
            throw new IllegalArgumentException(
                "--seg_qc_nuclei_dir '${dir}' is not a directory. It must be the --outdir of a " +
                "run that segmented the same native slides with the same seg_method.")
    }

    /**
     * Cross-parameter rules for --cleanup_level.
     *
     * The level is re-checked here too, for callers that bypass nf-schema.
     */
    static void validateCleanup(Map params) {
        def level = params.cleanup_level?.toString()
        if (!Layout.CLEANUP_LEVELS.contains(level))
            throw new IllegalArgumentException(
                "--cleanup_level '${level}' is not valid. Valid: " +
                "${Layout.CLEANUP_LEVELS}. 'none' (the default) publishes everything; " +
                "'final' publishes only ${Layout.FINAL_KINDS} plus run-level " +
                "${Layout.SURVIVING_RUN_LEVEL}.")
    }


    /*
     * Params nextflow.config turns into scalars while it is parsed, before any `-c` file
     * is merged, so a `-c` pin changes params.x but not the setting. The CLI,
     * -params-file and profiles work. tests/test_frozen_config_params.py discovers this
     * set from the config.
     */
    static final List<String> FROZEN_CONFIG_PARAMS = [
        'cleanup_work', 'enable_trace', 'trace_dir', 'concurrency', 'max_forks', 'queue_size',
    ].asImmutable()

    /*
     * Per-process maxForks caps (`Math.min(cap, derivedMaxForks)` in nextflow.config),
     * restated so validateFrozenConfig can recompute them; config cannot see lib/.
     * tests/test_concurrency_params.py checks they match.
     */
    static final Map<String, Integer> PER_PROCESS_MAX_FORKS_CAP = [
        REGISTER: 10, TILED_REG_TILE: 20, TILED_COARSE: 20, TILED_STITCH: 10,
    ].asImmutable()

    /** process.maxForks as nextflow.config derives it: max_forks if set, else concurrency. */
    static int derivedMaxForks(Map params) {
        return (params.max_forks != null ? params.max_forks : params.concurrency) as int
    }

    /** executor.queueSize as nextflow.config derives it: queue_size if set, else concurrency * 4. */
    static int derivedQueueSize(Map params) {
        return (params.queue_size != null ? params.queue_size : (params.concurrency as int) * 4) as int
    }

    /**
     * Refuse a run whose final params disagree with the scalars nextflow.config froze
     * from them.
     *
     * `config` is workflow.session.config. Each expected value is recomputed from params
     * as the config line does, so a mismatch means the param came from a `-c` file.
     *
     * `commandLine` is workflow.commandLine: a scope set by Nextflow's own
     * -with-trace/-with-report/-with-timeline is exempt (nf-test passes -with-trace).
     */
    static void validateFrozenConfig(Map params, Map config, String commandLine = '') {
        def mismatches = []
        def check = { String param, String key, Object expected, Object actual ->
            if (expected != actual) {
                mismatches << "${key} = ${actual} but ${param} resolves to ${expected}".toString()
            }
        }

        check('cleanup_work', 'cleanup', params.cleanup_work as boolean, config.cleanup as boolean)
        [trace: 'trace.txt', report: 'report.html', timeline: 'timeline.html'].each { String scope, String file ->
            if ((commandLine ?: '').contains("-with-${scope}")) {
                return   // overridden on the command line by Nextflow's own flag; not frozen
            }
            check('enable_trace', "${scope}.enabled".toString(),
                  params.enable_trace as boolean, config[scope]?.enabled as boolean)
            check('trace_dir', "${scope}.file".toString(),
                  "${params.trace_dir}/${file}".toString(), config[scope]?.file?.toString())
        }
        check('concurrency/queue_size', 'executor.queueSize', derivedQueueSize(params), config.executor?.queueSize as Integer)
        check('concurrency/max_forks', 'process.maxForks', derivedMaxForks(params), config.process?.maxForks as Integer)
        PER_PROCESS_MAX_FORKS_CAP.each { String name, Integer cap ->
            check('concurrency/max_forks', "process.withName:${name}.maxForks".toString(),
                  Math.min(cap, derivedMaxForks(params)), config.process?."withName:${name}"?.maxForks as Integer)
        }

        if (mismatches) {
            throw new IllegalArgumentException(
                "validateFrozenConfig: the resolved config disagrees with the final params:\n  - " +
                mismatches.join("\n  - ") + "\n" +
                "These settings are computed inside nextflow.config while it is parsed, which is " +
                "BEFORE any `-c` file is merged, so a `-c site.config` pin of " +
                "${FROZEN_CONFIG_PARAMS} changes the param but not the setting it drives. Pass " +
                "these with -params-file (required for the booleans), on the command line " +
                "(numbers and paths), or in a profile defined in nextflow.config -- not in a -c file.")
        }
    }

    /**
     * Refuse a run whose pixel size cannot possibly be legal.
     *
     * The default 'auto' is resolved from OME metadata by PREFLIGHT_SCALE, which fails
     * on a slide with no scale. This refuses an explicit null/empty value (present but
     * null, which nf-schema's `required` does not catch) and non-positive numbers.
     */
    static void validatePixelSize(Map params) {
        def raw = params.pixel_size
        if (raw == null || raw.toString().trim().isEmpty())
            throw new IllegalArgumentException(
                "--pixel_size was passed as null or empty. It is the micrometres-per-pixel " +
                "every measurement in this run is derived from, and defaults to 'auto'. Pass a " +
                "positive number, or 'auto' to read PhysicalSizeX from each image's own OME " +
                "metadata (which errors if an image carries no usable scale) -- or omit the " +
                "flag entirely to get 'auto'.")

        def text = raw.toString().trim()
        if (text.equalsIgnoreCase('auto')) return

        def value = text.isNumber() ? text.toBigDecimal() : null
        if (value == null || value <= 0)
            throw new IllegalArgumentException(
                "--pixel_size '${text}' is neither a positive number of micrometres per pixel " +
                "nor 'auto'.")
    }

    /**
     * STARE's retired cost tiers, kept only so checkRetiredParams can name the values a
     * stale `reg_tiled_mode` selected. The `high` row equals the nextflow.config
     * defaults (tests/test_stare_has_no_tiers.py).
     */
    static final Map<String, Map<String, Integer>> RETIRED_STARE_TIERS = [
        high  : [reg_tiled_tile: 2048, reg_tiled_halo: 256, reg_tiled_out_tile: 1024, reg_tiled_coarse_max_dim: 1024],
        medium: [reg_tiled_tile: 1024, reg_tiled_halo: 192, reg_tiled_out_tile:  768, reg_tiled_coarse_max_dim: 1024],
        low   : [reg_tiled_tile:  512, reg_tiled_halo: 128, reg_tiled_out_tile:  512, reg_tiled_coarse_max_dim:  512],
    ]

    /**
     * Handle a retired param that still reaches `params` (today only reg_tiled_mode).
     * Must run before validateParameters() (validation.ignoreParams lets it through).
     * Any value but `high` throws with that tier's knob values; `high` returns a
     * warning for the caller to log. Returns null when absent.
     *
     * Reads via containsKey/get so no line references the undeclared param
     * (tests/check_param_consistency.py).
     */
    static String checkRetiredParams(Map params) {
        if (params == null || !params.containsKey('reg_tiled_mode')) return null
        def raw = params.get('reg_tiled_mode')
        def mode = raw == null ? null : raw.toString().trim()
        if (mode == 'high') {
            return "--reg_tiled_mode is RETIRED and ignored: STARE no longer has cost tiers, and " +
                "'high' is exactly its single default configuration (reg_tiled_tile 2048, " +
                "reg_tiled_halo 256, reg_tiled_out_tile 1024, reg_tiled_coarse_max_dim 1024). " +
                "Drop it from your command line, params file or profile."
        }
        def row = RETIRED_STARE_TIERS[mode]
        def instead = row != null
            ? "To keep the former '${mode}' tier, set its four knobs directly: " +
              row.collect { k, v -> "${k} ${v}" }.join(', ') + "."
            : "For a former 'custom' tier, set the knobs you overrode directly " +
              "(reg_tiled_tile, reg_tiled_halo, reg_tiled_out_tile, reg_tiled_coarse_max_dim); " +
              "every knob you leave unset takes the former 'high' value, as custom did."
        throw new IllegalArgumentException(
            "--reg_tiled_mode '${raw}' is no longer supported: STARE's cost tiers were removed " +
            "(2026-11-04) and STARE runs one default configuration, the former 'high' row. " +
            "Refusing rather than silently running the 'high' values. ${instead} " +
            "Then remove reg_tiled_mode from your command line, params file or profile.")
    }

    /**
     * STARE's cross-parameter knob rule: COARSE's thumbnail floor.
     */
    static void validateStareKnobs(Map params) {
        // <= 0 disables decimation, handing the anchor the full-resolution plane (~50
        // warps + FFTs, with a memory request sized from this value). 256 px is the
        // anchor's own sweep resolution.
        if (params.reg_tiled_coarse_max_dim != null &&
            (params.reg_tiled_coarse_max_dim as int) < 256) {
            throw new IllegalArgumentException(
                "--reg_tiled_coarse_max_dim ${params.reg_tiled_coarse_max_dim} is below the " +
                "256 px floor. The COARSE anchor sweeps rotations at 256 px and refines at this " +
                "resolution, so a smaller refine thumbnail is coarser than its own seed, and a " +
                "value of 0 or less disables decimation entirely -- handing it the " +
                "full-resolution plane, which it warps and FFTs ~50 times. TILED_COARSE's " +
                "memory request is derived from this same value, so that would also " +
                "under-reserve the task. Use 256 or higher, or drop the " +
                "override and keep the default (1024).")
        }
    }

    /**
     * Resolve --quantify_compartments / --expanded_quantification / --embed_masks once,
     * in workflows/mirage.nf, and pass the map down to subworkflows. Config and module
     * script: blocks may still read the raw params. tests/test_compartment_mode_routing.py
     * enforces that nothing else does.
     */
    static Map compartmentMode(Map params) {
        return [
            compartments: params.quantify_compartments as boolean,
            expanded    : params.expanded_quantification as boolean,
            embedMasks  : params.embed_masks as boolean,
        ].asImmutable()
    }

    /**
     * Cross-parameter rules for the compartment trio (from compartmentMode):
     *
     *   expanded ⇒ compartments
     *   embedMasks ⇒ compartments && expanded   (otherwise no mask series is written,
     *                                            which add_cycle needs later)
     */
    static void validateCompartmentQuant(Map mode) {
        if (mode.expanded && !mode.compartments) {
            throw new IllegalArgumentException(
                "expanded_quantification requires quantify_compartments to also be true. " +
                "Both are booleans -- set them in a -params-file or a profile, never on " +
                "the command line (Nextflow 26 delivers every --param as a String)."
            )
        }
        if (mode.embedMasks && !(mode.compartments && mode.expanded)) {
            throw new IllegalArgumentException(
                "embed_masks requires both quantify_compartments and " +
                "expanded_quantification to also be true -- without both, the pyramid's " +
                "mask series (Image:1) is never written, and a run advertising " +
                "embed_masks that silently omits it is only discovered later, by whatever " +
                "reads Image:1 back out of the pyramid. All three are booleans -- " +
                "set them in a -params-file or a profile, never on the command line."
            )
        }
    }

    /**
     * The samplesheet columns `step` requires, from the one STEPS table.
     *
     * @param step one of STEP_ORDER
     *
     * Returns the step's requiredColumns list.
     * @throws IllegalArgumentException naming `step` when it is not in STEPS.
     */
    static List requiredColumnsForStep(String step) {
        def entry = STEPS.find { it.name == step }
        if (!entry) {
            throw new IllegalArgumentException("No column requirements defined for step: ${step}")
        }
        return entry.requiredColumns
    }

    /**
     * The checkpoint-CSV column to read when `step` is the run's entry point
     * (--start).
     */
    static String entryColumnForStep(String step) {
        def entry = STEPS.find { it.name == step }
        if (!entry) {
            throw new IllegalArgumentException("No entry column defined for step: ${step}")
        }
        return entry.entryColumn
    }

    /**
     * Whether `step` is this run's --start step, i.e. its input comes from
     * INPUT_CHECK rather than the previous step.
     */
    static boolean isEntryPoint(Map params, String step) {
        return params.start == step
    }

}

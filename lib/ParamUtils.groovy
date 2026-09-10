/*
 * ParamUtils - validation helpers for the pipeline's step-routing parameters.
 *
 * Scope note: single-parameter checks (is this a valid step name? a valid
 * segmentation backend? a reg_qc level in range?) are NOT here. They are stated
 * once in nextflow_schema.json and enforced by nf-schema's validateParameters()
 * at the top of workflows/mirage.nf. What remains below is what no JSON Schema
 * can express: cross-parameter rules (--stop must not precede --start,
 * --expanded_quantification implies --quantify_compartments), filesystem
 * prerequisites, and plain control-flow helpers.
 */
class ParamUtils {

    /*
     * The single table answering "what is a step?". Everything else that used
     * to restate the step vocabulary — STEP_ORDER, requiredColumnsForStep,
     * mirage.nf's entry_column map, knownArtifactKinds() below, and
     * nextflow_schema.json's start/stop enums — is either derived from this or
     * checked against it (the schema can't be generated at build time, so
     * tests/test_step_vocabulary_consistency.py asserts it agrees instead).
     *
     *   name            - the step's canonical identifier; also the value
     *                      --start/--stop accept.
     *   requiredColumns - samplesheet columns CsvUtils must find when this step
     *                      is the run's entry point (its own input, not a
     *                      downstream checkpoint column).
     *   entryColumn     - the checkpoint-CSV column INPUT_CHECK reads when this
     *                      step is the run's entry point (mirage.nf reads the
     *                      sheet exactly once, at --start).
     *   qcKinds         - the artifact-stream tags this step alone contributes
     *                      to FINAL_QC (see final_qc.nf). 'versions' and
     *                      'size_log' are cross-cutting — every step emits them
     *                      — so they are not listed per-step; they are added
     *                      once in UNIVERSAL_QC_KINDS below.
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
            // Empty by design, not an oversight: SEGMENT and the two property
            // extractors emit only 'versions' and 'size_log', which are
            // UNIVERSAL_QC_KINDS below, so they contribute nothing here. No QC
            // image belongs to segmentation alone — GENERATE_POSTPROCESSING_QC
            // consumes the mask AND the merged quantification table, so it stays
            // postprocessing's.
            qcKinds        : [],
        ],
        [
            name           : 'postprocessing',
            // 'cell_mask' and 'nuclei_mask' (beyond the registered-image base four) make
            // a plain csv/registered.csv -- still what every doc's `--start postprocessing`
            // example names, pre-dating the segmentation step -- fail HERE with a
            // clear "Missing required column" error. Without them, CsvUtils.validateInputCSV
            // and validateInputSemantics both pass (registered.csv satisfies the base
            // four), and the run dies much later inside segmentation.nf's
            // READ_SEGMENTED_CHECKPOINT -- which dereferences BOTH file(row.cell_mask) and
            // file(row.nuclei_mask) unconditionally -- with "Argument of 'file' function cannot be
            // null" -- a two-layer validation contract that silently skipped its job.
            //
            // 'id' (RULING R17, lib/Checkpoint.groovy) is here for the identical reason:
            // READ_SEGMENTED_CHECKPOINT builds its meta through Meta.fromCheckpointRow,
            // which throws on a row with no 'id' column -- but only once channel
            // construction actually runs, well past --dry_run. Requiring it here instead
            // makes a checkpoint written before this column existed fail at validation
            // time, with "Missing required column 'id'", visible under --dry_run like
            // every other required column -- not a confusing failure deep inside the
            // reader. (Meta.fromCheckpointRow's own row.containsKey('id') check stays as
            // defense in depth for any future caller that reaches it without going
            // through CsvUtils.validateInputCSV first.)
            requiredColumns: ['patient_id', 'id', 'registered_image', 'is_reference', 'channels', 'cell_mask', 'nuclei_mask'],
            entryColumn    : 'registered_image',
            qcKinds        : ['postprocess_qc'],
        ],
    ]

    // Artifact-stream tags every step contributes, so they belong to no single
    // step's qcKinds. See knownArtifactKinds() below.
    static final List UNIVERSAL_QC_KINDS = ['versions', 'size_log']

    /**
     * The complete FINAL_QC artifact vocabulary: every step's own qcKinds plus the
     * two kinds every step emits.
     *
     * This lives here rather than in final_qc.nf because it is a pure derivation of
     * the table above and because a `.nf` file has nowhere to put it: under
     * Nextflow 26's strict parser a bare top-level assignment is rejected
     * ("Statements cannot be mixed with script declarations"), and the obvious
     * alternative -- a script-level `def` -- is scoped to the file's implicit
     * run() method and is therefore invisible to the functions in that same file.
     * A static accessor has neither problem: it is visible from every call site,
     * from a subworkflow body, and from an nf-test `nextflow_function` test.
     *
     * Returns a fresh list each call, so no caller can mutate the vocabulary for
     * everyone else in the JVM.
     *
     * Both FINAL_QC call sites in workflows/mirage.nf hand-write these tags (21
     * literals across the two calls) and nothing else checks that the two
     * vocabularies agree -- this list is that check. See final_qc.nf's header.
     */
    static List knownArtifactKinds() {
        return STEPS.collectMany { it.qcKinds } + UNIVERSAL_QC_KINDS
    }

    static final List STEP_ORDER = STEPS.collect { it.name }

    /**
     * Require a non-blank --outdir.
     *
     * Without it every publishDir closure interpolates a null and the whole run
     * publishes under a literal 'null/' path -- which succeeds, so nothing else
     * catches it.
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
     * Effective registration-QC depth: 0 = none, 1 = DAPI overlay, 2 = + segmentation overlap.
     * Legacy skip_registration_qc=true forces 0. Defined once and shared by
     * registration.nf / add_cycle.nf so the QC gate has a single source of truth.
     */
    /*
     * WHY regQcLevelOf / microRegLevelOf ARE SCALAR
     *
     * Nextflow hashes the FREE VARIABLES a process's `script:` block references. A block
     * that says `ParamUtils.regQcLevel(params)` therefore hashes `params` -- the WHOLE
     * map, as one opaque entry -- and re-runs whenever ANY parameter anywhere changes.
     * Measured: changing only `--pyramid_resolutions` (postprocessing) re-ran REGISTER and
     * cascaded 20 of 28 tasks. `-dump-hashes` showed every individually-named entry
     * unchanged and only `params=ScriptBinding$ParamsMap@...` differing.
     *
     * A `params.reg_qc` / `params.reg_micro_reg` access is recorded as its OWN hash entry,
     * so passing the scalar keeps the process bound to just the parameters it actually reads.
     *
     * regQcLevel(Map) stays: workflow/subworkflow code (registration.nf, add_cycle.nf) is
     * not hashed this way, reads better with `params`, and there is exactly one
     * implementation behind it -- the scalar form is where the logic lives and the Map
     * form delegates to it, so the two cannot drift. `microRegLevelOf` has no Map
     * counterpart -- every call site is inside a `script:` block (register.nf,
     * warp_seg_qc.nf), so there is no workflow-level caller to justify one. CALL THE
     * `...Of` FORM FROM A `script:` BLOCK; either form elsewhere.
     *
     * `regQcLevelOf` is named rather than being an overload of `regQcLevel(Map)` because
     * `regQcLevel(Map)` and a same-arity `regQcLevel(def)` are ambiguous for a NULL
     * argument -- and null is the expected input here, it is what selects the default.
     * Groovy would pick the more specific Map candidate and NPE on `params.reg_qc`. A
     * distinct name removes the dispatch question entirely.
     */
    static int regQcLevelOf(def skipRegistrationQc, def regQc) {
        return skipRegistrationQc ? 0 : (regQc == null ? 2 : (regQc as int))
    }

    /** Map form -- see the note above. Delegates; holds no logic of its own. */
    static int regQcLevel(Map params) {
        return regQcLevelOf(params.skip_registration_qc, params.reg_qc)
    }

    /**
     * Effective micro-registration depth: 0 = none, 1 = micro-rigid only (refines slide.M),
     * 2 = micro-rigid + micro non-rigid (register_micro). VALIS controls the two passes
     * independently — micro_rigid_registrar_cls for the rigid refinement, the register_micro()
     * call for the non-rigid one — and this ordinal nests them (0 ⊂ 1 ⊂ 2), which forbids the
     * odd "micro non-rigid without micro-rigid" combination. The shipped default is 1
     * (micro-rigid only), declared in nextflow.config -- the single source of truth for it --
     * and restated in nextflow_schema.json; tests/test_micro_reg_default_is_one.py holds every
     * home to that number, this comment included. The `null` fallback below returns 2 and is
     * UNREACHABLE on the pipeline path: the schema declares reg_micro_reg non-nullable with its
     * own default, so validateParameters() has filled it in before any caller reaches here. It
     * is kept as the pre-v1.0.0 behaviour a direct Groovy caller used to get, not as a claim
     * about what ships. Single source of truth for register.nf / warp_seg_qc.nf so the QC can
     * honestly say what the 'rigid' stage means for a given run.
     */
    static int microRegLevelOf(def regMicroReg) {
        return regMicroReg == null ? 2 : (regMicroReg as int)
    }

    /**
     * Cross-parameter rules for --cleanup_level.
     *
     * The per-value enum is the schema's job (nextflow_schema.json); this layer
     * re-checks it for callers that never went through nf-schema. Runs before any
     * process is instantiated, like every other rule in this file.
     *
     * The level is re-checked here rather than trusted from the schema because
     * validateCleanup is also reachable from a caller that never went through
     * nf-schema (a test, or a future entry point), and a silently-unknown level
     * would then read as 'publish nothing'.
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
     * Settings nextflow.config derives from a param as a SCALAR. A scalar freezes at
     * the line it is written on, and nextflow.config is parsed IN FULL before any `-c`
     * file is merged -- so a `-c site.config` pin of one of these changes `params.x`
     * and reaches nothing that reads it. Measured 2026-09-09 (Nextflow 25.04.7): a -c
     * pin of concurrency=7 / cleanup_work=true / enable_trace=false left queueSize 20,
     * every maxForks 5, `cleanup` false and the trace enabled, while `params.*` printed
     * the pinned values. The CLI, a -params-file and a profile all reach them.
     *
     * tests/test_frozen_config_params.py DISCOVERS this set by scanning nextflow.config
     * and asserts it equals this list in both directions, so a new params-derived
     * scalar in the config fails the suite until it is checked here.
     */
    static final List<String> FROZEN_CONFIG_PARAMS = [
        'cleanup_work', 'enable_trace', 'trace_dir', 'concurrency', 'max_forks', 'queue_size',
    ].asImmutable()

    /*
     * The four per-process maxForks caps in nextflow.config's "Concurrency" block, keyed
     * by process name. Each is `Math.min(cap, derivedMaxForks)` there; this table lets
     * validateFrozenConfig recompute what the config SHOULD hold, since conf/*.config
     * cannot see lib/ classes and the config cannot share a helper.
     * tests/test_concurrency_params.py asserts the config literals match this table.
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
     * `config` is the resolved session config (workflow.session.config). Every
     * comparison recomputes the expected value from `params` exactly as the config
     * line does, so the only way they differ is that the param arrived by a route the
     * config line could not see -- a `-c` file. A silent mismatch here is the worst
     * kind: `nextflow config` prints the pinned param, the run honours the old scalar,
     * and for cleanup_work that means deleting (or keeping) work/ against the operator's
     * stated intent. Refusing at launch costs nothing; a wrong `cleanup` is discovered
     * at teardown, after the whole run.
     *
     * `commandLine` is workflow.commandLine. Nextflow's own `-with-trace <file>`,
     * `-with-report <file>` and `-with-timeline <file>` flags legitimately overwrite
     * that scope's `enabled` and `file` AFTER nextflow.config, so a scope named that
     * way on the command line is not checked. nf-test passes `-with-trace` on every
     * run -- measured 2026-09-09: without this exemption 43 of 235 stub tests refused
     * at launch on `trace.file = .../meta/trace.csv`.
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
     * `params.pixel_size` owns every micrometre in the pipeline -- GeoJSON centroids and areas,
     * the published pyramid's PhysicalSize, InstantSeg's rescaling target. It carried a literal
     * 0.325 default, which is one scanner's value standing in for everyone's: a run on any other
     * objective produced measurements uniformly wrong by the ratio of the two, with nothing said,
     * and the symptom surfaced in QuPath several steps and one repository away from the cause.
     *
     * The invariant that incident demands -- no scale is ever guessed -- still holds; only WHERE
     * it is enforced has moved. `nextflow.config` now defaults `pixel_size` to 'auto', which reads
     * the image's own OME metadata rather than inventing a number, and `PREFLIGHT_SCALE` resolves
     * that value for every slide before any heavy work and hard-fails the run the moment a slide
     * carries no usable scale (see bin/utils/pixel_size.py). So this method no longer needs to
     * refuse an unset value on the operator's behalf -- there is no unset value to reach it,
     * because the config supplies one. What it still refuses is null: null can only arrive if
     * someone explicitly passed one on the command line, which can only be a mistake, since it
     * overrides a working default with an unusable one.
     *
     * Cross-parameter/"must be set" layer, not the schema's: nf-schema's `required` fires on an
     * ABSENT key, and a null here is present-but-null. See the two-layer rule in CLAUDE.md.
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
     * Reject per-knob overrides that contradict the cost/accuracy tier they are used with.
     *
     * Both registration backends expose `high | medium | low | custom`. Individual knob overrides
     * are meaningful ONLY under `custom`, which is defined as "start from `high`, apply what the
     * user set". Allowing `--memory_mode low --reg_valis_max_non_rigid_dim 4096` would produce a
     * run that reports a tier it is not actually using, and nothing downstream could tell: the
     * tier name reaches the QC report and the benchmark tables, the resolved numbers do not.
     *
     * Runs before any process is instantiated (workflows/mirage.nf), so a contradictory
     * invocation costs nothing. This is the cross-parameter layer -- the per-value enums and
     * types are the schema's job (nextflow_schema.json), per the two-layer rule in CLAUDE.md.
     */
    static void validateRegPresets(Map params) {
        def offenders = { String modeName, Object mode, Map knobs ->
            if (mode == 'custom') return []
            return knobs.findAll { name, value -> value != null }
                        .collect { name, value -> "--${name} ${value}" }
        }

        def valisBad = offenders('memory_mode', params.memory_mode, [
            reg_valis_max_processed_dim: params.reg_valis_max_processed_dim,
            reg_valis_max_non_rigid_dim: params.reg_valis_max_non_rigid_dim,
        ])
        if (valisBad) {
            throw new IllegalArgumentException(
                "VALIS knob override(s) ${valisBad.join(', ')} were given, but " +
                "--memory_mode is '${params.memory_mode}'. Per-knob overrides only apply under " +
                "--memory_mode custom, which starts from the 'high' preset and keeps every knob " +
                "you do not set. Either pass --memory_mode custom, or drop the override(s) and " +
                "let the '${params.memory_mode}' preset supply them " +
                "(table: bin/utils/valis_config.py, MEMORY_PRESETS).")
        }

        def stareBad = offenders('reg_tiled_mode', params.reg_tiled_mode, [
            reg_tiled_tile          : params.reg_tiled_tile,
            reg_tiled_halo          : params.reg_tiled_halo,
            reg_tiled_upsample      : params.reg_tiled_upsample,
            reg_tiled_out_tile      : params.reg_tiled_out_tile,
            reg_tiled_coarse_max_dim: params.reg_tiled_coarse_max_dim,
        ])
        if (stareBad) {
            throw new IllegalArgumentException(
                "STARE knob override(s) ${stareBad.join(', ')} were given, but " +
                "--reg_tiled_mode is '${params.reg_tiled_mode}'. Per-knob overrides only apply " +
                "under --reg_tiled_mode custom, which starts from the 'high' preset and keeps " +
                "every knob you do not set. Either pass --reg_tiled_mode custom, or drop the " +
                "override(s) and let the '${params.reg_tiled_mode}' preset supply them " +
                "(table: lib/RegPresets.groovy, RegPresets.STARE).")
        }

        // COARSE's thumbnail bound has a FLOOR, and it is not cosmetic. bin/utils/tiled_io.py's
        // decimation_factor() treats `max_dim <= 0` as "no decimation" and returns factor 1, so
        // the matcher is handed the FULL-RESOLUTION plane -- and the matcher is DISK, a U-Net
        // that allocates activations over the whole plane at ~1.1 + 7.3*Mpx GB. A 26k x 26k
        // slide would need thousands of GB. That escape hatch was survivable when COARSE ran a
        // classical corner detector; it is now a guaranteed OOM.
        //
        // Worse, TILED_COARSE's memory closure DERIVES its request from this same value, so 0
        // computes 0 Mpx and asks for the 4 GB floor -- the smallest request in the table for
        // the largest possible job. A negative value is arithmetically even more perverse: the
        // closure squares it, so -1 yields a positive 0.000001 Mpx and again the floor.
        //
        // The schema's `minimum: 0` blocks negatives on the pipeline path; this is the
        // cross-parameter half that blocks the rest, with a message that says why. 256 px is
        // the floor because it is half of the 'low' tier (512) -- below that the anchor cannot
        // land inside any realistic halo anyway, so there is no legitimate value down there.
        if (params.reg_tiled_coarse_max_dim != null &&
            (params.reg_tiled_coarse_max_dim as int) < 256) {
            throw new IllegalArgumentException(
                "--reg_tiled_coarse_max_dim ${params.reg_tiled_coarse_max_dim} is below the " +
                "256 px floor. COARSE's matcher is DISK, a U-Net whose activation memory is " +
                "linear in thumbnail AREA (~1.1 + 7.3*Mpx GB), and a value of 0 or less " +
                "disables decimation entirely -- handing it the full-resolution plane, which " +
                "needs thousands of GB on a whole slide. TILED_COARSE's memory request is " +
                "derived from this same value, so a too-small bound also under-reserves the " +
                "task rather than merely slowing it. Use 512 (the 'low' tier) or higher, or " +
                "drop the override and let --reg_tiled_mode pick the tier " +
                "(table: lib/RegPresets.groovy, RegPresets.STARE).")
        }
    }

    /**
     * Resolve --quantify_compartments / --expanded_quantification / --embed_masks
     * into ONE immutable snapshot, the same seam --registration_method has
     * (subworkflows/local/registration.nf: read once, threaded down as an
     * argument, never re-read). Call this once per top-level entry point
     * (workflows/mirage.nf) and pass the returned map down to any subworkflow
     * that takes it as a `take:` argument or used to re-derive these booleans
     * itself (subworkflows/local/segmentation.nf's two workflows — SEGMENTATION
     * (`take:`, segmentation.nf:42) and READ_SEGMENTED_CHECKPOINT (`take:`,
     * segmentation.nf:275) — plus postprocess.nf, add_cycle.nf,
     * assemble_export.nf). Config
     * (`conf/modules.config`'s `ext.args`) and module `script:`/`stub:` blocks
     * (e.g. modules/local/quantify.nf) are explicitly out of scope: config cannot
     * take arguments or see this class, and a module reading its own param to
     * build its own flags is this repo's established pattern for both.
     * tests/test_compartment_mode_routing.py enforces that nothing else reads
     * the three raw params directly.
     */
    static Map compartmentMode(Map params) {
        return [
            compartments: params.quantify_compartments as boolean,
            expanded    : params.expanded_quantification as boolean,
            embedMasks  : params.embed_masks as boolean,
        ].asImmutable()
    }

    /**
     * Cross-parameter rules for the compartment-quantification trio, both derived
     * from `mode` (see compartmentMode above) rather than read raw:
     *
     *   expanded ⇒ compartments      (long-standing: per-compartment Mean/Sum
     *                                 needs a per-compartment signal to sum)
     *   embedMasks ⇒ compartments && expanded
     *                                 (assemble_export.nf's embed_masks gate --
     *                                 `params.embed_masks && params.quantify_compartments
     *                                 && params.expanded_quantification` -- decides
     *                                 whether the pyramid gets a mask series (Image:1).
     *                                 --embed_masks true with either sibling off used to
     *                                 exit 0 and silently publish a pyramid with NO mask
     *                                 series; a consumer reading Image:1 back out of it
     *                                 only discovers that much later.)
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
     * Derived rather than restated: STEPS is the single ordered definition of what a
     * step IS, and STEP_ORDER, entryColumnForStep and knownArtifactKinds all come
     * from the same rows.
     *
     * @param step one of STEP_ORDER
     *
     * Returns the step's requiredColumns list.
     * @throws IllegalArgumentException naming `step` when it is not in STEPS -- a
     *         typo surfaces here rather than as an empty requirement set that
     *         validates everything.
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
     * (--start). Replaces mirage.nf's hand-written entry_column map.
     */
    static String entryColumnForStep(String step) {
        def entry = STEPS.find { it.name == step }
        if (!entry) {
            throw new IllegalArgumentException("No entry column defined for step: ${step}")
        }
        return entry.entryColumn
    }

    /**
     * Whether `step` is this run's --start step -- i.e. whether the channel for
     * `step` must come from INPUT_CHECK.out.samples rather than the previous
     * step's direct output. Named helper so mirage.nf's routing reads its intent
     * instead of repeating the raw `params.start == '...'` comparison at every
     * branch point.
     */
    static boolean isEntryPoint(Map params, String step) {
        return params.start == step
    }

}

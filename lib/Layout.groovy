/*
 * Layout - the pipeline's one description of WHERE A FILE LANDS under --outdir.
 *
 * conf/modules.config decides where each process publishes. Everything that needs to
 * know that answer (checkpoint CSVs, the add_cycle reader, validators, onComplete)
 * asks this class instead of restating a path template. When the layout changes,
 * change it in conf/modules.config and here, nowhere else.
 *
 * The publishDir blocks are not routed through here (different evaluation context),
 * so the two must be kept in agreement by hand (tests/test_layout.py checks).
 *
 * Static; takes `outdir` as an argument and never reads params.
 */
class Layout {

    /* ------------------------------------------------------------------ *
     * Checkpoint CSVs
     * ------------------------------------------------------------------ */

    /** Run-level directory (relative to --outdir) holding the checkpoint CSVs. */
    static final String CSV_DIR = 'csv'

    /**
     * The checkpoint step names (CSV basenames): the --start/--stop names minus
     * "-ing". Use the constants, never the strings.
     */
    static final String PREPROCESSED  = 'preprocessed'
    static final String REGISTERED    = 'registered'
    static final String SEGMENTED     = 'segmented'
    static final String POSTPROCESSED = 'postprocessed'

    static final List<String> CHECKPOINT_STEPS =
        [PREPROCESSED, REGISTERED, SEGMENTED, POSTPROCESSED].asImmutable()

    /** The two checkpoints a `mode='add_cycle'` run reads out of --prior_outdir. */
    static final List<String> ADD_CYCLE_CHECKPOINTS =
        [REGISTERED, POSTPROCESSED].asImmutable()

    private static String requireStep(String step) {
        if (!CHECKPOINT_STEPS.contains(step))
            throw new Layout.UnknownStepException(
                "Layout: unknown checkpoint step: '${step}'. Valid: ${CHECKPOINT_STEPS}")
        return step
    }

    /**
     * Thrown for a step not in {@link #CHECKPOINT_STEPS}; distinct from Checkpoint's
     * equivalent error.
     */
    static class UnknownStepException extends IllegalArgumentException {
        UnknownStepException(String message) { super(message) }
    }

    private static String requireOutdir(def outdir) {
        def s = outdir?.toString()
        if (!s?.trim())
            throw new IllegalArgumentException(
                "Layout needs an output directory; got ${outdir == null ? 'null' : "'${outdir}'"}. " +
                "Pass params.outdir (or params.prior_outdir) explicitly.")
        return stripTrailingSlash(s)
    }

    private static String stripTrailingSlash(String s) {
        return s.length() > 1 && s.endsWith('/') ? s[0..-2] : s
    }

    /** `<outdir>/csv` - the collectFile storeDir every checkpoint CSV is written to. */
    static String checkpointDir(def outdir) {
        return "${requireOutdir(outdir)}/${CSV_DIR}"
    }

    /** `<step>.csv` - the collectFile `name:` for a checkpoint. */
    static String checkpointCsvName(String step) {
        return "${requireStep(step)}.csv"
    }

    /** `csv/<step>.csv` - outdir-relative, for messages and existence checks. */
    static String checkpointCsvRelative(String step) {
        return "${CSV_DIR}/${checkpointCsvName(step)}"
    }

    /** `<outdir>/csv/<step>.csv` - the absolute checkpoint path. */
    static String checkpointCsv(def outdir, String step) {
        return "${requireOutdir(outdir)}/${checkpointCsvRelative(step)}"
    }

    /* ------------------------------------------------------------------ *
     * Published directories
     * ------------------------------------------------------------------ */

    /**
     * The closed set of per-patient publish leaves: every `kind` a caller may pass to
     * patientDir()/publishedPath(). tests/test_layout.py checks it against
     * conf/modules.config in both directions.
     *
     * 'spatialdata' is absent: EXPORT_SPATIALDATA publishes to the patient root and
     * its .zarr output already carries `spatialdata/`, so there is no kind to ask for.
     *
     * Ordered as the pipeline produces them.
     */
    static final List<String> PUBLISHED_KINDS = [
        'converted',
        PREPROCESSED,
        REGISTERED,
        'segmentation',
        'cell_properties',
        'split_channels',
        'quantify',
        'quantification',
        'geojson',
        'pyramid',
    ].asImmutable()

    /**
     * The publish leaves that survive a cleaned run.
     *
     * At --cleanup_level=final only these are published at all (gating publishDir is
     * free; publish-then-delete would pay a full copy). SpatialData also survives:
     * its publishDir has no cleanup gate.
     */
    static final List<String> FINAL_KINDS = [
        'quantification',
        'geojson',
        'pyramid',
    ].asImmutable()

    /**
     * Derived, never restated.
     */
    static final List<String> INTERMEDIATE_KINDS =
        (PUBLISHED_KINDS - FINAL_KINDS).asImmutable()

    /**
     * Run-level directories (provenance and reporting) that survive every level.
     */
    static final List<String> SURVIVING_RUN_LEVEL = [
        'qc', CSV_DIR, 'size_logs',
    ].asImmutable()

    /**
     * Valid --cleanup_level values, in increasing aggressiveness.
     *
     * mode='add_cycle' and `--start <step>` re-enter a prior run's intermediates, so
     * they need 'none'; ParamUtils checks this at launch.
     */
    static final List<String> CLEANUP_LEVELS = ['none', 'final'].asImmutable()

    /** Reject an unknown publish kind at the call site rather than at path-resolution time. */
    static String requireKind(String kind) {
        if (!PUBLISHED_KINDS.contains(kind))
            throw new IllegalArgumentException(
                "Unknown published kind: '${kind}'. Valid: ${PUBLISHED_KINDS}")
        return kind
    }

    /**
     * `<outdir>/<patient_id>/<kind>` - the per-patient publish root.
     *
     * `kind` must be one of PUBLISHED_KINDS.
     */
    static String patientDir(def outdir, def patientId, String kind) {
        requireKind(kind)
        if (!patientId?.toString()?.trim())
            throw new IllegalArgumentException("Layout.patientDir: patient_id is required")
        return "${requireOutdir(outdir)}/${patientId}/${kind}"
    }

    /** `<outdir>/<kind>` - a run-level (not per-patient) publish directory. */
    static String runDir(def outdir, String kind) {
        if (!kind?.trim())
            throw new IllegalArgumentException("Layout.runDir: kind is required")
        return "${requireOutdir(outdir)}/${kind}"
    }

    /**
     * `<outdir>/<patient_id>/<kind>/[<producer subdir>/]<basename>` - where `file`
     * will be published.
     *
     * A process that writes into a named subdirectory of its task directory and
     * publishes with a `pattern:` that names that subdirectory gets the subdirectory
     * carried into the published path. conf/modules.config does this in three places:
     *
     *   REGISTER (VALIS)  pattern 'registered_slides/*_registered.ome.tiff'
     *                     -> <outdir>/<pid>/registered/registered_slides/<name>
     *   TILED_STITCH      pattern 'registered/*_registered.ome.tiff'
     *                     -> <outdir>/<pid>/registered/registered/<name>
     *   EXPORT_GEOJSON    output path("export/cells.geojson"), published bare
     *                     -> <outdir>/<pid>/geojson/export/cells.geojson
     *
     * Preserving the producer subdirectory is the only behaviour, so no caller can
     * record a basename-only path that does not exist.
     */
    static String publishedPath(def outdir, def patientId, String kind, def file) {
        def sub = producerSubdir(file)
        def rel = sub ? "${sub}/${basename(file)}" : basename(file)
        return "${patientDir(outdir, patientId, kind)}/${rel}"
    }

    /**
     * Like {@link #publishedPath}, but tolerant of a `file` that is ALREADY an
     * absolute path to a previously published artifact rather than a fresh output in
     * this run's own task directory -- e.g. a mask read back from an earlier
     * checkpoint CSV via `--start postprocessing`, which segmentation.nf's
     * READ_SEGMENTED_CHECKPOINT hands to POSTPROCESSING as a plain `file(row.cell_mask)`,
     * or a registered image read back via `--start segmentation`'s INPUT_CHECK.
     *
     * publishedPath would mis-record such a file (double-nesting the kind, or using
     * this run's outdir). A file whose parent or grandparent is a task directory is a
     * fresh output and goes through publishedPath; anything else is returned as-is.
     *
     * Only one level of producer subdirectory is detected. A deeper fresh output would
     * be recorded as its work-directory path; widen isUnderTaskDir before routing one here.
     */
    static String publishedOrAsIs(def outdir, def patientId, String kind, def file) {
        def path = resolve(file)
        if (path == null)
            throw new IllegalArgumentException("Layout.publishedOrAsIs: no file given")
        return isUnderTaskDir(path.parent)
            ? publishedPath(outdir, patientId, kind, path)
            : path.toString()
    }

    /**
     * Where a slide that was not registered is published (a single-slide patient, or
     * the tiled backend's reference). Nothing publishes it under `<pid>/registered/`;
     * it stays where its producer put it: `<pid>/converted/` with skip_preprocessing,
     * else `<pid>/preprocessed/`. A `--start registration` path is recorded as-is.
     */
    static String passthroughPath(def outdir, def patientId, def file, boolean skipPreprocessing) {
        return publishedOrAsIs(outdir, patientId, skipPreprocessing ? 'converted' : PREPROCESSED, file)
    }

    /**
     * The subdirectory a producer emitted `file` into, or '' when it emitted straight
     * into its task directory.
     *
     * Uses the structural isTaskDir test, not a hash-length guess.
     */
    static String producerSubdir(def file) {
        def path = resolve(file)
        def parent = path?.parent
        if (parent == null) return ''
        return isTaskDir(parent) ? '' : (parent.name?.toString() ?: '')
    }

    /**
     * True when `dir` is a Nextflow task directory: `<workDir>/<2 hex>/<30 hex>`.
     *
     * Both halves are checked so a hex-named output subdirectory is not mistaken for
     * one. 30..32 chars tolerates a wider future hash.
     */
    static boolean isTaskDir(def dir) {
        if (dir == null) return false
        def name = dir.name?.toString()
        if (!name || !name.matches(/^[0-9a-f]{30,32}$/)) return false
        def prefix = dir.parent?.name?.toString()
        return prefix != null && prefix.matches(/^[0-9a-f]{2}$/)
    }

    /**
     * True when `dir` is a task directory or one level below one. Checking the
     * grandparent is what tells a fresh `<work>/xx/<hash>/registered_slides/<name>`
     * from an already-published `.../registered/registered_slides/<name>`.
     * Deeper nesting is not detected (see publishedOrAsIs).
     */
    static boolean isUnderTaskDir(def dir) {
        return dir != null && (isTaskDir(dir) || isTaskDir(dir.parent))
    }

    /* ------------------------------------------------------------------ *
     * Internals
     * ------------------------------------------------------------------ */

    /**
     * Nextflow unwraps a single-element output glob to a bare Path but leaves a
     * multi-element one a List; callers hand us whichever they got.
     */
    private static def resolve(def file) {
        return file instanceof List ? (file ? file[0] : null) : file
    }

    private static String basename(def file) {
        def path = resolve(file)
        if (path == null)
            throw new IllegalArgumentException("Layout: no file given")
        return path.name.toString()
    }
}

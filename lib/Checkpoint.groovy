/*
========================================================================================
    Checkpoint — the checkpoint CSV's SCHEMA owner
========================================================================================
    Layout owns where a checkpoint CSV lives; this class owns its columns, header and
    row builder. Static, never reads params (usable from onComplete and lib_probe.nf).

    row() orders values by the declared columns and throws on a missing or unknown key.
    An empty value means "not produced" (e.g. nucleus_contours without
    --quantify_compartments); null is written as an empty field. nuclei_mask is always
    recorded: postprocess.nf joins on it. Values with a comma, quote or newline are
    RFC 4180-quoted (--outdir may contain a comma); readers use splitCsv(header: true).

    Every schema carries `id` after patient_id, and `pixel_size` last, so a reader
    (Meta.fromCheckpointRow) reads identity and scale back instead of re-deriving them;
    a file without those columns is refused with a message to re-run the step.
========================================================================================
*/

class Checkpoint {

    /*
     * The single table answering "what is in a checkpoint CSV?".
     *
     *   name    - the checkpoint's step name; the CSV basename Layout builds paths
     *             from, and a member of Layout.CHECKPOINT_STEPS.
     *   columns - the ordered column list: a published contract with every reader.
     */
    static final List<Map> STEPS = [
        [
            name   : 'preprocessed',
            columns: ['patient_id', 'id', 'preprocessed_image', 'is_reference', 'channels', 'pixel_size'].asImmutable(),
        ],
        [
            name   : 'registered',
            columns: ['patient_id', 'id', 'registered_image', 'is_reference', 'channels', 'pixel_size'].asImmutable(),
        ],
        [
            name   : 'segmented',
            columns: ['patient_id', 'id', 'registered_image', 'is_reference', 'channels',
                      'cell_mask', 'nuclei_mask', 'contours', 'nucleus_contours', 'pixel_size'].asImmutable(),
        ],
        [
            // per-patient rows: the writer records id = patient_id
            name   : 'postprocessed',
            columns: ['patient_id', 'id', 'cell_csv', 'cell_geojson', 'merged_csv', 'cell_mask', 'pyramid', 'pixel_size'].asImmutable(),
        ],
    ].asImmutable()

    /**
     * Whether a checkpoint CSV is written for `step` at `level`.
     *
     * Only at level 'none': at a cleaning level a manifest would name unpublished
     * files. That includes 'postprocessed', whose cell_mask column names the
     * segmentation mask, an intermediate.
     */
    static boolean writesAtLevel(String step, String level) {
        requireStep(step)
        return level == 'none'
    }

    private static Map requireStep(String step) {
        def entry = STEPS.find { it.name == step }
        if (!entry)
            throw new Checkpoint.UnknownStepException(
                "Checkpoint: unknown checkpoint step: '${step}'. Valid: ${STEPS*.name}")
        return entry
    }

    /**
     * Thrown for a step not in {@link #STEPS}; distinct from Layout's equivalent error.
     */
    static class UnknownStepException extends IllegalArgumentException {
        UnknownStepException(String message) { super(message) }
    }

    /**
     * The ordered column list for a checkpoint (immutable).
     */
    static List<String> columns(String step) {
        return requireStep(step).columns
    }

    /**
     * Assert that `step` still declares every column in `cols`.
     *
     * Call at every reader: splitCsv(header: true) reads an undeclared column as null,
     * which surfaces steps later as a path that does not resolve. Reports every missing
     * column; an empty `cols` throws.
     *
     * Throws {@link UnknownStepException} for an unknown step and
     * {@code IllegalStateException} for a missing column.
     */
    static void requireColumns(String step, Collection<String> cols) {
        def declared = columns(step)
        if (!cols)
            throw new IllegalArgumentException(
                "Checkpoint.requireColumns('${step}'): no columns requested. A caller " +
                "that checks nothing reads as covered; name the columns it indexes.")

        def missing = cols.findAll { !(it in declared) }
        if (missing)
            throw new IllegalStateException(
                "Checkpoint.requireColumns('${step}'): column(s) ${missing as List} are " +
                "read from the '${step}' checkpoint but Checkpoint no longer declares " +
                "them. Declared, in order: ${declared}. A reader indexing an undeclared " +
                "column gets an empty field, i.e. a path that does not resolve, several " +
                "steps later.")
    }

    /** The header line — the `seed:` value every writer passes to collectFile(). */
    static String header(String step) {
        return columns(step).join(',')
    }

    /**
     * One CSV row, values ordered by the DECLARED column list.
     *
     * Throws on a missing or unknown key. Values are not validated; null and ''
     * both become an empty field. Values containing a comma, double quote or newline
     * are RFC 4180-quoted.
     */
    static String row(String step, Map values) {
        def cols    = columns(step)
        def missing = cols.findAll { !values.containsKey(it) }
        if (missing)
            throw new IllegalArgumentException(
                "Checkpoint.row('${step}'): missing column(s) ${missing}. " +
                "Required, in order: ${cols}")

        def unknown = values.keySet().findAll { !(it in cols) }
        if (unknown)
            throw new IllegalArgumentException(
                "Checkpoint.row('${step}'): unknown column(s) ${unknown as List}. " +
                "Valid columns: ${cols}")

        return cols.collect { col -> quoteField(values[col]) }.join(',')
    }

    /**
     * One field, RFC 4180-encoded; null becomes the empty field.
     */
    private static String quoteField(def value) {
        if (value == null) return ''
        def s = value.toString()
        return (s.contains(',') || s.contains('"') || s.contains('\n'))
            ? '"' + s.replace('"', '""') + '"'
            : s
    }
}

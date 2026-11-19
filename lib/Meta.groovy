/*
========================================================================================
    Meta — the ONE constructor for every meta map
========================================================================================
    Every meta map is built here, from a samplesheet row or a checkpoint row, so each
    entry point yields the same key set and `--start X` behaves like a full run from X.
    (valis_adapter.nf's `[patient_id: ...]` REGISTER control tuple is the one allowed
    exception; see tests/test_meta_module.py.)

    Static; never reads params.
========================================================================================
*/

class Meta {

    /**
     * Every key a meta map must carry; requireComplete throws on a missing one.
     */
    static final List<String> REQUIRED_KEYS = [
        'patient_id', 'id', 'is_reference', 'channels',
        'keep_channels', 'channels_count', 'images_count',
    ].asImmutable()

    /**
     * Build a meta from a samplesheet row.
     *
     * @param row         the parsed CSV row (a Map; `row[imageColumn]` is this
     *                    step's entry image, e.g. what column
     *                    CsvUtils.parseCsvLine/splitCsv produced)
     * @param imageColumn which column holds this step's entry image
     * @param rowIndex    0-based index of this row within its patient's rows, in
     *                    samplesheet order. Used ONLY to disambiguate ids that
     *                    would otherwise collide -- see identityFor.
     * @param ctx         [keepChannelsBySlide: Map, imagesCount: Map,
     *                     channelsCount: Map, stemCounts: Map], pre-computed once by
     *                     the caller. stemCounts ("patientId::stem" -> row count) is
     *                     optional; absent, every stem is treated as unique.
     */
    static Map fromSamplesheetRow(Map row, String imageColumn, int rowIndex, Map ctx) {
        requirePresentInRow(row, 'patient_id')
        requirePresentInRow(row, imageColumn)

        def patientId = row.patient_id.toString().trim()
        def rawImage  = row[imageColumn].toString().trim()

        def meta = [
            patient_id  : patientId,
            id          : identityFor(patientId, rawImage, rowIndex, ctx),
            is_reference: row.is_reference?.toString()?.toLowerCase() == 'true',
            channels    : splitChannels(row.channels),
        ]
        // keepChannelsBySlide is keyed on meta.id (identityFor), so look up by the id
        // just assigned. identityFor's inputs must match what resolveKeptChannelsPerSlide
        // used, or a row silently inherits another row's keep_channels.
        return finish(meta, meta.id, ctx)
    }

    /**
     * Build a meta from a checkpoint row (a row of `csv/<step>.csv`, read back in
     * at a `--start` entry point). Produces the same key set as fromSamplesheetRow.
     *
     * @param row  the checkpoint CSV row, already split on the checkpoint's
     *             column list (see lib/Checkpoint.groovy, which owns that list)
     * @param step which checkpoint this row came from (a Checkpoint.STEPS name);
     *             the row is validated against that schema.
     * @param ctx  same shape as fromSamplesheetRow's ctx.
     */
    static Map fromCheckpointRow(Map row, String step, Map ctx) {
        if (!step?.toString()?.trim())
            throw new IllegalArgumentException("Meta.fromCheckpointRow: 'step' must not be blank")

        // Schema from Checkpoint, never restated; an unknown step throws.
        def schemaColumns = Checkpoint.columns(step)

        requirePresentInRow(row, 'patient_id')

        // Per-row checks first ('postprocessed' carries neither column), before the
        // file-level id/pixel_size checks below.
        if (schemaColumns.contains('is_reference'))
            requirePresentInRow(row, 'is_reference')
        if (schemaColumns.contains('channels'))
            requirePresentInRow(row, 'channels')

        // A file written before the `id` column existed has no 'id' key at all
        // (splitCsv only creates keys from the file's header); refuse it.
        if (!schemaColumns.contains('id'))
            throw new IllegalStateException(
                "Meta.fromCheckpointRow('${step}'): lib/Checkpoint.groovy's '${step}' schema " +
                "does not declare an 'id' column. Every STEPS entry must carry one (RULING R17) " +
                "-- this is a Checkpoint.groovy bug, not a bad checkpoint file.")
        if (!row.containsKey('id'))
            throw new IllegalStateException(
                "Meta.fromCheckpointRow('${step}'): this '${step}' checkpoint row has no 'id' " +
                "column. This checkpoint predates identity tracking (RULING R17) -- re-run the " +
                "step that WROTE this checkpoint so the regenerated file records identity, " +
                "rather than forcing it to be re-derived from a filename.")
        requirePresentInRow(row, 'id')

        // Same for `pixel_size`. Never fall back to params.pixel_size: on a --start path
        // it may be 'auto' with no image to resolve it from.
        if (!schemaColumns.contains('pixel_size'))
            throw new IllegalStateException(
                "Meta.fromCheckpointRow('${step}'): lib/Checkpoint.groovy's '${step}' schema does " +
                "not declare a 'pixel_size' column. Every STEPS entry must carry one -- this is a " +
                "Checkpoint.groovy bug, not a bad checkpoint file.")
        if (!row.containsKey('pixel_size'))
            throw new IllegalStateException(
                "Meta.fromCheckpointRow('${step}'): this '${step}' checkpoint row has no " +
                "'pixel_size' column. This checkpoint predates scale tracking -- re-run the step " +
                "that WROTE it so the regenerated manifest records the resolved micrometres-per-" +
                "pixel, rather than leaving every downstream measurement to be scaled by a value " +
                "nothing recorded.")
        requirePresentInRow(row, 'pixel_size')

        def meta = [
            patient_id  : row.patient_id.toString().trim(),
            // the id assigned at samplesheet read; never re-derive it from a filename
            id          : row.id.toString().trim(),
            is_reference: row.is_reference?.toString()?.toLowerCase() == 'true',
            channels    : splitChannels(row.channels),
            pixel_size  : row.pixel_size as Double,
        ]
        return finish(meta, meta.id, ctx)
    }

    /**
     * Identity, ASSIGNED at read time -- never derived from a basename alone.
     *
     * A basename stem alone collides in cyclic-IF layouts (cycle1/slide.ome.tiff and
     * cycle2/slide.ome.tiff). The stem is kept when unique within the patient and
     * suffixed with the row index only on a collision; deterministic for a fixed sheet,
     * as resume caching requires.
     *
     * Strips from the first '.', matching Nextflow's `simpleName`
     * ('slide.ome.tiff' -> 'slide').
     */
    static String identityFor(String patientId, String rawImage, int rowIndex, Map ctx) {
        def name = new File(rawImage.toString()).name
        def dot  = name.indexOf('.')
        def stem = dot >= 0 ? name.substring(0, dot) : name
        def base = stem.startsWith(patientId) ? stem : "${patientId}_${stem}"

        def counts = (ctx?.stemCounts ?: [:]) as Map
        def key    = "${patientId}::${stem}".toString()
        def n      = (counts[key] ?: 1) as int
        return n > 1 ? "${base}_${String.format('%03d', rowIndex)}" : base
    }

    // ----------------------------------------------------------------- private

    private static Map finish(Map meta, def slideKey, Map ctx) {
        // keep_channels is keyed on identity (slideKey)
        def perPatient = (ctx?.keepChannelsBySlide ?: [:] ) as Map
        def perSlide   = perPatient[meta.patient_id]
        // Absent vs empty, never `?:` ([] is falsy): an empty keep-set is legitimate
        // (all channels claimed by an earlier slide); only an absent one falls back.
        meta.keep_channels = (perSlide instanceof Map && perSlide.containsKey(slideKey))
            ? perSlide[slideKey]
            : meta.channels

        // channels_count must equal both the distinct channel names and the emitted
        // files (one consumer .unique()s, one does not). It is a per-patient total feeding
        // groupKey(patient_id, channels_count), so it must come from ctx: a wrong count
        // makes the group emit early or hang. No fallback.
        meta.channels_count = requirePerPatientCount(ctx, 'channelsCount', meta.patient_id,
            'CsvUtils.countChannelsPerPatient is the samplesheet-path example')

        // images_count: same rule, feeds groupKey in registration.nf.
        meta.images_count = requirePerPatientCount(ctx, 'imagesCount', meta.patient_id,
            'CsvUtils.countImagesPerPatient')

        requireComplete(meta)
        return meta
    }

    /**
     * ctx[ctxKey] must carry a per-patient entry. containsKey, not `?:`, so a real
     * count of 0 is accepted.
     *
     * @param ctx           the caller's context map
     * @param ctxKey        which per-patient map to read, e.g. 'channelsCount'
     * @param patientId     the patient this meta is being built for
     * @param resolverHint  named in the exception message: where a caller should
     *                      get this count from
     */
    private static int requirePerPatientCount(Map ctx, String ctxKey, String patientId, String resolverHint) {
        def counts = ctx?.get(ctxKey)
        if (!(counts instanceof Map) || !counts.containsKey(patientId))
            throw new IllegalArgumentException(
                "Meta: ctx.${ctxKey} has no entry for patient '${patientId}'. Every caller must " +
                "pre-compute a per-patient ${ctxKey} (${resolverHint}) covering every patient " +
                "before calling Meta. There is no fallback: a silently-wrong count desynchronises " +
                "a streaming groupTuple(size:)/groupKey downstream, which hangs or mis-groups far " +
                "from here.")
        return counts[patientId] as int
    }

    private static List<String> splitChannels(def raw) {
        return (raw ?: '').toString().split('\\|').collect { it.trim() }.findAll { it }
    }

    private static void requirePresentInRow(Map row, String col) {
        if (row == null || !row.containsKey(col) || row[col] == null || row[col].toString().trim() == '')
            throw new IllegalArgumentException(
                "Meta: required column '${col}' is missing or empty. Row: ${row}")
    }

    private static void requireComplete(Map meta) {
        def missing = REQUIRED_KEYS.findAll { !meta.containsKey(it) }
        if (missing)
            throw new IllegalStateException(
                "Meta is incomplete -- missing ${missing}. Every entry point must " +
                "produce the same key set, or --start X diverges from a full run.")
    }
}

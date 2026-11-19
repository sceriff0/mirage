/*
================================================================================
    SUBWORKFLOW: INPUT_CHECK
================================================================================
    Samplesheet in, sample channel out.

    The only place that turns a CSV row into a `[meta, file]` tuple and that
    pre-computes the per-patient image and channel totals injected into each meta
    as `images_count` / `channels_count`, so downstream `groupTuple(by:, size:)`
    can stream.

    Used by every linear entry point and by add_cycle; they differ only in which
    column holds the image. `take:` values arrive as plain Strings, not channels.

    Before `samples` is emitted, PREFLIGHT_SCALE reads the OME metadata of every
    image (no pixels) to resolve/verify params.pixel_size. The `.combine()` below
    makes every row depend on it, so a scale failure stops the run before any
    image is converted.
================================================================================
*/

include { PREFLIGHT_SCALE } from '../../modules/local/preflight_scale'

workflow INPUT_CHECK {
    take:
    samplesheet     // String  : path to the samplesheet CSV (already validated by CsvUtils)
    image_column    // String  : column holding the image this step consumes

    main:

    // Pre-count images and channels per patient for streaming groupTuple.
    // Callers have already validated the sheet.
    def patient_counts = CsvUtils.countImagesPerPatient(samplesheet)
    def channel_counts = CsvUtils.countChannelsPerPatient(samplesheet, image_column, params.nuclear_markers)
    // The per-SLIDE emit-set. channel_counts above is derived from this same resolver,
    // so the group size and the files that arrive cannot disagree.
    def keep_channels_by_slide = CsvUtils.resolveKeptChannelsPerSlide(samplesheet, image_column, params.nuclear_markers)

    // The reference is chosen once, from the samplesheet, so it never depends on task
    // timing and the resolved is_reference reaches csv/preprocessed.csv.
    def reference_image = CsvUtils.resolveReferenceRows(samplesheet, image_column)
    // Declared union: for FINAL_QC's run_summary.json manifest only, not meta.channels_count.
    def declared_channel_counts = CsvUtils.countDeclaredChannelsPerPatient(samplesheet)

    // Meta.identityFor's collision inputs: stem_counts detects two rows with the same stem
    // (cycle1/slide.ome.tiff + cycle2/slide.ome.tiff); row_index_by_key disambiguates them in
    // samplesheet order, never arrival order. Keyed "patientId::rawImageCell".
    // Each value is a list (the key can repeat), consumed by popping in the .map below.
    def stem_counts       = CsvUtils.stemCountsPerPatient(samplesheet, image_column)
    def row_index_by_key  = CsvUtils.rowIndexPerPatient(samplesheet, image_column)

    // Everything Meta.fromSamplesheetRow needs, computed once; it throws on a missing patient.
    def meta_ctx = [
        keepChannelsBySlide: keep_channels_by_slide,
        imagesCount        : patient_counts,
        channelsCount      : channel_counts,
        stemCounts         : stem_counts,
    ]

    ch_samples = Channel
        .fromPath(samplesheet, checkIfExists: true)
        // quote: '"' is required: without it splitCsv keeps the quotes in a quoted path,
        // and file() stages a dangling relative symlink.
        .splitCsv(header: true, quote: '"')
        .map { row ->
            def raw_image  = row[image_column]?.toString()?.trim()
            def patient_id = row.patient_id?.toString()?.trim()
            // Pop the front of this key's list so rows sharing a key each get their own
            // index. Relies on splitCsv emitting in file order, with nothing in between.
            def row_indices = row_index_by_key["${patient_id}::${raw_image}".toString()]
            def row_index   = (row_indices && !row_indices.isEmpty()) ? row_indices.remove(0) : 0

            def meta = Meta.fromSamplesheetRow(row, image_column, row_index, meta_ctx)

            // Use the resolved is_reference. A patient with no reference (add_cycle) keeps all false.
            meta = meta + [is_reference: reference_image[meta.patient_id] != null &&
                                          reference_image[meta.patient_id] == raw_image]

            return tuple(meta, file(row[image_column]))
        }

    // Fail if fewer rows reached the channel than the sheet declares (e.g. a malformed
    // line that splitCsv skips).
    if (patient_counts) {
        def expected_count = patient_counts.values().sum() ?: 0
        ch_samples.count().subscribe { emitted ->
            if (emitted < expected_count) {
                error "INPUT_CHECK(${samplesheet}): ${expected_count - emitted} row(s) did not reach the " +
                      "sample channel (expected ${expected_count} row(s), got ${emitted}). Check the sheet " +
                      "for malformed or unparseable rows."
            }
        }
    }

    // Metadata-only scale scan over every image (bin/preflight_scale.py); cheap, so ungated.
    PREFLIGHT_SCALE(ch_samples.map { meta, image -> image }.collect(sort: true))

    // combine() broadcasts the single report onto every row, so no consumer of `samples`
    // starts before PREFLIGHT_SCALE succeeds. The resolved scale goes into meta.pixel_size
    // for consumers with no image to read it from (params.pixel_size may be 'auto').
    // The report is keyed by the staged path (input_N/<basename>), so match on basename;
    // a missing entry, or same-named slides with different scales, is an error.
    ch_samples = ch_samples
        .combine(PREFLIGHT_SCALE.out.report)
        .map { meta, image, report ->
            def resolved = new groovy.json.JsonSlurper().parseText(report.text)
            def basename = image.getName()
            def matches = resolved.findAll { path, entry -> new File(path).name == basename }
            if (matches.isEmpty())
                throw new IllegalStateException(
                    "PREFLIGHT_SCALE resolved no scale for ${image} (looked for basename " +
                    "'${basename}' among ${resolved.keySet()}). Every staged image must " +
                    "appear in its report; a missing entry means the pre-flight and the " +
                    "sample channel disagree about what is being processed.")
            def scales = matches.values()*.pixel_size.unique()
            if (scales.size() > 1)
                throw new IllegalStateException(
                    "PREFLIGHT_SCALE reported ${scales.size()} different pixel sizes " +
                    "(${scales}) for basename '${basename}', shared by ${matches.keySet()}. " +
                    "Two same-named slides (e.g. two cyclic-IF cycles) resolved to different " +
                    "scales, and basename alone cannot say which belongs to ${image}.")
            tuple(meta + [pixel_size: scales[0]], image)
        }

    emit:
    // [meta, file] — meta carries images_count and channels_count.
    samples = ch_samples
    // The same totals for the QC report's manifest, as a value channel.
    // `channels` sized channels_count (after the nuclear drop); `declared_channels` is the
    // samplesheet's union, which the manifest reports.
    counts  = Channel.value([
        patients: patient_counts,
        channels: channel_counts,
        declared_channels: declared_channel_counts,
    ])
    versions  = PREFLIGHT_SCALE.out.versions
    size_logs = PREFLIGHT_SCALE.out.size_log
}

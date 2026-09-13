/*
 * CsvUtils - helpers for reading the pipeline's input sample sheet.
 *
 * Parses the input CSV (quote-aware), extracts per-sample metadata, and derives
 * the per-patient / per-channel counts that the workflow injects into meta maps
 * so channels can stream through groupTuple without buffering every sample.
 */
class CsvUtils {

    private static List<String> parseCsvLine(String line) {
        def fields = []
        def current = new StringBuilder()
        boolean inQuotes = false
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i)
            if (c == '"' as char) {
                // Handle escaped quotes ("") inside quoted fields
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"' as char) {
                    current.append('"')
                    i++  // skip the second quote
                } else {
                    inQuotes = !inQuotes
                }
            } else if (c == ',' as char && !inQuotes) {
                fields << current.toString().trim()
                current = new StringBuilder()
            } else {
                current.append(c)
            }
        }
        fields << current.toString().trim()
        return fields
    }

    /**
     * Read a CSV's lines, stripping a UTF-8 byte-order mark from the header if
     * present. Excel/Windows commonly save CSVs with a leading BOM (U+FEFF),
     * which otherwise gets glued onto the first header column name
     * ("<BOM>patient_id"), making every column lookup return -1 — silently
     * breaking image/channel counting (streaming groupTuple hangs) and input
     * validation. Stripping it once here protects every reader below.
     */
    private static List<String> readCsvLines(String csvPath) {
        def lines = new File(csvPath).readLines()
        if (lines && lines[0] && ((int) lines[0].charAt(0)) == 0xFEFF)
            lines[0] = lines[0].substring(1)
        return lines
    }

    /**
     * Count images per patient from a CSV file.
     * Returns a Map of patient_id -> count
     */
    static Map<String, Integer> countImagesPerPatient(String csvPath) {
        def file = new File(csvPath)
        if (!file.exists()) return [:]

        def counts = [:].withDefault { 0 }
        def lines = readCsvLines(file.path)
        if (lines.size() < 2) return [:]  // Header only or empty

        def header = parseCsvLine(lines[0])
        def patientIdx = header.findIndexOf { it == 'patient_id' }
        if (patientIdx == -1) return [:]

        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            if (cols.size() > patientIdx) {
                def patientId = cols[patientIdx].trim()
                if (patientId) counts[patientId]++  // ignore blank patient_id cells
            }
        }
        return counts
    }

    /**
     * Count the channels per patient that actually reach quantification.
     *
     * This is the size the postprocessing groupKeys are built from (feeds
     * meta.channels_count via input_check.nf), so it must equal the number of
     * single-channel TIFFs SPLIT_CHANNELS produces for the patient -- not the number
     * of channels the samplesheet declares. The two differ because the
     * emit-set is resolved per slide by resolveKeptChannelsPerSlide, which claims each
     * marker name exactly once per patient: the reference is walked first, then the
     * remaining slides in samplesheet order, and a slide keeps only names nothing has
     * claimed yet. A reference-less sheet declaring `DAPI|KI67|CD20` on one slide still
     * yields THREE markers; a second slide re-declaring `DAPI` adds nothing. Unioning
     * declared channels with no reference awareness (what this did before) over-counted,
     * and an over-counted groupKey never fills.
     *
     * Do NOT point run_summary.json's input manifest at this. The manifest should
     * report what the samplesheet declared, not what survives the nuclear-channel
     * drop -- that consumer is countDeclaredChannelsPerPatient below. Feeding the
     * manifest from THIS function is the exact regression closed in the branch that
     * added it: for add_cycle it silently reported 2 channels for a declared 3-channel
     * cycle. One number, one purpose; see that function's doc for the other half.
     *
     * @param csvPath        path to the samplesheet
     * @param imageColumn    the column holding the image this step consumes; passed
     *                       straight through to resolveReferenceRows so the reference
     *                       used for counting is the same row registration will use
     * @param nuclearMarkers params.nuclear_markers -- required, never defaulted here
     *
     * Returns a Map of patient_id -> channel count.
     */
    /**
     * patient_id -> the value of `imageColumn` on the row that IS that patient's
     * registration reference. THE one place the reference is decided.
     *
     * Rules, in order:
     *   1. the row declaring `is_reference=true` wins;
     *   2. otherwise the patient is absent from the returned map -- it has no
     *      reference, which is legitimate for mode='add_cycle' (whose reference is
     *      the prior run's and never a row in its sheet) and an ERROR anywhere else,
     *      raised by validateInputSemantics with a better message than this could give.
     *
     * THERE IS NO LONGER A RULE THAT INVENTS A REFERENCE. `--allow_auto_reference` used
     * to promote a patient's first samplesheet row when no row declared itself, and it
     * is gone: which slide every other slide is warped onto is the single most
     * consequential choice in a run, and picking it from row order is a guess wearing a
     * decision's clothes. Two sheets differing only in row order produced two different
     * alignments, and nothing downstream recorded that the pipeline had chosen. A
     * missing reference is now an error, always, on every path but add_cycle's.
     *
     * WHY THE DECISION STILL LIVES HERE. The promotion rule this method used to own
     * existed TWICE, resolved from two different orderings of the same data:
     *
     *   countChannelsPerPatient (below)  promoted rows[0]  -- samplesheet order
     *   subworkflows/local/registration.nf  promoted items[0] -- ARRIVAL order
     *
     * The second is a `.groupTuple()` result, so it is whichever slide finished
     * preprocessing first. Two runs of the same data could therefore register against
     * different references. Worse, the first sizes `channels_count`, which sizes the
     * streaming groupKey the whole pipeline's fan-in rests on: when the two copies
     * disagree AND the two slides differ in nuclear-marker content, the group is sized
     * for a slide that is not the reference. Latent rather than observed today only
     * because the test data's two slides both carry DAPI.
     *
     * Resolving at samplesheet-read time also puts the decision UPSTREAM of the first
     * checkpoint writer, which is what lets `is_reference=true` reach
     * `csv/preprocessed.csv`.
     *
     * THIS METHOD RESOLVES; IT DOES NOT VALIDATE. "Exactly one reference per patient"
     * and "no reference is an error unless this is add_cycle" stay in validateInputSemantics,
     * which runs first and has the better messages. Throwing here would break
     * add_cycle, whose zero-reference sheet is by design.
     *
     * @param imageColumn the column holding the image this step consumes -- the same
     *                    `entry_column` INPUT_CHECK carries forward, so both callers
     *                    key the resolution identically. Assumes that value is unique
     *                    per row within a patient: true for every checkpoint CSV
     *                    (paths are distinct) and for any sheet not listing one image
     *                    twice.
     */
    static Map<String, String> resolveReferenceRows(String csvPath, String imageColumn) {
        def file = new File(csvPath)
        if (!file.exists()) return [:]

        def lines = readCsvLines(file.path)
        if (lines.size() < 2) return [:]

        def header = parseCsvLine(lines[0])
        def patientIdx = header.findIndexOf { it == 'patient_id' }
        def imageIdx   = header.findIndexOf { it == imageColumn }
        def refIdx     = header.findIndexOf { it == 'is_reference' }
        if (patientIdx == -1 || imageIdx == -1) return [:]

        // Rows per patient, IN SAMPLESHEET ORDER -- rule 2 depends on that order, so
        // this must stay an ordered accumulation.
        def rowsByPatient = [:].withDefault { [] }
        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            if (cols.size() <= Math.max(patientIdx, imageIdx)) return
            def patientId = cols[patientIdx].trim()
            if (!patientId) return  // ignore blank patient_id cells
            // Lenient parse, matching countChannelsPerPatient below: validateInputSemantics
            // has already rejected malformed values with a better message.
            def isRef = refIdx != -1 && refIdx < cols.size() &&
                        cols[refIdx]?.trim()?.toLowerCase() == 'true'
            rowsByPatient[patientId] << [isRef, cols[imageIdx].trim()]
        }

        def resolved = [:]
        rowsByPatient.each { patientId, rows ->
            def declared = rows.find { it[0] }
            if (declared) { resolved[patientId] = declared[1]; return }
            // else: no reference for this patient -- omitted deliberately (rule 2).
            // Nothing is promoted in its place; see the note above.
        }
        return resolved
    }

    /**
     * patient_id -> channels of the REFERENCE row, from a checkpoint CSV.
     *
     * add_cycle's seed for resolveKeptChannelsPerSlide's `preClaimed`: the prior run's
     * pyramid already contains these markers, so a new cycle re-staining one adds
     * nothing. Read synchronously rather than from ch_prior_ref, because the keep-set
     * has to be known while the workflow is being constructed, not when a channel
     * happens to emit.
     */
    static Map<String, List<String>> referenceChannelsPerPatient(String csvPath) {
        def file = new File(csvPath)
        if (!file.exists()) return [:]

        def lines = readCsvLines(file.path)
        if (lines.size() < 2) return [:]

        def header      = parseCsvLine(lines[0])
        def patientIdx  = header.findIndexOf { it == 'patient_id' }
        def channelsIdx = header.findIndexOf { it == 'channels' }
        def refIdx      = header.findIndexOf { it == 'is_reference' }
        if (patientIdx == -1 || channelsIdx == -1 || refIdx == -1) return [:]

        def result = [:]
        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            if (cols.size() <= Math.max(patientIdx, Math.max(channelsIdx, refIdx))) return
            if (cols[refIdx]?.trim()?.toLowerCase() != 'true') return
            def patientId = cols[patientIdx].trim()
            if (!patientId) return
            result[patientId] = cols[channelsIdx].split('\\|')*.trim().findAll { it }
        }
        return result
    }

    /**
     * Each slide's exact emit-set: patient_id -> (ASSIGNED IDENTITY -> channels).
     *
     * THE INNER KEY IS Meta.identityFor's OUTPUT for that row -- the SAME identity
     * Meta.fromSamplesheetRow assigns as meta.id -- not the raw `<imageColumn>` cell.
     * A basename key (file(...).simpleName) silently weakened "the raw cell is unique
     * per patient" to "the FILENAME is unique per patient": two rows of one patient
     * under different directories (a cyclic-IF cohort with one directory per cycle)
     * then overwrote each other. Keying on the raw cell instead (an earlier fix)
     * closed that specific collision, but the raw cell is not guaranteed unique
     * either -- two rows can share an identical cell (a duplicate row) or both leave
     * it blank -- and either case collided the exact same way: the second pass
     * silently overwrote the first, sometimes with `[]`, and channels_count summed to
     * a number smaller than what SPLIT_CHANNELS actually emits. Keying on the
     * ASSIGNED identity instead inherits identityFor's own collision handling
     * (disambiguated by rowIndex whenever the stem would otherwise collide), so two
     * rows can never collapse into one entry regardless of what their raw cells look
     * like. `stemCounts` is computed once here, from this same samplesheet, via
     * stemCountsPerPatient -- the identical map Meta.fromSamplesheetRow's caller
     * (input_check.nf) computes and passes through ctx -- so the key written here and
     * the key `finish()` looks up by are provably the same function of the same row.
     *
     * THE keep-set rule, and the only place it exists. SPLIT_CHANNELS emits exactly what
     * this returns (via meta.keep_channels); countChannelsPerPatient sizes the
     * postprocessing groupKeys from it. Before this method the same rule lived in THREE
     * places -- MarkerUtils.splitOutputChannels, bin/split_multichannel.py, and
     * SPLIT_CHANNELS' stub block -- and a disagreement between them was a silent
     * groupTuple miscount rather than a crash.
     *
     * A channel is kept iff its upper-cased name has not already been claimed by an
     * earlier slide of the same patient, walking REFERENCE FIRST then samplesheet order.
     * Ordering the reference first is what makes "the reference wins" fall out of the
     * walk instead of needing a special case.
     *
     * NUCLEAR-NESS PLAYS NO PART IN THE DROP DECISION, deliberately, and the reason is
     * DETERMINISM, not a consumer that needs a file count.
     *
     * Claiming every kept name -- nuclear or not -- makes each marker name reach
     * SPLIT_CHANNELS exactly once per patient, and the winner is the reference, else the
     * samplesheet-order-first slide. Two slides sharing a marker used to be deduplicated
     * by ARRIVAL ORDER at a downstream `.unique()`, the exact scheduling-nondeterminism
     * add_cycle.nf warns about in its own dedup: which slide's copy reached
     * merged_quant.csv and the pyramid varied run to run. It also makes channels_count
     * (countChannelsPerPatient, below) EXACT against what actually arrives, because the
     * emitted-FILE count and the DISTINCT-NAME count are then the same number and the
     * sized groupKey is right for either consumer without knowing which it is.
     *
     * BE ACCURATE ABOUT WHY, because an earlier version of this comment was not: it said
     * groupTiffsByPatient "has no `.unique` and needs the FILE count". The FUNCTION has
     * no `.unique`, but BOTH of its callers dedup on [patient_id, marker] immediately
     * upstream of it -- subworkflows/local/postprocess.nf's `.unique { ... [patient_id,
     * marker] }` and subworkflows/local/add_cycle.nf's priority groupTuple on
     * [pid, marker]. So no live consumer needs the file count today, and the
     * under-count/ABORT scenario that claim cited is unreachable. The decision stands on
     * determinism and on one clean invariant; do not restate the unreachable one.
     *
     * `preClaimed` seeds the claimed set per patient. add_cycle passes the prior run's
     * reference channels, so a re-stained DAPI is dropped as redundant while a genuinely
     * new nuclear marker survives.
     *
     * @param nuclearMarkers validated via MarkerUtils.markerList so a malformed
     *        params.nuclear_markers still fails loudly here, even though the keep
     *        decision itself no longer branches on it.
     */
    static Map<String, Map<String, List<String>>> resolveKeptChannelsPerSlide(
            String csvPath, String imageColumn, def nuclearMarkers,
            Map<String, List<String>> preClaimed = [:]) {

        // Validate the parameter shape even though the keep rule does not branch on it: a
        // malformed params.nuclear_markers must not become silently harmless here.
        MarkerUtils.markerList(nuclearMarkers)

        def file = new File(csvPath)
        if (!file.exists()) return [:]

        def lines = readCsvLines(file.path)
        if (lines.size() < 2) return [:]

        def header      = parseCsvLine(lines[0])
        def patientIdx  = header.findIndexOf { it == 'patient_id' }
        def channelsIdx = header.findIndexOf { it == 'channels' }
        def imageIdx    = header.findIndexOf { it == imageColumn }
        if (patientIdx == -1 || channelsIdx == -1 || imageIdx == -1) return [:]

        // WHICH slide is the reference is resolved by resolveReferenceRows, not decided
        // here -- the same reason countChannelsPerPatient asks it rather than promoting
        // rows[0] itself.
        def referenceImage = resolveReferenceRows(csvPath, imageColumn)

        // Meta.identityFor's collision input, computed once from THIS SAME sheet --
        // the identical map input_check.nf computes independently and threads through
        // ctx.stemCounts for Meta.fromSamplesheetRow. Reading it here (rather than
        // accepting it as a parameter) is what makes the key written below and the
        // key `finish()` looks up by provably the same function of the same row,
        // without widening this method's signature or asking every caller to thread
        // it through.
        def stemCounts = stemCountsPerPatient(csvPath, imageColumn)

        // Rows per patient, IN SAMPLESHEET ORDER. The walk order below depends on it,
        // and `idx` (the row's 0-based position within ITS PATIENT, in that same
        // order) is Meta.identityFor's rowIndex argument -- captured here rather than
        // via a second pass/rowIndexPerPatient call, so it is trivially the same
        // order stemCounts was computed against.
        def rowsByPatient = [:].withDefault { [] }
        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            if (cols.size() <= Math.max(patientIdx, Math.max(channelsIdx, imageIdx))) return
            def patientId = cols[patientIdx].trim()
            if (!patientId) return  // ignore blank patient_id cells
            def rawImage = cols[imageIdx].trim()
            rowsByPatient[patientId] << [
                raw     : rawImage,
                channels: cols[channelsIdx].split('\\|')*.trim().findAll { it },
                idx     : rowsByPatient[patientId].size(),
            ]
        }

        def result = [:]
        rowsByPatient.each { patientId, rows ->
            // resolveReferenceRows returns the RAW cell; partitioning below still
            // compares raw cells (that is what resolveReferenceRows returns and what
            // is_reference is decided from), even though the OUTPUT map is no longer
            // keyed on one.
            def refCell = referenceImage[patientId]
            // Stable partition: reference row(s) first, everything else in declared
            // order. A patient with no reference at all (add_cycle's by-design
            // zero-reference sheet) simply walks in samplesheet order, since nothing
            // matches null.
            def ordered = rows.findAll { it.raw == refCell } +
                          rows.findAll { it.raw != refCell }

            def claimed = new HashSet<String>()
            (preClaimed[patientId] ?: []).each { claimed << it.toString().trim().toUpperCase() }

            def perSlide = [:]
            ordered.each { row ->
                def keep = []
                row.channels.each { ch ->
                    def name = ch.toUpperCase()
                    if (claimed.contains(name)) return
                    claimed << name
                    keep << ch
                }
                // Keyed on the ASSIGNED identity, not on the raw cell. Two rows for one
                // patient can share a raw cell (a duplicate row) or both leave it blank,
                // and keying on it meant the second pass overwrote the first -- sometimes
                // with [] -- so channels_count summed to less than what SPLIT_CHANNELS
                // actually emits and the streaming groupTuple(size:) it sizes either
                // emitted early with missing members or hung.
                perSlide[Meta.identityFor(patientId, row.raw, row.idx, [stemCounts: stemCounts])] = keep
            }
            result[patientId] = perSlide
        }
        return result
    }

    /**
     * Count the channels that will actually be EMITTED per patient -- the number a
     * `channels_count`-sized `groupKey` must equal.
     *
     * Derived from resolveKeptChannelsPerSlide, which is the rule SPLIT_CHANNELS
     * applies, so this method and SPLIT_CHANNELS cannot disagree. Summing the
     * per-slide keep-lists is exact in both directions at once: the resolver claims
     * each marker name once per patient, so the sum equals BOTH the number of TIFFs
     * emitted and the number of distinct names. Every consumer is therefore correct
     * without having to know which of the two it was handed.
     *
     * Do NOT substitute countDeclaredChannelsPerPatient. It has no reference and no
     * nuclear-marker awareness, so it over-counts a reference-less sheet by the
     * dropped nuclear channel -- and an over-counted groupKey never fills, so the
     * run hangs rather than failing.
     *
     * Returns an empty map on a missing file, a header-only sheet, or an absent
     * patient_id / channels / <imageColumn> column: every such guard lives in the
     * resolver rather than being repeated here.
     *
     * @param csvPath        path to the samplesheet
     * @param imageColumn    the column naming the step's input image
     * @param nuclearMarkers params.nuclear_markers, in any form MarkerUtils accepts
     *
     * Returns a Map of patient_id -> emitted channel count.
     */
    static Map<String, Integer> countChannelsPerPatient(String csvPath, String imageColumn, def nuclearMarkers) {
        // Every guard the old body carried inline -- missing file, header-only sheet,
        // absent patient_id/channels/<imageColumn> column -- now lives in the resolver,
        // which returns an empty map in each case. Reference resolution likewise: it
        // asks resolveReferenceRows, so this method and SPLIT_CHANNELS cannot disagree
        // about which slide is the reference.
        //
        // Derived from resolveKeptChannelsPerSlide, which IS the rule SPLIT_CHANNELS
        // applies (it emits exactly meta.keep_channels). Summing the per-slide list
        // sizes is safe precisely because that resolver claims each marker name once
        // per patient: the sum therefore equals BOTH the number of TIFFs emitted and the
        // number of distinct names, because those are the same number. Every
        // channels_count-sized groupKey downstream is then correct without its consumer
        // having to know which of the two it is being handed.
        //
        // NOT because some consumer needs the file count: both groupTiffsByPatient
        // callers dedup on [patient_id, marker] immediately upstream of it
        // (subworkflows/local/postprocess.nf's `.unique`, subworkflows/local/add_cycle.nf's
        // priority groupTuple), so the distinct-name count would in fact serve them
        // today. An earlier version of this comment claimed otherwise -- see
        // resolveKeptChannelsPerSlide's doc.
        //
        // This used to union upper-cased names into a HashSet, which gave the
        // distinct-name count but NOT the file count -- correct only while no two slides
        // could emit the same marker.
        def kept = resolveKeptChannelsPerSlide(csvPath, imageColumn, nuclearMarkers)
        return kept.collectEntries { patientId, perSlide ->
            [patientId, (perSlide.values().sum { it.size() } ?: 0)]
        }
    }

    /**
     * Count the DECLARED channels per patient: the union of the samplesheet's
     * `channels` column values, patient-wide -- no reference awareness, no
     * nuclear-marker awareness, no extra arguments. This is deliberately
     * countChannelsPerPatient's exact pre-Task-3 behaviour, kept alive for a
     * different consumer: run_summary.json's input manifest
     * (`manifest.totals.channels` / `manifest.patients[pid].channels`), which should
     * report what the samplesheet SAID, not what reaches QUANTIFY.
     *
     * Do NOT feed this into channels_count / the groupKey size. That reintroduces
     * the exact bug countChannelsPerPatient's exactness fixed: unioning declared
     * channels with no reference awareness over-counts a reference-less sheet by its
     * dropped nuclear channel, and an over-counted groupKey never fills (the run
     * hangs). See countChannelsPerPatient's doc for that consumer's requirements.
     *
     * @param csvPath path to the samplesheet
     *
     * Returns a Map of patient_id -> declared channel count.
     */
    static Map<String, Integer> countDeclaredChannelsPerPatient(String csvPath) {
        def file = new File(csvPath)
        if (!file.exists()) return [:]

        def channelSets = [:].withDefault { new HashSet<String>() }
        def lines = readCsvLines(file.path)
        if (lines.size() < 2) return [:]  // Header only or empty

        def header = parseCsvLine(lines[0])
        def patientIdx = header.findIndexOf { it == 'patient_id' }
        def channelsIdx = header.findIndexOf { it == 'channels' }
        if (patientIdx == -1 || channelsIdx == -1) return [:]

        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            if (cols.size() > Math.max(patientIdx, channelsIdx)) {
                def patientId = cols[patientIdx].trim()
                if (!patientId) return  // ignore blank patient_id cells
                def channels = cols[channelsIdx].split('\\|')*.trim().findAll { it }
                channelSets[patientId].addAll(channels*.toUpperCase())
            }
        }

        return channelSets.collectEntries { k, v -> [k, v.size()] }
    }

    /**
     * How many rows in each patient share a source-image stem. Meta.identityFor
     * consults this to decide whether a stem needs disambiguating -- so an id
     * only changes shape where it would otherwise collide, and every existing
     * non-colliding output filename is preserved byte-for-byte.
     *
     * The stem rule here MUST match Meta.identityFor's exactly -- both key on
     * "patientId::stem" -- or this map's counts land on a different stem than
     * identityFor is asking about and every lookup silently misses (n treated
     * as 1, no disambiguation, the exact collision this exists to catch).
     * identityFor strips EVERY extension (RULING R2, verified against this
     * repo's pinned Nextflow: `file('slide.ome.tiff').simpleName == 'slide'`,
     * not 'slide.ome') -- i.e. from the FIRST '.', not the last -- so this
     * strips the same way rather than a single-extension `.replaceFirst`.
     *
     * @param csvPath     path to the samplesheet
     * @param imageColumn the column holding this step's entry image
     */
    static Map<String, Integer> stemCountsPerPatient(String csvPath, String imageColumn) {
        def file = new File(csvPath)
        if (!file.exists()) return [:]

        def lines = readCsvLines(file.path)
        if (lines.size() < 2) return [:]

        def header = parseCsvLine(lines[0])
        def patientIdx = header.findIndexOf { it == 'patient_id' }
        def imageIdx   = header.findIndexOf { it == imageColumn }
        if (patientIdx == -1 || imageIdx == -1) return [:]

        def counts = [:].withDefault { 0 }
        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            if (cols.size() <= Math.max(patientIdx, imageIdx)) return
            def patientId = cols[patientIdx].trim()
            if (!patientId) return  // ignore blank patient_id cells
            def rawImage = cols[imageIdx].trim()
            if (!rawImage) return
            counts["${patientId}::${stemOf(rawImage)}".toString()]++
        }
        return counts
    }

    /**
     * Each patient's rows' 0-based positions, IN SAMPLESHEET ORDER, grouped by
     * "patientId::rawImageCell" (the same raw `<imageColumn>` cell
     * resolveReferenceRows/resolveKeptChannelsPerSlide key on), so a caller
     * building meta from a `splitCsv` row can look its own index up by content
     * instead of counting channel arrivals itself.
     *
     * RETURNS A LIST PER KEY, NOT A SCALAR -- "patientId::rawImageCell" is NOT
     * guaranteed unique. Two rows of one patient can share an identical raw
     * cell (a duplicate row) or both leave it blank; `validateInputSemantics`
     * does not reject either. A single scalar per key used to collapse under
     * that collision (last write wins), so BOTH rows read back the SAME index,
     * Meta.fromSamplesheetRow assigned them the SAME id, and one of the two
     * then silently displaced the other in ctx.keepChannelsBySlide (looked up
     * by that shared id) -- reproducing, one call site up, exactly the
     * row.raw-keying collapse this same task closes in
     * resolveKeptChannelsPerSlide.
     *
     * THE CALLER MUST CONSUME BY POSITION, NOT BY RE-READING THE VALUE: pop
     * (remove) the FIRST element of the matching key's list on every row it
     * processes, never just peek it. input_check.nf does exactly that. This is
     * safe -- and remains a pure function of the FILE's own row order rather
     * than of channel/task arrival order, which resume caching cannot depend
     * on -- PRECISELY BECAUSE `splitCsv()` reading a single path is a
     * synchronous, single-producer parse that emits in file order
     * deterministically. It is not the multi-producer, completion-order
     * dependent case `groupTuple()` is (see this class's callers for why THAT
     * ordering can't be trusted). Two rows sharing a key are visited in the
     * same relative order both when this method builds their list here and
     * when the caller's `.map` consumes it there -- both walks are driven by
     * the identical, single underlying file read -- so popping front-to-back
     * reunites each row with its own, file-order-derived index. This still
     * depends on nothing being inserted between `splitCsv` and that `.map`
     * that could reorder items (a pre-existing caveat, unchanged by this
     * method's contract).
     *
     * @param csvPath     path to the samplesheet
     * @param imageColumn the column holding this step's entry image
     */
    static Map<String, List<Integer>> rowIndexPerPatient(String csvPath, String imageColumn) {
        def file = new File(csvPath)
        if (!file.exists()) return [:]

        def lines = readCsvLines(file.path)
        if (lines.size() < 2) return [:]

        def header = parseCsvLine(lines[0])
        def patientIdx = header.findIndexOf { it == 'patient_id' }
        def imageIdx   = header.findIndexOf { it == imageColumn }
        if (patientIdx == -1 || imageIdx == -1) return [:]

        def nextIndex = [:].withDefault { 0 }
        def result = [:].withDefault { [] }
        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            if (cols.size() <= Math.max(patientIdx, imageIdx)) return
            def patientId = cols[patientIdx].trim()
            if (!patientId) return  // ignore blank patient_id cells
            def rawImage = cols[imageIdx].trim()
            def key = "${patientId}::${rawImage}".toString()
            result[key] << nextIndex[patientId]
            nextIndex[patientId] = nextIndex[patientId] + 1
        }
        return result
    }

    /**
     * The stem lib/Meta.groovy's identityFor derives an id from: everything
     * before the FIRST '.' in the filename. Kept as one private helper so
     * stemCountsPerPatient can never drift from identityFor's own inline copy
     * of this rule (Meta stays parameterless-static and cannot call back into
     * CsvUtils without inverting the module's dependency direction).
     */
    private static String stemOf(String rawImage) {
        def name = new File(rawImage.toString()).name
        def dot  = name.indexOf('.')
        return dot >= 0 ? name.substring(0, dot) : name
    }

    /**
     * The ctx {@link Meta#fromCheckpointRow} needs (`keepChannelsBySlide` /
     * `imagesCount` / `channelsCount`), computed ONCE from a checkpoint CSV's OWN
     * rows -- never from a samplesheet. Every checkpoint reader needs the identical
     * shape (mirroring INPUT_CHECK's samplesheet-side ctx assembly:
     * countImagesPerPatient / resolveKeptChannelsPerSlide / countChannelsPerPatient),
     * so it is written once here rather than re-derived independently by each one.
     *
     * WHY THIS ISN'T resolveKeptChannelsPerSlide AGAIN. That resolver keys its
     * per-slide map on the RAW image cell, because a samplesheet row has no assigned
     * `id` yet at the point it runs. A checkpoint row already has one -- a real,
     * assigned identity (RULING R17, lib/Checkpoint.groovy) -- so THIS resolver keys
     * `keepChannelsBySlide` on `id` directly, matching how `Meta.fromCheckpointRow`
     * looks its own row up (`finish()`'s `slideKey` argument is `meta.id`). Same
     * "claim each marker name once per patient, reference row(s) first" algorithm as
     * resolveKeptChannelsPerSlide, restated against a checkpoint's own columns
     * instead of a samplesheet's -- the two are independent implementations of one
     * rule, not one calling the other, because a checkpoint row's `channels` column
     * is the FULL declared list a writer recorded (`meta.channels.join('|')`), never
     * a samplesheet cell resolveKeptChannelsPerSlide could re-parse directly.
     *
     * A step whose schema does not carry `channels`/`is_reference` (currently only
     * 'postprocessed') yields an empty keepChannelsBySlide/channelsCount for every
     * patient rather than throwing: `Meta.fromCheckpointRow` already tolerates that
     * schema shape (it only requires `is_reference`/`channels` on a row when the
     * step's OWN schema declares them), so this helper must not be stricter than the
     * constructor it feeds.
     *
     * @param csvPath path to a checkpoint CSV (e.g. csv/segmented.csv)
     * @param step    the checkpoint step name (lib/Checkpoint.groovy STEPS) -- used
     *                only to know whether this schema carries channels/is_reference
     *                at all; the column POSITIONS are still read from the file's own
     *                header, never assumed from Checkpoint.columns(step)'s order.
     */
    static Map metaContextFromCheckpoint(String csvPath, String step) {
        def empty = [keepChannelsBySlide: [:], imagesCount: [:], channelsCount: [:]]

        def file = new File(csvPath)
        if (!file.exists()) return empty

        def lines = readCsvLines(file.path)
        if (lines.size() < 2) return empty  // Header only or empty

        def schemaColumns = Checkpoint.columns(step)
        def header     = parseCsvLine(lines[0])
        def patientIdx = header.findIndexOf { it == 'patient_id' }
        def idIdx      = header.findIndexOf { it == 'id' }
        if (patientIdx == -1 || idIdx == -1) return empty

        def hasChannels  = schemaColumns.contains('channels')     && header.contains('channels')
        def hasReference = schemaColumns.contains('is_reference') && header.contains('is_reference')
        def channelsIdx  = hasChannels  ? header.findIndexOf { it == 'channels' }     : -1
        def refIdx       = hasReference ? header.findIndexOf { it == 'is_reference' } : -1

        def rowsByPatient = [:].withDefault { [] }
        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            def neededIdx = [patientIdx, idIdx, channelsIdx, refIdx].findAll { it >= 0 }
            if (cols.size() <= (neededIdx ? neededIdx.max() : 0)) return
            def patientId = cols[patientIdx].trim()
            if (!patientId) return  // ignore blank patient_id cells
            def id = cols[idIdx].trim()
            if (!id) return  // an id-less row can't key keepChannelsBySlide; Meta itself rejects it
            def channels = hasChannels ? cols[channelsIdx].split('\\|')*.trim().findAll { it } : []
            def isRef    = hasReference && cols[refIdx].trim().toLowerCase() == 'true'
            rowsByPatient[patientId] << [id: id, channels: channels, isRef: isRef]
        }

        def imagesCount = rowsByPatient.collectEntries { patientId, rows -> [patientId, rows.size()] }

        def keepChannelsBySlide = [:]
        def channelsCount       = [:]
        rowsByPatient.each { patientId, rows ->
            // Stable partition: reference row(s) first, everything else in checkpoint
            // order -- the same rule resolveKeptChannelsPerSlide applies to a
            // samplesheet's rows.
            def ordered = rows.findAll { it.isRef } + rows.findAll { !it.isRef }
            def claimed = new HashSet<String>()
            def perSlide = [:]
            ordered.each { row ->
                def keep = []
                row.channels.each { ch ->
                    def name = ch.toUpperCase()
                    if (claimed.contains(name)) return
                    claimed << name
                    keep << ch
                }
                perSlide[row.id] = keep
            }
            keepChannelsBySlide[patientId] = perSlide
            channelsCount[patientId] = perSlide.values().sum { it.size() } ?: 0
        }

        return [
            keepChannelsBySlide: keepChannelsBySlide,
            imagesCount        : imagesCount,
            channelsCount      : channelsCount,
        ]
    }

    /**
     * Validate one already-parsed meta map, and return it unchanged.
     *
     * Checks, in order: `patient_id` is present; `is_reference` is a real Boolean
     * (not a truthy String); `channels` is a non-empty List with no null or blank
     * entry; and at least one channel is a nuclear/fiducial marker.
     *
     * The nuclear marker may sit at ANY position -- segmentation and the
     * registration fiducial both locate it by NAME -- and which names qualify comes
     * from params.nuclear_markers via MarkerUtils, never a hardcoded 'DAPI', which
     * used to reject an otherwise valid CELLTOX-only samplesheet before the run
     * could start.
     *
     * @param meta           the map to validate
     * @param nuclearMarkers params.nuclear_markers, in any form MarkerUtils accepts
     * @param context        included verbatim in every message, so a failure names
     *                       the row it came from
     *
     * Returns the same `meta` map.
     * @throws IllegalArgumentException on a missing patient_id, a non-Boolean
     *         is_reference, or a missing/empty/blank-entry channel list.
     * @throws IllegalStateException when no channel is a nuclear marker; the
     *         message lists both the accepted marker names and the channels found.
     */
    static Map validateMetadata(Map meta, def nuclearMarkers, String context = 'unknown') {

        if (!meta.patient_id)
            throw new IllegalArgumentException("Missing patient_id in ${context}")

        if (!(meta.is_reference instanceof Boolean))
            throw new IllegalArgumentException("is_reference must be boolean in ${context}")

        if (!(meta.channels instanceof List) || meta.channels.isEmpty())
            throw new IllegalArgumentException("channels must be a non-empty List in ${context}")

        if (meta.channels.any { it == null || it.trim().isEmpty() })
            throw new IllegalArgumentException("Empty channel name found for patient ${meta.patient_id}")

        // The nuclear/fiducial marker may appear at ANY position (segmentation and the
        // registration fiducial locate it by name, not index). Only its presence is
        // required, and WHICH names qualify comes from params.nuclear_markers via
        // MarkerUtils — not a hardcoded 'DAPI', which rejected an otherwise valid
        // CELLTOX-only samplesheet before the run could start.
        if (!MarkerUtils.hasNuclear(meta.channels, nuclearMarkers)) {
            throw new IllegalStateException("No nuclear channel (${MarkerUtils.markerList(nuclearMarkers).join(', ')}) found for patient ${meta.patient_id} (${context}). Found channels: ${meta.channels}")
        }

        return meta
    }

    /**
     * Strictly parse an is_reference cell. Accepts only 'true'/'false'
     * (case-insensitive); anything else is rejected so typos like "yes"
     * cannot be silently coerced to false and corrupt reference selection.
     */
    static Boolean parseIsReference(def value, String context = 'unknown') {
        def s = (value ?: '').toString().trim().toLowerCase()
        if (s == 'true')  return true
        if (s == 'false') return false
        throw new IllegalArgumentException("Invalid is_reference value '${value}' in ${context}. Must be 'true' or 'false'.")
    }

    /**
     * Build a validated meta map from one raw samplesheet row.
     *
     * Splits the `channels` cell on '|' and trims each name, parses `is_reference`
     * strictly through parseIsReference (so a typo like "yes" is rejected rather
     * than silently coerced to false and corrupting reference selection), then
     * hands the result to validateMetadata.
     *
     * @param row            the raw row, keyed by column name
     * @param nuclearMarkers params.nuclear_markers, in any form MarkerUtils accepts
     * @param context        prefix for every message; the patient id is appended
     *
     * Returns a Map with `patient_id`, `is_reference` and `channels`.
     * @throws IllegalArgumentException from parseIsReference or validateMetadata.
     * @throws IllegalStateException from validateMetadata's nuclear-marker check.
     */
    static Map parseMetadata(Map row, def nuclearMarkers, String context = 'parseMetadata') {

        def channels = row.channels
            ?.split('\\|')
            ?.collect { it.trim() } ?: []

        def meta = [
            patient_id  : row.patient_id?.toString()?.trim(),
            is_reference: parseIsReference(row.is_reference, "${context} (${row.patient_id})"),
            channels    : channels
        ]

        return validateMetadata(meta, nuclearMarkers, "${context} (${row.patient_id})")
    }

    /**
     * Structural check of a samplesheet: it exists, it is not empty, and its header
     * carries every required column.
     *
     * Structure only -- the per-row semantic checks are validateInputSemantics's. Run
     * before anything reads a row, so a missing column is named at launch instead of
     * surfacing as a null further down.
     *
     * @param csv           path to the samplesheet
     * @param required_cols the column names the current step needs, from
     *                      ParamUtils.requiredColumnsForStep
     *
     * @throws FileNotFoundException if the file does not exist.
     * @throws RuntimeException if the file has no lines at all.
     * @throws IllegalArgumentException naming the first required column missing
     *         from the header.
     */
    static void validateInputCSV(def csv, List required_cols) {

        def file = new File(csv)
        if (!file.exists())
            throw new FileNotFoundException("Input CSV not found: ${csv}")

        def lines = readCsvLines(file.path)
        if (lines.isEmpty())
            throw new RuntimeException("CSV is empty: ${csv}")

        def header = parseCsvLine(lines.first())

        required_cols.each {
            if (!(it in header))
                throw new IllegalArgumentException("Missing required column '${it}' in CSV: ${csv}")
        }
    }

    /**
     * Header columns this pipeline does not read, for a run entering at `step`.
     *
     * REPORTED, NEVER REJECTED, and the distinction is load-bearing. A checkpoint
     * CSV legitimately carries columns the entry step does not read -- a
     * csv/registered.csv read by `--start segmentation` has `id` and `pixel_size`
     * beyond that step's four required columns, and a segmented.csv has five more.
     * Rejecting an unknown column would break re-entry by construction.
     *
     * Accepting them SILENTLY is the other failure, and it is the one that bites:
     * validateInputCSV asserts the required columns are present and ignores
     * everything else, so a mistyped `channles` sits unnoticed beside a missing
     * `channels` and the run dies much later, somewhere else, on a null.
     *
     * The known set is the union of the entry step's required columns and EVERY
     * checkpoint step's columns, derived from ParamUtils.STEPS and
     * Checkpoint.STEPS rather than restated -- so a new column added to either
     * table stops being "unknown" without anything here changing.
     *
     * @return the unknown columns, in header order (empty when there are none)
     */
    static List<String> unknownColumns(def csv, String step) {
        def known = (ParamUtils.requiredColumnsForStep(step) +
                     Checkpoint.STEPS.collectMany { it.columns }) as Set
        def lines = readCsvLines(csv)
        if (lines.isEmpty()) return []
        return parseCsvLine(lines[0]).findAll { it && !(it in known) }
    }

    /**
     * Fail-fast semantic validation of the whole samplesheet, run at parse
     * time (and therefore visible under --dry_run). Complements the per-row
     * checks that otherwise only fire later during channel construction.
     *
     * Validates, for every data row: is_reference format, channel list /
     * nuclear-marker presence, and existence of the step's image file. Validates, per
     * patient: exactly one reference image (zero allowed only for mode=add_cycle,
     * whose reference is the prior run's; more than one is always ambiguous).
     *
     * @param csv                  path to the input samplesheet
     * @param step                 pipeline start step (selects the path column)
     * @param allowNoReference     whether a patient may legitimately declare NO
     *                             reference. TRUE only for mode=add_cycle. This no
     *                             longer causes anything to be promoted in its place --
     *                             it only suppresses the error.
     * @param nuclearMarkers       params.nuclear_markers — required, never defaulted here
     */
    static void validateInputSemantics(def csv, String step, boolean allowNoReference, def nuclearMarkers,
                                       boolean requireUniqueChannelSets = false) {

        // ParamUtils.STEPS is the single source of truth for "what is a step?"
        // (name / requiredColumns / entryColumn / qcKinds) -- see its header
        // comment in lib/ParamUtils.groovy. entryColumnForStep throws on an
        // unrecognised step rather than the old map literal's silent `null`,
        // but both call sites below only ever pass a step already validated
        // by nextflow_schema.json's enum (or the add_cycle branch's hardcoded
        // 'preprocessing'), so that stricter failure mode is unreachable in
        // practice and loud instead of silent if it ever is reached.
        def pathColumn = ParamUtils.entryColumnForStep(step)

        def lines = readCsvLines(csv)
        if (lines.size() < 2)
            throw new IllegalStateException("Input CSV contains no data rows: ${csv}")

        def header     = parseCsvLine(lines[0])
        def piIdx      = header.findIndexOf { it == 'patient_id' }
        def refIdx     = header.findIndexOf { it == 'is_reference' }
        def chIdx      = header.findIndexOf { it == 'channels' }
        def pathIdx    = pathColumn ? header.findIndexOf { it == pathColumn } : -1

        def refCounts = [:].withDefault { 0 }
        def rowCounts = [:].withDefault { 0 }
        // patient -> signature -> [row contexts]. Only consulted when
        // requireUniqueChannelSets: the caller says whether the registration backend
        // about to run pairs its outputs by channel set (RegBackends'
        // pairsOutputsBySignature), which is the only reason two slides of one patient
        // may not share one. The rule is RegisteredMatch.signature's -- lower-cased,
        // sorted -- so 'CD8|dapi|CD3' and 'DAPI|CD3|CD8' are the same set here exactly
        // as they are at pairing time.
        def sigRows   = [:].withDefault { [:].withDefault { [] } }

        lines.drop(1).eachWithIndex { line, i ->
            def cols = parseCsvLine(line)
            if (cols.every { it == null || it.trim().isEmpty() }) return  // skip blank lines

            def ctx = "row ${i + 2} of ${csv}"
            def row = [
                patient_id  : piIdx  >= 0 ? cols[piIdx]?.trim() : null,
                is_reference: refIdx >= 0 ? cols[refIdx] : null,
                channels    : chIdx  >= 0 && chIdx < cols.size() ? cols[chIdx] : null,
            ]

            // Per-row format + nuclear-channel validation (throws on problems).
            def parsed = parseMetadata(row, nuclearMarkers, ctx)

            // Image file must exist (resolved against the launch directory for
            // relative paths). Skipped only if the path column is absent.
            if (pathIdx >= 0 && pathIdx < cols.size()) {
                def p = cols[pathIdx].trim()
                if (!p)
                    throw new IllegalArgumentException("Empty path in column '${pathColumn}' for patient ${row.patient_id} (${ctx})")
                if (!new File(p).exists())
                    throw new FileNotFoundException("Input file does not exist: ${p} (patient ${row.patient_id}, ${ctx})")
            }

            rowCounts[row.patient_id]++
            if (parsed.is_reference) refCounts[row.patient_id]++  // reuse parsed value (no re-parse)
            if (requireUniqueChannelSets)
                sigRows[row.patient_id][RegisteredMatch.signature(parsed.channels as List)] << ctx
        }

        if (requireUniqueChannelSets) {
            sigRows.each { patientId, bySig ->
                bySig.findAll { _sig, rows -> rows.size() > 1 }.each { sig, rows ->
                    throw new IllegalStateException(
                        "Two slides of patient ${patientId} share the same channel set (${sig}): " +
                        "${rows.join(' and ')}. The registration backend pairs its registered " +
                        "outputs back to their slides by channel set, so these two cannot be told " +
                        "apart after registration -- the run would abort at that point, after the " +
                        "whole group had been registered. Drop one of the two rows, or give the " +
                        "repeat its own patient_id.")
                }
            }
        }

        rowCounts.each { patientId, _n ->
            def refs = refCounts[patientId]
            if (refs > 1)
                throw new IllegalStateException("Multiple reference images found for patient ${patientId} (${refs} found). Exactly one image per patient may set is_reference=true.")
            if (refs == 0 && !allowNoReference)
                throw new IllegalStateException("No reference image found for patient ${patientId}. Set is_reference=true on exactly one of that patient's images. There is no auto-promotion: the reference is the slide every other slide is warped onto, so the pipeline will not choose one for you.")
        }
    }
}
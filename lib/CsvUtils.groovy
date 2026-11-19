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
     * Read a CSV's lines, stripping a UTF-8 BOM (Excel) that would otherwise be glued
     * onto the first column name and break every column lookup.
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
     * patient_id -> the value of `imageColumn` on that patient's registration
     * reference row. The one place the reference is decided.
     *
     * The row declaring `is_reference=true` wins; otherwise the patient is absent from
     * the map. No reference is ever invented from row order. A missing reference is
     * legitimate only for mode='add_cycle' and is reported by validateInputSemantics,
     * which runs first; this method resolves, it does not validate.
     *
     * Resolved at samplesheet read (not from groupTuple arrival order) so the result is
     * deterministic and reaches csv/preprocessed.csv.
     *
     * @param imageColumn the column holding the image this step consumes (INPUT_CHECK's
     *                    entry column). Assumed unique per row within a patient.
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

        // rows per patient, in samplesheet order
        def rowsByPatient = [:].withDefault { [] }
        lines.drop(1).each { line ->
            def cols = parseCsvLine(line)
            if (cols.size() <= Math.max(patientIdx, imageIdx)) return
            def patientId = cols[patientIdx].trim()
            if (!patientId) return  // ignore blank patient_id cells
            // lenient: validateInputSemantics already rejected malformed values
            def isRef = refIdx != -1 && refIdx < cols.size() &&
                        cols[refIdx]?.trim()?.toLowerCase() == 'true'
            rowsByPatient[patientId] << [isRef, cols[imageIdx].trim()]
        }

        def resolved = [:]
        rowsByPatient.each { patientId, rows ->
            def declared = rows.find { it[0] }
            if (declared) { resolved[patientId] = declared[1]; return }
            // else: no reference; nothing is promoted in its place
        }
        return resolved
    }

    /**
     * patient_id -> channels of the REFERENCE row, from a checkpoint CSV.
     *
     * add_cycle's `preClaimed` seed for resolveKeptChannelsPerSlide: the prior pyramid
     * already has these markers. Read synchronously because the keep-set must be known
     * at workflow construction.
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
     * The inner key is Meta.identityFor's output for the row (the meta.id
     * fromSamplesheetRow assigns), not the raw image cell, which may repeat or be blank
     * within a patient. stemCounts is computed here from the same sheet as
     * input_check.nf computes it, so writer and reader keys match.
     *
     * This is the only definition of the keep-set: SPLIT_CHANNELS emits it (via
     * meta.keep_channels) and countChannelsPerPatient sums it.
     *
     * A channel is kept iff its upper-cased name was not already claimed by an earlier
     * slide of the patient, walking the reference first, then samplesheet order.
     * Nuclear-ness plays no part, so each name reaches SPLIT_CHANNELS exactly once and
     * deterministically, and the emitted-file count equals the distinct-name count.
     *
     * Within one sheet only nuclear names still repeat (validateInputSemantics refuses
     * repeated non-nuclear ones), but `preClaimed` can bring names from a prior run
     * (add_cycle), which must be deduplicated rather than refused.
     *
     * @param nuclearMarkers validated via MarkerUtils.markerList so a malformed value
     *        still fails here, though the keep rule does not branch on it.
     */
    static Map<String, Map<String, List<String>>> resolveKeptChannelsPerSlide(
            String csvPath, String imageColumn, def nuclearMarkers,
            Map<String, List<String>> preClaimed = [:]) {

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

        def referenceImage = resolveReferenceRows(csvPath, imageColumn)

        // same map input_check.nf passes as ctx.stemCounts
        def stemCounts = stemCountsPerPatient(csvPath, imageColumn)

        // Rows per patient in samplesheet order; `idx` (position within the patient) is
        // identityFor's rowIndex.
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
            // reference rows are matched on the raw cell
            def refCell = referenceImage[patientId]
            // Stable partition: reference first, then declared order (no reference:
            // samplesheet order).
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
                // keyed on the assigned identity, so duplicate/blank cells cannot collide
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
     * Sums resolveKeptChannelsPerSlide, the rule SPLIT_CHANNELS applies, so the count
     * equals both the emitted TIFFs and the distinct names.
     *
     * Do not substitute countDeclaredChannelsPerPatient: it over-counts, and an
     * over-counted groupKey never fills (the run hangs).
     *
     * Returns an empty map on a missing file, a header-only sheet, or an absent
     * patient_id / channels / <imageColumn> column.
     *
     * @param csvPath        path to the samplesheet
     * @param imageColumn    the column naming the step's input image
     * @param nuclearMarkers params.nuclear_markers, in any form MarkerUtils accepts
     *
     * Returns a Map of patient_id -> emitted channel count.
     */
    static Map<String, Integer> countChannelsPerPatient(String csvPath, String imageColumn, def nuclearMarkers) {
        def kept = resolveKeptChannelsPerSlide(csvPath, imageColumn, nuclearMarkers)
        return kept.collectEntries { patientId, perSlide ->
            [patientId, (perSlide.values().sum { it.size() } ?: 0)]
        }
    }

    /**
     * Count the declared channels per patient: the union of the `channels` cells, with
     * no reference or nuclear awareness. For run_summary.json's input manifest only,
     * which reports what the samplesheet said.
     *
     * Never use it for channels_count / a groupKey size (see countChannelsPerPatient).
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
     * How many rows in each patient share a source-image stem ("patientId::stem").
     * Meta.identityFor uses it to suffix only colliding ids. The stem rule must match
     * identityFor's (stemOf: everything before the first '.').
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
     * Each patient's 0-based row positions, in samplesheet order, grouped by
     * "patientId::rawImageCell", so a caller building meta from a splitCsv row can look
     * up its own index.
     *
     * A list per key, because the raw cell may repeat (or be blank) within a patient.
     * The caller must pop the first element per row (input_check.nf does). That is
     * deterministic because splitCsv of one file emits in file order, unlike
     * groupTuple; it relies on nothing reordering items between splitCsv and that map.
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
     * Everything before the first '.' in the filename: the same rule as
     * Meta.identityFor's inline copy (Meta does not depend on CsvUtils).
     */
    private static String stemOf(String rawImage) {
        def name = new File(rawImage.toString()).name
        def dot  = name.indexOf('.')
        return dot >= 0 ? name.substring(0, dot) : name
    }

    /**
     * The ctx {@link Meta#fromCheckpointRow} needs (`keepChannelsBySlide` /
     * `imagesCount` / `channelsCount`), computed from a checkpoint CSV's own rows.
     *
     * Same claim-once, reference-first rule as resolveKeptChannelsPerSlide, but keyed on
     * the row's recorded `id`. A schema without channels/is_reference ('postprocessed')
     * yields empty maps rather than throwing, matching fromCheckpointRow.
     *
     * @param csvPath path to a checkpoint CSV (e.g. csv/segmented.csv)
     * @param step    the Checkpoint.STEPS name; only decides whether channels /
     *                is_reference exist. Positions come from the file's own header.
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
            // reference first, then checkpoint order
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
     * The nuclear marker may be at any position (it is located by name); which names
     * qualify comes from params.nuclear_markers via MarkerUtils.
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
     * Splits `channels` on '|', parses `is_reference` strictly (parseIsReference), then
     * runs validateMetadata.
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
     * Structure only; per-row checks are validateInputSemantics's.
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
     * Reported, never rejected: checkpoint CSVs legitimately carry extra columns, but a
     * silent typo (`channles`) would otherwise fail much later on a null. Known = the
     * entry step's required columns plus every checkpoint column (derived from
     * ParamUtils.STEPS and Checkpoint.STEPS).
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
     * Fail-fast semantic validation of the whole samplesheet at parse time (visible
     * under --dry_run).
     *
     * Validates, for every data row: is_reference format, channel list /
     * nuclear-marker presence, no channel name repeated within the row's own slide,
     * and existence of the step's image file. Validates, per patient: exactly one
     * reference image (zero allowed only for mode=add_cycle, whose reference is the
     * prior run's), and that no non-nuclear channel is carried by two slides.
     *
     * @param csv                  path to the input samplesheet
     * @param step                 pipeline start step (selects the path column)
     * @param allowNoReference     whether a patient may declare no reference (only
     *                             mode=add_cycle); nothing is promoted in its place.
     * @param nuclearMarkers       params.nuclear_markers — required, never defaulted here
     */
    static void validateInputSemantics(def csv, String step, boolean allowNoReference, def nuclearMarkers,
                                       boolean requireUniqueChannelSets = false) {

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
        // patient -> channel-set signature -> rows. Checked only when the backend pairs
        // outputs by channel set (RegBackends.pairsOutputsBySignature); same signature
        // rule as RegisteredMatch.
        def sigRows   = [:].withDefault { [:].withDefault { [] } }

        // patient_id -> upper-cased channel -> [[name: as written, where: 'row N (<path>)']],
        // for the per-patient duplicate check below
        def channelOccurrences = [:]

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

            // image must exist (relative paths resolve against the launch directory)
            if (pathIdx >= 0 && pathIdx < cols.size()) {
                def p = cols[pathIdx].trim()
                if (!p)
                    throw new IllegalArgumentException("Empty path in column '${pathColumn}' for patient ${row.patient_id} (${ctx})")
                if (!new File(p).exists())
                    throw new FileNotFoundException("Input file does not exist: ${p} (patient ${row.patient_id}, ${ctx})")
            }

            // spreadsheet row number (+ image) for messages
            def rowPath = (pathIdx >= 0 && pathIdx < cols.size()) ? cols[pathIdx].trim() : null
            def where   = rowPath ? "row ${i + 2} (${rowPath})" : "row ${i + 2}"

            // No name may repeat within one slide (nuclear included): SPLIT_CHANNELS emits
            // one file per name, so channels_count would disagree with the files.
            def seenInRow = [:]
            parsed.channels.each { ch ->
                def key = ch.trim().toUpperCase()
                if (seenInRow.containsKey(key))
                    throw new IllegalArgumentException("Channel '${ch}' is listed more than once on the same slide, in ${ctx} (patient ${row.patient_id}, channels: ${parsed.channels.join('|')}). A channel name may appear at most once per slide -- nuclear markers included -- because the slide emits exactly one image per name. Remove the repeat, or rename one of the two if they really are different stains.")
                seenInRow[key] = ch

                def perPatient = channelOccurrences[row.patient_id]
                if (perPatient == null) {
                    perPatient = [:]
                    channelOccurrences[row.patient_id] = perPatient
                }
                if (perPatient[key] == null) perPatient[key] = []
                perPatient[key] << [name: ch, where: where]
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

            // A non-nuclear channel on two slides of one patient is refused: the keep-set
            // would silently keep one, chosen by row order. Nuclear markers (the fiducial)
            // are expected on every slide. Names compared trimmed and upper-cased.
            (channelOccurrences[patientId] ?: [:]).each { _key, occurrences ->
                if (occurrences.size() > 1 && !MarkerUtils.isNuclear(occurrences[0].name, nuclearMarkers))
                    throw new IllegalArgumentException("Channel '${occurrences[0].name}' appears on ${occurrences.size()} different slides of patient ${patientId} (${occurrences.collect { it.where }.join('; ')}), and it is not one of the nuclear markers (${MarkerUtils.markerList(nuclearMarkers).join(', ')}). Only a nuclear/fiducial channel may repeat across a patient's slides: every other marker is kept exactly once per patient, so one of these acquisitions would be dropped and which one survives is decided by samplesheet order rather than by you. Rename the channel on one of the slides if they are genuinely different acquisitions (e.g. PANCK_cycle1 / PANCK_cycle2), or add it to --nuclear_markers if it really is a nuclear stain (and if these entries name the same file, the row itself is duplicated: remove one).")
            }

            def refs = refCounts[patientId]
            if (refs > 1)
                throw new IllegalStateException("Multiple reference images found for patient ${patientId} (${refs} found). Exactly one image per patient may set is_reference=true.")
            if (refs == 0 && !allowNoReference)
                throw new IllegalStateException("No reference image found for patient ${patientId}. Set is_reference=true on exactly one of that patient's images. There is no auto-promotion: the reference is the slide every other slide is warped onto, so the pipeline will not choose one for you.")
        }
    }
}
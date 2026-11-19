/*
========================================================================================
    RegisteredMatch — pair registered output files back to their slide metas
========================================================================================
    VALIS renames its outputs, so filenames do not identify the slide (and parsing
    them breaks on patient ids with underscores). The OME channel set survives:
    bin/create_channels_manifest.py writes filename -> [channel names], and a file
    belongs to the slide whose samplesheet `channels` name the same set.

    Every mismatch throws: a wrong pairing would silently mislabel channels all the
    way to the GeoJSON. A duplicate file signature surfaces as an "unmatched" throw.

    Static; reads no params.
========================================================================================
*/

class RegisteredMatch {

    /**
     * The signature used for matching: channels lower-cased, sorted, joined with '|'.
     *
     * toSorted(), not sort(): `meta + [k: v]` is a shallow copy, so meta.channels is
     * shared and an in-place sort would reorder channels for every meta. Elements are
     * lower-cased before sorting so case does not change the order.
     */
    static String signature(List<String> channels) {
        if (channels == null) {
            throw new IllegalArgumentException('RegisteredMatch.signature: channels is null')
        }
        return channels.collect { it.toString().toLowerCase() }.toSorted().join('|')
    }

    /**
     * Pair registered output files back to their slide metas by OME channel signature.
     *
     * `manifest` is the parsed channels-manifest JSON written by
     * bin/create_channels_manifest.py: filename -> [channel names].
     * Returns [[meta, file], ...] in `metas` order.
     */
    static List<List> pair(List<Map> metas, List files, Map<String, List<String>> manifest) {
        if (files.size() != metas.size()) {
            // Null-tolerant, so building this message cannot throw a different error.
            def sigOrNull = { m -> m.channels == null ? '<null channels>' : signature(m.channels as List) }
            throw new IllegalStateException(
                "RegisteredMatch: count mismatch - ${metas.size()} slide(s) but " +
                "${files.size()} registered file(s). " +
                "Slide signatures: ${metas.collect { sigOrNull(it) }.join(', ')}. " +
                "Files: ${files.collect { it.name }.join(', ')}. " +
                'A registration that returns fewer files than slides is a partial failure; ' +
                'pairing what arrived would publish a short run as a complete one.')
        }

        def metaSignatures = metas.collect { signature(it.channels as List) }
        def duplicates = metaSignatures.countBy { it }.findAll { _sig, n -> n > 1 }.keySet()
        if (duplicates) {
            throw new IllegalStateException(
                "RegisteredMatch: duplicate signature - ${duplicates.join(', ')} " +
                "carried by more than one of the ${metas.size()} slides. Each slide must have a " +
                'unique combination of channels: two slides with the same channel set cannot be ' +
                'told apart from their registered output, and a map keyed on the signature would ' +
                'silently keep only the last of them.')
        }

        def fileBySignature = [:]
        files.each { f ->
            def channels = manifest[f.name]
            if (!channels) {
                throw new IllegalStateException(
                    "RegisteredMatch: unmatched - the channels manifest has no entry for " +
                    "${f.name}. Manifest keys: ${manifest.keySet().join(', ')}. " +
                    'Registered files must carry OME-XML channel names.')
            }
            fileBySignature[signature(channels as List)] = f
        }

        return [metas, metaSignatures].transpose().collect { meta, sig ->
            def f = fileBySignature[sig]
            if (f == null) {
                throw new IllegalStateException(
                    "RegisteredMatch: unmatched - no registered file carries the channel " +
                    "signature ${sig}. Available file signatures: " +
                    "${fileBySignature.keySet().join(', ')}. Check that the samplesheet " +
                    "'channels' column matches the OME-XML channel names.")
            }
            [meta, f]
        }
    }
}

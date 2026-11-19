/*
 * MarkerUtils - the pipeline's one nuclear/fiducial marker rule, for the Groovy
 * and Nextflow layer.
 *
 * Answers "is this channel nuclear?" from params.nuclear_markers. Which channels a
 * slide emits is decided elsewhere (CsvUtils.resolveKeptChannelsPerSlide), since that
 * needs patient-wide context.
 *
 * The marker list is always an argument; never add a default here (defaults live in
 * nextflow.config only).
 *
 * Matching is case-insensitive substring, to agree with bin/utils/metadata.py's
 * is_nuclear/pick_nuclear_index (e.g. 'DAPI_nuclear' is nuclear).
 */
class MarkerUtils {

    /**
     * The configured markers, trimmed and upper-cased.
     *
     * Every consumer reads params.nuclear_markers through this. It accepts all three
     * shapes the param arrives in, two of which break silently if read raw:
     *
     *   ['DAPI','CELLTOX']  nextflow.config / a params file
     *   'CELLTOX'           the CLI (iterating a String yields characters)
     *   ['DAPI,CELLTOX']    one comma string in a list (matches no channel)
     *
     * Every element is split on commas and whitespace. Throws when empty.
     */
    static List<String> markerList(def nuclearMarkers) {
        def raw
        if (nuclearMarkers == null)                      raw = []
        else if (nuclearMarkers instanceof CharSequence) raw = [nuclearMarkers]
        else if (nuclearMarkers instanceof Object[])     raw = nuclearMarkers as List
        else if (nuclearMarkers instanceof Collection)   raw = nuclearMarkers as List
        else                                             raw = [nuclearMarkers]

        def markers = raw
            .collectMany { it == null ? [] : it.toString().split(/[,\s]+/) as List }
            .collect { it?.trim()?.toUpperCase() }
            .findAll { it }
        if (!markers)
            throw new IllegalArgumentException(
                "No nuclear markers configured. Pass params.nuclear_markers; its default " +
                "lives in nextflow.config and nowhere else.")
        return markers
    }

    /** True when `channel` names one of the configured nuclear/fiducial markers. */
    static boolean isNuclear(def channel, def nuclearMarkers) {
        if (channel == null) return false
        def name = channel.toString().trim().toUpperCase()
        if (!name) return false
        return markerList(nuclearMarkers).any { name.contains(it) }
    }

    /** True when at least one of `channels` is a nuclear/fiducial marker. */
    static boolean hasNuclear(List channels, def nuclearMarkers) {
        return (channels ?: []).any { isNuclear(it, nuclearMarkers) }
    }

    /**
     * Index of the nuclear channel in `channels`, or -1 when there is none.
     *
     * Marker preference order wins over channel order, as in metadata.py's
     * pick_nuclear_index: markers ['DAPI','CELLTOX'], channels ['CELLTOX','CD8','DAPI']
     * gives 2. Must agree with the Python resolver (CellSAM's --dapi-channel uses it).
     */
    static int nuclearIndex(List channels, def nuclearMarkers) {
        def names = (channels ?: []).collect { it?.toString()?.trim()?.toUpperCase() ?: '' }
        for (marker in markerList(nuclearMarkers)) {
            def hit = names.findIndexOf { it && it.contains(marker) }
            if (hit >= 0) return hit
        }
        return -1
    }

}

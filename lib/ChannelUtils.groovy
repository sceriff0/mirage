class ChannelUtils {

    /**
     * Build pairwise tuples [moving_meta, reference_file, moving_file]
     * from a grouped patient bundle.
     */
    static List<List> toPairwiseTuples(def referenceItem, List allItems) {
        def referenceFile = referenceItem[1]
        return allItems
            .findAll { item -> !item[0].is_reference }
            .collect { movingItem -> [movingItem[0], referenceFile, movingItem[1]] }
    }

    /**
     * Extract the [meta, file] reference item from grouped tuples.
     */
    static List referenceTuple(def referenceItem) {
        return referenceItem
    }
}

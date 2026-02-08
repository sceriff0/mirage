nextflow.enable.dsl = 2

import static ChannelUtils.*

include { REGISTER_VALIS_PAIRS } from '../../../modules/local/register_valis_pairs'
include { PUBLISH_REFERENCE as PUBLISH_REFERENCE_VALIS_PAIRS } from '../../../modules/local/publish_reference'

workflow VALIS_PAIRS_ADAPTER {
    take:
    ch_grouped_meta

    main:
    ch_pairs = ch_grouped_meta
        .flatMap { _patient_id, reference_item, all_items ->
            toPairwiseTuples(reference_item, all_items)
        }

    REGISTER_VALIS_PAIRS(ch_pairs)

    ch_references = ch_grouped_meta
        .map { _patient_id, reference_item, _all_items ->
            referenceTuple(reference_item)
        }

    PUBLISH_REFERENCE_VALIS_PAIRS(ch_references)

    ch_all = PUBLISH_REFERENCE_VALIS_PAIRS.out.published.concat(REGISTER_VALIS_PAIRS.out.registered)
    ch_size_logs = REGISTER_VALIS_PAIRS.out.size_log

    emit:
    registered = ch_all
    size_logs = ch_size_logs
}

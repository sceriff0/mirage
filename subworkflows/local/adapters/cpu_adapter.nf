nextflow.enable.dsl = 2

import static ChannelUtils.*

include { CPU_REGISTER } from '../../../modules/local/register_cpu'
include { PUBLISH_REFERENCE as PUBLISH_REFERENCE_CPU } from '../../../modules/local/publish_reference'

workflow CPU_ADAPTER {
    take:
    ch_grouped_meta

    main:
    ch_pairs = ch_grouped_meta
        .flatMap { _patient_id, reference_item, all_items ->
            toPairwiseTuples(reference_item, all_items)
        }

    CPU_REGISTER(ch_pairs)

    ch_references = ch_grouped_meta
        .map { _patient_id, reference_item, _all_items ->
            referenceTuple(reference_item)
        }

    PUBLISH_REFERENCE_CPU(ch_references)

    ch_all = PUBLISH_REFERENCE_CPU.out.published.concat(CPU_REGISTER.out.registered)
    ch_size_logs = CPU_REGISTER.out.size_log

    emit:
    registered = ch_all
    size_logs = ch_size_logs
}

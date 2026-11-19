/*
================================================================================
    SUBWORKFLOW: FINAL_QC
================================================================================
    End-of-run aggregation, shared by the linear path and add_cycle. Each output
    has its own param gate:
      * the HTML QC report          (skipped by --skip_final_qc_report)
      * the input-size log rollup   (only under --enable_trace)

    Artifacts arrive as one stream of `[kind, file]` pairs; a kind nobody
    contributes yields an empty branch. (add_cycle contributes no preprocess_qc,
    registration_tre or postprocess_qc.)

    Recognised kinds:
      preprocess_qc | registration_qc | registration_tre | postprocess_qc | seg_qc
        | seg_residuals
        -> staged into the matching GENERATE_QC_REPORT input directory
      versions   -> deduplicated and collated into collated_versions.yml
      size_log   -> merged into raw_input_sizes.csv for AGGREGATE_SIZE_LOGS

    An unknown kind is an error (ParamUtils.knownArtifactKinds()): it would match
    no filter and drop that QC category from the report silently.

    `ch_run_facts` carries what differs between callers (the `stop` label and the
    manifest totals); the rest of run_summary.json is derived here.

    The manifest reads `facts.declared_channels` (the samplesheet's union), never
    `facts.channels` (channels_count after the nuclear drop, for groupKey sizing).
================================================================================
*/

// Strict-syntax Nextflow has no Groovy imports, so JsonOutput is spelled out in full below.

include { AGGREGATE_SIZE_LOGS } from '../../modules/local/aggregate_size_logs'
include { GENERATE_QC_REPORT  } from '../../modules/local/generate_qc_report'

// The artifact vocabulary is ParamUtils.knownArtifactKinds(), derived from STEPS' qcKinds
// plus UNIVERSAL_QC_KINDS. registration_tre is the method's own TRE estimate, in its own format.

// Pull one kind out of the tagged artifact stream (channels are broadcast, so repeat use is fine).
// Rejects an unknown `kind` literal, which would otherwise empty a report slot silently.
// `consumed` records each kind read, so FINAL_QC can fail on a declared kind nobody reads;
// it is a required argument so the check cannot be skipped.
def artifactsOf(ch_artifacts, String kind, Set consumed) {
    def known = ParamUtils.knownArtifactKinds()
    if (!(kind in known))
        throw new IllegalArgumentException("FINAL_QC: artifactsOf('${kind}') is not a known artifact kind: ${known.join(', ')}")
    consumed << kind
    return ch_artifacts.filter { it[0] == kind }.map { it[1] }
}

// The `manifest` block of run_summary.json: total and per-patient image/channel counts,
// from facts.declared_channels (see the header). Tested by final_qc_run_summary.nf.test.
def buildManifest(Map facts) {
    def patient_counts    = facts.patients
    def declared_channels = facts.declared_channels
    return [
        totals: [
            patients: patient_counts.size(),
            images: (patient_counts.values().sum() ?: 0),
            channels: (declared_channels.values().sum() ?: 0),
        ],
        patients: patient_counts.collectEntries { pid, imgs ->
            [(pid): [images: imgs, channels: (declared_channels[pid] ?: 0)]]
        },
    ]
}

// The run_summary.json content as a Map. Params, pipeline info and timestamp are arguments
// so an nf-test `nextflow_function` can call it without a live binding.
def buildRunSummary(Map facts, Map runParams, Map pipelineInfo, String timestamp) {
    return [
        pipeline: pipelineInfo,
        run: [
            timestamp: timestamp,
            mode: runParams.mode,
            start: runParams.start,
            stop: facts.stop,
        ],
        params: [
            registration_method: runParams.registration_method,
            seg_method: runParams.seg_method,
            quantify_compartments: runParams.quantify_compartments,
            expanded_quantification: runParams.expanded_quantification,
            pixel_size: runParams.pixel_size,
        ],
        manifest: buildManifest(facts),
    ]
}

workflow FINAL_QC {
    take:
    ch_artifacts   // [kind, file]  — see the kind list above
    ch_run_facts   // value channel of [stop: String, patients: Map, channels: Map,
                   //   declared_channels: Map]; the Maps are INPUT_CHECK.out.counts.
                   //   `channels` is unread here.

    main:

    // A fresh local list, so no other run in the JVM can mutate it.
    def known_kinds = ParamUtils.knownArtifactKinds()

    // Kinds some artifactsOf() call consumes; checked against known_kinds at the end.
    def consumed_kinds = [] as Set

    // Unknown-tag guard, outside both param gates so a typo fails every run.
    ch_artifacts.subscribe { item ->
        if (!(item[0] in known_kinds)) {
            error "FINAL_QC: unknown artifact kind '${item[0]}' for ${item[1]}. " +
                  "Known kinds: ${known_kinds.join(', ')}. An unknown tag matches no " +
                  "artifactsOf() filter, so that whole QC category would be dropped from the " +
                  "report without failing the run — fix the tag at the FINAL_QC call site in " +
                  "workflows/mirage.nf, or add the kind here and wire it to a report slot."
        }
    }

    if (!params.skip_final_qc_report) {
        // Run summary for the report's overview card, from INPUT_CHECK's counts.
        ch_run_summary = ch_run_facts
            .map { facts ->
                def timestamp = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss 'UTC'")
                    .format(new Date())
                def summary = buildRunSummary(
                    facts,
                    params,
                    [name: workflow.manifest.name, version: workflow.manifest.version],
                    timestamp
                )
                return groovy.json.JsonOutput.prettyPrint(
                    groovy.json.JsonOutput.toJson(summary))
            }
            .collectFile(name: 'run_summary.json')

        // Eight fixed slots in this order (nf-test snapshots pin them); a new kind needs a
        // new slot here and a new path() in modules/local/generate_qc_report.nf.
        // collect(sort: true), not collect(): path lists hash positionally, so arrival order
        // breaks -resume. collated_versions.yml is sorted for stable content.
        GENERATE_QC_REPORT(
            artifactsOf(ch_artifacts, 'preprocess_qc', consumed_kinds).collect(sort: true).ifEmpty([]),
            artifactsOf(ch_artifacts, 'registration_qc', consumed_kinds).collect(sort: true).ifEmpty([]),
            artifactsOf(ch_artifacts, 'registration_tre', consumed_kinds).collect(sort: true).ifEmpty([]),
            artifactsOf(ch_artifacts, 'postprocess_qc', consumed_kinds).collect(sort: true).ifEmpty([]),
            artifactsOf(ch_artifacts, 'versions', consumed_kinds).unique().collectFile(name: 'collated_versions.yml', sort: true),
            ch_run_summary,
            artifactsOf(ch_artifacts, 'seg_qc', consumed_kinds).collect(sort: true).ifEmpty([]),
            artifactsOf(ch_artifacts, 'seg_residuals', consumed_kinds).collect(sort: true).ifEmpty([]),
        )
    }

    if (params.enable_trace) {
        // Merge by content: several processes emit same-named *.size.csv logs.
        AGGREGATE_SIZE_LOGS(
            artifactsOf(ch_artifacts, 'size_log', consumed_kinds).collectFile(name: 'raw_input_sizes.csv', sort: true)
        )
    }

    // Every declared kind must have a consumer. Checked only with both outputs enabled,
    // since disabling one legitimately leaves its kinds unread.
    if (!params.skip_final_qc_report && params.enable_trace) {
        def unconsumed = known_kinds.toSet() - consumed_kinds
        if (unconsumed) {
            error "FINAL_QC: artifact kind(s) ${unconsumed as List} are declared but no " +
                  "artifactsOf() call consumes them, so anything tagged with them is dropped " +
                  "from every report while the run stays green. Wire each to a report slot in " +
                  "this file, or remove it from ParamUtils.STEPS' qcKinds / UNIVERSAL_QC_KINDS."
        }
    }

    // The versions of these two processes are not collected: they run after the collation.
    // AGGREGATE_SIZE_LOGS still publishes its own size_logs/versions.yml.
}

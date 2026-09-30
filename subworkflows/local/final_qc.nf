/*
================================================================================
    SUBWORKFLOW: FINAL_QC
================================================================================
    The pipeline's end-of-run aggregation, shared by the linear path and by
    add_cycle. Both used to carry their own copy of it — the add_cycle copy was
    introduced with the comment "Mirrors the standard-path aggregation below",
    and "mirrors" is exactly the property that rots.

    It owns two things, both gated by their own param so the router needs no `if`:
      * the HTML QC report          (skipped by --skip_final_qc_report)
      * the input-size log rollup   (only under --enable_trace)

    ---------------------------------------------------------------------------
    Interface
    ---------------------------------------------------------------------------
    Artifacts arrive as ONE stream of `[kind, file]` pairs rather than as eight
    positional channels. A caller mixes in what it has; a kind nobody contributes
    simply yields an empty branch here, which is what removes the
    `Channel.empty().collect().ifEmpty([])` placeholders both callers used to
    repeat. (add_cycle contributes no preprocess_qc / registration_tre /
    postprocess_qc: it calls PREPROCESSING internally without re-exposing its QC
    pngs, and it has no POSTPROCESSING step at all — masks are reused, not
    re-segmented. It DOES contribute seg_residuals, from SEG_QC.out.per_cell,
    same as the linear path — see subworkflows/local/add_cycle.nf.)

    Recognised kinds:
      preprocess_qc | registration_qc | registration_tre | postprocess_qc | seg_qc
        | seg_residuals
        -> staged into the matching GENERATE_QC_REPORT input directory
      versions   -> deduplicated and collated into collated_versions.yml
      size_log   -> merged into raw_input_sizes.csv for AGGREGATE_SIZE_LOGS

    An unrecognised kind is a hard ERROR (see ParamUtils.knownArtifactKinds()). Every
    consumer here is an `artifactsOf(...)` filter, so an unknown tag would match no
    filter and that whole QC category would vanish from the report while the run
    stayed green — under the previous positional signature the same typo was a
    parse-time resolution failure, and this guard is what keeps it loud.

    `ch_run_facts` carries the only run-summary inputs that genuinely differ
    between the two callers — the `stop` label and the sample manifest totals.
    Everything else in run_summary.json (pipeline name/version, timestamp,
    params.mode/start, the params block) is derived here, so the two paths cannot
    drift apart in a field neither of them meant to change. Nothing in this file
    branches on params.mode.

    `facts.channels` and `facts.declared_channels` are NOT interchangeable, and the
    manifest built below reads ONLY `declared_channels`: `channels` is
    channels_count's exact, post-nuclear-drop count (correct for sizing groupKey,
    wrong for an input manifest); `declared_channels` is the samplesheet's raw
    union (right for the manifest, wrong for groupKey — see
    CsvUtils.countDeclaredChannelsPerPatient / countChannelsPerPatient). Reading
    `channels` here instead is the exact regression this file's manifest carried
    for a while: add_cycle reported 2 channels for a declared 3-channel cycle.
================================================================================
*/

// No `import groovy.json.JsonOutput` here: Nextflow 26's strict parser has no
// Groovy import declaration at all (`include` is the only import form it knows),
// so JsonOutput is spelled out in full at its single use site in `main:` below.

include { AGGREGATE_SIZE_LOGS } from '../../modules/local/aggregate_size_logs'
include { GENERATE_QC_REPORT  } from '../../modules/local/generate_qc_report'

// The artifact vocabulary lives in ParamUtils.knownArtifactKinds()
// (lib/ParamUtils.groovy), derived from the STEPS table's per-step `qcKinds` plus
// UNIVERSAL_QC_KINDS. It used to be a bare top-level `KNOWN_ARTIFACT_KINDS = ...`
// here, with a comment explaining that it could not be a `def` because a
// script-level `def` is scoped to this file's implicit run() method and would be
// invisible to the functions below. Both halves of that bind are real, and
// Nextflow 26's strict parser rejects the bare assignment outright -- so the fix
// is to stop needing file-level state: a static accessor on ParamUtils is visible
// from the functions here, from FINAL_QC's `main:`, and from an nf-test
// `nextflow_function` test, with no script binding involved.
//
// The registration_tre entry is the registration method's OWN target-registration-error
// estimate. NOT named after a method: both backends produce one (VALIS a feature-distance
// CSV, STARE a *_tre.json) and a backend that produced none would contribute nothing, which
// the .collect().ifEmpty([]) below already tolerates.

// Pull one kind out of the tagged artifact stream. Nextflow channels are
// broadcast, so applying this repeatedly to the same source is fine.
//
// Validates `kind` itself, not just the tags flowing through `ch_artifacts`
// (that's the subscribe guard below): the eight artifactsOf() call sites in
// this file's `main:` are a second, textually separate copy of the vocabulary,
// and a typo there (e.g. 'seg_qcc') would match nothing and silently empty
// that report slot while every test stayed green -- the same failure class
// the subscribe guard exists to catch, from the other side.
//
// `consumed` is the third, mandatory argument rather than file-level state, and
// it is mandatory on purpose. It records which kinds some artifactsOf() call in
// this file actually reads, and FINAL_QC's `main:` checks it against the full
// vocabulary at the end (see the "Every declared kind must have a consumer"
// block). That guard closes the third way this seam breaks: a kind that IS
// declared in ParamUtils.STEPS' qcKinds, IS correctly tagged at the call site,
// and is read by NO artifactsOf() call -- silently dropped from the report while
// the run exits 0. The subscribe guard below catches an unknown TAG at the
// producer and the check here catches an unknown FILTER literal; neither watches
// that direction.
//
// A default value for `consumed` would make the guard opt-in and therefore
// forgettable, and a static mutable Set in lib/ would be shared across every
// run in the JVM. Threading it explicitly is what keeps "add an artifactsOf()
// call" and "that kind counts as consumed" the same edit.
def artifactsOf(ch_artifacts, String kind, Set consumed) {
    def known = ParamUtils.knownArtifactKinds()
    if (!(kind in known))
        throw new IllegalArgumentException("FINAL_QC: artifactsOf('${kind}') is not a known artifact kind: ${known.join(', ')}")
    consumed << kind
    return ch_artifacts.filter { it[0] == kind }.map { it[1] }
}

// Build the `manifest` block of run_summary.json: total and per-patient
// image/channel counts for the sample manifest. Pulled into its own function so it
// is unit-testable via nf-test's `nextflow_function`
// (final_qc_run_summary.nf.test) — GENERATE_QC_REPORT's stub never copies
// run_summary.json anywhere a workflow test could inspect its content, so before
// this the whole run_summary.json build was unverifiable by anything but a manual
// diff in a report.
//
// `facts.declared_channels` (NOT `facts.channels`) feeds this. See the header
// comment above and CsvUtils.countDeclaredChannelsPerPatient's doc for why the
// manifest and channels_count must not share a source.
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

// Build the complete run_summary.json content, as a Map (pre-JSON-encoding).
// params/pipeline-info/timestamp are explicit arguments — rather than this function
// reading params./workflow.manifest directly — purely so it is callable from an
// isolated nf-test `nextflow_function` test with no live params{}/workflow binding
// required. The live call site below (FINAL_QC's main:) passes the real
// params/workflow.manifest/timestamp; production behaviour is unchanged by this.
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
                   //   declared_channels: Map] — patients/channels/declared_channels
                   //   are INPUT_CHECK.out.counts. `channels` is NOT read by the
                   //   manifest below (see header comment) — it is deliberately unread
                   //   here. It is only present because this Map is INPUT_CHECK.out.counts
                   //   passed wholesale, not because some other consumer needs it.

    main:

    // The run's artifact vocabulary, read once. A local, not file-level state:
    // ParamUtils.knownArtifactKinds() returns a fresh list each call, so this
    // binding cannot be mutated for any other run sharing the JVM.
    def known_kinds = ParamUtils.knownArtifactKinds()

    // Kinds that some artifactsOf() call below actually consumes. Threaded into
    // every artifactsOf() call and checked against `known_kinds` at the end of
    // this workflow — see the comment on artifactsOf() above for what that
    // catches and why the Set is passed rather than held at file level.
    def consumed_kinds = [] as Set

    // Fail-fast vocabulary guard. Registered unconditionally (not inside either
    // param gate) so a typo is caught even on a run that asks for neither output.
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
        // Run-context summary for the report's overview card (pipeline, run,
        // params, sample manifest). Built from the counts INPUT_CHECK already
        // computed rather than by re-parsing the samplesheet.
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

        // GENERATE_QC_REPORT's input arity and order are fixed (nf-test snapshots
        // pin them) — this call must keep all EIGHT slots in this order. Nextflow
        // process inputs are statically declared and fixed-arity, so a new artifact
        // kind necessarily means a new slot here and a new `path(...)` in
        // modules/local/generate_qc_report.nf. That is a hard error when missed,
        // which is why the arity is not data-driven.
        // Every list slot below is collect(sort: true), never a bare collect().
        // GENERATE_QC_REPORT's inputs are `path` collections that Nextflow hashes
        // POSITIONALLY, and collect() emits in arrival order -- so identical reruns
        // produced different task hashes. collated_versions.yml carries sort: true for
        // the same reason at the level of file CONTENT: without it the collected yaml's
        // line order varied by completion order.
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
        // Merge by content (not by staging many same-named files) so AGGREGATE
        // receives a single file and cannot hit a work-dir name collision —
        // several processes emit identically-named *.size.csv logs.
        AGGREGATE_SIZE_LOGS(
            artifactsOf(ch_artifacts, 'size_log', consumed_kinds).collectFile(name: 'raw_input_sizes.csv', sort: true)
        )
    }

    // Every declared kind must have a consumer. See artifactsOf() above.
    //
    // Registered only when both outputs are enabled: skip_final_qc_report set to true,
    // or enable_trace set to false, legitimately leaves that half of the vocabulary
    // unconsumed, and failing then would break a supported run mode. The guard is
    // therefore about WIRING, checked on the default path, not about run policy.
    if (!params.skip_final_qc_report && params.enable_trace) {
        def unconsumed = known_kinds.toSet() - consumed_kinds
        if (unconsumed) {
            error "FINAL_QC: artifact kind(s) ${unconsumed as List} are declared but no " +
                  "artifactsOf() call consumes them, so anything tagged with them is dropped " +
                  "from every report while the run stays green. Wire each to a report slot in " +
                  "this file, or remove it from ParamUtils.STEPS' qcKinds / UNIVERSAL_QC_KINDS."
        }
    }

    // Neither GENERATE_QC_REPORT.out.versions nor AGGREGATE_SIZE_LOGS.out.versions is
    // consumed here, and neither was consumed by the router before this subworkflow
    // existed: both processes run AFTER the version collation they would belong to.
    // Left as-is to keep this extraction behaviour-preserving.
    //
    // The two are NOT equally invisible, though. GENERATE_QC_REPORT's publishDir
    // (conf/modules.config:139-145) drops versions.yml via saveAs; AGGREGATE_SIZE_LOGS'
    // (conf/modules.config:540-545) does NOT, so <outdir>/size_logs/versions.yml is a
    // published artifact whose single key is the fully-qualified process name — moving
    // this call in here changed it to "MIRAGE:FINAL_QC:AGGREGATE_SIZE_LOGS". See the
    // comment on that publishDir block.
}

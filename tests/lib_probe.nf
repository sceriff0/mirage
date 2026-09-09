/*
========================================================================================
    lib/ PROBE — the unit-test surface for lib/*.groovy
========================================================================================
    WHY THIS IS A .nf FILE AND NOT AN nf-test ASSERTION BLOCK.

    nf-test's assertion context is a separate Groovy shell that does NOT have the
    project's lib/ directory on its classpath: naming `Layout` there fails with
    `No such property: Layout` (see the header of tests/layout.nf.test). A
    Nextflow PIPELINE SCRIPT, by contrast, is compiled WITH lib/ on the
    classpath — so the assertions live here, in a script, and
    tests/lib_probe.nf.test only has to assert that this script ran.

    Run directly:   nextflow run tests/lib_probe.nf -lib lib
    Run in CI:      the nextflow-stub job, and tests/lib_probe.nf.test (tag: stub)

    This file declares no processes and touches no filesystem. Every statement is
    a pure call into a lib/ class plus an `assert`. A failing assert aborts the
    run with a nonzero exit — that is the whole mechanism.

    APPENDING: later tasks add their own assert blocks under a clearly commented
    section header, following the pattern below. Keep each class's assertions in
    its own banner-delimited section.

    THE workflow{} BLOCK HAS A HARD SIZE CEILING. Nextflow's DSL2 parser captures
    a top-level workflow{} block's body whole, as a single Groovy string constant,
    and the JVM class-file format caps any one String constant at 65535 UTF-16
    code units (org.codehaus.groovy.classgen.ClassCompletionVerifier
    #checkStringExceedingMaximumLength) -- confirmed empirically 2026-09-02: at
    60e5a2e the workflow{} block was already 64072 bytes, ~1.4KB below that
    ceiling, and appending a normally-commented section pushed it over with
    "String too long. The given string is N Unicode code units long...". A
    top-level `def` FUNCTION is not subject to this capture -- only the workflow{}
    block's own text counts against the limit. So a new section with any real
    amount of prose goes in its own top-level `def checkX() { ... }` function
    below, called as ONE statement from inside workflow{}; only add assertions
    directly inline in workflow{} if they are genuinely a one- or two-liner.
========================================================================================
*/

/**
 * ResourceReport — trace-path resolution and the missing-trace diagnostic.
 * Kept as a top-level function (see the header note above) so its assertions can
 * carry normal comments without threatening the workflow{} block's size ceiling.
 */
def checkResourceReport() {

    // ------------------------------------------------------------------ //
    // ResourceReport — where the trace file is looked for
    // ------------------------------------------------------------------ //
    // Nextflow resolves trace.file (nextflow.config's "${params.trace_dir}/trace.txt")
    // against launchDir. main.nf's onComplete handler used to hand the RAW param to
    // cmd.execute(), which resolves against the JVM's working directory instead --
    // the same thing only when the pipeline happened to be launched from the
    // directory it runs in. These four assertions pin the resolution rule.
    assert ResourceReport.tracePath('/abs/trace', '/launch')  == '/abs/trace/trace.txt'
    assert ResourceReport.tracePath('.trace', '/launch')      == '/launch/.trace/trace.txt'
    assert ResourceReport.tracePath('./.trace/', '/launch')   == '/launch/.trace/trace.txt'
    assert ResourceReport.tracePath('a/b', '/launch/')        == '/launch/a/b/trace.txt'

    // A blank trace_dir must throw rather than produce a literal 'null/trace.txt'.
    def blankTraceDir = false
    try { ResourceReport.tracePath('  ', '/launch') }
    catch (IllegalArgumentException ignored) { blankTraceDir = true }
    assert blankTraceDir : 'ResourceReport.tracePath must reject a blank trace_dir'

    // A relative trace_dir with no launchDir cannot be resolved and must say so,
    // rather than silently returning the relative path it was given.
    def blankLaunchDir = false
    try { ResourceReport.tracePath('.trace', '') }
    catch (IllegalArgumentException ignored) { blankLaunchDir = true }
    assert blankLaunchDir : 'ResourceReport.tracePath must reject a relative trace_dir with no launchDir'

    // The diagnostic must NAME the path and state whether tracing was on. A message
    // that says only "could not generate resource report" leaves an operator with no
    // way to tell a misconfigured trace_dir from a deliberately disabled trace.
    def traceOn  = ResourceReport.missingTraceMessage('/launch/.trace/trace.txt', true)
    def traceOff = ResourceReport.missingTraceMessage('/launch/.trace/trace.txt', false)

    assert traceOn.contains('/launch/.trace/trace.txt')  : 'the ON message must name the path'
    assert traceOff.contains('/launch/.trace/trace.txt') : 'the OFF message must name the path'
    assert traceOn  != traceOff : 'the two cases must be distinguishable'
    assert traceOn.contains('enable_trace')  : 'the ON message must name the parameter'
    assert traceOff.contains('enable_trace') : 'the OFF message must name the parameter'
    // No CLI flag form anywhere: Nextflow 26 delivers every --param as a String, so
    // a message telling an operator to pass one would be advice that cannot work.
    assert !traceOn.contains('--enable_trace')  : 'must not suggest a CLI boolean'
    assert !traceOff.contains('--enable_trace') : 'must not suggest a CLI boolean'
}

/**
 * ProcessEnvelope — the size-log row and container identity in versions.yml.
 *
 * Lives outside `workflow { ... }` deliberately: see "THE WORKFLOW BLOCK HAS A HARD
 * SIZE CEILING" above. Called once from the workflow block, before the final println.
 */
def checkProcessEnvelope() {
    // ------------------------------------------------------------------ //
    // ProcessEnvelope — the size-log row
    // ------------------------------------------------------------------ //
    // The rendered strings are asserted BYTE FOR BYTE, because they are shell
    // text that no `-stub` run and no snapshot ever evaluates: `-stub` skips
    // script: entirely, and the aggregated CSV records only the row's VALUES.
    // If the `$(`/`${` escaping is wrong here, .command.sh gets the literal
    // command text instead of a byte count and nothing downstream notices --
    // the exact failure that shipped once in versions() (see probe()'s comment).
    assert ProcessEnvelope.SIZE_LOG_COLUMNS == ['process', 'sample_id', 'filename', 'bytes']

    def oneFile = ProcessEnvelope.sizeLog('MIRAGE:PRE:CONVERT_IMAGE', 'P001',
                                          ['P001_slide.ome.tiff'], 'P001_slide.CONVERT_IMAGE.size.csv')
    assert oneFile == 'size_bytes=$(stat -L --printf="%s\\n" P001_slide.ome.tiff 2>/dev/null | awk \'{s+=$1} END {print s+0}\')\n' +
                      'echo "MIRAGE:PRE:CONVERT_IMAGE,P001,P001_slide.ome.tiff,${size_bytes}" > P001_slide.CONVERT_IMAGE.size.csv'

    // A staged path keeps only its BASENAME in the filename column -- `mov/x.ome.tiff`
    // is stageAs positioning, not part of the file's identity.
    def staged = ProcessEnvelope.sizeLog('MIRAGE:REG:TILED_STITCH', 'P001',
                                         ['mov/x.ome.tiff'], 'P001.TILED_STITCH.size.csv')
    assert staged.contains('MIRAGE:REG:TILED_STITCH,P001,x.ome.tiff,${size_bytes}')
    assert staged.contains('stat -L --printf="%s\\n" mov/x.ome.tiff ')

    // More than one path -> the literal `inputs/`, and every path is stat'ed in ONE call.
    def many = ProcessEnvelope.sizeLog('MIRAGE:REG:REGISTER', 'P001',
                                       ['ref/*', 'input_*/*'], 'P001.REGISTER.size.csv')
    assert many.contains('stat -L --printf="%s\\n" ref/* input_*/* 2>/dev/null')
    assert many.contains('MIRAGE:REG:REGISTER,P001,inputs/,${size_bytes}')

    // A SOLE path that is a glob is `inputs/` too: a glob is by construction more
    // than one path, so naming it after its own wildcard ('*') would be a lie.
    def glob = ProcessEnvelope.sizeLog('MIRAGE:POST:MERGE_AND_PYRAMID', 'P001',
                                       ['channels/*'], 'P001.MERGE_AND_PYRAMID.size.csv')
    assert glob.contains('MIRAGE:POST:MERGE_AND_PYRAMID,P001,inputs/,${size_bytes}')

    // The stub row carries the SAME process name as the real row. That is the whole
    // point: `STUB` as a process name made a stub CSV structurally incomparable with
    // a real one, and bin/generate_resource_report.py had to special-case it.
    assert ProcessEnvelope.sizeLogStub('MIRAGE:PRE:CONVERT_IMAGE', 'P001', 'P001_slide.CONVERT_IMAGE.size.csv') ==
           'echo "MIRAGE:PRE:CONVERT_IMAGE,P001,stub,0" > P001_slide.CONVERT_IMAGE.size.csv'

    // An empty path list is a caller bug, not an empty measurement: `stat` with no
    // operands would report 0 bytes for a task that certainly read something.
    def noPaths = false
    try { ProcessEnvelope.sizeLog('P', 'S', [], 'x.size.csv') }
    catch (IllegalArgumentException ignored) { noPaths = true }
    assert noPaths : 'ProcessEnvelope.sizeLog must reject an empty shellPaths list'

    // ------------------------------------------------------------------ //
    // ProcessEnvelope — container identity in versions.yml
    // ------------------------------------------------------------------ //
    // WHY versions.yml AND NOT THE TRACE. The trace records what Nextflow was told
    // to run; versions.yml is what the RUN ITSELF reports, and it is the artifact
    // that ships with the results. Ruling R6 makes the image a first-class part of
    // 1.0.0's reproducibility claim, and an image tag in this repo is descriptive
    // (:preprocess, :tiled), not immutable -- so "which image produced this" is a
    // question the outputs must answer for themselves.
    def withContainer = ProcessEnvelope.versions('MIRAGE:POST:MERGE_AND_PYRAMID',
                                                 ['tifffile'], 'bolt3x/mirage-merge:1.0.0')
    def containerLines = withContainer.readLines()
    assert containerLines[0] == 'cat <<-END_VERSIONS > versions.yml'
    assert containerLines[1] == '"MIRAGE:POST:MERGE_AND_PYRAMID":'
    assert containerLines[2].startsWith('    python: $(python --version')
    // Immediately after python:, before the tool probes -- the position is asserted,
    // not just the presence, because bin/generate_qc_report.py renders the block in
    // file order into a published table.
    assert containerLines[3] == '    container: bolt3x/mirage-merge:1.0.0'
    assert containerLines[4].startsWith('    tifffile: $(python -c ')

    // A null container renders NO line: an absent container must not become the
    // string 'null' in a published report.
    assert !ProcessEnvelope.versions('P', ['tifffile'], null).contains('container:')
    assert !ProcessEnvelope.versions('P', ['tifffile']).contains('container:')

    // The stub reports `stub`, never the real image name: a stub run produced no
    // evidence about any image, and claiming one would be a fabricated provenance
    // record in exactly the artifact that exists to carry provenance.
    def stubContainer = ProcessEnvelope.versionsStub('MIRAGE:POST:MERGE_AND_PYRAMID',
                                                     ['tifffile'], 'bolt3x/mirage-merge:1.0.0')
    assert stubContainer.readLines()[3] == '    container: stub'
    assert !stubContainer.contains('bolt3x')

    // The bash-only pair, for a module with no Python interpreter at all.
    // AGGREGATE_SIZE_LOGS runs in bolt3x/mirage-preprocess:1.0.0; routing it through
    // versions() would run `python --version 2>&1` and write the SHELL'S ERROR MESSAGE
    // ("bash: python: command not found") into a published report as a version
    // number, because that heredoc pipes stderr into the value.
    def bashVersions = ProcessEnvelope.versionsBash('MIRAGE:FINAL_QC:AGGREGATE_SIZE_LOGS', 'bolt3x/mirage-preprocess:1.0.0')
    assert bashVersions.readLines() == [
        'cat <<-END_VERSIONS > versions.yml',
        '"MIRAGE:FINAL_QC:AGGREGATE_SIZE_LOGS":',
        '    bash: $(bash --version | head -n1 | sed \'s/GNU bash, version //\')',
        '    container: bolt3x/mirage-preprocess:1.0.0',
        'END_VERSIONS',
    ]
    assert !bashVersions.contains('python:')
    def bashStub = ProcessEnvelope.versionsBashStub('MIRAGE:FINAL_QC:AGGREGATE_SIZE_LOGS', 'bolt3x/mirage-preprocess:1.0.0')
    assert bashStub.readLines() == [
        'cat <<-END_VERSIONS > versions.yml',
        '"MIRAGE:FINAL_QC:AGGREGATE_SIZE_LOGS":',
        '    bash: stub',
        '    container: stub',
        'END_VERSIONS',
    ]
}

/**
 * RegisteredMatch — pairing VALIS's outputs back to their slide metas.
 *
 * Lives outside `workflow { ... }` deliberately: see "THE WORKFLOW BLOCK HAS A HARD
 * SIZE CEILING" above. Called once from the workflow block, before the final println.
 */
def checkRegisteredMatch() {
    // ------------------------------------------------------------------ //
    // RegisteredMatch — pairing VALIS's outputs back to their slide metas
    // ------------------------------------------------------------------ //
    // WHY BY CHANNEL SIGNATURE AND NOT BY FILENAME. VALIS renames its outputs; the
    // only thing carried through registration that identifies a slide is its OME
    // channel set, which bin/create_channels_manifest.py writes out as
    // filename -> [channel names]. Getting this wrong does not fail -- it silently
    // attaches the wrong patient's metadata to a registered image.
    assert RegisteredMatch.signature(['PANCK', 'dapi', 'CD3']) == 'cd3|dapi|panck'
    // Lower-case FIRST, then sort: 'CD3' and 'cd3' must land in the same place.
    assert RegisteredMatch.signature(['cd3', 'DAPI']) == RegisteredMatch.signature(['CD3', 'dapi'])

    // toSorted(), never sort(): meta.channels is a SHARED List reference across
    // every meta built with `meta + [k: v]` (Map.plus is a SHALLOW clone), so an
    // in-place sort here would silently reorder a sibling meta's channel list --
    // and channel ORDER is what --dapi-channel and the pyramid writer index on.
    def sharedChannels = ['PANCK', 'DAPI']
    RegisteredMatch.signature(sharedChannels)
    assert sharedChannels == ['PANCK', 'DAPI'] : 'RegisteredMatch.signature must not mutate its argument'

    def refMeta = [patient_id: 'P1', id: 'P1_ref', channels: ['DAPI', 'PANCK', 'SMA']]
    def movMeta = [patient_id: 'P1', id: 'P1_mov', channels: ['DAPI', 'CD3']]
    def refFile = file('/tmp/P1_DAPI_PANCK_SMA_registered.ome.tiff')
    def movFile = file('/tmp/P1_DAPI_CD3_registered.ome.tiff')
    def manifest = [
        'P1_DAPI_PANCK_SMA_registered.ome.tiff': ['DAPI', 'PANCK', 'SMA'],
        'P1_DAPI_CD3_registered.ome.tiff'      : ['CD3', 'DAPI'],
    ]

    // Files deliberately in the OPPOSITE order to metas: the pairing is by
    // signature, and the RESULT is in metas order.
    def paired = RegisteredMatch.pair([refMeta, movMeta], [movFile, refFile], manifest)
    assert paired.size() == 2
    assert paired[0][0].id == 'P1_ref'
    assert paired[0][1].name == 'P1_DAPI_PANCK_SMA_registered.ome.tiff'
    assert paired[1][0].id == 'P1_mov'
    assert paired[1][1].name == 'P1_DAPI_CD3_registered.ome.tiff'

    // The manifest's channel ORDER is irrelevant -- it comes from OME-XML, the meta's
    // comes from the samplesheet, and neither is authoritative over the other.
    def reordered = RegisteredMatch.pair([movMeta], [movFile],
                                         ['P1_DAPI_CD3_registered.ome.tiff': ['DAPI', 'CD3']])
    assert reordered[0][1].name == 'P1_DAPI_CD3_registered.ome.tiff'

    // Exception 1: count mismatch. VALIS returning fewer files than slides is a
    // partial failure, and pairing what arrived would publish a short run as a
    // complete one.
    def countMismatch = ''
    try { RegisteredMatch.pair([refMeta, movMeta], [refFile], manifest) }
    catch (IllegalStateException e) { countMismatch = e.message }
    assert countMismatch.startsWith('RegisteredMatch: count mismatch') : "got: ${countMismatch}"

    // Exception 1b: count mismatch where one meta's channels is null. Building the
    // diagnostic itself must not throw IllegalArgumentException from signature() --
    // that would replace "count mismatch" with a confusing "channels is null".
    def nullChannelsMeta = [patient_id: 'P1', id: 'P1_null', channels: null]
    def countMismatchNullChannels = ''
    try { RegisteredMatch.pair([refMeta, nullChannelsMeta], [refFile], manifest) }
    catch (IllegalStateException e) { countMismatchNullChannels = e.message }
    assert countMismatchNullChannels.startsWith('RegisteredMatch: count mismatch') : "got: ${countMismatchNullChannels}"

    // Exception 2: duplicate signature. Two slides with the same channel set are
    // indistinguishable to this rule, and a map keyed on the signature would
    // silently keep only the last one.
    def dupMeta = [patient_id: 'P1', id: 'P1_dup', channels: ['PANCK', 'DAPI', 'SMA']]
    def dupSig = ''
    try { RegisteredMatch.pair([refMeta, dupMeta], [refFile, movFile], manifest) }
    catch (IllegalStateException e) { dupSig = e.message }
    assert dupSig.startsWith('RegisteredMatch: duplicate signature') : "got: ${dupSig}"

    // Exception 3a: a registered file the manifest says nothing about.
    def noManifestEntry = ''
    try { RegisteredMatch.pair([refMeta], [refFile], [:]) }
    catch (IllegalStateException e) { noManifestEntry = e.message }
    assert noManifestEntry.startsWith('RegisteredMatch: unmatched') : "got: ${noManifestEntry}"
    // Distinguish 3a from 3b by message substring, not merely by shared prefix -- see
    // lib/RegisteredMatch.groovy's two distinct 'unmatched' throw sites.
    assert noManifestEntry.contains('the channels manifest has no entry for') : "3a message: ${noManifestEntry}"

    // Exception 3b: a meta whose channel set no registered file carries.
    def noFileForMeta = ''
    try {
        RegisteredMatch.pair([refMeta], [refFile],
                             ['P1_DAPI_PANCK_SMA_registered.ome.tiff': ['CD8', 'FOXP3']])
    }
    catch (IllegalStateException e) { noFileForMeta = e.message }
    assert noFileForMeta.startsWith('RegisteredMatch: unmatched') : "got: ${noFileForMeta}"
    assert noFileForMeta.contains('no registered file carries the channel') : "3b message: ${noFileForMeta}"

    // Exception 3c: a duplicate FILE signature (as opposed to 3b's duplicate META
    // signature, already rejected earlier). refMeta and movMeta have DISTINCT
    // signatures, so the meta-side duplicate check does not fire -- but both
    // manifest entries below name refMeta's channel set, so fileBySignature
    // collapses them to one map slot and movFile's entry is overwritten. This is
    // the third behaviour change documented in lib/RegisteredMatch.groovy's
    // docstring: the old closure would have silently paired the surviving file to
    // both metas, whereas here movMeta's signature has no file left and the
    // "unmatched" throw below is fatal.
    def dupFileSig = ''
    try {
        RegisteredMatch.pair([refMeta, movMeta], [refFile, movFile],
                             ['P1_DAPI_PANCK_SMA_registered.ome.tiff': ['DAPI', 'PANCK', 'SMA'],
                              'P1_DAPI_CD3_registered.ome.tiff'      : ['DAPI', 'PANCK', 'SMA']])
    }
    catch (IllegalStateException e) { dupFileSig = e.message }
    assert dupFileSig.startsWith('RegisteredMatch: unmatched') : "got: ${dupFileSig}"
}

/**
 * Checkpoint — requireColumns, the drift guard three readers copied.
 *
 * Lives outside `workflow { ... }` deliberately: see "THE WORKFLOW BLOCK HAS A HARD
 * SIZE CEILING" above. Called once from the workflow block, before the final println.
 */
def checkCheckpoint() {
    // ------------------------------------------------------------------ //
    // Checkpoint — requireColumns, the drift guard three readers copied
    // ------------------------------------------------------------------ //
    // Three readers each carried a hand-copied 7-line `.each { col -> if (!(col in
    // Checkpoint.columns(...))) throw ... }` block: add_cycle.nf twice and
    // segmentation.nf once. Three copies of one rule, each free to word its message
    // differently and to fall behind. This is that rule, once.
    //
    // `assert cond : 'msg'`, never `assert cond, 'msg'`, and `closure.call(x)`, never
    // `closure(x)` — the second spelling of each does not parse under Nextflow 26 and
    // silently skipped every assertion in this file in CI's latest-everything leg.
    // Guarded by tests/test_lib_probe_parses_on_nf26.py.

    // The satisfied case returns quietly.
    Checkpoint.requireColumns('registered', ['patient_id', 'registered_image', 'channels'])
    Checkpoint.requireColumns('postprocessed', ['patient_id', 'merged_csv', 'cell_mask', 'pyramid'])

    // A column the step does not declare must throw, and the message must name BOTH the
    // missing column and what IS declared — a reader that indexes a column the writer
    // stopped emitting reads an empty field, which is a path that does not resolve.
    def missingCol = false
    def missingColMsg = ''
    try {
        Checkpoint.requireColumns('registered', ['patient_id', 'no_such_column'])
    } catch (IllegalStateException e) {
        missingCol = true
        missingColMsg = e.message
    }
    assert missingCol : 'Checkpoint.requireColumns must reject a column the step does not declare'
    assert missingColMsg.contains('no_such_column') : 'the message must name the missing column'
    assert missingColMsg.contains('registered') : 'the message must name the step'
    assert missingColMsg.contains('patient_id') : 'the message must list the declared columns'

    // Every missing column at once, not just the first: a reader that lost two columns
    // should learn both in one run rather than one per edit-and-rerun cycle.
    def bothNamed = ''
    try {
        Checkpoint.requireColumns('registered', ['alpha_gone', 'beta_gone'])
    } catch (IllegalStateException e) {
        bothNamed = e.message
    }
    assert bothNamed.contains('alpha_gone') && bothNamed.contains('beta_gone') :
        'Checkpoint.requireColumns must name every missing column, not only the first'

    // An unknown STEP is Checkpoint's own exception type, not the missing-column one:
    // a caller can tell "this schema changed" from "this step never existed".
    def badCkptStep = false
    try { Checkpoint.requireColumns('nonsense', ['patient_id']) }
    catch (Checkpoint.UnknownStepException ignored) { badCkptStep = true }
    assert badCkptStep : 'Checkpoint.requireColumns must reject an unknown step'

    // An empty request is a caller that checks nothing, which is worse than no call at
    // all — it reads as covered.
    def emptyCols = false
    try { Checkpoint.requireColumns('registered', []) }
    catch (IllegalArgumentException ignored) { emptyCols = true }
    assert emptyCols : 'Checkpoint.requireColumns must reject an empty column list'
}

/**
 * RegBackends — the registration backend's identity, in one table.
 *
 * Lives outside `workflow { ... }` deliberately: see "THE WORKFLOW BLOCK HAS A HARD
 * SIZE CEILING" above. Called once from the workflow block, before the final println.
 *
 * The method name was re-decided in five places: nextflow_schema.json's
 * registration_method enum, workflows/mirage.nf's add_cycle allowlist,
 * register_patient.nf's three-arm dispatch, seg_qc.nf's join-shape branch, and
 * lib/WarpBackends.groovy's own two-key map. Adding a backend meant finding all
 * five; missing one was silent, and the silence had a direction -- the
 * register_patient dispatch used to read `if (tiled) ... else VALIS`, so a method
 * the schema gained but that file did not know about registered with VALIS and
 * REPORTED SUCCESS. A whole benchmark arm would have measured VALIS twice under
 * two labels.
 */
def checkRegBackends() {
    assert RegBackends.methods() == ['valis', 'tiled'] :
        'RegBackends.methods() must be the BACKENDS keys in declaration order'

    // The adapter each method dispatches to. A String, not a workflow reference: a
    // Nextflow workflow cannot be invoked by name out of a map, so register_patient.nf
    // keeps a two-arm `if` -- what this replaces is the DECISION, not the call.
    assert RegBackends.of('valis').adapter == 'VALIS_ADAPTER'
    assert RegBackends.of('tiled').adapter == 'TILED_ADAPTER'

    // The seg-QC join shape. VALIS produces ONE registrar pickle per patient; the
    // tiled backend produces one transform manifest per moving SLIDE. seg_qc.nf branches
    // on this rather than on the method name, so a third backend has to DECLARE which of
    // the two shapes it has instead of silently inheriting one by falling through.
    assert RegBackends.segQcJoin('valis') == 'per_patient'
    assert RegBackends.segQcJoin('tiled') == 'per_slide'

    // The optional emits, as a null-object contract: an adapter for a method that
    // produces no TRE, or no pre-micro checkpoint, emits Channel.empty() and consumers
    // tolerate zero artifacts. Recorded here so a third backend answers both questions
    // in the table rather than in a consumer's if-chain.
    assert RegBackends.of('valis').hasStageCheckpoint
    assert !RegBackends.of('valis').hasIntrinsicTre
    assert !RegBackends.of('tiled').hasStageCheckpoint
    assert RegBackends.of('tiled').hasIntrinsicTre

    // Which run modes each backend supports. Only 'linear' exists on this branch
    // (the dev branch adds 'add_cycle', VALIS-only).
    assert RegBackends.supportsMode('valis', 'linear') :
        'valis must support the linear (default registration) run mode'
    assert RegBackends.supportsMode('tiled', 'linear') :
        'tiled must support the linear (default registration) run mode'
    assert !RegBackends.supportsMode('tiled', 'add_cycle') :
        'no backend on this branch declares add_cycle -- that mode lives on dev only'

    // An unknown method throws, and the message names the valid ones. This is the
    // property that makes the dispatch's fall-through loud instead of a silent VALIS.
    def badRegMethod = false
    def badRegMsg = ''
    try { RegBackends.of('ashlar') }
    catch (IllegalArgumentException e) { badRegMethod = true; badRegMsg = e.message }
    assert badRegMethod : 'RegBackends.of must reject an unknown method'
    assert badRegMsg.contains('ashlar') : 'the message must name what was asked for'
    assert badRegMsg.contains('valis') && badRegMsg.contains('tiled') :
        'the message must list the valid methods'

    // supportsMode and segQcJoin go through of(), so they reject the same way rather
    // than answering false/null for a name that does not exist -- a false here reads as
    // "that backend does not support add_cycle", which is a different statement.
    def badModeLookup = false
    try { RegBackends.supportsMode('ashlar', 'linear') }
    catch (IllegalArgumentException ignored) { badModeLookup = true }
    assert badModeLookup : 'RegBackends.supportsMode must reject an unknown method'

    def badJoinLookup = false
    try { RegBackends.segQcJoin('ashlar') }
    catch (IllegalArgumentException ignored) { badJoinLookup = true }
    assert badJoinLookup : 'RegBackends.segQcJoin must reject an unknown method'

    // The two backend tables must agree on WHICH backends exist. WarpBackends keys the
    // reg_qc=2 warp; RegBackends keys the registration itself. A method in one and not
    // the other is a run that registers and then cannot be scored, or vice versa.
    assert RegBackends.methods().toSorted() == WarpBackends.methods().toSorted() :
        'RegBackends and WarpBackends declare different backend sets'

    // ... and on which WARP each method uses, which is the one field they share.
    RegBackends.methods().each { m ->
        assert RegBackends.of(m).warp == m :
            "RegBackends.of('${m}').warp must name the WarpBackends key for that method"
        assert WarpBackends.of(RegBackends.of(m).warp) != null
    }

    // The table is immutable: a caller mutating it would silently rewrite every later
    // lookup in the run.
    def regTableImmutable = false
    try { RegBackends.BACKENDS.put('ashlar', [:]) }
    catch (UnsupportedOperationException ignored) { regTableImmutable = true }
    assert regTableImmutable : 'RegBackends.BACKENDS must be immutable'
}

// CsvUtils.unknownColumns — report, never reject. Out-of-band like the other
// check*() functions above: the workflow{} block is already within ~160 bytes
// of DSL2's 65535-UTF-16-code-unit single-String-constant ceiling (see the file
// header note), so a new assertion block cannot be inlined there.
def checkCsvUtilsUnknownColumns() {
    def unkCsv = File.createTempFile('unknowncols', '.csv')
    unkCsv.text = '''patient_id,path_to_file,is_reference,channels,operator,scan_date
P1,/x/a.tiff,true,DAPI|CD3,AB,2026-01-31
'''
    assert CsvUtils.unknownColumns(unkCsv.path, 'preprocessing') == ['operator', 'scan_date'] :
        'unknownColumns must report the unknown columns in header order'

    // A legitimate CHECKPOINT header must warn about nothing: a csv/registered.csv
    // read by --start segmentation carries id and pixel_size beyond that step's
    // four required columns. Rejecting -- or even warning about -- those would make
    // every re-entry noisy, which is how a real warning stops being read.
    def ckptCsv = File.createTempFile('ckptcols', '.csv')
    ckptCsv.text = '''patient_id,id,registered_image,is_reference,channels,pixel_size
P1,P1_ref,/x/a.tiff,true,DAPI|CD3,0.325
'''
    assert CsvUtils.unknownColumns(ckptCsv.path, 'segmentation') == [] :
        'a legitimate checkpoint header must produce no unknown columns'

    // The typo case this exists for: `channles` is unknown, and `channels` being
    // absent is validateInputCSV's job -- two different messages for two different
    // mistakes.
    def typoCsv = File.createTempFile('typocols', '.csv')
    typoCsv.text = '''patient_id,path_to_file,is_reference,channles
P1,/x/a.tiff,true,DAPI|CD3
'''
    assert CsvUtils.unknownColumns(typoCsv.path, 'preprocessing') == ['channles'] :
        'unknownColumns must name a mistyped column'

    // An empty file is not a crash: validateInputCSV owns that error.
    def emptyCsv = File.createTempFile('emptycols', '.csv')
    emptyCsv.text = ''
    assert CsvUtils.unknownColumns(emptyCsv.path, 'preprocessing') == [] :
        'unknownColumns must tolerate an empty file rather than throwing over it'
}

workflow {

    // ------------------------------------------------------------------ //
    // Layout — checkpoint paths
    // ------------------------------------------------------------------ //
    assert Layout.checkpointDir('/out')                              == '/out/csv'
    assert Layout.checkpointDir('/out/')                             == '/out/csv'
    assert Layout.checkpointCsvName(Layout.REGISTERED)               == 'registered.csv'
    assert Layout.checkpointCsvRelative(Layout.POSTPROCESSED)        == 'csv/postprocessed.csv'
    assert Layout.checkpointCsv('/out', Layout.PREPROCESSED)         == '/out/csv/preprocessed.csv'

    // A step name that is not a checkpoint step must throw, not silently produce
    // '<outdir>/csv/nonsense.csv'.
    def badStep = false
    try { Layout.checkpointCsvName('nonsense') }
    catch (IllegalArgumentException ignored) { badStep = true }
    assert badStep : 'Layout.checkpointCsvName must reject an unknown step'

    // A blank outdir must throw rather than produce a literal 'null/csv'.
    def badOutdir = false
    try { Layout.checkpointDir('  ') }
    catch (IllegalArgumentException ignored) { badOutdir = true }
    assert badOutdir : 'Layout.checkpointDir must reject a blank outdir'

    // ------------------------------------------------------------------ //
    // MarkerUtils — the nuclear-marker rule
    // ------------------------------------------------------------------ //
    // markerList normalises three incompatible shapes. The bare-String case is the
    // dangerous one: iterated as a List it yields CHARACTERS, so 'PANCK' would match
    // any channel containing 'C'.
    assert MarkerUtils.markerList(['DAPI', 'CELLTOX']) == ['DAPI', 'CELLTOX']
    assert MarkerUtils.markerList('DAPI,CELLTOX')      == ['DAPI', 'CELLTOX']
    assert MarkerUtils.markerList('DAPI CELLTOX')      == ['DAPI', 'CELLTOX']
    assert MarkerUtils.markerList('PANCK')             == ['PANCK']

    // Matching is case-insensitive SUBSTRING, deliberately: 'DAPI_nuclear' is nuclear.
    assert MarkerUtils.isNuclear('DAPI_nuclear', ['DAPI'])
    assert MarkerUtils.isNuclear('dapi', ['DAPI'])
    assert !MarkerUtils.isNuclear('CD3', ['DAPI'])

    // MARKER PREFERENCE ORDER BEATS CHANNEL ORDER. With markers ['DAPI','CELLTOX'] and
    // channels ['CELLTOX','CD8','DAPI'] the answer is 2, not 0. SEGMENT's CellSAM
    // backend feeds this straight to --dapi-channel, so a wrong answer segments the
    // wrong channel rather than failing.
    assert MarkerUtils.nuclearIndex(['CELLTOX', 'CD8', 'DAPI'], ['DAPI', 'CELLTOX']) == 2
    assert MarkerUtils.nuclearIndex(['CD8', 'CELLTOX'], ['DAPI', 'CELLTOX'])         == 1

    assert MarkerUtils.hasNuclear(['CD3', 'DAPI'], ['DAPI'])
    assert !MarkerUtils.hasNuclear(['CD3', 'CD8'], ['DAPI'])


    // ------------------------------------------------------------------ //
    // CsvUtils.resolveKeptChannelsPerSlide — THE keep-set rule
    // ------------------------------------------------------------------ //
    // Each marker name is claimed exactly once per patient, reference first then
    // samplesheet order. That single invariant does two things. It makes the winner of a
    // shared marker DETERMINISTIC -- two slides carrying the same name used to be
    // deduplicated by ARRIVAL ORDER at a downstream .unique(), so which copy reached
    // merged_quant.csv and the pyramid varied run to run. And it collapses the
    // distinct-name count and the emitted-file count into ONE number, which is what lets
    // a single channels_count size every downstream groupKey correctly.
    //
    // It is NOT that some consumer needs the file count: both groupTiffsByPatient callers
    // dedup on [patient_id, marker] immediately upstream (postprocess.nf's `.unique`,
    // add_cycle.nf's priority groupTuple), so the distinct-name count would serve them
    // today. An earlier version of this comment said otherwise.
    // The inner map is keyed on Meta.identityFor's output (meta.id), not the raw
    // image cell -- see resolveKeptChannelsPerSlide's doc. For a non-colliding stem,
    // identityFor ignores its rowIndex argument entirely (n<=1 short-circuits before
    // rowIndex is ever consulted), so `0` is a safe stand-in below wherever a row's
    // actual samplesheet-order index isn't the point of the assertion.
    def keepCsv = File.createTempFile('keepset', '.csv')
    keepCsv.text = '''patient_id,image,channels,is_reference
P1,ref.tiff,DAPI|KI67|CD20,true
P1,cyc2.tiff,CELLTOX|CD8,false
P1,cyc3.tiff,CELLTOX|FOXP3,false
'''
    def keep = CsvUtils.resolveKeptChannelsPerSlide(keepCsv.path, 'image', ['DAPI','CELLTOX'])
    assert keep['P1'][Meta.identityFor('P1', 'ref.tiff', 0, [:])]  == ['DAPI', 'KI67', 'CD20']
    assert keep['P1'][Meta.identityFor('P1', 'cyc2.tiff', 0, [:])] == ['CELLTOX', 'CD8']  // CELLTOX unclaimed -> KEPT
    assert keep['P1'][Meta.identityFor('P1', 'cyc3.tiff', 0, [:])] == ['FOXP3']           // CELLTOX claimed by cyc2
    def keepFlat = keep['P1'].values().flatten()
    assert keepFlat.size() == keepFlat.toSet().size()
    assert keepFlat.size() == 6
    keepCsv.delete()

    // The reference wins wherever it sits in the sheet: ordering is reference-first,
    // not row-order, so a reference declared LAST still claims its markers first.
    def keepLateRef = File.createTempFile('keeplate', '.csv')
    keepLateRef.text = '''patient_id,image,channels,is_reference
P2,cyc2.tiff,DAPI|CD8,false
P2,ref.tiff,DAPI|KI67,true
'''
    def lateRef = CsvUtils.resolveKeptChannelsPerSlide(keepLateRef.path, 'image', ['DAPI','CELLTOX'])
    assert lateRef['P2'][Meta.identityFor('P2', 'ref.tiff', 0, [:])]  == ['DAPI', 'KI67']
    assert lateRef['P2'][Meta.identityFor('P2', 'cyc2.tiff', 0, [:])] == ['CD8']   // DAPI already claimed by the reference
    keepLateRef.delete()

    // preClaimed seeds the claimed set — add_cycle passes the prior run's reference
    // channels, so a re-stained DAPI is redundant but a NEW nuclear marker survives.
    def keepPrior = File.createTempFile('keepprior', '.csv')
    keepPrior.text = '''patient_id,image,channels,is_reference
P3,cyc4.tiff,DAPI|CELLTOX|CD8,false
'''
    def seeded = CsvUtils.resolveKeptChannelsPerSlide(
        keepPrior.path, 'image', ['DAPI','CELLTOX'], ['P3': ['DAPI', 'KI67']])
    assert seeded['P3'][Meta.identityFor('P3', 'cyc4.tiff', 0, [:])] == ['CELLTOX', 'CD8']  // DAPI pre-claimed; CELLTOX new
    keepPrior.delete()

    // BASENAME COLLISION: two rows of one patient can share an image BASENAME while
    // living in different directories -- a cyclic-IF cohort with one directory per cycle
    // is the ordinary case, not a pathological one. identityFor disambiguates via each
    // row's samplesheet-order index (0, 1) whenever the stem would otherwise collide,
    // and resolveKeptChannelsPerSlide keys its map on exactly that -- so both rows land
    // as distinct entries even though their basenames, and therefore their un-disambiguated
    // stems, are identical.
    def dupCsv = File.createTempFile('keepdup', '.csv')
    dupCsv.text = '''patient_id,image,channels,is_reference
P4,/data/c1/slide.tiff,DAPI|CD3,true
P4,/data/c2/slide.tiff,DAPI|CD8,false
'''
    def dupStemCounts = CsvUtils.stemCountsPerPatient(dupCsv.path, 'image')
    def dup = CsvUtils.resolveKeptChannelsPerSlide(dupCsv.path, 'image', ['DAPI','CELLTOX'])
    assert dup['P4'].size() == 2                              // two rows, two entries
    assert dup['P4'][Meta.identityFor('P4', '/data/c1/slide.tiff', 0, [stemCounts: dupStemCounts])] == ['DAPI', 'CD3']  // the reference claims DAPI
    assert dup['P4'][Meta.identityFor('P4', '/data/c2/slide.tiff', 1, [stemCounts: dupStemCounts])] == ['CD8']          // DAPI already claimed
    assert CsvUtils.countChannelsPerPatient(dupCsv.path, 'image', ['DAPI','CELLTOX'])['P4'] == 3
    dupCsv.delete()

    // RAW-CELL COLLISION -- THE BUG THIS TASK FIXES. Two rows of one patient can share
    // the exact same raw `<imageColumn>` cell (a duplicate row) or both leave it blank,
    // which used to collide the map when it was keyed on `row.raw`: the second pass
    // silently overwrote the first -- here, with []. P6's reference row keeps
    // DAPI+CD3; the second row declares only DAPI, already claimed, so its OWN keep-set
    // is legitimately []. Under the old row.raw keying, perSlide['same.tiff'] would be
    // ['DAPI','CD3'] and then get overwritten by []. Keying on the ASSIGNED identity
    // instead means the two rows can never collapse into one entry, because identityFor
    // disambiguates them by rowIndex the moment stemCounts says their stem collides --
    // exactly as it does for a basename collision above, even though here the raw cells
    // are not just same-stem but IDENTICAL strings.
    def rawDupCsv = File.createTempFile('rawdup', '.csv')
    rawDupCsv.text = '''patient_id,image,channels,is_reference
P6,same.tiff,DAPI|CD3,true
P6,same.tiff,DAPI,false
'''
    def rawDupStemCounts = CsvUtils.stemCountsPerPatient(rawDupCsv.path, 'image')
    def rawDupId0 = Meta.identityFor('P6', 'same.tiff', 0, [stemCounts: rawDupStemCounts])
    def rawDupId1 = Meta.identityFor('P6', 'same.tiff', 1, [stemCounts: rawDupStemCounts])
    assert rawDupId0 != rawDupId1 : 'two rows sharing a raw cell must still get distinct identities'
    def rawDupKept = CsvUtils.resolveKeptChannelsPerSlide(rawDupCsv.path, 'image', ['DAPI','CELLTOX'])
    assert rawDupKept['P6'].size() == 2 : 'both rows must produce their own entry, not one overwriting the other'
    assert rawDupKept['P6'].values().every { it != null } : 'the collapse bug left a slide with no entry at all'
    assert rawDupKept['P6'][rawDupId0] == ['DAPI', 'CD3']
    assert rawDupKept['P6'][rawDupId1] == []  // legitimately empty: DAPI already claimed by the reference
    def rawDupFlat = rawDupKept['P6'].values().flatten()
    assert rawDupFlat.size() == 2 : 'the collapse bug summed this to 0 (second row overwrote the first with [])'
    assert CsvUtils.countChannelsPerPatient(rawDupCsv.path, 'image', ['DAPI','CELLTOX'])['P6'] == 2
    rawDupCsv.delete()

    // NOTE: the block above only proves resolveKeptChannelsPerSlide's OWN internal
    // row index (rowsByPatient[patientId].size(), captured while it builds its rows)
    // is collision-safe -- it never goes through CsvUtils.rowIndexPerPatient. THAT
    // function is the SEPARATE, real production call site input_check.nf's `.map`
    // closure actually uses to get rowIndex for Meta.fromSamplesheetRow, and it had
    // its OWN, independent collision: it used to key a SCALAR by
    // "patientId::rawImageCell", so two rows sharing a raw cell (or both blank) had
    // the SECOND row's index silently overwrite the first's, and BOTH rows read back
    // the SAME index -- Meta.fromSamplesheetRow then assigned them the SAME meta.id,
    // and one row's real keep-set silently displaced the other's under that shared id
    // (a wrong-but-PRESENT entry, not a clean miss -- see lib/Meta.groovy's
    // fromSamplesheetRow doc). Pinned directly, against the real function, below.
    // Pinned END TO END -- through Nextflow's own splitCsv and input_check.nf's real
    // `.map` closure, not a hand-rebuilt copy of it -- by
    // tests/subworkflows/local/input_check.nf.test's "two rows sharing the identical
    // raw path cell get distinct id and keep-set" case.
    def rowIdxCsv = File.createTempFile('rowindexdup', '.csv')
    rowIdxCsv.text = '''patient_id,image,channels,is_reference
P8,same.tiff,DAPI|CD3,true
P8,same.tiff,DAPI,false
P8,other.tiff,CD20,false
'''
    def rowIdx = CsvUtils.rowIndexPerPatient(rowIdxCsv.path, 'image')
    assert rowIdx['P8::same.tiff']  == [0, 1] : 'two rows sharing a raw cell must each keep their OWN index, in file order, not one overwriting the other'
    assert rowIdx['P8::other.tiff'] == [2]
    rowIdxCsv.delete()

    // AN EMPTY KEEP-SET IS AN ANSWER, NOT AN ABSENCE. A slide whose every declared
    // channel was already claimed contributes NO new markers, and countChannelsPerPatient
    // counts it as contributing ZERO. Its entry must therefore be PRESENT and EMPTY:
    // consumers have to be able to tell "this slide emits nothing" (emit nothing) from
    // "this slide has no entry" (fall back to its declared list). Groovy's `?:` cannot --
    // it treats [] as falsy -- so every lookup of this map uses containsKey/an explicit
    // null test, and input_check.nf does the lookup where meta.id is in scope.
    // Resolving the empty entry to the FULL declared list emitted a duplicate marker NAME
    // across two slides of one patient, which is exactly what the one-name-per-patient
    // invariant exists to forbid.
    def emptyCsv = File.createTempFile('keepempty', '.csv')
    emptyCsv.text = '''patient_id,image,channels,is_reference
P5,ref.tiff,DAPI|PANCK|SMA,true
P5,mov1.tiff,DAPI,false
'''
    def emptyKeep = CsvUtils.resolveKeptChannelsPerSlide(emptyCsv.path, 'image', ['DAPI','CELLTOX'])
    def emptyMov1Id = Meta.identityFor('P5', 'mov1.tiff', 0, [:])
    assert emptyKeep['P5'].containsKey(emptyMov1Id)   // present ...
    assert emptyKeep['P5'][emptyMov1Id] == []         // ... and EMPTY, not the declared list
    assert emptyKeep['P5'][Meta.identityFor('P5', 'ref.tiff', 0, [:])] == ['DAPI', 'PANCK', 'SMA']
    assert CsvUtils.countChannelsPerPatient(emptyCsv.path, 'image', ['DAPI','CELLTOX'])['P5'] == 3
    emptyCsv.delete()

    // THE invariant countChannelsPerPatient exists to satisfy: the count equals BOTH the
    // number of TIFFs emitted AND the number of distinct marker names, because the
    // one-name-per-patient rule makes those the same number. Pinning both equalities here
    // is what stops a future change to the resolver from quietly making channels_count
    // right for one downstream grouping and wrong for the other.
    def invCsv = File.createTempFile('keepcount', '.csv')
    invCsv.text = '''patient_id,image,channels,is_reference
P1,ref.tiff,DAPI|KI67|CD20,true
P1,cyc2.tiff,CELLTOX|CD8,false
P1,cyc3.tiff,CELLTOX|FOXP3,false
'''
    def invCounts = CsvUtils.countChannelsPerPatient(invCsv.path, 'image', ['DAPI','CELLTOX'])
    def invKept   = CsvUtils.resolveKeptChannelsPerSlide(invCsv.path, 'image', ['DAPI','CELLTOX'])
    def invFlat   = invKept['P1'].values().flatten()
    assert invCounts['P1'] == 6
    assert invCounts['P1'] == invFlat.size()          // == emitted TIFF count (pyramid)
    assert invCounts['P1'] == invFlat.toSet().size()  // == distinct names     (quant)
    invCsv.delete()

    // THE SAME INVARIANT, but forced through the ONE case invCsv above never
    // exercises: a duplicate marker name declared TWICE ON THE SAME ROW (not
    // across two slides). invCsv's only duplicate, CELLTOX, is claimed by
    // cyc2 and re-declared by cyc3 -- a CROSS-slide collision. A samplesheet
    // row's own `channels` cell is free-text and nothing upstream forbids
    // `DAPI|DAPI|KI67`, so the within-row path through the SAME claimed-set
    // loop needs its own pin: if a future change replaced the live,
    // incrementally-updated `claimed` set with one only updated AFTER each
    // row finishes (e.g. to "parallelise" the per-row scan), cross-slide
    // dedup would still hold but a same-row repeat would stop being caught,
    // desynchronising the emitted-file count from the distinct-name count
    // for exactly the reason this dual invariant exists.
    //
    // P9's reference row declares DAPI twice and P9's second row declares
    // CELLTOX twice -- two independent within-row repeats, one per slide, so
    // this cannot pass by accident of only one slide being exercised.
    def dupWithinRowCsv = File.createTempFile('keepdupwithinrow', '.csv')
    dupWithinRowCsv.text = '''patient_id,image,channels,is_reference
P9,ref.tiff,DAPI|DAPI|KI67|CD20,true
P9,cyc2.tiff,CELLTOX|CELLTOX,false
'''
    def dupWithinRowKept = CsvUtils.resolveKeptChannelsPerSlide(dupWithinRowCsv.path, 'image', ['DAPI','CELLTOX'])
    // The repeat is dropped WITHIN the row itself -- not merely deduplicated
    // later -- so each per-slide keep-list is already free of it.
    assert dupWithinRowKept['P9'][Meta.identityFor('P9', 'ref.tiff', 0, [:])]  == ['DAPI', 'KI67', 'CD20']
    assert dupWithinRowKept['P9'][Meta.identityFor('P9', 'cyc2.tiff', 0, [:])] == ['CELLTOX']
    def dupWithinRowCounts = CsvUtils.countChannelsPerPatient(dupWithinRowCsv.path, 'image', ['DAPI','CELLTOX'])
    def dupWithinRowFlat   = dupWithinRowKept['P9'].values().flatten()
    // Naively summing the DECLARED cells (4 + 2 = 6) is exactly the over-count
    // a within-row repeat would produce if it leaked through -- the four
    // distinct names below are the number every one of these must agree on.
    assert dupWithinRowFlat.size() == 4          : 'a same-row repeat leaked through as an extra emitted file'
    assert dupWithinRowFlat.toSet().size() == 4  : 'a same-row repeat leaked through as an extra distinct name'
    assert dupWithinRowCounts['P9'] == 4
    assert dupWithinRowCounts['P9'] == dupWithinRowFlat.size()          // == emitted TIFF count (pyramid)
    assert dupWithinRowCounts['P9'] == dupWithinRowFlat.toSet().size()  // == distinct names     (quant)
    dupWithinRowCsv.delete()

    // ------------------------------------------------------------------ //
    // ParamUtils — the step vocabulary
    // ------------------------------------------------------------------ //
    assert ParamUtils.STEP_ORDER == ['preprocessing', 'registration', 'segmentation', 'postprocessing']
    assert ParamUtils.entryColumnForStep('registration')      == 'preprocessed_image'
    assert ParamUtils.requiredColumnsForStep('preprocessing') ==
        ['patient_id', 'path_to_file', 'is_reference', 'channels']

    assert  ParamUtils.shouldRun('registration',   'preprocessing', 'postprocessing')
    assert !ParamUtils.shouldRun('preprocessing',  'registration',  'postprocessing')
    assert !ParamUtils.shouldRun('postprocessing', 'preprocessing', 'registration')

    // 'segmentation' is now a REAL step (added by this task) and must not be used
    // here as the "unknown step" example any more -- use a name that is genuinely
    // absent from STEPS instead.
    def badTarget = false
    try { ParamUtils.shouldRun('quantification', 'preprocessing', 'postprocessing') }
    catch (IllegalArgumentException ignored) { badTarget = true }
    assert badTarget : 'ParamUtils.shouldRun must reject an unknown step'

    // Every step's entryColumn must be one of its own requiredColumns, or the
    // samplesheet parser reads a column validation never demanded.
    ParamUtils.STEPS.each { step ->
        assert step.entryColumn in step.requiredColumns :
            "step '${step.name}': entryColumn '${step.entryColumn}' not in requiredColumns"
    }

    // ------------------------------------------------------------------ //
    // Checkpoint — filename AND columns, one owner
    // ------------------------------------------------------------------ //
    assert Checkpoint.columns(Layout.PREPROCESSED) ==
        ['patient_id', 'id', 'preprocessed_image', 'is_reference', 'channels', 'pixel_size']
    assert Checkpoint.columns(Layout.REGISTERED) ==
        ['patient_id', 'id', 'registered_image', 'is_reference', 'channels', 'pixel_size']
    assert Checkpoint.columns(Layout.POSTPROCESSED) ==
        ['patient_id', 'id', 'cell_csv', 'cell_geojson', 'merged_csv', 'cell_mask', 'pyramid', 'pixel_size']

    // The header IS the seed: string the three writers pass to collectFile. These
    // three literals are the published contract — Group A must not change them.
    // RULING R17 (Task 4.3) added 'id'; this task appended 'pixel_size' LAST.
    assert Checkpoint.header(Layout.PREPROCESSED)  == 'patient_id,id,preprocessed_image,is_reference,channels,pixel_size'
    assert Checkpoint.header(Layout.REGISTERED)    == 'patient_id,id,registered_image,is_reference,channels,pixel_size'
    assert Checkpoint.header(Layout.POSTPROCESSED) == 'patient_id,id,cell_csv,cell_geojson,merged_csv,cell_mask,pyramid,pixel_size'

    // row() emits values in DECLARED COLUMN ORDER regardless of map insertion order.
    // This is the whole point: a writer can no longer transpose two columns.
    assert Checkpoint.row(Layout.REGISTERED, [
        channels: 'DAPI|CD3', patient_id: 'P001', id: 'P001_x',
        registered_image: '/out/P001/registered/x.ome.tiff', is_reference: false,
        pixel_size: 0.325,
    ]) == 'P001,P001_x,/out/P001/registered/x.ome.tiff,false,DAPI|CD3,0.325'

    // RFC 4180 QUOTING (Task 4.5). `--outdir` is an arbitrary filesystem path and
    // every published path is built from it -- a comma anywhere in it used to shift
    // every later column of a bare `.join(',')` row. A value containing a comma is
    // now wrapped in double quotes, so the field count is preserved on read-back
    // (Nextflow's `splitCsv(header: true)` parses RFC 4180 quoting natively).
    assert Checkpoint.row(Layout.REGISTERED, [
        patient_id: 'P001', id: 'P001_x',
        registered_image: '/out,dir/P001/registered/x,y.ome.tiff',
        is_reference: false, channels: 'DAPI|CD3', pixel_size: 0.325,
    ]) == 'P001,P001_x,"/out,dir/P001/registered/x,y.ome.tiff",false,DAPI|CD3,0.325'

    // A value containing a double quote is wrapped AND its embedded quotes are
    // doubled -- the RFC 4180 escape -- so a naive strip-the-outer-quotes reader
    // (as opposed to splitCsv) would still see the original text back.
    assert Checkpoint.row(Layout.REGISTERED, [
        patient_id: 'P001', id: 'P001_x',
        registered_image: '/out/say "hi"/x.ome.tiff',
        is_reference: false, channels: 'DAPI|CD3', pixel_size: 0.325,
    ]) == 'P001,P001_x,"/out/say ""hi""/x.ome.tiff",false,DAPI|CD3,0.325'

    // A `null` VALUE (as opposed to a missing KEY, rejected above) now writes as an
    // EMPTY field, not the literal four-character text `null`. Before this task a
    // null-valued column read back as a bogus four-character path/id indistinguishable
    // from a real one; an empty field is what every "artifact not produced" caller
    // already writes (see the class doc's EMPTY VALUES note), and is what
    // Meta.fromCheckpointRow's requirePresentInRow already treats as absent.
    assert Checkpoint.row(Layout.REGISTERED, [
        patient_id: 'P001', id: 'P001_x', registered_image: null,
        is_reference: false, channels: 'DAPI|CD3', pixel_size: 0.325,
    ]) == 'P001,P001_x,,false,DAPI|CD3,0.325'

    // A missing column must throw, not silently emit an empty field — an empty field
    // is a checkpoint row naming a path that does not exist, which is exactly the
    // failure csv/postprocessed.csv shipped with for two releases.
    def missingCol = false
    try { Checkpoint.row(Layout.REGISTERED, [patient_id: 'P001']) }
    catch (IllegalArgumentException ignored) { missingCol = true }
    assert missingCol : 'Checkpoint.row must reject a missing column'

    // An unknown key must throw too (pixel_size supplied, so this is the real check).
    def unknownCol = false
    try {
        Checkpoint.row(Layout.REGISTERED, [
            patient_id: 'P001', id: 'P001_x', registered_image: '/x', is_reference: false,
            channels: 'DAPI', pixel_size: 0.325, typo_column: 'oops'
        ])
    }
    catch (IllegalArgumentException ignored) { unknownCol = true }
    assert unknownCol : 'Checkpoint.row must reject an unknown column'

    // Checkpoint and Layout must agree on the step vocabulary. Two tables that drift
    // is the exact failure this extraction exists to prevent.
    assert Checkpoint.STEPS*.name == Layout.CHECKPOINT_STEPS

    // A step's requiredColumns used to EQUAL the previous step's full checkpoint
    // column list -- that was true before RULING R17. It no longer is, by design:
    // 'id' was added to every Checkpoint.STEPS schema (every checkpoint FILE now
    // carries identity), but 'registration'/'segmentation' still enter through
    // INPUT_CHECK (Meta.fromSamplesheetRow), which derives id itself from the entry
    // image column and never reads a persisted 'id' value -- so requiring it in
    // CsvUtils.validateInputCSV there would rightly reject an OLDER checkpoint
    // (preprocessed.csv/registered.csv written before this task) that INPUT_CHECK
    // could actually still read correctly. Only 'postprocessing' actually
    // dereferences 'id' (via READ_SEGMENTED_CHECKPOINT's Meta.fromCheckpointRow),
    // so only IT gained 'id' in requiredColumns (see lib/ParamUtils.groovy's
    // comment on that entry). The invariant is now "every OTHER column" equality,
    // not full-list equality. 'pixel_size' joins 'id' in the exclusion set: those
    // two steps re-resolve scale via PREFLIGHT_SCALE, not a persisted value.
    assert ParamUtils.STEPS.find { it.name == 'registration' }.requiredColumns ==
        Checkpoint.columns(Layout.PREPROCESSED) - ['id', 'pixel_size']
    assert ParamUtils.STEPS.find { it.name == 'segmentation' }.requiredColumns ==
        Checkpoint.columns(Layout.REGISTERED) - ['id', 'pixel_size']
    assert ParamUtils.STEPS.find { it.name == 'postprocessing' }.requiredColumns ==
        ['patient_id', 'id', 'registered_image', 'is_reference', 'channels', 'cell_mask', 'nuclei_mask']
    assert ParamUtils.STEPS.find { it.name == 'postprocessing' }.requiredColumns
        .every { it in Checkpoint.columns(Layout.SEGMENTED) }

    // columns() must hand out an IMMUTABLE list. Checkpoint.STEPS' outer list is
    // .asImmutable() but a naive implementation leaves each entry's inner `columns`
    // ArrayList mutable, so a caller mutating the returned list permanently corrupts
    // the schema every later header()/row() call in the run sees. Demonstrated before
    // the fix: `Checkpoint.columns(Layout.PREPROCESSED) << 'injected'` succeeded
    // silently and mutated the live table in place, so a subsequent header() call
    // returned 'patient_id,id,preprocessed_image,is_reference,channels,injected'.
    def columnsThrew = false
    try { Checkpoint.columns(Layout.PREPROCESSED) << 'injected' }
    catch (UnsupportedOperationException ignored) { columnsThrew = true }
    assert columnsThrew : 'Checkpoint.columns() must return an immutable list; mutating it must throw'
    assert Checkpoint.columns(Layout.PREPROCESSED) ==
        ['patient_id', 'id', 'preprocessed_image', 'is_reference', 'channels', 'pixel_size'] :
        'Checkpoint.columns() schema must be unaffected by the mutation attempt above'

    // ------------------------------------------------------------------ //
    // The artifact vocabulary
    // ------------------------------------------------------------------ //
    def kinds = ParamUtils.STEPS.collectMany { it.qcKinds } + ParamUtils.UNIVERSAL_QC_KINDS
    // Order is load-bearing here: the kind order is the report-slot order, so this is
    // an ordered-literal equality, not a set/sorted comparison. That same equality
    // already subsumes uniqueness -- a duplicate member makes this list one element
    // longer than the literal on the right, which fails equality before any separate
    // uniqueness check could run. (A prior version of this file carried a
    // `kinds.size() == kinds.toSet().size()` line after this assert; it was dead code,
    // unreachable by construction, and has been removed rather than kept "just in case".)
    assert kinds == ['preprocess_qc', 'registration_qc', 'registration_tre', 'seg_qc',
                     'seg_residuals', 'postprocess_qc', 'versions', 'size_log']

    // ------------------------------------------------------------------ //
    // The segmentation checkpoint
    // ------------------------------------------------------------------ //
    assert Layout.SEGMENTED == 'segmented'
    assert Layout.CHECKPOINT_STEPS == ['preprocessed', 'registered', 'segmented', 'postprocessed']
    assert Checkpoint.columns(Layout.SEGMENTED) == [
        'patient_id', 'id', 'registered_image', 'is_reference', 'channels',
        'cell_mask', 'nuclei_mask', 'contours', 'nucleus_contours', 'pixel_size',
    ]
    assert Checkpoint.header(Layout.SEGMENTED) ==
        'patient_id,id,registered_image,is_reference,channels,cell_mask,nuclei_mask,contours,nucleus_contours,pixel_size'

    // Empty string means "artifact not produced" — nucleus_contours is empty when
    // --quantify_compartments is false (nuclei_mask is NOT similarly gated: SEGMENT
    // always produces it — see Checkpoint's EMPTY VALUES note). row() must accept ''
    // (it is a value, not a missing key) and emit it as an empty field regardless of
    // which column carries it.
    assert Checkpoint.row(Layout.SEGMENTED, [
        patient_id: 'P001', id: 'P001_a', registered_image: '/o/P001/registered/a.tif',
        is_reference: true, channels: 'DAPI|CD3',
        cell_mask: '/o/P001/segmentation/P001_cell_mask.tif',
        nuclei_mask: '/o/P001/segmentation/P001_nuclei_mask.tif',
        contours: '/o/P001/cell_properties/contours.json',
        nucleus_contours: '',
        pixel_size: 0.325,
    ]) == 'P001,P001_a,/o/P001/registered/a.tif,true,DAPI|CD3,/o/P001/segmentation/P001_cell_mask.tif,/o/P001/segmentation/P001_nuclei_mask.tif,/o/P001/cell_properties/contours.json,,0.325'

    // The step vocabulary gains one entry.
    assert ParamUtils.entryColumnForStep('segmentation') == 'registered_image'

    // ------------------------------------------------------------------ //
    // Layout.isUnderTaskDir / publishedOrAsIs — the fix for Critical 1
    // (--start segmentation recording paths that do not exist). A one-level
    // isTaskDir(path.parent) check cannot tell a FRESH one-level-nested output
    // (REGISTER's registered_slides/) from an ALREADY-PUBLISHED path with the
    // same producer-subdirectory name -- both have a parent literally named
    // 'registered_slides', which never matches the hex-hash pattern either way.
    // These six cases are exactly what review's own probe checked directly
    // against lib/, unreachable from any stub-run assertion at the workflow
    // level: reverting isUnderTaskDir to a parent-only check `isTaskDir(dir)`
    // makes case 2 fail (a live registered_slides/ output would be treated as
    // already-published and returned verbatim as its work-directory path).
    // ------------------------------------------------------------------ //
    def taskHash   = 'c' * 30
    def workDirPre = "/work/ab/${taskHash}"

    // 1. Fresh, flat (no producer subdirectory) -- e.g. SEGMENT's cell_mask.
    def freshFlat = file("${workDirPre}/P001_cell_mask.tif")
    assert Layout.isUnderTaskDir(freshFlat.parent)
    assert Layout.publishedOrAsIs('/out', 'P001', 'segmentation', freshFlat) ==
        Layout.publishedPath('/out', 'P001', 'segmentation', freshFlat)
    assert Layout.publishedOrAsIs('/out', 'P001', 'segmentation', freshFlat) ==
        '/out/P001/segmentation/P001_cell_mask.tif'

    // 2. Fresh, ONE level down -- REGISTER's registered_slides/. THE case a
    // parent-only isTaskDir check gets wrong (see the comment above).
    def freshNested = file("${workDirPre}/registered_slides/P001_x_registered.ome.tiff")
    assert Layout.isUnderTaskDir(freshNested.parent)
    assert Layout.publishedOrAsIs('/out', 'P001', Layout.REGISTERED, freshNested) ==
        '/out/P001/registered/registered_slides/P001_x_registered.ome.tiff'

    // 3. Already-published, same producer-subdirectory NAME as case 2 -- a prior
    // run's registered.csv entry read back via --start segmentation's INPUT_CHECK.
    // Neither the parent ('registered_slides') nor the grandparent ('registered')
    // is a task hash, so this must be returned AS-IS, not reconstructed against
    // the new outdir (that was Critical 1).
    def publishedNested = file('/prior/P001/registered/registered_slides/P001_x_registered.ome.tiff')
    assert !Layout.isUnderTaskDir(publishedNested.parent)
    assert Layout.publishedOrAsIs('/out', 'P001', Layout.REGISTERED, publishedNested) ==
        '/prior/P001/registered/registered_slides/P001_x_registered.ome.tiff'

    // 4. A single-slide passthrough, published under 'preprocessed/' but asked
    // about with kind REGISTERED -- the exact P002 shape from Critical 1's repro.
    // The kind argument must be IGNORED once the as-is branch fires: returning
    // the file's own path unchanged is what makes the wrong kind harmless here.
    def publishedPassthrough = file('/prior/P002/preprocessed/P002_ref_corrected.ome.tif')
    assert !Layout.isUnderTaskDir(publishedPassthrough.parent)
    assert Layout.publishedOrAsIs('/out', 'P002', Layout.REGISTERED, publishedPassthrough) ==
        '/prior/P002/preprocessed/P002_ref_corrected.ome.tif'

    // 5. Already-published, flat (no producer subdirectory) -- a prior run's
    // segmentation.nf cell_mask read back via --start postprocessing.
    def publishedFlat = file('/prior/P001/segmentation/P001_cell_mask.tif')
    assert !Layout.isUnderTaskDir(publishedFlat.parent)
    assert Layout.publishedOrAsIs('/out', 'P001', 'segmentation', publishedFlat) ==
        '/prior/P001/segmentation/P001_cell_mask.tif'

    // 6. Fresh, ONE level down via nuclei/ -- EXTRACT_NUCLEI_PROPERTIES's own
    // producer subdirectory (the fix for I5's published-path regression).
    def freshNuclei = file("${workDirPre}/nuclei/contours.json")
    assert Layout.isUnderTaskDir(freshNuclei.parent)
    assert Layout.publishedOrAsIs('/out', 'P001', 'cell_properties', freshNuclei) ==
        '/out/P001/cell_properties/nuclei/contours.json'

    // passthroughPath now delegates to publishedOrAsIs pinned to PREPROCESSED --
    // assert the delegation is behaviourally identical, not just present.
    assert Layout.passthroughPath('/out', 'P001', freshFlat) ==
        Layout.publishedOrAsIs('/out', 'P001', Layout.PREPROCESSED, freshFlat)
    assert Layout.passthroughPath('/out', 'P002', publishedPassthrough) ==
        Layout.publishedOrAsIs('/out', 'P002', Layout.PREPROCESSED, publishedPassthrough)

    // ------------------------------------------------------------------ //
    // ParamUtils.compartmentMode / validateCompartmentQuant -- the
    // --quantify_compartments seam (mirrors --registration_method's: resolved
    // once, threaded down as an argument; see workflows/mirage.nf's
    // `compartment_mode` and tests/test_compartment_mode_routing.py).
    // ------------------------------------------------------------------ //

    // 1. Plain field mapping, all three flags on.
    def modeAllOn = ParamUtils.compartmentMode([
        quantify_compartments: true, expanded_quantification: true, embed_masks: true,
    ])
    assert modeAllOn == [compartments: true, expanded: true, embedMasks: true]

    // 2. All three off.
    def modeAllOff = ParamUtils.compartmentMode([
        quantify_compartments: false, expanded_quantification: false, embed_masks: false,
    ])
    assert modeAllOff == [compartments: false, expanded: false, embedMasks: false]

    // 3. The map is immutable -- a caller cannot mutate the shared snapshot out
    // from under another reader of the same resolved value.
    try {
        modeAllOn.compartments = false
        assert false : "compartmentMode() map must be immutable"
    } catch (UnsupportedOperationException ignored) {
        // expected
    }

    // 4. validateCompartmentQuant: expanded requires compartments (pre-existing
    // rule, now driven off the resolved map instead of two raw booleans).
    ParamUtils.validateCompartmentQuant([compartments: true, expanded: true, embedMasks: false])   // ok
    ParamUtils.validateCompartmentQuant([compartments: true, expanded: false, embedMasks: false])  // ok
    try {
        ParamUtils.validateCompartmentQuant([compartments: false, expanded: true, embedMasks: false])
        assert false : "expanded=true, compartments=false must be rejected"
    } catch (IllegalArgumentException ignored) {
        // expected
    }

    // 5. validateCompartmentQuant: the delayed-cost gap this task closes --
    // embedMasks requires BOTH compartments AND expanded. Each way to violate
    // that must be rejected, not just the case where compartments is off.
    ParamUtils.validateCompartmentQuant([compartments: true, expanded: true, embedMasks: true])    // ok
    try {
        ParamUtils.validateCompartmentQuant([compartments: false, expanded: false, embedMasks: true])
        assert false : "embedMasks=true with compartments=false, expanded=false must be rejected"
    } catch (IllegalArgumentException ignored) {
        // expected -- this exact combination used to exit 0 and silently publish
        // a pyramid with no mask series (assemble_export.nf's embed_masks gate).
    }
    try {
        ParamUtils.validateCompartmentQuant([compartments: true, expanded: false, embedMasks: true])
        assert false : "embedMasks=true with expanded=false must be rejected even when compartments=true"
    } catch (IllegalArgumentException ignored) {
        // expected
    }

    // validateRegPresets — COARSE's 256 px floor. 0/negatives are DANGEROUS, not
    // merely invalid: decimation_factor reads <=0 as "no decimation" (full-res
    // plane into a U-Net) and the memory closure squares the value, so both ask
    // for the 4 GB floor for the largest possible job. Rationale: ParamUtils.
    def sb = [memory_mode: 'high', reg_tiled_mode: 'custom',
              reg_valis_max_processed_dim: null, reg_valis_max_non_rigid_dim: null,
              reg_tiled_tile: null, reg_tiled_halo: null,
              reg_tiled_upsample: null, reg_tiled_out_tile: null]
    [null, 512, 256].each { ParamUtils.validateRegPresets(sb + [reg_tiled_coarse_max_dim: it]) }
    [0, -1, 255, 16].each { bad ->
        def no = false
        try { ParamUtils.validateRegPresets(sb + [reg_tiled_coarse_max_dim: bad]) }
        catch (IllegalArgumentException ignored) { no = true }
        assert no : "validateRegPresets must reject reg_tiled_coarse_max_dim=${bad}"
    }

    // ------------------------------------------------------------------ //
    // Layout — the published-kind vocabulary
    // ------------------------------------------------------------------ //
    assert Layout.requireKind('segmentation') == 'segmentation'
    assert Layout.PUBLISHED_KINDS.contains(Layout.REGISTERED)
    assert Layout.PUBLISHED_KINDS.contains('split_channels')

    def badKind = false
    try { Layout.requireKind('segmentaton') }   // typo, deliberately
    catch (IllegalArgumentException ignored) { badKind = true }
    assert badKind : 'Layout.requireKind must reject an unknown kind'

    // patientDir must reject it too — that is the call site the typo reaches from.
    def badPatientKind = false
    try { Layout.patientDir('/out', 'P001', 'segmentaton') }
    catch (IllegalArgumentException ignored) { badPatientKind = true }
    assert badPatientKind : 'Layout.patientDir must reject an unknown kind'

    // ------------------------------------------------------------------ //
    // ProcessEnvelope — the versions.yml envelope
    // ------------------------------------------------------------------ //
    def envVersions     = ProcessEnvelope.versions('TEST:PROC', ['numpy', 'skimage'])
    def envVersionsStub = ProcessEnvelope.versionsStub('TEST:PROC', ['numpy', 'skimage'])

    // The python: row is prepended automatically, in both renderings. A BARE `$(`, not
    // `\$(` -- `<<-END_VERSIONS` is an unquoted heredoc, so bash performs command
    // substitution on the former and prints the latter as literal text. This assertion
    // is the one that would have caught the over-escaping bug: it asserted `\\$(` (an
    // escaped, unexecuted dollar) until a reviewer caught that the real published
    // versions.yml was showing shell commands instead of version numbers.
    assert envVersions.contains('python: $(python --version')
    assert !envVersions.contains('python: \\$(python --version')
    assert envVersionsStub.contains('python: stub')

    // skimage -> scikit-image: the import name and the published YAML key differ, and
    // bin/generate_qc_report.py's hand-rolled parser is keyed on the YAML key, not the
    // import name.
    assert envVersions.contains('scikit-image: $(python -c "import skimage;')
    assert !envVersions.contains('skimage: $(python -c "import skimage;')
    assert envVersionsStub.contains('scikit-image: stub')

    // The property this whole task exists to guarantee: versions() and versionsStub()
    // must never be able to name a different set of tools. Comparing the two heredocs'
    // YAML KEYS (the text before each ':') is what -stub could never see for itself,
    // because -stub never evaluates the script: block that versions() renders.
    def keysOf = { String heredoc ->
        heredoc.readLines()
            .findAll { it.startsWith('    ') }
            .collect { it.trim().split(':')[0].trim() }
            .toSet()
    }
    assert keysOf.call(envVersions) == keysOf.call(envVersionsStub)
    assert keysOf.call(envVersions) == ['python', 'numpy', 'scikit-image'].toSet()

    // ------------------------------------------------------------------ //
    // SegBackends — the segmentation seam's versions.yml tool lists
    // ------------------------------------------------------------------ //
    // Declared as BARE MODULE NAMES, not rendered YAML. That is what lets
    // modules/local/segment.nf hand the SAME list to ProcessEnvelope.versions()
    // (script:) and ProcessEnvelope.versionsStub() (stub:). Before this, the table
    // stored six pre-rendered probe strings, which a stub block cannot reuse -- so
    // segment.nf's stub: hand-wrote `seg_method: <name>` instead, a key that is not a
    // tool and that no real run ever emits. `-stub` never evaluates script:, so nothing
    // in the blocking gate could see the two key sets were disjoint.
    assert SegBackends.methods().toSorted() == ['cellsam', 'instantseg', 'stardist']
    assert SegBackends.of('stardist').versionTools   == ['tensorflow']
    assert SegBackends.of('instantseg').versionTools == ['instanseg', 'torch']
    assert SegBackends.of('cellsam').versionTools    == ['cellSAM', 'torch']

    // `torch` is deliberately repeated across two backends rather than hoisted into a
    // shared list: exactly ONE backend runs per task, and stardist loads no torch at
    // all, so a shared list would fabricate a torch row for every StarDist run.
    assert !SegBackends.of('stardist').versionTools.contains('torch')

    // The property the whole envelope exists for, asserted end-to-end for every
    // backend: rendering the SAME list through versions() and versionsStub() yields
    // the SAME YAML keys. Compare keys (the text before each ':'), because the values
    // are exactly what differs between a real run and a stub.
    def yamlKeysOf = { String heredoc ->
        heredoc.readLines()
            .findAll { it.startsWith('    ') }
            .collect { it.trim().split(':')[0].trim() }
    }
    SegBackends.methods().each { m ->
        def tools = SegBackends.of(m).versionTools
        def real  = yamlKeysOf.call(ProcessEnvelope.versions('SEGMENT', tools))
        def stub  = yamlKeysOf.call(ProcessEnvelope.versionsStub('SEGMENT', tools))
        assert real == stub : "SEGMENT/${m}: script: and stub: versions.yml keys differ"
        assert real == ['python'] + tools : "SEGMENT/${m}: unexpected versions.yml keys ${real}"
        // The old stub key must not come back under any backend.
        assert !real.contains('seg_method')
    }

    // ------------------------------------------------------------------ //
    // WarpBackends — one seam for the reg_qc=2 warp
    // ------------------------------------------------------------------ //
    assert WarpBackends.methods().toSorted() == ['tiled', 'valis']
    // Digest-pinned (ruling R6): no tag, see tests/test_base_images_are_digest_pinned.py.
    assert WarpBackends.container('valis') == 'cdgatenbee/valis-wsi@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6'
    assert WarpBackends.container('tiled') == 'bolt3x/mirage-tiled:1.0.0'
    assert WarpBackends.of('valis').stages == ['native', 'rigid', 'non_rigid', 'micro']
    assert WarpBackends.of('tiled').stages == ['native', 'rigid', 'refined']

    // The tiled backend must pass --method tiled; VALIS must not.
    assert WarpBackends.of('tiled').flags([:]).any { it.contains('--method tiled') }
    // No JVM heap flag may leak onto a JVM-free backend (see the WarpBackends header).
    assert !WarpBackends.of('tiled').flags([:]).any { it.contains('--jvm-heap-gb') }
    def valisFlags = WarpBackends.of('valis').flags(
        [ref_slide: 'R', moving_slide: 'M', stage_checkpoint: null, micro_reg: 2])
    assert !valisFlags.any { it.contains('--method') }
    assert valisFlags.any { it.contains('--micro-reg 2') }
    // Absent checkpoint is a null object, not an empty flag.
    assert !valisFlags.any { it.contains('--checkpoint-dir') }
    assert WarpBackends.of('valis').flags(
        [ref_slide: 'R', moving_slide: 'M', stage_checkpoint: 'ckpt/', micro_reg: 2]
    ).any { it.contains('--checkpoint-dir ckpt/') }

    def badMethod = false
    try { WarpBackends.of('stare') }
    catch (IllegalArgumentException ignored) { badMethod = true }
    assert badMethod : 'WarpBackends.of must reject an unknown method'

    // ------------------------------------------------------------------ //
    // Meta — the one constructor for every meta map (Task 4.1)
    // ------------------------------------------------------------------ //
    assert Meta.REQUIRED_KEYS.containsAll([
        'patient_id', 'id', 'is_reference', 'channels',
        'keep_channels', 'channels_count', 'images_count',
    ])

    // identityFor: RULING R2, verified directly against Nextflow's own
    // file(...).simpleName earlier in this task -- it strips EVERY extension
    // (file('slide.ome.tiff').simpleName == 'slide'), not just the last one.
    // identityFor must reproduce that for a non-colliding row.
    assert Meta.identityFor('P1', 'slide.ome.tiff', 0, [:]) == 'P1_slide'
    assert Meta.identityFor('P1', 'slide.tiff', 0, [:])     == 'P1_slide'
    assert Meta.identityFor('P1', 'slide.tif', 0, [:])      == 'P1_slide'
    assert Meta.identityFor('P1', 'noext', 0, [:])          == 'P1_noext'
    // Already patient-prefixed: not re-prefixed.
    assert Meta.identityFor('P1', 'P1_slide.ome.tiff', 0, [:]) == 'P1_slide'
    // A collision (stemCounts says two rows of P1 share the 'slide' stem) is
    // disambiguated by rowIndex; a non-colliding stem for a different patient
    // sharing the same stem text is NOT (the counts map is keyed per-patient).
    def collideCtx = [stemCounts: ['P1::slide': 2]]
    assert Meta.identityFor('P1', 'cycle1/slide.ome.tiff', 0, collideCtx) == 'P1_slide_000'
    assert Meta.identityFor('P1', 'cycle2/slide.ome.tiff', 1, collideCtx) == 'P1_slide_001'
    assert Meta.identityFor('P2', 'slide.ome.tiff', 0, collideCtx)        == 'P2_slide'

    // fromSamplesheetRow: full construction, REQUIRED_KEYS present, dual invariant
    // upheld (channels_count comes from ctx.channelsCount, the per-PATIENT total --
    // NOT meta.keep_channels.size(), which is only a per-SLIDE count. Conflating
    // the two was a bug in this task's own brief: CsvUtils.countChannelsPerPatient
    // sums per-slide keep-set sizes ACROSS a patient's slides, so a single slide's
    // keep_channels.size() under-reports it whenever a patient has more than one
    // slide).
    // keepChannelsBySlide is keyed on the ASSIGNED identity (meta.id), matching what
    // CsvUtils.resolveKeptChannelsPerSlide now produces and what finish() looks up by
    // -- 'P1_img1' is identityFor('P1', 'img1.ome.tiff', 0, [:]), asserted below.
    def ssCtx = [
        keepChannelsBySlide: [P1: ['P1_img1': ['DAPI', 'CD3']]],
        imagesCount        : [P1: 2],
        channelsCount      : [P1: 5],  // e.g. this slide's 2 + a second slide's 3
    ]
    def ssRow  = [patient_id: 'P1', image: 'img1.ome.tiff', channels: 'DAPI|CD3', is_reference: 'true']
    def ssMeta = Meta.fromSamplesheetRow(ssRow, 'image', 0, ssCtx)
    assert Meta.REQUIRED_KEYS.every { ssMeta.containsKey(it) }
    assert ssMeta.patient_id     == 'P1'
    assert ssMeta.id             == 'P1_img1'
    assert ssMeta.is_reference   == true
    assert ssMeta.channels       == ['DAPI', 'CD3']
    assert ssMeta.keep_channels  == ['DAPI', 'CD3']
    assert ssMeta.channels_count == 5
    assert ssMeta.images_count   == 2

    // keep_channels ABSENT vs EMPTY: a slide with no entry in keepChannelsBySlide
    // falls back to its declared channels; a slide with an EXPLICIT empty list
    // (every marker already claimed by an earlier slide) keeps [], not the
    // declared list -- `?:` cannot tell these apart because [] is falsy in Groovy.
    // Both ctx maps below must also carry channelsCount AND imagesCount now (fix
    // rounds 1 and 2: there is no more per-slide/default fallback for either --
    // see the channels_count-missing and images_count-missing blocks further down).
    // 'P1_img2' is identityFor('P1', 'img2.ome.tiff', 0, [:]) -- same reasoning as ssCtx.
    def emptyKeepCtx = [keepChannelsBySlide: [P1: ['P1_img2': []]], channelsCount: [P1: 0], imagesCount: [P1: 1]]
    def emptyKeepRow = [patient_id: 'P1', image: 'img2.ome.tiff', channels: 'DAPI|CD3', is_reference: 'false']
    def emptyKeepMeta = Meta.fromSamplesheetRow(emptyKeepRow, 'image', 0, emptyKeepCtx)
    assert emptyKeepMeta.keep_channels == []
    assert emptyKeepMeta.channels_count == 0
    def absentKeepCtx = [keepChannelsBySlide: [P1: [:]], channelsCount: [P1: 2], imagesCount: [P1: 1]]
    def absentKeepMeta = Meta.fromSamplesheetRow(emptyKeepRow, 'image', 0, absentKeepCtx)
    assert absentKeepMeta.keep_channels == ['DAPI', 'CD3']

    // ---- Fix round 1, item 1: channels_count has NO fallback -----------------
    //
    // finish() used to fall back to meta.keep_channels.size() (a per-SLIDE count)
    // when ctx.channelsCount had no entry for the patient. That silently produced
    // a too-low count for any multi-slide patient, and a too-low channels_count
    // feeds groupKey(patient_id, channels_count) -- the group then emits early
    // with missing members, or never emits, far from this call site. There is now
    // no fallback and no opt-out: every caller must pre-compute a real per-patient
    // total (see finish()'s comment for why no legitimate caller lacks one).
    def noCountRow = [patient_id: 'P1', image: 'img3.ome.tiff', channels: 'DAPI|CD3', is_reference: 'false']

    // Trigger: ctx carries no channelsCount key at all.
    def missingChannelsCountKey = false
    try { Meta.fromSamplesheetRow(noCountRow, 'image', 0, [:]) }
    catch (IllegalArgumentException ignored) { missingChannelsCountKey = true }
    assert missingChannelsCountKey : 'Meta.fromSamplesheetRow must reject ctx with no channelsCount map at all'

    // Trigger: ctx.channelsCount is populated, but not for THIS patient (the
    // "computed the map, forgot one patient" case -- the one the review named).
    def missingChannelsCountForPatient = false
    try { Meta.fromSamplesheetRow(noCountRow, 'image', 0, [channelsCount: [P2: 9]]) }
    catch (IllegalArgumentException ignored) { missingChannelsCountForPatient = true }
    assert missingChannelsCountForPatient : 'Meta.fromSamplesheetRow must reject ctx.channelsCount missing THIS patient'

    // Satisfy: supply BOTH required per-patient counts, watch it pass.
    def satisfiedMeta = Meta.fromSamplesheetRow(noCountRow, 'image', 0, [channelsCount: [P1: 2], imagesCount: [P1: 1]])
    assert satisfiedMeta.channels_count == 2

    // A GENUINE zero is not "missing": containsKey, not a truthy/`?:` check, is
    // what tells these apart -- the same distinction keep_channels' ABSENT-vs-EMPTY
    // rule already has to make.
    def zeroCountMeta = Meta.fromSamplesheetRow(noCountRow, 'image', 0, [channelsCount: [P1: 0], imagesCount: [P1: 1]])
    assert zeroCountMeta.channels_count == 0

    // ---- Fix round 2: images_count has NO fallback either --------------------
    //
    // Same defect, one field over, and WORSE while it lasted: the old expression
    // was `(ctx?.imagesCount ?: [:])[patientId] ?: 1` -- a bare `?:` on the looked-up
    // value, which in Groovy treats an explicit `0` as falsy too. So a genuine
    // `images_count: 0` would have been silently COERCED to `1`, not merely
    // defaulted to a differently-wrong number the way the old channels_count bug
    // was. images_count is equally load-bearing: it feeds
    // groupKey(meta.patient_id, meta.images_count) in
    // subworkflows/local/registration.nf:77. Same fix, same helper
    // (requirePerPatientCount), same reasoning: no opt-out, because every caller
    // already computes a full per-patient imagesCount map up front
    // (CsvUtils.countImagesPerPatient).

    // Trigger: ctx carries no imagesCount key at all (channelsCount present, so
    // this isolates the images_count check specifically).
    def missingImagesCountKey = false
    try { Meta.fromSamplesheetRow(noCountRow, 'image', 0, [channelsCount: [P1: 2]]) }
    catch (IllegalArgumentException ignored) { missingImagesCountKey = true }
    assert missingImagesCountKey : 'Meta.fromSamplesheetRow must reject ctx with no imagesCount map at all'

    // Trigger: ctx.imagesCount is populated, but not for THIS patient.
    def missingImagesCountForPatient = false
    try { Meta.fromSamplesheetRow(noCountRow, 'image', 0, [channelsCount: [P1: 2], imagesCount: [P2: 9]]) }
    catch (IllegalArgumentException ignored) { missingImagesCountForPatient = true }
    assert missingImagesCountForPatient : 'Meta.fromSamplesheetRow must reject ctx.imagesCount missing THIS patient'

    // Satisfy: supply the entry, watch it pass.
    def satisfiedImagesMeta = Meta.fromSamplesheetRow(noCountRow, 'image', 0, [channelsCount: [P1: 2], imagesCount: [P1: 3]])
    assert satisfiedImagesMeta.images_count == 3

    // THE bug this round exists to kill: ctx.imagesCount = [P1: 0] must yield
    // images_count == 0, NOT 1. The old bare `?:` would have coerced this.
    def zeroImagesMeta = Meta.fromSamplesheetRow(noCountRow, 'image', 0, [channelsCount: [P1: 2], imagesCount: [P1: 0]])
    assert zeroImagesMeta.images_count == 0 : 'a genuine images_count of 0 must NOT be coerced to 1'

    // ---- Fix round 1, item 2: fromCheckpointRow validates against the SCHEMA --
    //
    // lib/Checkpoint.groovy owns the column list per step; fromCheckpointRow now
    // reads it from there instead of trusting whatever the row happens to carry.
    // RULING R17 landed in Task 4.3: every Checkpoint.STEPS schema now carries an
    // 'id' column, so these calls can finally reach a successful return (see the
    // full-construction case near the end of this block) -- the schema-level "this
    // step doesn't declare id yet" throw is no longer reachable for a REAL step
    // name (only Checkpoint.UnknownStepException remains reachable that way, via
    // the 'not_a_real_step' case below). What replaces it as the load-bearing
    // migration check is row.containsKey('id') -- a REAL old checkpoint FILE has no
    // 'id' column in its header, so splitCsv(header:true) never puts an 'id' key in
    // the row Map at all, distinct from "the column exists but this row's value is
    // blank" (a separate, more mundane problem covered further down).

    // 'preprocessed' declares is_reference AND channels (Checkpoint.columns).
    // Missing is_reference is caught BEFORE channels is even inspected.
    def missingIsReference = false
    try { Meta.fromCheckpointRow([patient_id: 'P1', channels: 'DAPI'], 'preprocessed', [:]) }
    catch (IllegalArgumentException ignored) { missingIsReference = true }
    assert missingIsReference : "Meta.fromCheckpointRow must reject a 'preprocessed' row with no is_reference"

    // Satisfy is_reference -- the failure moves to the id gate (IllegalStateException),
    // because this row (an id-less-file shape) still carries no 'id' key at all. The
    // message must be something a user can act on: name the fix, not just the symptom.
    def afterIsReferenceFixed = null
    try { Meta.fromCheckpointRow([patient_id: 'P1', is_reference: 'true', channels: 'DAPI'], 'preprocessed', [:]) }
    catch (IllegalStateException e) { afterIsReferenceFixed = e.message }
    assert afterIsReferenceFixed?.contains('predates identity tracking') &&
           afterIsReferenceFixed?.contains('re-run the step') :
        'fixing is_reference on an id-less preprocessed row must move the failure on to the id gate, with an actionable message'

    // Missing channels specifically (is_reference present) on the same schema.
    def missingChannelsCol = false
    try { Meta.fromCheckpointRow([patient_id: 'P1', is_reference: 'true'], 'preprocessed', [:]) }
    catch (IllegalArgumentException ignored) { missingChannelsCol = true }
    assert missingChannelsCol : "Meta.fromCheckpointRow must reject a 'preprocessed' row with no channels"

    // 'postprocessed' declares NEITHER is_reference NOR channels (Checkpoint.columns).
    // A row missing both must NOT be rejected for those -- it must fall straight
    // through to the id gate, proving the schema-conditional checks are actually
    // schema-conditional and not just always-on.
    def postprocessedFailure = null
    try { Meta.fromCheckpointRow([patient_id: 'P1'], 'postprocessed', [:]) }
    catch (IllegalStateException e) { postprocessedFailure = e.message }
    assert postprocessedFailure?.contains('predates identity tracking') :
        "a 'postprocessed' row with no is_reference/channels must fail on the id gate, not on those fields"

    // The column EXISTS (the row Map has an 'id' key) but this one row's VALUE is
    // blank -- a different, more mundane problem (a malformed/hand-edited file),
    // correctly caught by requirePresentInRow's generic message instead of the
    // richer "predates identity tracking" one.
    def blankIdValue = false
    try {
        Meta.fromCheckpointRow(
            [patient_id: 'P1', id: '', is_reference: 'true', channels: 'DAPI'],
            'preprocessed', [channelsCount: [P1: 1], imagesCount: [P1: 1]])
    }
    catch (IllegalArgumentException ignored) { blankIdValue = true }
    assert blankIdValue : 'a row with a blank id VALUE (column present) must fail requirePresentInRow, not the id-gate message'

    // Fully satisfied: fromCheckpointRow can now actually return a complete meta --
    // unreachable before Task 4.3 added the id column to every schema.
    def ckCtx = [
        keepChannelsBySlide: [P1: [P1_slide: ['DAPI', 'CD3']]],
        imagesCount        : [P1: 1],
        channelsCount      : [P1: 2],
    ]
    def ckRow  = [patient_id: 'P1', id: 'P1_slide', is_reference: 'true', channels: 'DAPI|CD3', pixel_size: '0.325']
    def ckMeta = Meta.fromCheckpointRow(ckRow, 'preprocessed', ckCtx)
    assert Meta.REQUIRED_KEYS.every { ckMeta.containsKey(it) }
    assert ckMeta.patient_id     == 'P1'
    assert ckMeta.id             == 'P1_slide'
    assert ckMeta.is_reference   == true
    assert ckMeta.channels       == ['DAPI', 'CD3']
    assert ckMeta.keep_channels  == ['DAPI', 'CD3']
    assert ckMeta.channels_count == 2
    assert ckMeta.images_count   == 1
    // A real number, not the raw CSV string.
    assert ckMeta.pixel_size == 0.325d

    // Old checkpoint, no 'pixel_size' key: throw, never fall back to params.pixel_size.
    def noPxRow = [patient_id: 'P1', id: 'P1_slide', is_reference: 'true', channels: 'DAPI|CD3']
    def missingPixelSize = false
    try { Meta.fromCheckpointRow(noPxRow, 'preprocessed', ckCtx) }
    catch (IllegalStateException e) { missingPixelSize = e.message.contains('predates scale tracking') }
    assert missingPixelSize

    // An unknown step name is rejected via Checkpoint.columns, not silently accepted.
    def unknownStep = false
    try { Meta.fromCheckpointRow([patient_id: 'P1'], 'not_a_real_step', [:]) }
    catch (IllegalArgumentException ignored) { unknownStep = true }
    assert unknownStep : 'Meta.fromCheckpointRow must reject an unknown step name'

    // ------------------------------------------------------------------ //
    // CsvUtils.metaContextFromCheckpoint (Task 4.3) -- the checkpoint-side
    // twin of the samplesheet-side ctx assembly INPUT_CHECK does by hand
    // (countImagesPerPatient / resolveKeptChannelsPerSlide / countChannelsPerPatient).
    // ------------------------------------------------------------------ //
    def ckpTmp = File.createTempFile('meta_ctx_checkpoint', '.csv')
    ckpTmp.deleteOnExit()
    ckpTmp.text =
        'patient_id,id,registered_image,is_reference,channels\n' +
        'P1,P1_ref,/x/ref.tiff,true,DAPI|PANCK|SMA\n' +
        // The moving slide re-declares DAPI (a re-stain) -- claim-once must drop it,
        // exactly as resolveKeptChannelsPerSlide does for a samplesheet's rows.
        'P1,P1_mov,/x/mov.tiff,false,DAPI|CD3\n'
    def checkpointCtx = CsvUtils.metaContextFromCheckpoint(ckpTmp.path, 'registered')
    assert checkpointCtx.imagesCount == [P1: 2]
    assert checkpointCtx.channelsCount == [P1: 4]  // 3 (ref) + 1 (mov's CD3; DAPI already claimed)
    assert checkpointCtx.keepChannelsBySlide.P1.P1_ref == ['DAPI', 'PANCK', 'SMA']
    assert checkpointCtx.keepChannelsBySlide.P1.P1_mov == ['CD3']

    // A nonexistent file must degrade to the same empty shape every other CsvUtils
    // reader uses for "file not found" -- never throw, since a caller learns that
    // from Meta.fromCheckpointRow's own per-patient-count checks instead.
    def missingCtx = CsvUtils.metaContextFromCheckpoint('/no/such/checkpoint.csv', 'registered')
    assert missingCtx == [keepChannelsBySlide: [:], imagesCount: [:], channelsCount: [:]]

    // A schema with no channels/is_reference ('postprocessed') must not throw --
    // it degrades to empty keep-sets/zero counts rather than being stricter than
    // the Meta.fromCheckpointRow it feeds.
    def ppTmp = File.createTempFile('meta_ctx_postprocessed', '.csv')
    ppTmp.deleteOnExit()
    ppTmp.text = 'patient_id,id,cell_csv,cell_geojson,merged_csv,cell_mask,pyramid\n' +
        'P1,P1,/x/a.csv,/x/b.geojson,/x/c.csv,/x/d.tif,/x/e.tiff\n'
    def ppCtx = CsvUtils.metaContextFromCheckpoint(ppTmp.path, 'postprocessed')
    assert ppCtx.imagesCount == [P1: 1]
    assert ppCtx.channelsCount == [P1: 0]
    assert ppCtx.keepChannelsBySlide == [P1: [P1: []]]

    // Fail loudly, naming the missing key, rather than defaulting.
    def missingPatientId = false
    try { Meta.fromSamplesheetRow([image: 'x.tiff', channels: 'DAPI'], 'image', 0, [:]) }
    catch (IllegalArgumentException ignored) { missingPatientId = true }
    assert missingPatientId : 'Meta.fromSamplesheetRow must reject a row with no patient_id'

    def missingImageCol = false
    try { Meta.fromSamplesheetRow([patient_id: 'P1', channels: 'DAPI'], 'image', 0, [:]) }
    catch (IllegalArgumentException ignored) { missingImageCol = true }
    assert missingImageCol : 'Meta.fromSamplesheetRow must reject a row missing the image column'

    def blankStep = false
    try { Meta.fromCheckpointRow([patient_id: 'P1', id: 'x'], '  ', [:]) }
    catch (IllegalArgumentException ignored) { blankStep = true }
    assert blankStep : 'Meta.fromCheckpointRow must reject a blank step'

    // ------------------------------------------------------------------ //
    // ResourceReport — see checkResourceReport() above workflow{}; kept out of
    // this block because it is close to the JVM class-file string-constant
    // ceiling (see the file header note above).
    // ------------------------------------------------------------------ //
    checkResourceReport()

    // ProcessEnvelope — size-log row + container identity in versions.yml.
    // See checkProcessEnvelope() above the workflow block, and the header
    // comment's "HARD SIZE CEILING" note for why it lives outside this block.
    checkProcessEnvelope()

    // RegisteredMatch — pairing VALIS's outputs back to their slide metas.
    // See checkRegisteredMatch() above the workflow block.
    checkRegisteredMatch()

    // Checkpoint.requireColumns — the drift guard three readers copied.
    // See checkCheckpoint() above the workflow block.
    checkCheckpoint()

    // RegBackends — the registration backend's identity, in one table.
    // See checkRegBackends() above the workflow block.
    checkRegBackends()

    // CsvUtils.unknownColumns — report, never reject.
    // See checkCsvUtilsUnknownColumns() above the workflow block.
    checkCsvUtilsUnknownColumns()

    // println, NOT log.info: nf-test's underlying `nextflow ... -quiet` run
    // suppresses log.info from stdout entirely (observed directly: a log.info
    // line here never appears in workflow.stdout under nf-test, even though the
    // exact same script printed it fine under a plain `nextflow run`). println
    // writes straight to stdout regardless of -quiet, so it is what
    // tests/lib_probe.nf.test's `workflow.stdout.any { ... }` assertion needs.
    println "LIB PROBE: all assertions passed"
}

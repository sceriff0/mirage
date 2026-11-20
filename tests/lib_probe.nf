/*
    lib/ probe: the unit-test surface for lib/*.groovy.

    nf-test's assertion context cannot see lib/ (`No such property: Layout`), but a
    Nextflow script is compiled with lib/ on the classpath, so the assertions live
    here and tests/lib_probe.nf.test only asserts that this script ran.

    Run directly:   nextflow run tests/lib_probe.nf -lib lib
    Run in CI:      the nextflow-stub job, and tests/lib_probe.nf.test (tag: stub)

    No processes. Each statement is a call into a lib/ class plus an `assert`; a
    failing assert exits nonzero. Some checks write and delete temp files.

    Must parse on Nextflow 25 and 26: `assert cond : 'msg'` and `closure.call(x)`,
    never `assert cond, 'msg'` or `closure(x)`.

    Size ceiling: Nextflow 25's parser stores the workflow{} body as one Java string
    constant, at most 65535 bytes of modified UTF-8 (an em-dash costs 3 bytes). Keep
    that block ASCII and short; new checks go in a top-level `def checkX()` called
    from it, which does not count. tests/test_lib_probe_parses_on_nf26.py pins a margin.
*/

/** ResourceReport: trace-path resolution and the missing-trace diagnostic. */
def checkResourceReport() {

    // Nextflow resolves trace.file against launchDir, not the JVM's working directory;
    // these cases pin that rule.
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

    // The diagnostic must name the path and say whether tracing was on, so an operator
    // can tell a wrong trace_dir from a disabled trace.
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

/** ProcessEnvelope: the size-log row and container identity in versions.yml. */
def checkProcessEnvelope() {
    // The size-log row, asserted byte for byte: it is shell text that no `-stub` run or
    // snapshot evaluates, so wrong `$(`/`${` escaping would silently publish the command
    // text instead of a byte count.
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

    // The stub row carries the same process name as the real row, so the CSVs compare.
    assert ProcessEnvelope.sizeLogStub('MIRAGE:PRE:CONVERT_IMAGE', 'P001', 'P001_slide.CONVERT_IMAGE.size.csv') ==
           'echo "MIRAGE:PRE:CONVERT_IMAGE,P001,stub,0" > P001_slide.CONVERT_IMAGE.size.csv'

    // An empty path list is a caller bug, not an empty measurement: `stat` with no
    // operands would report 0 bytes for a task that certainly read something.
    def noPaths = false
    try { ProcessEnvelope.sizeLog('P', 'S', [], 'x.size.csv') }
    catch (IllegalArgumentException ignored) { noPaths = true }
    assert noPaths : 'ProcessEnvelope.sizeLog must reject an empty shellPaths list'

    // Container identity in versions.yml: image tags are descriptive, not immutable, so
    // the outputs themselves must record which image produced them.
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

    // The stub reports `stub`: a stub run is no evidence about any image.
    def stubContainer = ProcessEnvelope.versionsStub('MIRAGE:POST:MERGE_AND_PYRAMID',
                                                     ['tifffile'], 'bolt3x/mirage-merge:1.0.0')
    assert stubContainer.readLines()[3] == '    container: stub'
    assert !stubContainer.contains('bolt3x')

    // The bash-only pair, for an image with no Python: versions() would record
    // "bash: python: command not found" as a version number.
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

/** RegisteredMatch: pairing VALIS's outputs back to their slide metas. */
def checkRegisteredMatch() {
    // By channel signature, not filename: VALIS renames its outputs, and the OME channel
    // set (bin/create_channels_manifest.py) is the only identity that survives. A wrong
    // pairing does not fail; it attaches the wrong metadata to a registered image.
    assert RegisteredMatch.signature(['PANCK', 'dapi', 'CD3']) == 'cd3|dapi|panck'
    // Lower-case FIRST, then sort: 'CD3' and 'cd3' must land in the same place.
    assert RegisteredMatch.signature(['cd3', 'DAPI']) == RegisteredMatch.signature(['CD3', 'dapi'])

    // toSorted(), never sort(): meta.channels is shared across metas built with
    // `meta + [k: v]` (a shallow copy), and channel order is what --dapi-channel indexes.
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

    // Exception 3c: two manifest entries with the same FILE signature collapse to one map
    // slot, so movMeta has no file left and the "unmatched" throw fires.
    def dupFileSig = ''
    try {
        RegisteredMatch.pair([refMeta, movMeta], [refFile, movFile],
                             ['P1_DAPI_PANCK_SMA_registered.ome.tiff': ['DAPI', 'PANCK', 'SMA'],
                              'P1_DAPI_CD3_registered.ome.tiff'      : ['DAPI', 'PANCK', 'SMA']])
    }
    catch (IllegalStateException e) { dupFileSig = e.message }
    assert dupFileSig.startsWith('RegisteredMatch: unmatched') : "got: ${dupFileSig}"
}

/** Checkpoint.requireColumns: the missing-column check checkpoint readers share. */
def checkRegQcLevel() {
    // reg_qc is the one QC switch: 0 turns it off, null means the default 2.
    assert ParamUtils.regQcLevelOf(0) == 0 : 'reg_qc 0 must mean no registration QC'
    assert ParamUtils.regQcLevelOf(1) == 1 : 'reg_qc 1 must stay 1'
    assert ParamUtils.regQcLevelOf(null) == 2 : 'unset reg_qc must mean 2'
    assert ParamUtils.regQcLevel([reg_qc: 0]) == 0 : 'map form must agree with regQcLevelOf'
}

def checkCheckpoint() {

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
 * RegBackends: the registration backend's identity, in one table. A method the
 * schema knows but a dispatch site does not must throw, not fall through to VALIS.
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

    // Which backend pairs its registered outputs back to metas by channel SET
    // (RegisteredMatch), and so cannot accept two slides of one patient with the same
    // set. VALIS renames its outputs; the tiled adapter carries the meta itself.
    assert RegBackends.of('valis').pairsOutputsBySignature :
        'VALIS pairs by signature -- that is what RegisteredMatch exists for'
    assert !RegBackends.of('tiled').pairsOutputsBySignature :
        'the tiled adapter keeps meta through its fan-out and never pairs by signature'

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

// CsvUtils.unknownColumns: report, never reject.
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

// CsvUtils.validateInputSemantics: the duplicate-channel refusal. A non-nuclear
// marker on two slides of one patient has no safe interpretation (only one copy is
// kept, chosen by order), so it is refused at launch. Nuclear markers may repeat:
// DAPI is the registration fiducial.
def checkDuplicateChannelRefusal() {

    // validateInputSemantics also checks that each row's image EXISTS, so the two
    // slides need real files; their contents are never read.
    def dupDir = File.createTempDir()
    ['P001_ref.ome.tiff', 'P001_mov1.ome.tiff'].each { name ->
        new File(dupDir, name).text = 'not an image; only its existence is validated'
    }

    // One two-slide P001 sheet per case, returning its path.
    def sheetFor = { String tag, String refChannels, String movChannels ->
        def f = new File(dupDir, "dup_${tag}.csv")
        f.text = 'patient_id,path_to_file,is_reference,channels\n' +
                 "P001,${dupDir.path}/P001_ref.ome.tiff,true,${refChannels}\n" +
                 "P001,${dupDir.path}/P001_mov1.ome.tiff,false,${movChannels}\n"
        return f.path
    }

    // The refusal message, or null when the sheet was accepted.
    def refusalFor = { String sheetPath, def markers ->
        def message = null
        try { CsvUtils.validateInputSemantics(sheetPath, 'preprocessing', false, markers) }
        catch (IllegalArgumentException e) { message = e.message }
        return message
    }

    def nuclear = ['DAPI', 'CELLTOX']

    // (a) DAPI on both slides is the ORDINARY cyclic-IF sheet: accepted.
    CsvUtils.validateInputSemantics(
        sheetFor.call('nuclear', 'DAPI|PANCK|SMA', 'DAPI|CD3|CD8'), 'preprocessing', false, nuclear)

    // (b) PANCK on both slides is refused, naming the patient, the channel and BOTH rows.
    def panck = refusalFor.call(
        sheetFor.call('panck', 'DAPI|PANCK|SMA', 'DAPI|PANCK|CD3'), nuclear)
    assert panck != null : 'a non-nuclear channel on two slides of one patient must be refused'
    assert panck.contains('P001') : "the refusal must name the patient: ${panck}"
    assert panck.contains('PANCK') : "the refusal must name the channel: ${panck}"
    assert panck.contains('row 2') && panck.contains('row 3') :
        "the refusal must name both offending rows: ${panck}"
    assert panck.contains('P001_ref.ome.tiff') && panck.contains('P001_mov1.ome.tiff') :
        "the refusal must name both slides: ${panck}"
    assert panck.contains('nuclear_markers') : "the refusal must name the remedy: ${panck}"

    // (c) Case and surrounding whitespace must not hide the duplicate: the comparison
    // is trimmed and upper-cased, exactly as resolveKeptChannelsPerSlide's keep-set is.
    def cased = refusalFor.call(
        sheetFor.call('cased', 'DAPI| panck |SMA', 'DAPI|PANCK|CD3'), nuclear)
    assert cased != null : 'a case/whitespace variant of the same channel is still a duplicate'
    assert cased.contains('P001') && cased.contains('panck') :
        "the refusal names the channel as WRITTEN on its first slide: ${cased}"

    // (d) A scalar params.nuclear_markers (the only shape a command line can supply)
    // still classifies DAPI as nuclear -- MarkerUtils.markerList normalises it, so the
    // sheet is accepted rather than refused on a String iterated as characters.
    CsvUtils.validateInputSemantics(
        sheetFor.call('scalar', 'DAPI|PANCK|SMA', 'DAPI|CD3|CD8'), 'preprocessing', false, 'DAPI')

    // (e) A name repeated WITHIN one slide's own channels cell is refused whatever it
    // is -- nuclear or not. validateMetadata checks for blank names and for the
    // PRESENCE of a nuclear marker, but never for a repeat, so this rule lives here.
    def within = refusalFor.call(
        sheetFor.call('within', 'DAPI|PANCK|PANCK', 'DAPI|CD3'), nuclear)
    assert within != null : 'a channel repeated within one slide must be refused'
    assert within.contains('PANCK') && within.contains('row 2') :
        "the within-slide refusal must name the channel and its row: ${within}"
    def withinNuclear = refusalFor.call(
        sheetFor.call('withinnuc', 'DAPI|DAPI|PANCK', 'DAPI|CD3'), nuclear)
    assert withinNuclear != null : 'the within-slide rule does not exempt a nuclear marker'

    // Delete the temp tree (two images and six CSVs per run).
    dupDir.deleteDir()

    println 'LIB PROBE: checkDuplicateChannelRefusal passed'
}

// Direct cases for validators with no other unit-test surface: validatePixelSize,
// validateCleanup, isEntryPoint, CsvUtils.parseIsReference and Layout's private
// requireOutdir (through publishedPath()).
def checkParamValidators() {

    // ------------------------------------------------------------------ //
    // ParamUtils.validatePixelSize: 'auto' in any case, or a positive number.
    // ------------------------------------------------------------------ //
    ['auto', 'AUTO', ' Auto ', '0.5', ' 0.325 ', 0.325, 1].each { v ->
        ParamUtils.validatePixelSize([pixel_size: v])
    }
    [null, '', '   ', '0', '-1', '-0.5', 'abc', '0.5um'].each { v ->
        def rejected = false
        try { ParamUtils.validatePixelSize([pixel_size: v]) }
        catch (IllegalArgumentException e) { rejected = e.message.contains('--pixel_size') }
        assert rejected : "validatePixelSize must reject ${v.inspect()} and name --pixel_size"
    }

    // ------------------------------------------------------------------ //
    // ParamUtils.validateCleanup: exactly Layout.CLEANUP_LEVELS, case-sensitive.
    // ------------------------------------------------------------------ //
    Layout.CLEANUP_LEVELS.each { level ->
        ParamUtils.validateCleanup([cleanup_level: level])
    }
    ['FINAL', 'None', 'garbage', '', null].each { v ->
        def rejected = false
        try { ParamUtils.validateCleanup([cleanup_level: v]) }
        catch (IllegalArgumentException e) {
            rejected = Layout.CLEANUP_LEVELS.every { e.message.contains(it) }
        }
        assert rejected : "validateCleanup must reject ${v.inspect()} and list every valid level"
    }

    // ------------------------------------------------------------------ //
    // ParamUtils.isEntryPoint: true for params.start only.
    // ------------------------------------------------------------------ //
    ParamUtils.STEP_ORDER.each { step ->
        assert ParamUtils.isEntryPoint([start: step], step) : "isEntryPoint must be true for its own step ${step}"
        ParamUtils.STEP_ORDER.findAll { it != step }.each { other ->
            assert !ParamUtils.isEntryPoint([start: step], other) : "isEntryPoint(${step}) must be false for ${other}"
        }
    }

    // ------------------------------------------------------------------ //
    // CsvUtils.parseIsReference: trimmed, lower-cased, then exactly true/false.
    // ------------------------------------------------------------------ //
    assert CsvUtils.parseIsReference('true', 'row 1')   == true
    assert CsvUtils.parseIsReference('false', 'row 1')  == false
    assert CsvUtils.parseIsReference(' TRUE ', 'row 1') == true  : 'the parser trims and lower-cases'
    assert CsvUtils.parseIsReference('False', 'row 1')  == false
    // No 'TRUE ' here: it trims to a valid value and is ACCEPTED above.
    ['yes', 'no', '1', '0', 'y', '', null, 'reference'].each { v ->
        def rejected = false
        try { CsvUtils.parseIsReference(v, 'samplesheet.csv row 3') }
        catch (IllegalArgumentException e) {
            rejected = e.message.contains("Must be 'true' or 'false'") && e.message.contains('samplesheet.csv row 3')
        }
        assert rejected : "parseIsReference must reject ${v.inspect()} naming the context"
    }

    // ------------------------------------------------------------------ //
    // Layout.requireOutdir is private; every public path builder goes through it.
    // ------------------------------------------------------------------ //
    // A real Path, not a bare String: producerSubdir/basename read .parent/.name.
    def geojsonFile = file('cells.geojson')
    [null, '', '   '].each { v ->
        def rejected = false
        try { Layout.publishedPath(v, 'P001', 'geojson', geojsonFile) }
        catch (IllegalArgumentException e) { rejected = e.message.contains('output directory') }
        assert rejected : "Layout must refuse an outdir of ${v.inspect()}"
    }
    assert Layout.publishedPath('/out/', 'P001', 'geojson', geojsonFile) ==
           Layout.publishedPath('/out', 'P001', 'geojson', geojsonFile) : 'a trailing slash is stripped'

    // The duplicate-channel refusal; kept out of workflow{} for its size ceiling.
    checkDuplicateChannelRefusal()

    println 'LIB PROBE: checkParamValidators passed'
}

// CsvUtils.validateInputSemantics with requireUniqueChannelSets. Rule B (opt-in, two
// slides sharing one whole channel set) runs before Rule A (always on, a non-nuclear
// channel shared across a patient's slides). Rows 3 and 4 here trip both, so `false`
// is refused by Rule A and `true` by Rule B with its own message.
def checkUniqueChannelSets() {
    def ref  = File.createTempFile('dupsig_ref', '.tiff'); ref.text = 'x'
    def mov1 = File.createTempFile('dupsig_mov1', '.tiff'); mov1.text = 'x'
    def mov2 = File.createTempFile('dupsig_mov2', '.tiff'); mov2.text = 'x'
    def csv = File.createTempFile('dupsig', '.csv')
    csv.text = """patient_id,path_to_file,is_reference,channels
P7,${ref.path},true,DAPI|PANCK|SMA
P7,${mov1.path},false,DAPI|CD3|CD8
P7,${mov2.path},false,CD8|dapi|CD3
"""
    // default and explicit false: Rule B is off, but Rule A refuses CD3, the first
    // non-nuclear channel repeated across the patient's slides (DAPI is nuclear, skipped).
    [null, false].each { arg ->
        def refusedA = false
        def msgA = ''
        try {
            arg == null ? CsvUtils.validateInputSemantics(csv.path, 'preprocessing', false, 'DAPI')
                        : CsvUtils.validateInputSemantics(csv.path, 'preprocessing', false, 'DAPI', arg)
        }
        catch (IllegalArgumentException e) { refusedA = true; msgA = e.message }
        assert refusedA : "Rule A must refuse shared non-nuclear channels even when requireUniqueChannelSets is ${arg.inspect()}"
        assert msgA.contains("Channel 'CD3'")            : "must name CD3, the first non-nuclear repeat in declaration order: ${msgA}"
        assert msgA.contains('patient P7')               : "must name the patient: ${msgA}"
        assert msgA.contains('appears on 2 different slides') : "must use Rule A's wording: ${msgA}"
        assert msgA.contains('row 3') && msgA.contains('row 4') : "must name both rows: ${msgA}"
    }

    def refused = false
    def msg = ''
    try { CsvUtils.validateInputSemantics(csv.path, 'preprocessing', false, 'DAPI', true) }
    catch (IllegalStateException e) { refused = true; msg = e.message }
    assert refused : 'two slides of one patient with one channel set must be refused when the backend pairs by signature'
    assert msg.contains('patient P7')      : "must name the patient: ${msg}"
    assert msg.contains('cd3|cd8|dapi')    : "must name the signature as RegisteredMatch computes it: ${msg}"
    assert msg.contains('row 3') && msg.contains('row 4') : "must name both rows: ${msg}"
    assert !msg.contains('row 2')          : "the reference has a different set and must not be named: ${msg}"

    // a second patient with its own copy of the set is not a collision across patients
    def csv2 = File.createTempFile('dupsig_two_patients', '.csv')
    csv2.text = """patient_id,path_to_file,is_reference,channels
P7,${ref.path},true,DAPI|CD3|CD8
P8,${mov1.path},true,DAPI|CD3|CD8
"""
    CsvUtils.validateInputSemantics(csv2.path, 'preprocessing', false, 'DAPI', true)
    [ref, mov1, mov2, csv, csv2]*.delete()
}

// Checks called from workflow{}, kept out of it for its size ceiling (file header).

def checkLayoutAndMarkerUtils() {
    // ------------------------------------------------------------------ //
    // Layout - checkpoint paths
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
    // MarkerUtils - the nuclear-marker rule
    // ------------------------------------------------------------------ //
    // markerList normalises three incompatible shapes. The bare-String case is the
    // dangerous one: iterated as a List it yields CHARACTERS, so 'PANCK' would match
    // any channel containing 'C'.
    assert MarkerUtils.markerList(['DAPI', 'CELLTOX']) == ['DAPI', 'CELLTOX']
    assert MarkerUtils.markerList('DAPI,CELLTOX')      == ['DAPI', 'CELLTOX']
    assert MarkerUtils.markerList('DAPI CELLTOX')      == ['DAPI', 'CELLTOX']
    assert MarkerUtils.markerList('PANCK')             == ['PANCK']

    // Matching is case-insensitive substring: 'DAPI_nuclear' is nuclear.
    assert MarkerUtils.isNuclear('DAPI_nuclear', ['DAPI'])
    assert MarkerUtils.isNuclear('dapi', ['DAPI'])
    assert !MarkerUtils.isNuclear('CD3', ['DAPI'])

    // Marker preference order beats channel order: markers ['DAPI','CELLTOX'] over
    // channels ['CELLTOX','CD8','DAPI'] gives 2, not 0. CellSAM passes this straight to
    // --dapi-channel, so a wrong answer segments the wrong channel silently.
    assert MarkerUtils.nuclearIndex(['CELLTOX', 'CD8', 'DAPI'], ['DAPI', 'CELLTOX']) == 2
    assert MarkerUtils.nuclearIndex(['CD8', 'CELLTOX'], ['DAPI', 'CELLTOX'])         == 1

    assert MarkerUtils.hasNuclear(['CD3', 'DAPI'], ['DAPI'])
    assert !MarkerUtils.hasNuclear(['CD3', 'CD8'], ['DAPI'])


}

def checkKeepSetRule() {
    // ------------------------------------------------------------------ //
    // CsvUtils.resolveKeptChannelsPerSlide - the keep-set rule
    // ------------------------------------------------------------------ //
    // Each marker name is claimed once per patient, reference first then samplesheet
    // order. That makes the winner of a shared marker deterministic, and makes the
    // distinct-name count equal the emitted-file count, so one channels_count sizes
    // every downstream groupKey.
    // The inner map is keyed on Meta.identityFor's output (meta.id). For a
    // non-colliding stem identityFor ignores rowIndex, so `0` stands in below.
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

    // preClaimed seeds the claimed set - add_cycle passes the prior run's reference
    // channels, so a re-stained DAPI is redundant but a NEW nuclear marker survives.
    def keepPrior = File.createTempFile('keepprior', '.csv')
    keepPrior.text = '''patient_id,image,channels,is_reference
P3,cyc4.tiff,DAPI|CELLTOX|CD8,false
'''
    def seeded = CsvUtils.resolveKeptChannelsPerSlide(
        keepPrior.path, 'image', ['DAPI','CELLTOX'], ['P3': ['DAPI', 'KI67']])
    assert seeded['P3'][Meta.identityFor('P3', 'cyc4.tiff', 0, [:])] == ['CELLTOX', 'CD8']  // DAPI pre-claimed; CELLTOX new
    keepPrior.delete()

    // Basename collision: two rows of one patient can share an image basename in
    // different directories (one directory per cycle is ordinary). identityFor adds the
    // row index when stems collide, so both rows are distinct entries.
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

    // Raw-cell collision: two rows sharing the exact same image cell (or both blank)
    // must stay two entries. P6's second row declares only DAPI, already claimed, so its
    // keep-set is legitimately []; keying on the assigned identity stops it overwriting
    // the reference row's ['DAPI','CD3'].
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

    // The block above exercises resolveKeptChannelsPerSlide's own row index. The
    // production call site, input_check.nf, gets rowIndex from
    // CsvUtils.rowIndexPerPatient; two rows sharing a raw cell must get distinct indexes
    // there too, or Meta.fromSamplesheetRow gives them the same meta.id. Pinned directly
    // below, and end to end by input_check.nf.test's "two rows sharing the identical raw
    // path cell get distinct id and keep-set" case.
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

    // An empty keep-set is an answer, not an absence: a slide whose channels were all
    // claimed must have a present, empty entry, so consumers can tell "emit nothing" from
    // "no entry, use the declared list". Groovy's `?:` treats [] as falsy, so lookups use
    // containsKey or an explicit null test. Falling back to the declared list would emit
    // a duplicate marker name across two slides.
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

    // The invariant countChannelsPerPatient exists for: the count equals both the number
    // of TIFFs emitted and the number of distinct marker names. Both are pinned so a
    // resolver change cannot make channels_count right for one grouping only.
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

    // The same invariant for a name repeated within one row (`DAPI|DAPI|KI67`), which the
    // cross-slide case above never exercises: the claimed set must update per channel,
    // not per row. P9 repeats DAPI on the reference row and CELLTOX on the second row,
    // so both slides are exercised.
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

}

def checkSegQcNucleiDir() {
    // unset: nothing to check
    ParamUtils.validateSegQcNucleiDir([reg_qc: 2])
    // an existing directory at reg_qc=2: accepted
    def d = File.createTempDir()
    ParamUtils.validateSegQcNucleiDir([reg_qc: 2, seg_qc_nuclei_dir: d.path])
    // a missing directory: refused, naming it
    def msg = ''
    try { ParamUtils.validateSegQcNucleiDir([reg_qc: 2, seg_qc_nuclei_dir: d.path + '_nope']) }
    catch (IllegalArgumentException e) { msg = e.message }
    assert msg.contains('is not a directory') : "a missing --seg_qc_nuclei_dir must be refused: ${msg}"
    // set below reg_qc=2: refused (nothing reads it)
    msg = ''
    try { ParamUtils.validateSegQcNucleiDir([reg_qc: 1, seg_qc_nuclei_dir: d.path]) }
    catch (IllegalArgumentException e) { msg = e.message }
    assert msg.contains('only reg_qc=2') : "seg_qc_nuclei_dir below reg_qc=2 must be refused: ${msg}"
    d.deleteDir()
    println "LIB PROBE: checkSegQcNucleiDir passed"
}

def checkPassthroughPath() {
    // Layout.passthroughPath: PREPROCESSED when BaSiC ran, 'converted' when
    // skip_preprocessing (the default) left CONVERT_IMAGE's output as the slide.
    def fresh = file("/work/ab/${'c' * 30}/P001_ref.ome.tif")
    assert Layout.passthroughPath('/out', 'P001', fresh, false) ==
        Layout.publishedOrAsIs('/out', 'P001', Layout.PREPROCESSED, fresh)
    assert Layout.passthroughPath('/out', 'P001', fresh, false) == '/out/P001/preprocessed/P001_ref.ome.tif'
    assert Layout.passthroughPath('/out', 'P001', fresh, true) == '/out/P001/converted/P001_ref.ome.tif'
    // an already-published path (a --start samplesheet) is recorded as is, whatever the kind
    def prior = file('/prior/P002/preprocessed/P002_ref_corrected.ome.tif')
    assert Layout.passthroughPath('/out', 'P002', prior, true) == prior.toString()
    println "LIB PROBE: checkPassthroughPath passed"
}

// ParamUtils.checkRetiredParams: a stale reg_tiled_mode must never silently run the
// high values. medium/low/custom (or anything else) throw naming the knobs to set,
// high returns a warning, absent returns null.
def checkRetiredStareTier() {
    assert ParamUtils.checkRetiredParams([:]) == null : "absent reg_tiled_mode must be a no-op"
    assert ParamUtils.checkRetiredParams([reg_tiled_tile: 512]) == null : "a live STARE knob is not retired"
    def warn = ParamUtils.checkRetiredParams([reg_tiled_mode: 'high'])
    assert warn != null && warn.contains('RETIRED') : "high must warn, not throw: ${warn}"
    def expect = [
        medium: 'reg_tiled_tile 1024, reg_tiled_halo 192, reg_tiled_out_tile 768, reg_tiled_coarse_max_dim 1024',
        low   : 'reg_tiled_tile 512, reg_tiled_halo 128, reg_tiled_out_tile 512, reg_tiled_coarse_max_dim 512',
        custom: 'set the knobs you overrode directly',
        bogus : 'set the knobs you overrode directly',
    ]
    expect.each { mode, needle ->
        def msg = null
        try { ParamUtils.checkRetiredParams([reg_tiled_mode: mode]) }
        catch (IllegalArgumentException e) { msg = e.message }
        assert msg != null : "reg_tiled_mode=${mode} must be REFUSED, not run with the high values"
        assert msg.contains('no longer supported') && msg.contains(needle) :
            "reg_tiled_mode=${mode}: the refusal must name ${needle}; got: ${msg}"
    }
    // a CLI value arrives as a String with whatever whitespace the shell left
    assert ParamUtils.checkRetiredParams([reg_tiled_mode: ' high ']) != null : "trimmed high warns"
    println "LIB PROBE: checkRetiredStareTier passed"
}

workflow {

    // Layout checkpoint paths, MarkerUtils and the keep-set rule (functions above).
    checkLayoutAndMarkerUtils()
    checkKeepSetRule()

    // ------------------------------------------------------------------ //
    // ParamUtils - the step vocabulary
    // ------------------------------------------------------------------ //
    assert ParamUtils.STEP_ORDER == ['preprocessing', 'registration', 'segmentation', 'postprocessing']
    assert ParamUtils.entryColumnForStep('registration')      == 'preprocessed_image'
    assert ParamUtils.requiredColumnsForStep('preprocessing') ==
        ['patient_id', 'path_to_file', 'is_reference', 'channels']

    assert  ParamUtils.shouldRun('registration',   'preprocessing', 'postprocessing')
    assert !ParamUtils.shouldRun('preprocessing',  'registration',  'postprocessing')
    assert !ParamUtils.shouldRun('postprocessing', 'preprocessing', 'registration')

    // Use a name absent from STEPS ('segmentation' is a real step).
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
    // Checkpoint - filename AND columns, one owner
    // ------------------------------------------------------------------ //
    assert Checkpoint.columns(Layout.PREPROCESSED) ==
        ['patient_id', 'id', 'preprocessed_image', 'is_reference', 'channels', 'pixel_size']
    assert Checkpoint.columns(Layout.REGISTERED) ==
        ['patient_id', 'id', 'registered_image', 'is_reference', 'channels', 'pixel_size']
    assert Checkpoint.columns(Layout.POSTPROCESSED) ==
        ['patient_id', 'id', 'cell_csv', 'cell_geojson', 'merged_csv', 'cell_mask', 'pyramid', 'pixel_size']

    // The header is the seed string the writers pass to collectFile; these literals
    // are the published contract.
    assert Checkpoint.header(Layout.PREPROCESSED)  == 'patient_id,id,preprocessed_image,is_reference,channels,pixel_size'
    assert Checkpoint.header(Layout.REGISTERED)    == 'patient_id,id,registered_image,is_reference,channels,pixel_size'
    assert Checkpoint.header(Layout.POSTPROCESSED) == 'patient_id,id,cell_csv,cell_geojson,merged_csv,cell_mask,pyramid,pixel_size'

    // row() emits values in declared column order regardless of map insertion order.
    assert Checkpoint.row(Layout.REGISTERED, [
        channels: 'DAPI|CD3', patient_id: 'P001', id: 'P001_x',
        registered_image: '/out/P001/registered/x.ome.tiff', is_reference: false,
        pixel_size: 0.325,
    ]) == 'P001,P001_x,/out/P001/registered/x.ome.tiff,false,DAPI|CD3,0.325'

    // RFC 4180 quoting: --outdir is an arbitrary path, so a value containing a comma is
    // quoted and the field count survives read-back (splitCsv parses RFC 4180 natively).
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

    // A null value (unlike a missing key) writes an empty field, not the text `null`:
    // empty is what "artifact not produced" means to every reader.
    assert Checkpoint.row(Layout.REGISTERED, [
        patient_id: 'P001', id: 'P001_x', registered_image: null,
        is_reference: false, channels: 'DAPI|CD3', pixel_size: 0.325,
    ]) == 'P001,P001_x,,false,DAPI|CD3,0.325'

    // A missing column must throw: an empty field would be a checkpoint row naming a
    // path that does not exist.
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

    // Checkpoint and Layout must agree on the step vocabulary.
    assert Checkpoint.STEPS*.name == Layout.CHECKPOINT_STEPS

    // A step's requiredColumns equal the previous step's checkpoint columns except 'id'
    // and 'pixel_size': registration and segmentation enter through INPUT_CHECK, which
    // derives id itself and re-resolves scale via PREFLIGHT_SCALE, so requiring either
    // would reject an older checkpoint INPUT_CHECK can still read.
    assert ParamUtils.STEPS.find { it.name == 'registration' }.requiredColumns ==
        Checkpoint.columns(Layout.PREPROCESSED) - ['id', 'pixel_size']
    assert ParamUtils.STEPS.find { it.name == 'segmentation' }.requiredColumns ==
        Checkpoint.columns(Layout.REGISTERED) - ['id', 'pixel_size']
    assert ParamUtils.STEPS.find { it.name == 'postprocessing' }.requiredColumns ==
        ['patient_id', 'id', 'registered_image', 'is_reference', 'channels', 'cell_mask', 'nuclei_mask']
    assert ParamUtils.STEPS.find { it.name == 'postprocessing' }.requiredColumns
        .every { it in Checkpoint.columns(Layout.SEGMENTED) }

    // columns() must hand out an immutable list: a caller mutating it would corrupt the
    // schema every later header()/row() call sees.
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
    // Order matters: the kind order is the report-slot order, so this is an ordered
    // equality, which also rules out duplicates.
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

    // Empty string means "artifact not produced" - nucleus_contours is empty when
    // --quantify_compartments is false (nuclei_mask is NOT similarly gated: SEGMENT
    // always produces it - see Checkpoint's EMPTY VALUES note). row() must accept ''
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
    // Layout.isUnderTaskDir / publishedOrAsIs: a fresh output one level down
    // (REGISTER's registered_slides/) and an already-published path with the same
    // subdirectory name both have a non-hash parent; only the full check tells them
    // apart. A parent-only isTaskDir check fails case 2.
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

    // 2. Fresh, one level down: REGISTER's registered_slides/ (a parent-only check fails it).
    def freshNested = file("${workDirPre}/registered_slides/P001_x_registered.ome.tiff")
    assert Layout.isUnderTaskDir(freshNested.parent)
    assert Layout.publishedOrAsIs('/out', 'P001', Layout.REGISTERED, freshNested) ==
        '/out/P001/registered/registered_slides/P001_x_registered.ome.tiff'

    // 3. Already published, same subdirectory name as case 2 (a prior run's registered.csv
    // read back by --start segmentation): returned as is, not rebuilt against the new outdir.
    def publishedNested = file('/prior/P001/registered/registered_slides/P001_x_registered.ome.tiff')
    assert !Layout.isUnderTaskDir(publishedNested.parent)
    assert Layout.publishedOrAsIs('/out', 'P001', Layout.REGISTERED, publishedNested) ==
        '/prior/P001/registered/registered_slides/P001_x_registered.ome.tiff'

    // 4. A single-slide passthrough published under preprocessed/ but asked about with
    // kind REGISTERED: the as-is branch ignores the kind, which makes it harmless.
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

    // 6. Fresh, one level down via nuclei/ (EXTRACT_NUCLEI_PROPERTIES's subdirectory).
    def freshNuclei = file("${workDirPre}/nuclei/contours.json")
    assert Layout.isUnderTaskDir(freshNuclei.parent)
    assert Layout.publishedOrAsIs('/out', 'P001', 'cell_properties', freshNuclei) ==
        '/out/P001/cell_properties/nuclei/contours.json'

    // Layout.passthroughPath -- see checkPassthroughPath() above the workflow block.
    checkPassthroughPath()
    checkSegQcNucleiDir()

    // ------------------------------------------------------------------ //
    // ParamUtils.compartmentMode / validateCompartmentQuant: --quantify_compartments is
    // resolved once and threaded down as an argument.
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

    // 4. validateCompartmentQuant: expanded requires compartments.
    ParamUtils.validateCompartmentQuant([compartments: true, expanded: true, embedMasks: false])   // ok
    ParamUtils.validateCompartmentQuant([compartments: true, expanded: false, embedMasks: false])  // ok
    try {
        ParamUtils.validateCompartmentQuant([compartments: false, expanded: true, embedMasks: false])
        assert false : "expanded=true, compartments=false must be rejected"
    } catch (IllegalArgumentException ignored) {
        // expected
    }

    // 5. embedMasks requires both compartments and expanded; each violation is rejected.
    ParamUtils.validateCompartmentQuant([compartments: true, expanded: true, embedMasks: true])    // ok
    try {
        ParamUtils.validateCompartmentQuant([compartments: false, expanded: false, embedMasks: true])
        assert false : "embedMasks=true with compartments=false, expanded=false must be rejected"
    } catch (IllegalArgumentException ignored) {
        // expected: otherwise a pyramid with no mask series is published silently
    }
    try {
        ParamUtils.validateCompartmentQuant([compartments: true, expanded: false, embedMasks: true])
        assert false : "embedMasks=true with expanded=false must be rejected even when compartments=true"
    } catch (IllegalArgumentException ignored) {
        // expected
    }

    // validateStareKnobs: COARSE's 256 px floor. 0 or negative is dangerous, not just
    // invalid: decimation reads <=0 as "none" (a full-res plane into a U-Net) and the
    // memory closure squares the value.
    def sb = [reg_tiled_tile: 2048, reg_tiled_halo: 256,
              reg_tiled_out_tile: 1024]
    [null, 512, 256].each { ParamUtils.validateStareKnobs(sb + [reg_tiled_coarse_max_dim: it]) }
    [0, -1, 255, 16].each { bad ->
        def no = false
        try { ParamUtils.validateStareKnobs(sb + [reg_tiled_coarse_max_dim: bad]) }
        catch (IllegalArgumentException ignored) { no = true }
        assert no : "validateStareKnobs must reject reg_tiled_coarse_max_dim=${bad}"
    }
    checkRetiredStareTier()

    // ------------------------------------------------------------------ //
    // Layout - the published-kind vocabulary
    // ------------------------------------------------------------------ //
    assert Layout.requireKind('segmentation') == 'segmentation'
    assert Layout.PUBLISHED_KINDS.contains(Layout.REGISTERED)
    assert Layout.PUBLISHED_KINDS.contains('split_channels')

    def badKind = false
    try { Layout.requireKind('segmentaton') }   // typo, deliberately
    catch (IllegalArgumentException ignored) { badKind = true }
    assert badKind : 'Layout.requireKind must reject an unknown kind'

    // patientDir must reject it too - that is the call site the typo reaches from.
    def badPatientKind = false
    try { Layout.patientDir('/out', 'P001', 'segmentaton') }
    catch (IllegalArgumentException ignored) { badPatientKind = true }
    assert badPatientKind : 'Layout.patientDir must reject an unknown kind'

    // ------------------------------------------------------------------ //
    // ProcessEnvelope - the versions.yml envelope
    // ------------------------------------------------------------------ //
    def envVersions     = ProcessEnvelope.versions('TEST:PROC', ['numpy', 'skimage'])
    def envVersionsStub = ProcessEnvelope.versionsStub('TEST:PROC', ['numpy', 'skimage'])

    // The python: row is prepended in both renderings. A bare `$(`, not `\$(`:
    // `<<-END_VERSIONS` is an unquoted heredoc, so bash substitutes the former and prints
    // the latter literally.
    assert envVersions.contains('python: $(python --version')
    assert !envVersions.contains('python: \\$(python --version')
    assert envVersionsStub.contains('python: stub')

    // skimage -> scikit-image: the import name and the published YAML key differ, and
    // bin/generate_qc_report.py's hand-rolled parser is keyed on the YAML key, not the
    // import name.
    assert envVersions.contains('scikit-image: $(python -c "import skimage;')
    assert !envVersions.contains('skimage: $(python -c "import skimage;')
    assert envVersionsStub.contains('scikit-image: stub')

    // versions() and versionsStub() must name the same set of tools. Compare YAML keys:
    // -stub never evaluates the script: block that versions() renders.
    def keysOf = { String heredoc ->
        heredoc.readLines()
            .findAll { it.startsWith('    ') }
            .collect { it.trim().split(':')[0].trim() }
            .toSet()
    }
    assert keysOf.call(envVersions) == keysOf.call(envVersionsStub)
    assert keysOf.call(envVersions) == ['python', 'numpy', 'scikit-image'].toSet()

    // ------------------------------------------------------------------ //
    // SegBackends - the segmentation seam's versions.yml tool lists
    // ------------------------------------------------------------------ //
    // Declared as bare module names, not rendered YAML, so segment.nf hands the same list
    // to ProcessEnvelope.versions() (script:) and versionsStub() (stub:).
    assert SegBackends.methods().toSorted() == ['cellsam', 'instantseg', 'stardist']
    assert SegBackends.of('stardist').versionTools   == ['tensorflow']
    assert SegBackends.of('instantseg').versionTools == ['instanseg', 'torch']
    assert SegBackends.of('cellsam').versionTools    == ['cellSAM', 'torch']

    // `torch` is repeated per backend, not shared: one backend runs per task and stardist
    // loads no torch.
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
    // WarpBackends - one seam for the reg_qc=2 warp
    // ------------------------------------------------------------------ //
    assert WarpBackends.methods().toSorted() == ['tiled', 'valis']
    // REGISTER's image: containers/valis, FROM the digest-pinned upstream valis-wsi 1.2.0.
    assert WarpBackends.container('valis') == 'bolt3x/mirage-valis:1.0.0'
    assert WarpBackends.container('tiled') == 'bolt3x/mirage-stare:1.0.0'
    assert WarpBackends.of('valis').stages == ['native', 'rigid', 'non_rigid', 'micro']
    assert WarpBackends.of('tiled').stages == ['native', 'rigid', 'refined']

    // The tiled backend must pass --method tiled; VALIS must not.
    assert WarpBackends.of('tiled').flags([:]).any { it.contains('--method tiled') }
    // No JVM heap flag may leak onto a JVM-free backend (see the WarpBackends header).
    assert !WarpBackends.of('tiled').flags([:]).any { it.contains('--jvm-heap-gb') }
    def valisFlags = WarpBackends.of('valis').flags(
        [ref_slide: 'R', moving_slide: 'M', stage_checkpoint: null, micro_reg: 0])
    assert !valisFlags.any { it.contains('--method') }
    assert valisFlags.any { it.contains('--micro-reg 0') }
    // Absent checkpoint is a null object, not an empty flag.
    assert !valisFlags.any { it.contains('--checkpoint-dir') }
    assert WarpBackends.of('valis').flags(
        [ref_slide: 'R', moving_slide: 'M', stage_checkpoint: 'ckpt/', micro_reg: 0]
    ).any { it.contains('--checkpoint-dir ckpt/') }

    def badMethod = false
    try { WarpBackends.of('stare') }
    catch (IllegalArgumentException ignored) { badMethod = true }
    assert badMethod : 'WarpBackends.of must reject an unknown method'

    // ------------------------------------------------------------------ //
    // Meta - the one constructor for every meta map
    // ------------------------------------------------------------------ //
    assert Meta.REQUIRED_KEYS.containsAll([
        'patient_id', 'id', 'is_reference', 'channels',
        'keep_channels', 'channels_count', 'images_count',
    ])

    // identityFor matches Nextflow's file(...).simpleName, which strips every extension
    // ('slide.ome.tiff' -> 'slide'), for a non-colliding row.
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

    // fromSamplesheetRow: full construction. channels_count comes from ctx.channelsCount,
    // the per-patient total, not keep_channels.size(), which counts one slide only.
    // keepChannelsBySlide is keyed on meta.id: 'P1_img1' is
    // identityFor('P1', 'img1.ome.tiff', 0, [:]).
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

    // keep_channels absent vs empty: no entry falls back to the declared channels; an
    // explicit [] (every marker already claimed) stays [] (`?:` cannot tell them apart).
    // Both ctx maps carry channelsCount and imagesCount, which have no fallback.
    // 'P1_img2' is identityFor('P1', 'img2.ome.tiff', 0, [:]).
    def emptyKeepCtx = [keepChannelsBySlide: [P1: ['P1_img2': []]], channelsCount: [P1: 0], imagesCount: [P1: 1]]
    def emptyKeepRow = [patient_id: 'P1', image: 'img2.ome.tiff', channels: 'DAPI|CD3', is_reference: 'false']
    def emptyKeepMeta = Meta.fromSamplesheetRow(emptyKeepRow, 'image', 0, emptyKeepCtx)
    assert emptyKeepMeta.keep_channels == []
    assert emptyKeepMeta.channels_count == 0
    def absentKeepCtx = [keepChannelsBySlide: [P1: [:]], channelsCount: [P1: 2], imagesCount: [P1: 1]]
    def absentKeepMeta = Meta.fromSamplesheetRow(emptyKeepRow, 'image', 0, absentKeepCtx)
    assert absentKeepMeta.keep_channels == ['DAPI', 'CD3']

    // ---- channels_count has no fallback ----------------------------------------
    // A per-slide fallback under-counts a multi-slide patient, and a low count makes
    // groupKey(patient_id, channels_count) emit early or never. Every caller must supply
    // a per-patient total.
    def noCountRow = [patient_id: 'P1', image: 'img3.ome.tiff', channels: 'DAPI|CD3', is_reference: 'false']

    // Trigger: ctx carries no channelsCount key at all.
    def missingChannelsCountKey = false
    try { Meta.fromSamplesheetRow(noCountRow, 'image', 0, [:]) }
    catch (IllegalArgumentException ignored) { missingChannelsCountKey = true }
    assert missingChannelsCountKey : 'Meta.fromSamplesheetRow must reject ctx with no channelsCount map at all'

    // Trigger: ctx.channelsCount is populated, but not for this patient.
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

    // ---- images_count has no fallback either -----------------------------------
    // It feeds groupKey(meta.patient_id, meta.images_count) in registration.nf, and a bare
    // `?:` would coerce a genuine 0 to the default. Every caller computes a full
    // per-patient imagesCount map (CsvUtils.countImagesPerPatient).

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

    // ctx.imagesCount = [P1: 0] must yield images_count == 0, not 1.
    def zeroImagesMeta = Meta.fromSamplesheetRow(noCountRow, 'image', 0, [channelsCount: [P1: 2], imagesCount: [P1: 0]])
    assert zeroImagesMeta.images_count == 0 : 'a genuine images_count of 0 must NOT be coerced to 1'

    // ---- fromCheckpointRow validates against the schema ------------------------
    // lib/Checkpoint.groovy owns the column list per step. A checkpoint written before
    // identity tracking has no 'id' column, so its rows carry no 'id' key at all
    // (distinct from a present but blank value, covered further down).

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

    // Fully satisfied: fromCheckpointRow returns a complete meta.
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
    // CsvUtils.metaContextFromCheckpoint: the checkpoint-side twin of the ctx
    // INPUT_CHECK assembles from a samplesheet.
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

    // The checks below are defined above workflow{} (size ceiling).
    checkResourceReport()

    // ProcessEnvelope - size-log row + container identity in versions.yml.
    checkProcessEnvelope()

    // RegisteredMatch - pairing VALIS's outputs back to their slide metas.
    checkRegisteredMatch()

    // Checkpoint.requireColumns
    checkRegQcLevel()
    checkCheckpoint()

    // RegBackends - the registration backend table.
    checkRegBackends()

    // CsvUtils.unknownColumns - report, never reject.
    checkCsvUtilsUnknownColumns()

    checkParamValidators()

    // CsvUtils.validateInputSemantics(..., requireUniqueChannelSets).
    checkUniqueChannelSets()

    // println, not log.info: nf-test runs nextflow with -quiet, which drops log.info
    // from stdout, and lib_probe.nf.test asserts on workflow.stdout.
    println "LIB PROBE: all assertions passed"
}

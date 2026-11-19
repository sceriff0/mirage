/*
========================================================================================
    ProcessEnvelope — the versions.yml boilerplate, rendered once
========================================================================================
    Renders the versions.yml heredoc and the *.size.csv row for both script: and
    stub: from one call each, so the two blocks cannot name different tools (-stub
    never evaluates script:, so CI could not see them diverge).

    Output is joined with '\n' at column zero; the caller indents it, because
    Nextflow stripIndent()s the finished script.

    Static; reads no params.
========================================================================================
*/

class ProcessEnvelope {

    /*
     * Python module name -> the YAML key the QC report expects. Only modules whose import
     * name differs from their reported name need an entry; everything else reports under
     * its own name.
     *
     * These are the keys of every published versions.yml (and collated_versions.yml);
     * changing one changes published data.
     */
    private static final Map<String, String> YAML_KEY = [
        'skimage'   : 'scikit-image',
        'sklearn'   : 'scikit-learn',
        'PIL'       : 'pillow',
        'yaml'      : 'pyyaml',
    ].asImmutable()

    private static String yamlKey(String tool) {
        // Not get(tool, tool): Groovy's get(key, default) puts into the map, which throws
        // on an immutable map.
        return YAML_KEY.getOrDefault(tool, tool)
    }

    /** The version-probe line for one tool, as it appears inside the heredoc. */
    private static String probe(String tool) {
        // One level of escaping: `\$` yields a bare `$`, which bash must see to run the
        // substitution (the heredoc is unquoted). The caller splices this string verbatim,
        // so `\\\$` would reach .command.sh as `\$(...)` and print the command text.
        return "    ${yamlKey(tool)}: \$(python -c \"import ${tool}; print(${tool}.__version__)\" 2>/dev/null || echo \"unknown\")"
    }

    /**
     * The full versions.yml heredoc for a `script:` block.
     *
     * `python:` is prepended automatically. Pass tools in report order.
     */
    static String versions(String process, List<String> tools, String container = null) {
        def lines = ['cat <<-END_VERSIONS > versions.yml',
                     "\"${process}\":",
                     // single-quoted: the bare `$(...)` must reach bash unescaped
                     '    python: $(python --version 2>&1 | sed \'s/Python //\')']
        // task.container is set even with no container engine; the null check avoids
        // writing 'null' from the 2-arg form.
        if (container) { lines << "    container: ${container}" }
        lines += tools.collect { probe(it) }
        lines << 'END_VERSIONS'
        return lines.join('\n')
    }

    /**
     * The same heredoc for a `stub:` block, every value the literal `stub`.
     *
     * Takes the same tools list as versions(), so the two blocks agree.
     */
    static String versionsStub(String process, List<String> tools, String container = null) {
        def lines = ['cat <<-END_VERSIONS > versions.yml',
                     "\"${process}\":",
                     '    python: stub']
        // `stub`, not the image name: a stub run used no image.
        if (container) { lines << '    container: stub' }
        lines += tools.collect { "    ${yamlKey(it)}: stub" }
        lines << 'END_VERSIONS'
        return lines.join('\n')
    }

    /**
     * The heredoc for a module with NO Python interpreter.
     *
     * Used by AGGREGATE_SIZE_LOGS: versions() always prepends `python:`, which would
     * record the shell's error message where Python is missing.
     */
    static String versionsBash(String process, String container = null) {
        def lines = ['cat <<-END_VERSIONS > versions.yml',
                     "\"${process}\":",
                     '    bash: $(bash --version | head -n1 | sed \'s/GNU bash, version //\')']
        if (container) { lines << "    container: ${container}" }
        lines << 'END_VERSIONS'
        return lines.join('\n')
    }

    /** The same heredoc for a `stub:` block, every value the literal `stub`. */
    static String versionsBashStub(String process, String container = null) {
        def lines = ['cat <<-END_VERSIONS > versions.yml',
                     "\"${process}\":",
                     '    bash: stub']
        if (container) { lines << '    container: stub' }
        lines << 'END_VERSIONS'
        return lines.join('\n')
    }

    /**
     * Column order of every *.size.csv row and of AGGREGATE_SIZE_LOGS' header.
     *
     * Rows and the header are both built from this list. bin/generate_resource_report.py
     * declares the same tuple (tests/test_size_log_schema_has_one_owner.py).
     */
    static final List<String> SIZE_LOG_COLUMNS = ['process', 'sample_id', 'filename', 'bytes'].asImmutable()

    /**
     * The `filename` cell for a row: the basename of the sole path, or the literal
     * `inputs/` when the measurement covers more than one file.
     *
     * A glob counts as more than one file. A staged prefix is dropped
     * ('mov/x.tif' -> 'x.tif').
     */
    private static String filenameCell(List<String> shellPaths) {
        if (shellPaths.size() != 1) { return 'inputs/' }
        def p = shellPaths[0]
        if (p.contains('*') || p.contains('?')) { return 'inputs/' }
        def slash = p.lastIndexOf('/')
        return slash < 0 ? p : p.substring(slash + 1)
    }

    private static String row(Map cells, String outFile) {
        return 'echo "' + SIZE_LOG_COLUMNS.collect { cells[it] }.join(',') + '" > ' + outFile
    }

    /**
     * The size-log lines for a `script:` block: one `stat` of every path in
     * `shellPaths`, summed, then ONE row into `outFile`.
     *
     * `shellPaths` entries reach bash as-is. Pass staged inputs double-quoted
     * ("${image_file}") so the call site interpolates them; a single-quoted `${x}`
     * becomes an unset shell variable and a silent zero. Globs ('channels/*') are fine.
     * A missing file counts as 0 bytes.
     */
    static String sizeLog(String process, String sampleId, List<String> shellPaths, String outFile) {
        if (!shellPaths) {
            throw new IllegalArgumentException(
                "ProcessEnvelope.sizeLog: shellPaths must not be empty (${process})")
        }
        // Single-quoted: bare `$` reaches bash; `\\n` is backslash + n for stat --printf.
        def stat = 'size_bytes=$(stat -L --printf="%s\\n" ' + shellPaths.join(' ') +
                   ' 2>/dev/null | awk \'{s+=$1} END {print s+0}\')'
        def cells = [process: process, sample_id: sampleId,
                     filename: filenameCell(shellPaths), bytes: '${size_bytes}']
        return stat + '\n' + row(cells, outFile)
    }

    /**
     * The size-log line for a `stub:` block: `process,sampleId,stub,0`.
     *
     * Uses the real process name, so stub rows stay comparable with real ones.
     */
    static String sizeLogStub(String process, String sampleId, String outFile) {
        def cells = [process: process, sample_id: sampleId, filename: 'stub', bytes: '0']
        return row(cells, outFile)
    }
}

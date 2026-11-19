/*
========================================================================================
    ResourceReport — where the trace is, and what to say when it is not there
========================================================================================
    Used by main.nf's onComplete to build the resource report. Nextflow resolves
    trace.file against launchDir, not the JVM's working directory, so a relative
    trace_dir must be resolved the same way. The generator script exits 0 on a missing
    trace, so main.nf must say so itself.

    Kept in lib/ so tests/lib_probe.nf can test it. Static, no params, no filesystem,
    no logging (main.nf owns the log calls).
========================================================================================
*/

class ResourceReport {

    /** The file Nextflow's trace scope writes inside `trace_dir`. */
    static final String TRACE_FILENAME = 'trace.txt'

    /**
     * Absolute path of the run's trace file.
     *
     * @param traceDir   params.trace_dir, raw. Absolute passes through; relative is
     *                   resolved against launchDir, matching how Nextflow itself
     *                   resolves trace.file.
     * @param launchDir  workflow.launchDir as a String. Required only when traceDir
     *                   is relative.
     * @return           "<resolved trace dir>/trace.txt", normalised.
     * @throws IllegalArgumentException on a blank traceDir, or on a blank launchDir
     *                   with a relative traceDir.
     */
    static String tracePath(String traceDir, String launchDir) {
        if (!traceDir?.trim())
            throw new IllegalArgumentException(
                'ResourceReport.tracePath: trace_dir is required')
        def dir = java.nio.file.Paths.get(traceDir.trim())
        if (!dir.isAbsolute()) {
            if (!launchDir?.trim())
                throw new IllegalArgumentException(
                    "ResourceReport.tracePath: launchDir is required to resolve the " +
                    "relative trace_dir '${traceDir}'")
            dir = java.nio.file.Paths.get(launchDir.trim()).resolve(dir)
        }
        return dir.resolve(TRACE_FILENAME).normalize().toString()
    }

    /**
     * The one line main.nf logs when no resource report can be produced.
     *
     * Names the path looked for and whether the trace was enabled (a config fix vs a
     * choice). Never suggests a CLI boolean flag: Nextflow 26 passes CLI params as Strings.
     *
     * @param tracePath    the absolute path from tracePath(), for the operator to check
     * @param enableTrace  params.enable_trace, as captured at launch
     */
    static String missingTraceMessage(String tracePath, boolean enableTrace) {
        def why = enableTrace
            ? "enable_trace was ON for this run, so Nextflow should have written it -- " +
              "check that trace_dir is writable and that this run reached completion"
            : "enable_trace was OFF for this run, so Nextflow wrote no trace -- turn it " +
              "on with a -params-file or a profile to collect one"
        return "No resource report was generated: no trace at ${tracePath}. ${why}."
    }
}

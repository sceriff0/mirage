#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
================================================================================
    MIRAGE WSI Processing Pipeline
================================================================================
    Preprocessing, Registration, Segmentation, Quantification, GeoJSON export
    https://github.com/sceriff0/mirage
================================================================================
*/

include { MIRAGE } from './workflows/mirage'

/*
================================================================================
    POST-RUN: COMPUTATIONAL RESOURCE REPORT (best-effort)
================================================================================
    trace.txt is only finalized at workflow completion, so the resource report
    is generated here rather than as an in-DAG process. Any failure (no python3
    on the head node, missing trace) logs a warning and never fails the run;
    the script is also runnable by hand against an existing outdir + trace dir.

    Why this is a function called from a handler rather than the handler itself:

      * Nextflow 26's strict parser rejects a top-level `workflow.onComplete { }`
        with "Statements cannot be mixed with script declarations". The handler
        has to be registered from inside the entry `workflow { }` block.
      * A closure registered from there does NOT see the script binding. Inside
        it, `params`, `workflow` and `projectDir` all evaluate to null — under
        the v1 parser as well as v2, so this is a Nextflow scoping fact and not
        a migration artifact. Everything the report needs is therefore captured
        into a plain Map in the workflow body (where those names resolve) and
        handed in as `cfg`.
      * `log` resolves normally inside a script-level function, which is why the
        warn/info calls live here rather than in the handler closure.

    `cfg` deliberately carries only raw params and plain Strings (outdir, trace_dir,
    launch_dir, project_dir): every Layout call stays inside the try below, so a
    Layout failure is still caught and downgraded to a warning exactly as before,
    rather than being hoisted to DAG-construction time where it would fail the run.

    launch_dir is captured for the same reason the others are, and used because
    Nextflow resolves `trace.file` against launchDir while cmd.execute() resolves a
    relative path against the JVM's working directory. lib/ResourceReport.groovy owns
    that arithmetic so tests/lib_probe.nf can assert it -- a script-level function in
    this file has no unit-test surface of its own.
*/
def generateResourceReport(Map cfg) {
    // The path Nextflow's trace scope actually wrote to. lib/ResourceReport.groovy
    // owns the resolution rule and the diagnostic text; see its header for why a
    // relative trace_dir resolved against the JVM's working directory was wrong, and
    // why the previous silence was worse than the wrong path.
    def trace_txt
    try {
        trace_txt = ResourceReport.tracePath(cfg.trace_dir, cfg.launch_dir)
    } catch (Exception e) {
        log.warn "Could not resolve the trace path (non-fatal): ${e.message}"
        return
    }

    if (!cfg.enable_trace) {
        log.warn ResourceReport.missingTraceMessage(trace_txt, false)
        return
    }
    if (!new File(trace_txt).exists()) {
        // Deliberately do NOT invoke the script here. bin/generate_resource_report.py
        // renders "Trace data not available" and exits 0 for a missing trace, so
        // running it would produce a page that LOOKS like a report and a log line
        // that says the report was written. An absent file plus this warning is the
        // honest outcome.
        log.warn ResourceReport.missingTraceMessage(trace_txt, true)
        return
    }

    try {
        def script    = "${cfg.project_dir}/bin/generate_resource_report.py"
        def size_log  = "${Layout.runDir(cfg.outdir, 'size_logs')}/input_sizes.csv"
        def qc_dir    = Layout.runDir(cfg.outdir, 'qc')
        def out_html  = "${qc_dir}/mirage_resource_report.html"
        new File(qc_dir).mkdirs()
        // Derived from trace_txt's own resolved parent, not the raw cfg.trace_dir:
        // trace_txt is already the RESOLVED absolute path (ResourceReport.tracePath
        // above), so its parent is unambiguous regardless of whether trace_dir was
        // passed relative or absolute. Building these two from the raw param instead
        // silently pointed at the wrong directory under the exact relative-trace_dir
        // launch this class's header describes.
        def trace_dir_resolved = new File(trace_txt).parent
        def cmd = [
            'python3', script,
            '--trace', trace_txt,
            '--size-log', size_log,
            '--output', out_html,
            '--native-report', "${trace_dir_resolved}/report.html",
            '--native-timeline', "${trace_dir_resolved}/timeline.html",
        ]
        def proc = cmd.execute()
        proc.waitForProcessOutput(System.out, System.err)
        if (proc.exitValue() == 0) {
            log.info "Resource report: ${out_html}"
        } else {
            log.warn "Resource report generation exited ${proc.exitValue()} (non-fatal)."
        }
    } catch (Exception e) {
        log.warn "Could not generate resource report (non-fatal): ${e.message}"
    }
}

/*
================================================================================
    POST-RUN: EXPLAIN AN EMPTY CHECKPOINT DIRECTORY
================================================================================
    At a cleaning --cleanup_level no checkpoint manifest is written, because the
    artifacts every manifest names were not published (Checkpoint.writesAtLevel).
    An empty csv/ is then indistinguishable from a run that crashed before its
    first checkpoint, so the directory says which it is.

    Written from a handler rather than a process for the same reason the resource
    report is: it is provenance about the run as a whole, it must not appear in
    the DAG, and it must not change the task graph or the stub manifest's task
    count. `cfg` carries raw params for the reason generateResourceReport's
    comment gives -- the handler closure cannot see `params`.
*/
def writeCheckpointReadme(Map cfg) {
    if (cfg.cleanup_level == 'none') {
        return
    }
    try {
        def dir = Layout.checkpointDir(cfg.outdir)
        new File(dir).mkdirs()
        new File(dir, 'README.txt').text = """\
No checkpoint manifest is written at --cleanup_level=${cfg.cleanup_level}.

A manifest names WHERE a step's artifacts landed so a later run can re-enter
there. At this level those artifacts are not published at all, so every manifest
would record paths that do not exist -- including postprocessed.csv, whose
cell_mask column names a segmentation mask that was not published even though
its other columns are final artifacts.

Re-run with --cleanup_level none if you need to re-enter this output with
--start <step>, or use it as the --prior_outdir of a --mode add_cycle run.
add_cycle is refused at any other level, at launch.
"""
        log.info "Checkpoint directory note: ${dir}/README.txt"
    } catch (Exception e) {
        log.warn "Could not write the checkpoint README (non-fatal): ${e.message}"
    }
}

/*
================================================================================
    POST-RUN: ANNOUNCE DROPPED TASKS
================================================================================
    A `retry-then-drop` errorStrategy lets the run report success with an
    artifact missing. That is a deliberate policy (conf/modules.config names it,
    tests/test_error_strategy_policy.py pins it), but it must not be SILENT —
    "it OOMed, the strategy swallowed it, and the run went green" is the exact
    history that made the CSE scorer look permanently broken.

    The announcement lives here rather than in the errorStrategy closure that
    knows the most about it, for the reason the closure's own comment gives:
    `log` is unbound in conf/*.config under the v1 config parser, so logging
    from a closure there resolves against ConfigObject and ABORTS the run past
    the retry budget — the opposite of 'ignore'. Reproduced 2026-08-25 on
    NXF_VER=25.04.7, which manifest.nextflowVersion still accepts.

    Same script-level-function shape as generateResourceReport above, and for
    the same reason: `log` and `workflow` resolve in a script-level function but
    are null inside the handler closure. `workflow.stats` is final by the time
    onComplete fires, so reading the count here is safe.
*/
def announceDroppedTasks() {
    try {
        def dropped = workflow.stats.ignoredCount
        if (dropped) {
            log.warn "${dropped} task(s) failed and were DROPPED by a retry-then-drop " +
                     "policy. This run is reported as SUCCESSFUL and its output tree is " +
                     "INCOMPLETE — one optional artifact is missing per dropped task. " +
                     "Grep the log above for [ERROR] to see which. Only SEG_QUALITY_EVAL " +
                     "and MERGE_SEG_EVAL carry that policy; if a CSE QualityScore is the " +
                     "artifact that went missing, lower --cse_max_pixels and re-run."
        }
    } catch (Exception e) {
        log.warn "Could not report the dropped-task count (non-fatal): ${e.message}"
    }
}

/*
================================================================================
    RUN MAIN WORKFLOW
================================================================================
*/


/*
================================================================================
    POST-RUN: PRUNE THE EMPTY PUBLISH DIRECTORIES A CLEANING LEVEL LEAVES
================================================================================
    publishDir creates its target directory when the process starts, before the
    `saveAs:` gate has decided whether any file lands there. At a cleaning
    --cleanup_level every intermediate's gate returns null for every file, so a
    successful run leaves an empty converted/, preprocessed/, split_channels/,
    segmentation/, quantify/, cell_properties/ and registered/summary/ per patient
    (measured 2026-09-09: seven on the stub dataset). They are noise that reads as
    "something was supposed to be here", so they go. Only EMPTY directories, and
    only outside any .zarr store; never a file. tests/cleanup_work.sh asserts none
    remain.
*/
def pruneEmptyPublishDirs(Map cfg) {
    if (cfg.cleanup_level == 'none') {
        return
    }
    try {
        def root = new File(cfg.outdir.toString())
        if (!root.isDirectory()) {
            return
        }
        def dirs = []
        root.eachDirRecurse { File d -> dirs << d }
        // Deepest first, so a directory left empty by its children's removal goes too.
        dirs.sort { File d -> -d.toPath().getNameCount() }
        def removed = 0
        dirs.each { File d ->
            if (d.path.contains('.zarr')) {
                return
            }
            def entries = d.list()
            if (entries != null && entries.length == 0 && d.delete()) {
                removed += 1
            }
        }
        if (removed > 0) {
            log.info "Removed ${removed} empty publish director${removed == 1 ? 'y' : 'ies'} under ${root}: " +
                     "intermediates are not published at --cleanup_level=${cfg.cleanup_level}."
        }
    } catch (Exception e) {
        log.warn "Could not prune empty publish directories (non-fatal): ${e.message}"
    }
}

workflow {
    main:

    // Captured before MIRAGE() so the handler is registered even if the
    // workflow throws during DAG construction — that is when the previous
    // top-level registration took effect, and onComplete fires on failed runs
    // too.
    def report_cfg = [
        enable_trace : params.enable_trace,
        outdir       : params.outdir,
        trace_dir    : params.trace_dir,
        launch_dir   : "${workflow.launchDir}",
        project_dir  : "${projectDir}",
        cleanup_level: params.cleanup_level,
    ]
    workflow.onComplete {
        announceDroppedTasks()
        writeCheckpointReadme(report_cfg)
        pruneEmptyPublishDirs(report_cfg)
        generateResourceReport(report_cfg)
    }

    MIRAGE()
}

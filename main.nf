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
    trace.txt is only final at workflow completion, so the report is built here, not
    in the DAG. Any failure logs a warning and never fails the run.

    Called from onComplete, whose closure cannot see `params`, `workflow` or
    `projectDir`; everything is captured into `cfg` in the workflow body. `log`
    resolves in a script-level function. Layout calls stay inside the try so a
    failure is downgraded to a warning. lib/ResourceReport.groovy resolves the trace
    path against launchDir, as Nextflow's trace scope does.
*/
def generateResourceReport(Map cfg) {
    // The path Nextflow's trace scope actually wrote to.
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
        // Do not run the script: on a missing trace it writes a placeholder page that
        // looks like a report.
        log.warn ResourceReport.missingTraceMessage(trace_txt, true)
        return
    }

    try {
        def script    = "${cfg.project_dir}/bin/generate_resource_report.py"
        def size_log  = "${Layout.runDir(cfg.outdir, 'size_logs')}/input_sizes.csv"
        def qc_dir    = Layout.runDir(cfg.outdir, 'qc')
        def out_html  = "${qc_dir}/mirage_resource_report.html"
        new File(qc_dir).mkdirs()
        // From the resolved trace path, not the raw (possibly relative) trace_dir.
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
    At a cleaning --cleanup_level no checkpoint manifest is written
    (Checkpoint.writesAtLevel), so csv/ gets a README saying why it is empty rather
    than looking like a crashed run. A handler, not a process, so the DAG is unchanged.
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
    A `retry-then-drop` errorStrategy lets a run succeed with an artifact missing;
    this makes that visible. It lives here because `log` is unbound in a
    conf/*.config closure on Nextflow 25 and would abort the run.
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
    publishDir creates its directory before `saveAs:` decides, so a cleaning
    --cleanup_level leaves empty intermediate directories per patient. Remove only
    empty directories, never inside a .zarr store. tests/cleanup_work.sh checks it.
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

    // Registered before MIRAGE() so onComplete also fires if DAG construction throws.
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

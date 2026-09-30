/*
 * WARP_SEG_QC - staged, fixed-correspondence segmentation-overlap QC (reg_qc = 2)
 *
 * Loads the registration transform (a VALIS registrar pickle, or a STARE transform
 * manifest for the tiled method) and warps the reference and moving native-cell
 * GeoJSONs through each registration stage in turn. Cell-to-cell correspondence is
 * established ONCE at the rigid stage (optimal one-to-one assignment minimising total
 * centroid distance within 1.5 median nuclear radii, solved exactly per connected
 * component — see bin/utils/cell_pairs.py's match_lsa; the older mutual-nearest-centroid
 * rule remains selectable via params.seg_qc_pairing) and then held fixed, so each
 * stage's per-pair IoU and centroid residual describe the same cells and the deltas are
 * pure registration effects.
 *
 * The pre-micro stage checkpoint from REGISTER is optional but load-bearing on the
 * VALIS path: VALIS composes the micro residual into the same displacement field, so
 * without the checkpoint the 'non_rigid' stage cannot be separated from 'micro' and is
 * not reported (the output records `stages_separable: false` and says why). The tiled
 * method needs no checkpoint at all — its stages are separable by construction — so it
 * always receives `[]` (a null object, not a different input shape).
 *
 * Scoring is per-pair on a local window, so peak memory is one nucleus rather than one
 * slide — this process no longer allocates whole-slide label images.
 *
 * Everything that differs between the two registration backends — container, the
 * method-specific CLI flags, the stage list, the versions.yml rows, and the stub-JSON
 * extras — comes from lib/WarpBackends.groovy, keyed on the `method` input. This is
 * the same shape modules/local/segment.nf already uses for its three segmentation
 * backends via lib/SegBackends.groovy.
 */
process WARP_SEG_QC {
    tag "${meta.patient_id}:${moving_geojson.simpleName}"

    container { WarpBackends.container(method) }

    input:
    tuple val(meta), val(method), path(transform), val(ref_slide), val(moving_slide), path(ref_geojson), path(moving_geojson), path(stage_checkpoint)

    output:
    tuple val(meta), path("*_seg_qc.json"), emit: metrics
    // Final-stage per-cell registration residuals, spatially joinable onto cell_mask.
    tuple val(meta), path("*_reg_residuals.csv"), optional: true, emit: per_cell
    path "versions.yml"                   , emit: versions
    path("*.size.csv")                    , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def backend = WarpBackends.of(method)
    def args    = task.ext.args ?: ''
    def prefix  = "${meta.patient_id}_${moving_geojson.simpleName}"
    // Tell the QC the run's micro depth so it can state honestly whether 'rigid' includes
    // micro-rigid (level >= 1). Single-source via ParamUtils, same as REGISTER. Unused by
    // the tiled backend's flags closure, but harmless to compute either way.
    def ctx = [meta: meta, ref_slide: ref_slide, moving_slide: moving_slide,
               stage_checkpoint: stage_checkpoint, micro_reg: ParamUtils.microRegLevelOf(params.reg_micro_reg)]
    // `?: ['']` guards against a future backend whose `flags` closure returns `[]`: with
    // no fallback, `backend_flags` would be the empty string and the interpolation below
    // (column 0, right after a line ending in `\`) would land a BLANK line between two
    // backslash-continued lines, ending the shell command early and letting `${args}`
    // execute as its own separate command. A single empty-string flag renders as a
    // harmless whitespace-only continuation line instead. Not reachable today — both
    // backends emit at least one flag.
    def backend_flags = (backend.flags(ctx) ?: ['']).collect { "        ${it} \\" }.join('\n')
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${moving_geojson}"], "${prefix}.WARP_SEG_QC.size.csv")}

    warp_seg_qc.py \\
        --pickle ${transform} \\
        --ref-slide '${ref_slide}' \\
        --moving-slide '${moving_slide}' \\
        --ref-geojson ${ref_geojson} \\
        --moving-geojson ${moving_geojson} \\
        --output ${prefix}_seg_qc.json \\
        --per-cell-csv ${prefix}_reg_residuals.csv \\
        --patient-id ${meta.patient_id} \\
${backend_flags}
        ${args}

    ${ProcessEnvelope.versions(task.process, backend.versionTools, task.container)}
    """

    stub:
    def backend = WarpBackends.of(method)
    def prefix  = "${meta.patient_id}_${moving_geojson.simpleName}"
    def ctx     = [micro_reg: ParamUtils.microRegLevelOf(params.reg_micro_reg)]
    def stages     = backend.stages
    def stage_stub = [n_pairs: 0, n_pairs_scored: 0, iou_n: 0, displacement_px_n: 0]
    // Built as base + extras + rest (rather than one flat literal) so each backend's
    // stubExtras land at their ORIGINAL position in the JSON rather than being appended
    // at the end: Groovy's Map#plus() inserts a right-hand key at the end only if it is
    // NEW, so the valis backend's micro_reg/rigid_includes_micro_rigid keys land right
    // after stages_separable (where the pre-merge VALIS stub always put them) and the
    // tiled backend's empty stubExtras changes nothing.
    def base = [
        patient_id      : meta.patient_id,
        moving          : moving_slide,
        reference       : ref_slide,
        stages_separable: true,
    ]
    def rest = [
        stage_order     : stages,
        stages          : stages.collectEntries { [(it): stage_stub] },
        delta_vs_anchor : (stages - 'rigid').collectEntries { [(it): [:]] },
        matching        : [method: 'lsa_centroid', anchor_stage: 'rigid', n_pairs: 0],
        counts          : [features_ref: 0, features_moving: 0],
    ]
    def stub_json = groovy.json.JsonOutput.toJson(base + backend.stubExtras(ctx) + rest)
    """
    echo '${stub_json}' > ${prefix}_seg_qc.json
    printf 'moving,ref_x,ref_y,residual_px,iou,stage\\n' > ${prefix}_reg_residuals.csv
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${prefix}.WARP_SEG_QC.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, backend.versionTools, task.container)}
    """
}

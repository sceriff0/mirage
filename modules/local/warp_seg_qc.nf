/*
 * WARP_SEG_QC - staged, fixed-correspondence segmentation-overlap QC (reg_qc = 2)
 *
 * Warps the reference and moving native-cell GeoJSONs through each stage of the
 * registration transform (VALIS registrar pickle or STARE manifest). Cell pairs are
 * fixed once at the rigid stage (optimal one-to-one assignment within 1.5 median
 * nuclear radii, bin/utils/cell_pairs.py; params.seg_qc_pairing selects the rule), so
 * per-stage IoU and centroid residuals describe the same cells. The headline
 * `full_transform` record re-pairs in the final stage's frame.
 *
 * On VALIS, without REGISTER's pre-micro checkpoint 'non_rigid' cannot be separated
 * from 'micro' (recorded as `stages_separable: false`). The tiled method needs none
 * and always receives `[]`.
 *
 * Scoring is per pair on a local window, so peak memory is one nucleus. Everything
 * backend-specific comes from lib/WarpBackends.groovy, keyed on the `method` input.
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
    // Micro depth, so the QC can say whether 'rigid' includes micro-rigid: 0 at
    // reg_valis_mode 'default', 2 at 'hires'. Unused by the tiled backend.
    def micro_reg = params.reg_valis_mode == 'hires' ? 2 : 0
    def ctx = [meta: meta, ref_slide: ref_slide, moving_slide: moving_slide,
               stage_checkpoint: stage_checkpoint, micro_reg: micro_reg]
    // `?: ['']`: an empty flag list would leave a blank line after a `\` continuation,
    // ending the command early and running `${args}` on its own.
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
    def ctx     = [micro_reg: params.reg_valis_mode == 'hires' ? 2 : 0]   // as in script:
    def stages     = backend.stages
    def stage_stub = [n_pairs: 0, n_pairs_scored: 0, iou_n: 0, displacement_px_n: 0]
    // base + extras + rest keeps each backend's stubExtras right after stages_separable
    // in the JSON key order.
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
        // The headline record: the final stage, paired in its own frame (bin/warp_seg_qc.py).
        full_transform  : [stage: stages[-1]] + stage_stub +
                          [matching: [method: 'lsa_centroid', anchor_stage: stages[-1], n_pairs: 0]],
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

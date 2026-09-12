/*
 * TILED_SOLVE - STARE fan-out step 3/4: assemble the manifest from per-tile control points.
 *
 * One cheap per-slide reduction (kilobytes): gathers every tile's control point and writes the
 * self-contained transform manifest (M0 + gated mesh) the stitch and reg_qc=2 warper consume.
 * Control points are gated on correlation CONFIDENCE (--max-error) and range (--max-disp)
 * before the TRE gate, which keeps background out of the deformation mesh. Tiles only PARTLY
 * on-section are a known remaining exposure -- see the note on _accept in bin/tiled_solve.py.
 */
process TILED_SOLVE {
    tag "${meta.patient_id}:${meta.channels.join('_')}"
    label 'process_single'

    container 'bolt3x/mirage-tiled:1.0.0'

    input:
    tuple val(meta), path(m0), path(controls, stageAs: 'ctrl_?/*')

    output:
    tuple val(meta), path("*_manifest.json"), emit: manifest
    tuple val(meta), path("*_tre.json")     , emit: tre
    path "versions.yml"                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix    = "${meta.patient_id}_${meta.channels.join('_')}"
    def slidename = meta.channels.join('_')
    def gate      = params.reg_tiled_gate_tre
    // --solver: which SOLVE algorithm builds the mesh from the gated control points
    // ('robust' | 'legacy', see stare.solve). A correctness knob like the gates, so it is a
    // param rather than a tier value.
    def solver    = params.reg_tiled_solver
    // --max-error / --max-disp: the confidence and range gates on the control points, from
    // conf/modules.config's ext.args for this process.
    def args      = task.ext.args ?: ''
    """
    tiled_solve.py \\
        --m0 ${m0} \\
        --controls 'ctrl_*/*_ctrl.json' \\
        --gate-tre ${gate} \\
        --solver ${solver} \\
        ${args} \\
        --moving-name '${slidename}' \\
        --out-manifest ${prefix}_manifest.json \\
        --out-tre ${prefix}_tre.json

    ${ProcessEnvelope.versions(task.process, [], task.container)}
    """

    stub:
    def prefix    = "${meta.patient_id}_${meta.channels.join('_')}"
    def slidename = meta.channels.join('_')
    """
    echo '{"ref_slide":"ref","slides":{"ref":{"M0":[[1,0,0],[0,1,0],[0,0,1]],"mesh":null},"${slidename}":{"M0":[[1,0,0],[0,1,0],[0,0,1]],"mesh":null,"out_shape":[16,16]}}}' > ${prefix}_manifest.json
    echo '{"coarse_tre_px":0,"n_inliers":0,"n_tiles":0,"mesh_refined":false,"rigid_tre_px":{"mean":null,"p50":null,"p90":null,"max":null},"tiles":[]}' > ${prefix}_tre.json
    ${ProcessEnvelope.versionsStub(task.process, [], task.container)}
    """
}

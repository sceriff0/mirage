/*
========================================================================================
    WarpBackends — per-method knobs for the reg_qc=2 segmentation-overlap warp
========================================================================================
    WARP_SEG_QC scores registration by warping reference and moving native-cell GeoJSONs
    through each registration stage and comparing per-pair IoU and centroid residual
    against a correspondence fixed at the rigid anchor. The SCORER is method-agnostic —
    bin/warp_seg_qc.py takes `--method` and builds its warper from either a VALIS
    registrar pickle or a DRAPE transform manifest.

    The two PROCESSES were not. modules/local/warp_seg_qc.nf and warp_seg_qc_tiled.nf
    shared ~77% of their bodies — the whole output: block including its comment, the
    when: block, both tag lines, 8 of 12 flags, 7 of 9 stub-JSON keys — and differed only
    in container, one input element, four flags, the stage list and two stub keys.

    WHY THIS REVERSES A RECORDED DECISION. seg_qc.nf's header argued the split was
    deliberate: merging would need a conditional container directive and conditional
    inputs, "method knowledge pushed back INSIDE a process body". That argument was sound
    when written, but modules/local/segment.nf now does exactly that for three
    segmentation backends via lib/SegBackends.groovy, and shipped green. Holding both
    positions — one table for SEGMENT, two processes for WARP_SEG_QC — costs more than
    either position alone. This class is SegBackends' shape applied to the same problem.

    THE ABSENT STAGE CHECKPOINT IS A NULL OBJECT, NOT A DIFFERENT INPUT SHAPE. The VALIS
    branch already passes `[]` for patients whose checkpoint is missing (seg_qc.nf's
    ch_ckpt_by_patient makes that join total precisely so a missing checkpoint costs a
    stage rather than dropping the patient). The tiled method needs no checkpoint at all,
    so it passes `[]` for every slide. One 8-element input tuple serves both.

    SHELL FRAGMENTS ARE RETURNED AS A LIST OF LINES, never pre-indented — the caller
    joins them with its own indentation, because Nextflow applies stripIndent() to the
    finished script. Same rule as SegBackends.

    `--jvm-heap-gb 8` LIVES HERE, NOT IN conf/modules.config's ext.args, EVEN THOUGH
    CLAUDE.md says tool arguments belong in ext.args. That rule is for genuinely TUNABLE
    values; this is a hardcoded literal that is also backend-specific — exactly like
    `--method tiled` two lines below. It used to live in a
    `params.registration_method == 'tiled' ? '' : '--jvm-heap-gb 8'` ternary in
    conf/modules.config, which reads the GLOBAL PARAM while WARP_SEG_QC selects its
    backend from the `method` INPUT — on the linear path the two always agree, but the
    flag was one config edit away from leaking into a tiled render (observed once, by
    review, in this task). Resolving it from `backend.flags`, which already closes over
    `method`, makes the leak structurally impossible instead of conditionally absent, and
    keeps conf/modules.config's `ext.args` free for its actual purpose: user-supplied
    tunables layered on top, never backend selection.

    All methods are static; nothing here reads params. `method` arrives as an argument,
    which is what keeps params.registration_method read exactly once on the linear path
    (in subworkflows/local/registration.nf).
========================================================================================
*/

class WarpBackends {

    private static final Map<String, Map> BACKENDS = [
        valis: [
            // Same image as REGISTER's classic path, so the registrar pickle loads and
            // scikit-image/scipy are present. DIGEST-PINNED (R6) to the same digest
            // modules/local/register.nf uses: cdgatenbee/valis-wsi:1.0.0 @ 2026-09-02.
            container   : 'cdgatenbee/valis-wsi@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6',
            stages      : ['native', 'rigid', 'non_rigid', 'micro'],
            versionTools: ['valis', 'skimage', 'scipy'],
            flags       : { ctx ->
                // VALIS composes the micro residual into the same displacement field, so
                // without the pre-micro checkpoint 'non_rigid' cannot be separated from
                // 'micro'. Optional but load-bearing; absent as `[]`.
                def out = [
                    "--moving-name '${ctx.moving_slide}'",
                    "--reference-name '${ctx.ref_slide}'",
                    "--micro-reg ${ctx.micro_reg}",
                ]
                if (ctx.stage_checkpoint) out << "--checkpoint-dir ${ctx.stage_checkpoint}"
                // WARP_SEG_QC stages no TIFFs, so the auto-sizer in valis_config.init_jvm
                // always fell through to its 8 GB floor; pinning it makes the reservation a
                // decision instead of an accident of the input-directory scan. A VALIS-only
                // knob, resolved here (not conf/modules.config's ext.args) so it can never
                // leak onto the JVM-free tiled backend regardless of what
                // params.registration_method says — see the class header.
                out << '--jvm-heap-gb 8'
                return out
            },
            stubExtras  : { ctx -> [
                micro_reg                 : ctx.micro_reg,
                rigid_includes_micro_rigid: ctx.micro_reg >= 1,
            ] },
        ],
        tiled: [
            // JVM-free slim image; no BioFormats.
            container   : 'bolt3x/mirage-tiled:1.0.0',
            stages      : ['native', 'rigid', 'refined'],
            versionTools: ['skimage', 'scipy'],
            flags       : { _ctx -> ['--method tiled'] },
            stubExtras  : { _ctx -> [:] },
        ],
    ].asImmutable()

    /**
     * The registration methods WARP_SEG_QC can score, in BACKENDS order.
     *
     * Must equal the set of methods nextflow_schema.json's `registration_method`
     * enum accepts (currently 'valis' and 'tiled') and lib/RegBackends.groovy's
     * `methods()`: a method that registers but cannot be warp-scored would fail
     * only at reg_qc=2, on a real run.
     */
    static List<String> methods() {
        return BACKENDS.keySet() as List
    }

    /**
     * The backend row for `method` -- container, stages, versionTools, flags, stubExtras.
     *
     * Keyed on the `method` ARGUMENT, never params.registration_method: SEG_QC takes
     * the method as a parameter so there is one decision site rather than two reads
     * that can drift.
     *
     * @throws IllegalArgumentException naming `method` and listing methods().
     */
    static Map of(String method) {
        def backend = BACKENDS[method]
        if (!backend)
            throw new IllegalArgumentException(
                "Unknown registration method for WARP_SEG_QC: '${method}'. Valid: ${methods()}")
        return backend
    }

    /** Read by WARP_SEG_QC's container directive. */
    static String container(String method) {
        return of(method).container
    }
}

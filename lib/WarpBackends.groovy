/*
========================================================================================
    WarpBackends — per-method knobs for the reg_qc=2 segmentation-overlap warp
========================================================================================
    One WARP_SEG_QC process serves both backends (as SEGMENT does via SegBackends);
    this table holds what differs: container, stages, versions.yml tools, flags and
    stub-JSON extras. A missing stage checkpoint is passed as `[]` (always, for tiled).

    Shell fragments are lists of lines, never pre-indented: the caller indents them
    because Nextflow stripIndent()s the finished script.

    Backend-specific fixed flags (e.g. --jvm-heap-gb 8) live here rather than in
    ext.args, so they follow the `method` input and cannot leak onto the other backend.

    Static; `method` is an argument, so params.registration_method is read once
    (in subworkflows/local/registration.nf).
========================================================================================
*/

class WarpBackends {

    private static final Map<String, Map> BACKENDS = [
        valis: [
            // Same image as REGISTER, so the registrar pickle loads in the VALIS that wrote it.
            container   : 'bolt3x/mirage-valis:1.0.0',
            stages      : ['native', 'rigid', 'non_rigid', 'micro'],
            versionTools: ['valis', 'skimage', 'scipy'],
            flags       : { ctx ->
                // Without the pre-micro checkpoint 'non_rigid' cannot be separated from 'micro'.
                def out = [
                    "--moving-name '${ctx.moving_slide}'",
                    "--reference-name '${ctx.ref_slide}'",
                    "--micro-reg ${ctx.micro_reg}",
                    // REGISTER's frame, so per-cell residuals land on the mask's cells.
                    "--crop reference",
                ]
                if (ctx.stage_checkpoint) out << "--checkpoint-dir ${ctx.stage_checkpoint}"
                // No TIFFs are staged, so the JVM auto-sizer cannot size it; pin it.
                out << '--jvm-heap-gb 8'
                return out
            },
            stubExtras  : { ctx -> [
                micro_reg                 : ctx.micro_reg,
                rigid_includes_micro_rigid: ctx.micro_reg >= 1,
            ] },
        ],
        tiled: [
            // STARE's image: JVM-free and slim (no BioFormats, no torch).
            container   : 'bolt3x/mirage-stare:1.0.0',
            stages      : ['native', 'rigid', 'refined'],
            versionTools: ['skimage', 'scipy'],
            flags       : { _ctx -> ['--method tiled'] },
            stubExtras  : { _ctx -> [:] },
        ],
    ].asImmutable()

    /**
     * The registration methods WARP_SEG_QC can score, in BACKENDS order.
     *
     * Must equal the schema's `registration_method` enum and RegBackends.methods().
     */
    static List<String> methods() {
        return BACKENDS.keySet() as List
    }

    /**
     * The backend row for `method` -- container, stages, versionTools, flags, stubExtras.
     * Keyed on the `method` argument, never params.registration_method.
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

/*
========================================================================================
    RegBackends — what a registration backend IS, in one table
========================================================================================
    One table for the registration backends (cf. SegBackends). WarpBackends keeps the
    reg_qc=2 warp's own knobs, but both must list the same backends; the schema enum
    must match too (tests/test_reg_backends.py).

      1. nextflow_schema.json's `registration_method` enum (the valid names)
      3. subworkflows/local/register_patient.nf's dispatch (which adapter to invoke)
      4. subworkflows/local/seg_qc.nf's branch (which join shape the transforms have)
      5. lib/WarpBackends.groovy's own two-key map (the reg_qc=2 warp's knobs)

    Optional emits use a null object: an adapter without a stage checkpoint or TRE
    emits Channel.empty() and consumers tolerate zero artifacts.

    Static; `method` is an argument, so params.registration_method is read once
    (in subworkflows/local/registration.nf).
========================================================================================
*/

class RegBackends {

    /*
     * The single table answering "what is this registration backend?".
     *
     *   adapter            the subworkflow name register_patient.nf invokes.
     *   segQcJoin          'per_patient' (one transform per patient, VALIS pickle) or
     *                      'per_slide' (one per moving slide, tiled manifest).
     *   hasStageCheckpoint whether the method writes a pre-micro stage checkpoint.
     *   hasIntrinsicTre    whether the method estimates its own TRE.
     *
     *                      hasStageCheckpoint and hasIntrinsicTre are the ADAPTER
     *                      CONTRACT'S declared optional emits (stage_checkpoint,
     *                      intrinsic_tre — the null-object rule described in this file's
     *                      header): what an adapter for this method is allowed to fill,
     *                      versus what it must emit Channel.empty() for. No production
     *                      call site reads these two fields yet — register_patient.nf
     *                      passes the adapters' emits straight through unrenamed rather
     *                      than branching on them — so today tests/lib_probe.nf's
     *                      checkRegBackends() is their only reader. They stay in the
     *                      table anyway because the master plan for this class mandates
     *                      them as part of the contract, and because the alternative is
     *                      the thing this class exists to prevent: a future consumer
     *                      that needs to know "does this method have a stage checkpoint"
     *                      discovering the answer by reading an adapter instead of
     *                      asking the table.
     *   supportedModes     the run modes this backend may run under. Only 'linear'
     *                      exists on this branch; the dev branch adds 'add_cycle'.
     *   pairsOutputsBySignature
     *                      outputs lose slide identity and are paired back by channel set
     *                      (RegisteredMatch). workflows/mirage.nf uses it to refuse, at
     *                      launch, a patient whose slides share one channel set.
     *   warp               the WarpBackends key for this method.
     */
    static final Map<String, Map> BACKENDS = [
        valis: [adapter: 'VALIS_ADAPTER', segQcJoin: 'per_patient', hasStageCheckpoint: true,
                hasIntrinsicTre: false, supportedModes: ['linear'],              warp: 'valis',
                pairsOutputsBySignature: true],
        tiled: [adapter: 'TILED_ADAPTER', segQcJoin: 'per_slide',   hasStageCheckpoint: false,
                hasIntrinsicTre: true,  supportedModes: ['linear'],              warp: 'tiled',
                pairsOutputsBySignature: false],
    ].asImmutable()

    /** The backend names this table knows, in declaration order. */
    static List<String> methods() {
        return BACKENDS.keySet() as List
    }

    /**
     * The backend entry for `method`.
     *
     * Throws on an unknown name, so accessors never answer `false` for a backend
     * that does not exist.
     */
    static Map of(String method) {
        def backend = BACKENDS[method]
        if (!backend)
            throw new IllegalArgumentException(
                "Unknown registration method '${method}'. Valid: ${methods().join(', ')}. " +
                "(nextflow_schema.json's registration_method enum and this table must be " +
                "widened together; tests/test_reg_backends.py asserts they agree.)")
        return backend
    }

    /** Whether `method` may run under `mode` ('linear' here; dev adds 'add_cycle'). */
    static boolean supportsMode(String method, String mode) {
        return mode in of(method).supportedModes
    }

    /** 'per_patient' | 'per_slide' — the shape seg_qc.nf joins the transforms with. */
    static String segQcJoin(String method) {
        return of(method).segQcJoin
    }
}

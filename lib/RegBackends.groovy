/*
========================================================================================
    RegBackends — what a registration backend IS, in one table
========================================================================================
    One table for the registration backends (cf. SegBackends). WarpBackends keeps the
    reg_qc=2 warp's own knobs, but both must list the same backends; the schema enum
    must match too (tests/test_reg_backends.py).

    The adapter is a String: a workflow cannot be invoked by name from a map, so
    register_patient.nf keeps an `if`. Unknown methods throw rather than fall through
    to VALIS.

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
     *                      (These two are the adapter's optional emits; today only
     *                      tests/lib_probe.nf reads them.)
     *   supportedModes     the params.mode values this backend may run under.
     *   pairsOutputsBySignature
     *                      outputs lose slide identity and are paired back by channel set
     *                      (RegisteredMatch). workflows/mirage.nf uses it to refuse, at
     *                      launch, a patient whose slides share one channel set.
     *   warp               the WarpBackends key for this method.
     */
    static final Map<String, Map> BACKENDS = [
        valis: [adapter: 'VALIS_ADAPTER', segQcJoin: 'per_patient', hasStageCheckpoint: true,
                hasIntrinsicTre: false, supportedModes: ['linear', 'add_cycle'], warp: 'valis',
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

    /** Whether `method` may run under `mode` (params.mode: 'linear' | 'add_cycle'). */
    static boolean supportsMode(String method, String mode) {
        return mode in of(method).supportedModes
    }

    /** 'per_patient' | 'per_slide' — the shape seg_qc.nf joins the transforms with. */
    static String segQcJoin(String method) {
        return of(method).segQcJoin
    }
}

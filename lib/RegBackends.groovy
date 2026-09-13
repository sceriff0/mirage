/*
========================================================================================
    RegBackends — what a registration backend IS, in one table
========================================================================================
    The registration method's identity was re-decided in five places:

      1. nextflow_schema.json's `registration_method` enum (the valid names)
      2. workflows/mirage.nf's add_cycle allowlist (which names that mode accepts)
      3. subworkflows/local/register_patient.nf's dispatch (which adapter to invoke)
      4. subworkflows/local/seg_qc.nf's branch (which join shape the transforms have)
      5. lib/WarpBackends.groovy's own two-key map (the reg_qc=2 warp's knobs)

    Adding or retiring a backend meant finding all five and keeping them consistent, and
    missing one was silent. The silence had a DIRECTION, which is the part worth keeping
    in mind: register_patient.nf's dispatch once read `if (tiled) ... else VALIS`, so any
    method the schema enum gained but that file did not know about registered with VALIS
    and reported success. A benchmark arm would have measured VALIS twice under two
    labels and no test could have told.

    This class is lib/SegBackends.groovy's shape applied to the same problem, one layer
    up: SegBackends keys the three segmenters, WarpBackends keys the reg_qc=2 warp, and
    this keys registration itself. WarpBackends stays a separate table on purpose -- its
    fields are that process's knobs (container, stage list, flag closures) and belong
    with it -- but the two must agree on WHICH backends exist, which
    tests/lib_probe.nf and tests/test_reg_backends.py both assert.

    THE ADAPTER IS A STRING, NOT A REFERENCE. A Nextflow workflow cannot be invoked by
    name out of a map, so register_patient.nf keeps a two-arm `if`. What moves here is
    the DECISION -- which adapter, which join shape, which modes -- not the call.

    OPTIONAL EMITS ARE A NULL-OBJECT CONTRACT, recorded here rather than discovered in a
    consumer. `hasStageCheckpoint` and `hasIntrinsicTre` say what an adapter can fill;
    an adapter for a method that produces neither emits `Channel.empty()` and every
    consumer tolerates zero artifacts. A third backend answers both questions in this
    table instead of adding a required input somewhere.

    All methods are static; nothing here reads params. `method` arrives as an argument,
    which is what keeps params.registration_method read exactly once on the linear path
    (in subworkflows/local/registration.nf).
========================================================================================
*/

class RegBackends {

    /*
     * The single table answering "what is this registration backend?".
     *
     *   adapter            the subworkflow name register_patient.nf invokes. A String;
     *                      see the header on why it cannot be a reference.
     *   segQcJoin          'per_patient' when the method produces ONE transform per
     *                      patient (VALIS's registrar pickle), 'per_slide' when it
     *                      produces one per moving slide (the tiled backend's transform
     *                      manifest). seg_qc.nf builds a different join for each.
     *   hasStageCheckpoint whether the method writes a pre-micro stage checkpoint the
     *                      reg_qc=2 warp can separate 'non_rigid' from 'micro' with.
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
     *   supportedModes     the params.mode values this backend may run under.
     *   warp               the lib/WarpBackends.groovy key for this method. Equal to the
     *                      method name today, and stated rather than assumed so that a
     *                      future backend reusing another's warp does not have to be
     *                      discovered by reading WarpBackends.
     */
    static final Map<String, Map> BACKENDS = [
        valis: [adapter: 'VALIS_ADAPTER', segQcJoin: 'per_patient', hasStageCheckpoint: true,
                hasIntrinsicTre: false, supportedModes: ['linear', 'add_cycle'], warp: 'valis'],
        tiled: [adapter: 'TILED_ADAPTER', segQcJoin: 'per_slide',   hasStageCheckpoint: false,
                hasIntrinsicTre: true,  supportedModes: ['linear'],              warp: 'tiled'],
    ].asImmutable()

    /** The backend names this table knows, in declaration order. */
    static List<String> methods() {
        return BACKENDS.keySet() as List
    }

    /**
     * The backend entry for `method`.
     *
     * Throws on an unknown name rather than returning null. Every accessor below goes
     * through this for that reason: a `supportsMode('ashlar', 'add_cycle')` that answered
     * `false` would read as "that backend does not support add_cycle", which is a
     * different and much more plausible-looking statement than "there is no such
     * backend".
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

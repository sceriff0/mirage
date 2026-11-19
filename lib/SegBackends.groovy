/*
 * SegBackends - the segmentation backend table.
 *
 * Everything that differs between SEGMENT's backends, keyed by `params.seg_method`:
 *
 *   container    the image; read by SEGMENT's `container` directive (directive
 *                closures can see a lib class but not script-block variables).
 *   entrypoint   the bin/ script on $PATH (must be tracked executable).
 *   flags        resolved (not tunable) arguments: output prefix, nuclear channel index.
 *                Tunables live in conf/modules.config's ext.args.
 *   guard        the backend's precondition, as shell lines.
 *   versionTools Python module names; ProcessEnvelope renders them for both script:
 *                and stub:.
 *
 * `flags` and `guard` take [meta, prefix, params: ctxParams(params)] so the cache key
 * stays narrow. Shell fragments are lists of lines, never pre-indented: the caller
 * indents them because Nextflow stripIndent()s the script. Lines containing `$` use
 * single-quoted strings so shell expansions reach the shell verbatim.
 */
class SegBackends {

    /**
     * The parameters the backends above read off `ctx.params`, and the only ones SEGMENT
     * is allowed to depend on.
     *
     * A slice, not `params`: a bare `params` in SEGMENT's script: would put every
     * parameter in its task hash. Built in segmentation.nf (workflow code is not hashed).
     *
     * A backend that reads a new param needs its key here; a missing key reads as null
     * and renders an empty flag (tests/test_seg_backends_ctx_params.py checks).
     */
    static final List<String> CTX_PARAM_KEYS = [
        'nuclear_markers',
        'instanseg_model_dir',
        'cellsam_model_path',
    ].asImmutable()

    /** The `ctx.params` value SEGMENT should be given. Call from workflow code, never from a `script:` block. */
    static Map ctxParams(Map params) {
        return params.subMap(CTX_PARAM_KEYS)
    }

    private static final Map BACKENDS = [

        // StarDist reads channel 0 (--dapi-channel 0), so channel 0 must be nuclear.
        // CONVERT_IMAGE reorders channels; the guard checks it did.
        stardist: [
            container : 'bolt3x/mirage-stardist:1.0.0',
            entrypoint: 'segment.py',
            flags     : { ctx -> '--dapi-channel 0' },
            guard     : { ctx ->
                def channels = ctx.meta.channels ?: []
                def ok = MarkerUtils.isNuclear(channels ? channels[0] : null, ctx.params.nuclear_markers)
                def markers = MarkerUtils.markerList(ctx.params.nuclear_markers).join(', ')
                [
                    '# Runtime validation: channel 0 must be the nuclear marker, because',
                    '# StarDist is invoked with --dapi-channel 0.',
                    "if [ \"${ok}\" != \"true\" ]; then",
                    "    echo \"ERROR: a nuclear channel (${markers}) must be in channel 0 for segmentation\"",
                    "    echo \"Found channels: ${channels ? channels.join(', ') : 'Unknown'}\"",
                    "    echo \"Channel 0: ${channels ? channels[0] : 'Unknown'}\"",
                    '    exit 1',
                    'fi',
                    'echo "Validated: channel 0 is a configured nuclear marker"',
                ]
            },
            // Must be importable in this image (the probe reports a missing one as 'unknown').
            versionTools: ['tensorflow'],
        ],

        // InstanSeg is channel-invariant (no nuclear check). Its model cache defaults to
        // site-packages, read-only under Singularity, so it is redirected to the configured
        // path or the task work dir.
        instantseg: [
            container : 'bolt3x/mirage-instanseg:1.0.0',
            entrypoint: 'segment_instantseg.py',
            flags     : { ctx -> "--prefix ${ctx.prefix}" },
            guard     : { ctx ->
                def cacheDir = ctx.params.instanseg_model_dir ?: '$PWD/.instanseg_cache'
                [
                    '# Redirect the InstanSeg model cache away from the read-only image.',
                    "export INSTANSEG_BIOIMAGEIO_PATH=\"${cacheDir}\"",
                    'mkdir -p "$INSTANSEG_BIOIMAGEIO_PATH"',
                    'echo "InstanSeg model cache: $INSTANSEG_BIOIMAGEIO_PATH"',
                    'echo "Note: InstanSeg is channel-invariant; no nuclear-channel-0 check is enforced."',
                ]
            },
            versionTools: ['instanseg', 'torch'],
        ],

        // CellSAM segments the resolved nuclear channel; refuse index -1, which NumPy
        // would wrap to the last channel. Warn early when weights must be downloaded
        // without DEEPCELL_ACCESS_TOKEN.
        cellsam: [
            container : 'bolt3x/mirage-cellsam:1.0.0',
            entrypoint: 'segment_cellsam.py',
            flags     : { ctx ->
                "--prefix ${ctx.prefix} " +
                "--dapi-channel ${MarkerUtils.nuclearIndex(ctx.meta.channels ?: [], ctx.params.nuclear_markers)}"
            },
            guard     : { ctx ->
                def channels = ctx.meta.channels ?: []
                def idx = MarkerUtils.nuclearIndex(channels, ctx.params.nuclear_markers)
                def markers = MarkerUtils.markerList(ctx.params.nuclear_markers).join(', ')
                def modelPath = ctx.params.cellsam_model_path ?: ''
                [
                    "echo \"Note: CellSAM segments the nuclear channel resolved to index ${idx}; the cell mask is expanded from nuclei.\"",
                    '',
                    '# CellSAM is NOT channel-invariant. Fail loudly when no nuclear channel',
                    '# exists rather than let a negative index wrap around under NumPy.',
                    "if [ \"${idx}\" -lt 0 ]; then",
                    "    echo \"ERROR: no nuclear channel (${markers}) found in [${channels ? channels.join(', ') : 'Unknown'}] for ${ctx.meta.patient_id}\"",
                    '    echo "CellSAM requires a nuclear channel to be present."',
                    '    exit 1',
                    'fi',
                    '',
                    '# Warn early if weight auto-download will be attempted without credentials.',
                    "if [ -z \"${modelPath}\" ] && [ -z \"\${DEEPCELL_ACCESS_TOKEN:-}\" ]; then",
                    '    echo "WARNING: --model-path unset and DEEPCELL_ACCESS_TOKEN is empty; CellSAM weight auto-download will likely fail."',
                    'fi',
                ]
            },
            versionTools: ['cellSAM', 'torch'],
        ],
    ]

    /** The backend names this table knows, in declaration order. */
    static List<String> methods() {
        return BACKENDS.keySet() as List
    }

    /**
     * The backend entry for `method`.
     *
     * Throws on an unknown name: `seg_method` has no schema enum, so this is where a
     * typo is caught.
     */
    static Map of(String method) {
        def backend = BACKENDS[method]
        if (!backend)
            throw new IllegalArgumentException(
                "Unknown seg_method '${method}'. Valid values: ${methods().join(', ')}.")
        return backend
    }

    /** The container image for `method` -- for SEGMENT's `container` directive. */
    static String container(String method) {
        return of(method).container
    }
}

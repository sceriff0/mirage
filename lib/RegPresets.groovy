/*
 * RegPresets - the cost/accuracy tier tables for the two registration backends.
 *
 * Both backends expose the same four-value vocabulary:
 *
 *     high | medium | low | custom
 *
 * `high` is the shipped default and, with one exception, the historical behaviour: every value
 * in the `high` row below is what `nextflow.config` used to declare as that param's literal
 * default -- EXCEPT `coarse_max_dim`, which is now the anchor's REFINE resolution (1024 at `high`
 * and `medium`, 512 at `low`; it shipped as 4096, then 2048 while COARSE was a U-Net matcher). The
 * long note on STARE below says why. Do not restate this row as "unchanged".
 * `medium` and `low` trade accuracy for memory and wall-clock. `custom` starts from `high` and
 * applies whichever individual knobs the user set; anything left unset stays at the `high` value.
 *
 * WHY THE PARAMS ARE null-DECLARED
 * --------------------------------
 * The per-knob override params (`reg_tiled_tile`, ...) are declared `= null` in nextflow.config
 * rather than carrying literal defaults. That is the repo's existing "null means derive a value
 * at runtime" contract -- the same one `reg_jvm_heap_gb` uses -- and it is the ONLY shape that
 * `tests/test_no_duplicate_param_defaults.py` permits a `?:` fallback for. It is also forced:
 * Nextflow evaluates the `params {}` block BEFORE merging CLI arguments, so a preset lookup
 * written there would never see `--reg_tiled_mode low`. That was verified, not assumed: a params
 * block resolving `(mode == 'low') ? 111 : 999` still returned 999 under `--mode low`. Resolution
 * therefore has to happen after the CLI merge -- here, or inline in a config closure.
 *
 * WHY conf/modules.config DUPLICATES THE STARE TABLE
 * --------------------------------------------------
 * `conf/*.config` cannot see `lib/*.groovy` at all: the class name resolves silently against
 * ConfigObject and only fails when the closure runs. TILED_REG_TILE's and TILED_STITCH's
 * `memory = { ... }` closures need the resolved tile/halo/out_tile to size their request, so they
 * inline this table. That duplication is forced, not a style choice -- see the long comment at the
 * top of conf/modules.config for the four alternatives that were tried against Nextflow 26's
 * strict parser and rejected. `tests/test_reg_presets_inlined_in_config.py` pins the two copies
 * together so they cannot drift.
 *
 * An inline map lookup on a CLI-provided param inside a `memory = {}` closure was verified to work
 * on BOTH Nextflow 25.04.7 and 26.04.6, including the `?:` fallback for an unrecognised mode.
 *
 * VALIS IS NOT HERE ON PURPOSE
 * ----------------------------
 * The VALIS tier table stays in `bin/utils/valis_config.py` because its rows hold Python objects
 * -- `feature_detectors.SuperPointFD` and a `SuperGlueMatcher()` instance -- which cannot be
 * expressed in Groovy. Splitting only its numeric half up here would create exactly the two-homes
 * drift this class exists to prevent. `modules/local/register.nf` passes `--memory-mode` plus any
 * explicit numeric overrides, and `register.py` resolves them against the same table.
 */
class RegPresets {

    /** The tier vocabulary, shared by both backends. Mirrored by the schema enums. */
    static final List<String> MODES = ['high', 'medium', 'low', 'custom']

    /** The tier every unset value falls back to, and the shipped default for both backends. */
    static final String DEFAULT_MODE = 'high'

    /*
     * STARE / tiled cost tiers.
     *
     * `tile` + 2*`halo` is the per-task window that drives TILED_REG_TILE's memory request, and
     * `out_tile` drives TILED_STITCH's, so these three are the memory axis. `coarse_max_dim` is
     * the resolution the global transform is solved at -- STARE's counterpart to VALIS's
     * `reg_max_image_dim`, and its dominant runtime-and-accuracy knob. (The guard that ties this
     * axis to VALIS's is benchmarks/tests/test_build_run_plan.py, which exists only on the
     * `benchmarking` branch -- there is no benchmarks/ directory on this one.)
     *
     * Gating and resolution knobs -- reg_tiled_max_disp, reg_tiled_stride, reg_tiled_nuclear_index --
     * are deliberately NOT tiered. The range gate sets what counts as an acceptable displacement
     * vector, a correctness question, not a cost/accuracy trade; the stride is the mesh resolution
     * and is crossed with the tiers in the benchmark arms rather than tied to one.
     *
     * `coarse_max_dim` is the resolution COARSE REFINES its rigid anchor at (stare/coarse_align.py:
     * a 256 px NCC rotation sweep, then +-3 deg at this thumbnail, then sub-pixel translation). It
     * is no longer a memory knob: the anchor is FFT-based and measured 0.25 / 0.43 / 0.40 GB peak
     * RSS for the whole TILED_COARSE stage at 512 / 1024 / 2048 px, band reads of a 16k x 16k
     * tiled OME-TIFF included (the retired DISK+LightGlue U-Net needed ~1.1 + 7.3 * Mpx GB, i.e.
     * ~32 GB at 2048). Above 1024 it buys nothing the pipeline needs: the anchor only has to land
     * inside the per-tile halo, and SOLVE's robust affine absorbs any rotation left over. At
     * 1024 its quantisation bound (half a 0.25 deg step at the half-diagonal) is ~1.6 thumbnail
     * px, i.e. ~65 px full-res on a 40k slide -- inside every tier's halo. So `high` and `medium`
     * share 1024 and `low` keeps 512 (half the CPU of the ~50 refine evaluations).
     */
    static final Map<String, Map<String, Integer>> STARE = [
        high  : [tile: 2048, halo: 256, out_tile: 1024, coarse_max_dim: 1024],
        medium: [tile: 1024, halo: 192, out_tile:  768, coarse_max_dim: 1024],
        low   : [tile:  512, halo: 128, out_tile:  512, coarse_max_dim:  512],
    ]

    /** The STARE knobs that a tier owns, i.e. the ones `--reg_tiled_mode` moves. */
    static final List<String> STARE_KEYS = ['tile', 'halo', 'out_tile', 'coarse_max_dim']

    /**
     * Map a STARE tier key to the pipeline param that overrides it.
     *
     * Kept explicit rather than derived by string concatenation so that renaming a param is a
     * compile-visible edit here instead of a silent lookup miss at runtime.
     */
    static final Map<String, String> STARE_PARAM_OF = [
        tile          : 'reg_tiled_tile',
        halo          : 'reg_tiled_halo',
        out_tile      : 'reg_tiled_out_tile',
        coarse_max_dim: 'reg_tiled_coarse_max_dim',
    ]

    /**
     * The STARE tier row for `mode`.
     *
     * `custom` resolves to the `high` row, which is what makes "anything the user did not set
     * stays at the high value" true. An unrecognised or null mode also falls back to `high`
     * rather than throwing: ParamUtils validates the enum up front, so reaching here with a bad
     * value means validation was bypassed, and a resource closure is the worst possible place to
     * raise (conf/modules.config's errorStrategy has an 'ignore' branch that would swallow it).
     */
    private static Map<String, Integer> stareRow(String mode) {
        return STARE[(mode == 'custom' || !mode) ? DEFAULT_MODE : mode] ?: STARE[DEFAULT_MODE]
    }

    /**
     * Resolve one STARE knob: the explicit override if the user set one, else the tier value.
     *
     * Takes the mode and the override as SCALARS, never the `params` map. A process `script:`
     * block that passes `params` into a helper makes Nextflow hash the whole map, so any
     * unrelated parameter change re-runs the task and everything downstream -- see the resume
     * hashing rules in CLAUDE.md.
     *
     * Uses an explicit null test rather than `?:` because `?:` is falsy-coalescing: a legitimate
     * `--reg_tiled_halo 0` would be silently rewritten to the tier value.
     */
    static int stare(String mode, String key, Object override) {
        if (override != null) {
            return override as int
        }
        def row = stareRow(mode)
        if (!row.containsKey(key)) {
            throw new IllegalArgumentException(
                "Unknown STARE preset key '${key}'. Known keys: ${STARE_KEYS.join(', ')}"
            )
        }
        return row[key]
    }
}

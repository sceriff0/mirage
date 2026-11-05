/*
 * RegPresets - the cost/accuracy tier VOCABULARY of the VALIS registration backend.
 *
 *     high | medium | low | custom
 *
 * `high` is the shipped default; `medium` and `low` trade accuracy for memory and wall-clock;
 * `custom` starts from `high` and applies whichever individual knobs the user set (anything
 * left unset stays at the `high` value). The vocabulary is pinned against
 * `bin/utils/valis_config.py`'s MEMORY_MODES and the schema's `memory_mode` enum by
 * tests/test_reg_presets_inlined_in_config.py.
 *
 * STARE HAS NO TIERS (removed 2026-11-04)
 * ---------------------------------------
 * STARE used to carry a `reg_tiled_mode` tier table here (with its key list, param map and
 * resolver), duplicated inline in four conf/modules.config closures. It runs one default configuration now: the former `high` row is the plain
 * nextflow.config default of `reg_tiled_tile` / `reg_tiled_halo` / `reg_tiled_out_tile` /
 * `reg_tiled_coarse_max_dim`, read directly by the modules and the config closures. A stale
 * `reg_tiled_mode` is answered at launch by ParamUtils.checkRetiredParams, which keeps the
 * retired rows only to name their values in the refusal.
 *
 * WHY THE VALIS TIER-OWNED PARAMS ARE null-DECLARED
 * -------------------------------------------------
 * The per-knob override params (`reg_valis_max_processed_dim`, ...) are declared `= null` in
 * nextflow.config rather than carrying literal defaults. Nextflow evaluates the `params {}`
 * block BEFORE merging CLI arguments, so a preset lookup written there would never see
 * `--memory_mode low`; resolution has to happen after the CLI merge.
 *
 * VALIS'S TABLE IS NOT HERE ON PURPOSE
 * ------------------------------------
 * The VALIS tier table stays in `bin/utils/valis_config.py` because its rows hold Python objects
 * -- `feature_detectors.SuperPointFD` and a `SuperGlueMatcher()` instance -- which cannot be
 * expressed in Groovy. `modules/local/register.nf` passes `--memory-mode` plus any explicit
 * numeric overrides, and `register.py` resolves them against that table.
 */
class RegPresets {

    /** The VALIS tier vocabulary. Mirrored by the schema's memory_mode enum. */
    static final List<String> MODES = ['high', 'medium', 'low', 'custom']

    /** The tier every unset value falls back to, and the shipped default. */
    static final String DEFAULT_MODE = 'high'
}

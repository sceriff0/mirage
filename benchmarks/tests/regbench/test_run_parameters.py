"""T29 (M6): ``run_parameters`` names each run's settings from the method modules' own
constants and resolvers (imported read-only), never from a hand-written list."""

from benchmarks.regbench import variants
from benchmarks.regbench.methods import (
    ashlar,
    deeperhistreg,
    palom,
    stare,
    valis_mirage,
)
from benchmarks.regbench.paper import tables


def _got(name):
    return {k: (v, src) for k, v, src in tables.module_parameters(variants.run_of(name))}


def test_stare_names_its_tier_stride_and_preprocessing():
    got = _got("stare_medium_s128_none")
    assert got["tier"][0] == "medium" and got["stride"][0] == 128
    assert got["tile"][0] == stare.TIERS["medium"]["tile"]
    assert got["max_dim"][0] == stare.TIERS["medium"]["max_dim"]
    assert got["sprep"][0] == "none"
    assert "methods.stare" in got["tier"][1]
    assert _got("stare_clahe")["sprep"][0] == "clahe"


def test_valis_names_its_sizes_and_micro_level():
    got = _got("valis_mirage_low_m0")
    assert got["memory_mode"][0] == "low" and got["micro_reg"][0] == 0
    assert got["micro_reg_fraction"][0] == valis_mirage.ARGS["micro_reg_fraction"]
    assert got["max_image_dim_px"][0] == valis_mirage.ARGS["max_image_dim_px"]
    # the tier's sizes are the pipeline's MEMORY_PRESETS row, read from its source
    assert got["max_processed_image_dim_px"][0] == 512
    assert got["max_non_rigid_registration_dim_px"][0] == 512
    assert "valis_config.py" in got["max_processed_image_dim_px"][1]
    assert _got("valis_mirage")["max_processed_image_dim_px"][0] == 2048


def test_palom_ashlar_and_deeperhistreg():
    p = _got("palom")
    assert p["thumbnail_size"][0] == palom.THUMBNAIL_SIZE
    assert p["n_keypoints"][0] == palom.N_KEYPOINTS
    a = _got("affine_ashlar")
    assert (a["max_shift_um"][0], a["noise_frac"][0], a["affine"][0]) == (15.0, 0.002, "stare")
    assert a["tile"][0] == ashlar.TILE
    d = _got("deeperhistreg_hr_gpu")
    assert d["config"][0] == deeperhistreg.HR_CONFIG
    assert d["device"][0] == "cuda:0" and d["fluo"][0] is False
    assert d["paper_resolution_px"][0] == deeperhistreg.CONFIGS[deeperhistreg.HR_CONFIG]
    assert _got("deeperhistreg_fluo_cpu")["fluo"][0] is True


def test_a_changed_module_constant_moves_the_table(monkeypatch):
    # read from the module, not restated: change the module, the row follows
    monkeypatch.setitem(stare.TIERS, "high", {**stare.TIERS["high"], "tile": 999})
    monkeypatch.setattr(palom, "N_KEYPOINTS", 7)
    assert _got("stare")["tile"][0] == 999
    assert _got("palom")["n_keypoints"][0] == 7


def test_the_table_carries_the_module_rows_beside_the_recorded_ones(monkeypatch, tmp_path):
    monkeypatch.setattr(tables, "terminal_rows", lambda ds: {"stare_medium": "stare_medium"})
    rows = tables.run_parameters(tmp_path, "multiplex")
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys))  # (row, key) is the table's key
    by = {r["key"]: r for r in rows}
    assert by["tile"]["value"] == str(stare.TIERS["medium"]["tile"])
    assert by["tier"]["from"].startswith("benchmarks.regbench.methods.stare")

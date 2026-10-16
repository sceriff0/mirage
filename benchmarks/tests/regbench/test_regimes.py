import pytest

from benchmarks.regbench import regimes as rg


def test_every_regime_has_its_four_levels():
    for name, r in rg.REGIMES.items():
        assert set(r) == {"baseline", "default", "alt", "pct"}, name


def test_baseline_is_stare_unchanged():
    for name, r in rg.REGIMES.items():
        assert r["baseline"].sprep == "none", name  # R4: STARE's plain row always runs


def test_cyclic_if_default_is_the_baseline_so_two_levels():
    keys = [k for k, _ in rg.stare_levels("cyclic_if")]
    assert keys == ["baseline", "alt", "pct"]


def test_cyclic_if_alt_is_clahe_on_the_channel():
    assert rg.level_key(rg.REGIMES["cyclic_if"]["alt"]) == "clahe"


def test_percentile_level_keys():
    assert rg.level_key(rg.Level("channel", "percentile")) == "percentile"


def test_the_only_regime_is_the_cohorts():
    assert set(rg.REGIMES) == {"cyclic_if"}
    assert rg.COHORT_REGIME == "cyclic_if"
    assert rg.DATASET_REGIME == {"multiplex": "cyclic_if"}
    assert [k for k, _ in rg.stare_levels("cyclic_if")] == ["baseline", "alt", "pct"]
    assert [rg.level_key(lv) for _, lv in rg.stare_levels("cyclic_if")] == [
        "none",
        "clahe",
        "percentile",
    ]


def test_each_dataset_maps_to_a_regime():
    from benchmarks.regbench import DATASETS

    for d in DATASETS:
        assert rg.regime_of(d) in rg.REGIMES


def test_unknown_dataset_names_itself():
    with pytest.raises(KeyError, match="no_such"):
        rg.regime_of("no_such")

import pytest

from benchmarks.regbench import regimes as rg


def test_every_regime_has_its_four_levels():
    """``bf_same`` has a fifth, ``none_haem`` (user ruling 2026-10-19: the manuscript's
    HyReCo STARE row, Fig. 3f, haematoxylin with NO intensity preprocessing)."""
    extra = {"bf_same": {"none_haem"}}
    for name, r in rg.REGIMES.items():
        assert set(r) == {"baseline", "default", "alt", "pct"} | extra.get(
            name, set()
        ), name


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
    assert rg.level_key(rg.Level("haem", "percentile")) == "haem_percentile"


def test_the_cohorts_regime_is_unchanged():
    assert rg.COHORT_REGIME == "cyclic_if"
    assert [k for k, _ in rg.stare_levels("cyclic_if")] == ["baseline", "alt", "pct"]
    assert [rg.level_key(lv) for _, lv in rg.stare_levels("cyclic_if")] == [
        "none",
        "clahe",
        "percentile",
    ]


def test_the_regimes_are_the_two_this_plan_uses():
    """Restored from 5a46f778 without ``dapi_he`` and ``cytassist`` (no dataset of this
    plan uses them), and without ``bf_consecutive`` since its only dataset was removed
    (user ruling 2026-10-19). ``grey_smooth`` stays a proxy STARE's adapter accepts."""
    assert set(rg.REGIMES) == {"cyclic_if", "bf_same"}
    assert rg.PROXIES == ("channel", "grey", "haem", "grey_smooth")
    assert rg.SPREPS == ("none", "clahe", "percentile")


def test_the_brightfield_levels_are_the_archived_ones_and_haem_none():
    assert rg.stare_levels("bf_same") == (
        ("baseline", rg.Level("grey", "none")),
        ("default", rg.Level("haem", "clahe")),
        ("alt", rg.Level("grey", "clahe")),
        ("pct", rg.Level("haem", "percentile")),
        ("none_haem", rg.Level("haem", "none")),
    )
    # an RGB baseline is STARE unchanged on the case's own grey proxy: the name `none`
    assert rg.level_key(rg.Level("grey", "none")) == "none"
    # the new level's run names: stare_haem_none, stare_s128_haem_none, ...
    assert rg.level_key(rg.Level("haem", "none")) == "haem_none"


def test_every_dataset_of_the_plan_has_its_regime_now():
    # the two datasets removed by the user's ruling of 2026-10-18 are not here
    want = {
        "multiplex": "cyclic_if",
        # one HyReCo dataset (ruling 2026-10-19 (hyreco)), in hyreco_a's and hyreco_b's place
        "hyreco": "bf_same",
        "mcmicro": "cyclic_if",
        "phenocycler": "cyclic_if",
        "rashid": "cyclic_if",
    }
    assert rg.DATASET_REGIME == want
    for name, regime in want.items():
        assert rg.regime_of(name) == regime
    assert rg.regime_of("phenocycler@stare_mosaic") == "cyclic_if"
    assert rg.regime_of("mcmicro@stage") == "cyclic_if"


def test_each_dataset_maps_to_a_regime():
    from benchmarks.regbench import DATASETS

    for d in DATASETS:
        assert rg.regime_of(d) in rg.REGIMES
    for d in rg.DATASET_REGIME:
        assert rg.regime_of(d) in rg.REGIMES


def test_unknown_dataset_names_itself():
    with pytest.raises(KeyError, match="no_such"):
        rg.regime_of("no_such")
    with pytest.raises(KeyError, match="no_such"):
        rg.regime_of("no_such@stage")


# regimes.regime_of(dataset) and the STARE levels of each regime as e898d763 had them (the
# commit before the two datasets of the ruling of 2026-10-18 were removed), for every dataset that remains and the
# cohort: removing the two moves nothing of the others' STARE runs.
_AT_E898D763 = {
    "multiplex": "cyclic_if",
    "hyreco_a": "bf_same",
    "hyreco_b": "bf_same",
    "mcmicro": "cyclic_if",
    "phenocycler": "cyclic_if",
    "rashid": "cyclic_if",
}
_LEVELS_AT_E898D763 = {
    "cyclic_if": (
        ("baseline", ("channel", "none")),
        ("alt", ("channel", "clahe")),
        ("pct", ("channel", "percentile")),
    ),
    "bf_same": (
        ("baseline", ("grey", "none")),
        ("default", ("haem", "clahe")),
        ("alt", ("grey", "clahe")),
        ("pct", ("haem", "percentile")),
    ),
}
# user ruling 2026-10-19: bf_same gains one level AFTER its four (the order of the
# others, and so every other run's name and options, does not move); the brightfield
# landmark set of bf_consecutive and that regime were removed the same day
_ADDED_SINCE = {"bf_same": (("none_haem", ("haem", "none")),)}


# ruling 2026-10-19 (hyreco): HyReCo's two subsets are one dataset, in the same regime
_MERGED = {"hyreco_a": "hyreco", "hyreco_b": "hyreco"}


def test_the_remaining_datasets_regimes_are_those_of_e898d763():
    from benchmarks.regbench import COST_ONLY_KINDS, PUBLIC

    now = {_MERGED.get(d, d) for d in _AT_E898D763}
    # T32: the synthetic scaling set is cost only and names its runs
    # (variants.SYNTHETIC_RUNS): it has no STARE preprocessing regime
    regimed = {d for d, s in PUBLIC.items() if s["kind"] not in COST_ONLY_KINDS}
    assert now == regimed | {"multiplex"}
    assert "synthetic" in PUBLIC and "synthetic" not in rg.DATASET_REGIME
    for old, regime in _AT_E898D763.items():
        dataset = _MERGED.get(old, old)
        assert rg.regime_of(dataset) == regime, dataset
        assert rg.regime_of(f"{dataset}@raw") == regime, dataset
        got = tuple((k, tuple(lv)) for k, lv in rg.stare_levels(regime))
        assert got == _LEVELS_AT_E898D763[regime] + _ADDED_SINCE.get(regime, ()), (
            dataset
        )
    assert set(rg.DATASET_REGIME) == now

"""Registration benchmark: each method run natively and judged with the transform it
estimated, on synthetic, semi-synthetic and multiplex-IF cases under one contract."""

# The cohort's datasets: what `score`, `tissue` and the launcher take when none is named.
DATASETS = ("synthetic", "semisynth", "multiplex")
# External landmark sets (HyReCo, brightfield). Never scored with the cohort: they have
# their own run list (`variants.runs_for`), their own scorer (`regbench.landmarks`) and
# their own tables. `hyreco_a` is the tuning set, `hyreco_b` the report set.
EXTERNAL_DATASETS = ("hyreco_a", "hyreco_b")
ALL_DATASETS = DATASETS + EXTERNAL_DATASETS
METHODS = (
    "initial",
    "stare",
    "valis",
    "deeperhistreg",
    "valis_mirage",
    "ashlar",
    "palom",
)

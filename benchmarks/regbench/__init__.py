"""Registration benchmark: each method run natively and judged with the transform it
estimated, on the multiplex-IF cohort (and, until they go, two external landmark sets)."""

# The cohort's datasets: what `score`, `tissue` and the launcher take when none is named.
DATASETS = ("multiplex",)
# External landmark sets (brightfield). Never scored with the cohort: each has its own
# run list (`variants.runs_for`), its own scorer and its own tables, and none is part of
# the default submission.
#   HyReCo (`regbench.landmarks`): `hyreco_a` is the tuning set, `hyreco_b` the report set.
#   ANHIR (`regbench.challenge`): `anhir` is the 230 training pairs, each marked as on
#   the tuning or the report side; `anhir_tune` is the tuning side alone, where the
#   preprocessing candidates of the `stare_prep` row are compared and nothing else runs.
HYRECO_DATASETS = ("hyreco_a", "hyreco_b")
ANHIR_TUNE, ANHIR = "anhir_tune", "anhir"
ANHIR_DATASETS = (ANHIR_TUNE, ANHIR)
EXTERNAL_DATASETS = HYRECO_DATASETS + ANHIR_DATASETS
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

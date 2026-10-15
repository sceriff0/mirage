"""Registration benchmark: each method run natively and judged with the transform it
estimated, on the multiplex-IF cohort."""

# The cohort's datasets: what `score`, `tissue` and the launcher take when none is named.
DATASETS = ("multiplex",)
METHODS = (
    "initial",
    "stare",
    "valis",
    "deeperhistreg",
    "valis_mirage",
    "ashlar",
    "palom",
)

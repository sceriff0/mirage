"""Registration benchmark: each method run natively and judged with the transform it
estimated, on synthetic, semi-synthetic and multiplex-IF cases under one contract."""

DATASETS = ("synthetic", "semisynth", "multiplex")
METHODS = ("initial", "stare", "valis", "deeperhistreg", "valis_mirage", "ashlar")

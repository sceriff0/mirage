"""Registration benchmark: each method run natively and judged with the transform it
estimated, on synthetic, semi-synthetic and multiplex-IF cases under one contract."""

# The cohort's datasets: what `score`, `tissue` and the launcher take when none is named.
DATASETS = ("synthetic", "semisynth", "multiplex")
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
# Raw-tile cyclic-IF datasets (`regbench.datasets.rawtiles`): external too, their own
# tables (`tables/<name>/`), never pooled with the cohort. Each is prepared once per
# STITCHING FRONT-END into a case set `<name>@<front end>`: `stage`, `ashlar_edge` and
# `stare_mosaic` stitch every cycle into a mosaic (ordinary multiplex-style cases every
# method runs on); `raw` keeps the tiles, which ASHLAR alone reads natively.
RAW_TILE_DATASETS = ("mcmicro", "rashid", "phenocycler", "codex")
RAW = "raw"
# `frontends.FRONT_ENDS` + `raw`, written out so that importing this package reads no
# image library; test_rawtiles.py holds the two equal
RAW_FRONT_ENDS = ("stage", "ashlar_edge", "stare_mosaic", RAW)
RAW_TILE_CASE_SETS = tuple(f"{d}@{fe}" for d in RAW_TILE_DATASETS for fe in RAW_FRONT_ENDS)
EXTERNAL_DATASETS = HYRECO_DATASETS + ANHIR_DATASETS + RAW_TILE_CASE_SETS
# `score --dataset <name>` takes the raw-tile base name and scores every case set of it
ALL_DATASETS = DATASETS + EXTERNAL_DATASETS + RAW_TILE_DATASETS


def case_set(name):
    """``(base dataset, front end)`` of a case-set name: ``("phenocycler", "raw")`` for
    ``phenocycler@raw``; ``(name, None)`` for a name without ``@``."""
    base, sep, fe = str(name).partition("@")
    return (base, fe) if sep else (base, None)


def is_raw_tiles(name):
    """True for a raw-tile dataset or one of its case sets."""
    return case_set(name)[0] in RAW_TILE_DATASETS
METHODS = (
    "initial",
    "stare",
    "valis",
    "deeperhistreg",
    "valis_mirage",
    "ashlar",
    "palom",
)

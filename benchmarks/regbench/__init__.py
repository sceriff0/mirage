"""Registration benchmark: each method run natively and judged with the transform it
estimated, on the multiplex-IF cohort, and beside it on public datasets that are never
pooled with it."""

# The cohort's datasets: what `score`, `tissue` and the launcher take when none is named.
DATASETS = ("multiplex",)
METHODS = (
    "initial",
    "stare",
    "valis",
    "deeperhistreg",
    "ashlar",
    "palom",
)

# The public datasets of plan F: scored BESIDE the cohort, each into tables of its own
# (``<out>/tables/<dataset>/``, ``<out>/paper/<dataset>/``), never pooled with it and
# never read by default (``DATASETS`` stays the cohort). ``kind`` is what a case of the
# set is scored on, and so which scorer ``score.score`` hands it to: ``landmarks`` (the
# dataset's landmark pairs), ``tissue`` (the tissue-level block of the written images)
# or ``nuclei`` (the matched cells, as the cohort). ``modality`` is the case's
# (``methods.input_rule``). ``raw_tiles``: the set also has its unstitched tiles, run as
# ``<name>@raw`` (a front end; ``base_name``). HyReCo is ONE set, ``hyreco``: subset A
# and, when it is downloaded, subset B, scored together (ruling 2026-10-19).
_BF_LANDMARKS = {"kind": "landmarks", "modality": "brightfield"}
_NUCLEI = {"kind": "nuclei", "modality": "fluorescence"}
PUBLIC = {
    # T32 (user 2026-10-19): generated locally (`regbench synth`), never downloaded; COST
    # only -- the scorer never runs nuclei, pairs or landmarks on it (score.SYNTHETIC).
    # The paper's synthetic matrix (Fig. 2c, Supplementary Fig. S4).
    "synthetic": {"kind": "synthetic", "modality": "fluorescence"},
}
# ARCHIVED public datasets (user ruling 2026-11-06, "still don't delete, archive if you
# can"): the 261105 manuscripts report none of them. They are not in PUBLIC, so no
# launcher, scorer or table takes them (a name of one is refused as an unknown dataset);
# their entries are kept here as they were, and HyReCo's own code (its case writer, its
# download script and the landmark scorer) is under benchmarks/archive/ (README there).
# The machinery a public set runs through stays and is tested on the test-only sets of
# benchmarks/tests/regbench/testpublic.py. Re-activating one is moving its entry back.
ARCHIVED_PUBLIC = {
    "hyreco": dict(_BF_LANDMARKS),
    "mcmicro": {**_NUCLEI, "raw_tiles": True},
    "phenocycler": {**_NUCLEI, "raw_tiles": True},
    "rashid": {**_NUCLEI, "raw_tiles": True},
}
KINDS = ("landmarks", "tissue", "nuclei")
# the kinds whose sets are scored for cost alone (no accuracy table of any kind)
COST_ONLY_KINDS = ("synthetic",)


def base_name(dataset):
    """The dataset a case set belongs to: ``mcmicro@raw`` (a front end of ``mcmicro``)
    is ``mcmicro``; any other name is itself."""
    return str(dataset).split("@", 1)[0]


def public_of(dataset):
    """``PUBLIC``'s entry of a dataset, looked up by its base name (``base_name``), or
    None for a dataset that is not public (the cohort)."""
    return PUBLIC.get(base_name(dataset)) if dataset is not None else None

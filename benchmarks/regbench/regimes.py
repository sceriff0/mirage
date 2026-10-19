"""Per-modality preprocessing regimes for STARE (rulings R4, R5 of 2026-10-14).

Restored from commit 5a46f778 (``benchmarks/regbench/regimes.py``) without its ``dapi_he``
and ``cytassist`` regimes, which no dataset of plan F uses; ``DATASET_REGIME`` names every
dataset of plan F now, so this module is not edited after the Task 8 freeze.

A regime names four LEVELS of STARE's input: ``baseline`` (STARE unchanged: the
fluorescence channel as it is, or the case's own grey proxy of an RGB slide; ``--prep
none``), ``default`` (the modality's chosen preprocessing), ``alt`` (its main alternative)
and ``pct`` (a percentile window). Every STARE cross runs all of them, deduplicated.
``proxy`` is the single-channel image STARE is given: the channel of a fluorescence slide,
or a benchmark-side one-channel image of an RGB slide (listed in ``benchmark_side``);
``sprep`` is STARE's own ``--prep`` (STARE 1.3.0, not a departure). VALIS and every other
method run as shipped (R2).

Why each choice (the reasoning of 2026-10-14, not a measurement):
Percentile is benchmarked as an extra level ``pct`` in every regime (the default proxy +
``--prep percentile``) although STARE's coarse step already applies the same (1, 99.9)
window and its tile matcher is gain-invariant: it measures whether clipping the tails matters.

cyclic_if       same nuclei, same stain -> none; alt CLAHE on the channel (what VALIS does
                on DAPI by default)
bf_same         H&E -> IHC re-stain: haematoxylin is the one shared signal, the IHC
                counterstain is pale -> haem + CLAHE; alt grey + CLAHE
bf_consecutive  different cells, shared architecture -> grey + CLAHE (DeeperHistReg's own
                shipped preprocessing is the same); alt grey_smooth

A raw-tile case set is named ``<name>@<front end>`` and has the regime of ``<name>``.
"""

from __future__ import annotations

from typing import NamedTuple

PROXIES = ("channel", "grey", "haem", "grey_smooth")
SPREPS = ("none", "clahe", "percentile")


class Level(NamedTuple):
    proxy: str
    sprep: str


_FLUO = Level("channel", "none")
_RGB = Level("grey", "none")

REGIMES = {
    "cyclic_if": {
        "baseline": _FLUO,
        "default": _FLUO,
        "alt": Level("channel", "clahe"),
        "pct": Level("channel", "percentile"),
    },
    "bf_same": {
        "baseline": _RGB,
        "default": Level("haem", "clahe"),
        "alt": Level("grey", "clahe"),
        "pct": Level("haem", "percentile"),
    },
    "bf_consecutive": {
        "baseline": _RGB,
        "default": Level("grey", "clahe"),
        "alt": Level("grey_smooth", "none"),
        "pct": Level("grey", "percentile"),
    },
}

DATASET_REGIME = {
    "multiplex": "cyclic_if",
    "anhir": "bf_consecutive",
    "hyreco": "bf_same",
    "mcmicro": "cyclic_if",
    "phenocycler": "cyclic_if",
    "rashid": "cyclic_if",
}


COHORT_REGIME = "cyclic_if"  # the cohort (dataset None) is cyclic IF


def regime_of(dataset):
    try:
        return DATASET_REGIME[str(dataset).split("@", 1)[0]]
    except KeyError:
        raise KeyError(
            f"dataset {dataset!r} has no regime in regimes.DATASET_REGIME"
        ) from None


def stare_levels(regime):
    seen, out = set(), []
    for key in ("baseline", "default", "alt", "pct"):
        lv = REGIMES[regime][key]
        if lv not in seen:
            seen.add(lv)
            out.append((key, lv))
    return tuple(out)


def level_key(level):
    if level in (_FLUO, _RGB):
        return "none"
    proxy = "" if level.proxy == "channel" else f"{level.proxy}_"
    return f"{proxy}{level.sprep}"

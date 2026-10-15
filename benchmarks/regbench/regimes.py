"""Per-modality preprocessing regimes for STARE (rulings R4, R5 of 2026-10-14).

A regime names three LEVELS of STARE's input: ``baseline`` (STARE unchanged: the
fluorescence channel as it is, or inverted luminance for an RGB slide; ``--prep none``),
``default`` (the modality's chosen preprocessing) and ``alt`` (its main alternative).
Every STARE cross runs all three, deduplicated. ``proxy`` is the benchmark-side
single-channel image of an RGB slide (listed in ``benchmark_side``); ``sprep`` is STARE's
own ``--prep`` (STARE 1.3.0, not a departure). VALIS and every other method run as
shipped (R2).

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
dapi_he         DAPI vs haematoxylin, unrelated intensities -> haem + CLAHE on both sides
                (fluorescence side: channel + CLAHE); alt grey + CLAHE
cytassist       nuclei likely unresolved at CytAssist resolution -> grey + CLAHE; alt haem + CLAHE
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
    "dapi_he": {
        "baseline": _RGB,
        "default": Level("haem", "clahe"),
        "alt": Level("grey", "clahe"),
        "pct": Level("haem", "percentile"),
    },
    "cytassist": {
        "baseline": _RGB,
        "default": Level("grey", "clahe"),
        "alt": Level("haem", "clahe"),
        "pct": Level("grey", "percentile"),
    },
}

DATASET_REGIME = {
    "synthetic": "cyclic_if",
    "semisynth": "cyclic_if",
    "multiplex": "cyclic_if",
    "hyreco_a": "bf_same",
    "hyreco_b": "bf_same",
    "anhir_tune": "bf_consecutive",
    "anhir": "bf_consecutive",
}


COHORT_REGIME = "cyclic_if"  # the cohort (dataset None) is cyclic IF; a non-IF cohort would need its own entry


def regime_of(dataset):
    try:
        return DATASET_REGIME[dataset]
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

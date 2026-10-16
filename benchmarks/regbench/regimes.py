"""The cohort's preprocessing regime for STARE (rulings R4, R5 of 2026-10-14).

The cohort is cyclic IF, and its regime names four LEVELS of STARE's input: ``baseline``
(STARE unchanged: the fluorescence channel as it is), ``default`` (the same), ``alt``
(CLAHE on the channel, what VALIS does on DAPI by default) and ``pct`` (a percentile
window). Every STARE cross runs all of them, deduplicated: the levels are ``none``,
``clahe`` and ``percentile``. ``sprep`` is STARE's own preprocessing option (STARE 1.3.0,
not a departure); ``proxy`` is the single-channel image it is given, and is the channel.
VALIS and every other method run as shipped (R2).

Percentile is benchmarked as an extra level in the regime (the channel +
``sprep=percentile``) although STARE's coarse step already applies the same (1, 99.9)
window and its tile matcher is gain-invariant: it measures whether clipping the tails
matters.
"""

from __future__ import annotations

from typing import NamedTuple

PROXIES = ("channel",)
SPREPS = ("none", "clahe", "percentile")


class Level(NamedTuple):
    proxy: str
    sprep: str


_FLUO = Level("channel", "none")

REGIMES = {
    "cyclic_if": {
        "baseline": _FLUO,
        "default": _FLUO,
        "alt": Level("channel", "clahe"),
        "pct": Level("channel", "percentile"),
    },
}

DATASET_REGIME = {
    "multiplex": "cyclic_if",
}


COHORT_REGIME = "cyclic_if"  # the cohort (dataset None) is cyclic IF


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
    if level == _FLUO:
        return "none"
    proxy = "" if level.proxy == "channel" else f"{level.proxy}_"
    return f"{proxy}{level.sprep}"

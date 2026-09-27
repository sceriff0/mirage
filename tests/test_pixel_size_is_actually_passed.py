"""Every process that converts pixels to micrometres must be HANDED the scale.

WHAT THIS CAUGHT. `modules/local/export_geojson.nf` did not pass `--pixel_size` at all,
and `bin/export_geojson.py` carried `default=0.325` in its own argparse. So every
"Centroid X µm", "Centroid Y µm", "MORPH: Area µm²", "MORPH: Perimeter µm",
"MORPH: Convex Area µm²" and axis length written into `cells.geojson` was computed at
0.325 µm/px no matter what the run configured. A run on a 0.5 µm/px objective published
measurements 35% wrong, silently, and the symptom surfaced one repository away -- in
`qupath-extension-flowpath`, which reads exactly those keys.

The bug needed BOTH halves to hide: a module that forgot to pass the flag, and a script
default that made the omission invisible. So this guard checks both halves, statically:

  1. no `bin/*.py` may carry a hardcoded pixel-size default in its argparse, and
  2. every module whose script calls one of those consumers must pass the flag.

Static, not behavioural, on purpose. `-stub` never evaluates a `script:` block, so a stub
run cannot see a rendered command; and the rendered-command nf-test pattern used
elsewhere in this suite could not be made to FAIL for this case when the fix was backed
out (it kept reporting the flag present), so it would have been a guard that proves
nothing -- the exact failure mode CLAUDE.md's "watch every new guard fail" rule is about.

Reads `tests/nfmodel`, never a private regex over the raw file. The first draft of this
guard did use one, and passed with the fix removed: the module still mentioned
`--pixel_size` in the COMMENT explaining why the flag matters, and a raw substring search
cannot tell a comment from an invocation. `strip_comments` is the right view here rather
than `strip_comments_and_strings`, because the flag under assertion lives inside the
script block's quoted shell text -- blanking strings would blank the very thing being
checked.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.nfmodel import processes, strip_comments

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "bin"

# The flags that carry a micrometres-per-pixel scale into a script.
PIXEL_FLAGS = ("--pixel_size", "--pixel-size", "--physical-size-x", "--physical-size-y")


# A pixel size is only ever legitimately absent from a module that never asks for one.
# These are the scripts that DO take a scale, so a module invoking one must supply it.
def _scripts_taking_a_scale() -> set[str]:
    taking = set()
    for py in sorted(BIN.glob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        # argparse registrations span lines, so match the flag inside any add_argument(...)
        for call in re.findall(r"add_argument\((.*?)\)", text, re.S):
            if any(f'"{flag}"' in call or f"'{flag}'" in call for flag in PIXEL_FLAGS):
                taking.add(py.name)
    return taking


def test_no_bin_script_hardcodes_a_pixel_size_default():
    """Half one: the script must not be able to supply a scale nobody chose.

    `params.pixel_size` has no shipped default precisely so that a run asserts a scale or
    asks for 'auto'. An argparse `default=` re-introduces one below the schema, where no
    validation can see it -- and a second declaration of a default is what this repo's
    "parameter defaults live in nextflow.config only" rule exists to prevent.
    """
    offenders = []
    for py in sorted(BIN.glob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        for call in re.findall(r"add_argument\((.*?)\)", text, re.S):
            if not any(f'"{f}"' in call or f"'{f}'" in call for f in PIXEL_FLAGS):
                continue
            m = re.search(r"default\s*=\s*([^,\)\s]+)", call)
            if m and m.group(1) not in ("None",):
                offenders.append(f"{py.name}: default={m.group(1)}")
    assert not offenders, (
        "these scripts can supply a pixel size nobody configured, which is how "
        "cells.geojson came to be written at 0.325 regardless of --pixel_size:\n  "
        + "\n  ".join(offenders)
    )


def test_every_module_invoking_a_scale_consumer_passes_the_scale():
    """Half two: a module that runs such a script must hand it the scale.

    Reads the module source rather than a rendered command, because a rendered command
    only exists at runtime and `-stub` never produces one.
    """
    consumers = _scripts_taking_a_scale()
    assert consumers, "found no bin/ script taking a pixel-size flag; the parse broke"

    offenders = []
    for name, proc in sorted(processes().items()):
        # The script: block only, with comments removed and quoted text kept -- the flag
        # is inside the shell heredoc, and a comment naming it is not an invocation.
        body = strip_comments(proc.script_body or "")
        called = [c for c in consumers if c in body]
        if not called:
            continue
        if not any(flag in body for flag in PIXEL_FLAGS):
            offenders.append(
                f"{name} runs {', '.join(sorted(called))} without a scale flag"
            )
    assert not offenders, (
        "these modules invoke a script that converts pixels to micrometres but never "
        "pass it one, so the script falls back to whatever its own default is:\n  "
        + "\n  ".join(offenders)
    )

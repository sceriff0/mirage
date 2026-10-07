#!/usr/bin/env python3
"""before_after_pairs.py -- the Before | After figures from panels ALREADY drawn.

No rendering: it reads an `_anchor/` directory a supplementary S4 / S7 run left behind
(the bare panels and their `*_overlay.json`) and writes one figure per moving round and
variant into `<root>/before_after/`, titled Before / After, with the scale-bar text and
the channel names as editable text in the PDF. Seconds per figure.

    python3 ~/pipelines/mirage/benchmarks/before_after_pairs.py S4
    python3 ~/pipelines/mirage/benchmarks/before_after_pairs.py S7/052 S7/10338 --formats pdf

Each ROOT is a directory holding `_anchor/` (S4 itself; S7/<patient> for S7). The patient
is read from the manifests, so nothing else needs naming.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import supplementary as sp  # noqa: E402


def patients_of(root: Path) -> list[str]:
    """The patients with a manifest under ROOT/_anchor, from the manifests themselves."""
    found = []
    for mf in sorted((root / "_anchor").glob("*_overlay.json")):
        pid = str(json.loads(mf.read_text()).get("patient") or "")
        if pid and pid not in found:
            found.append(pid)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("roots", nargs="+", type=Path, metavar="ROOT")
    ap.add_argument(
        "--labels",
        choices=sp.OVERLAY_LABELS,
        default="editable",
        help="editable (default) = write the words as text over bare panels; burned = "
        "the panels already carry them; none = write nothing",
    )
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)
    ctx = SimpleNamespace(dpi=a.dpi, formats=a.formats)
    total = 0
    for root in a.roots:
        if not (root / "_anchor").is_dir():
            print(f"{root}: no _anchor/ directory, skipped", file=sys.stderr)
            continue
        pids = patients_of(root)
        if not pids:
            print(f"{root}/_anchor: no *_overlay.json, skipped", file=sys.stderr)
            continue
        for pid in pids:
            written = sp._compose_pairs(ctx, root, pid, a.labels)
            total += len(written)
            print(
                f"{root}: {len(written)} file(s) for {pid} in {root / 'before_after'}"
            )
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())

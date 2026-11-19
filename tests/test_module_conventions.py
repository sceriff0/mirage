#!/usr/bin/env python3
"""Guard against multi-process module files.

Convention: one process per file in `modules/local/`. A second `process` block
forces callers into a two-name `include` and hides a process inside a module
named after another. Only `modules/local/*.nf` is scanned.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = ROOT / "modules" / "local"

# `process NAME {` at the start of a line (ignoring leading whitespace).
PROCESS_RE = re.compile(r"^\s*process\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.MULTILINE)


def find_module_process_counts() -> dict[Path, list[str]]:
    """Return {file: [process names]} for every modules/local/*.nf file."""
    counts: dict[Path, list[str]] = {}
    for path in sorted(MODULES_DIR.glob("*.nf")):
        text = path.read_text()
        counts[path] = PROCESS_RE.findall(text)
    return counts


def test_one_process_per_module_file():
    """Every modules/local/*.nf file must define exactly one `process` block."""
    counts = find_module_process_counts()

    assert counts, "No modules/local/*.nf files found -- scan pattern may be stale."

    offending = []
    for path, names in counts.items():
        if len(names) != 1:
            offending.append(
                f"{path.relative_to(ROOT)}: defines {len(names)} process block(s) "
                f"({', '.join(names) if names else 'none'}) -- expected exactly 1"
            )

    assert not offending, (
        f"{len(offending)} module file(s) violate CLAUDE.md's \"one process per "
        'file in modules/local/" convention. Split each offending file so it '
        "defines exactly one process, and update its include sites:\n"
        + "\n".join(offending)
    )

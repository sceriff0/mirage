"""Has a results root been slimmed (``regbench slim``, ``paper/slim.py``), and to what level?

``<root>/SLIMMED.txt`` is written by ``slim --apply``; its ``level:`` line is ``images``
(the written images and work folders are gone: re-scoring still works) or ``csv`` (the
maps and records are gone too: nothing can be re-scored or reused). Read by ``score``
(which refuses ``csv``), ``reuse`` and ``tissue`` (which say ``slimmed: <level>`` instead
of a missing-file error) and by the launcher (``slim --status``). Standard library only."""

from __future__ import annotations

from pathlib import Path

NAME = "SLIMMED.txt"
LEVELS = ("images", "csv")  # in order: csv removes everything images does, and more


def level(root):
    """``"images"``, ``"csv"`` or None (not slimmed, or no readable level line)."""
    try:
        text = (Path(root) / NAME).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "level" and value.strip() in LEVELS:
            return value.strip()
    return None


def said(root):
    """``"slimmed: <level>"`` or ``""``."""
    lv = level(root)
    return f"slimmed: {lv}" if lv else ""

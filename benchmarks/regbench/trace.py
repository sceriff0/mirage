"""Nextflow's ``trace.txt``, read with the standard library: one dict per task.

A port of ``benchmarks/analysis/lib/load.py::parse_trace`` and of its unit parsers
(``analysis/lib/parsing.py``: ``parse_to_gb``, ``parse_duration``), which go with the
analysis layer; compute-all (``compute_all.py``) is what reads it now.

**Every column is read by its NAME, never by its position.** The trace's columns are
those of ``nextflow.config``'s ``trace.fields``, in that order, and a site or a later
release can change the list; a reader that counts columns reads the wrong one without a
word (the trace column order gotcha). A field the trace does not have reads as NaN.

Units are Nextflow's own: its memory figures print binary multiples under the names
``KB``, ``MB``, ``GB``, ``TB`` (1 GB = 2**30 bytes), and every ``*_gb`` here is in those
units. Durations print as ``1d 2h 3m 4s`` or ``120ms``; the original parser had no
``d``, and a REGISTER task can run longer than a day. ``-`` and an empty cell are NaN
(``NAN``, one object, so two parses of the same file compare equal).
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

NAN = math.nan

FIELDS = (
    "task_id",
    "process",
    "tag",
    "name",
    "status",
    "exit",
    "submit",
    "start",
    "complete",
    "duration_s",
    "realtime_s",
    "pcpu",
    "cpus",
    "memory_gb",
    "peak_rss_gb",
    "peak_vmem_gb",
    "read_gb",
    "write_gb",
)

_MEM_UNITS = {
    "B": 1 / 2**30,
    "KB": 1 / 2**20,
    "MB": 1 / 2**10,
    "GB": 1.0,
    "TB": 2.0**10,
}
_DURATION_UNITS = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


def _blank(val):
    return val is None or str(val).strip() in ("", "-")


def to_gb(val) -> float:
    """``'1.2 GB'`` / ``'512 MB'`` / ``'1073741824'`` (bytes) / ``'0'`` -> GB (2**30
    bytes). ``'-'`` and ``''`` -> NaN."""
    if _blank(val):
        return NAN
    s = str(val).strip()
    if s == "0":
        return 0.0
    m = re.match(r"^([\d.]+)\s*([KMGT]?B)$", s, re.IGNORECASE)
    if not m:
        try:
            return float(s) / 2**30  # a bare number is bytes
        except ValueError:
            return NAN
    return float(m.group(1)) * _MEM_UNITS[m.group(2).upper()]


def to_seconds(val) -> float:
    """``'1d 2h 3m 4s'`` / ``'2m 3s'`` / ``'100ms'`` / ``'1.5s'`` -> seconds. ``'-'``
    and ``''`` -> NaN, and so is text with no unit in it."""
    if _blank(val):
        return NAN
    s = str(val).strip()
    total, matched = 0.0, False
    # milliseconds first: the m of ms is not minutes
    for ms in re.findall(r"([\d.]+)\s*ms", s):
        total += float(ms) / 1000.0
        matched = True
    s = re.sub(r"[\d.]+\s*ms", " ", s)
    for value, unit in re.findall(r"([\d.]+)\s*([dhms])", s):
        total += float(value) * _DURATION_UNITS[unit]
        matched = True
    return total if matched else NAN


def _number(val, kind=float):
    if _blank(val):
        return NAN
    s = str(val).strip().rstrip("%").strip()
    try:
        return kind(s)
    except ValueError:
        try:
            return kind(float(s))
        except ValueError:
            return NAN


def _text(val):
    return "" if _blank(val) else str(val).strip()


def parse(path) -> list[dict]:
    """One dict per line of ``trace.txt`` (``FIELDS``, in that order), every column
    found by its header name."""
    with open(Path(path), newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        out = []
        for raw in reader:
            get = raw.get
            out.append(
                {
                    "task_id": _number(get("task_id"), int),
                    "process": _text(get("process")),
                    "tag": _text(get("tag")),
                    "name": _text(get("name")),
                    "status": _text(get("status")),
                    "exit": _number(get("exit"), int),
                    "submit": _text(get("submit")),
                    "start": _text(get("start")),
                    "complete": _text(get("complete")),
                    "duration_s": to_seconds(get("duration")),
                    "realtime_s": to_seconds(get("realtime")),
                    "pcpu": _number(get("%cpu")),
                    "cpus": _number(get("cpus"), int),
                    "memory_gb": to_gb(get("memory")),
                    "peak_rss_gb": to_gb(get("peak_rss")),
                    "peak_vmem_gb": to_gb(get("peak_vmem")),
                    "read_gb": to_gb(get("rchar")),
                    "write_gb": to_gb(get("wchar")),
                }
            )
    return out

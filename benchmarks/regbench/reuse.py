"""Would a finished result be reused by THIS code? Read-only; see README "Reuse".

For every ``<out>/<variant>/<dataset>/<case>/run.json`` it recomputes the key the
launcher would compute now (``methods.reuse_key``) and names the components of the
recorded key that differ: ``execution``, ``code``, ``inputs``, ``method`` (the four of
the key), ``points`` (the case's points were re-prepared) and ``record`` (the result
is not ``ok``, or its ``warped.npz`` is gone). Nothing is written and no reuse
behaviour changes.

A method whose adapter cannot be loaded or whose key cannot be computed here (a
stand-in, a private plug-in that is not enabled) is ``reusable: no`` with the detail
``KEY_NOT_COMPUTABLE`` and ``code`` among the differences: its code bytes cannot be
fingerprinted, so they cannot be shown equal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import methods
from .cases import load_case, stale_points

COMPONENTS = ("execution", "code", "inputs", "method")
KEY_NOT_COMPUTABLE = "key not computable here"


def _records(out, dataset):
    for path in sorted(Path(out).glob(f"*/{dataset}/*/run.json")):
        try:
            yield path, json.loads(path.read_text())
        except (OSError, ValueError):
            yield path, None


def _key_now(method, cases, out, rec, label):
    """The key the launcher would compute now, or ``(None, reason)``."""
    set_here = False
    try:
        mod = methods.load(method)
        if getattr(mod, "STAGES", None) and not os.environ.get("REGBENCH_STAGES"):
            os.environ["REGBENCH_STAGES"] = "jobs"  # what the cohort's batch script states
            set_here = True
        return methods.reuse_key(method, cases, out, rec.get("opts") or {}, label), ""
    except (Exception, SystemExit) as exc:  # SystemExit: methods.load, argument errors
        return None, f"{KEY_NOT_COMPUTABLE}: {exc}"
    finally:
        if set_here:
            os.environ.pop("REGBENCH_STAGES", None)


def report(cases_root, out, dataset="multiplex"):
    rows = []
    for path, rec in _records(out, dataset):
        variant, case_id = path.parts[-4], path.parts[-2]
        row = {"variant": variant, "case_id": case_id, "method": "", "label": "",
               "reusable": "no", "differs": "", "detail": ""}
        if not rec or not rec.get("ok"):
            row.update(differs="record", detail="no ok record")
            rows.append(row)
            continue
        method, label = rec.get("method", ""), rec.get("label") or None
        row.update(method=method, label=label or "")
        try:
            cases = [load_case(Path(cases_root) / dataset / c)
                     for c in (rec.get("group_cases") or [case_id])]
        except (OSError, ValueError, KeyError) as exc:
            row.update(differs="inputs", detail=f"case not loadable: {exc}")
            rows.append(row)
            continue
        now, detail = _key_now(method, cases, out, rec, label)
        was = rec.get("reuse_key") or {}
        if now is None:
            differs = ["code"]
        else:
            differs = [c for c in COMPONENTS if now.get(c) != was.get(c)]
        case = next((c for c in cases if c.case_id == case_id), cases[0])
        made_for = rec.get("points_sha256")
        if not isinstance(made_for, dict) or stale_points(made_for, case):
            differs.append("points")
        if not (path.parent / "warped.npz").exists():
            differs.append("record")
            detail = detail or "warped.npz is missing"
        row.update(reusable="no" if differs else "yes", differs=",".join(differs),
                   detail=detail)
        rows.append(row)
    return rows

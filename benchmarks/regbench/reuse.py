"""Would a finished result be reused by THIS code? Read-only; see README "Reuse".

For every ``<out>/<variant>/<dataset>/<case>/run.json`` it recomputes the key the
launcher would compute now (``methods.reuse_key``) and names the components of the
recorded key that differ: ``execution``, ``code``, ``inputs``, ``method`` (the four of
the key), ``points`` (the case's points were re-prepared) and ``record`` (the result
is not ``ok``, or its ``warped.npz`` is gone). Nothing is written and no reuse
behaviour changes.

The report is **per result, not per unit; ``opts`` and a requested realcheck are not
compared**. The launcher reuses a unit only when every variant of every case of it is
ok, its ``group_cases`` and ``opts`` equal the current ones and a requested realcheck
exists; a row here whose siblings (same method, label, ``group_cases``) include a ``no``
is marked ``sibling variant not ok`` in ``detail`` (``reusable`` stays what it is for
the record itself). The per-record predicate is ``methods.record_gaps``, shared with
the launcher's ``_reuse``.

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
from .cases import load_case

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
    rows, units = [], {}
    for path, rec in _records(out, dataset):
        variant, case_id = path.parts[-4], path.parts[-2]
        row = {"variant": variant, "case_id": case_id, "method": "", "label": "",
               "reusable": "no", "differs": "", "detail": ""}
        if not isinstance(rec, dict) or not rec.get("ok"):
            row.update(differs="record", detail="no ok record")
            if isinstance(rec, dict):
                units.setdefault(_unit(rec), []).append(row)
            rows.append(row)
            continue
        method, label = rec.get("method", ""), rec.get("label") or None
        row.update(method=method, label=label or "")
        units.setdefault(_unit(rec), []).append(row)
        try:
            cases = [load_case(Path(cases_root) / dataset / c)
                     for c in (rec.get("group_cases") or [case_id])]
        except (OSError, ValueError, KeyError) as exc:
            row.update(differs="inputs", detail=f"case not loadable: {exc}")
            rows.append(row)
            continue
        now, detail = _key_now(method, cases, out, rec, label)
        was = rec.get("reuse_key")
        case = next((c for c in cases if c.case_id == case_id), cases[0])
        # not computable here: compare the record with itself, and name `code` below
        gaps = methods.record_gaps(rec, case, path.parent, was if now is None else now)
        differs = (["code"] if now is None else []) + gaps
        if gaps == ["record"] and not detail:
            detail = "warped.npz is missing"
        row.update(reusable="no" if differs else "yes", differs=",".join(differs),
                   detail=detail)
        rows.append(row)
    for unit_rows in units.values():
        if any(r["reusable"] == "no" for r in unit_rows):
            for r in unit_rows:
                if r["reusable"] == "yes":
                    r["detail"] = "sibling variant not ok"
    return rows


def _unit(rec):
    return (rec.get("method", ""), rec.get("label") or "",
            tuple(rec.get("group_cases") or ()))

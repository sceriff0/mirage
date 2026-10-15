"""The golden snapshot of the cohort's score tables, the reference every removal task is
checked against: the ``multiplex`` rows of the tables, on their key columns and their
value columns.

Regenerate with ``python -m benchmarks.tests.regbench.make_cohort_golden``. A later
task may update the golden only deliberately, with the reason stated in its commit
message; never by regenerating it to make a test pass.

Two kinds of table:

* the SCORE tables (``SCORE_TABLES``): the key columns plus ``VALUE_COLUMNS`` (and the
  paired tables' ``d_*`` differences), nothing else.
* the ACCOUNTING tables (``ACCOUNTING_TABLES``: failures, failure_log, attrition, the
  three ``*_missing`` tables, cost_summary, resources, summary_resources): EVERY column
  except the ones ``excluded`` names, i.e. the counts, statuses, causes and reasons.

Columns left out of the accounting tables, because they differ from one build of the
cohort to the next (measured: two builds, compared) or from one machine to the next:

* seconds and clock: any ``*_s`` / ``*_s_<stat>`` column (register_s, write_s, prep_s,
  elapsed_s, total_to_image_s, wasted_wall_s, ...) and ``wall_*``, ``elapsed_*``.
* CPU, GPU and memory measurements: ``cpu_h*``, ``*_cpu_h*``, ``gb_h*``, ``gpu_h*``,
  ``cpu_efficiency*``, ``*rss*``, ``cgroup*``, ``*_peak_gb``, ``cpu_h_used_per_gpx*``.
* where it ran: ``host``, ``cpu_model``, ``gpu_model``, ``device_name``, ``blas``,
  ``threads_env``, ``cpus_node``, ``cpus_affinity``, ``workers_seen``, ``cpus_requested``,
  ``cpus_asked`` (the machine's CPU count), ``launch_id``, ``submission``, and
  the paths ``log_path`` and ``written_file``.

The ``selfcheck_*`` columns of the score tables are not in VALUE_COLUMNS either.

``synthetic``: ``scored_cohort.build`` makes one ``synthetic`` case and the scoring runs
it. ``test_the_golden_does_not_need_the_synthetic_case`` builds the cohort WITHOUT it
and shows the multiplex golden is the same (measured when this was written), so a task that removes
the ``synthetic`` dataset may drop that case freely."""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
from pathlib import Path

from benchmarks.regbench import score as score_mod

from . import scored_cohort

# The brief's keys (``variant`` is called ``method`` in the tables) and the other columns
# that tell two rows of a table apart; a column a table lacks is skipped.
KEY_COLUMNS = (
    "dataset",
    "variant",
    "method",
    "case_id",
    "group",
    "unit",
    "tier",
    "set",
    "family",
    "pose",
    "subset",
    "against",
    "owner",
    "in_common_block",
    "run",
    "attempt",
    "stage",
)
_VALUES = (
    "pair_fraction pair_fraction_excess dice_matched iou_p50 frac_iou_ge_0.5 "
    "displacement_um_p50 displacement_um_p90 within_1um within_3um count_ratio "
    "fold_pct sd_log_jac n_unmapped coverage nn_distance fallback_frac "
    "imputed_initial excluded_from_summaries"
).split()
VALUE_COLUMNS = tuple(
    name + suffix for name in _VALUES for suffix in ("", "_shared", "_ranked", "_exact")
)
_BASE = (
    "cells summary_cells common_block paired_vs_primary paired_summary "
    "nonband_cells nonband_summary"
).split()
SCORE_TABLES = tuple(_BASE + [name + "_nuclei" for name in _BASE])
ACCOUNTING_TABLES = (
    "failures failure_log attrition nuclei_missing nonband_missing tissue_missing "
    "cost_summary resources summary_resources"
).split()
TABLES = SCORE_TABLES + tuple(ACCOUNTING_TABLES)
_EXCLUDED = re.compile(
    r"(_s|_s_per_round|_s_(med|mean|common_med|common_mean))$"
    r"|^(wall|elapsed)_|_wall_s$"
    r"|cpu_h|gb_h|gpu_h|cpu_efficiency|rss|cgroup|_peak_gb$"
    r"|^(host|cpu_model|gpu_model|device_name|blas|threads_env|cpus_node|cpus_affinity)$"
    r"|^(workers_seen|cpus_requested|cpus_asked)$"
    r"|^(launch_id|submission|log_path|written_file)$"
)
GOLDEN = Path(__file__).parent / "fixtures" / "cohort_golden.json"


def excluded(column):
    return bool(_EXCLUDED.search(column))


def _value(text):
    try:
        number = float(text)
    except ValueError:
        return text
    # NaN is not equal to itself, so it is stored as null
    return None if number != number else round(number, 9)


def _columns(table, names):
    if table in SCORE_TABLES:
        return [c for c in names if c in VALUE_COLUMNS or c.startswith("d_")]
    return [c for c in names if c not in KEY_COLUMNS and not excluded(c)]


def project(out):
    """The golden form of the tables ``score`` wrote under ``out``."""
    got = {}
    for table in TABLES:
        path = Path(out) / "tables" / f"{table}.csv"
        if not path.exists():
            got[table] = []
            continue
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            names = reader.fieldnames or []
            values = _columns(table, names)
            keep = [c for c in KEY_COLUMNS if c in names] + values
            rows = [
                {c: (_value(r[c]) if c in values else r[c]) for c in keep}
                for r in reader
                if r.get("dataset") == "multiplex"
            ]
        rows.sort(
            key=lambda r: (
                tuple(str(r.get(k, "")) for k in KEY_COLUMNS),
                json.dumps(r, sort_keys=True),
            )
        )
        got[table] = rows
    return got


def snapshot(out_dir, build=None):
    root, out, _ = (build or scored_cohort.build)(out_dir)
    score_mod.score(root, out, **scored_cohort.SCORE_KW)
    return project(out)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        snap = snapshot(Path(tmp))
    GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print({t: len(r) for t, r in snap.items()}, file=sys.stderr)

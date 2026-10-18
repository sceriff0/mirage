"""The few patients of the CPU cross, and a cases folder that holds only them.

The CPU cross (README, "The CPU cross") runs each family's best-Dice row at 4 and 8 CPUs
on a few patients (user ruling 2026-10-18: 4 and 8 only). Which ones is a rule, not a
choice: ``pick`` takes the patients at the smallest, median and largest total moving
gigapixels (the sum of ``Case.mov_hw`` over a patient's rounds), ties broken by name. ``write`` makes a cases folder of those patients
alone, as symlinks to their prepared case folders, so ``REGBENCH_CASES`` can point at it
and the launcher sizes its arrays from it; nothing is copied and nothing prepared again.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .cases import groups, list_cases

RULE = "smallest, median and largest by total moving gigapixels"


def _gigapixels(case):
    hw = case.mov_hw
    if not hw or len(hw) < 2:
        raise SystemExit(
            f"{case.dataset}/{case.case_id}: its case.json names no mov_hw, and the "
            "patients are ranked by their moving slides' size"
        )
    return int(hw[0]) * int(hw[1]) / 1e9


def sizes(cases_root, dataset="multiplex"):
    """``[(patient, total moving gigapixels)]``, smallest first, ties by name."""
    by_group = groups(list_cases(cases_root, dataset))
    if not by_group:
        raise SystemExit(f"no {dataset} case is prepared under {cases_root}")
    totals = [(g, sum(_gigapixels(c) for c in cs)) for g, cs in by_group.items()]
    return sorted(totals, key=lambda t: (t[1], t[0]))


def pick(cases_root, dataset="multiplex", n=3):
    """The patients at ``n`` evenly spaced ranks of ``sizes``, both ends included: for
    3, the smallest, the median (the lower one of an even count) and the largest. Fewer
    when the dataset has fewer patients. ``[(patient, gigapixels)]``, smallest first."""
    if n < 1:
        raise SystemExit(f"--n {n}: at least one patient")
    ranked = sizes(cases_root, dataset)
    last = len(ranked) - 1
    if n == 1:
        ranks = [last // 2]
    else:
        ranks = [i * last // (n - 1) for i in range(n)]
    return [ranked[r] for r in sorted(set(ranks))]


def write(cases_root, groups_wanted, dest, dataset="multiplex"):
    """``dest/<dataset>/<case_id>``: a symlink to each case folder of the named patients,
    and ``dest/manifest.csv`` copied from ``cases_root`` when it is there (the scorer's
    attrition table reads it). Refuses a ``dest`` that exists and is not empty, and a
    patient that is not prepared, before anything is written. Returns the number of
    cases linked."""
    root, dest = Path(cases_root).resolve(), Path(dest)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise SystemExit(f"{dest} already exists and is not empty: name a new folder")
    by_group = groups(list_cases(root, dataset))
    wanted = [g for g in dict.fromkeys(str(g).strip() for g in groups_wanted) if g]
    if not wanted:
        raise SystemExit("no patient named")
    missing = [g for g in wanted if g not in by_group]
    if missing:
        raise SystemExit(
            f"{dataset}: no patient {', '.join(missing)} is prepared under {root}"
        )
    (dest / dataset).mkdir(parents=True, exist_ok=True)
    n = 0
    for g in wanted:
        for case in by_group[g]:
            source = root / dataset / str(case.case_id)
            (dest / dataset / str(case.case_id)).symlink_to(source.resolve())
            n += 1
    if (root / "manifest.csv").exists():
        shutil.copyfile(root / "manifest.csv", dest / "manifest.csv")
    return n


def staged_stare(out):
    """The STARE result folders under ``out`` whose run was its stage jobs: ``run.json``
    of method ``stare`` whose reuse key says ``execution: jobs`` (``methods.reuse_key``),
    or, in a record without that part, whose ``info.jobs.execution`` does
    (``methods/stare.py``). A ``local`` launch (the CPU cross) into such a folder would
    run every one of those units again under another key and replace it, so the
    launcher refuses it (``slurm/submit_cohort.sh``). Unreadable records are skipped."""
    import json

    found = []
    for path in sorted(Path(out).glob("*/*/*/run.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("method") != "stare":
            continue
        key = rec.get("reuse_key")
        execution = key.get("execution") if isinstance(key, dict) else None
        if execution is None:
            jobs = (rec.get("info") or {}).get("jobs")
            execution = jobs.get("execution") if isinstance(jobs, dict) else None
        if execution == "jobs":
            found.append(path.parent)
    return found

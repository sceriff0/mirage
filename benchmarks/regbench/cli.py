"""``python -m benchmarks.regbench``: prepare cases, run a method, score everything.

``prepare <dataset>``  write the cases of ``synthetic`` / ``semisynth`` / ``multiplex`` (idempotent)
``list``               the number of prepared cases of a dataset (sizes a SLURM array), their
                       ids (``--ids``) or their patients (``--groups``)
``run``                register ONE case (``--case-id``), ONE patient (``--group``), or the
                       ``--index``-th of them: the index counts cases for a method that runs
                       per pair and patients for one that runs per patient
``score``              score every variant under ``--out``; writes ``<out>/tables``. Every
                       result name of the run list is expected (a missing one is a "not
                       run" row) and the primary rows lead; ``--only-run`` scores just
                       what is there
``runs``               the run list, one tab-separated line per run, for the launcher
                       (columns: ``benchmarks/regbench/variants.py``); ``--all`` adds the
                       paused runs
``gate``               from one run's ``selfcheck.json``: the status that decides whether
                       the run may start, and the rows to leave out of the results
                       (``score --drop``). Exit code as ``selfcheck``
``selfcheck``          check that a method's points follow the image its own writer produced,
                       on a synthetic pair; exit 0 ``ok``, 1 ``inconsistent``, 2
                       ``not_checked``, 3 ``crashed``. Anything but 0 blocks the method

``run --realcheck`` does the same kind of check on the real case it has just registered
and exits 3 when it is flagged.

Run from the repository root.
"""

from __future__ import annotations

import argparse

from . import DATASETS, METHODS
from .resources import n_cpus


def build_parser():
    ap = argparse.ArgumentParser(prog="regbench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare").add_subparsers(dest="dataset", required=True)

    def common(p):
        p.add_argument(
            "--cases", required=True, help="folder the cases are written under"
        )
        p.add_argument(
            "--index",
            type=int,
            default=None,
            help="prepare only this one case (SLURM_ARRAY_TASK_ID)",
        )
        p.add_argument("--workers", type=int, default=n_cpus())
        p.add_argument(
            "--force", action="store_true", help="redo cases that already exist"
        )

    s = prep.add_parser("synthetic")
    common(s)
    s.add_argument(
        "--suite",
        default="full",
        choices=("test", "ci", "full", "scale", "dev", "truth"),
        help="`truth` is the frozen known-truth suite of truth_spec.json",
    )

    m = prep.add_parser("multiplex")
    common(m)
    m.add_argument(
        "--manifest",
        required=True,
        help="CSV: case_id,reference,moving,... (for the arms cohort: the output of "
        "`python -m benchmarks.regbench.arms_manifest`)",
    )
    m.add_argument(
        "--diameter",
        type=float,
        default=20.0,
        help="nuclear diameter in px, for the built-in segmenter and the tile halo",
    )
    m.add_argument("--tile", type=int, default=2048)
    m.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="score a random sample of this many moving cells (0 = all); not for "
        "nuclei given as GeoJSON",
    )

    ss = prep.add_parser("semisynth")
    common(ss)
    ss.add_argument(
        "--image",
        default=None,
        help="OME-TIFF with a real nuclear channel (required without --manifest)",
    )
    ss.add_argument("--channel", type=int, default=0)
    ss.add_argument(
        "--mov-image",
        default=None,
        help="another round of the same section, already registered to --image",
    )
    ss.add_argument("--mov-channel", type=int, default=0)
    ss.add_argument(
        "--name", default=None, help="prefix of the case ids (default: the file name)"
    )
    ss.add_argument("--n", type=int, default=4096, help="window side, px")
    ss.add_argument(
        "--windows", type=int, default=8, help="test windows (one more is dev)"
    )
    ss.add_argument(
        "--residual-um",
        type=float,
        default=2.0,
        help="RMS displacement of the multiscale family, micrometres",
    )
    ss.add_argument("--diameter", type=float, default=20.0, help="nuclear diameter, px")
    ss.add_argument(
        "--noise",
        type=float,
        default=1.0,
        help="added noise, in units of the image's background noise (same-image variant)",
    )
    ss.add_argument("--pixel-size-um", type=float, default=None)
    ss.add_argument(
        "--truth",
        action="store_true",
        help="the frozen known-truth cases of truth_spec.json: window side, windows, "
        "noise and amplitudes come from the spec. With --manifest for every patient "
        "of the cohort; with --name --image --channel for one patient",
    )
    ss.add_argument(
        "--manifest",
        default=None,
        help="with --truth: the cohort's list, either the rounds.csv of "
        "`python -m benchmarks.regbench.arms_manifest` or the arms' input.csv; each "
        "patient's reference round and nuclear channel are read from it",
    )
    ss.add_argument(
        "--patients",
        default=None,
        metavar="ID,ID,...",
        help="with --manifest: only these patients (an id not in the manifest is an "
        "error)",
    )
    ss.add_argument(
        "--patients-file",
        default=None,
        help="with --manifest: the same subset, one id per line (# starts a comment)",
    )

    lst = sub.add_parser("list")
    lst.add_argument("--cases", required=True)
    lst.add_argument("--dataset", required=True, choices=DATASETS)
    what = lst.add_mutually_exclusive_group()
    what.add_argument("--ids", action="store_true", help="print the case ids")
    what.add_argument(
        "--groups", action="store_true", help="print the patients, one per line"
    )

    r = sub.add_parser("run")
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--dataset", required=True, choices=DATASETS)
    r.add_argument("--method", required=True, choices=METHODS)
    which = r.add_mutually_exclusive_group(required=True)
    which.add_argument("--case-id", default=None, help="one case (a pair method)")
    which.add_argument("--group", default=None, help="every case of one patient")
    which.add_argument(
        "--index",
        type=int,
        default=None,
        help="SLURM_ARRAY_TASK_ID: into the cases for a pair method, into the "
        "patients for a per-patient one",
    )
    r.add_argument(
        "--label",
        default=None,
        help="name this configuration's results instead of the method's name",
    )
    r.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a method option; repeatable (see the method's module docstring)",
    )
    r.add_argument(
        "--work", default=None, help="scratch for the method's intermediates"
    )
    r.add_argument("--keep-work", action="store_true")
    r.add_argument(
        "--realcheck",
        action="store_true",
        help="after the method's writer, check that each mapping follows the image "
        "written for it (realcheck.json); exit 3 when one is flagged",
    )

    chk = sub.add_parser("selfcheck")
    chk.add_argument("--method", required=True, choices=METHODS)
    chk.add_argument("--label", default=None)
    chk.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE")
    chk.add_argument("--work", required=True, help="where the pair and results go")
    chk.add_argument(
        "--mask",
        default=None,
        choices=("overlap", "seams"),
        help="leave out the dots the method says to (ASHLAR `overlap`: tile overlap "
        "bands; PALOM `seams`: edges between blocks with different shifts)",
    )
    chk.add_argument(
        "--m0-from",
        default=None,
        help="the --work of a `stare` selfcheck on the same pair (anchored methods)",
    )
    chk.add_argument(
        "--size",
        action="append",
        default=None,
        metavar="HxW[:HxW]",
        help="check on this pair (reference[:moving]) instead of the method's own "
        "sizes; repeatable",
    )
    chk.add_argument("--seed", type=int, default=0)
    chk.add_argument("--keep-work", action="store_true")
    chk.add_argument(
        "--in-process",
        action="store_true",
        help="do not start a process per pair (what that process itself is given)",
    )

    sc = sub.add_parser("score")
    sc.add_argument("--cases", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--dataset", nargs="*", default=None, choices=DATASETS)
    sc.add_argument("--workers", type=int, default=n_cpus())
    sc.add_argument(
        "--dev",
        action="store_true",
        help="score the dev_* cases instead of leaving them out",
    )
    sc.add_argument(
        "--only-run",
        action="store_true",
        help='score only the variants that have results: no "not run" rows for the '
        "rest of the run list and no primary rows (a laptop run of one method)",
    )

    sc.add_argument(
        "--drop",
        action="append",
        default=[],
        metavar="VARIANT",
        help="leave this result row out of every table and name it in dropped.csv: a "
        "row whose own selfcheck failed without blocking its method (`gate` prints "
        "them). Repeatable",
    )
    sc.add_argument(
        "--no-tissue",
        action="store_true",
        help="leave out the tissue-level block of the written images (tissue*.csv)",
    )

    ts = sub.add_parser(
        "tissue",
        help="compute the tissue-level pieces of every written image (needs OpenCV); "
        "`score` then reads them",
    )
    ts.add_argument("--cases", required=True)
    ts.add_argument("--out", required=True)
    ts.add_argument("--dataset", nargs="*", default=None, choices=DATASETS)
    ts.add_argument("--workers", type=int, default=n_cpus())
    ts.add_argument("--dev", action="store_true")
    ts.add_argument("--only-run", action="store_true")

    gt = sub.add_parser(
        "gate",
        help="from a run's selfcheck.json: may the run start, and which of its rows "
        "are left out of the results",
    )
    gt.add_argument("--name", required=True, help="a run name (`runs`, column 1)")
    gt.add_argument("--selfcheck", required=True, help="the run's selfcheck.json")

    rn = sub.add_parser("runs")
    rn.add_argument(
        "--header", action="store_true", help="print the column names first"
    )
    rn.add_argument(
        "--all",
        action="store_true",
        help="also print the paused runs (0 in the `active` column)",
    )
    return ap


def _prepare(a):
    import importlib

    return importlib.import_module(f".datasets.{a.dataset}", __package__).prepare(a)


def _run(a):
    from .cases import groups, list_cases
    from .methods import parse_opts, run_case, run_group, scope_of

    cases = list_cases(a.cases, a.dataset)
    by_group = groups(cases)
    per_patient = scope_of(a.method) == "group"
    if a.case_id is not None:
        if per_patient:
            raise SystemExit(
                f"{a.method} runs per patient: give --group or --index, not --case-id"
            )
        picked = [c for c in cases if c.case_id == str(a.case_id)]
        if not picked:
            raise SystemExit(f"{a.dataset}/{a.case_id} is not prepared under {a.cases}")
    elif a.group is not None:
        picked = by_group.get(str(a.group))
        if not picked:
            raise SystemExit(
                f"{a.dataset}: no patient {a.group!r} is prepared under {a.cases}"
            )
    else:
        units = list(by_group.values()) if per_patient else [[c] for c in cases]
        if a.index >= len(units):
            what = "patients" if per_patient else "cases"
            print(
                f"index {a.index}: only {len(units)} {a.dataset} {what}; nothing to do"
            )
            return 0
        picked = units[a.index]
    opts = parse_opts(a.opt)
    rest = (opts, a.label, a.work, a.keep_work)
    if per_patient or a.group is not None:
        done = run_group(a.method, picked, a.out, *rest, realcheck=a.realcheck)
    else:
        case = picked[0]
        got = run_case(a.method, case, a.out, *rest, realcheck=a.realcheck)
        done = {case.case_id: got}
    for case_id, variants in done.items():
        print(
            f"{a.dataset}/{case_id} [{a.label or a.method}]: "
            f"{', '.join(variants) if variants else 'FAILED'}"
        )
    # A failure is recorded in run.json and scored at the unregistered pose; the exit code
    # says whether the method produced anything at all.
    if not any(done.values()):
        return 1
    if a.realcheck and _realcheck_flagged(a, picked, done):
        return 3
    return 0


def _realcheck_flagged(a, picked, done):
    """Print each realcheck; True when one is flagged."""
    import json

    from .cases import result_dir

    flagged = False
    by_id = {c.case_id: c for c in picked}
    for case_id, variants in done.items():
        for variant in variants:
            d = result_dir(a.out, variant, by_id[case_id])
            try:
                check = json.loads((d / "run.json").read_text()).get("realcheck")
            except (OSError, ValueError):
                check = None
            if check:
                print(f"{a.dataset}/{case_id} [{variant}] realcheck: {check['status']}")
                flagged = flagged or bool(check.get("flag"))
    return flagged


def _selfcheck(a):
    from . import selfcheck
    from .methods import parse_opts

    record = selfcheck.run(
        a.method,
        a.work,
        parse_opts(a.opt),
        a.label,
        a.mask,
        a.m0_from,
        [selfcheck.parse_size(s) for s in a.size] if a.size else None,
        a.seed,
        a.in_process,
        a.keep_work,
    )
    for variant, status in record["variants"].items():
        print(f"selfcheck [{variant}]: {status}")
    print(f"selfcheck [{a.label or a.method}]: {record['status']}")
    return selfcheck.EXIT_CODE[record["status"]]


def _list(a):
    from .cases import groups, list_cases

    cases = list_cases(a.cases, a.dataset)
    if a.ids:
        print("\n".join(c.case_id for c in cases))
    elif a.groups:
        print("\n".join(groups(cases)))
    else:
        print(len(cases))
    return 0


def _runs(a):
    from . import variants

    if a.header:
        print("\t".join(variants.RUNS_COLUMNS))
    for row in variants.runs_table(paused=a.all):
        print("\t".join(row))
    return 0


def _gate(a):
    """One line, tab-separated: the status that decides whether the run may start (the
    worst over its gating rows), and the rows to leave out of the results (``-`` for
    none). The exit code is that status's, as ``selfcheck`` returns it: 0 only for ok."""
    import json
    from pathlib import Path

    from . import selfcheck, variants

    try:
        variants.run_of(a.name)
    except KeyError as exc:
        raise SystemExit(f"gate: {exc.args[0]}; see `runs --all`") from None
    record = json.loads(Path(a.selfcheck).read_text())
    status = variants.gate_status(a.name, record)
    dropped = variants.dropped_variants(a.name, record)
    print(f"{status}\t{','.join(dropped) or variants.NONE}")
    return selfcheck.EXIT_CODE[status]


def _dropped(names):
    """The ``--drop`` names, refused when one is a typo or a row that gates a run."""
    from . import variants

    for name in names:
        try:
            ok = variants.droppable(name)
        except KeyError as exc:
            raise SystemExit(f"--drop: {exc.args[0]}") from None
        if not ok:
            raise SystemExit(
                f"--drop {name}: that row gates its run (`runs`, last column); when "
                "its selfcheck fails the method is blocked, the row is not dropped"
            )
    return list(names)


def _tissue(a):
    """The pieces of every written image, left beside each reduced plane (tissue.json)."""
    from . import score, tissue, variants
    from .cases import list_cases

    if not tissue.have_cv2():
        raise SystemExit(
            "tissue: OpenCV (cv2) is not importable in this environment; the pieces "
            "are the released ACCREDIT code's and need it"
        )
    expected = None if a.only_run else variants.expected_variants()
    jobs = []
    for ds in a.dataset or DATASETS:
        cases = [
            c for c in list_cases(a.cases, ds) if c.case_id.startswith("dev_") == a.dev
        ]
        found = score.find_variants(a.out, ds, expected) if cases else []
        jobs += [(a.out, c, found) for c in cases]
    if not jobs:
        raise SystemExit(f"no prepared cases under {a.cases}")
    got = tissue.score_all(jobs, a.workers)
    print(
        f"tissue: {len(got['tissue'])} row(s), {len(got['tissue_missing'])} without "
        f"one, {len(got['tissue_dose'])} dose row(s); `score` writes the tables"
    )
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "prepare":
        return _prepare(a)
    if a.cmd == "list":
        return _list(a)
    if a.cmd == "run":
        return _run(a)
    if a.cmd == "selfcheck":
        return _selfcheck(a)
    if a.cmd == "runs":
        return _runs(a)
    if a.cmd == "gate":
        return _gate(a)
    if a.cmd == "tissue":
        return _tissue(a)
    from . import score, variants

    expected = None if a.only_run else variants.expected_variants()
    primary = None if a.only_run else list(variants.PRIMARY)
    score.score(
        a.cases,
        a.out,
        a.dataset,
        a.workers,
        dev=a.dev,
        expected=expected,
        primary=primary,
        with_tissue=not a.no_tissue,
        dropped=_dropped(a.drop),
    )
    return 0

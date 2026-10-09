"""``python -m benchmarks.regbench``: prepare cases, run a method, score everything.

``prepare <dataset>``  write the cases of ``synthetic`` / ``semisynth`` / ``multiplex`` (idempotent)
``list``               the number of prepared cases of a dataset (sizes a SLURM array), their
                       ids (``--ids``) or their patients (``--groups``)
``run``                register ONE case (``--case-id``), ONE patient (``--group``), or the
                       ``--index``-th of them: the index counts cases for a method that runs
                       per pair and patients for one that runs per patient
``score``              score every variant under ``--out``; writes ``<out>/tables``

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
        "--manifest", required=True, help="CSV: case_id,reference,moving,..."
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
        help="score a random sample of this many moving cells (0 = all)",
    )

    ss = prep.add_parser("semisynth")
    common(ss)
    ss.add_argument(
        "--image", required=True, help="OME-TIFF with a real nuclear channel"
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
        help="the frozen known-truth cases of truth_spec.json for the patient --name: "
        "window side, windows, noise and amplitudes come from the spec",
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
    if per_patient or a.group is not None:
        done = run_group(a.method, picked, a.out, opts, a.label, a.work, a.keep_work)
    else:
        case = picked[0]
        got = run_case(a.method, case, a.out, opts, a.label, a.work, a.keep_work)
        done = {case.case_id: got}
    for case_id, variants in done.items():
        print(
            f"{a.dataset}/{case_id} [{a.label or a.method}]: "
            f"{', '.join(variants) if variants else 'FAILED'}"
        )
    # A failure is recorded in run.json and scored at the unregistered pose; the exit code
    # says whether the method produced anything at all.
    return 0 if any(done.values()) else 1


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


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "prepare":
        return _prepare(a)
    if a.cmd == "list":
        return _list(a)
    if a.cmd == "run":
        return _run(a)
    from .score import score

    score(a.cases, a.out, a.dataset, a.workers, dev=a.dev)
    return 0

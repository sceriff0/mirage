"""``python -m regbench``: prepare cases, run a method on one case, score everything.

``prepare <dataset>``  write the cases of ``synthetic`` / ``anhir`` / ``multiplex`` (idempotent)
``list``               the number of prepared cases of a dataset (sizes a SLURM array), or their ids
``run``                register ONE case with ONE method (``--case-id`` or ``--index``)
``score``              score every variant under ``--out``; writes ``<out>/tables``
``calibrate``          pick each method's options from its runs on the ``dev_*`` cases
``anhir-submission``   lay a variant's warped ANHIR landmarks out for the challenge server

Run with ``PYTHONPATH=src:benchmarks`` from the repository root.
"""

from __future__ import annotations

import argparse
import os

from . import DATASETS, METHODS


def _cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def build_parser():
    ap = argparse.ArgumentParser(prog="regbench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare").add_subparsers(dest="dataset", required=True)

    def common(p):
        p.add_argument("--cases", required=True, help="folder the cases are written under")
        p.add_argument("--index", type=int, default=None,
                       help="prepare only this one case (SLURM_ARRAY_TASK_ID)")
        p.add_argument("--workers", type=int, default=_cpus())
        p.add_argument("--force", action="store_true", help="redo cases that already exist")

    s = prep.add_parser("synthetic")
    common(s)
    s.add_argument("--suite", default="full", choices=("test", "ci", "full", "scale", "dev"))

    an = prep.add_parser("anhir")
    common(an)
    an.add_argument("--data-root", required=True, help="folder holding dataset_medium.csv")
    an.add_argument("--proxy", default="lum", choices=("lum", "hema"))
    an.add_argument("--status", default="training", choices=("training", "evaluation", "all"),
                    help="evaluation pairs have hidden targets: see `anhir-submission`")
    an.add_argument("--tissue", nargs="*", default=None)
    an.add_argument("--case-id", nargs="*", type=int, default=None)
    an.add_argument("--limit", type=int, default=None)

    m = prep.add_parser("multiplex")
    common(m)
    m.add_argument("--manifest", required=True, help="CSV: case_id,reference,moving,...")
    m.add_argument("--diameter", type=float, default=20.0,
                   help="nuclear diameter in px, for the built-in segmenter and the tile halo")
    m.add_argument("--tile", type=int, default=2048)
    m.add_argument("--max-cells", type=int, default=0,
                   help="score a random sample of this many moving cells (0 = all)")

    ss = prep.add_parser("semisynth")
    common(ss)
    ss.add_argument("--image", required=True, help="OME-TIFF with a real nuclear channel")
    ss.add_argument("--channel", type=int, default=0)
    ss.add_argument("--mov-image", default=None,
                    help="another round of the same section, already registered to --image")
    ss.add_argument("--mov-channel", type=int, default=0)
    ss.add_argument("--name", default=None, help="prefix of the case ids (default: the file name)")
    ss.add_argument("--n", type=int, default=4096, help="window side, px")
    ss.add_argument("--windows", type=int, default=8, help="test windows (one more is dev)")
    ss.add_argument("--residual-um", type=float, default=2.0,
                    help="RMS displacement of the multiscale family, micrometres")
    ss.add_argument("--diameter", type=float, default=20.0, help="nuclear diameter, px")
    ss.add_argument("--noise", type=float, default=1.0,
                    help="added noise, in units of the image's background noise (same-image variant)")
    ss.add_argument("--pixel-size-um", type=float, default=None)

    hy = prep.add_parser("hyreco")
    common(hy)
    hy.add_argument("--data-root", default=".", help="folder holding HE/ and PHH3/")
    hy.add_argument("--ref-dir", default=None)
    hy.add_argument("--mov-dir", default=None)
    hy.add_argument("--pixel-size-um", type=float, default=None,
                    help="default: the TIFF's resolution tags")

    lst = sub.add_parser("list")
    lst.add_argument("--cases", required=True)
    lst.add_argument("--dataset", required=True, choices=DATASETS)
    lst.add_argument("--ids", action="store_true")

    r = sub.add_parser("run")
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--dataset", required=True, choices=DATASETS)
    r.add_argument("--method", required=True, choices=METHODS)
    r.add_argument("--case-id", default=None)
    r.add_argument("--index", type=int, default=None, help="index into `list` (SLURM_ARRAY_TASK_ID)")
    r.add_argument("--label", default=None,
                   help="name this configuration's results instead of the method's name")
    r.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="a method option; repeatable (see the method's module docstring)")
    r.add_argument("--workers", type=int, default=_cpus())
    r.add_argument("--work", default=None, help="scratch for the method's intermediates")
    r.add_argument("--keep-work", action="store_true")

    sc = sub.add_parser("score")
    sc.add_argument("--cases", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--dataset", nargs="*", default=None, choices=DATASETS)
    sc.add_argument("--workers", type=int, default=_cpus())
    sc.add_argument("--dev", action="store_true",
                    help="score the dev_* cases instead of leaving them out")

    cal = sub.add_parser("calibrate")
    cal.add_argument("--cases", required=True)
    cal.add_argument("--out", required=True, help="where the calibration runs were written")
    cal.add_argument("--lock", default=None, help="default: <out>/params.lock.json")
    cal.add_argument("--opts-for", default=None, metavar="METHOD",
                     help="print the recommended arm's options of one method and exit")
    cal.add_argument("--print-configs", action="store_true",
                     help="list `method label opts` of every candidate, for the submit script")

    sm = sub.add_parser("anhir-submission")
    sm.add_argument("--cases", required=True)
    sm.add_argument("--out", required=True)
    sm.add_argument("--data-root", required=True)
    sm.add_argument("--variant", required=True, help="e.g. stare, valis_micro, deeperhistreg")
    sm.add_argument("--dest", required=True)
    return ap


def _prepare(a):
    import importlib

    return importlib.import_module(f".datasets.{a.dataset}", __package__).prepare(a)


def _run(a):
    from .resources import set_thread_env

    set_thread_env(a.method, a.workers)  # before NumPy / torch load their thread pools
    from .cases import list_cases
    from .methods import parse_opts, run_case

    cases = list_cases(a.cases, a.dataset)
    if a.case_id is not None:
        picked = [c for c in cases if c.case_id == str(a.case_id)]
        if not picked:
            raise SystemExit(f"{a.dataset}/{a.case_id} is not prepared under {a.cases}")
    elif a.index is not None:
        if a.index >= len(cases):
            print(f"index {a.index}: only {len(cases)} {a.dataset} cases; nothing to do")
            return 0
        picked = [cases[a.index]]
    else:
        raise SystemExit("give --case-id or --index")
    case = picked[0]
    done = run_case(a.method, case, a.out, parse_opts(a.opt), a.label, a.workers, a.work,
                    a.keep_work)
    print(f"{a.dataset}/{case.case_id} [{a.label or a.method}]: "
          f"{', '.join(done) if done else 'FAILED'}")
    # A failure is recorded in run.json and scored at the initial pose; the exit code says
    # whether the method produced anything at all.
    return 0 if done else 1


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "prepare":
        return _prepare(a)
    if a.cmd == "list":
        from .cases import list_cases

        cases = list_cases(a.cases, a.dataset)
        print("\n".join(c.case_id for c in cases) if a.ids else len(cases))
        return 0
    if a.cmd == "run":
        return _run(a)
    if a.cmd == "calibrate":
        from . import calibrate

        if a.opts_for:
            lock = a.lock or f"{a.out}/params.lock.json"
            print(" ".join(f"{k}={v}" for k, v in calibrate.recommended(a.opts_for, lock)[0].items()))
            return 0
        return calibrate.main(a)
    if a.cmd == "anhir-submission":
        from .datasets import anhir

        anhir.export_submission(a.cases, a.out, a.data_root, a.variant, a.dest)
        return 0
    from .score import score

    score(a.cases, a.out, a.dataset, a.workers, dev=a.dev)
    return 0

"""``python -m benchmarks.regbench``: prepare cases, run a method, score everything.

``prepare <dataset>``  write the cases of ``synthetic`` / ``semisynth`` / ``multiplex`` (idempotent);
                       ``prepare hyreco --subset a|b`` writes the external landmark set
                       ``hyreco_a`` (tuning) or ``hyreco_b`` (report); ``prepare anhir
                       --source DIR`` writes ANHIR's training pairs (``anhir``, and its
                       tuning side again as ``anhir_tune``)
``list``               the number of prepared cases of a dataset (sizes a SLURM array), their
                       ids (``--ids``) or their patients (``--groups``)
``run``                register ONE case (``--case-id``), ONE patient (``--group``), or the
                       ``--index``-th of them: the index counts cases for a method that runs
                       per pair and patients for one that runs per patient
``score-case``         score one case and keep its record (``<out>/cache/scores``); ``score``
                       reads the records that are still valid and computes the rest
``score``              score every variant under ``--out``; writes ``<out>/tables``. Every
                       result name of the run list is expected (a missing one is a "not
                       run" row) and the primary rows lead; ``--only-run`` scores just
                       what is there
``runs``               the run list, one tab-separated line per run, for the launcher
                       (columns: ``benchmarks/regbench/variants.py``); ``--all`` adds the
                       paused runs; ``--dataset hyreco_b`` prints the runs of an
                       external brightfield set

An external landmark set (``hyreco_a``, ``hyreco_b``, ``anhir_tune``, ``anhir``) is never
scored with the cohort: ``score`` without ``--dataset`` reads the cohort's datasets only,
and ``score --dataset hyreco_b`` writes that set's own tables (``<out>/tables/hyreco_b/``,
``benchmarks/regbench/landmarks.py``; for ANHIR ``benchmarks/regbench/challenge.py``).
The two kinds cannot be named in one call.
``prep-select``        from the results of the preprocessing candidates on ANHIR's tuning
                       side (``anhir_tune``): the evidence table and the candidate the
                       ``stare_prep`` row is to run with; ``--freeze`` writes it into the
                       tracked ``prep_frozen.json``
``submissions``        the submission ids the selfcheck records of a results folder carry
``gate``               from one run's ``selfcheck.json``: the status that decides whether
                       the run may start, and the rows to leave out of the results
                       (``score --drop``). Exit code as ``selfcheck``
``realgate``           from the realcheck of a run's first unit: may the rest of the run
                       start. Exit 1 when the gating row is flagged
``selfcheck``          check that a method's points follow the image its own writer produced,
                       on a synthetic pair; exit 0 ``ok``, 1 ``inconsistent``, 2
                       ``not_checked``, 3 ``crashed``. Anything but 0 blocks the method

``run --realcheck`` does the same kind of check on the real case it has just registered
and exits 3 when it is flagged. ``run --retry-oom`` and ``selfcheck --retry-memory`` are
what a later rung of the memory ladder runs: they do nothing unless the attempt before
ran out of memory (``benchmarks/regbench/ladder.py``).

A method whose registration is several stage processes per round (``runs``, column
``stages``) is never run without a statement of who starts them, ``REGBENCH_STAGES``
(``benchmarks/regbench/methods/stare.py``): ``jobs`` from the cohort's batch script,
``local`` from ``run --stages local`` and from ``selfcheck``, which states it for its own
run. ``run`` without one touches nothing and ends with ``cause=not_submitted``.

Run from the repository root.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import os

from . import ALL_DATASETS, DATASETS, EXTERNAL_DATASETS, plugins
from . import METHODS as PUBLIC_METHODS
from .resources import n_cpus

# the public methods, and those of a private method plug-in when one is enabled
METHODS = PUBLIC_METHODS + plugins.method_names()


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
    m.add_argument(
        "--nuclei-dir",
        default=None,
        help="where `regbench nuclei` left each slide's bare nuclei (an existing "
        "label image that passed its checks, or a StarDist re-run)",
    )
    m.add_argument(
        "--nuclei-fallback",
        default="",
        choices=("", "shrink"),
        help="`shrink`: for a slide with no bare nuclei, use its outline with the "
        "10 px rim taken off (labelled shrunk_outlines, exact: false; too small "
        "where cells touch). Never used unless given",
    )

    nu = sub.add_parser(
        "nuclei",
        help="bare nuclei per native slide of a manifest: check the label images "
        "that exist, re-run StarDist where there is none",
    )
    nu.add_argument("--manifest", required=True, help="the `prepare multiplex` CSV")
    nu.add_argument("--out", required=True, help="one folder per slide is kept here")
    nu.add_argument("--index", type=int, default=None, help="one slide (array task)")
    nu.add_argument("--workers", type=int, default=1)
    nu.add_argument("--diameter", type=float, default=20.0, help="sizes the tile halo")
    nu.add_argument("--tile", type=int, default=2048)
    nu.add_argument(
        "--force",
        action="store_true",
        help="decide every chosen slide again, whatever is kept for it: a label "
        "image is checked again, a kept StarDist re-run is set aside and run again",
    )
    mode = nu.add_mutually_exclusive_group()
    mode.add_argument("--count", action="store_true", help="print the number of slides")
    mode.add_argument(
        "--list",
        action="store_true",
        help="print one line per slide: usable nuclei and their source, or why not",
    )
    mode.add_argument(
        "--command",
        action="store_true",
        help="print the segment.py call of each slide that still needs the re-run",
    )
    mode.add_argument(
        "--segment",
        action="store_true",
        help="run that call here (an environment with StarDist), then check it",
    )
    mode.add_argument(
        "--settings",
        action="store_true",
        help="print the StarDist model and settings of the re-run and where each is "
        "from (REGBENCH_STARDIST_*, a site config, the repository's defaults)",
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

    hy = prep.add_parser(
        "hyreco",
        help="HyReCo's re-stained H&E / PHH3 pairs (benchmarks/regbench/datasets/"
        "hyreco.py): subset a is the tuning set, subset b the report set",
    )
    common(hy)
    hy.add_argument(
        "--subset",
        required=True,
        choices=("a", "b"),
        help="a: the nine pairs settings are chosen on (hyreco_a). b: the 54 pairs "
        "that are reported (hyreco_b); takes its settings from --settings-from",
    )
    hy.add_argument(
        "--data-root", default=None, help="a folder holding HE/ and PHH3/ (subset A)"
    )
    hy.add_argument("--ref-dir", default=None, help="the subset's HE/ folder")
    hy.add_argument("--mov-dir", default=None, help="the subset's PHH3/ folder")
    hy.add_argument(
        "--proxy",
        default=None,
        choices=("luminance", "haematoxylin"),
        help="subset a only: the one-channel image a one-channel method is given "
        "(default luminance, inverted)",
    )
    hy.add_argument(
        "--reduce",
        default=None,
        type=int,
        choices=(1, 2),
        help="subset a only: block-average the proxy by this factor (default 1)",
    )
    hy.add_argument(
        "--settings-from",
        default=None,
        help="subset b only, required: the cases folder holding the prepared hyreco_a "
        "cases; their settings are the ones subset b is prepared with",
    )
    hy.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="a measured pixel size, instead of the file's (must be within 2 %% of "
        "the published 0.24)",
    )
    hy.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="prepare a subset of which some pairs are missing; every case is marked",
    )
    hy.add_argument(
        "--count",
        action="store_true",
        help="print the number of pairs of the subset and exit (sizes a prepare array)",
    )
    hy.add_argument(
        "--keep-shared-ids",
        action="store_true",
        help="subset b only: keep the four ids subset A also has (29, 108, 361, 464) "
        "in the report set. Only when the files show they are different slides",
    )

    an = prep.add_parser(
        "anhir",
        help="ANHIR's 230 training pairs (benchmarks/regbench/datasets/anhir.py): "
        "`anhir`, and its tuning side again as `anhir_tune`",
    )
    common(an)
    an.add_argument(
        "--source",
        default=None,
        help="a folder holding dataset_medium.csv, images/ and landmarks/ (the "
        "unpacked archives); it is only read",
    )
    an.add_argument("--table", default=None, help="the cover table, if elsewhere")
    an.add_argument("--images", default=None, help="the unpacked image archive")
    an.add_argument("--landmarks", default=None, help="the unpacked landmark archive")
    an.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="prepare the pairs whose images are there; every case is marked",
    )
    an.add_argument(
        "--count",
        action="store_true",
        help="print the number of training pairs and exit (sizes a prepare array)",
    )

    ps = sub.add_parser(
        "prep-select",
        help="choose the stare_prep preprocessing from the tuning side's results",
    )
    ps.add_argument("--cases", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--workers", type=int, default=n_cpus())
    ps.add_argument(
        "--freeze",
        action="store_true",
        help="write the choice and its evidence into benchmarks/regbench/"
        "prep_frozen.json (then commit it)",
    )
    ps.add_argument("--frozen-file", default=None, help="another file to write (tests)")

    lst = sub.add_parser("list")
    lst.add_argument("--cases", required=True)
    lst.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    what = lst.add_mutually_exclusive_group()
    what.add_argument("--ids", action="store_true", help="print the case ids")
    what.add_argument(
        "--groups", action="store_true", help="print the patients, one per line"
    )
    what.add_argument(
        "--sizes",
        action="store_true",
        help="print three numbers on one line: cases, patients, and the rounds of the "
        "largest patient (what the launcher sizes its arrays and time limits from)",
    )
    what.add_argument(
        "--current-spec",
        action="store_true",
        help="--dataset synthetic: print nothing, and exit non-zero when a prepared "
        "case is of a truth_spec.json other than the current one (the launcher asks "
        "before its first job: the scorer refuses such a result as a whole)",
    )

    r = sub.add_parser("run")
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--dataset", required=True, choices=ALL_DATASETS)
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
        "--stages",
        default=None,
        choices=("jobs", "local"),
        help="for a method whose registration is several stage processes per round: "
        "who starts them. local: this run itself (a run by hand). jobs: the "
        "scheduler did, with `stage`, and this run reads their records. Unset, "
        "REGBENCH_STAGES says; with neither, the run is refused",
    )
    r.add_argument(
        "--realcheck",
        action="store_true",
        help="after the method's writer, check that each mapping follows the image "
        "written for it (realcheck.json); exit 3 when one is flagged",
    )
    r.add_argument(
        "--retry-oom",
        action="store_true",
        help="a later rung of the memory ladder: run only when this unit's last "
        "run.json says it ran out of memory; otherwise leave it alone and exit 0",
    )
    r.add_argument(
        "--ladder",
        default=None,
        choices=("first", "climb", "again"),
        help="which task of the memory ladder this is (ladder.py). first: run the "
        "unit. climb (= --retry-oom): run only when its last attempt ended for "
        "memory. again: the same rung once more, only when the attempt before was "
        "lost to the scheduler (node failure, preemption, a cancelled job, a task "
        "that never started). The rung is REGBENCH_ATTEMPT",
    )
    r.add_argument(
        "--previous-state",
        default="",
        help="with --ladder: the scheduler's state of this unit's task just before "
        "(sacct). It decides only where run.json still says `killed`",
    )
    r.add_argument(
        "--accounting",
        default=None,
        metavar="FILE",
        help="with --ladder: `sacct -n -P -o JobID,State,ElapsedRaw,TotalCPU` of "
        "this unit's earlier tasks. A record that still says `killed` is settled by "
        "the state of the task that wrote it, and keeps that task's elapsed time",
    )
    r.add_argument(
        "--accounting-wanted",
        default=None,
        metavar="FILE",
        help="with --ladder: when a record still says `killed` and the accounting has "
        "no end state for the task that wrote it, write those tasks' ids to FILE "
        "(one per line), run nothing and exit 75: the caller waits for the "
        "accounting and calls again with --accounting-waited",
    )
    r.add_argument(
        "--accounting-waited",
        action="store_true",
        help="with --ladder: the accounting is as complete as it will be (it was "
        "waited for, or there is none to ask). A record that still says `killed` "
        "without an end state is then run again by the climb task at its rung",
    )
    r.add_argument(
        "--force",
        action="store_true",
        help="run a unit whose results are all ok already (also REGBENCH_FORCE=1). "
        "Without it such a unit is reused: nothing runs, nothing is overwritten",
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
        help="the --work of a `stare` selfcheck on the same cases (anchored methods)",
    )
    chk.add_argument(
        "--size",
        action="append",
        default=None,
        metavar="HxW[:HxW[:HxW]]",
        help="check on this case instead of the method's own: a pair "
        "(reference[:moving]), or one patient (reference:moving:moving, every round "
        "registered in one run by a per-patient method); repeatable",
    )
    chk.add_argument("--seed", type=int, default=0)
    chk.add_argument("--keep-work", action="store_true")
    chk.add_argument(
        "--in-process",
        action="store_true",
        help="do not start a process per pair (what that process itself is given)",
    )
    chk.add_argument(
        "--retry-memory",
        action="store_true",
        help="a later attempt with more memory: run only when the record already in "
        "--work ended for memory; otherwise leave it and exit with its status",
    )
    chk.add_argument(
        "--previous-state",
        default="",
        help="with --retry-memory: the scheduler's state of the earlier attempt",
    )

    st = sub.add_parser(
        "settle",
        help="after everything: give every result that still says `killed` the cause "
        "the scheduler's accounting names (out of memory, time limit, lost)",
    )
    st.add_argument("--out", required=True)
    st.add_argument(
        "--accounting",
        required=True,
        metavar="FILE",
        help="`sacct -n -P -o JobID,State,ElapsedRaw,TotalCPU` of the jobs",
    )

    sb = sub.add_parser(
        "submissions",
        help="the submission ids the selfcheck records under <out>/logs/selfcheck "
        "carry, one per line with the runs that carry it (`submit_cohort.sh "
        "--score-only` asks this when the folder has no logs/submission)",
    )
    sb.add_argument("--out", required=True)

    sc = sub.add_parser("score")
    sc.add_argument("--cases", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        choices=ALL_DATASETS,
        help="default: the cohort's datasets. An external landmark set (hyreco_a, "
        "hyreco_b, anhir_tune, anhir) is scored only when named, alone, into tables "
        "of its own",
    )
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
        "--submission",
        default=None,
        metavar="ID",
        help="score only what this submission ran or reused (run.json: submissions; "
        "the launcher's REGBENCH_SUBMISSION). A result another submission left, of a "
        "run this one blocked for instance, is `not run`",
    )
    sc.add_argument(
        "--gates",
        default=None,
        metavar="DIR",
        help="the launcher's gate records (<out>/logs/gates): every run a gate blocked "
        "is named in the summary and in launch.csv",
    )
    sc.add_argument(
        "--no-tissue",
        action="store_true",
        help="leave out the tissue-level block of the written images (tissue*.csv)",
    )
    sc.add_argument(
        "--manifest",
        default=os.environ.get("REGBENCH_MANIFEST") or None,
        metavar="CSV",
        help="the `prepare multiplex` manifest the cohort's cases were made from "
        "(default: REGBENCH_MANIFEST): its number of rounds is the first column of "
        "attrition.csv. Nothing else reads it",
    )
    sc.add_argument(
        "--fresh",
        action="store_true",
        help="read no kept record (<out>/cache/scores): compute every case and every "
        "pair record again and replace them. Never needed for correctness: a record "
        "is reused only when everything it rests on is unchanged",
    )

    so = sub.add_parser(
        "score-case",
        help="score ONE case of a cohort dataset and keep its record under "
        "<out>/cache/scores; no table is written. `score` reads the record instead "
        "of computing the case (one array task per case on a cluster)",
    )
    so.add_argument("--cases", required=True)
    so.add_argument("--out", required=True)
    so.add_argument("--dataset", required=True, choices=DATASETS)
    which = so.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--index",
        type=int,
        help="its place among all prepared cases of the dataset (SLURM_ARRAY_TASK_ID; "
        "the order of `list --ids`)",
    )
    which.add_argument("--case-id")
    so.add_argument(
        "--workers",
        type=int,
        default=n_cpus(),
        help="processes: the case's pair records are computed side by side",
    )
    so.add_argument("--dev", action="store_true", help="as `score --dev`")
    so.add_argument("--only-run", action="store_true", help="as `score --only-run`")
    so.add_argument(
        "--drop",
        action="append",
        default=[],
        metavar="VARIANT",
        help="as `score --drop`; the record is reused by a `score` given the same ones",
    )
    so.add_argument(
        "--submission",
        default=None,
        metavar="ID",
        help="as `score --submission`; the record is reused by a `score` of that one",
    )
    so.add_argument("--fresh", action="store_true", help="as `score --fresh`")

    ts = sub.add_parser(
        "tissue",
        help="compute the tissue-level pieces of every written image (needs OpenCV); "
        "`score` then reads them",
    )
    ts.add_argument("--cases", required=True)
    ts.add_argument("--out", required=True)
    ts.add_argument("--dataset", nargs="*", default=None, choices=ALL_DATASETS)
    ts.add_argument(
        "--index",
        type=int,
        default=None,
        help="only this case of the ONE dataset named: its place among all prepared "
        "cases (SLURM_ARRAY_TASK_ID, as `score-case --index`)",
    )
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
    gt.add_argument(
        "--for",
        dest="run",
        default=None,
        help="the run that is gated, when it stands behind another run's selfcheck "
        "(default: --name)",
    )
    gt.add_argument(
        "--record",
        default=None,
        help="also write the verdict here, one tab-separated line: run, status, rows "
        "to drop (or -), blocked 0|1, the reference comparison (or -)",
    )
    gt.add_argument(
        "--allow-unchecked",
        action="store_true",
        help="let the run start when its selfcheck ran and could not decide "
        "(not_checked); every other verdict still blocks (REGBENCH_ALLOW_UNCHECKED)",
    )
    gt.add_argument(
        "--advisory",
        action="store_true",
        help="the selfcheck is advice, not a gate: the run starts whatever the verdict "
        "(inconsistent, crashed, not_checked, no record), and the verdict is recorded "
        "as it is (REGBENCH_SELFCHECK_ADVISORY=1)",
    )
    gt.add_argument(
        "--allow-inconsistent",
        action="store_true",
        help="let the run start although its selfcheck MEASURED a disagreement "
        "(inconsistent): the owner's override, per run (REGBENCH_ALLOW_INCONSISTENT). "
        "The verdict is recorded as it is; a crash or a missing record still blocks",
    )

    rg = sub.add_parser(
        "realgate",
        help="from the realcheck of a run's first unit: may the rest of the run start",
    )
    rg.add_argument("--cases", required=True)
    rg.add_argument("--out", required=True)
    rg.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    rg.add_argument("--name", required=True, help="a run name (`runs`, column 1)")
    rg.add_argument("--method", required=True, choices=METHODS)
    rg.add_argument("--label", default=None)
    rg.add_argument(
        "--variant", required=True, help="the row whose realcheck gates the run"
    )
    rg.add_argument("--index", type=int, default=0, help="the unit that went ahead")
    rg.add_argument("--record", default=None, help="as `gate --record`")
    rg.add_argument("--allow-unchecked", action="store_true")

    sg = sub.add_parser(
        "stage",
        help="one stage job of one round of a run whose registration is several "
        "scheduler jobs (`runs`, column `stages`): prep, coarse, tile or solve",
    )
    sg.add_argument("--cases", required=True)
    sg.add_argument("--out", required=True)
    sg.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    sg.add_argument("--method", required=True, choices=METHODS)
    sg.add_argument("--label", default=None)
    sg.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE")
    sg.add_argument(
        "--stage", required=True, choices=("prep", "coarse", "tile", "solve")
    )
    sg.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="1 runs the job; 2 to 4 run it only when the attempt before ended the "
        "way the pipeline retries (regbench.stages)",
    )
    where = sg.add_mutually_exclusive_group(required=True)
    where.add_argument("--index", type=int, default=None, help="the case (round)")
    where.add_argument(
        "--flat",
        type=int,
        default=None,
        help="with --stage tile: one tile, counted over every case's tiles laid end "
        "to end as --layout lists them",
    )
    sg.add_argument("--layout", default=None, help="the file `stages --layout` wrote")
    sg.add_argument("--force", action="store_true", help="also REGBENCH_FORCE=1")

    sl = sub.add_parser(
        "stages",
        help="size the tile arrays of a run on a dataset: writes the layout file "
        "(one line per case: index, case id, tiles, offset) and prints `<cases> "
        "<tiles>`",
    )
    sl.add_argument("--cases", required=True)
    sl.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    sl.add_argument("--method", required=True, choices=METHODS)
    sl.add_argument("--label", default=None)
    sl.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE")
    sl.add_argument("--layout", required=True)
    sl.add_argument(
        "--due",
        type=int,
        default=None,
        metavar="ATTEMPT",
        help="write no layout: READ it, and print what the tile submitter puts into "
        "the queue at that attempt, one index per line. Attempt 1: every tile that "
        "is due, counted over every case's tiles laid end to end. Attempts 2 to 4: "
        "every case (round) that has a tile the pipeline would retry",
    )
    sl.add_argument("--out", default=None, help="with --due: the results folder")

    pf = sub.add_parser(
        "preflight",
        help="inside an image, before anything is submitted: do the modules a run "
        "needs import, is the package the version its adapter enforces, does numpy "
        "resolve inside the image, are the folders writable (regbench.preflight)",
    )
    what = pf.add_mutually_exclusive_group(required=True)
    what.add_argument("--method", choices=METHODS)
    what.add_argument("--scorer", action="store_true", help="the scorer's image")
    pf.add_argument("--image", default="", help="the image's name, for the lines")
    pf.add_argument("--writable", action="append", default=[], metavar="DIR")
    pf.add_argument(
        "--cases",
        default=None,
        help="with --dataset: the slides of one prepared case of each dataset must be "
        "readable from inside the image",
    )
    pf.add_argument("--dataset", action="append", default=[], choices=ALL_DATASETS)

    rn = sub.add_parser("runs")
    rn.add_argument(
        "--header", action="store_true", help="print the column names first"
    )
    rn.add_argument(
        "--scorer-image",
        action="store_true",
        help="print only the image `prepare`, `gate`, `tissue` and `score` run in",
    )
    rn.add_argument(
        "--all",
        action="store_true",
        help="also print the paused runs (0 in the `active` column)",
    )
    rn.add_argument(
        "--dataset",
        default=None,
        choices=ALL_DATASETS,
        help="the runs of this dataset: for an external brightfield set, only the "
        "runs it takes; for anhir_tune, the preprocessing candidates",
    )
    rn.add_argument(
        "--order",
        default=None,
        metavar="NAMES",
        help="print the runs in the order a launcher is to SUBMIT them: these family "
        "or run names first (space-separated), the rest after, and a run that reads "
        "another's results never before it; `default` is the cohort's default order. Nothing else reads this order",
    )
    rn.add_argument(
        "--order-notes",
        action="store_true",
        help="with --order: print instead one line per run that was moved behind the "
        "run it reads",
    )
    rn.add_argument(
        "--stage-allocation",
        action="store_true",
        help="print only, for the runs that are several scheduler jobs per round "
        "(column `stages`), one line per stage and attempt: stage, attempt, CPUs, "
        "memory in GB, time limit (the pipeline's allocation of that process)",
    )
    rn.add_argument(
        "--prep-status",
        action="store_true",
        help="print only `<status>\t<candidate>\t<run>` of the frozen preprocessing "
        "(status: provisional or selected; run: the row that uses it)",
    )
    rn.add_argument(
        "--excluded",
        action="store_true",
        help="with --dataset: print instead the runs left out, each with its reason",
    )
    return ap


def _prepare(a):
    import importlib

    return importlib.import_module(f".datasets.{a.dataset}", __package__).prepare(a)


@contextlib.contextmanager
def _stages_stated(method, stated):
    """For the time of one command, say who starts the stage processes of a method
    whose registration is several per round (its ``STAGES``): ``stated`` goes into the
    variable the method reads (its ``JOBS_ENV``) and what was there comes back after.
    None states nothing. A method that is one job per unit has nothing to state."""
    from .methods import load

    mod = load(method)
    name = getattr(mod, "JOBS_ENV", None) if getattr(mod, "STAGES", None) else None
    if stated is None or name is None:
        yield
        return
    before = os.environ.get(name)
    os.environ[name] = stated
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = before


def _stages_refused(a, picked):
    """True when the unit's run does not say who starts its stage processes (the
    method's ``launcher``): said on the task's end line as the launcher's failure,
    before a record is read or written."""
    from . import failure_log, resources, stages
    from .methods import load

    mod = load(a.method)
    if not getattr(mod, "STAGES", None):
        if a.stages:
            raise SystemExit(f"{a.method} is one job per unit: --stages says nothing")
        return False
    try:
        mod.launcher()
    except stages.StageFailed as err:
        print(f"{a.dataset} [{a.label or a.method}]: {err}")
        failure_log.emit_end(
            "run",
            failure_log.outcome(False, err.regbench_cause),
            err.regbench_cause,
            run=a.label or a.method,
            dataset=a.dataset,
            unit="+".join(c.case_id for c in picked),
            rung=resources.host_info()["attempt"],
        )
        return True
    return False


def _run(a):
    from .cases import groups, list_cases
    from .methods import parse_opts, run_case, run_group, scope_of

    # an option the method does not read: refused before anything is looked at
    opts = parse_opts(a.opt, a.method)
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
    if _stages_refused(a, picked):
        return 1
    task = a.ladder or ("climb" if a.retry_oom else None)
    if task:
        from pathlib import Path

        from . import ladder, resources

        host = resources.host_info()
        heads = ladder.unit_heads(a.out, a.method, a.label, picked)
        ids = "+".join(c.case_id for c in picked)
        accounting = ladder.read_accounting(a.accounting)
        exits = ladder.read_exits(a.out)
        if a.accounting_wanted and not a.accounting_waited:
            pending = ladder.unsettled(
                list(heads), a.previous_state, accounting, host["submission"], exits
            )
            wanted = sorted({launch for _, launch in pending if launch})
            if wanted:
                Path(a.accounting_wanted).write_text("".join(f"{w}\n" for w in wanted))
                print(
                    f"{a.dataset}/{ids} [{a.label or a.method}] {task}, rung "
                    f"{host['attempt']}: {len(pending)} result(s) still say killed "
                    f"and the scheduler's accounting has no end state for "
                    f"{', '.join(wanted)}; waiting for it, nothing decided yet"
                )
                return ladder.EXIT_ACCOUNTING_PENDING
        go, why, lost = ladder.unit_decision(
            list(heads),
            task,
            host["attempt"],
            a.previous_state,
            accounting,
            host["submission"],
            exits,
            a.accounting_waited,
            host["mem_requested_gb"],
        )
        print(
            f"{a.dataset}/{ids} [{a.label or a.method}] {task}, rung "
            f"{host['attempt']}: {why}; " + ("running" if go else "nothing to do")
        )
        if lost:
            n = ladder.mark_lost(heads, host["attempt"] - 1, host["submission"])
            print(f"{a.dataset}/{ids}: {n} result(s) recorded as infrastructure")
        if not go:
            _unit_end(a, picked)
            return 0
    rest = (opts, a.label, a.work, a.keep_work)
    more = {"realcheck": a.realcheck, "resume": not a.force}
    if per_patient or a.group is not None:
        done = run_group(a.method, picked, a.out, *rest, **more)
    else:
        case = picked[0]
        got = run_case(a.method, case, a.out, *rest, **more)
        done = {case.case_id: got}
    for case_id, variants in done.items():
        print(
            f"{a.dataset}/{case_id} [{a.label or a.method}]: "
            f"{', '.join(variants) if variants else 'FAILED'}"
        )
    _unit_end(a, picked)
    # A failure is recorded in run.json and scored at the unregistered pose; the exit code
    # says whether the method produced anything at all.
    if not any(done.values()):
        return 1
    if a.realcheck and _realcheck_flagged(a, picked, done):
        return 3
    return 0


def _stage_opts(a):
    from .methods import parse_opts

    opts = parse_opts(a.opt, a.method)
    opts["label"] = a.label or ""
    return opts


def _stages(a):
    """`stages`: the layout of a run's tile arrays on a dataset."""
    from . import stages
    from .cases import list_cases
    from .methods import load

    mod, opts = load(a.method), _stage_opts(a)
    cases = list_cases(a.cases, a.dataset)
    if a.due is not None:
        import os

        if not a.out:
            raise SystemExit("--due needs --out (the stage records are under it)")
        layout = stages.read_layout(a.layout)
        if [c for c, _ in layout] != [c.case_id for c in cases]:
            raise SystemExit(
                f"{a.layout} does not list the {a.dataset} cases under {a.cases}: the "
                "cases changed after the arrays were sized. Submit again"
            )
        force = os.environ.get("REGBENCH_FORCE") == "1"
        offset = 0
        for index, (case, (_, n)) in enumerate(zip(cases, layout)):
            due = mod.tiles_due(
                case, mod.stage_dir_of(a.out, a.label, case), opts, a.due, force
            )
            if a.due == 1:
                for row in due:
                    if row < n:
                        print(offset + row)
            elif due:
                print(index)
            offset += n
        return 0
    rows = [(c.case_id, mod.tile_count(c, opts)) for c in cases]
    print(len(rows), stages.write_layout(a.layout, rows))
    return 0


def _stage(a):
    """`stage`: one stage job. The exit code is 0 whatever became of the stage: its
    record says, and the round's last job reads it."""
    import os

    from . import failure_log, resources, stages
    from .cases import list_cases
    from .methods import check_inputs, load

    mod, opts = load(a.method), _stage_opts(a)
    cases = list_cases(a.cases, a.dataset)
    index, row = a.index, None
    if a.flat is not None:
        if a.stage != "tile" or not a.layout:
            raise SystemExit("--flat goes with --stage tile and --layout")
        layout = stages.read_layout(a.layout)
        if [c for c, _ in layout] != [c.case_id for c in cases]:
            raise SystemExit(
                f"{a.layout} does not list the {a.dataset} cases under {a.cases}: the "
                "cases changed after the arrays were sized. Submit again"
            )
        at = stages.locate([n for _, n in layout], a.flat)
        if at is None:
            print(f"tile {a.flat}: past the last tile of {a.dataset}; nothing to do")
            return 0
        index, row = at
    if index >= len(cases):
        print(f"index {index}: only {len(cases)} {a.dataset} cases; nothing to do")
        return 0
    case = cases[index]
    # a slide that cannot be read from here is the launch's failure, said before the
    # stage starts (the round's last job records it as `infrastructure`)
    try:
        check_inputs([case])
    except resources.InputUnreadable as exc:
        raise SystemExit(f"stage {a.stage}: {exc}") from exc
    force = a.force or os.environ.get("REGBENCH_FORCE") == "1"
    stage_dir = mod.stage_dir_of(a.out, a.label, case)
    done = mod.run_stage(case, stage_dir, opts, a.stage, row, a.attempt, "jobs", force)
    if a.flat is not None and done and done[0][4] is not None:
        planned = len(mod.Round(case, stage_dir, opts).plan())
        if planned != layout[index][1]:
            # the launcher's layout and COARSE's plan disagree: said on the task's
            # end line like every other end, as the launcher's failure
            print(
                f"{a.dataset}/{case.case_id}: COARSE planned {planned} tiles, the "
                f"arrays were sized for {layout[index][1]}"
            )
            failure_log.emit_end(
                "stage",
                "failed",
                stages.NOT_SUBMITTED,
                run=a.label or a.method,
                dataset=a.dataset,
                unit=case.case_id,
                stage=a.stage,
                attempt=a.attempt,
                planned=planned,
                sized_for=layout[index][1],
            )
            return 1
    state, reason = "ok", ""
    for stage, r, ran, why, rec in done:
        at = f" row {r}" if r is not None else ""
        print(
            f"{a.dataset}/{case.case_id} [{a.label or a.method}] {stage}{at}, "
            f"attempt {a.attempt}: {why}; "
            + ("ran" if ran else "nothing to do")
            + (f", exit {rec.get('exit')}" if ran and rec else "")
        )
        if rec is None:
            state, reason = "waiting", ""
        elif not rec.get("ok") and state == "ok":
            state, reason = "failed", str(rec.get("cause") or "exception")
    failure_log.emit_end(
        "stage",
        state,
        reason,
        run=a.label or a.method,
        dataset=a.dataset,
        unit=case.case_id,
        stage=a.stage,
        attempt=a.attempt,
    )
    return 0


def _unit_end(a, picked):
    """The task's end line (``failure_log.end_line``): what the unit's records say now,
    in the failure log's words."""
    import json

    from . import failure_log, ladder

    records = []
    for path in ladder.unit_paths(a.out, a.method, a.label, picked):
        try:
            records.append(json.loads(open(path).read()))
        except (OSError, ValueError):
            records.append({})
    status, cause = failure_log.unit_end(records)
    failure_log.emit_end(
        "run",
        status,
        cause,
        run=a.label or a.method,
        dataset=a.dataset,
        unit="+".join(c.case_id for c in picked),
        rung=next((r.get("attempt") for r in records if r), "-"),
    )


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

    if a.retry_memory:
        import json
        from pathlib import Path

        from . import ladder, resources

        path = Path(a.work) / "selfcheck.json"
        submission = resources.host_info()["submission"]
        go, why = ladder.selfcheck_retry(path, a.previous_state, submission)
        print(f"selfcheck [{a.label or a.method}]: {why}")
        if not go:
            try:
                status = json.loads(path.read_text()).get("status")
            except (OSError, ValueError):
                status = selfcheck.NOT_CHECKED
            _selfcheck_end(a, status if status in selfcheck.EXIT_CODE else "crashed")
            return selfcheck.EXIT_CODE.get(
                status, selfcheck.EXIT_CODE[selfcheck.CRASHED]
            )
    record = selfcheck.run(
        a.method,
        a.work,
        parse_opts(a.opt, a.method),
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
    _selfcheck_end(a, record["status"])
    return selfcheck.EXIT_CODE[record["status"]]


def _selfcheck_end(a, status):
    from . import failure_log

    failure_log.emit_end(
        "selfcheck",
        status,
        failure_log.SELFCHECK_CAUSE.get(status, ""),
        run=a.label or a.method,
    )


def _list(a):
    from .cases import groups, list_cases

    if a.current_spec:
        from .datasets import synthetic

        if a.dataset != "synthetic":
            raise SystemExit("--current-spec is the synthetic dataset's check")
        synthetic.refuse_other_spec(a.cases, nothing="Nothing was submitted.")
        return 0
    cases = list_cases(a.cases, a.dataset)
    if a.ids:
        print("\n".join(c.case_id for c in cases))
    elif a.groups:
        print("\n".join(groups(cases)))
    elif a.sizes:
        by_group = groups(cases)
        print(len(cases), len(by_group), max(map(len, by_group.values()), default=0))
    else:
        print(len(cases))
    return 0


def _runs(a):
    from . import variants

    if a.scorer_image:
        print(variants.SCORER_IMAGE)
        return 0
    if a.stage_allocation:
        from . import stages

        for stage in stages.ALLOCATION:
            for attempt in range(1, stages.ATTEMPTS + 1):
                got = stages.allocation(stage, attempt)
                print(
                    f"{stage}\t{attempt}\t{got['cpus']}\t{got['mem_gb']}\t{got['time']}"
                )
        return 0
    if a.prep_status:
        from . import prep

        record = prep.load_frozen()
        print(f"{record['status']}\t{record['selected']}\t{prep.ROW}")
        return 0
    if a.excluded:
        for name, why in variants.excluded_for(a.dataset):
            print(f"{name}\t{why}")
        return 0
    if a.header:
        print("\t".join(variants.RUNS_COLUMNS))
    order = None if a.order is None else a.order.split()
    if order == ["default"]:
        order = list(variants.DEFAULT_ORDER)
    if a.order_notes:
        listed = [variants.run_of(r[0]) for r in variants.runs_table(a.all, a.dataset)]
        for note in variants.submission_order(order, listed)[1]:
            print(note)
        return 0
    for row in variants.runs_table(paused=a.all, dataset=a.dataset, order=order):
        print("\t".join(row))
    return 0


def split_datasets(names):
    """``(cohort datasets | None, external dataset | None)`` of a ``--dataset`` list.
    The two kinds are never scored together: an external landmark set has its own
    tables, and the cohort's default never includes one."""
    names = list(names or [])
    external = [d for d in names if d in EXTERNAL_DATASETS]
    if not external:
        return (names or None), None
    if len(names) != 1:
        raise SystemExit(
            f"--dataset {' '.join(names)}: an external landmark set "
            f"({', '.join(EXTERNAL_DATASETS)}) is scored alone, into tables of its "
            "own; name one of them and nothing else"
        )
    return None, external[0]


def _write_gate(path, run, status, dropped, blocked, reference):
    """One line a shell can read: run, status, rows to drop, blocked, reference, and
    the log of the gate job that wrote it (for the failure log)."""
    from pathlib import Path

    from . import variants

    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "\t".join(
            [
                run,
                status,
                ",".join(dropped) or variants.NONE,
                "1" if blocked else "0",
                reference or variants.NONE,
                os.environ.get("REGBENCH_LOG") or variants.NONE,
            ]
        )
        + "\n"
    )


def _reference_status(record, names):
    """The worst status of the second comparison a method that redraws the reference
    has (ASHLAR): ``pairs[].variants.<row>.reference.status``. It is already folded
    into each row's own status; it is recorded apart so the summary can name it."""
    from . import selfcheck

    found = [
        ((pair.get("variants") or {}).get(v) or {}).get("reference", {}).get("status")
        for pair in record.get("pairs") or []
        for v in names
    ]
    found = [f for f in found if f]
    return selfcheck.worst(found) if found else ""


def _gate(a):
    """One line, tab-separated: the status that decides whether the run may start (the
    worst over its gating rows), and the rows to leave out of the results (``-`` for
    none). The exit code is that status's, as ``selfcheck`` returns it: 0 only for ok.

    A selfcheck that left no record (its job was cancelled, or killed before it could
    write) is ``not_checked``: the run is blocked, and the record says why.

    ``--allow-unchecked`` (a run named in ``REGBENCH_ALLOW_UNCHECKED``) lets a run start
    whose selfcheck RAN and could not decide (``not_checked``), and nothing else: not a
    measured disagreement (``inconsistent``), not a crash, and not a record that is
    missing, unreadable or another submission's."""
    import json
    from pathlib import Path

    from . import selfcheck, variants

    run = a.run or a.name
    try:
        variants.run_of(a.name)
        variants.run_of(run)
    except KeyError as exc:
        raise SystemExit(f"gate: {exc.args[0]}; see `runs --all`") from None
    from . import ladder, resources

    try:
        record = json.loads(Path(a.selfcheck).read_text())
    except (OSError, ValueError) as exc:
        print(f"gate: no selfcheck record at {a.selfcheck} ({type(exc).__name__})")
        record = {}
    submission = resources.host_info()["submission"]
    if record and not ladder.mine(record, submission):
        # an older launch's verdict says nothing about this one's images and code
        print(
            f"gate: the selfcheck record at {a.selfcheck} is of another submission "
            f"({record.get('submission') or 'none recorded'}), not of {submission}: "
            "refused, the run is not checked"
        )
        record = {}
    status = variants.gate_status(a.name, record)
    # rows are dropped from the run whose selfcheck this is, and from no other
    dropped = variants.dropped_variants(a.name, record) if run == a.name else []
    reference = _reference_status(record, variants.gate_variants(a.name))
    allowed = a.allow_unchecked and status == selfcheck.NOT_CHECKED and bool(record)
    # the owner's override of a MEASURED disagreement, named per run: the verdict stays
    # `inconsistent` in the record, in the end line and in the summary
    overridden = (
        getattr(a, "allow_inconsistent", False)
        and status == selfcheck.INCONSISTENT
        and bool(record)
    )
    allowed = allowed or overridden
    advisory = getattr(a, "advisory", False) and status in selfcheck.BLOCKING
    allowed = allowed or advisory
    blocked = status in selfcheck.BLOCKING and not allowed
    print(f"{status}\t{','.join(dropped) or variants.NONE}")
    _write_gate(a.record, run, status, dropped, blocked, reference)
    from . import failure_log

    failure_log.emit_end(
        "gate",
        "blocked" if blocked else "may_start",
        "gate_blocked" if blocked else "",
        run=run,
        gate="selfcheck",
        verdict=status,
        dropped=",".join(dropped) or "-",
    )
    if advisory:
        print(
            f"gate: the selfcheck is ADVISORY (REGBENCH_SELFCHECK_ADVISORY=1): {run} "
            f"starts although its verdict is {status}. The record keeps the verdict"
        )
        return 0
    if overridden:
        print(
            f"gate: {run} is named in REGBENCH_ALLOW_INCONSISTENT: it starts ALTHOUGH "
            f"its selfcheck measured a disagreement ({status}). Its point mapping does "
            "not follow the image its writer draws on the selfcheck's pairs"
        )
        return 0
    if allowed:
        print(f"gate: {run} is allowed to start unchecked ({status})")
        return 0
    if blocked and a.allow_unchecked:
        print(
            f"gate: {run} is named in REGBENCH_ALLOW_UNCHECKED, which lets through a "
            f"selfcheck that ran and could not decide (not_checked) and nothing else: "
            + ("there is no record of this submission" if not record else status)
            + "; blocked"
        )
    return selfcheck.EXIT_CODE[status]


REAL_OK, REAL_FLAGGED, REAL_NOT_CHECKED = "ok", "flagged", "not_checked"


def _realgate(a):
    """May the rest of a run start, from the realcheck of the unit that went ahead.

    Reads ``realcheck.json`` of the gating row for every case of that unit. ``flagged``
    (exit 1) when one is flagged; ``not_checked`` when there is none (the unit failed,
    or wrote no image), which does NOT block: only a measured disagreement does, and
    ``--allow-unchecked`` does not let that one through."""
    import json

    from .cases import groups, list_cases, result_dir
    from .methods import scope_of

    cases = list_cases(a.cases, a.dataset)
    by_group = groups(cases)
    units = (
        list(by_group.values())
        if scope_of(a.method) == "group"
        else [[c] for c in cases]
    )
    from . import ladder, resources

    submission = resources.host_info()["submission"]
    picked = units[a.index] if a.index < len(units) else []
    found, flagged = 0, []
    for case in picked:
        d = result_dir(a.out, a.variant, case)
        try:
            run = json.loads((d / "run.json").read_text())
            check = json.loads((d / "realcheck.json").read_text())
        except (OSError, ValueError):
            continue
        if not run.get("ok") or not ladder.mine(run, submission):
            continue  # a realcheck an earlier launch left answers for nothing here
        found += 1
        if check.get("flag"):
            flagged.append(case.case_id)
    status = REAL_FLAGGED if flagged else REAL_OK if found else REAL_NOT_CHECKED
    blocked = bool(flagged)
    print(
        f"{status}\t{a.variant}\t{found} of {len(picked)} realcheck(s)"
        + (f"\tflagged: {','.join(flagged)}" if flagged else "")
    )
    _write_gate(a.record, a.name, f"realcheck_{status}", [], blocked, "")
    from . import failure_log

    failure_log.emit_end(
        "gate",
        "blocked" if blocked else "may_start",
        "gate_blocked" if blocked else ("realcheck_flagged" if flagged else ""),
        run=a.name,
        gate="realcheck",
        verdict=status,
    )
    return 1 if blocked else 0


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


def _score_one(a):
    """One case's record (``score.score_case``), made exactly as ``score`` would."""
    from . import failure_log, score, variants

    if a.submission:
        os.environ[score.SUBMISSION_ENV] = a.submission
    status = score.score_case(
        a.cases,
        a.out,
        a.dataset,
        index=a.index,
        case_id=a.case_id,
        workers=a.workers,
        dev=a.dev,
        expected=None if a.only_run else variants.expected_variants(),
        dropped=_dropped(a.drop),
        row_checks=variants.row_checks(
            os.path.join(a.out, "logs", "selfcheck"), score.of_this_submission
        ),
        fresh=a.fresh,
    )
    unit = a.case_id if a.index is None else f"index_{a.index}"
    failure_log.emit_end(
        "score_case", "ok", dataset=a.dataset, unit=unit, record=status
    )
    return 0


def _tissue(a):
    """The pieces of every written image, left beside each reduced plane (tissue.json)."""
    from . import score, tissue, variants
    from .cases import list_cases

    if split_datasets(a.dataset)[1]:
        raise SystemExit(
            f"tissue: not computed for {a.dataset[0]}: {tissue.NOT_BRIGHTFIELD}"
        )
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
        if a.index is not None:
            everything = list_cases(a.cases, ds)
            if len(a.dataset or []) != 1 or not 0 <= a.index < len(everything):
                raise SystemExit(
                    f"tissue --index {a.index}: name ONE dataset, and an index among "
                    f"its {len(everything)} prepared case(s)"
                )
            one = everything[a.index].case_id
            cases = [c for c in cases if c.case_id == one]
            if not cases:
                print(f"tissue: {ds}/{one}: a dev_* case, not computed here")
                return 0
        jobs += [(a.out, c, found) for c in cases]
    if not jobs:
        raise SystemExit(f"no prepared cases under {a.cases}")
    got = tissue.score_all(jobs, a.workers)
    print(
        f"tissue: {len(got['tissue'])} row(s), {len(got['tissue_missing'])} without "
        f"one, {len(got['tissue_dose'])} dose row(s); `score` writes the tables"
    )
    return 0


# The commands a shell script reads the standard output of (a table, a count, a list of
# array indices). While one of them runs, `sys.stdout` IS the standard error: a library
# that logs or warns on "stdout" (STARE installs a root handler on it at its first
# logger) writes there, and only this module's own `print` reaches the real one.
ANSWERS = ("list", "runs", "stages", "submissions")
_ANSWER = None


def print(*args, **kwargs):  # noqa: A001  (this module's answers, see ANSWERS)
    if _ANSWER is not None and "file" not in kwargs:
        kwargs["file"] = _ANSWER
    builtins.print(*args, **kwargs)


def main(argv=None):
    """Run one command. For a command in ``ANSWERS`` everything but its answer goes to
    the standard error."""
    global _ANSWER
    a = build_parser().parse_args(argv)
    if a.cmd not in ANSWERS:
        return _main(a)
    import sys

    _ANSWER, sys.stdout = sys.stdout, sys.stderr
    try:
        return _main(a)
    finally:
        sys.stdout.flush()
        sys.stdout, _ANSWER = _ANSWER, None


def _main(a):
    if a.cmd == "nuclei":
        from . import nuclei

        return nuclei.main(a)
    if a.cmd == "preflight":
        from . import preflight

        return preflight.main(a)
    if a.cmd == "prepare":
        return _prepare(a)
    if a.cmd == "prep-select":
        from . import challenge

        challenge.select_prep(
            a.cases, a.out, freeze=a.freeze, path=a.frozen_file, workers=a.workers
        )
        return 0
    if a.cmd == "list":
        return _list(a)
    if a.cmd == "run":
        with _stages_stated(a.method, a.stages):
            return _run(a)
    if a.cmd == "selfcheck":
        # the selfcheck has no scheduler jobs of its own: it starts its stages itself
        with _stages_stated(a.method, "local"):
            return _selfcheck(a)
    if a.cmd == "runs":
        return _runs(a)
    if a.cmd == "stage":
        return _stage(a)
    if a.cmd == "stages":
        return _stages(a)
    if a.cmd == "gate":
        return _gate(a)
    if a.cmd == "realgate":
        return _realgate(a)
    if a.cmd == "tissue":
        return _tissue(a)
    if a.cmd == "score-case":
        return _score_one(a)
    if a.cmd == "submissions":
        import sys

        from . import ladder

        found, unreadable = ladder.selfcheck_submissions(a.out)
        for path in unreadable:
            print(f"submissions: not a readable record: {path}", file=sys.stderr)
        for sub, runs in found.items():
            if sub:  # a record that names none is evidence of no submission
                print(f"{sub}\t{','.join(runs)}")
        return 0
    if a.cmd == "settle":
        from . import ladder, resources

        changed = ladder.settle_all(
            a.out,
            ladder.read_accounting(a.accounting),
            resources.host_info()["submission"],
        )
        print(
            "settle: "
            + (", ".join(f"{n} {c}" for c, n in sorted(changed.items())) or "nothing")
            + " (results that said `killed`, by the scheduler's accounting)"
        )
        return 0
    from . import score, variants

    if getattr(a, "submission", None):
        os.environ[score.SUBMISSION_ENV] = a.submission

    cohort, external = split_datasets(a.dataset)
    if external:
        from . import ANHIR_DATASETS, challenge, landmarks

        common = dict(
            expected=None if a.only_run else variants.expected_variants(external),
            primary=variants.primary_for(external),
            dropped=_dropped(a.drop),
        )
        # the gates' verdicts, as for the cohort: a blocked run is not a failure
        if external in ANHIR_DATASETS:
            challenge.score(
                a.cases, a.out, external, a.workers, gates=a.gates, **common
            )
        else:
            landmarks.score(
                a.cases, a.out, external, a.workers, gates=a.gates, **common
            )
        return 0
    expected = None if a.only_run else variants.expected_variants()
    primary = None if a.only_run else list(variants.PRIMARY)
    score.score(
        a.cases,
        a.out,
        cohort,
        a.workers,
        dev=a.dev,
        expected=expected,
        primary=primary,
        with_tissue=not a.no_tissue,
        dropped=_dropped(a.drop),
        gates=a.gates,
        run_variants=variants.run_variants(),
        row_checks=variants.row_checks(
            os.path.join(a.out, "logs", "selfcheck"), score.of_this_submission
        ),
        mask_flags=variants.mask_flags(
            os.path.join(a.out, "logs", "selfcheck"), score.of_this_submission
        ),
        selfcheck_masks=variants.selfcheck_mask_rows(
            os.path.join(a.out, "logs", "selfcheck"), score.of_this_submission
        ),
        fresh=a.fresh,
        manifest=a.manifest,
    )
    from . import failure_log

    failure_log.emit_end("score", "ok")
    return 0

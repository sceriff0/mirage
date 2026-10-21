"""``python -m benchmarks.regbench``: prepare cases, run a method, score everything.

``prepare <dataset>``  write the cases of ``multiplex`` or ``hyreco`` (subset A and,
                       when it is there, subset B, in one set), idempotent
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
``compute-all``        ``plan``, ``params``, ``sheet``: the full pipeline per patient and registration
                       backend, traced (``compute_all.py``, ``slurm/compute_all.sbatch``)
``runs``               the run list, one tab-separated line per run, for the launcher
                       (columns: ``benchmarks/regbench/variants.py``); ``--all`` adds the
                       paused runs
``pick-patients``      the CPU cross's patients: smallest, median and largest by total
                       moving gigapixels (``subset.py``), one ``patient<TAB>gpx`` line each
``subset-cases``       a cases folder of the named patients alone (symlinks to their case
                       folders), for ``REGBENCH_CASES`` of the CPU cross
``staged-results``     the STARE results under ``--out`` its stage jobs made (the launcher
                       refuses ``REGBENCH_STARE_EXECUTION=local`` into such a folder)
``local``              STARE's stages on ONE cohort slide on a laptop, stored as
                       ``<run>_laptop`` with the machine's spec in its run.json
                       (``local.py``)
``cross-runs``         the CPU cross's rows: each family's best-Dice row of a scored main
                       run (DeeperHistReg's best CPU row), one ``family<TAB>row`` line each
                       and the ``--rows`` line (``paper/cpu_scaling.py``)
``cpu-scaling``        the CPU-scaling table: the cross's rows at 4 and 8 CPUs, per
                       patient and repeat (``paper/cpu_scaling.py``); ``--main MAIN``
                       takes the rows from ``cross-runs``; a GPU row is refused
``plots``              the paper's figures from the paper tables (``paper/plots.py``);
                       ``--tables DIR --to DIR`` draws them from any folder of the paper's
                       CSVs (a ``collect`` folder: drafts)
``overlays``           the overlay zooms (before, after rigid, after the whole transform) of
                       one field of a cohort case, per method's best row (``paper/overlays.py``)
``collect``            the interim collection: the existing arm, sweep and interim regbench
                       results in the paper tables' columns, NA for what is still to come
                       (``paper/collect.py``; never read by ``score`` or ``paper``)
``slim``               keep the CSVs, free the space of a results, arms or sweep root: the
                       written images and work folders (``images``), and the maps and
                       records too (``csv``, after a ``collect``); a dry run unless
                       ``--apply`` (``paper/slim.py``)

``submissions``        the submission ids the selfcheck records of a results folder carry
``verdict``            what every run's selfcheck and realcheck said, one line each per
                       run, into ``<out>/logs/verdicts.tsv`` (``score --verdicts`` reads
                       it). Always exits 0: the checks are REPORTED, never a gate
``selfcheck``          check that a method's points follow the image its own writer produced,
                       on a generated pair; exit 0 ``ok``, 1 ``inconsistent``, 2
                       ``not_checked``, 3 ``crashed`` (for a run by hand; nothing waits for
                       it and no run or row is kept out because of it)

``run --realcheck`` does the same kind of check on the real case it has just registered
and exits 3 when it is flagged (reported, never a gate). ``run --retry-oom`` and
``selfcheck --retry-memory`` are what a later rung of the memory ladder runs: they do
nothing unless the attempt before ran out of memory (``benchmarks/regbench/ladder.py``).

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
import sys

from . import COST_ONLY_KINDS, DATASETS, PUBLIC, compute_all, plugins, public_of
from . import METHODS as PUBLIC_METHODS
from .resources import n_cpus

# the public methods, and those of a private method plug-in when one is enabled
METHODS = PUBLIC_METHODS + plugins.method_names()
# what `--dataset` takes where a public dataset can be named: the cohort, then the public
# datasets (regbench.PUBLIC), each scored into tables of its own and never pooled
ALL_DATASETS = DATASETS + tuple(PUBLIC)
# the sets scored for cost alone (T32): the CPU cross and its subset take them too
COST_ONLY_SETS = tuple(d for d, s in PUBLIC.items() if s["kind"] in COST_ONLY_KINDS)


def build_parser():
    ap = argparse.ArgumentParser(prog="regbench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    prepare_sub = sub.add_parser("prepare").add_subparsers(
        dest="dataset", required=True
    )

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

    m = prepare_sub.add_parser("multiplex")
    common(m)
    m.add_argument(
        "--manifest",
        required=True,
        help="CSV: case_id,reference,moving,... (for the cohort: the output of "
        "`python -m benchmarks.regbench.manifest`, written by launch.sbatch --prepare)",
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
        help="where `regbench nuclei` left each slide's bare nuclei (its StarDist run)",
    )
    m.add_argument(
        "--nuclei-fallback",
        default="",
        choices=("", "shrink"),
        help="`shrink`: for a slide with no bare nuclei, use its outline with the "
        "10 px rim taken off (labelled shrunk_outlines, exact: false; too small "
        "where cells touch). Never used unless given",
    )

    hy = prepare_sub.add_parser(
        "hyreco",
        help="HyReCo's re-stained pairs (a public landmark set, scored apart: "
        "regbench.landmarks)",
    )
    common(hy)
    hy.add_argument(
        "--data-root",
        required=True,
        help="a folder holding HyReCo/{HE,PHH3} (subset A) and, when downloaded, "
        "HyReCo-Additional/{HE,PHH3} (subset B), as download/hyreco.sh leaves it",
    )
    hy.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="a measured pixel size (default: the file's, else the paper's 0.24)",
    )
    hy.add_argument(
        "--proxy",
        default=None,
        choices=("luminance", "haematoxylin"),
        help="STARE's one-channel proxy (default luminance)",
    )
    hy.add_argument(
        "--reduce",
        type=int,
        default=None,
        choices=(1, 2),
        help="block-average the proxy (default 1, native)",
    )
    hy.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="prepare the pairs that are there and mark every case so",
    )
    hy.add_argument(
        "--count",
        action="store_true",
        help="print the number of pairs of the subsets that are there",
    )

    nu = sub.add_parser(
        "nuclei",
        help="nuclei and outlines per native slide of a manifest: StarDist, its cell "
        "mask traced to the slide's outlines (the manifest's ref_cells / mov_cells)",
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
        help="decide every chosen slide again, whatever is kept for it: a kept "
        "StarDist run (and the outlines traced from it) is set aside and run again",
    )
    mode = nu.add_mutually_exclusive_group()
    mode.add_argument("--count", action="store_true", help="print the number of slides")
    mode.add_argument(
        "--pending",
        action="store_true",
        help="print the slides the nuclei array has anything to do for, as one "
        "--array list (empty: every slide has usable nuclei and its outlines)",
    )
    mode.add_argument(
        "--failed",
        type=int,
        default=None,
        metavar="EXIT",
        help="(nuclei.sbatch) write down that the StarDist run of the --index slide "
        "exited EXIT: the slide's record and `prepare multiplex`'s drop reason say so",
    )
    nu.add_argument(
        "--mem-gb", type=float, default=None, help="with --failed: the task's memory"
    )
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

    lo = sub.add_parser(
        "local",
        help="STARE's stages on one cohort slide on a laptop (stored as <run>_laptop)",
    )
    lo.add_argument("--cases", required=True)
    lo.add_argument("--out", required=True)
    lo.add_argument("--dataset", default="multiplex", choices=DATASETS)
    lo.add_argument("--case-id", required=True)
    lo.add_argument("--run", required=True, help="a STARE run of `regbench runs`")
    lo.add_argument("--workers", type=int, default=None, help="default: every core")
    lo.add_argument(
        "--submission",
        default=None,
        help="the submission this result belongs to (REGBENCH_SUBMISSION, as a cluster "
        "run records it): `score --submission ID` admits it then",
    )
    lo.add_argument("--force", action="store_true", help="run a result that is ok already")

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
    st.add_argument(
        "--cases",
        default=None,
        help="the prepared cases: with it, the units of a chain no watcher link "
        "finished whose last task had not ended are recorded as infrastructure "
        "(chains.mark_in_flight)",
    )

    cm = sub.add_parser(
        "chain-mark",
        help="(the chains' watcher, slurm/retry.sbatch) write units whose ladder could "
        "not go on down as the launcher's: a rung the partition cannot grant, or a "
        "step sbatch refused (chains.mark)",
    )
    cm.add_argument("--cases", required=True)
    cm.add_argument("--out", required=True)
    cm.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    cm.add_argument("--method", required=True, choices=METHODS)
    cm.add_argument("--label", default=None)
    cm.add_argument("--index", type=int, action="append", required=True)
    cm.add_argument("--rung-gb", type=float, required=True)
    cm.add_argument(
        "--cause",
        required=True,
        choices=("not_submitted", "capped", "submit_refused", "watcher_gave_up"),
        help="not_submitted: the launcher did not; capped: a rung the partition cannot "
        "grant, above a memory failure (the method's, imputed); submit_refused: sbatch "
        "refused the step after its retries; watcher_gave_up: the queue could not be "
        "read (chains.MARK_CAUSES)",
    )
    cm.add_argument("--why", required=True)

    st = sub.add_parser(
        "status",
        help="where a results folder stands (read-only): per run and variant, ok / "
        "reused / failed by cause / missing against the case list, the jobs still "
        "queued (squeue, when there is one), and what to resubmit (status.py)",
    )
    st.add_argument("--out", required=True)
    st.add_argument("--cases", default=None, help="default: REGBENCH_CASES")
    st.add_argument("--dataset", default=None, choices=ALL_DATASETS)
    st.add_argument("--submission", default=None, help="default: <out>/logs/submission")
    st.add_argument("--runs", default=None, help="only these runs (space-separated)")
    st.add_argument("--csv", action="store_true", help="the table as CSV")
    st.add_argument("--no-queue", action="store_true", help="do not ask squeue")
    st.add_argument(
        "--summary-to",
        default=None,
        help="(score_cohort.sbatch) append the chains no watcher finished to this "
        "summary.md, and print nothing",
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
        help="default: the cohort's datasets. A public dataset is scored alone into "
        "<out>/tables/<dataset>/, never pooled with the cohort",
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
        "--submission",
        default=None,
        metavar="ID",
        help="score only what this submission ran or reused (run.json: submissions; "
        "the launcher's REGBENCH_SUBMISSION). A result another submission left is "
        "`not run`",
    )
    sc.add_argument(
        "--verdicts",
        default=None,
        metavar="TSV",
        help="what every run's selfcheck and realcheck said (`verdict`, "
        "<out>/logs/verdicts.tsv): in launch.csv, in the summary and in the "
        "selfcheck_status column of every row. Nothing is left out because of it",
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

    ru = sub.add_parser(
        "reuse",
        help="READ-ONLY: per finished result, would the launcher reuse it, and which "
        "part of its reuse key differs (execution, code, inputs, method, points, "
        "record). Writes nothing under --out",
    )
    ru.add_argument("--cases", required=True)
    ru.add_argument("--out", required=True)
    ru.add_argument("--dataset", default="multiplex")
    ru.add_argument("--csv", default=None, help="also write the rows here")

    so = sub.add_parser(
        "score-case",
        help="score ONE case of a cohort dataset and keep its record under "
        "<out>/cache/scores; no table is written. `score` reads the record instead "
        "of computing the case (one array task per case on a cluster)",
    )
    so.add_argument("--cases", required=True)
    so.add_argument("--out", required=True)
    so.add_argument("--dataset", required=True, choices=ALL_DATASETS)
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

    vd = sub.add_parser(
        "verdict",
        help="what every run's selfcheck and realcheck said, one line each per run, "
        "into <out>/logs/verdicts.tsv. Always exits 0: reported, never a gate",
    )
    vd.add_argument("--out", required=True)
    vd.add_argument(
        "--runs",
        default=None,
        help="run names (`runs`, column 1), separated by spaces or commas (default: "
        "every active run)",
    )
    vd.add_argument(
        "--submission",
        default=os.environ.get("REGBENCH_SUBMISSION") or "",
        metavar="ID",
        help="the submission whose records are read (default: REGBENCH_SUBMISSION); a "
        "record of another one is reported not_checked",
    )

    sg = sub.add_parser(
        "stage",
        help="one stage job of one round of a run whose registration is several "
        "scheduler jobs (`runs`, column `stages`): coarse, tile or solve",
    )
    sg.add_argument("--cases", required=True)
    sg.add_argument("--out", required=True)
    sg.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    sg.add_argument("--method", required=True, choices=METHODS)
    sg.add_argument("--label", default=None)
    sg.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE")
    sg.add_argument("--stage", required=True, choices=("coarse", "tile", "solve"))
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

    pk = sub.add_parser(
        "pick-patients",
        help="the CPU cross's patients: smallest, median and largest by total moving "
        "gigapixels",
    )
    pk.add_argument("--cases", required=True)
    pk.add_argument("--dataset", default="multiplex", choices=DATASETS)
    pk.add_argument(
        "--n", type=int, default=3, help="evenly spaced ranks, both ends included"
    )
    ss = sub.add_parser(
        "subset-cases", help="a cases folder of the named patients alone (symlinks)"
    )
    ss.add_argument("--cases", required=True)
    ss.add_argument("--groups", required=True, help="patients, comma-separated")
    ss.add_argument("--to", required=True, help="a new (or empty) folder")
    ss.add_argument("--dataset", default="multiplex", choices=DATASETS + COST_ONLY_SETS)
    sx = sub.add_parser(
        "staged-results",
        help="the STARE result folders under --out that its stage jobs made, one per "
        "line (the launcher refuses a `local` launch into such a folder)",
    )
    sx.add_argument("--out", required=True)

    pa = sub.add_parser(
        "paper",
        help="the paper tables (benchmarks/regbench/paper/tables.py) from the scorer's "
        "tables: <out>/paper/<table>.csv for the cohort, <out>/paper/<dataset>/ otherwise",
    )
    pa.add_argument("--out", required=True, help="the results folder `score` wrote to")
    pa.add_argument("--dataset", default="multiplex", choices=ALL_DATASETS)
    pa.add_argument(
        "--cases",
        default=None,
        help="the folder the cases were prepared under: the input size of a cost point "
        "is read from its slide files (blank without it)",
    )

    sy = sub.add_parser(
        "synth",
        help="the synthetic scaling set (benchmarks/regbench/synth.py): 11 panels of 3 "
        "channels per size, cost only; generated once under <to>/synthetic/",
    )
    sy.add_argument("--source", default="", help="a real slide: its nuclear channel is the texture")
    sy.add_argument("--to", required=True, help="the cases folder (REGBENCH_CASES_SYNTHETIC)")
    sy.add_argument("--grid", default="default", choices=("default", "small"))
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument(
        "--channel", type=int, default=None,
        help="the source's nuclear channel (default: the first named DAPI/Hoechst, else 0)",
    )
    syw = sy.add_mutually_exclusive_group()
    syw.add_argument("--estimate", action="store_true", help="print the bytes only")
    syw.add_argument("--plan", action="store_true", help="print the bytes, fix the texture and the units")
    syw.add_argument("--unit", type=int, default=None, help="write one (size, panel) slide")
    syw.add_argument("--finish", action="store_true", help="write the cases, case.json last")
    syw.add_argument("--units", action="store_true", help="print the number of units")
    syw.add_argument("--status", action="store_true", help="complete / planned / partial / absent")
    syw.add_argument(
        "--results-estimate", action="store_true",
        help="print the bytes STARE's uncompressed rounds of the set will take (the plan's "
        "sizes once planned; the STARE runs among --runs)",
    )
    sy.add_argument(
        "--runs", default=None,
        help="with --results-estimate: the runs of the launch, by space or comma (default: "
        "every run of the set)",
    )

    pl = sub.add_parser(
        "plots",
        help="the paper's figures (benchmarks/regbench/paper/plots.py) from the paper tables: "
        "<out>/paper/figures/ for the cohort, <out>/paper/<dataset>/figures/ otherwise; a figure "
        "whose input table is absent is written as a placeholder naming it",
    )
    pl.add_argument("--out", default=None, help="the results folder `paper` wrote to")
    pl.add_argument("--dataset", default="multiplex", choices=ALL_DATASETS)
    pl.add_argument("--only", default=None, help="figure names, comma-separated")
    pl.add_argument(
        "--tables",
        default=None,
        metavar="DIR",
        help="draw from this folder of paper-schema CSVs instead (a `collect` folder or a "
        "paper/ folder; nothing else is read), into --to; every plot and overlay is written, "
        "a placeholder where its inputs are absent or all NA",
    )
    pl.add_argument("--to", default=None, help="with --tables: the folder the figures go to")
    pl.add_argument(
        "--draft",
        action="store_true",
        help="with --tables: stamp 'DRAFT — interim data (<sources>)' on every figure and add "
        "_draft to its name (implied when the folder holds the collector's README.txt)",
    )
    pl.add_argument(
        "--results",
        default=None,
        help="with --tables: the results folder whose images the overlay zooms are drawn from "
        "(with --cases); without it the overlays are placeholders naming what they need",
    )
    pl.add_argument("--cases", default=None, help="with --results: the prepared cases folder")

    ov = sub.add_parser(
        "overlays",
        help="the overlay zooms (benchmarks/regbench/paper/overlays.py): one field of one "
        "case before registration, after the rigid stage and after the whole transform, for "
        "each method's best row (paper/method_best.csv); writes <out>/paper/figures/ and "
        "<out>/paper/overlay_rois.csv",
    )
    ov.add_argument("--cases", required=True, help="the folder the cases were prepared under")
    ov.add_argument("--out", required=True, help="the results folder `paper` wrote to")
    ov.add_argument("--case-id", default=None, help="draw this case (default: chosen by rule)")
    ov.add_argument("--roi-um", type=float, default=300.0, help="side of the field, in um")
    ov.add_argument("--dataset", default="multiplex", choices=DATASETS)

    cr = sub.add_parser(
        "cross-runs",
        help="the CPU cross's rows (benchmarks/regbench/paper/cpu_scaling.py): each "
        "family's best-Dice row of the scored MAIN run (tables.best_of), DeeperHistReg's "
        "best CPU row; refused when MAIN has no scored tables",
    )
    cr.add_argument("--out", required=True, help="the MAIN results folder (scored)")
    cr.add_argument("--dataset", default="multiplex", choices=DATASETS)

    cs = sub.add_parser(
        "cpu-scaling",
        help="the CPU-scaling table (benchmarks/regbench/paper/cpu_scaling.py): wall time, "
        "CPU-hours and peak memory of the cross's rows at 4 and 8 CPUs, per patient and "
        "repeat (a folder's rep<k> subfolders, REGBENCH_REPEAT), e.g. --point 4=OUT_C4 "
        "--point 8=OUT_C8 --main MAIN --to O/paper/cpu_scaling.csv; a GPU row is refused",
    )
    cs.add_argument("--point", action="append", required=True, metavar="CPUS=OUT",
                    help="a results folder that ran at CPUS; repeat, also for one count")
    cs.add_argument("--rows", default=None,
                    help="result names, comma-separated (default: --main's cross-runs)")
    cs.add_argument("--main", default=None,
                    help="the scored MAIN run the rows are chosen from (cross-runs)")
    cs.add_argument("--to", required=True)
    cs.add_argument("--dataset", default="multiplex", choices=DATASETS + COST_ONLY_SETS)

    co = sub.add_parser(
        "collect",
        help="the interim collection (benchmarks/regbench/paper/collect.py): every paper table "
        "under --to from the existing arm, sweep and interim regbench results, NA for every "
        "measurement still to come; interim only, never read by `score` or `paper`",
    )
    co.add_argument("--to", required=True,
                    help="a folder outside every folder read (refused inside a results folder)")
    co.add_argument("--arms", default=None, help="the arm results root (<arm>/<patient>/qc/...)")
    co.add_argument("--sweep", default=None, help="the sweep results root (<run_id>/trace/...)")
    co.add_argument("--sweep-plan", default=None, help="the sweep's run plan CSV (run_id)")
    co.add_argument("--regbench", action="append", default=[], metavar="FOLDER",
                    help="an interim regbench results folder; repeat for several")
    co.add_argument("--cases", default=None,
                    help="the folder the cohort cases were prepared under: one expected per_pair "
                    "row per (case, terminal row of the run list)")

    sm = sub.add_parser(
        "slim",
        help="keep the CSVs, free the space (benchmarks/regbench/paper/slim.py): a DRY RUN "
        "unless --apply. --level images removes the written images and work folders "
        "(re-scoring still works); --level csv also the maps and records (needs --collected)",
    )
    sm.add_argument("--root", required=True,
                    help="a regbench results folder, an arms root or a sweep root")
    sm.add_argument("--level", default=None, choices=("images", "csv"))
    sm.add_argument("--collected", default=None, metavar="DIR",
                    help="--level csv: a `collect` folder (or the root's tables/) newer than "
                    "every run it covers, with rows of this root")
    sm.add_argument("--apply", action="store_true", help="delete (default: a dry run)")
    sm.add_argument("--status", action="store_true",
                    help="print `slimmed: images|csv` (or `slimmed: no`) and exit")

    rn = sub.add_parser("runs")
    rn.add_argument(
        "--header", action="store_true", help="print the column names first"
    )
    rn.add_argument(
        "--scorer-image",
        action="store_true",
        help="print only the image `prepare`, `verdict`, `tissue` and `score` run in",
    )
    rn.add_argument(
        "--dataset",
        default=None,
        choices=ALL_DATASETS,
        help="the runs of this dataset",
    )
    rn.add_argument(
        "--order",
        default=None,
        metavar="NAMES",
        help="print the runs in the order a launcher is to SUBMIT them: these family "
        "or run names first (space-separated), the rest after, and a run that reads "
        "another's results never before it; `default` is the default order "
        "(variants.default_order: the cohort's, STARE first on a public set). Nothing "
        "else reads this order",
    )
    rn.add_argument(
        "--order-notes",
        action="store_true",
        help="with --order: print instead one line per run that was moved behind the "
        "run it reads",
    )
    rn.add_argument(
        "--all",
        action="store_true",
        help="also print the paused runs (0 in the `active` column)",
    )
    rn.add_argument(
        "--stage-allocation",
        action="store_true",
        help="print only, for the runs that are several scheduler jobs per round "
        "(column `stages`), one line per stage and attempt: stage, attempt, CPUs, "
        "memory in GB, time limit (the pipeline's allocation of that process)",
    )

    ca = sub.add_parser(
        "compute-all",
        help="the full pipeline per patient and registration backend, traced "
        "(benchmarks/regbench/compute_all.py; slurm/compute_all.sbatch)",
    ).add_subparsers(dest="compute_cmd", required=True)
    cp = ca.add_parser(
        "plan",
        help="one tab-separated line per run, in the order they are submitted: "
        "backend, patient, launch folder, run name",
    )
    cp.add_argument("--input", required=True, help="the cohort's samplesheet")
    cp.add_argument("--out", required=True, help="the results folder (REGBENCH_OUT)")
    cpp = ca.add_parser(
        "params",
        help="write one run's -params-file, typed by nextflow_schema.json",
    )
    cpp.add_argument("--backend", required=True, choices=compute_all.BACKENDS)
    cpp.add_argument("--out", required=True)
    cs = ca.add_parser(
        "sheet",
        help="write the samplesheet's header and one patient's rows; with --sizes, "
        "the sum of that patient's slide files in GB",
    )
    cs.add_argument("--input", required=True)
    cs.add_argument("--patient", required=True)
    cs.add_argument("--out", required=True)
    cs.add_argument("--sizes", default=None)
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
            # no later step of the chain cures this: the watcher (retry.sbatch) takes a
            # run end line WITHOUT `wants=` for every step, which would climb every rung
            wants="-",
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
            time_limit=host.get("time_limit"),
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
            # One round's error costs that round only: its tiles are not listed and
            # the others still are. Raised out of here, the feeder's fallback would
            # submit every tile of every round, reused ones included.
            try:
                due = mod.tiles_due(
                    case, mod.stage_dir_of(a.out, a.label, case), opts, a.due, force
                )
            except (Exception, SystemExit) as exc:
                print(
                    f"{case.case_id}: no tiles listed, the round could not be read "
                    f"({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
                due = []
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
    paths = ladder.unit_paths(a.out, a.method, a.label, picked)
    peaks = [
        (r.get("resources") or {}).get("peak_rss_mb") for r in records if r
    ]
    peaks = [float(p) for p in peaks if isinstance(p, (int, float))]
    failure_log.emit_end(
        "run",
        status,
        cause,
        run=a.label or a.method,
        dataset=a.dataset,
        unit="+".join(c.case_id for c in picked),
        rung=next((r.get("attempt") for r in records if r), "-"),
        # what the chain's watcher may still submit for it (failure_log.chain_wants)
        wants=",".join(failure_log.chain_wants(records)) or "-",
        peak_rss_mb=f"{max(peaks):.0f}" if peaks else "-",
        record=str(paths[0]) if paths else "-",
        records=len(paths),
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
    if a.header:
        print("\t".join(variants.RUNS_COLUMNS))
    order = None if a.order is None else a.order.split()
    if order == ["default"]:
        order = variants.default_order(a.dataset)
    if a.order_notes:
        listed = [variants.run_of(r[0]) for r in variants.runs_table(a.all, a.dataset)]
        for note in variants.submission_order(order, listed)[1]:
            print(note)
        return 0
    for row in variants.runs_table(paused=a.all, dataset=a.dataset, order=order):
        print("\t".join(row))
    return 0


VERDICT_COLUMNS = ("run", "check", "status", "reference", "rows", "log")
NO_RECORD = "no selfcheck record"


def _selfcheck_line(out, name, submission):
    """The ``verdicts.tsv`` line of a run's selfcheck: never raises; a record that is
    missing, unreadable or another submission's is ``not_checked`` with the reason in
    ``log``."""
    import json
    from pathlib import Path

    from . import ladder, selfcheck, variants

    checker = variants.checked_by(name)
    path = Path(out) / "logs" / "selfcheck" / checker / "selfcheck.json"
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise ValueError("not a selfcheck record")
        log = record.get("log_path") or variants.NONE
    except (OSError, ValueError):
        record, log = {}, NO_RECORD
    if record and not ladder.mine(record, submission):
        # an older launch's verdict says nothing about this one's images and code
        log = (
            f"selfcheck record of another submission "
            f"({record.get('submission') or 'none recorded'})"
        )
        record = {}
    if not isinstance(record, dict):
        record, log = {}, NO_RECORD
    # read with the rows of the run whose selfcheck it is (for a run checked by another)
    status, rows, reference = variants.selfcheck_status(checker, record)
    if not record:
        status = selfcheck.NOT_CHECKED
    if checker != name:
        log = f"via {checker}: {log}"
    return [name, "selfcheck", status, reference or variants.NONE, ",".join(rows), log]


def _realcheck_line(out, name, submission):
    """The ``verdicts.tsv`` line of a run's realcheck (``-`` for a run without one)."""
    from . import variants

    status, row, flagged, unreadable = variants.realcheck_status(name, out, submission)
    log = variants.NONE
    said = []
    if flagged:
        said.append("flagged on " + ",".join(flagged))
    if unreadable:
        said.append("realcheck.json unreadable on " + ",".join(unreadable))
    if said:
        log = "; ".join(said)
    elif status == variants.REAL_NOT_CHECKED:
        log = "no realcheck of this submission beside an ok result"
    return [name, "realcheck", status, variants.NONE, row, log]


def _verdict(a):
    """``<out>/logs/verdicts.tsv``: per run one ``selfcheck`` line and one ``realcheck``
    line (``VERDICT_COLUMNS``). REPORTED, never a gate: it always exits 0, and nothing
    of a run or of a table waits for it or is left out because of it. A run whose
    selfcheck left no record is ``not_checked``, ``log`` = ``no selfcheck record``."""
    from pathlib import Path

    from . import failure_log, variants

    names = (a.runs or "").replace(",", " ").split()
    if not names:
        names = [variants.run_name(r) for r in variants.runs_for(None)]
    lines = []
    for name in dict.fromkeys(names):
        try:
            variants.run_of(name)
        except KeyError:
            print(f"verdict: no run named {name!r}; nothing to report for it")
            continue
        lines.append(_selfcheck_line(a.out, name, a.submission))
        lines.append(_realcheck_line(a.out, name, a.submission))
    path = Path(a.out) / "logs" / "verdicts.tsv"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tsv.tmp")
        tmp.write_text(
            "".join("\t".join(row) + "\n" for row in [VERDICT_COLUMNS, *lines])
        )
        tmp.replace(path)
    except OSError as exc:
        print(f"verdict: could not write {path}: {exc}")
    for row in lines:
        print("\t".join(row))
    failure_log.emit_end("verdict", "ok", runs=len(lines) // 2)
    return 0


def _no_case_scoring(dataset):
    """Why ``dataset`` has no per-case record and no tissue piece, or None when it has
    them. A cost-only set (T32) is scored from its resources alone; a landmark set's
    scorer (``regbench.landmarks``) reads its landmarks and the results, never a per-case
    record or a tissue piece (final review I3: the launcher submits neither job for it)."""
    from . import variants

    if variants.is_cost_only_set(dataset):
        return "is scored for cost alone (no per-case record)"
    spec = public_of(dataset)
    if spec is not None and spec["kind"] == "landmarks":
        return (
            "is a landmark set: its scorer (regbench.landmarks) reads no per-case "
            "record and no tissue piece"
        )
    return None


def _score_one(a):
    """One case's record (``score.score_case``), made exactly as ``score`` would."""
    from . import failure_log, score, variants

    why = _no_case_scoring(a.dataset)
    if why is not None:
        print(f"score-case: {a.dataset} {why}")
        return 0
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
        expected=None
        if a.only_run
        else variants.expected_variants(
            a.dataset if public_of(a.dataset) is not None else None
        ),
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

    if not tissue.have_cv2():
        raise SystemExit(
            "tissue: OpenCV (cv2) is not importable in this environment; the pieces "
            "are the released ACCREDIT code's and need it"
        )
    jobs = []
    for ds in a.dataset or DATASETS:
        if variants.is_cost_only_set(ds):
            print(f"tissue: {ds} is scored for cost alone (no tissue block)")
            continue
        why = _no_case_scoring(ds)
        if why is not None:
            print(f"tissue: {ds} {why}")
            continue
        expected = (
            None
            if a.only_run
            else variants.expected_variants(ds if public_of(ds) is not None else None)
        )
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
    if not jobs and all(_no_case_scoring(d) for d in a.dataset or DATASETS):
        return 0
    if not jobs:
        raise SystemExit(f"no prepared cases under {a.cases}")
    got = tissue.score_all(jobs, a.workers)
    print(
        f"tissue: {len(got['tissue'])} row(s), {len(got['tissue_missing'])} without "
        "one; `score` writes the tables"
    )
    return 0


# The commands a shell script reads the standard output of (a table, a count, a list of
# array indices). While one of them runs, `sys.stdout` IS the standard error: a library
# that logs or warns on "stdout" (STARE installs a root handler on it at its first
# logger) writes there, and only this module's own `print` reaches the real one.
ANSWERS = ("list", "runs", "stages", "submissions", "compute-all")
_ANSWER = None


def print(*args, **kwargs):  # noqa: A001  (this module's answers, see ANSWERS)
    if _ANSWER is not None and "file" not in kwargs:
        kwargs["file"] = _ANSWER
    builtins.print(*args, **kwargs)


def _reuse_report(a):
    import csv

    from . import reuse

    rows = reuse.report(a.cases, a.out, a.dataset)
    cols = ["variant", "case_id", "method", "label", "reusable", "differs", "detail"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    ok = [r for r in rows if r["differs"] != "record"]
    print(
        f"# reusable: {sum(r['reusable'] == 'yes' for r in ok)} of {len(ok)} ok results"
    )
    return 0


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


def _compute_all(a):
    import json
    import sys

    from . import params_json

    try:
        if a.compute_cmd == "plan":
            for r in compute_all.plan(a.input, a.out):
                print(f"{r.backend}\t{r.patient}\t{r.launch_dir}\t{r.name}")
            return 0
        if a.compute_cmd == "sheet":
            n = compute_all.sheet(a.input, a.patient, a.out, a.sizes)
            print(f"{n}")
            return 0
    except ValueError as exc:
        print(f"compute-all {a.compute_cmd}: {exc}", file=sys.stderr)
        return 2
    want = compute_all.params(a.backend)
    pairs = [
        f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in want.items()
    ]
    got = params_json.build(pairs)
    if got != want:  # the schema types a value otherwise than compute-all states it
        print(f"compute-all params: typed {got}, stated {want}", file=sys.stderr)
        return 2
    with open(a.out, "w") as fh:
        fh.write(json.dumps(got, indent=2) + "\n")
    return 0


def _main(a):
    if a.cmd == "compute-all":
        return _compute_all(a)
    if a.cmd == "synth":
        from . import synth

        return synth.main(a)
    if a.cmd == "nuclei":
        from . import nuclei

        return nuclei.main(a)
    if a.cmd == "local":
        from . import local

        return local.main(a)
    if a.cmd == "preflight":
        from . import preflight

        return preflight.main(a)
    if a.cmd == "prepare":
        return _prepare(a)
    if a.cmd == "paper":
        from .paper import tables

        return tables.main(a)
    if a.cmd == "plots":
        from .paper import plots

        if a.tables:
            return plots.main_tables(a)
        if not a.out:
            raise SystemExit("plots: --out (the results folder) or --tables (with --to)")
        return plots.main(a)
    if a.cmd == "slim":
        from .paper import slim

        return slim.main(a)
    if a.cmd == "overlays":
        from .paper import overlays

        return overlays.main(a)
    if a.cmd == "collect":
        from .paper import collect

        return collect.main(a)
    if a.cmd == "cpu-scaling":
        from .paper import cpu_scaling

        return cpu_scaling.main(a)
    if a.cmd == "cross-runs":
        from .paper import cpu_scaling

        return cpu_scaling.main_cross_runs(a)
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
    if a.cmd == "pick-patients":
        from . import subset

        print(f"# rule: {subset.RULE}")
        for group, gpx in subset.pick(a.cases, a.dataset, a.n):
            print(f"{group}\t{gpx:.3f}")
        return 0
    if a.cmd == "staged-results":
        from . import subset

        for folder in subset.staged_stare(a.out):
            print(folder)
        return 0
    if a.cmd == "subset-cases":
        from . import subset

        n = subset.write(a.cases, a.groups.split(","), a.to, a.dataset)
        print(
            f"subset-cases: {n} case(s) of {a.groups} linked under {a.to}/{a.dataset}"
        )
        return 0
    if a.cmd == "stage":
        return _stage(a)
    if a.cmd == "stages":
        return _stages(a)
    if a.cmd == "verdict":
        return _verdict(a)
    if a.cmd == "tissue":
        return _tissue(a)
    if a.cmd == "score-case":
        return _score_one(a)
    if a.cmd == "reuse":
        return _reuse_report(a)
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
        from . import chains, ladder, resources

        accounting = ladder.read_accounting(a.accounting)
        submission = resources.host_info()["submission"]
        changed = ladder.settle_all(a.out, accounting, submission)
        print(
            "settle: "
            + (", ".join(f"{n} {c}" for c, n in sorted(changed.items())) or "nothing")
            + " (results that said `killed`, by the scheduler's accounting)"
        )
        # a chain no link of its watcher finished: what it was waiting for is still in
        # the queue, and its results are not the method's (chains.mark_in_flight)
        for plan in chains.unfinished(a.out, submission):
            print(
                f"settle: WARNING the chain {plan.name} was not finished by any link "
                "of its watcher (time limit, a lost node): what it still had to "
                "submit was not submitted"
            )
        flying = chains.mark_in_flight(a.out, accounting, submission, a.cases)
        for where, n in sorted(flying.items()):
            print(
                f"settle: WARNING {n} result(s) of "
                + ("tasks still running" if where == "-" else f"the chain {where}")
                + " the scheduler had not ended: recorded as infrastructure (or "
                "watcher_gave_up), not the method's"
            )
        lost = chains.read_unmarked(a.out) if a.cases else []
        for r in lost:
            print(
                f"settle: WARNING unit {r['index'] or '?'} of the chain {r['plan']} "
                f"could not be listed ({r['reason']}): nothing of it was written down "
                "(logs/unmarked_units.csv; an `unmarked_unit` row of losses.csv)"
            )
        print(f"settle: n_unmarked_units={flying.n_unmarked_units}")
        return 0
    if a.cmd == "status":
        from . import status

        return status.main(a)
    if a.cmd == "chain-mark":
        from . import chains, resources

        n = chains.mark(
            a.out, a.cases, a.dataset, a.method, a.label, a.index, a.rung_gb,
            a.cause, a.why, resources.host_info()["submission"],
        )  # fmt: skip
        print(
            f"chain-mark: {n} result(s) of {a.label or a.method} on {a.dataset} "
            f"(unit(s) {', '.join(map(str, a.index))}) recorded as {a.cause} (or "
            f"not_submitted, where capped was not over a memory failure): the "
            f"{a.rung_gb:g} GB rung was not submitted ({a.why})"
        )
        return 0
    from . import score, variants

    if getattr(a, "submission", None):
        os.environ[score.SUBMISSION_ENV] = a.submission

    named = list(a.dataset or [])
    public = [d for d in named if public_of(d) is not None]
    cohort = [d for d in named if public_of(d) is None] or None
    # every kind is checked before anything is scored: a dataset whose scorer is not
    # here yet stops the call with nothing written, the cohort's tables neither
    for d in public:
        kind = public_of(d)["kind"]
        if kind not in score.SCORED_KINDS:
            raise SystemExit(
                score.NO_SCORER_YET.format(
                    ds=d, kind=kind, scorer=score.KIND_SCORER[kind]
                )
            )
    # the cohort, then each public dataset in a call of its own, with its own run list
    # (variants.expected_variants / primary_for): never pooled (score.tables_dir)
    nothing, names = [], []
    for ds in ([cohort] if cohort or not public else []) + [[d] for d in public]:
        one = None if ds is None or public_of(ds[0]) is None else ds[0]
        expected = None if a.only_run else variants.expected_variants(one)
        if a.only_run:
            primary = None
        elif one is None:
            primary = list(variants.PRIMARY)
        else:
            primary = variants.primary_for(one)
        got = score.score(
            a.cases,
            a.out,
            ds,
            a.workers,
            dev=a.dev,
            expected=expected,
            primary=primary,
            with_tissue=not a.no_tissue,
            verdicts=a.verdicts,
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
            manifest=a.manifest if one is None else None,
        )
        scored = [got] if one is None else list((got.get("public") or {}).values())
        # a tissue dataset's block counts as scored: it has no case rows to count
        nothing += [g.get("n_scored_cases", 1) for g in scored]
        names.append(one or "the cohort")
    from . import failure_log

    if nothing and not any(nothing):
        # rule 3: non-zero only when NOTHING could be scored; losses.csv is written
        failure_log.emit_end("score", "failed", "score_error")
        raise SystemExit(
            f"score: no case of {', '.join(names)} could be scored (every one raised: "
            "score_errors.csv, losses.csv)"
        )
    failure_log.emit_end("score", "ok")
    return 0

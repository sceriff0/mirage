"""A stand-in SLURM for the end-to-end launcher test (test_launcher_e2e.py): `sbatch`,
`squeue`, `sacct`, `sinfo` and `scancel`, each a two-line shell script that runs
``python -m benchmarks.tests.regbench.slurm_sim <command> ...``. Test-fixture code.

The scheduler is SIMPLE AND DETERMINISTIC:

* a job submitted by the launcher (outside any job) is QUEUED. :func:`drain` then runs
  the queue in submission order; every dependency (``afterany``/``afterok``) names a
  job submitted earlier, so that order honours them, and :func:`drain` checks it;
* a job submitted from INSIDE a running job (a chain's watcher, a tile submitter) runs
  at once, before ``sbatch`` returns its id, so when the watcher asks ``squeue`` the job
  has left the queue: the watcher's wait is never a deadlock;
* each task is ``bash <script> <args>`` with the job's own environment (``--export=ALL``:
  the submitter's), ``SLURM_*`` set as the scheduler sets them, and its output where
  ``--output`` says (``%x %j %A %a``). Its state is COMPLETED (exit 0), FAILED (other),
  or OUT_OF_MEMORY / TIMEOUT when the task was killed by a signal;
* ``SIM_STUB="rb-sc-* rb-score ..."``: the jobs whose NAME matches one of these
  patterns are not run; each task leaves an ok end line (``$REGBENCH_OUT/logs/end/<name>-<id>.end``) and nothing else;
* ``SIM_KILL_ON_SQUEUE=<job name>``: the first ``squeue`` that job asks kills the job's
  whole process group with SIGKILL (a watcher ended at its time limit, or lost with its
  node), once. Its state is TIMEOUT.

Everything is kept in ``$SIM_DIR`` (jobs.json, a lock file, the calls of each command).
"""

from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

COMMANDS = ("sbatch", "squeue", "sacct", "sinfo", "scancel")


def sim_dir():
    return Path(os.environ["SIM_DIR"])


def install(bin_dir, python=sys.executable):
    """Write the five stand-ins into ``bin_dir``."""
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    for cmd in COMMANDS:
        path = bin_dir / cmd
        path.write_text(
            "#!/bin/bash\n"
            f'PYTHONPATH="{repo}${{PYTHONPATH:+:$PYTHONPATH}}" exec "{python}" -m '
            f'benchmarks.tests.regbench.slurm_sim {cmd} "$@"\n'
        )
        path.chmod(0o755)
    return bin_dir


@contextmanager
def _locked():
    d = sim_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load():
    try:
        return json.loads((sim_dir() / "jobs.json").read_text())
    except (OSError, ValueError):
        return {"next": 100, "jobs": {}}


def _save(db):
    tmp = sim_dir() / "jobs.json.new"
    tmp.write_text(json.dumps(db, indent=1))
    os.replace(tmp, sim_dir() / "jobs.json")


def jobs():
    """``{id: job}`` as the scheduler holds them (for the test's assertions)."""
    return _load()["jobs"]


def _log(cmd, argv):
    sim_dir().mkdir(parents=True, exist_ok=True)
    with open(sim_dir() / f"{cmd}.log", "a") as fh:
        fh.write(shlex.join(argv) + "\n")


def indices(spec):
    """``"0-2,5%20"`` -> ``[0, 1, 2, 5]``."""
    out = []
    for part in spec.split("%")[0].split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _mem_mb(text):
    m = re.match(r"^(\d+)([KMGT]?)$", text or "")
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2) or "M"
    return n * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]


# ── sbatch ────────────────────────────────────────────────────────────────────
def sbatch(argv):
    _log("sbatch", argv)
    opts, rest, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if rest:
            rest.append(a)
        elif a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
        elif a in ("--parsable", "--export=ALL", "--no-requeue"):
            opts[a[2:]] = ""
        elif a.startswith("-"):
            opts[a.lstrip("-")] = ""
        else:
            rest.append(a)
        i += 1
    if not rest:
        print("sbatch: error: no script", file=sys.stderr)
        return 1
    deps = re.findall(r"after\w*:([0-9_:]+)", opts.get("dependency", ""))
    dep_ids = [d for chunk in deps for d in chunk.split(":") if d]
    inside = os.environ.get("SIM_IN_JOB") == "1"
    with _locked():
        db = _load()
        job_id = str(db["next"])
        db["next"] += 1
        db["jobs"][job_id] = {
            "id": job_id,
            "name": opts.get("job-name", Path(rest[0]).name),
            "script": rest[0],
            "args": rest[1:],
            "opts": opts,
            "deps": dep_ids,
            "env": dict(os.environ),
            "state": "PENDING",
            "tasks": {},
            "by": os.environ.get("SLURM_JOB_NAME", ""),
        }
        _save(db)
    if inside:
        run_job(job_id)
    print(job_id)
    return 0


def _task_env(job, task_id, index):
    env = {
        k: v
        for k, v in job["env"].items()
        if not k.startswith(("SLURM_", "SIM_KILLED"))
    }
    env.update(
        SIM_IN_JOB="1",
        SLURM_JOB_ID=task_id,
        SLURM_JOB_NAME=job["name"],
        SLURM_CPUS_PER_TASK=job["opts"].get("cpus-per-task", "1"),
    )
    mem = _mem_mb(job["opts"].get("mem", ""))
    if mem is not None:
        env["SLURM_MEM_PER_NODE"] = str(int(mem))
    if index is not None:
        env.update(SLURM_ARRAY_JOB_ID=job["id"], SLURM_ARRAY_TASK_ID=str(index))
    return env


def _output(job, task_id, index):
    pattern = job["opts"].get("output") or f"slurm-{task_id}.out"
    return (
        pattern.replace("%x", job["name"])
        .replace("%A", job["id"])
        .replace("%a", "" if index is None else str(index))
        .replace("%j", task_id)
    )


def _set(job_id, **fields):
    with _locked():
        db = _load()
        db["jobs"][job_id].update(fields)
        _save(db)


def _set_task(job_id, key, rec):
    with _locked():
        db = _load()
        db["jobs"][job_id]["tasks"][key] = rec
        _save(db)


def run_job(job_id):
    """Run every task of one job, in order; record each task's state."""
    job = _load()["jobs"][job_id]
    _set(job_id, state="RUNNING")
    spec = job["opts"].get("array")
    tasks = [(f"{job_id}_{i}", i) for i in indices(spec)] if spec else [(job_id, None)]
    states = []
    for task_id, index in tasks:
        out = _output(job, task_id, index)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        argv = ["bash", job["script"], *job["args"]]
        stub = job["env"].get("SIM_STUB", "").split()
        if any(fnmatch.fnmatchcase(job["name"], pattern) for pattern in stub):
            # a job this test does not run: it leaves an ok end line where env.sh
            # would name it, and nothing else
            end = (
                job["env"].get("REGBENCH_OUT", ".")
                + f"/logs/end/{job['name']}-{task_id}.end"
            )
            kind = Path(job["script"]).stem
            argv = ["bash", "-c", f'mkdir -p "$(dirname "{end}")"; '
                    f'echo "REGBENCH_END kind={kind} status=ok cause= whose=" > "{end}"; '
                    'echo "slurm_sim: stubbed"']  # fmt: skip
        with open(out, "a") as fh:
            p = subprocess.run(
                argv,
                env=_task_env(job, task_id, index),
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
            )  # fmt: skip
        code = p.returncode
        if code < 0:
            state = "TIMEOUT" if -code == signal.SIGKILL else "CANCELLED"
        else:
            state = "COMPLETED" if code == 0 else "FAILED"
        states.append(state)
        _set_task(job_id, str(index) if index is not None else "-", {
            "id": task_id, "state": state, "exit": code, "log": out,
            "elapsed": int(time.time() - t0),
        })  # fmt: skip
    final = (
        "COMPLETED"
        if all(s == "COMPLETED" for s in states)
        else next(s for s in states if s != "COMPLETED")
    )
    _set(job_id, state=final)


def drain(max_jobs=10000):
    """Run the queue (jobs submitted outside any job) in submission order, each once its
    dependencies have ended; return the ids run."""
    ran = []
    while len(ran) < max_jobs:
        db = _load()
        pending = [
            j for j in sorted(db["jobs"].values(), key=lambda j: int(j["id"]))
            if j["state"] == "PENDING"
        ]  # fmt: skip
        if not pending:
            return ran
        job = pending[0]
        for dep in job["deps"]:
            state = db["jobs"].get(dep.split("_")[0], {}).get("state")
            assert state not in (None, "PENDING", "RUNNING"), (job["name"], dep, state)
        run_job(job["id"])
        ran.append(job["id"])
    raise AssertionError("the queue did not drain")


# ── squeue, sacct, sinfo, scancel ─────────────────────────────────────────────
def squeue(argv):
    _log("squeue", argv)
    me = os.environ.get("SLURM_JOB_NAME", "")
    kill = os.environ.get("SIM_KILL_ON_SQUEUE", "")
    marker = sim_dir() / f"killed-{kill}"
    if kill and me == kill and not marker.exists():
        marker.write_text("1")
        os.killpg(os.getpgid(0), signal.SIGKILL)
    ids, name, fmt = None, None, "%i"
    for i, a in enumerate(argv):
        if a == "-j":
            ids = set(argv[i + 1].split(","))
        elif a.startswith("--name="):
            name = a.split("=", 1)[1]
        elif a == "-o":
            fmt = argv[i + 1]
    for job in _load()["jobs"].values():
        if job["state"] not in ("PENDING", "RUNNING"):
            continue
        if ids is not None and job["id"] not in ids:
            continue
        if name is not None and job["name"] != name:
            continue
        print(
            fmt.replace("%i", job["id"]).replace("%j", job["name"])
            .replace("%T", job["state"]).replace("%F", job["id"]).replace("%A", job["id"])
        )  # fmt: skip
    return 0


def sacct(argv):
    _log("sacct", argv)
    ids, fields = [], ["JobID", "State"]
    for i, a in enumerate(argv):
        if a == "-j":
            ids = argv[i + 1].split(",")
        elif a in ("-o", "--format"):
            fields = argv[i + 1].split(",")
        elif a.startswith("--format="):
            fields = a.split("=", 1)[1].split(",")
    sep = "|"
    db = _load()["jobs"]
    for job_id in ids:
        job = db.get(job_id.split("_")[0])
        if not job:
            continue
        for key, task in job["tasks"].items():
            values = {
                "JobID": task["id"], "State": task["state"], "JobName": job["name"],
                "ExitCode": f"{max(task['exit'], 0)}:0",
                "ElapsedRaw": str(task["elapsed"]), "Elapsed": "00:00:01",
                "TotalCPU": "00:00:01", "UserCPU": "00:00:01", "SystemCPU": "00:00:00",
            }  # fmt: skip
            print(sep.join(values.get(f, "") for f in fields))
    return 0


def sinfo(argv):
    _log("sinfo", argv)
    for line in (os.environ.get("SIM_SINFO") or "cpu*|720000|").split(";"):
        print(line)
    return 0


def scancel(argv):
    _log("scancel", argv)
    return 0


def main(argv):
    cmd, rest = argv[0], argv[1:]
    return {"sbatch": sbatch, "squeue": squeue, "sacct": sacct, "sinfo": sinfo,
            "scancel": scancel}[cmd](rest)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

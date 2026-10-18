"""``regbench local``: STARE's stages on ONE cohort slide, on a laptop.

The result is a run like any other, stored under the label ``<run>_laptop`` (so the row
``stare_laptop``) and made through the same ``cli._run`` as ``run``, with
``REGBENCH_STAGES=local`` (this process starts the stages) and ``--opt workers=N``.

**It belongs to a submission the way a cluster result does.** ``--submission ID`` sets
``REGBENCH_SUBMISSION`` for the run, which is where ``resources.host_info`` reads the id
the driver writes into ``run.json: submissions``. ``score --submission ID`` then admits the
result as it admits any other of that submission (``score.of_this_submission``, unchanged);
with no id the result carries none, and a score of no submission takes it too. A result
made without the id and copied into a submission's folder later is made again with
``--submission`` (``--force`` runs a result that is ``ok`` already).

**The machine is in ``run.json``** (key ``machine``, ``machine()``'s dict), because a
laptop's spec is part of what its numbers mean; ``paper.tables`` reads it from there for
``laptop.csv``. The laptop row is never a compared row and never the primary one: the
primary rows are named (``variants.PRIMARY``), and the pairwise table leaves ``*_laptop``
out (``score.is_laptop``).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SUFFIX = "_laptop"
KEYS = (
    "cpu_model",
    "n_cores_logical",
    "n_cores_physical",
    "mem_total_gb",
    "os",
    "python",
)


def _sysctl(key):
    """``sysctl -n key`` (macOS), or None."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _read_text(path):
    """The text of a file, or None."""
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _first(fn):
    """``fn()``, or None when it fails: the machine's spec never stops a run."""
    try:
        return fn()
    except Exception:  # noqa: BLE001  (a probe of the host: any failure is "unknown")
        return None


def _cpu_model(system):
    if system == "Darwin":
        return _sysctl("machdep.cpu.brand_string")
    text = _read_text("/proc/cpuinfo") or ""
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip() or None
    return None


def _mem_gb(system, psutil):
    if psutil is not None:
        return round(psutil.virtual_memory().total / 1024**3, 3)
    if system == "Darwin":
        raw = _sysctl("hw.memsize")
        return round(int(raw) / 1024**3, 3) if raw else None
    for line in (_read_text("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal"):
            return round(int(line.split()[1]) / 1024**2, 3)  # kB
    return None


def machine():
    """The laptop's spec: every key of ``KEYS`` has a value or None. Never raises."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        psutil = None
    system = _first(platform.system) or ""
    return {
        "cpu_model": _first(lambda: _cpu_model(system)),
        "n_cores_logical": _first(os.cpu_count),
        "n_cores_physical": _first(
            lambda: psutil.cpu_count(logical=False) if psutil is not None else None
        ),
        "mem_total_gb": _first(lambda: _mem_gb(system, psutil)),
        "os": _first(platform.platform),
        "python": _first(platform.python_version),
    }


def _refuse(message):
    print(f"regbench local: {message}", file=sys.stderr)
    raise SystemExit(2)


def _stamp_machine(a, cases, label):
    """Put ``machine()`` into every ``run.json`` the run wrote for its case."""
    from . import methods
    from .score import result_dir

    spec = machine()
    for variant in methods.variant_names(a.method, label):
        path = result_dir(a.out, variant, cases[0]) / "run.json"
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rec.setdefault("machine", spec)  # a reused result keeps ITS machine
        path.write_text(json.dumps(rec, indent=2))


def main(a):
    from . import cli, variants
    from .cases import list_cases

    try:
        run = variants.run_of(a.run)
    except KeyError:
        _refuse(f"{a.run!r} is not a run (see `regbench runs`)")
    if run.method != "stare":
        _refuse(f"{a.run} is a {run.method} run: only STARE's stages are run here")
    cases = [c for c in list_cases(a.cases, a.dataset) if c.case_id == str(a.case_id)]
    if not cases:
        _refuse(f"{a.dataset}/{a.case_id} is not prepared under {a.cases}")
    label = a.run + SUFFIX
    workers = a.workers or os.cpu_count() or 1
    opts = [f"{k}={v}" for k, v in run.opts.items() if k != "workers"]
    ns = SimpleNamespace(
        cases=a.cases,
        out=a.out,
        dataset=a.dataset,
        method=run.method,
        case_id=str(a.case_id),
        group=None,
        index=None,
        label=label,
        opt=[*opts, f"workers={workers}"],
        work=None,
        keep_work=False,
        stages=None,
        realcheck=False,
        retry_oom=False,
        ladder=None,
        previous_state="",
        accounting=None,
        accounting_wanted=None,
        accounting_waited=False,
        force=bool(a.force),
    )
    env = {"REGBENCH_STAGES": "local"}
    if a.submission:
        env["REGBENCH_SUBMISSION"] = str(a.submission)
    kept = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        code = cli._run(ns)
    finally:
        for k, v in kept.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _stamp_machine(ns, cases, label)
    return code

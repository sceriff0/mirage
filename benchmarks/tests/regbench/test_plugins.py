"""Private method plug-ins (``regbench.plugins``): off by default, and when on, one hook
feeds the method loader, the run list, the scorer's seam owners and the selfcheck.

Each case runs in its own interpreter, because the run list is built at import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

FAKE = """
import sys
import numpy as np
from benchmarks.regbench.mapping import MOV_TO_REF, PointMap

NAME = "fakeprivate"
SCOPE = "pair"
VARIANTS = ("fakeprivate",)
MASKS = ("seams",)


def register(case, work, opts):
    yield "fakeprivate", PointMap(lambda xy: np.asarray(xy, dtype=float) + 1.0, MOV_TO_REF), {
        "seconds_of_model": 2.5, "checked_with": opts.get("checked_with", "")}


METHODS = {NAME: sys.modules[__name__]}
RUNS = [dict(method=NAME, label=None, opts={}, image="FAKE_IMAGE", gpu=False, after=None),
        dict(method=NAME, label="fakeprivate_r2", opts={"repeat": "2"}, image="FAKE_IMAGE",
             gpu=False, after=None)]
FAMILY = {NAME: NAME}
SEAM_OWNERS = (NAME,)
SELFCHECK_SIZES = {NAME: [((512, 640), None)]}
SELFCHECK_OPTS = {NAME: {"checked_with": "its own options"}}
SELFCHECK_GATED_BY = {"fakeprivate_r2": NAME}
COST_COLUMNS = {"model_s": "seconds_of_model"}
"""

PROBE = """
import json
from benchmarks.regbench import cli, methods, plugins, score, selfcheck, variants
names = [variants.run_name(r) for r in variants.RUNS]
out = {
    "counts": list(variants.counts()),
    "expected": len(variants.expected_variants()),
    "names": names,
    "table_rows": len(variants.runs_table()),
    "cli_methods": list(cli.METHODS),
    "seam_owners": list(score.SEAM_OWNERS),
    "sizes": "fakeprivate" in selfcheck.SIZES,
    "family": variants.family("fakeprivate_r2"),
    "checked_by": variants.checked_by("fakeprivate_r2") if "fakeprivate_r2" in names else None,
    "cohort": [variants.run_name(r) for r in variants.runs_for("multiplex")],
    "loaded": plugins.load() is not None,
}
if "fakeprivate" in names:
    mod = methods.load("fakeprivate")
    out["scope"] = methods.scope_of("fakeprivate")
    out["variants"] = methods.variant_names("fakeprivate", "fakeprivate_r2")
    out["cost"] = score.resource_row(
        {"method": "fakeprivate", "info": {"seconds_of_model": 2.5}, "resources": {}}
    ).get("model_s")
else:
    out["cost_columns"] = "model_s" in score.resource_row({"method": "stare", "resources": {}})
print(json.dumps(out))
"""


def probe(tmp_path, switch, with_plugin=True):
    folder = tmp_path / "plug"
    folder.mkdir(exist_ok=True)
    if with_plugin:
        (folder / "plugin.py").write_text(FAKE)
    env = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_PRIVATE")}
    env["REGBENCH_PRIVATE_DIR"] = str(folder)
    if switch is not None:
        env["REGBENCH_PRIVATE"] = switch
    got = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=REPO, env=env, capture_output=True, text=True
    )
    assert got.returncode == 0, got.stderr[-2000:]
    return json.loads(got.stdout.strip().splitlines()[-1])


def public_counts():
    """The public run list, derived from the tables: (runs, GPU runs, per patient, cohort result
    names, cohort runs). STARE: every stride x level of every regime, by name, and the two tier
    runs (two result names each); ASHLAR: its cells (one each); the others are written out
    below."""
    from benchmarks.regbench import regimes as rg
    from benchmarks.regbench import variants
    from benchmarks.regbench.methods import ashlar

    tiers = {f"stare_{t}" for t in variants.STARE_TIERS}  # stare_medium, stare_low
    stare = {
        variants.stare_cell(s_, k, lv)[0]
        for reg in rg.REGIMES
        for k, lv in rg.stare_levels(reg)
        for s_ in variants.STARE_STRIDES
    } | tiers
    # 11 VALIS (the pipeline's at 3 tiers x 3 micro depths, 1.2.0 twice) + 8 DeeperHistReg
    # + PALOM
    others = 20
    other_names = (
        9 * 3 + 3 + 3 + 8 * 2 + 2
    )  # valis_mirage x 9, valis120, valis120_hires, dhr, palom
    cohort = {
        variants.stare_cell(s_, k, lv)[0]
        for k, lv in rg.stare_levels(rg.COHORT_REGIME)
        for s_ in variants.STARE_STRIDES
    } | tiers
    runs = len(stare) + others + len(ashlar.ASHLAR_CELLS)  # counts(): every active run
    # the cohort's own list (runs_for(None), the table, the expected names) is its regime's
    cohort_runs = len(cohort) + others + len(ashlar.ASHLAR_CELLS)
    names = 2 * len(cohort) + other_names + len(ashlar.ASHLAR_CELLS)
    return runs, 4, 11 + len(ashlar.ASHLAR_CELLS), names, cohort_runs


def public(got):
    runs, gpu, group, names, cohort = public_counts()
    assert got["counts"] == [runs, gpu, group] and got["expected"] == names
    assert got["table_rows"] == cohort and not got["loaded"]
    assert "fakeprivate" not in got["cli_methods"] + got["seam_owners"] + got["names"]
    assert not got["sizes"] and got["family"] is None and not got["cost_columns"]


def test_without_the_switch_nothing_changes(tmp_path):
    public(probe(tmp_path, None))
    public(probe(tmp_path, "0"))


def test_with_the_switch_and_no_plugin_nothing_changes(tmp_path):
    public(probe(tmp_path, "1", with_plugin=False))


def test_an_enabled_plugin_is_picked_up_everywhere(tmp_path):
    got = probe(tmp_path, "1")
    assert got["loaded"]
    runs, gpu, group, names, cohort = public_counts()
    assert got["counts"] == [runs + 2, gpu, group] and got["expected"] == names + 2
    assert got["names"][-2:] == ["fakeprivate", "fakeprivate_r2"]
    assert got["table_rows"] == cohort + 2
    assert got["cli_methods"][-1] == "fakeprivate"
    assert got["seam_owners"][-1] == "fakeprivate" and got["sizes"]
    assert got["family"] == "fakeprivate" and got["checked_by"] == "fakeprivate"
    assert "fakeprivate" in got["cohort"]
    assert got["scope"] == "pair" and got["variants"] == ["fakeprivate_r2"]
    assert got["cost"] == 2.5


def test_the_runs_command_lists_a_plugin_run_only_when_enabled(tmp_path):
    folder = tmp_path / "plug"
    folder.mkdir()
    (folder / "plugin.py").write_text(FAKE)
    base = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_PRIVATE")}

    def runs(extra):
        got = subprocess.run(
            [sys.executable, "-m", "benchmarks.regbench", "runs"],
            cwd=REPO,
            env={**base, **extra},
            capture_output=True,
            text=True,
        )
        assert got.returncode == 0, got.stderr[-2000:]
        return [line.split("\t") for line in got.stdout.splitlines() if line.strip()]

    off = runs({"REGBENCH_PRIVATE_DIR": str(folder)})
    on = runs({"REGBENCH_PRIVATE_DIR": str(folder), "REGBENCH_PRIVATE": "1"})
    n = public_counts()[4]
    assert len(off) == n and on[:n] == off
    assert [r[0] for r in on[n:]] == ["fakeprivate", "fakeprivate_r2"]
    assert (
        on[n][4] == "FAKE_IMAGE"
        and on[n][9] == "seams"
        and on[n + 1][8] == "fakeprivate"
    )


def test_selfcheck_options_of_a_plugin_reach_the_method(tmp_path, monkeypatch):
    folder = tmp_path / "plug"
    folder.mkdir()
    (folder / "plugin.py").write_text(FAKE)
    code = (
        "from benchmarks.regbench import plugins;"
        "print(plugins.get('SELFCHECK_OPTS')['fakeprivate']['checked_with'])"
    )
    env = {**os.environ, "REGBENCH_PRIVATE": "1", "REGBENCH_PRIVATE_DIR": str(folder)}
    got = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True
    )
    assert got.stdout.strip() == "its own options", got.stderr[-800:]


def test_the_default_plugin_folder_is_ignored_by_git():
    got = subprocess.run(
        ["git", "check-ignore", "-q", "benchmarks/regbench/private/plugin.py"], cwd=REPO
    )
    assert got.returncode == 0  # never tracked: see benchmarks/regbench/plugins.py
    text = (REPO / ".gitignore").read_text()
    assert "benchmarks/regbench/private/" in text.splitlines()

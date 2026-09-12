"""A subset re-run must produce exactly what a full re-run would.

After a code change confined to one component (STARE's SOLVE stage, say), only
the arms that change need to run again. The claim that makes this safe has two
halves, and both are asserted here rather than argued:

1. THE CLOSURE IS RIGHT (benchmarks/impact.py). ``--changed tiled`` selects every
   tiled arm AND every row that depends on one -- a QC cross that resumes a tiled
   base would otherwise keep scoring the OLD registration from the OLD session --
   and nothing else: no VALIS arm, not the shared preprocessing, no segmentation
   arm (they resume the VALIS reference), no ashlar arm (scored on VALIS nuclei).

2. THE ANALYSIS READS THE UNION. Replace ONLY the affected arms' files in a
   results root and the tables come out byte-identical to a root where a full
   re-run laid down the same files -- order-independent -- while every
   unaffected arm's rows are byte-identical to before and every affected arm's
   rows changed. Built on a synthetic root in the REAL arm layout
   (``<root>/<arm>/<patient>/qc/registration/*_seg_qc.json``, size logs and
   trace under ``<root>/<arm>/``), the way test_make_tables.py builds the sweep's.

Plus the plumbing that carries those two into practice: a subset plan's rows are
byte-identical lines of the full plan under the full plan's header, arms.csv
stays the FULL manifest during a subset build, and run_arms.sh's ARMS_REPLACE=1
moves exactly the plan's previous results aside (never deletes), refuses a base
whose crosses the plan lacks, and stays refuse-only without it.

Every test in this file was watched failing before it was trusted: the closure
tests against a closure with the dependency step removed, the equivalence test
against an analysis that only read the sweep's ``out/`` layout (which returned
no registration rows for ANY arm and so passed the equality vacuously -- the
reason the affected-rows-changed assertion exists), and the ARMS_REPLACE tests
against the launcher before the block was added.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from benchmarks import impact
from benchmarks.analysis import make_figures, make_tables
from benchmarks.analysis.lib import quality
from benchmarks.build_arm_plan import arms_manifest_rows, build_arm_plan, csv_fields
from benchmarks.build_run_plan import build_run_plan, select_runs
from benchmarks.tests.test_build_arm_plan import _fake_nextflow, _plan_csv

BENCH = Path(__file__).parents[1]
REPO = BENCH.parent
PATIENTS = ("P001", "P002")

# A small arms.yaml with every arm KIND the shipped one has, so the closure has
# every edge to traverse: 2 VALIS + 1 tiled base, a segmenter and a pairing cross
# of each, one ashlar arm on the VALIS reference's nuclei, two segmentation arms
# resuming the VALIS reference, one compute profile.
FIXTURE_ARMS = {
    "baseline": {
        "reg_qc": 2,
        "seg_method": "instantseg",
        "memory_mode": "high",
        "reg_micro_reg": 1,
        "registration_method": "valis",
        "seg_qc_pairing": "lsa",
    },
    "registration_arms": {
        "valis": {"memory_mode": ["high"], "reg_micro_reg": [1, 2]},
        "tiled": {
            "enabled": True,
            "reg_tiled_mode": ["low"],
            "reg_tiled_gate_tre": [1.0],
        },
    },
    "qc_segmenter_cross": {
        "cross": "all",
        "reference_arm": "valis_high_micro2",
        "seg_method": ["instantseg", "stardist"],
    },
    "qc_pairing_cross": {
        "cross": "all",
        "reference_arm": "valis_high_micro2",
        "seg_qc_pairing": ["lsa", "mutual_nn"],
    },
    "segmentation_arms": {
        "from_arm": "valis_high_micro2",
        "seg_method": ["instantseg", "stardist"],
    },
    "compute_profile": {"patients": [], "repeats": 1},
    "external_baseline": {
        "ashlar": {
            "enabled": True,
            "from_arm": "valis_high_micro2",
            "tile_size": [1024],
            "overlap_fraction": 0.1,
            "maximum_shift_um": [30],
        }
    },
}


@pytest.fixture(scope="module")
def plan() -> list[dict]:
    return build_arm_plan(FIXTURE_ARMS)


def _by_kind(plan):
    out: dict[str, list] = {}
    for r in plan:
        out.setdefault(r["arm_kind"], []).append(r)
    return out


def test_the_fixture_has_every_arm_kind(plan):
    """The closure tests below are only as strong as the edges the fixture offers."""
    kinds = _by_kind(plan)
    assert set(kinds) == {
        "preprocess",
        "registration",
        "registration_qc",
        "external",
        "segmentation",
        "compute",
    }
    assert len(kinds["registration"]) == 3 and len(kinds["registration_qc"]) == 6
    assert len(plan) == 14


# ---------------------------------------------------------------------------
# 1. The closure
# ---------------------------------------------------------------------------


def _names(rows):
    return {r["arm"] for r in rows}


def test_changed_tiled_is_every_tiled_arm_plus_its_crosses_and_nothing_else(plan):
    """The case this feature ships for: a change in STARE's SOLVE stage."""
    sub = impact.affected_rows(plan, ["tiled"])
    tiled_bases = {r["arm"] for r in plan if r["registration_method"] == "tiled"}
    tiled_crosses = {
        r["arm"]
        for r in plan
        if r["arm_kind"] == "registration_qc" and r["resume_run"] in tiled_bases
    }
    assert tiled_bases and tiled_crosses
    assert _names(sub) == tiled_bases | tiled_crosses
    # and, spelled out, what must NOT be there
    for r in plan:
        if r["arm"] in _names(sub):
            continue
        assert r["registration_method"] != "tiled"
        assert r["resume_run"] not in tiled_bases
    assert not any(
        r["arm_kind"] in ("preprocess", "segmentation", "external") for r in sub
    )
    assert not any("valis" in r["arm"] for r in sub)
    # `stare` is the same component under the docs' other name
    assert _names(impact.affected_rows(plan, ["stare"])) == _names(sub)


def test_changed_valis_pulls_in_everything_that_resumes_or_scores_against_valis(plan):
    sub = _names(impact.affected_rows(plan, ["valis"]))
    valis_bases = {r["arm"] for r in plan if r["registration_method"] == "valis"}
    assert valis_bases <= sub
    # crosses of a VALIS base, the segmentation arms (from_arm = VALIS reference),
    # the ashlar arm (ext_from_arm = VALIS reference) and the compute profile
    # (baseline registration_method = valis) are all affected
    for r in plan:
        if r["arm_kind"] == "registration_qc":
            assert (r["arm"] in sub) == (r["resume_run"] in valis_bases), r["arm"]
        if r["arm_kind"] in ("segmentation", "external", "compute"):
            assert r["arm"] in sub, r["arm"]
    assert not any(r["registration_method"] == "tiled" for r in plan if r["arm"] in sub)
    assert "preprocess_shared" not in sub


def test_changed_seg_backend_selects_the_rows_that_run_that_backend(plan):
    """A segmentation backend runs both as the segmentation step and as the QC
    segmenter (SEG_QC_SEGMENT is SEGMENT under an alias), so a stardist change
    touches seg_stardist AND every _segstardist cross, and nothing else."""
    sub = _names(impact.affected_rows(plan, ["seg:stardist"]))
    expect = {r["arm"] for r in plan if r["seg_method"] == "stardist"}
    assert expect == sub
    assert "seg_stardist" in sub and any(a.endswith("_segstardist") for a in sub)
    assert not any(a.endswith("_pairmutual_nn") for a in sub)
    # instantseg is the baseline instrument, so it reaches every base arm too
    sub_i = _names(impact.affected_rows(plan, ["seg:instantseg"]))
    assert {r["arm"] for r in plan if r["arm_kind"] == "registration"} <= sub_i


def test_changed_qc_is_every_row_that_scores_and_not_the_preprocess_run(plan):
    sub = _names(impact.affected_rows(plan, ["qc"]))
    assert sub == {r["arm"] for r in plan} - {"preprocess_shared"}


def test_ashlar_arms_are_untouched_by_a_tiled_change_and_reachable_alone(plan):
    """The external arms are scored on the VALIS reference's nuclei and carry no
    registration_method, so a STARE/SOLVE change leaves them byte-identical --
    and `--only 'ashlar.*'` (or `--changed ashlar`) re-runs exactly the external
    arms and nothing else, which is how their resource rows are collected once."""
    ext = {r["arm"] for r in plan if r["arm_kind"] == "external"}
    assert ext
    assert not ext & _names(impact.affected_rows(plan, ["tiled"]))
    assert not ext & _names(impact.affected_rows(plan, ["stare"]))
    assert _names(impact.affected_rows(plan, only=r"ashlar.*")) == ext
    assert _names(impact.affected_rows(plan, ["ashlar"])) == ext
    # nothing depends on an external arm, so the closure adds nothing
    assert all(not impact.dependants(plan, a) for a in ext)


def test_changed_preprocess_is_the_whole_plan(plan):
    """Everything resumes from preprocess_shared's checkpoint."""
    assert _names(impact.affected_rows(plan, ["preprocess"])) == _names(plan)


def test_only_regex_seeds_by_name_and_still_closes(plan):
    sub = _names(impact.affected_rows(plan, only=r"^valis_high_micro2$"))
    assert "valis_high_micro2" in sub
    assert {"valis_high_micro2_segstardist", "valis_high_micro2_pairmutual_nn"} <= sub
    assert {"seg_instantseg", "seg_stardist", "ashlar_t1024_s30"} <= sub
    assert "valis_high_micro1" not in sub and "compute_all" not in sub
    # a regex that seeds a cross alone selects that cross alone (nothing depends on it)
    assert _names(impact.affected_rows(plan, only=r"_pairmutual_nn$")) == {
        r["arm"] for r in plan if r["arm"].endswith("_pairmutual_nn")
    }


def test_closure_is_transitive_over_hand_built_rows():
    rows = [
        {
            "run_id": "a",
            "arm": "a",
            "resume_run": "",
            "from_arm": "",
            "ext_from_arm": "",
        },
        {
            "run_id": "b",
            "arm": "b",
            "resume_run": "",
            "from_arm": "a",
            "ext_from_arm": "",
        },
        {
            "run_id": "c",
            "arm": "c",
            "resume_run": "b",
            "from_arm": "",
            "ext_from_arm": "",
        },
        {
            "run_id": "d",
            "arm": "d",
            "resume_run": "",
            "from_arm": "",
            "ext_from_arm": "c",
        },
        {
            "run_id": "e",
            "arm": "e",
            "resume_run": "",
            "from_arm": "",
            "ext_from_arm": "",
        },
    ]
    got = impact.closure(rows, [rows[0]])
    assert [r["arm"] for r in got] == ["a", "b", "c", "d"]
    # the SAME objects, in plan order -- nothing rewritten
    assert all(any(g is r for r in rows) for g in got)
    assert [r["arm"] for r in impact.closure(rows, [rows[2]])] == ["c", "d"]
    assert impact.dependants(rows, "b") == [rows[2]]


def test_unknown_component_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown changed component 'bogus'"):
        impact.component_predicate("bogus")
    with pytest.raises(ValueError):
        impact.component_predicate("seg:")


# ---------------------------------------------------------------------------
# 2. The subset plan is row-identical to the full plan; arms.csv stays full
# ---------------------------------------------------------------------------


def _run_builder(tmp_path, cfg, *extra):
    arms = tmp_path / "arms.yaml"
    arms.write_text(yaml.safe_dump(cfg))
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\n"
        + "".join(f"{p},/x/{p}.tif,true,DAPI\n" for p in PATIENTS)
    )
    tag = hashlib.md5(" ".join(extra).encode()).hexdigest()[:6]
    out = tmp_path / f"plan_{tag}.csv"
    root = tmp_path / f"root_{tag}"
    r = subprocess.run(
        [
            "python3",
            str(BENCH / "build_arm_plan.py"),
            "--arms",
            str(arms),
            "--input",
            str(sheet),
            "--out",
            str(out),
            "--results-root",
            str(root),
            *extra,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return out, root / "arms.csv", r.stdout


@pytest.mark.parametrize(
    "extra",
    [
        ("--changed", "tiled"),
        ("--only", "^valis_high_micro2$"),
        ("--changed", "seg:stardist"),
    ],
)
def test_subset_plan_is_a_row_identical_subset_under_the_full_header(tmp_path, extra):
    full, full_manifest, _ = _run_builder(tmp_path, FIXTURE_ARMS)
    sub, sub_manifest, out = _run_builder(tmp_path, FIXTURE_ARMS, *extra)
    full_lines = full.read_text().splitlines()
    sub_lines = sub.read_text().splitlines()
    assert sub_lines[0] == full_lines[0], "the subset must carry the FULL plan's header"
    assert 1 < len(sub_lines) < len(full_lines)
    # every subset line is a full-plan line, byte for byte, in the full plan's order
    positions = [full_lines.index(ln) for ln in sub_lines[1:]]
    assert positions == sorted(positions)
    assert "SUBSET" in out and "FULL plan" in out
    # arms.csv is the FULL manifest, byte-identical to the one a full build writes
    assert sub_manifest.read_bytes() == full_manifest.read_bytes()
    n_manifest = len(sub_manifest.read_text().splitlines()) - 1
    n_ranked = sum(
        1
        for r in build_arm_plan(FIXTURE_ARMS)
        if r["arm_kind"] in ("registration", "registration_qc", "external")
    )
    assert n_manifest == n_ranked


def test_the_full_header_is_wider_than_the_subsets_own_keys(plan):
    """only_patient lives on the compute rows alone; a tiled subset has none, so
    writing it under its own keys would shrink the header -- the case the
    `fields=` argument exists for."""
    sub = impact.affected_rows(plan, ["tiled"])
    lead = ["run_id", "arm_kind", "arm"]
    assert "only_patient" in csv_fields(plan, lead)
    assert "only_patient" not in csv_fields(sub, lead)


def test_a_selection_matching_nothing_is_refused(tmp_path):
    arms = tmp_path / "arms.yaml"
    arms.write_text(yaml.safe_dump(FIXTURE_ARMS))
    sheet = tmp_path / "input.csv"
    sheet.write_text("patient_id,path_to_file,is_reference,channels\nP1,/x,true,DAPI\n")
    r = subprocess.run(
        [
            "python3",
            str(BENCH / "build_arm_plan.py"),
            "--arms",
            str(arms),
            "--input",
            str(sheet),
            "--out",
            str(tmp_path / "p.csv"),
            "--only",
            "^nothing_matches_this$",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "selects no row" in r.stderr
    assert not (tmp_path / "p.csv").exists()


def test_sweep_subset_is_row_identical_and_keeps_run_ids(tmp_path):
    """build_run_plan.py assigns run_id by enumeration, so the filter must run
    AFTER expansion or a subset would renumber -- and re-run into the wrong dirs."""
    sweep = yaml.safe_load((BENCH / "configs" / "sweep.yaml").read_text())
    plan = build_run_plan(sweep, repeats=1)
    sub = select_runs(plan, ["tiled"])
    assert sub and all(r["registration_method"] == "tiled" for r in sub)
    assert all(any(s is r for r in plan) for s in sub), (
        "rows are the plan's own objects"
    )
    assert {r["run_id"] for r in plan if r["registration_method"] == "tiled"} == {
        r["run_id"] for r in sub
    }
    # --only on varied_axis, the name an operator knows a block by. The tiled
    # method spans TWO blocks since the delta grid landed (the launched 9 cells
    # at reg_tiled_solver=legacy, and their 9 replicas at robust), so the block
    # name selects a strict subset of the method.
    by_axis = select_runs(plan, [], only=r"registration_method_grid:tiled")
    delta = select_runs(plan, [], only=r"delta_grid:solver_robust")
    assert {r["run_id"] for r in by_axis} | {r["run_id"] for r in delta} == {
        r["run_id"] for r in sub
    }
    assert {r["run_id"] for r in by_axis}.isdisjoint({r["run_id"] for r in delta})
    # and through the CLI, byte-identical lines under the full header
    full = tmp_path / "full.csv"
    part = tmp_path / "tiled.csv"
    for out, extra in ((full, []), (part, ["--only-method", "tiled"])):
        r = subprocess.run(
            [
                "python3",
                str(BENCH / "build_run_plan.py"),
                "--sweep",
                str(BENCH / "configs" / "sweep.yaml"),
                "--out",
                str(out),
                "--repeats",
                "1",
                *extra,
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
    full_lines, part_lines = (
        full.read_text().splitlines(),
        part.read_text().splitlines(),
    )
    assert part_lines[0] == full_lines[0]
    assert set(part_lines[1:]) < set(full_lines[1:])
    assert len(part_lines) - 1 == len(sub)


# ---------------------------------------------------------------------------
# 3. The synthetic results root, in the REAL arm layout
# ---------------------------------------------------------------------------

_TRACE_HEADER = (
    "task_id\tprocess\ttag\tname\tstatus\texit\tsubmit\tstart\tcomplete\tduration"
    "\trealtime\t%cpu\tcpus\tmemory\tpeak_rss\tpeak_vmem\trchar\twchar\n"
)

_PROCESSES = {
    "preprocess": ["MIRAGE:PREPROCESSING:CONVERT_IMAGE"],
    "registration": ["MIRAGE:REGISTRATION:REGISTER", "MIRAGE:REGISTRATION:WARP_SEG_QC"],
    "registration_qc": ["MIRAGE:REGISTRATION:WARP_SEG_QC"],
    "segmentation": ["MIRAGE:POSTPROCESSING:SEGMENT"],
    "compute": [
        "MIRAGE:PREPROCESSING:CONVERT_IMAGE",
        "MIRAGE:REGISTRATION:REGISTER",
        "MIRAGE:POSTPROCESSING:SEGMENT",
    ],
    "external": [],  # ashlar runs outside Nextflow: QC JSON only, no trace
}


def _h(arm: str) -> int:
    return int(hashlib.md5(arm.encode()).hexdigest()[:8], 16)


def _numbers(arm: str, version: int) -> dict:
    """Deterministic per arm, and DIFFERENT per version: version 2 is 'the same
    code path after the change', so every number moves."""
    h = _h(arm)
    return {
        "rss_gb": 1 + h % 50 + 3 * version,
        "realtime_s": 60 + h % 600 + 10 * version,
        "dice": round(0.5 + (h % 300) / 1000 + 0.05 * version, 4),
        "disp_um": round(0.2 + (h % 100) / 100 + 0.1 * version, 4),
        "valis_d": round(1.0 + (h % 40) / 10 + 0.5 * version, 3),
        "bytes": 2**30 * (1 + h % 8),
    }


def write_arm(root: Path, row: dict, version: int) -> None:
    """One arm's results, laid out as run_arms.sh publishes them:
    <root>/<arm>/{trace/trace.txt, size_logs/input_sizes.csv, <patient>/...}."""
    arm = row["arm"]
    n = _numbers(arm, version)
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    procs = _PROCESSES[row["arm_kind"]]
    if procs:
        lines = [_TRACE_HEADER]
        sizes = ["process,sample_id,filename,bytes\n"]
        for i, proc in enumerate(procs, start=1):
            leaf = proc.split(":")[-1]
            lines.append(
                f"{i}\t{proc}\t{arm}\t{leaf} ({arm})\tCOMPLETED\t0\t-\t-\t-\t"
                f"{n['realtime_s'] + 30}s\t{n['realtime_s']}s\t150%\t4\t64 GB\t"
                f"{n['rss_gb']} GB\t{n['rss_gb'] + 1} GB\t1 GB\t1 GB\n"
            )
            sizes.append(f"{proc},{arm},{arm}.ome.tif,{n['bytes']}\n")
        (d / "trace").mkdir(exist_ok=True)
        (d / "trace" / "trace.txt").write_text("".join(lines))
        (d / "size_logs").mkdir(exist_ok=True)
        (d / "size_logs" / "input_sizes.csv").write_text("".join(sizes))
    scores = row["arm_kind"] in (
        "registration",
        "registration_qc",
        "external",
        "compute",
    )
    if scores:
        stages = (
            ["rigid", "non_rigid"]
            if row["registration_method"] == "valis"
            else ["rigid", "refined"]
        )
        for pat in PATIENTS:
            q = d / pat / "qc" / "registration"
            q.mkdir(parents=True, exist_ok=True)
            (q / f"{pat}_cycle2_seg_qc.json").write_text(
                json.dumps(
                    {
                        "patient_id": pat,
                        "moving": "cycle2",
                        "reference": f"{pat}_ref",
                        "stage_order": stages,
                        "stages": {
                            "rigid": {
                                "n_pairs": 500,
                                "dice_matched": round(n["dice"] - 0.2, 4),
                                "displacement_um_p50": n["disp_um"] + 1.0,
                            },
                            stages[1]: {
                                "n_pairs": 500,
                                "dice_matched": n["dice"],
                                "displacement_um_p50": n["disp_um"],
                            },
                        },
                        "delta_vs_anchor": {
                            stages[1]: {
                                "dice_matched": 0.2,
                                "displacement_um_p50": -1.0,
                            }
                        },
                        "matching": {"pair_fraction": 0.9},
                    }
                )
            )
            if row["registration_method"] == "valis":
                s = d / pat / "registered" / "summary"
                s.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [{"name": "cycle2", "non_rigid_D": n["valis_d"], "n_matches": 200}]
                ).to_csv(s / f"{pat}_summary.csv", index=False)


def build_root(root: Path, plan: list[dict], version_of) -> None:
    for row in plan:
        write_arm(root, row, version_of(row))


def _plan_file(tmp_path: Path, plan: list[dict]) -> Path:
    p = tmp_path / "arm_plan.csv"
    p.write_text(_plan_csv(plan))
    return p


def run_analysis(root: Path, plan_csv: Path, outdir: Path) -> dict[str, bytes]:
    """Every CSV `make arm-tables` produces: make_tables' six + make_figures' tidy
    files, keyed by filename. Figures themselves are not compared (PDF/SVG bytes
    carry timestamps); every number they draw is in these CSVs."""
    make_tables.build_paper_data(root, plan_csv, outdir / "tables")
    make_figures.run(root, plan_csv, "none", outdir / "figures", formats=("svg",))
    out = {}
    for f in sorted(outdir.rglob("*.csv")):
        out[f.relative_to(outdir).as_posix()] = f.read_bytes()
    return out


def test_the_fixture_root_is_read_in_the_arm_layout(tmp_path, plan):
    """The harvesters used to hardcode the sweep's out/ segment and returned
    NOTHING for an arm results root -- registration_accuracy.csv was empty for
    the whole arm benchmark while the trace tables filled normally. With that
    bug, the equivalence test below passes vacuously, so this is asserted first."""
    root = tmp_path / "root"
    build_root(root, plan, lambda r: 1)
    plan_csv = _plan_file(tmp_path, plan)
    reg = quality.harvest_registration_qc(root, plan_csv)
    scoring = {
        r["arm"]
        for r in plan
        if r["arm_kind"] in ("registration", "registration_qc", "external", "compute")
    }
    assert set(reg["run_id"]) == scoring
    assert len(reg) == len(scoring) * len(PATIENTS) * 2  # two stages each
    rtre = quality.harvest_valis_rtre(root, plan_csv)
    assert set(rtre["run_id"]) == {
        r["arm"]
        for r in plan
        if r["arm"] in scoring and r["registration_method"] == "valis"
    }
    # and the sweep layout is untouched: out/ wins when it exists
    (root / "valis_high_micro2" / "out").mkdir()
    assert "valis_high_micro2" not in set(
        quality.harvest_registration_qc(root, plan_csv)["run_id"]
    )


def _rows(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    return df[df["run_id"] == run_id].reset_index(drop=True)


def test_subset_rerun_tables_equal_a_full_rerun(tmp_path, plan):
    """The whole point, end to end.

    A: the completed benchmark.  B: the SAME root after a `--changed tiled` subset
    re-run (only the affected arms' files replaced, the old ones moved aside as
    ARMS_REPLACE does).  C: a fresh root a FULL re-run would produce -- unaffected
    arms regenerate the same numbers (their code did not change), affected arms
    the new ones -- laid down in plan order rather than B's replace order.
    """
    affected = impact.affected_rows(plan, ["tiled"])
    aff = {r["arm"] for r in affected}
    assert aff and aff < {r["arm"] for r in plan}
    plan_csv = _plan_file(tmp_path, plan)

    root_a = tmp_path / "A"
    build_root(root_a, plan, lambda r: 1)
    tables_a = run_analysis(root_a, plan_csv, tmp_path / "out_A")

    root_b = tmp_path / "B"
    shutil.copytree(root_a, root_b)
    for r in reversed(affected):  # reverse order: the union must not depend on order
        shutil.move(str(root_b / r["arm"]), str(root_b / ".replaced" / "ts" / r["arm"]))
        write_arm(root_b, r, 2)
    tables_b = run_analysis(root_b, plan_csv, tmp_path / "out_B")

    root_c = tmp_path / "C"
    build_root(root_c, plan, lambda r: 2 if r["arm"] in aff else 1)
    tables_c = run_analysis(root_c, plan_csv, tmp_path / "out_C")

    # (a) the subset re-run root and the full re-run root give the same tables
    assert set(tables_b) == set(tables_c) == set(tables_a)
    assert {
        "tables/runs_master.csv",
        "tables/registration_accuracy.csv",
        "tables/param_matrix.csv",
        "figures/measurements.csv",
    } <= set(tables_b)
    for name in tables_b:
        assert tables_b[name] == tables_c[name], (
            f"{name} differs between subset and full re-run"
        )

    # (b) unaffected arms' rows are byte-identical before and after; (c) affected changed
    keyed = (
        "tables/runs_master.csv",
        "tables/param_matrix.csv",
        "tables/registration_accuracy.csv",
    )
    for name in keyed:
        a = pd.read_csv(tmp_path / "out_A" / name)
        b = pd.read_csv(tmp_path / "out_B" / name)
        assert not a.empty
        present = set(a["run_id"])
        for arm in {r["arm"] for r in plan} - aff:
            if arm in present:
                pd.testing.assert_frame_equal(_rows(a, arm), _rows(b, arm))
        changed = [arm for arm in aff if arm in present]
        assert changed, f"no affected arm reaches {name}"
        for arm in changed:
            assert not _rows(a, arm).equals(_rows(b, arm)), f"{arm} unchanged in {name}"
    # the moved-aside copies are invisible to the analysis (it reads the plan's arms only)
    assert (root_b / ".replaced" / "ts").is_dir()


# ---------------------------------------------------------------------------
# 4. run_arms.sh ARMS_REPLACE
# ---------------------------------------------------------------------------


def _launch_cfg():
    cfg = yaml.safe_load((BENCH / "configs" / "arms.yaml").read_text())
    cfg["registration_arms"]["valis"]["memory_mode"] = ["high"]
    cfg["registration_arms"]["valis"]["reg_micro_reg"] = [1, 2]
    cfg["registration_arms"]["tiled"]["reg_tiled_mode"] = ["low"]
    cfg["registration_arms"]["tiled"]["reg_tiled_gate_tre"] = [1.0]
    cfg["external_baseline"]["ashlar"]["enabled"] = False
    cfg["segmentation_arms"]["seg_method"] = []
    cfg["qc_segmenter_cross"]["cross"] = "all"
    cfg["qc_pairing_cross"]["cross"] = "all"
    return cfg


@pytest.fixture
def launched(tmp_path):
    """A results root after a FULL launch with the stub nextflow."""
    plan = build_arm_plan(_launch_cfg())
    root = tmp_path / "arm_results"
    root.mkdir()
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/x/a.tif,true,DAPI\n"
    )
    log = tmp_path / "launches.log"
    _fake_nextflow(tmp_path / "bin", log)
    env = dict(
        os.environ,
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        ARMS_CONCURRENCY="4",
    )
    env.pop("ARMS_REPLACE", None)

    def run(rows, **extra_env):
        plan_csv = (
            tmp_path / f"plan_{len(rows)}_{extra_env.get('ARMS_REPLACE', '0')}.csv"
        )
        plan_csv.write_text(
            "\n".join(
                [_plan_csv(plan).splitlines()[0]]
                + [
                    ln
                    for ln in _plan_csv(plan).splitlines()[1:]
                    if ln.split(",")[0] in {r["run_id"] for r in rows}
                ]
            )
            + "\n"
        )
        before = len(log.read_text().splitlines()) if log.exists() else 0
        r = subprocess.run(
            ["bash", str(BENCH / "run_arms.sh"), str(plan_csv), str(sheet), str(root)],
            env=dict(env, **extra_env),
            capture_output=True,
            text=True,
            timeout=300,
        )
        new = (
            [ln.split("|") for ln in log.read_text().splitlines()[before:]]
            if log.exists()
            else []
        )
        return r, [ln[1] for ln in new if ln[0] != "LOCKFAIL"]

    r, names = run(plan)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(names) == sorted(f"arms-{p['run_id']}" for p in plan)
    return plan, root, run


def test_without_arms_replace_a_subset_relaunch_is_refused_and_names_the_switch(
    launched,
):
    plan, root, run = launched
    sub = impact.affected_rows(plan, ["tiled"])
    r, names = run(sub)
    assert names == [], "the relaunch reached nextflow"
    assert "ARMS_REPLACE=1" in r.stderr
    assert not (root / ".replaced").exists()


def test_arms_replace_moves_exactly_the_subset_aside_and_relaunches_it(launched):
    plan, root, run = launched
    sub = impact.affected_rows(plan, ["tiled"])
    sub_ids = {r["run_id"] for r in sub}
    base = "tiled_low_gate1"
    # the tiled base, its two segmenter crosses, its pairing cross, its solver cross
    assert base in sub_ids and len(sub) == 5
    untouched = [p["run_id"] for p in plan if p["run_id"] not in sub_ids]
    before = {
        u: (root / u).stat().st_mtime_ns for u in untouched if (root / u).exists()
    }
    valis_hist = (
        root / ".launch" / "valis_high_micro2" / ".nextflow" / "history"
    ).read_bytes()

    r, names = run(sub, ARMS_REPLACE="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(names) == sorted(f"arms-{i}" for i in sub_ids), names

    replaced = list((root / ".replaced").iterdir())
    assert len(replaced) == 1, replaced
    ts = replaced[0]
    # every subset arm's previous results moved aside, the base's launch dir with them
    for i in sub_ids:
        assert (ts / i).is_dir(), f"{i} not moved aside"
        assert (root / i).is_dir(), f"{i} not relaunched"
    assert (ts / ".launch" / base / ".nextflow" / "history").exists()
    assert (root / ".launch" / base / ".nextflow" / "history").exists(), (
        "base relaunched"
    )
    # the cross arms have no launch dir of their own -- nothing else under .launch moved
    assert sorted(p.name for p in (ts / ".launch").iterdir()) == [base]
    # unaffected arms untouched: same dirs, same history bytes
    for u, m in before.items():
        assert (root / u).stat().st_mtime_ns == m, f"{u} was touched"
        assert not (ts / u).exists()
    assert (
        root / ".launch" / "valis_high_micro2" / ".nextflow" / "history"
    ).read_bytes() == valis_hist
    assert "moved aside" in r.stdout and "not deleted" in r.stdout


def test_arms_replace_refuses_a_base_whose_crosses_are_not_in_the_plan(launched):
    """A hand-filtered plan: the tiled base without its four crosses (three QC
    instruments and the solver cross)."""
    plan, root, run = launched
    base_only = [p for p in plan if p["run_id"] == "tiled_low_gate1"]
    crosses = [p["run_id"] for p in plan if p["resume_run"] == "tiled_low_gate1"]
    assert len(crosses) == 4
    r, names = run(base_only, ARMS_REPLACE="1")
    assert r.returncode != 0
    assert names == [], "refused, yet something launched"
    for c in crosses:
        assert c in r.stderr, f"the refusal does not name {c}"
    assert "--changed" in r.stderr
    assert not (root / ".replaced").exists(), "refused, yet something was moved"
    assert (root / "tiled_low_gate1").is_dir()


def test_arms_replace_relaunches_a_cross_alone_without_touching_its_base(launched):
    plan, root, run = launched
    cross = "tiled_low_gate1_pairmutual_nn"
    base = "tiled_low_gate1"
    hist = root / ".launch" / base / ".nextflow" / "history"
    base_line = next(
        ln for ln in hist.read_text().splitlines() if f"\tarms-{base}\t" in ln
    )
    r, names = run([p for p in plan if p["run_id"] == cross], ARMS_REPLACE="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert names == [f"arms-{cross}"]
    ts = next((root / ".replaced").iterdir())
    assert (ts / cross).is_dir()
    assert not (ts / ".launch" / base / "work").exists(), "the base's launch dir moved"
    assert (ts / ".launch" / base / ".nextflow" / f"history.before-{cross}").exists()
    assert (root / ".launch" / base / "work").is_dir()
    assert base_line in hist.read_text(), "the base's own history line was lost"
    # the cross resumed the base's session, as a normal launch would
    assert f"arms-{cross}" in hist.read_text()


def test_arms_replace_with_nothing_to_replace_is_a_plain_launch(tmp_path):
    plan = build_arm_plan(_launch_cfg())
    sub = impact.affected_rows(plan, ["tiled"])
    root = tmp_path / "fresh"
    root.mkdir()
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/x/a.tif,true,DAPI\n"
    )
    log = tmp_path / "launches.log"
    _fake_nextflow(tmp_path / "bin", log)
    plan_csv = tmp_path / "plan.csv"
    lines = _plan_csv(plan).splitlines()
    keep = {r["run_id"] for r in sub}
    plan_csv.write_text(
        "\n".join([lines[0]] + [ln for ln in lines[1:] if ln.split(",")[0] in keep])
        + "\n"
    )
    # the tiled base resumes preprocess_shared's checkpoint, which the fresh root lacks:
    # write it so the launch is not SKIPped on the missing csv
    (root / "preprocess_shared" / "csv").mkdir(parents=True)
    (root / "preprocess_shared" / "csv" / "preprocessed.csv").write_text("patient_id\n")
    r = subprocess.run(
        ["bash", str(BENCH / "run_arms.sh"), str(plan_csv), str(sheet), str(root)],
        env=dict(
            os.environ,
            PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            ARMS_REPLACE="1",
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "nothing to replace" in r.stdout
    assert not (root / ".replaced").exists()
    assert len(log.read_text().splitlines()) == len(sub)


# ---------------------------------------------------------------------------
# 5. The Makefile and the sweep launcher carry the same rules
# ---------------------------------------------------------------------------


def _recipe(makefile: str, target: str) -> str:
    m = re.search(rf"^{re.escape(target)}:(.*)$\n((?:\t.*\n)*)", makefile, re.M)
    assert m, f"no target {target}"
    return m.group(1), m.group(2)


def test_make_targets_keep_the_full_plan_for_the_tables():
    mk = (REPO / "Makefile").read_text()
    deps, recipe = _recipe(mk, "arm-plan-subset")
    assert "--changed" in recipe and "--only" in recipe
    assert "$(SUBSET_PLAN)" in recipe and "$(ROOT)_plan.csv" not in recipe, (
        "the subset must not overwrite the full plan"
    )
    deps, recipe = _recipe(mk, "arm-rerun")
    assert "arm-plan-subset" in deps and "ARMS_REPLACE=1" in recipe
    deps, recipe = _recipe(mk, "arm-tables")
    assert "arm-rerun" not in deps and "arm-run" not in deps
    assert "$(ROOT)_plan.csv" in recipe, "arm-tables must read the FULL plan"
    assert "arm-plan-subset" in mk.split(".PHONY:")[1].split("\n\n")[0]
    assert "arm-rerun" in mk.split(".PHONY:")[1].split("\n\n")[0]


def test_run_sweep_refuses_a_rerun_unless_sweep_replace_moves_it_aside():
    code = "\n".join(
        ln
        for ln in (BENCH / "run_sweep.sh").read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "SWEEP_REPLACE" in code
    assert ".replaced/" in code and 'mv "$run_dir"' in code
    assert "rm -rf" not in code.split("SWEEP_REPLACE")[0], (
        "the launcher must never delete"
    )
    i = code.index('"${SWEEP_REPLACE:-0}" == "1"')
    assert "continue" in code[i : i + 800], "without the switch the run is skipped"


def test_submit_arms_builds_a_subset_plan_under_its_own_name():
    code = "\n".join(
        ln
        for ln in (BENCH / "submit_arms.sh").read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "arm_plan.subset.csv" in code
    assert "--changed" in code and "--only" in code
    # the analysis hint at the end still names the FULL plan
    tail = code[code.index("Arms finished") :]
    assert "arm_plan.csv" in tail and "subset" not in tail


def test_arms_manifest_is_independent_of_the_selection():
    """The function main() writes arms.csv from takes the full plan; feeding it
    a subset would shrink the manifest, so it must never be handed one."""
    # Rebuilt here rather than taken from the module fixture: arms_manifest_rows
    # reads the labels the LAST build_arm_plan() call recorded.
    plan = build_arm_plan(FIXTURE_ARMS)
    full = arms_manifest_rows(plan)
    sub = arms_manifest_rows(impact.affected_rows(plan, ["tiled"]))
    assert len(sub) < len(full)
    src = (BENCH / "build_arm_plan.py").read_text()
    body = src[src.index("def main(") :]
    assert "arms_manifest_rows(plan)" in body and "arms_manifest_rows(rows)" not in body

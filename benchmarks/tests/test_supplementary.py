"""The supplementary set is drawn from one arm root, with the choices made by looking.

Through the real renderers, on the mosaic tests' synthetic arms (QC composites + reg_qc=2
scorer JSONs), laid out as a unified results root: VALIS, STARE and ASHLAR arms
side by side, told apart only by the plan's `method` column. What is pinned:

  * `best` is the arm with the highest median final-stage Dice, `high` the configured
    tier, and picks.csv says which arm and why;
  * every method set and config of the mosaic is drawn on the SAME ROIs -- a column may
    differ from its neighbour only by the method that registered it;
  * S4 puts Before and one After per method side by side on one crop, with the per-mode
    values the legend needs; S8 splits the scorer's numbers by case and by panel pair;
  * a figure with nothing on disk says so and does not cost the others.
"""

from __future__ import annotations

import csv
import json
import shutil

import pytest
import yaml

from benchmarks import supplementary as sp
from benchmarks.tests import test_reg_mosaic as tm

ARMS = {
    # arm dir            source   method  kind
    "valis_high_micro2": ("armB", "valis", "registration"),
    "valis_low_micro0": ("armB", "valis", "registration"),
    "tiled_high_s128": ("armA", "stare", "registration"),
    "ashlar_t1024_s240": ("armB", "ashlar", "external"),
    # a QC cross re-scores its base: never a candidate for `best`, however high it scores
    "valis_high_micro2_segstardist": ("armA", "valis", "registration_qc"),
}


@pytest.fixture(scope="module")
def unified(tmp_path_factory):
    src = tm.arm_root.__wrapped__(tmp_path_factory)
    root = tmp_path_factory.mktemp("unified") / "arm_results"
    root.mkdir()
    for arm, (from_arm, _, _) in ARMS.items():
        shutil.copytree(src / from_arm, root / arm)
    # valis_low_micro0 scores BETTER than the configured high tier: `best` must find it.
    for j in (root / "valis_low_micro0").rglob("*_seg_qc.json"):
        d = json.loads(j.read_text())
        d["stages"]["refined"]["dice_matched"] = 0.88
        j.write_text(json.dumps(d))
    plan = root.parent / "arm_plan.csv"
    with open(plan, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "arm_kind", "arm", "method", "seg_method", "resume_run"])
        for arm, (_, method, kind) in ARMS.items():
            base = "valis_high_micro2" if kind == "registration_qc" else ""
            w.writerow([arm, kind, arm, method, "instantseg", base])
    cfg = yaml.safe_load(
        (sp.REPO_ROOT / "benchmarks" / "configs" / "supplementary.yaml").read_text()
    )
    cfg["high"]["ashlar"] = "ashlar_t1024_s240"
    cfg["mosaic"].update(variants=2, patch_um=32, rows=2, kinds=["overlay"])
    cfg["S4"].update(field_um=48, zoom_um=0, variants=2, rounds=["CD3"])
    cfg["S7"]["patients"] = []
    cfg["options"].update(formats="png", dpi=50)
    conf = root.parent / "supplementary.yaml"
    conf.write_text(yaml.safe_dump(cfg))
    return root, plan, conf


def _run(unified, out, *extra):
    root, plan, conf = unified
    return sp.main(
        [
            "--results",
            str(root),
            "--plan",
            str(plan),
            "--config",
            str(conf),
            "-o",
            str(out),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def drawn(unified, tmp_path_factory):
    out = tmp_path_factory.mktemp("supp")
    rc = _run(unified, out, "--only", "mosaic,S4,S8")
    return rc, out


def test_best_is_the_top_scored_arm_and_high_the_configured_one(drawn):
    rc, out = drawn
    picks = {
        (r["method"], r["config"]): r
        for r in csv.DictReader((out / "picks.csv").open())
    }
    assert picks[("valis", "high")]["arm"] == "valis_high_micro2"
    assert picks[("valis", "best")]["arm"] == "valis_low_micro0"
    assert float(picks[("valis", "best")]["median_dice"]) == pytest.approx(0.88)
    assert picks[("stare", "high")]["arm"] == "tiled_high_s128"
    assert all(p["arm"] != "valis_high_micro2_segstardist" for p in picks.values()), (
        "a QC cross is the same registration measured another way, never a config"
    )


def test_every_mosaic_set_and_config_shows_the_same_tissue(drawn):
    rc, out = drawn
    assert rc == 0
    rois = {}
    for d in sorted((out / "mosaic").glob("*_*/v*")):
        if d.parent.name == "_anchor":
            continue
        (m,) = list(d.glob("P1*_rois.json"))
        rois[(d.parent.name, d.name)] = [
            (r["y"], r["x"]) for r in json.loads(m.read_text())["rois"]
        ]
        assert list(d.glob("P1*_mosaic.png")), d
    names = {k[0] for k in rois}
    assert names == {"all_high"}, (
        "the mosaic is drawn at the high tier only (mosaic.configs)"
    )
    for v in ("v1", "v2"):
        assert len({tuple(r) for (s, vv), r in rois.items() if vv == v}) == 1, v
    assert rois[("all_high", "v1")] != rois[("all_high", "v2")], "variants must differ"


def test_s4_puts_before_and_each_method_on_one_crop_with_its_values(drawn):
    rc, out = drawn
    figs = sorted((out / "S4" / "all_high").glob("v*/P1_all_high_v*.png"))
    assert len(figs) == 2, figs
    crops = set()
    for method in ("valis", "stare", "ashlar"):
        (j,) = list(
            (out / "S4" / "panels" / f"{method}_high" / "v1").glob("*_overlay.json")
        )
        c = json.loads(j.read_text())["crop"]
        crops.add((c["y"], c["x"]))
    assert len(crops) == 1, f"one crop for every method, got {crops}"
    vals = list(
        csv.DictReader(
            next((out / "S4" / "all_high" / "v1").glob("*_values.csv")).open()
        )
    )
    assert [v["method"] for v in vals] == ["VALIS", "STARE", "ASHLAR"]
    assert float(vals[1]["median_dice_matched"]) == pytest.approx(0.92)


def test_s8_splits_the_scorer_by_case_and_by_panel_pair(drawn):
    rc, out = drawn
    assert (out / "S8" / "all_high" / "S8_all_high.png").is_file()
    by_pair = list(
        csv.DictReader((out / "S8" / "all_high" / "S8_values_by_panel_pair.csv").open())
    )
    assert {r["panel_pair"] for r in by_pair} == {"DAPI_CD3", "DAPI_CD8"}, (
        "VALIS (file stem) and the manifest backends (channel set) name a moving slide "
        "differently; both must land on one panel-pair label"
    )


def test_an_unnamed_tier_means_high_so_best_is_drawn_only_when_asked(
    drawn, unified, tmp_path
):
    """User ruling 2026-09-30: a legend that names no registration tier means the HIGH one.
    `best` is still computed into picks.csv, but no figure is drawn at it by default."""
    rc, out = drawn
    drawn_dirs = {d.name for f in ("mosaic", "S4", "S8") for d in (out / f).glob("*_*")}
    assert drawn_dirs and not {d for d in drawn_dirs if d.endswith("_best")}, drawn_dirs
    root, plan, conf = unified
    cfg = yaml.safe_load(conf.read_text())
    cfg["configs"] = ["high", "best"]
    both = tmp_path / "both.yaml"
    both.write_text(yaml.safe_dump(cfg))
    assert (
        sp.main(
            [
                "--results",
                str(root),
                "--plan",
                str(plan),
                "--config",
                str(both),
                "-o",
                str(tmp_path / "o"),
                "--only",
                "S8",
            ]
        )
        == 0
    )
    assert (tmp_path / "o" / "S8" / "all_best" / "S8_all_best.png").is_file()


def test_a_missing_high_arm_falls_back_to_another_high_arm_never_a_lower_tier(
    unified, tmp_path
):
    """valis_low_micro0 scores best; if the configured high arm is absent the `high` pick
    must stay on the high tier (valis_high_micro2), and a method with NO high-tier arm is
    left out of `high` rather than silently drawn at a lower tier."""
    root, plan, conf = unified
    cfg = yaml.safe_load(conf.read_text())
    cfg["high"]["valis"] = "valis_high_micro9"  # not on disk
    cfg["high"]["stare"] = "tiled_low_s64"  # not on disk; stare HAS a high arm
    c = tmp_path / "c.yaml"
    c.write_text(yaml.safe_dump(cfg))
    assert (
        sp.main(
            [
                "--results",
                str(root),
                "--plan",
                str(plan),
                "--config",
                str(c),
                "-o",
                str(tmp_path),
                "--check",
            ]
        )
        == 0
    )
    picks = {
        (r["method"], r["config"]): r
        for r in csv.DictReader((tmp_path / "picks.csv").open())
    }
    assert picks[("valis", "high")]["arm"] == "valis_high_micro2"
    assert "high tier" in picks[("valis", "high")]["why"]
    assert picks[("stare", "high")]["arm"] == "tiled_high_s128"
    assert sp.tier_of("valis_low_micro0") == "low"
    assert sp.tier_of("tiled_high_s128") == "high"
    assert sp.tier_of("ashlar_t1024_s240") == ""


def test_check_reports_every_figure_and_draws_nothing(unified, tmp_path):
    root, plan, conf = unified
    assert (
        sp.main(
            [
                "--results",
                str(root),
                "--plan",
                str(plan),
                "--config",
                str(conf),
                "-o",
                str(tmp_path),
                "--check",
                "--ihc",
                str(tmp_path / "no_ihc"),
            ]
        )
        == 0
    )
    rows = {r["figure"]: r for r in csv.DictReader((tmp_path / "check.csv").open())}
    expected = {
        "mosaic",
        "S2",
        "S3a",
        "S3b",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
        "S10",
        "S11",
    }
    assert expected <= set(rows), set(rows)
    assert rows["mosaic"]["status"] == "READY", rows["mosaic"]
    assert rows["S8"]["status"] == "READY" and "high" in rows["S8"]["detail"]
    assert rows["S6"]["status"] == "MISSING", "the fixture has no segmentation arm"
    assert rows["S3a"]["status"] == "MISSING"
    assert rows["S11"]["status"] == "MISSING"
    assert not (tmp_path / "mosaic").exists() and not (tmp_path / "S8").exists()


def test_the_index_lists_every_variant_and_not_the_working_files(drawn):
    rc, out = drawn
    page = (out / "index.html").read_text()
    assert "mosaic/all_high/v2/" in page and "S4/all_high/v1/" in page
    assert "_anchor" not in page and "/panels/" not in page


def test_a_figure_with_nothing_on_disk_is_skipped_not_fatal(unified, tmp_path):
    assert _run(unified, tmp_path, "--only", "S3,S5,S6,S2") == 0
    assert (tmp_path / "index.html").is_file()


def test_dry_run_renders_nothing(unified, tmp_path):
    assert _run(unified, tmp_path, "--only", "mosaic", "--dry-run") == 0
    assert not (tmp_path / "mosaic").exists()
    assert "reg_mosaic" in (tmp_path / "commands.txt").read_text()


def test_a_plan_without_the_method_column_is_refused(unified, tmp_path):
    root, plan, conf = unified
    old = tmp_path / "old_plan.csv"
    rows = list(csv.DictReader(plan.open()))
    with open(old, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=[k for k in rows[0] if k != "method"], extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="method"):
        sp.main(
            [
                "--results",
                str(root),
                "--plan",
                str(old),
                "--config",
                str(conf),
                "-o",
                str(tmp_path / "o"),
            ]
        )


def test_the_one_job_script_dry_runs_both_halves(unified, tmp_path):
    """submit_supplementary.sh: the ihc half is announced, the mirage half prints every
    render, nothing is drawn, and the plan's missing `method` column is refused."""
    import os
    import subprocess

    root, plan, conf = unified
    ihc = tmp_path / "ihc"
    (ihc / "figures").mkdir(parents=True)
    (ihc / "figures" / "_common.R").write_text("")
    env = dict(
        os.environ,
        OUT=str(tmp_path / "out"),
        SRC_DIR=str(sp.REPO_ROOT),
        RESULTS=str(root),
        PLAN=str(plan),
        CONFIG=str(conf),
        IHC=str(ihc),
        ONLY="mosaic+S10",
        DRY_RUN="1",
        RENDER_EXEC="",
    )
    script = sp.REPO_ROOT / "benchmarks" / "submit_supplementary.sh"
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "benchmarks/ihc/supplementary.R S10" in r.stdout
    assert "[dry-run]" in r.stdout and "reg_mosaic" in r.stdout
    assert not (tmp_path / "out" / "mosaic").exists()
    bare = tmp_path / "bare_plan.csv"
    bare.write_text("run_id,arm_kind,arm\nx,registration,x\n")
    r = subprocess.run(
        ["bash", str(script)],
        env=dict(env, PLAN=str(bare)),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1 and "no `method` column" in r.stderr


def test_the_one_job_script_check_writes_check_csv_and_draws_nothing(unified, tmp_path):
    import os
    import subprocess

    root, plan, conf = unified
    env = dict(
        os.environ,
        OUT=str(tmp_path / "out"),
        SRC_DIR=str(sp.REPO_ROOT),
        RESULTS=str(root),
        PLAN=str(plan),
        CONFIG=str(conf),
        IHC="",
        CHECK="1",
        RENDER_EXEC="",
    )
    script = sp.REPO_ROOT / "benchmarks" / "submit_supplementary.sh"
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader((tmp_path / "out" / "check.csv").open()))
    assert {"mosaic", "S8", "S11"} <= {x["figure"] for x in rows}
    assert not (tmp_path / "out" / "mosaic").exists()
    assert not (tmp_path / "out" / "S8").exists()


def test_s7_and_s8_need_only_valis_but_s4_and_the_mosaic_need_a_comparator(
    unified, tmp_path
):
    """S7 is ONE method before vs after, S8 the per-arm scores: VALIS alone draws both.
    S4 and the mosaic compare methods, so with VALIS alone they stay MISSING."""
    root, plan, conf = unified
    rows = [r for r in csv.DictReader(plan.open()) if r["method"] == "valis"]
    vplan = tmp_path / "valis_plan.csv"
    with open(vplan, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = yaml.safe_load(conf.read_text())
    cfg["S4"]["patient"] = "P0"  # so S7 ("every case except S4's") draws P1
    cfg["S7"].update(field_um=48, zoom_um=0, variants=1, rounds=["CD3"])
    c = tmp_path / "c.yaml"
    c.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "o"
    args = ["--results", str(root), "--plan", str(vplan), "--config", str(c)]
    assert sp.main([*args, "-o", str(out), "--only", "S7,S8"]) == 0
    chk = {r["figure"]: r["status"] for r in csv.DictReader((out / "check.csv").open())}
    assert chk["S7"] == "READY" and chk["S8"] == "READY", chk
    assert chk["S4"] == "MISSING" and chk["mosaic"] == "MISSING", chk
    assert list((out / "S7" / "P1" / "valis_high").glob("v1/P1_valis_high_v1.png"))
    assert (out / "S8" / "valis_high" / "S8_valis_high.png").is_file()


def test_the_cohort_cuts_every_number_and_unquoted_ids_are_refused(unified, tmp_path):
    root, plan, conf = unified
    cfg = yaml.safe_load(conf.read_text())
    cfg["patients"] = ["P1", "P9"]  # P9 is not on disk
    c = tmp_path / "c.yaml"
    c.write_text(yaml.safe_dump(cfg))
    args = ["--results", str(root), "--plan", str(plan), "--config", str(c)]
    assert sp.main([*args, "-o", str(tmp_path / "o"), "--check"]) == 0
    chk = {
        r["figure"]: r for r in csv.DictReader((tmp_path / "o" / "check.csv").open())
    }
    assert chk["cohort"]["status"] == "PARTIAL" and "'P9'" in chk["cohort"]["detail"]
    cfg["patients"] = ["033", 46]  # 046 unquoted: YAML gives the int 38
    c.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="quote every case id"):
        sp.main([*args, "-o", str(tmp_path / "p"), "--check"])


def test_the_cohort_defaults_to_the_arms_samplesheet(unified, tmp_path):
    root, plan, conf = unified
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,channel_1\n046,a.tif,DAPI\n046,b.tif,DAPI\n"
    )
    assert sp.read_input_patients(sheet) == ["046"], "ids stay strings, deduplicated"
    sheet.write_text("patient_id,path_to_file\nP1,a.tif\nP7,b.tif\n")
    args = ["--results", str(root), "--plan", str(plan), "--config", str(conf)]
    assert (
        sp.main([*args, "-o", str(tmp_path / "o"), "--check", "--input", str(sheet)])
        == 0
    )
    chk = {
        r["figure"]: r for r in csv.DictReader((tmp_path / "o" / "check.csv").open())
    }
    d = chk["cohort"]["detail"]
    assert "--input samplesheet" in d and "'P7'" in d, d


def test_s6_renders_every_backend_at_the_same_several_regions(
    tmp_path_factory, tmp_path
):
    """S6: N regions picked ONCE on the reference, every backend x mask drawn there."""
    from benchmarks.tests import test_reg_zoom as tz

    src, _ = tz.seg_run.__wrapped__(tmp_path_factory)
    root = tmp_path / "arm_results"
    root.mkdir()
    backends = ("instantseg", "stardist", "cellsam")
    for m in backends:
        shutil.copytree(src, root / f"seg_{m}")
        seg = root / f"seg_{m}" / "csv" / "segmented.csv"
        seg.write_text(seg.read_text().replace(str(src), str(root / f"seg_{m}")))
    shutil.copytree(src, root / "valis_high_micro2")
    plan = tmp_path / "arm_plan.csv"
    with open(plan, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "arm_kind", "arm", "method", "seg_method", "resume_run"])
        w.writerow(
            ["valis_high_micro2"] * 1
            + ["registration", "valis_high_micro2", "valis", "instantseg", ""]
        )
        for m in backends:
            w.writerow([f"seg_{m}", "segmentation", f"seg_{m}", "seg", m, ""])
    cfg = yaml.safe_load(
        (sp.REPO_ROOT / "benchmarks" / "configs" / "supplementary.yaml").read_text()
    )
    cfg["S6"].update(field_um=20, crop_px=64, regions=3)
    cfg["options"].update(formats="png", dpi=50)
    conf = tmp_path / "s.yaml"
    conf.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "o"
    sp.main(
        [
            "--results",
            str(root),
            "--plan",
            str(plan),
            "--config",
            str(conf),
            "-o",
            str(out),
            "--only",
            "S6",
        ]
    )
    regions = list(csv.DictReader((out / "S6" / "S6_regions.csv").open()))
    assert len(regions) == 3
    assert len({(r["y_px"], r["x_px"]) for r in regions}) == 3
    for k in (1, 2, 3):
        for m in backends:
            for mask in ("nuclei", "cell", "both"):
                assert (
                    out / "S6" / f"r{k}" / f"{m}_{mask}" / "P1_crop.png"
                ).is_file(), (k, m, mask)
    assert (out / "S6" / "S6_nuclei_cell.png").is_file()


def test_s5_draws_the_cost_figure_when_traces_exist(unified, tmp_path):
    """S5 used to pass formats as a STRING to save_fig, which iterates it: 'png' became
    'p', 'n', 'g' and every S5 draw failed. Only reachable with a trace on disk."""
    root, plan, conf = unified
    work = tmp_path / "arm_results"
    shutil.copytree(root, work)
    for arm in ("valis_high_micro2", "tiled_high_s128"):
        t = work / arm / "trace" / "trace.txt"
        t.parent.mkdir(parents=True, exist_ok=True)
        proc = "REGISTER" if arm.startswith("valis") else "TILED_SOLVE"
        t.write_text(
            "task_id\tprocess\ttag\tstatus\texit\tpeak_rss\tpeak_vmem\trealtime\t"
            "duration\tcpus\tstart\tcomplete\n"
            f"1\tMIRAGE:REGISTRATION:{proc}\tP1\tCOMPLETED\t0\t4 GB\t5 GB\t600s\t605s\t2\t"
            "2026-01-01 00:00:00\t2026-01-01 00:10:00\n"
        )
    # a real plan carries the backend and tier columns cost-by-tier keys on
    rows = list(csv.DictReader(plan.open()))
    tiers = {
        "valis_high_micro2": dict(
            registration_method="valis", memory_mode="high", reg_micro_reg="2"
        ),
        "tiled_high_s128": dict(
            registration_method="tiled", reg_tiled_mode="high", reg_tiled_stride="128"
        ),
    }
    cols = list(rows[0]) + [
        "registration_method",
        "memory_mode",
        "reg_micro_reg",
        "reg_tiled_mode",
        "reg_tiled_stride",
    ]
    full = tmp_path / "plan.csv"
    with open(full, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **tiers.get(r["arm"], {})})
    out = tmp_path / "o"
    args = ["--results", str(work), "--plan", str(full), "--config", str(conf)]
    assert sp.main([*args, "-o", str(out), "--only", "S5"]) == 0
    assert list((out / "S5").glob("S5_cost_by_tier_all.png")), sorted(
        p.name for p in (out / "S5").iterdir()
    )


def test_gallery_draws_every_arm_backend_and_channel_on_shared_tissue(
    unified, tmp_path_factory, tmp_path
):
    """The gallery: per case, every registration method's high arm on the SAME crop per
    moving round, every segmentation backend at the SAME regions, every channel of the
    case at those regions too -- nothing picked independently per arm or backend."""
    from benchmarks.tests import test_reg_zoom as tz

    root, plan, conf = unified
    work = tmp_path / "arm_results"
    shutil.copytree(root, work)
    seg_src, _ = tz.seg_run.__wrapped__(tmp_path_factory)
    rows = list(csv.DictReader(plan.open()))
    for m in ("instantseg", "stardist"):
        d = work / f"seg_{m}"
        shutil.copytree(seg_src, d)
        seg = d / "csv" / "segmented.csv"
        seg.write_text(seg.read_text().replace(str(seg_src), str(d)))
        rows.append(
            dict(
                rows[0],
                run_id=f"seg_{m}",
                arm=f"seg_{m}",
                arm_kind="segmentation",
                method="seg",
                seg_method=m,
                resume_run="",
            )
        )
    full = tmp_path / "plan.csv"
    with open(full, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = yaml.safe_load(conf.read_text())
    cfg["gallery"] = {
        "overlay": {"field_um": [48], "zoom_um": 0, "variants": 2},
        "zoom": {
            "field_um": [20],
            "masks": ["both", "nuclei"],
            "regions": 2,
            "crop_px": 64,
        },
        # the mosaic fixture's registered slides are empty stubs; seg_run's are real
        "channels": {
            "field_um": 20,
            "crop_px": 64,
            "autoscale": ["clean"],
            "arm": "seg_instantseg",
        },
    }
    c = tmp_path / "c.yaml"
    c.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "o"
    sp.main(
        [
            "--results",
            str(work),
            "--plan",
            str(full),
            "--config",
            str(c),
            "-o",
            str(out),
            "--only",
            "gallery",
        ]
    )
    g = out / "gallery"
    # overlays: one crop per (round, variant), identical across the methods
    crops = {}
    for j in (g / "overlay").rglob("*_overlay.json"):
        man = json.loads(j.read_text())
        key = (man["round"], j.parent.name)
        crops.setdefault(key, set()).add((man["crop"]["y"], man["crop"]["x"]))
    arms = {p.name for p in (g / "overlay").iterdir()}
    assert (
        arms
        == {
            "valis_high_micro2",
            "tiled_high_gate1",
            "tiled_high_s128",
            "ashlar_t1024_s240",
        }
        or len(arms) >= 3
    ), arms
    assert crops and all(len(v) == 1 for v in crops.values()), crops
    assert {k[1] for k in crops} == {"v1", "v2"}
    # segmentation: every backend x mask at each region, the regions shared
    for k in ("r1", "r2"):
        for m in ("instantseg", "stardist"):
            for mask in ("both", "nuclei"):
                assert list((g / "zoom" / m / f"f20_{mask}" / k).glob("P1_zoom.png"))
                assert list(
                    (g / "crops" / m / f"f20_p64_{mask}" / k).glob("P1_crop.png")
                )
    # channels: every channel the case carries, at the segmentation regions
    chans = sp._slide_channels(work / "seg_instantseg", "P1")
    assert len(chans) >= 2, chans
    for k in ("r1", "r2"):
        got = {
            p.name
            for p in (g / "crops" / "channels" / "f20_p64_clean" / k).glob("*_crop.png")
        }
        assert got == {f"P1_{ch}_crop.png" for ch in chans}, got


def test_a_panels_caption_reads_that_panels_own_manifest(tmp_path):
    """A variant directory holds one render per moving round. The caption under a panel
    must come from the manifest written with THAT panel, not from whichever manifest the
    directory lists first (S4 printed 0.12 in the image and 0.46 under it)."""
    from benchmarks import supplementary as supp

    for rnd, dice in (("A_first", 0.46), ("Z_last", 0.12)):
        (tmp_path / f"046_{rnd}_after.png").write_bytes(b"")
        (tmp_path / f"046_{rnd}_overlay.json").write_text(
            json.dumps(
                {
                    "numbers": {
                        "after": {
                            "dice_matched": dice,
                            "slide_displacement_px": 38.2,
                            "stage": "refined",
                        }
                    }
                }
            )
        )
    note = supp._numbers_note(tmp_path / "046_Z_last_after.png", "after")
    assert note == "Dice = 0.12  Δ = 38.2 px*   [refined]"
    assert supp._numbers_note(tmp_path / "046_missing_after.png", "after") == ""


def test_cell_note_gives_micrometres_when_the_scorer_reported_pixels_only():
    """The manifest backends' scorer has no pixel size and reports px; the figure does."""
    from benchmarks import reg_mosaic as rm

    qc = rm.SegQC(
        stage="refined", dice=0.5, displacement_um=None, displacement_px=10.0, n_pairs=9
    )
    note, vals = rm.cell_note(qc, 0, 0, 100, 0.5, 5)
    assert "5.0 µm" in note and "px" not in note
    assert vals["slide_displacement_um"] == 5.0


def _agreement_frame():
    import pandas as pd

    return pd.DataFrame(
        [
            ("P1", "cellsam", "instantseg", 0.80),
            ("P1", "cellsam", "stardist", 0.60),
            ("P1", "instantseg", "stardist", 0.70),
            ("P2", "cellsam", "instantseg", 0.90),
            ("P2", "cellsam", "stardist", 0.40),
        ],
        columns=["patient_id", "method_a", "method_b", "foreground_dice"],
    )


def test_agreement_matrix_is_the_cohort_median_or_one_patients_own_values():
    """The matrix is symmetric with a unit diagonal; over the cohort it is the median per
    pair, with `patient` it is that section alone, and a pair nobody scored stays NaN
    instead of reading as perfect agreement or as zero."""
    import numpy as np

    from benchmarks import supplementary as sp

    order = ["stardist", "instantseg", "cellsam"]
    mat, n = sp.agreement_matrix(_agreement_frame(), order)
    assert n == 2 and list(mat.index) == order
    assert mat.loc["cellsam", "instantseg"] == pytest.approx(0.85)
    assert mat.loc["instantseg", "cellsam"] == pytest.approx(0.85)
    assert mat.loc["stardist", "cellsam"] == pytest.approx(0.50)
    assert all(mat.loc[m, m] == 1.0 for m in order)

    one, n1 = sp.agreement_matrix(_agreement_frame(), order, patient="P2")
    assert n1 == 1 and one.loc["cellsam", "instantseg"] == pytest.approx(0.90)
    assert np.isnan(one.loc["instantseg", "stardist"])  # P2 has no such row


def test_agreement_heatmap_writes_the_matrix_as_its_own_figure(tmp_path):
    from benchmarks import supplementary as sp

    csv_path = tmp_path / "S6_pairwise_agreement.csv"
    _agreement_frame().to_csv(csv_path, index=False)
    written = sp.agreement_heatmap(csv_path, tmp_path / "heat", formats="png,pdf")
    assert [p.name for p in written] == ["heat.png", "heat.pdf"]
    assert all(p.stat().st_size > 0 for p in written)
    # no usable row: nothing is written rather than an identity matrix posing as data
    assert sp.agreement_heatmap(_agreement_frame().iloc[0:0], tmp_path / "none") == []
    assert not (tmp_path / "none.png").exists()

# benchmarks/tests/regbench/test_slim.py
"""``regbench slim``: keep the CSVs, free the space taken by images and work folders.

Stand-in regbench, arms and sweep trees with a tiny file of every class; the dry run, the
two levels, the ``--collected`` rule for ``csv``, ``SLIMMED.txt``, the refusals (``/``,
``$HOME``, a root of no known layout, a path in the checkout), symlinks never followed, and
what still works afterwards: re-scoring after ``images`` (tables byte-identical, the
tissue block read from the kept reduced planes), and nothing after ``csv`` (``score``
refuses, ``reuse`` says ``slimmed: csv``, the launcher says it re-runs)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from benchmarks.regbench import cli, reuse, score, slimmed
from benchmarks.regbench.paper import collect, schema, slim

REPO = Path(__file__).resolve().parents[3]


def _touch(path, text="x" * 10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _old(root, seconds=10_000):
    """Every file under ``root`` made ``seconds`` older (a collection written after it is
    newer than every run it covers)."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            st = p.stat()
            os.utime(p, (st.st_atime - seconds, st.st_mtime - seconds))


# ── stand-in trees ───────────────────────────────────────────────────────────
def regbench_tree(root):
    d = root / "stare" / "multiplex" / "P1_r1"
    keep_images = {
        "maps": [_touch(d / "warped.npz")],
        "run": [_touch(d / "run.json", json.dumps({"ok": True, "method": "stare"}))],
        "tissue_plane": [_touch(d / "tissue" / "moving_eval.tif")],
        "tissue_cache": [_touch(root / "cache" / "tissue" / "multiplex" / "P1_r1" / "reference_eval.tif"),
                         _touch(root / "cache" / "tissue" / "multiplex" / "P1_r1" / "initial_eval.tif")],
        "records": [_touch(root / "cache" / "scores" / "multiplex" / "P1_r1" / "case.rec"),
                    _touch(d / "stages" / "round.json")],
        "csv": [_touch(root / "tables" / "cells.csv", "a,b\n1,2\n"),
                _touch(root / "logs" / "verdicts.tsv")],
        "tissue_json": [_touch(d / "tissue" / "tissue.json", "{}")],
    }
    images = {
        "native": [_touch(d / "native" / "out.ome.tif", "y" * 100)],
        "stages": [_touch(d / "stages" / "coarse.tif", "y" * 50)],
        "work": [_touch(d / "work" / "scratch.bin", "y" * 70)],
    }
    return keep_images, images


def arms_tree(root):
    arm = root / "tiled_high_s64"
    keep = [
        _touch(arm / "csv" / "registered.csv", "patient_id,id,channels\nP1,P1_a,DAPI|CD8\n"),
        _touch(arm / "P1" / "qc" / "registration" / "P1_a_seg_qc.json",
               json.dumps({"moving": "P1_a", "full_transform": {"dice_matched": 0.8}})),
        _touch(arm / "trace" / "trace.txt", "task_id\tname\n"),
        _touch(arm / "size_logs" / "input_sizes.csv", "a\n"),
    ]
    csv_only_gone = [_touch(arm / "P1" / "geojson" / "cells.geojson"),
                     _touch(arm / "registered" / "transform" / "P1_registrar.pickle")]
    images = [
        _touch(arm / "registered" / "P1_a_registered.ome.tif", "y" * 100),
        _touch(arm / "P1" / "pyramid" / "P1.ome.tiff", "y" * 100),
        _touch(arm / "work" / "ab" / "cdef" / ".command.sh", "y" * 30),
    ]
    return keep, csv_only_gone, images


def sweep_tree(root):
    run = root / "run_001"
    keep = [_touch(run / "trace" / "trace.txt", "task_id\tname\n"),
            _touch(run / "out" / "size_logs" / "input_sizes.csv", "a\n")]
    images = [_touch(run / "out" / "registered" / "P1.ome.tif", "y" * 40),
              _touch(run / "work" / "12" / "3456" / "x.tif", "y" * 40)]
    return keep, images


def _paths(removals):
    return {Path(r.path) for r in removals}


# ── plan / dry run ───────────────────────────────────────────────────────────
def test_images_level_lists_exactly_the_image_and_work_classes(tmp_path):
    root = tmp_path / "out"
    keep, images = regbench_tree(root)
    plan = slim.plan(root, "images")
    got = _paths(plan)
    assert got == {images["native"][0], images["stages"][0], images["work"][0].parent}
    assert {r.kind for r in plan} == {"native images", "stage images", "method work"}
    work = next(r for r in plan if r.kind == "method work")
    assert work.is_dir and work.bytes == 70
    for paths in keep.values():
        for p in paths:
            assert p not in got and not any(p.is_relative_to(g) for g in got), p


def test_the_default_is_a_dry_run_and_deletes_nothing(tmp_path, capsys):
    root = tmp_path / "out"
    keep, images = regbench_tree(root)
    before = sorted(str(p) for p in root.rglob("*"))
    summary = slim.run(root, "images", apply=False)
    assert summary["applied"] is False and summary["bytes"] == 220
    assert sorted(str(p) for p in root.rglob("*")) == before
    assert not (root / slim.SLIMMED).exists()
    assert cli.main(["slim", "--root", str(root), "--level", "images"]) == 0
    said = capsys.readouterr().out
    assert "dry run" in said and "native images" in said and "--apply" in said
    assert sorted(str(p) for p in root.rglob("*")) == before


def test_apply_deletes_and_writes_slimmed_txt(tmp_path):
    root = tmp_path / "out"
    keep, images = regbench_tree(root)
    assert cli.main(["slim", "--root", str(root), "--level", "images", "--apply"]) == 0
    for paths in images.values():
        for p in paths:
            assert not p.exists(), p
    for paths in keep.values():
        for p in paths:
            assert p.exists(), p
    text = (root / slim.SLIMMED).read_text()
    assert "level: images" in text and "bytes freed: 220" in text
    assert "overlay" in text  # what can no longer be done
    assert slimmed.level(root) == "images"


def test_arms_and_sweep_roots_are_recognised_by_their_layout(tmp_path):
    arms = tmp_path / "arms"
    keep, csv_only, images = arms_tree(arms)
    assert slim.layout(arms) == "arms"
    got = _paths(slim.plan(arms, "images"))
    assert got == {images[0], images[1], arms / "tiled_high_s64" / "work"}
    sweep = tmp_path / "sweep"
    skeep, simages = sweep_tree(sweep)
    assert slim.layout(sweep) == "sweep"
    got = _paths(slim.plan(sweep, "images"))
    assert got == {simages[0], sweep / "run_001" / "work"}
    assert not (_paths(slim.plan(sweep, "images")) & set(skeep))


# ── the csv level ────────────────────────────────────────────────────────────
def test_csv_without_a_collection_is_refused_naming_the_rule(tmp_path):
    root = tmp_path / "arms"
    arms_tree(root)
    with pytest.raises(SystemExit, match="--collected"):
        slim.plan(root, "csv")
    with pytest.raises(SystemExit, match="--collected"):
        slim.run(root, "csv", apply=True)
    assert (root / "tiled_high_s64" / "work").exists()


def test_csv_with_a_stale_or_empty_collection_is_refused(tmp_path):
    root = tmp_path / "arms"
    arms_tree(root)
    # a collection of another root: no row for this one
    other = tmp_path / "other_arms"
    arms_tree(other)
    elsewhere = tmp_path / "c_other"
    collect.run(elsewhere, other, None, None, [], None)
    with pytest.raises(SystemExit, match="row"):
        slim.plan(root, "csv", collected=elsewhere)
    # a collection older than a run it covers
    c = tmp_path / "c"
    collect.run(c, root, None, None, [], None)
    _old(c)
    with pytest.raises(SystemExit, match="newer"):
        slim.plan(root, "csv", collected=c)


def test_csv_with_a_collection_keeps_only_the_csv_level_set(tmp_path):
    root = tmp_path / "arms"
    keep, csv_only, images = arms_tree(root)
    _old(root)
    c = tmp_path / "c"
    counts = collect.run(c, root, None, None, [], None)
    assert counts["per_pair"] > 0
    summary = slim.run(root, "csv", apply=True, collected=c)
    assert summary["applied"]
    left = {p for p in root.rglob("*") if p.is_file()}
    assert left == set(keep) | {root / slim.SLIMMED}
    text = (root / slim.SLIMMED).read_text()
    assert "level: csv" in text and "re-scor" in text and str(c) in text


def test_csv_on_a_regbench_root_keeps_run_json_and_the_csvs(tmp_path):
    root = tmp_path / "out"
    keep, images = regbench_tree(root)
    _old(root)
    c = tmp_path / "c"
    collect.run(c, None, None, None, [], None)
    # a collection that read nothing of this root has no row of it
    with pytest.raises(SystemExit, match="row"):
        slim.plan(root, "csv", collected=c)
    # its own tables/ folder, newer than every run, with rows
    tables = root / "tables"
    _touch(tables / "cells.csv", "a,b\n1,2\n")
    slim.run(root, "csv", apply=True, collected=tables)
    left = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert left == {"stare/multiplex/P1_r1/run.json", "tables/cells.csv", "logs/verdicts.tsv",
                    slim.SLIMMED}
    assert slimmed.level(root) == "csv"


# ── safety ───────────────────────────────────────────────────────────────────
def test_a_symlink_out_of_root_is_never_followed(tmp_path):
    outside = tmp_path / "precious"
    big = _touch(outside / "slide.ome.tif", "z" * 500)
    _touch(outside / "work" / "keep.tif", "z" * 5)
    root = tmp_path / "out"
    regbench_tree(root)
    d = root / "stare" / "multiplex" / "P1_r1"
    os.symlink(big, d / "native" / "linked.ome.tif")
    os.symlink(outside, d / "stages" / "linked_dir")
    d2 = root / "stare" / "multiplex" / "P1_r2"
    _touch(d2 / "run.json", "{}")
    os.symlink(outside / "work", d2 / "work")  # a method work folder that is a link
    plan = slim.plan(root, "images")
    for r in plan:
        assert not Path(r.path).is_symlink()
        assert Path(r.path).resolve().is_relative_to(root.resolve())
    slim.run(root, "images", apply=True)
    assert big.read_text() == "z" * 500 and (outside / "work" / "keep.tif").exists()
    assert (d / "native" / "linked.ome.tif").is_symlink() and (d2 / "work").is_symlink()


def test_slash_home_unknown_layouts_and_the_checkout_are_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="refus"):
        slim.plan(Path("/"), "images")
    home = tmp_path / "home"
    regbench_tree(home)
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(SystemExit, match="HOME"):
        slim.plan(home, "images")
    with pytest.raises(SystemExit, match="HOME"):  # an ancestor of $HOME
        slim.plan(tmp_path, "images")
    monkeypatch.undo()
    plain = tmp_path / "plain"
    _touch(plain / "photo.tif")
    with pytest.raises(SystemExit, match="layout"):
        slim.plan(plain, "images")
    with pytest.raises(SystemExit, match="checkout"):
        slim.plan(REPO / "benchmarks", "images")
    with pytest.raises(SystemExit, match="checkout"):  # a root holding the checkout
        slim.plan(REPO.parent, "images")
    assert (plain / "photo.tif").exists()


def test_apply_alone_deletes(tmp_path):
    root = tmp_path / "out"
    _, images = regbench_tree(root)
    cli.main(["slim", "--root", str(root), "--level", "images"])
    assert images["native"][0].exists()
    cli.main(["slim", "--root", str(root), "--level", "images", "--apply"])
    assert not images["native"][0].exists()


# ── what still works ─────────────────────────────────────────────────────────
def _tables_bytes(out):
    return {p.name: p.read_bytes() for p in sorted((Path(out) / "tables").glob("*"))
            if p.is_file()}


def test_rescoring_after_images_gives_the_same_tables(tmp_path):
    from . import scored_cohort

    root, out, _ = scored_cohort.build(tmp_path)
    score.score(root, out, **scored_cohort.SCORE_KW)
    first = _tables_bytes(out)
    for d in sorted(Path(out).glob("*/multiplex/*")):
        if (d / "run.json").exists():
            _touch(d / "native" / "written.ome.tif", "y" * 64)
            _touch(d / "work" / "scratch.tif", "y" * 8)
    summary = slim.run(out, "images", apply=True)
    assert summary["bytes"] > 0 and not list(Path(out).glob("*/multiplex/*/native/*.tif"))
    score.score(root, out, fresh=True, **scored_cohort.SCORE_KW)
    assert _tables_bytes(out) == first


def test_the_tissue_block_is_rescored_from_the_kept_reduced_planes(tmp_path_factory):
    pytest.importorskip("cv2")
    from benchmarks.regbench import methods, tissue
    from benchmarks.regbench.cases import result_dir

    from . import test_tissue as tt

    root, case, ref, _ = _unwrap(tt.slide_case)(tmp_path_factory)
    out = tmp_path_factory.mktemp("slim_tissue_out")
    methods.run_case(tt._writer(lambda _: tt._stack(ref)), case, out, {"workers": 1})
    first = score.score(root, out, datasets=["multiplex"])
    d = result_dir(out, "drawn", case)
    native = json.loads((d / "run.json").read_text())["native"]["files"][0]
    assert Path(native).exists()
    slim.run(out, "images", apply=True)
    assert not Path(native).exists()
    assert (d / "tissue" / tissue.MOVING_PLANE).exists()
    (d / "tissue" / tissue.RESULT).unlink()  # the pieces are computed again, from the plane
    again = score.score(root, out, datasets=["multiplex"])
    assert [(r["variant"], r["qcs"]) for r in again["tissue"]] == [
        (r["variant"], r["qcs"]) for r in first["tissue"]]
    assert {r["variant"] for r in again["tissue"]} >= {"initial", "drawn"}


def test_score_refuses_a_csv_slimmed_root(tmp_path):
    out = tmp_path / "out"
    _touch(out / slim.SLIMMED, "level: csv\n")
    with pytest.raises(SystemExit, match="slimmed: csv"):
        score.score(tmp_path / "cases", out)
    with pytest.raises(SystemExit, match="slimmed: csv"):
        score.score_case(tmp_path / "cases", out, "multiplex", case_id="P1_r1")


def test_pair_code_does_not_name_the_slimmed_check():
    import inspect

    for name in score.PAIR_CODE_MEMBERS:
        assert "SLIMMED" not in inspect.getsource(getattr(score, name)), name
    src = inspect.getsource(score.score)
    assert "slimmed" in src.split('"""', 2)[2][:400]  # at the entry of score()


def test_reuse_reports_a_slimmed_run_as_such(tmp_path, monkeypatch):
    out = tmp_path / "out"
    d = out / "stare" / "multiplex" / "P1_r1"
    _touch(d / "run.json", json.dumps({"ok": True, "method": "stare", "group_cases": ["P1_r1"]}))
    monkeypatch.setattr(reuse, "load_case", lambda p: type("C", (), {"case_id": "P1_r1"})())
    monkeypatch.setattr(reuse, "_key_now", lambda *a: ({"k": 1}, ""))
    monkeypatch.setattr(reuse.methods, "record_gaps", lambda rec, case, path, key: ["record"])
    (row,) = reuse.report(tmp_path / "cases", out)
    assert "slimmed" not in row["detail"]
    _touch(out / slim.SLIMMED, "level: csv\n")
    (row,) = reuse.report(tmp_path / "cases", out)
    assert row["reusable"] == "no" and "slimmed: csv" in row["detail"]
    assert "warped.npz is missing" not in row["detail"]


def test_the_launcher_says_a_csv_slimmed_folder_is_run_again(tmp_path):
    from . import test_submit_cohort as tsc

    site = _unwrap(tsc.site)(_Factory(tmp_path))
    out = site[0] / "out-slimmed"
    _touch(out / slim.SLIMMED, "level: csv\n")
    done, _, _ = tsc.launch(site, log="slimmed")
    assert done.returncode == 0, done.stderr
    said = done.stdout
    assert "slimmed: csv" in said and "runs again" in said
    shutil.rmtree(out)
    plain, _, _ = tsc.launch(site, log="slimmed")
    assert "SLIMMED.txt" not in plain.stdout and "slimmed: " not in plain.stdout
    # the same jobs: what decides a reuse is the run's own check, not the launcher
    assert tsc._sbatch_lines(said) == tsc._sbatch_lines(plain.stdout)
    _touch(out / slim.SLIMMED, "level: csv\n")
    refused, _, _ = tsc.launch(site, "--score-only", env={"REGBENCH_SUBMISSION": "s1"},
                              log="slimmed")
    assert refused.returncode != 0 and "can no longer be scored" in refused.stdout + refused.stderr


def _unwrap(fixture):
    """The function under a pytest fixture (``__wrapped__``; pytest 8.4 names it
    otherwise)."""
    got = getattr(fixture, "__wrapped__", None)
    return got if got is not None else fixture._get_wrapped_function()


class _Factory:
    def __init__(self, base):
        self.base = Path(base)

    def mktemp(self, name):
        p = self.base / name
        p.mkdir()
        return p


def test_the_slimmed_file_name_is_one_constant():
    assert slim.SLIMMED == slimmed.NAME == "SLIMMED.txt"
    assert schema.TABLES  # the plots read these and nothing else (test_draft_plots.py)

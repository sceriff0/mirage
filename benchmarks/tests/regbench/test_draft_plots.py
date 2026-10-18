# benchmarks/tests/regbench/test_draft_plots.py
"""Draft plots from a collection: ``regbench plots --tables <dir> --to <dir> [--draft]``.

A collection (``paper/collect.py``) of stand-in arms is drawn: one file set per plot, each
named ``_draft`` and stamped ``DRAFT — interim data (<sources>)``; a plot whose inputs are
all ``NA`` is a placeholder naming the table and columns; the overlay zooms, which need
images, are placeholders naming what is missing; and after ``slim --level csv`` of the
arms root the same plots come out of the CSVs alone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from benchmarks.regbench import cli  # noqa: E402
from benchmarks.regbench.paper import (  # noqa: E402
    collect,
    overlays,
    plots,
    schema,
    slim,
)

from .test_slim import _old  # noqa: E402

PANELS = ("DAPI_CD8_CD3", "DAPI_CD4_FOXP3")


def _arm(root, arm, dice):
    d = root / arm
    (d / "csv").mkdir(parents=True, exist_ok=True)
    lines = ["patient_id,id,channels"]
    for p in range(6):
        pid = f"P{p}"
        for k, panel in enumerate(PANELS):
            mid = f"{pid}_{panel}"
            lines.append(f"{pid},{mid},{panel.replace('_', '|')}")
            q = d / pid / "qc" / "registration"
            q.mkdir(parents=True, exist_ok=True)
            (q / f"{mid}_seg_qc.json").write_text(json.dumps(
                {"moving": mid, "full_transform": {"dice_matched": dice - 0.01 * p - 0.02 * k}}))
    (d / "csv" / "registered.csv").write_text("\n".join(lines) + "\n")
    (d / "P0" / "registered").mkdir(parents=True, exist_ok=True)
    (d / "P0" / "registered" / "P0.ome.tif").write_text("y" * 100)


@pytest.fixture
def collection(tmp_path):
    arms = tmp_path / "arms"
    _arm(arms, "tiled_high_s64", 0.85)
    _arm(arms, "tiled_medium_s64", 0.80)
    _arm(arms, "valis_high_micro2", 0.70)
    _old(arms)
    c = tmp_path / "collected"
    collect.run(c, arms, None, None, [], None)
    return arms, c


def _drawn(lines):
    return {ln.split("\t")[0]: ln.split("\t")[1] for ln in lines if "\t" in ln}


def test_every_plot_is_written_once_as_a_draft(collection, tmp_path, capsys):
    arms, c = collection
    to = tmp_path / "figs"
    assert cli.main(["plots", "--tables", str(c), "--to", str(to)]) == 0
    said = _drawn(capsys.readouterr().out.splitlines())
    names = [*plots.PLOTS, *overlays.OVERLAYS]
    assert set(said) == set(names)
    for name in names:
        assert (to / f"{name}_draft.png").is_file(), name
        assert (to / f"{name}_draft_source_data.csv").is_file(), name
        assert not (to / f"{name}.png").exists(), name
    stamp = "DRAFT — interim data (arms)"
    for name in names:
        assert stamp in (to / f"{name}_draft.svg").read_text(), name
    assert said["fig3b_dice_best"] == "drawn"


def test_an_all_na_plot_is_a_placeholder_naming_its_table_and_columns(collection, tmp_path):
    _, c = collection
    to = tmp_path / "figs"
    got = plots.render_tables(c, to)
    # the arms carry no displacement: every displacement_um_p50 is NA
    res = got["fig3c_displacement_best"]
    assert res["placeholder"]
    src = (to / "fig3c_displacement_best_draft_source_data.csv").read_text()
    assert "per_pair" in src and "displacement_um_p50" in src
    # a table with no row at all is named too
    res = got["figS_cpu_scaling"]
    assert res["placeholder"]
    assert "cpu_scaling" in (to / "figS_cpu_scaling_draft_source_data.csv").read_text()


def test_overlays_without_their_images_are_placeholders_naming_them(collection, tmp_path):
    _, c = collection
    to = tmp_path / "figs"
    got = plots.render_tables(c, to)
    for name in overlays.OVERLAYS:
        assert got[name]["placeholder"]
        src = (to / f"{name}_draft_source_data.csv").read_text()
        assert "warped.npz" in src and "native" in src


def test_plots_read_only_the_schema_csvs(collection, tmp_path, monkeypatch):
    _, c = collection
    (c / "not_a_table.csv").write_text("a\n1\n")
    opened = []
    real = plots._read

    def spy(path):
        opened.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(plots, "_read", spy)
    plots.render_tables(c, tmp_path / "figs")
    assert opened and {n.removesuffix(".csv") for n in opened} <= set(schema.TABLES)


def test_a_scorer_tables_folder_is_drawn_without_draft_unless_asked(collection, tmp_path):
    _, c = collection
    plain = tmp_path / "plain"
    plain.mkdir()
    for name in schema.TABLES:
        (plain / f"{name}.csv").write_bytes((c / f"{name}.csv").read_bytes())
    got = plots.render_tables(plain, tmp_path / "f1")
    assert all("_draft" not in Path(r["files"][0]).name for r in got.values())
    got = plots.render_tables(plain, tmp_path / "f2", draft=True)
    assert all(Path(r["files"][0]).stem.endswith("_draft") for r in got.values())


def test_plots_render_after_the_source_roots_are_slimmed_to_csv(collection, tmp_path):
    arms, c = collection
    before = plots.render_tables(c, tmp_path / "before")
    slim.run(arms, "csv", apply=True, collected=c)
    assert not list(arms.rglob("*.ome.tif")) and list(arms.rglob("*_seg_qc.json"))
    after = plots.render_tables(c, tmp_path / "after")
    assert {k: bool(v.get("placeholder")) for k, v in after.items()} == {
        k: bool(v.get("placeholder")) for k, v in before.items()}
    for name in plots.PLOTS:
        a = (tmp_path / "after" / f"{name}_draft_source_data.csv").read_text()
        b = (tmp_path / "before" / f"{name}_draft_source_data.csv").read_text()
        assert a == b, name

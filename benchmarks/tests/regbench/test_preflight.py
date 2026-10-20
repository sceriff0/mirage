"""`regbench preflight`: what each image is asked before anything is submitted."""

from __future__ import annotations

import sys
import types

import pytest

from benchmarks.regbench import cli, methods, preflight, variants

pytest.importorskip("stare")


def test_every_method_of_the_run_list_has_its_package_named():
    used = {r.method for r in variants.RUNS}
    assert used <= set(preflight.PACKAGE)
    # the harness reads every image through STARE's reader, in every image
    assert "stare.slide_io" in preflight.COMMON and "stare.slide_io" in preflight.SCORER
    assert {"zarr", "imagecodecs", "tifffile", "numpy"} <= set(preflight.COMMON)
    # the version each adapter enforces is the one the preflight compares with
    for method in ("ashlar", "valis", "valis_mirage", "deeperhistreg", "palom"):
        assert methods.load(method).VERSION


def test_an_environment_that_holds_what_the_run_needs_passes(tmp_path, capsys):
    notes, failures = preflight.check("stare", [tmp_path / "out"])
    assert failures == []
    assert any(n.startswith("numpy from ") for n in notes)
    assert f"writable: {tmp_path / 'out'}" in notes
    code = cli.main(["preflight", "--method", "stare", "--image", "STARE_REGBENCH"])
    said = capsys.readouterr()
    assert code == 0 and said.out.splitlines()[-1] == (
        "preflight [STARE_REGBENCH stare] ok"
    )


def test_a_package_that_is_not_there_is_one_line(capsys):
    """This machine has no ASHLAR: its adapter's imports fail one by one, and nothing
    else is reported as failing."""
    pytest.importorskip("numpy")
    try:
        import ashlar  # noqa: F401

        pytest.skip("ashlar is installed here")
    except ImportError:
        pass
    notes, failures = preflight.check("ashlar")
    assert [f.split(":")[0] for f in failures] == [
        "cannot import ashlar",
        "cannot import benchmarks.ashlar.original",
        "cannot import benchmarks.ashlar.retile",
    ][: len(failures)] and failures[0].startswith("cannot import ashlar: ")
    code = cli.main(["preflight", "--method", "ashlar", "--image", "ASHLAR"])
    said = capsys.readouterr()
    assert code == 1
    assert said.err.splitlines() == [
        f"preflight FAIL [ASHLAR ashlar] {f}" for f in failures
    ]
    assert said.out.splitlines()[-1] == (
        f"preflight [ASHLAR ashlar] {len(failures)} failure(s)"
    )


def test_the_version_the_adapter_enforces_is_surfaced_and_not_weakened(monkeypatch):
    """ASHLAR's version comes from versioneer: whatever the installed package prints is
    reported beside what the adapter runs, and the adapter's own check is as it was."""
    fake = types.ModuleType("ashlar")
    fake.__file__ = "/image/site-packages/ashlar/__init__.py"
    fake.__version__ = "1.20.0+3.gabc1234"
    monkeypatch.setitem(sys.modules, "ashlar", fake)
    for name in preflight.NEEDS["ashlar"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    notes, failures = preflight.check("ashlar")
    assert len(failures) == 1
    assert "ashlar says its version is '1.20.0+3.gabc1234'" in failures[0]
    assert "runs '1.20.0' and nothing else" in failures[0]
    fake.__version__ = methods.load("ashlar").VERSION
    notes, failures = preflight.check("ashlar")
    assert (
        failures == [] and "ashlar 1.20.0 (the version the adapter enforces)" in notes
    )
    assert methods.load("ashlar").VERSION == "1.20.0"


def test_numpy_from_the_users_own_site_packages_is_a_failure(monkeypatch):
    import numpy

    home = str(__import__("pathlib").Path(numpy.__file__).resolve().parents[1])
    monkeypatch.setattr(preflight, "_user_sites", lambda: [home])
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    notes, failures = preflight.check("stare")
    (line,) = [f for f in failures if f.startswith("numpy resolves OUTSIDE the image")]
    assert numpy.__file__ in line and "PYTHONNOUSERSITE is 'not set' here" in line
    assert preflight.outside_the_image("") is False


def test_a_folder_that_cannot_be_written_is_a_failure(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        notes, failures = preflight.check("stare", [locked, tmp_path / "fine"])
    finally:
        locked.chmod(0o700)
    assert len(failures) == 1 and failures[0].startswith(
        f"{locked} is not visible or not writable from inside the image"
    )
    assert f"writable: {tmp_path / 'fine'}" in notes


def test_slides_that_cannot_be_read_inside_the_image_are_a_failure(tmp_path, capsys):
    """The prepared case of the dataset: a slide folder that is not bound into the image
    would fail every unit of every run, and is said before anything is submitted."""
    import json
    from dataclasses import asdict

    from benchmarks.regbench.cases import Case

    there = tmp_path / "slide.ome.tif"
    there.write_bytes(b"x")
    d = tmp_path / "cases" / "multiplex" / "c1"
    d.mkdir(parents=True)

    def prepared(mov):
        case = Case(
            dataset="multiplex", case_id="c1", group="g", ref_nuclear="", mov_nuclear="",
            ref_image=str(there), mov_image=mov, modality="fluorescence", diagonal=1.0,
        )  # fmt: skip
        (d / "case.json").write_text(json.dumps({**asdict(case), "dir": str(d)}))

    prepared("/not/bound/slides/r2.ome.tif")
    notes, failures = preflight.unreadable_slides(tmp_path / "cases", ["multiplex"])
    assert notes == []
    (failure,) = failures
    assert failure.startswith("multiplex/c1: its image file(s) cannot be read from")
    assert "/not/bound/slides/r2.ome.tif" in failure and "REGBENCH_BIND" in failure
    assert str(there) not in failure
    argv = ["preflight", "--method", "stare", "--image", "STARE", "--cases",
            str(tmp_path / "cases"), "--dataset", "multiplex"]  # fmt: skip
    assert cli.main(argv) == 1
    said = capsys.readouterr()
    assert "preflight FAIL [STARE stare] multiplex/c1: its image file(s)" in said.err
    # the same case with both slides readable: a note, and the command passes
    prepared(str(there))
    notes, failures = preflight.unreadable_slides(tmp_path / "cases", ["multiplex"])
    assert notes == ["readable: the 2 image file(s) of multiplex/c1"] and not failures
    assert cli.main(argv) == 0
    # no cases named: nothing of the kind is asked
    assert not [n for n in preflight.check("stare")[0] if n.startswith("readable")]


def test_the_scorers_image_is_checked_for_what_the_scorer_imports(monkeypatch, capsys):
    """`--scorer` asks for the scorer's list and for nothing of a method: a module of
    that list that does not import is one FAIL line naming it and exit 1; a list that
    imports whole is `ok` and exit 0."""
    # what the scorer's image must hold (prepare, gate, tissue with OpenCV, score)
    assert set(preflight.SCORER) == {
        "numpy", "scipy", "skimage", "tifffile", "zarr", "stare.slide_io", "cv2"
    }  # fmt: skip
    argv = ["preflight", "--scorer", "--image", "STARE_REGBENCH"]
    missing = "regbench_no_such_module"
    monkeypatch.setattr(preflight, "SCORER", ("numpy", missing, "tifffile"))
    notes, failures = preflight.check(scorer=True)
    assert len(failures) == 1
    assert failures[0].startswith(f"cannot import {missing}: ModuleNotFoundError")
    assert cli.main(argv) == 1
    said = capsys.readouterr()
    assert said.err.splitlines() == [
        f"preflight FAIL [STARE_REGBENCH the scorer] {failures[0]}"
    ]
    assert said.out.splitlines()[-1] == (
        "preflight [STARE_REGBENCH the scorer] 1 failure(s)"
    )
    # nothing of a method is asked of the scorer's image: no package, no version
    monkeypatch.setattr(preflight, "SCORER", ("numpy", "tifffile"))
    notes, failures = preflight.check("ashlar", scorer=True)
    assert failures == [] and not [n for n in notes if "ashlar" in n]
    assert cli.main(argv) == 0
    said = capsys.readouterr()
    assert said.err == ""
    assert said.out.splitlines()[-1] == "preflight [STARE_REGBENCH the scorer] ok"


def test_a_stare_older_than_the_pin_is_a_failure(monkeypatch):
    """2026-10-20 on the cluster: bolt3x/mirage-stare-regbench:1.0.0 had been pushed on
    2026-10-11 with STARE 1.2.1, and requirements/stare.txt pins 1.3.0 (whose `--prep`
    the adapter passes). Every STARE unit failed with "unrecognized arguments: --prep
    none" while the preflight said ok. methods/stare.py is frozen and declares no
    VERSION, so the preflight compares STARE with the ONE pin, requirements/stare.txt."""
    pin = preflight.pinned_stare()
    assert pin == "1.3.0"
    for name in ("stare", *preflight.NEEDS["stare"], "stare.slide_io"):
        fake = types.ModuleType(name)
        fake.__file__ = f"/image/site-packages/{name.replace('.', '/')}.py"
        monkeypatch.setitem(sys.modules, name, fake)
    sys.modules["stare"].__version__ = "1.2.1"
    notes, failures = preflight.check("stare")
    (line,) = failures
    assert "stare says its version is '1.2.1'" in line
    assert "requirements/stare.txt pins '1.3.0'" in line
    sys.modules["stare"].__version__ = pin
    notes, failures = preflight.check("stare")
    assert failures == [] and "stare 1.3.0 (the version requirements/stare.txt pins)" in notes

"""The synthetic dataset's suites and its known-truth refusal. (The generator's own tests
moved with it to ``test_scene.py``.)"""

import argparse
import json

import pytest

from benchmarks.regbench import truth
from benchmarks.regbench.datasets import synthetic


def test_scale_suite_runs_one_deformation_from_1024_to_65536():
    suite = synthetic.SUITES["scale"]
    names = sorted(suite)
    assert [suite[k]["n"] for k in names] == [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
    ]
    deform = {
        tuple((k, str(v)) for k, v in s.items() if k not in ("n",))
        for s in suite.values()
    }
    assert len(deform) == 1  # only the size differs
    # a huge slide is described without allocating it
    scene = synthetic.Scene({**suite["scale_65536"], "n": 65536})
    assert scene.mask.shape == (2048, 2048) and len(scene.ref[0]) > 2_000_000
    assert scene.tile("mov", 40000, 50000, 40256, 50256).shape == (256, 256)


def _prepared(root, case_id, sha):
    d = root / "synthetic" / case_id
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps({"case_id": case_id, "extra": {"truth_spec_sha256": sha}})
    )
    return d


@pytest.mark.parametrize("force", [False, True])
def test_a_case_of_another_truth_spec_is_refused_not_reused(tmp_path, force):
    """A cases folder that holds a known-truth case of another `truth_spec.json`:
    `prepare` neither keeps it as done nor writes over it, with or without --force."""
    current = truth.load_spec()[truth.HASH_KEY]
    old = _prepared(tmp_path, "blob_grid_rot", "a" * 64)
    before = (old / "case.json").read_text()
    # the array task whose case this is: without the refusal it would find the case
    # there and call it done (or, forced, write the current spec's case over it)
    index = sorted(synthetic.truth_suite()).index("blob_grid_rot")
    args = argparse.Namespace(
        cases=str(tmp_path), suite="truth", index=index, workers=1, force=force
    )
    with pytest.raises(SystemExit) as refused:
        synthetic.prepare(args)
    said = str(refused.value)
    assert str(old) in said and "aaaaaaaaaaaa" in said and current[:12] in said
    assert "new, empty cases folder" in said and "REGBENCH_CASES" in said
    assert "Nothing was prepared." in said
    assert (old / "case.json").read_text() == before
    assert sorted(p.name for p in old.iterdir()) == ["case.json"]  # no image written


def test_cases_of_the_current_spec_and_cases_of_none_are_not_refused(tmp_path):
    current = truth.load_spec()[truth.HASH_KEY]
    done = _prepared(tmp_path, "blob_grid_rot", current)
    _prepared(tmp_path, "smoke_0", "")  # a suite that is not the frozen one
    (tmp_path / "synthetic" / "half").mkdir()  # a case folder without a record yet
    synthetic.refuse_other_spec(tmp_path)
    synthetic.refuse_other_spec(tmp_path / "nothing_there")
    # and a case of the current spec that is there is still not made twice
    spec = synthetic.truth_suite()["blob_grid_rot"]
    assert synthetic.prepare_one(tmp_path, "blob_grid_rot", spec) == done
    assert sorted(p.name for p in done.iterdir()) == ["case.json"]

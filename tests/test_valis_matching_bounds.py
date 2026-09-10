"""REGISTER's peak is SuperGlue matching, and two bounds on it must actually land.

Measured 2026-09-10 (tma profile, 134 patients x 5 cores, ~2800 px each): every REGISTER
task was OOM-killed at "Matching images 0/10", climbing the 32/64/96/128 GB ramp and dying at
128 GB with a 96 GiB JVM heap it never touched. The pixels were tiny; the cost was not pixels.
VALIS 1.0.0 caps keypoints at `feature_detectors.MAX_FEATURES = 20000`, SuperGlue's attention
and Sinkhorn tensors are quadratic in that count, and serial_rigid.py matches every pair at
once on `multiprocessing.cpu_count() - 1` threads -- the NODE's core count, not the task's.

bin/utils/valis_config.py applies two bounds:

  1. `MAX_KEYPOINTS` (5000) is written into `feature_detectors.MAX_FEATURES` BEFORE
     `MEMORY_PRESETS` constructs its `SuperGlueMatcher()` instances, because the matcher
     copies the value into its config in `__init__` (feature_matcher.py:1007). Applied after,
     the detector would honour 5000 while the matcher still budgeted for 20000.
  2. `bound_cpu_count(task.cpus)` rebinds `multiprocessing.cpu_count` so the six VALIS
     pools that size themselves on it see the allocation register.nf passes as `--cpus`.

Both are checked against a FAKE valis whose feature_detectors / feature_matcher mirror the
1.0.0 source shapes that matter (a module constant, a function with a bound default, a
matcher that snapshots the constant at construction). No valis install is needed.
"""

from __future__ import annotations

import multiprocessing
import re
import sys
import types
from pathlib import Path

import pytest

from tests.nfmodel import processes

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_MAX_FEATURES = 20000


@pytest.fixture
def fake_valis(monkeypatch):
    """Install a valis package that reproduces the three shapes the cap must reach."""
    fd = types.ModuleType("valis.feature_detectors")
    fd.MAX_FEATURES = UPSTREAM_MAX_FEATURES

    def filter_features(kp, desc, n_keep=fd.MAX_FEATURES):  # default bound at def time
        return kp[:n_keep], desc[:n_keep]

    fd.filter_features = filter_features
    fd.SuperPointFD = type("SuperPointFD", (), {})

    fm = types.ModuleType("valis.feature_matcher")

    class SuperGlueMatcher:
        def __init__(self):
            # valis_lib/feature_matcher.py:1007 -- copied at construction.
            self.config = {"superpoint": {"max_keypoints": fd.MAX_FEATURES}}

    fm.SuperGlueMatcher = SuperGlueMatcher

    micro = types.ModuleType("valis.micro_rigid_registrar")
    micro.MicroRigidRegistrar = type("MicroRigidRegistrar", (), {})
    nonrigid = types.ModuleType("valis.non_rigid_registrars")
    nonrigid.OpticalFlowWarper = type("OpticalFlowWarper", (), {})

    pkg = types.ModuleType("valis")
    pkg.feature_detectors, pkg.feature_matcher = fd, fm
    pkg.micro_rigid_registrar, pkg.non_rigid_registrars = micro, nonrigid
    for name, mod in {
        "valis": pkg,
        "valis.feature_detectors": fd,
        "valis.feature_matcher": fm,
        "valis.micro_rigid_registrar": micro,
        "valis.non_rigid_registrars": nonrigid,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "valis_config", raising=False)
    monkeypatch.syspath_prepend(str(ROOT / "bin" / "utils"))
    return pkg


@pytest.fixture
def valis_config(fake_valis):
    import valis_config as vc

    return vc


def test_the_cap_is_a_real_reduction(valis_config):
    assert 0 < valis_config.MAX_KEYPOINTS < UPSTREAM_MAX_FEATURES


def test_the_cap_reaches_the_module_constant(valis_config, fake_valis):
    assert fake_valis.feature_detectors.MAX_FEATURES == valis_config.MAX_KEYPOINTS


def test_the_cap_reaches_filter_features_bound_default(valis_config, fake_valis):
    """`filter_features(kp, desc, n_keep=MAX_FEATURES)` froze 20000 into its default at
    definition time; the OpenCV-detector paths call it with no argument."""
    kp, desc = fake_valis.feature_detectors.filter_features(
        list(range(UPSTREAM_MAX_FEATURES)), list(range(UPSTREAM_MAX_FEATURES))
    )
    assert len(kp) == len(desc) == valis_config.MAX_KEYPOINTS


@pytest.mark.parametrize("tier", ["high", "medium", "low"])
def test_the_cap_was_applied_before_the_matchers_were_built(valis_config, tier):
    """The assertion with teeth: the matcher instance in the preset row snapshotted the
    capped value, which is only true if the cap ran before MEMORY_PRESETS was built."""
    matcher = valis_config.MEMORY_PRESETS[tier]["matcher"]
    assert matcher.config["superpoint"]["max_keypoints"] == valis_config.MAX_KEYPOINTS
    assert valis_config.MEMORY_PRESETS[tier]["num_features"] == valis_config.MAX_KEYPOINTS


def test_bound_cpu_count_makes_valis_see_the_allocation(valis_config, monkeypatch):
    monkeypatch.setattr(multiprocessing, "cpu_count", multiprocessing.cpu_count)
    assert valis_config.bound_cpu_count(8) == 8
    assert multiprocessing.cpu_count() == 8


def test_bound_cpu_count_never_drops_below_two(valis_config, monkeypatch):
    """VALIS sizes its pools as cpu_count() - 1; joblib refuses n_jobs=0."""
    monkeypatch.setattr(multiprocessing, "cpu_count", multiprocessing.cpu_count)
    assert valis_config.bound_cpu_count(1) == 2
    assert multiprocessing.cpu_count() - 1 >= 1


def test_bound_cpu_count_falls_back_to_the_affinity_mask(valis_config, monkeypatch):
    monkeypatch.setattr(multiprocessing, "cpu_count", multiprocessing.cpu_count)
    n = valis_config.bound_cpu_count(None)
    assert n >= 2
    assert multiprocessing.cpu_count() == n


def test_register_nf_passes_task_cpus_to_register_py():
    """The bound is only as good as its input: REGISTER must hand task.cpus over."""
    body = processes()["REGISTER"].script_body
    assert re.search(r"--cpus \$\{task\.cpus\}", body), (
        "modules/local/register.nf no longer passes `--cpus ${task.cpus}` to register.py, "
        "so VALIS sizes its matching pool on the node's core count again"
    )


def test_register_py_accepts_the_flag_and_applies_it_before_the_registrar():
    src = (ROOT / "bin" / "register.py").read_text()
    assert '"--cpus"' in src, "register.py lost its --cpus argument"
    bound_at = src.index("bound_cpu_count(n_cpus)")
    built_at = src.index("registrar_kwargs = build_registrar_kwargs(")
    assert bound_at < built_at, (
        "bound_cpu_count must run before the registrar is built -- VALIS reads cpu_count() "
        "when its pools start, and the first pool is the rigid matcher inside register()"
    )

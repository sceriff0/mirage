"""Guard against the BioFormats-JVM crash on a read-only $HOME.

On clusters where $HOME lives on a read-only mount (e.g. /hpcnfs), any stage that
starts the BioFormats JVM (e.g. warp_seg_qc via valis_config.init_jvm) crashes:

    OSError: [Errno 30] Read-only file system: '/hpcnfs'
      ... jgo.resolve_dependencies -> os.makedirs(workspace)

The pinned scyjava (<1.11) derives its jgo ``cache_dir`` / ``m2_repo``
from ``pathlib.Path.home()`` at import time (scyjava/config.py) and passes them
*explicitly* into ``jgo.resolve_dependencies`` (scyjava/_jvm.py). It NEVER reads the
``JGO_CACHE_DIR`` / ``M2_REPO`` environment variables, so the Dockerfile ENV knobs are
inert. With $HOME on a read-only mount jgo's ``os.makedirs($HOME/.jgo)`` dies with EROFS.

The timing-immune lever is ``scyjava.config.set_cache_dir`` / ``set_m2_repo`` — they
overwrite the frozen module globals that ``get_cache_dir()`` returns. This is a static
behavioural test (scyjava.config is faked) so it runs in CI without a valis install.
"""

import os
import sys
import types
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parent.parent / "bin" / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))


class _FakeConfig:
    def __init__(self):
        self.cache_dir = None
        self.m2_repo = None

    def set_cache_dir(self, d):
        self.cache_dir = d

    def set_m2_repo(self, d):
        self.m2_repo = d


@pytest.fixture
def fake_scyjava(monkeypatch):
    """Inject a fake ``scyjava.config`` that records set_cache_dir / set_m2_repo calls."""
    fake = _FakeConfig()
    scyjava_pkg = types.ModuleType("scyjava")
    config_mod = types.ModuleType("scyjava.config")
    config_mod.set_cache_dir = fake.set_cache_dir
    config_mod.set_m2_repo = fake.set_m2_repo
    scyjava_pkg.config = config_mod
    monkeypatch.setitem(sys.modules, "scyjava", scyjava_pkg)
    monkeypatch.setitem(sys.modules, "scyjava.config", config_mod)
    return fake


def _fn():
    import jvm_cache  # noqa: WPS433  (imported lazily so the fake scyjava is in place)

    return jvm_cache.point_jvm_cache_off_readonly_home


def test_writable_home_leaves_scyjava_untouched(monkeypatch, fake_scyjava):
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/home/dev")
    monkeypatch.setattr(os, "access", lambda p, m: True)
    assert _fn()() is None
    assert fake_scyjava.cache_dir is None
    assert fake_scyjava.m2_repo is None


def test_readonly_home_prefers_baked_root_cache(monkeypatch, fake_scyjava):
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/hpcnfs/home")
    monkeypatch.setattr(os, "access", lambda p, m: False)
    monkeypatch.delenv("MIRAGE_JVM_HOME", raising=False)
    monkeypatch.setattr(
        os.path, "isdir", lambda p: p in ("/root/.jgo", "/root/.m2/repository")
    )
    chosen = _fn()()
    assert chosen == "/root/.jgo"
    assert fake_scyjava.cache_dir == "/root/.jgo"
    assert fake_scyjava.m2_repo == "/root/.m2/repository"


def test_mirage_jvm_home_override_wins(monkeypatch, fake_scyjava):
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/hpcnfs/home")
    monkeypatch.setattr(os, "access", lambda p, m: False)
    monkeypatch.setenv("MIRAGE_JVM_HOME", "/opt/jvmcache")
    monkeypatch.setattr(
        os.path,
        "isdir",
        lambda p: p in ("/opt/jvmcache/.jgo", "/opt/jvmcache/.m2/repository"),
    )
    chosen = _fn()()
    assert chosen == "/opt/jvmcache/.jgo"
    assert fake_scyjava.cache_dir == "/opt/jvmcache/.jgo"
    assert fake_scyjava.m2_repo == "/opt/jvmcache/.m2/repository"


def test_readonly_home_no_baked_cache_falls_back_to_writable_scratch(
    monkeypatch, fake_scyjava, tmp_path
):
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/hpcnfs/home")
    monkeypatch.setattr(os, "access", lambda p, m: False)
    monkeypatch.delenv("MIRAGE_JVM_HOME", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(os.path, "isdir", lambda p: False)  # nothing baked anywhere
    made = []
    monkeypatch.setattr(os, "makedirs", lambda p, **k: made.append(p))
    chosen = _fn()()
    expected = os.path.join(str(tmp_path), "mirage_jvm", ".jgo")
    assert chosen == expected
    assert fake_scyjava.cache_dir == expected
    assert expected in made  # scratch cache dir is actually created

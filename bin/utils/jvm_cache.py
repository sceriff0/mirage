"""Redirect scyjava's BioFormats-JVM (jgo/Maven) cache off a read-only $HOME.

Import-light on purpose (no ``valis`` import) so the guard is unit-testable in CI without a
valis install, and so it can be called from any entrypoint before the JVM starts.

Why this exists
---------------
Two real consumers pin scyjava for two different reasons. The VALIS runtime path
(``bin/utils/valis_config.py``, running in the external ``cdgatenbee/valis-wsi`` image) pins
``scyjava<1.11``: that scyjava computes its jgo ``cache_dir`` and Maven ``m2_repo`` from
``pathlib.Path.home()`` at *import time* and passes them **explicitly** into
``jgo.resolve_dependencies`` (in scyjava's config and _jvm modules). It never consults the
``JGO_CACHE_DIR`` / ``M2_REPO`` environment variables — so setting those (e.g. the Dockerfile ENV)
has no effect on where the JVM cache lands. On an HPC compute node ``$HOME`` (e.g.
``/hpcnfs/...``) is a read-only mount, so jgo's first-run ``os.makedirs($HOME/.jgo)`` dies with
``OSError [Errno 30] Read-only file system`` before the JVM can start — the REG_PREP /
REG_MICRO_PREP / REG_FINALIZE crash.

The convert path (``bin/utils/ome_io._open_bioio`` / ``bin/convert_image.py``,
``containers/convert``) runs scyjava 1.12.5 instead, where
``scyjava.config.set_cache_dir``/``set_m2_repo`` are verified present — confirmed by the Task 13
build — so the same timing-immune lever below applies to both consumers despite the version gap.

The timing-immune lever is ``scyjava.config.set_cache_dir`` / ``set_m2_repo``: they overwrite the
frozen module globals that ``get_cache_dir()`` returns, regardless of when scyjava was imported.
Call :func:`point_jvm_cache_off_readonly_home` after valis/scyjava are importable and *before*
``registration.init_jvm``.
"""

import os


def point_jvm_cache_off_readonly_home():
    """Point scyjava's jgo cache at a usable location when ``$HOME`` is read-only.

    No-op when ``$HOME`` is writable (local/dev, or an image with a writable ``$HOME`` cache), so
    those paths keep scyjava's defaults untouched.

    Preference order when ``$HOME`` is read-only:

    1. A **warm baked cache** shipped in the image (``/root/.jgo`` + ``/root/.m2``, made
       world-readable at build). jgo short-circuits on an existing ``buildSuccess`` workspace, so
       pointing scyjava here needs no writes and no network — the offline-node golden path.
       Override the search root with ``$MIRAGE_JVM_HOME``.
    2. **Node-local writable scratch** (``$TMPDIR``, set by SLURM/Nextflow; else ``/tmp``). jgo will
       (re)build its workspace here; that needs the BioFormats jars reachable via a baked m2 repo
       (preferred, kept read-only) or the network. Only helps if the jars are available.

    Returns the chosen jgo cache dir (str), or ``None`` when no redirect was needed/possible.
    """
    try:
        from scyjava import config as sjconf
    except Exception:
        # scyjava missing or its layout changed — leave defaults; the JVM start will fail loudly.
        return None

    home = os.path.expanduser("~")
    if os.access(home, os.W_OK):
        return (
            None  # writable $HOME — leave scyjava's Path.home()-derived defaults alone
        )

    # 1) prefer a warm baked cache (exists -> jgo reuses it: no writes, no network)
    for root in (os.environ.get("MIRAGE_JVM_HOME", "").strip(), "/root"):
        if root and os.path.isdir(os.path.join(root, ".jgo")):
            cache_dir = os.path.join(root, ".jgo")
            sjconf.set_cache_dir(cache_dir)
            m2_repo = os.path.join(root, ".m2", "repository")
            if os.path.isdir(m2_repo):
                sjconf.set_m2_repo(m2_repo)
            return cache_dir

    # 2) fall back to node-local writable scratch (jgo rebuilds the workspace here)
    scratch = os.environ.get("TMPDIR") or "/tmp"
    cache_dir = os.path.join(scratch, "mirage_jvm", ".jgo")
    os.makedirs(cache_dir, exist_ok=True)
    sjconf.set_cache_dir(cache_dir)

    # Keep the readable baked m2 jars if present so the rebuild can run offline; else a fresh
    # scratch m2 (which needs network to populate).
    baked_m2 = "/root/.m2/repository"
    if os.path.isdir(baked_m2):
        sjconf.set_m2_repo(baked_m2)
    else:
        m2_repo = os.path.join(scratch, "mirage_jvm", ".m2", "repository")
        os.makedirs(m2_repo, exist_ok=True)
        sjconf.set_m2_repo(m2_repo)

    return cache_dir

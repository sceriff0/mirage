"""Redirect scyjava's BioFormats-JVM (jgo/Maven) cache off a read-only $HOME.

No ``valis`` import, so it is testable without valis and callable before the JVM starts.

scyjava derives its jgo cache and Maven repo from ``Path.home()`` at import time and ignores
``JGO_CACHE_DIR`` / ``M2_REPO``. On a read-only cluster ``$HOME``, jgo's first
``os.makedirs($HOME/.jgo)`` fails with ``OSError [Errno 30] Read-only file system``.
``scyjava.config.set_cache_dir`` / ``set_m2_repo`` overwrite those values whenever scyjava
was imported. Used by the VALIS path (valis_config.py) and the convert path
(ome_io._open_bioio, convert_image.py). Call before the JVM starts.
"""

import os


def point_jvm_cache_off_readonly_home():
    """Point scyjava's jgo cache at a usable location when ``$HOME`` is read-only.

    No-op when ``$HOME`` is writable (local/dev, or an image with a writable ``$HOME`` cache), so
    those paths keep scyjava's defaults untouched.

    Preference order when ``$HOME`` is read-only:

    1. A warm cache baked into the image (``/root/.jgo`` + ``/root/.m2``): jgo reuses an
       existing workspace, so no writes and no network. ``$MIRAGE_JVM_HOME`` overrides the root.
    2. Node-local scratch (``$TMPDIR``, else ``/tmp``), where jgo rebuilds its workspace; this
       needs the jars from a baked m2 repo or the network.

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

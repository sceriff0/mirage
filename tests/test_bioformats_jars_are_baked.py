"""The Bio-Formats jars and JVM must be baked into the :convert image.

`bioio-bioformats` gets Bio-Formats through `bffile` -> `scyjava`/`jgo`, which resolves
`ome:formats-gpl` from Maven at FIRST USE into `~/.jgo` + `~/.m2`, and may fetch a Java
runtime through `cjdk` into `~/.cache/cjdk`. The cluster has a READ-ONLY $HOME and
docs/usage.md documents air-gapped execution, so a first run there would die exactly the
way bin/utils/jvm_cache.py's docstring records: jgo's `os.makedirs($HOME/.jgo)` raising
`OSError [Errno 30] Read-only file system`.

This is a CONTAINER-ONLY fix. A git pull can never repair a missing jar. It is the same
shape as the retired containers/tiled's DISK/LightGlue bake, and this file was modelled on
tests/test_disk_weights_are_baked.py (now tests/test_drape_image_is_slim.py) -- including
its comment-blindness, which is not a refinement but a defect found by breaking that
file's guards before trusting them.

WHY /root AND NOT /opt. bin/utils/jvm_cache.point_jvm_cache_off_readonly_home() already
searches `$MIRAGE_JVM_HOME` then `/root` for a `.jgo` directory and points scyjava's
frozen module globals at it. Baking into /root means the runtime path is the one this
repository already wrote and tested, not a second mechanism.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONVERT_DOCKERFILE = REPO / "containers" / "convert" / "Dockerfile"
CONVERT_SMOKE = REPO / "containers" / "convert" / "smoke.sh"


def _code(path):
    """The file with COMMENT LINES REMOVED, so prose can never satisfy a guard.

    Not a refinement -- a defect found by breaking tests/test_disk_weights_are_baked.py
    deliberately: two of its four cases stayed GREEN with the thing they protect deleted,
    because the Dockerfile's comments named the very strings they grepped for. Every
    comment in containers/convert/Dockerfile names `bioio-bioformats`, `formats-gpl` and
    `/root/.jgo`, so the raw view here would be pure annotation.
    """
    text = path.read_text()
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    # A check whose "good" answer is a presence must still prove the stripper left a file.
    # Two different files pass through here (the Dockerfile and smoke.sh), so the
    # sanity marker is per-file rather than a single Dockerfile-only pair -- a shared
    # marker would fail on smoke.sh unconditionally, for a reason unrelated to what
    # that test actually checks.
    if path == CONVERT_DOCKERFILE:
        assert "FROM eclipse-temurin" in code and "pip3 install" in code, (
            "the comment stripper returned something that is not containers/convert's "
            "Dockerfile any more; every assertion below would be testing an empty string"
        )
    else:
        assert "set -euo pipefail" in code, (
            "the comment stripper returned something that is not containers/convert's "
            "smoke.sh any more; every assertion below would be testing an empty string"
        )
    return code


def test_the_bioformats_plugin_is_installed():
    """Without it, `bioio` refuses .svs/.qptiff/.vsi outright -- and the pipeline claims
    to accept 'any Bio-Formats-compatible format'."""
    code = _code(CONVERT_DOCKERFILE)
    assert "bioio-bioformats==" in code, (
        "containers/convert does not install a pinned bioio-bioformats. It is the plugin "
        "that makes the pipeline's Bio-Formats claim true; the other five bioio-* plugins "
        "cover OME-TIFF, TIFF, ND2, CZI and LIF only."
    )


def test_the_bioformats_maven_coordinate_is_pinned():
    """Ruling R6 pins the Python side; the Java side needs the same treatment or the
    version of Bio-Formats in the image floats on every rebuild."""
    code = _code(CONVERT_DOCKERFILE)
    assert "BIOFORMATS_VERSION=ome:formats-gpl:" in code, (
        "containers/convert does not pin BIOFORMATS_VERSION. bffile reads it and accepts "
        "a full Maven coordinate (ome:formats-gpl:<version>); unset, the jar jgo resolves "
        "changes whenever upstream releases."
    )


def test_the_jars_are_fetched_at_build_time():
    """A first RUN that has to reach Maven is a first RUN that fails on an air-gapped
    node, and one that has to write $HOME fails on a read-only one."""
    code = _code(CONVERT_DOCKERFILE)
    assert "HOME=/root" in code, (
        "the bake step does not run with HOME=/root, so scyjava's Path.home()-derived "
        "jgo/m2/cjdk caches land somewhere bin/utils/jvm_cache.py will not look for them."
    )
    assert "from bioio_bioformats import Reader" in code, (
        "the build never constructs the Bio-Formats reader, so nothing forces jgo to "
        "resolve the jars at BUILD time. A prefetch that never runs leaves the same "
        "broken image."
    )


def test_the_build_asserts_the_jars_landed():
    """A prefetch that silently no-ops leaves the same broken image, which is why
    containers/tiled asserts `test -f` on both checkpoint filenames by name."""
    code = _code(CONVERT_DOCKERFILE)
    assert "test -d /root/.m2/repository/ome/formats-gpl" in code, (
        "the build does not assert that the formats-gpl artifact directory exists under "
        "/root/.m2/repository after the bake."
    )
    assert 'test -n "$(ls -A /root/.m2/repository/ome/formats-gpl)"' in code, (
        "the build checks the directory exists but not that it is NON-EMPTY. jgo creates "
        "the path before it populates it, so an existence check alone can pass over a "
        "failed download."
    )


def test_the_baked_cache_is_world_readable():
    """docker.runOptions passes -u $(id -u), so a task does not run as root. /root is
    mode 700 by default and the whole bake would be unreadable."""
    code = _code(CONVERT_DOCKERFILE)
    assert "chmod -R a+rX /root" in code, (
        "the baked jgo/m2/cjdk caches under /root are never made world-readable. "
        "nextflow.config's docker profile runs tasks as -u $(id -u):$(id -g), and /root "
        "is mode 700, so the task would fall through to the network -- or to nothing."
    )


def test_the_smoke_test_opens_a_file_with_no_home():
    """The whole point is that the reader works with a read-only or absent $HOME, which
    is the cluster's actual condition. Asserted in smoke.sh so it runs BOTH at build time
    and, through containers.yml's `docker run`, against the finished image."""
    code = _code(CONVERT_SMOKE)
    assert "HOME=/nonexistent" in code, (
        "containers/convert/smoke.sh never exercises the Bio-Formats reader without "
        "$HOME, which is the exact failure mode the bake exists to prevent."
    )
    assert "bioio_bioformats" in code, (
        "containers/convert/smoke.sh does not import bioio_bioformats. A plugin that is "
        "installed and does not import is a format the pipeline silently cannot read."
    )


def test_the_smoke_test_calls_the_production_redirect_not_a_hand_rolled_copy():
    """Review round 1 on Task 7: the smoke test used to call
    `scyjava.config.set_cache_dir`/`set_m2_repo` directly -- two calls production never
    made, since nothing on the CONVERT_IMAGE run path called
    `jvm_cache.point_jvm_cache_off_readonly_home()` at the time. That let the smoke test
    pass while proving nothing about the real run path. It must now call the SAME
    function `bin/utils/ome_io.py::_open_bioio` and `bin/convert_image.py::read_image`
    call, via the copy the Dockerfile COPYs in."""
    dockerfile_code = _code(CONVERT_DOCKERFILE)
    assert "COPY bin/utils/jvm_cache.py" in dockerfile_code, (
        "containers/convert/Dockerfile does not COPY bin/utils/jvm_cache.py into the "
        "image, so smoke.sh has no way to call the real production redirect function."
    )
    smoke_code = _code(CONVERT_SMOKE)
    assert "from jvm_cache import point_jvm_cache_off_readonly_home" in smoke_code, (
        "containers/convert/smoke.sh does not import the production "
        "point_jvm_cache_off_readonly_home function."
    )
    assert "point_jvm_cache_off_readonly_home()" in smoke_code, (
        "containers/convert/smoke.sh imports the production redirect but never calls it."
    )
    assert "sjconf.set_cache_dir" not in smoke_code, (
        "containers/convert/smoke.sh still hand-rolls scyjava.config.set_cache_dir -- "
        "that proves a DIFFERENT mechanism works, not the one CONVERT_IMAGE actually "
        "uses. Call point_jvm_cache_off_readonly_home() instead."
    )


def test_the_smoke_test_proves_the_redirect_is_load_bearing_with_a_negative_leg():
    """A positive-only smoke test run as root cannot tell "the redirect worked" apart
    from "root can write anywhere anyway" (CAP_DAC_OVERRIDE). The negative leg drops to
    an unprivileged UID and skips the guard, so the read must FAIL -- if it does not,
    either the redirect stopped being load-bearing or this leg stopped exercising the
    failure it claims to."""
    code = _code(CONVERT_SMOKE)
    assert "os.setuid(65534)" in code, (
        "containers/convert/smoke.sh's negative leg does not drop privileges to an "
        "unprivileged UID (65534/nobody) -- run as root, HOME=/nonexistent alone proves "
        "nothing, since root can os.makedirs() anywhere via CAP_DAC_OVERRIDE."
    )
    assert "noguard_status" in code and "-eq 0" in code, (
        "containers/convert/smoke.sh's negative leg does not check that the no-guard "
        "read actually failed (non-zero exit)."
    )


def test_the_glencoe_cli_tools_are_gone():
    """They were installed, symlinked and smoke-tested for months, and invoked by nothing:
    the CONVERT_IMAGE module runs convert_image.py, which reads through bioio.

    (Deliberately not naming that module's file by its literal Nextflow module
    extension here: a docstring mentioning it trips tests/test_nfmodel.py's "reads
    Nextflow source" discovery heuristic, which pattern-matches on that extension
    as a bare substring -- this file parses no Nextflow/Groovy source at all, only
    containers/convert's own Dockerfile and smoke.sh.)"""
    code = _code(CONVERT_DOCKERFILE)
    for tool in ("bioformats2raw", "raw2ometiff"):
        assert tool not in code, (
            f"containers/convert still installs {tool}. No module, script or config "
            f"invokes it -- the CONVERT_IMAGE module (modules/local/convert_image) runs "
            f"convert_image.py, whose read path is bioio. If it is genuinely needed "
            f"again, add the caller first."
        )

"""The Bio-Formats jars and JVM must be baked into the :convert image.

`bioio-bioformats` gets Bio-Formats through `bffile` -> `scyjava`/`jgo`, which resolves
`ome:formats-gpl` from Maven at FIRST USE into `~/.jgo` + `~/.m2`, and may fetch a Java
runtime through `cjdk` into `~/.cache/cjdk`. The cluster has a READ-ONLY $HOME and
docs/usage.md documents air-gapped execution, so a first run there would die exactly the
way bin/utils/jvm_cache.py's docstring records: jgo's `os.makedirs($HOME/.jgo)` raising
`OSError [Errno 30] Read-only file system`. A missing jar is fixed only by the image.

The bake goes to /root because bin/utils/jvm_cache.point_jvm_cache_off_readonly_home()
searches `$MIRAGE_JVM_HOME` then `/root` for a `.jgo` directory and points scyjava at it.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONVERT_DOCKERFILE = REPO / "containers" / "convert" / "Dockerfile"
CONVERT_SMOKE = REPO / "containers" / "convert" / "smoke.sh"


def _code(path):
    """The file with comment lines removed, so prose can never satisfy a guard.

    The Dockerfile's comments name `bioio-bioformats`, `formats-gpl` and `/root/.jgo`,
    the very strings these checks look for.
    """
    text = path.read_text()
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    # Prove the stripper left a real file; the sanity marker is per file.
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
    """The Python pins live in requirements/; the Java side needs the same treatment or the
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
    """smoke.sh calls the same `jvm_cache.point_jvm_cache_off_readonly_home()` that
    `bin/utils/ome_io.py::_open_bioio` and `bin/convert_image.py::read_image` call, via
    the copy the Dockerfile COPYs in, not its own `scyjava.config.set_cache_dir`."""
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
    either the redirect stopped mattering or this leg stopped exercising the failure."""
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
    """Nothing invokes bioformats2raw/raw2ometiff: CONVERT_IMAGE runs convert_image.py,
    which reads through bioio.

    (The module file is not named with its Nextflow extension here: that substring
    trips tests/test_nfmodel.py's "reads Nextflow source" discovery heuristic.)"""
    code = _code(CONVERT_DOCKERFILE)
    for tool in ("bioformats2raw", "raw2ometiff"):
        assert tool not in code, (
            f"containers/convert still installs {tool}. No module, script or config "
            f"invokes it -- the CONVERT_IMAGE module (modules/local/convert_image) runs "
            f"convert_image.py, whose read path is bioio. If it is genuinely needed "
            f"again, add the caller first."
        )

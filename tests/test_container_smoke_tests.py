#!/usr/bin/env python3
"""Guard: every container image carries a real smoke test, and it is wired up twice.

WHAT THIS IS FOR. Until 2026-09-01 an image could stop building, or build and then
fail on its first import, and nothing in CI would say so: `build-images.yml` ran on a
push to main touching `containers/**` and on a manual dispatch, and on nothing else.
`containers/regqc` carried a `pip install git+.../cudipy.git` for months after that
GitHub repository ceased to exist -- the image could not be built AT ALL -- and the
suite stayed green throughout. CI redesign Phase 6 closed that with three mechanisms:
a changed-only build on every PR, a full nightly matrix, and a per-image
`containers/<dir>/smoke.sh` that is RUN BY THE DOCKERFILE and again by
`.github/workflows/containers.yml` against the built image.

The smoke script is the part that rots silently, so it is the part guarded here:

  1. every image in containers/images.json has one;
  2. every Dockerfile COPYs and RUNs ITS OWN (a copy/paste that names a sibling's
     script would still build, and would assert the wrong stack);
  3. containers.yml runs it against the built image;
  4. it is not empty ceremony -- it must invoke Python and name at least two
     third-party modules; and
  5. EVERY MODULE IT IMPORTS IS ACTUALLY INSTALLED BY THAT IMAGE.

(5) is the one that earns this file. The smoke scripts assert imports, and a
mistyped module name (`skimage` vs `scikit-image`, `instanseg` vs `instanseg-torch`,
`cellSAM` vs `cellsam`) turns a smoke test into a build failure that only a real
container build -- twenty minutes on a runner, and on the release path a half-published
image set -- can discover. Comparing the imported names against what the Dockerfile and
its requirements files actually install catches that here, in a second, with no Docker.

It is a static check and it says so: it cannot prove an import SUCCEEDS, only that the
package providing it was asked for. The build is what proves the rest.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# The shared workflow/composite-action resolver, imported rather than re-derived --
# the rule this repo adopted after seven guards each carried a private parse of the
# same files, each with a different blind spot.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
ci_actions = importlib.import_module("ci_actions")

REPO = Path(__file__).resolve().parent.parent
CONTAINERS = REPO / "containers"
IMAGES_JSON = CONTAINERS / "images.json"
WORKFLOW = REPO / ".github" / "workflows" / "containers.yml"

# Where the Dockerfile puts the script, and therefore what containers.yml runs.
IN_IMAGE_PATH = "/usr/local/bin/mirage-smoke.sh"

# Import name -> the distribution name pip is asked for. Only the ones that DIFFER
# need an entry; anything absent is assumed to be its own distribution, which is true
# for numpy, scipy, pandas, tifffile, zarr, numcodecs, imagecodecs, torch, kornia,
# csbdeep, stardist, spatialdata, anndata, geopandas, shapely, xmltodict and matplotlib.
# `bioio_bioformats` underscores where its distribution hyphenates, and `scyjava` arrives
# transitively through bioio-bioformats -> bffile. bin/utils/jvm_cache.py (which
# containers/convert/smoke.sh imports -- see LOCALLY_COPIED_MODULES below) imports
# scyjava.config directly (it is the timing-immune lever for the jgo cache), so
# containers/convert names scyjava in its own pip line rather than relying on the
# transitive install -- an install nothing in the Dockerfile names is invisible to
# test_container_harmonisation.py too.
IMPORT_TO_DISTRIBUTION = {
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "instanseg": "instanseg-torch",
    "PIL": "pillow",
    "bioio_bioformats": "bioio-bioformats",
    "scyjava": "scyjava",
}

# {container: {import name}} for a module a smoke script imports that is NOT a pip
# distribution at all -- it is a first-party bin/utils/*.py file the Dockerfile COPYs
# into the image directly, so `declared_distributions` (a pip-install scan) can never
# find it. containers/convert is the one case today: smoke.sh calls the PRODUCTION
# `bin/utils/jvm_cache.py::point_jvm_cache_off_readonly_home()` (Task 7 review round 1)
# rather than hand-rolling its own scyjava.config calls, and the Dockerfile COPYs that
# one file onto its own PYTHONPATH entry for exactly that reason -- see
# containers/convert/Dockerfile's "COPY bin/utils/jvm_cache.py" comment.
# `test_locally_copied_modules_are_actually_copied_by_name` below is what keeps this
# honest: an entry here that the Dockerfile does not literally COPY is a hard failure,
# not a silent permanent exemption.
LOCALLY_COPIED_MODULES = {
    "convert": {"jvm_cache"},
}

# {container: {import name: repo-relative package dir}} for a FIRST-PARTY PACKAGE a smoke
# script imports -- repository code the Dockerfile COPYs as a directory and pip-installs
# from that path, so `declared_distributions` (which reads requirement tokens and skips
# path installs) cannot see it either. containers/tiled is the case: `stare` is the STARE
# method itself (packages/stare), which bin/tiled_*.py shim over.
# `test_locally_installed_packages_are_actually_installed_by_name` keeps an entry honest
# the same way LOCALLY_COPIED_MODULES' meta-test does: the Dockerfile must literally COPY
# that directory AND pip-install the destination, and smoke.sh must actually import it.
LOCALLY_INSTALLED_PACKAGES = {
    "tiled": {"stare": "packages/stare"},
}


def _entries() -> list[dict]:
    return json.loads(IMAGES_JSON.read_text())


def _dirs() -> list[str]:
    return [e["dir"] for e in _entries()]


def smoke_path(name: str) -> Path:
    return CONTAINERS / name / "smoke.sh"


def dockerfile(name: str) -> Path:
    return CONTAINERS / name / "Dockerfile"


def _strip_hash_comments(text: str) -> str:
    """Drop whole-line `#` comments.

    Both a Dockerfile and a shell script comment with `#`, and both of these scans
    would otherwise be satisfied by prose: the requirement scan by a comment naming a
    package the image does not install, and the import scan by a comment naming a
    module the script does not import. A trailing `#` is left alone -- Dockerfiles do
    not use it and stripping it would corrupt a URL fragment.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


# Trailing-comment stripping lives in the shared resolver, not here -- see
# ci_actions.strip_shell_comments, and this repo's rule about seven guards each
# carrying a private parse of the same files.
_shell_commands_only = ci_actions.strip_shell_comments


# AN IMPORT MUST START A STATEMENT. Both patterns are matched against a chunk that
# has already been cut at every newline, `;` and quote character, and both are
# anchored -- so `import numpy, scipy` counts and the words "can import pyplot",
# which appear INSIDE a print() string in containers/segeval/smoke.sh, do not.
#
# That is not a hypothetical: an unanchored first draft of this scan reported that
# containers/segeval installs no distribution called `pyplot`. Prose satisfying a
# text-matching guard is this repository's most repeated defect, and a string
# literal is prose the shell-comment stripper cannot see.
#
# Cutting at quotes is also what makes the scan work at all on the three scripts
# whose Python is written on ONE line inside `"$PY" -c "..."` -- a `^\s*import`
# form over raw lines matched nothing whatsoever in containers/tiled/smoke.sh, which
# this file's own "imports at least two modules" case caught on its first run.
_CHUNK_SPLIT_RE = re.compile(r"""[\n;"']""")
_FROM_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\b")
_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*)")

# Python's own modules, which no image installs.
_STDLIB = {"os", "sys", "json", "pathlib", "argparse", "logging", "time", "csv", "glob"}


def imported_modules(script: str) -> set[str]:
    """Top-level module names the smoke script imports, comment-blind.

    The script's Python lives inside a `"$PY" -c "..."` string, so this is a scan and
    not a parse. Shell comments are stripped first, so a comment naming a module
    cannot satisfy the requirement check below; see _CHUNK_SPLIT_RE for the string
    literals, which it cannot strip.
    """
    names: set[str] = set()
    for chunk in _CHUNK_SPLIT_RE.split(_strip_hash_comments(script)):
        match = _FROM_RE.match(chunk)
        if match:
            names.add(match.group(1).split(".")[0])
            continue
        match = _IMPORT_RE.match(chunk)
        if match:
            for token in match.group(1).split(","):
                names.add(token.strip().split(".")[0])
    return {n for n in names if n and n not in _STDLIB}


# A pip requirement token: `name`, `name==1.2`, `name[extra]>=1.2`, quoted or not.
_REQ_TOKEN_RE = re.compile(
    r"^[\"']?([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?(?:[<>=!~].*)?[\"']?$"
)


def _tokens_from_pip_line(chunk: str) -> set[str]:
    found: set[str] = set()
    for raw in chunk.split():
        if raw.startswith("-") or raw in {"&&", "\\", "|"}:
            continue
        if raw.startswith("git+") or "github.com" in raw:
            # `git+https://github.com/vanvalenlab/cellSAM.git@<sha>` installs `cellSAM`.
            repo = raw.rstrip("/").split("/")[-1]
            found.add(repo.split(".git")[0].split("@")[0])
            continue
        match = _REQ_TOKEN_RE.match(raw)
        if match:
            found.add(match.group(1))
    return found


def declared_distributions(name: str) -> set[str]:
    """Distributions this image asks pip for, from its Dockerfile AND the
    requirements/ files that Dockerfile COPYs.

    Both sources are needed and neither is enough: containers/tiled installs
    everything through `requirements/tiled.txt`, containers/merge names every pin
    inline, and containers/regqc does both.
    """
    # Join Dockerfile line continuations FIRST, so one `RUN pip install a \` + `b`
    # is one logical line. Scanning the raw text instead is how a first draft came
    # back claiming this image installs distributions called `run`, `copy` and `env`.
    text = _strip_hash_comments(dockerfile(name).read_text()).replace("\\\n", " ")
    declared: set[str] = set()

    # Inline pins: everything after `pip install` on each logical line, split on `&&`
    # first so that `pip install --upgrade pip && pip install numpy` contributes two
    # invocations rather than one whose tail swallows the second command's own words.
    for line in text.splitlines():
        for command in line.split("&&"):
            match = re.search(r"\bpip[0-9]*\s+(?:-m\s+pip\s+)?install\b(.*)$", command)
            if match:
                declared |= _tokens_from_pip_line(match.group(1))

    # Plus everything in every requirements file the Dockerfile COPYs -- EXCEPT
    # constraints.txt, and that exclusion is what gives this check its teeth.
    # A constraints file does not install anything; it caps what a resolve may pick.
    # requirements/constraints.txt is the harmonised table and names nearly every
    # package in the repository, so counting it as "declared" would mean almost any
    # module name passed, including a wrong one. Nine of the eleven Dockerfiles COPY
    # it, so this is the difference between a real check and a formality.
    for req in sorted(set(re.findall(r"requirements/[A-Za-z0-9._-]+\.txt", text))):
        if Path(req).name == "constraints.txt":
            continue
        path = REPO / req
        if not path.is_file():
            continue
        for line in _strip_hash_comments(path.read_text()).splitlines():
            token = line.split("#")[0].strip()
            if not token or token.startswith("-"):
                continue
            declared |= _tokens_from_pip_line(token)

    return {d.lower() for d in declared}


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _dirs())
def test_every_image_has_a_smoke_script(name):
    assert smoke_path(name).is_file(), (
        f"containers/{name}/ has no smoke.sh. Every image carries one: it is what turns "
        "'the layers built' into 'the stack imports', and containers.yml runs it against "
        "the built image on a pull request that touched this directory."
    )


@pytest.mark.parametrize("name", _dirs())
def test_the_dockerfile_runs_its_own_smoke_script(name):
    """A COPY of a SIBLING's script would still build, and would assert the wrong stack.

    That is the realistic failure here -- these eleven blocks were written by copying
    one -- and it has no symptom: the image builds, the smoke test passes, and it
    proved nothing about the image it is in.
    """
    text = _strip_hash_comments(dockerfile(name).read_text())
    expected_copy = f"COPY containers/{name}/smoke.sh {IN_IMAGE_PATH}"
    seen_copies = re.findall(r"COPY \S*smoke\.sh \S*", text)
    assert expected_copy in text, (
        f"containers/{name}/Dockerfile does not `{expected_copy}`. Either it never runs "
        "its smoke test, or -- worse -- it COPYs another image's, which builds fine and "
        f"asserts the wrong stack. Its smoke.sh COPY line(s): {seen_copies}"
    )
    assert re.search(rf"RUN\s+bash\s+{re.escape(IN_IMAGE_PATH)}", text), (
        f"containers/{name}/Dockerfile COPYs its smoke.sh but never RUNs it, so the "
        "check ships inside the image without ever having been executed at build time."
    )


@pytest.mark.parametrize("name", _dirs())
def test_the_smoke_script_actually_checks_something(name):
    """An empty or near-empty smoke.sh passes every build and proves nothing --
    which is this repository's single most repeated failure mode."""
    body = _strip_hash_comments(smoke_path(name).read_text())
    assert '"$PY" -c' in body, (
        f"containers/{name}/smoke.sh runs no Python. Whatever else it does, the thing "
        "these scripts exist for is the import check."
    )
    modules = imported_modules(body)
    assert len(modules) >= 2, (
        f"containers/{name}/smoke.sh imports only {sorted(modules)}. A one-module smoke "
        "test passes for an image whose actual stack is broken; assert the modules this "
        "image's bin/ scripts really import."
    )


# `ps` must START a command. Anchored for the same reason the import scan is: the word
# "procps" appears in prose in several of these headers, and an unanchored search for it
# would be satisfied by the comment explaining why the check exists.
_PS_CHECK_RE = re.compile(r"(?m)^\s*ps\s+-e\b")


@pytest.mark.parametrize("name", _dirs())
def test_every_smoke_script_proves_ps_exists(name):
    """Nextflow's task-metrics wrapper hard-exits BEFORE the script block without `ps`.

    `nextflow/executor/command-trace.txt` opens with

        command -v ps &>/dev/null || { >&2 echo "Command 'ps' required by nextflow ..."; exit 1; }

    and `params.enable_trace` defaults to true, so the wrapper is injected into EVERY
    task of EVERY run. An image without procps therefore fails every task with exit
    status 1 and empty stdout -- a failure that reads as "the tool crashed silently",
    not as "a system package is missing". Debian/Ubuntu bases do not ship procps:
    `python:*-slim` and `ubuntu:22.04` both lack it, and `AGGREGATE_SIZE_LOGS` ran in
    bare `ubuntu:22.04` until 2026-09-02.

    COMMENT-BLIND. Every one of these scripts explains procps in prose above the check;
    reading the raw text would let the explanation satisfy the guard while the command
    was deleted. That is this repository's most-repeated defect.
    """
    body = _strip_hash_comments(smoke_path(name).read_text())
    assert _PS_CHECK_RE.search(body), (
        f"containers/{name}/smoke.sh never runs `ps -e`. The image may or may not have "
        "procps -- nothing proves it, and the symptom on the cluster is exit status 1 "
        "with empty stdout on every task. Add:\n"
        '    ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: '
        'nextflow task-metrics wrapper can run"'
    )


@pytest.mark.parametrize("name", _dirs())
def test_every_module_the_smoke_script_imports_is_installed_by_the_image(name):
    """The check that earns this file.

    A mistyped module name turns a smoke test into a BUILD failure, discovered twenty
    minutes into a runner job -- or, on the release path, after the tag exists and half
    the image set is published. The Dockerfile and its requirements files say what pip
    was asked for; compare against that here instead.

    STATIC, and it says so: this proves the package was requested, not that the import
    succeeds. The build proves the rest. It is still the difference between finding a
    typo in a second and finding it in a release.
    """
    declared = declared_distributions(name)
    assert declared, (
        f"no pip requirement was parsed out of containers/{name}/Dockerfile at all, so "
        "the comparison below is against an empty set. The scan, not the image, is what "
        "changed."
    )
    locally_copied = LOCALLY_COPIED_MODULES.get(name, set())
    locally_installed = LOCALLY_INSTALLED_PACKAGES.get(name, {})
    missing = []
    for module in sorted(imported_modules(smoke_path(name).read_text())):
        if module in locally_copied or module in locally_installed:
            continue
        distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
        if distribution.lower() not in declared:
            missing.append(f"{module} (would come from {distribution!r})")
    assert not missing, (
        f"containers/{name}/smoke.sh imports module(s) containers/{name}/Dockerfile "
        f"never installs: {missing}.\n"
        "Either the module name is wrong (the import name and the distribution name "
        "differ often enough that tests/test_container_smoke_tests.py keeps a table of "
        "them), or it arrives transitively -- in which case name it in the Dockerfile, "
        "because an install nothing in that file names is invisible to "
        "tests/test_container_harmonisation.py's 'installs what it imports' guard too.\n"
        f"Declared distributions: {sorted(declared)}"
    )


@pytest.mark.parametrize(
    "name,module",
    sorted(
        (name, module)
        for name, modules in LOCALLY_COPIED_MODULES.items()
        for module in modules
    ),
)
def test_locally_copied_modules_are_actually_copied_by_name(name, module):
    """Every LOCALLY_COPIED_MODULES entry must be a real COPY, by name, in that
    container's own Dockerfile -- otherwise the entry is a silent permanent exemption
    from test_every_module_the_smoke_script_imports_is_installed_by_the_image rather
    than a documented, verified special case."""
    text = _strip_hash_comments(dockerfile(name).read_text())
    assert re.search(rf"COPY\s+\S*/{re.escape(module)}\.py\s+\S+", text), (
        f"LOCALLY_COPIED_MODULES[{name!r}] names {module!r}, but "
        f"containers/{name}/Dockerfile has no `COPY .../{module}.py <dest>` line -- "
        "either the exemption is stale (the smoke script no longer imports it, or the "
        "Dockerfile install changed shape) or the COPY was removed without removing "
        "this entry."
    )
    assert module in imported_modules(smoke_path(name).read_text()), (
        f"LOCALLY_COPIED_MODULES[{name!r}] names {module!r}, but "
        f"containers/{name}/smoke.sh does not actually import it -- the entry has "
        "outlived its reason and should be removed."
    )


@pytest.mark.parametrize(
    "name,module,src_dir",
    sorted(
        (name, module, src_dir)
        for name, packages in LOCALLY_INSTALLED_PACKAGES.items()
        for module, src_dir in packages.items()
    ),
)
def test_locally_installed_packages_are_actually_installed_by_name(
    name, module, src_dir
):
    """Every LOCALLY_INSTALLED_PACKAGES entry must be a real `COPY <src_dir> <dest>` plus
    a `pip install ... <dest>` in that container's own Dockerfile, and a real import in
    its smoke.sh -- otherwise it is a silent permanent exemption from
    test_every_module_the_smoke_script_imports_is_installed_by_the_image."""
    text = _strip_hash_comments(dockerfile(name).read_text())
    text = re.sub(r"\\\s*\n", " ", text)
    copy = re.search(rf"COPY\s+{re.escape(src_dir)}/?\s+(\S+)", text)
    assert copy, (
        f"LOCALLY_INSTALLED_PACKAGES[{name!r}] names {module!r} from {src_dir}, but "
        f"containers/{name}/Dockerfile has no `COPY {src_dir} <dest>` line."
    )
    dest = copy.group(1).rstrip("/")
    assert re.search(rf"pip3?\s+install\b[^\n&]*\s{re.escape(dest)}/?(?:\s|$)", text), (
        f"containers/{name}/Dockerfile COPYs {src_dir} to {dest} but never "
        f"`pip install`s it -- a stray directory, not an installed package."
    )
    assert module in imported_modules(smoke_path(name).read_text()), (
        f"LOCALLY_INSTALLED_PACKAGES[{name!r}] names {module!r}, but "
        f"containers/{name}/smoke.sh does not actually import it -- the entry has "
        "outlived its reason and should be removed."
    )


def test_the_workflow_runs_the_smoke_script_against_the_built_image():
    """Half the value of one definition is that it runs in BOTH places.

    The Dockerfile runs it on an intermediate layer; containers.yml runs it through
    `docker run`, which is what the cluster does -- so an ENTRYPOINT, a CMD or a
    read-only $HOME that breaks starting the image shows up there and nowhere else.

    COMMENT-BLIND, AND THAT COST A DRAFT. The first version of this case matched the
    whole file text, so replacing the invocation with
    `true  # bash /usr/local/bin/mirage-smoke.sh` -- the plausible way somebody
    disables it "temporarily" -- left it PASSING. It is read off `run:` bodies via
    ci_actions.run_scripts (no YAML comment can satisfy it) and then through a
    quote-aware trailing-`#` strip, because ci_actions drops WHOLE-LINE shell comments
    only. Both halves were needed; each alone still passed the break.
    """
    commands = _shell_commands_only(ci_actions.run_scripts(WORKFLOW))
    # The invocation is written over three continued lines, so the window spans them.
    assert re.search(
        rf"docker run[\s\S]{{0,400}}?bash\s+{re.escape(IN_IMAGE_PATH)}", commands
    ), (
        f"{WORKFLOW.name} never `docker run`s {IN_IMAGE_PATH}, so the smoke script is "
        "exercised only at build time and the 'one definition, two places' claim in "
        "containers/<dir>/smoke.sh's header is false. (A comment does not count -- this "
        "reads `run:` bodies with their shell comments, trailing ones included, removed.)"
    )

    # `load:` is a `with:` value rather than a command, so it is read STRUCTURALLY --
    # which is comment-proof by construction.
    build = yaml.safe_load(WORKFLOW.read_text())["jobs"]["build"]
    loads = [
        str((step.get("with") or {}).get("load", ""))
        for step in build["steps"]
        if "docker/build-push-action" in str(step.get("uses", ""))
    ]
    assert loads, "containers.yml's `build` job has no docker/build-push-action step"
    assert any("smoke" in value for value in loads), (
        "the build step does not `load:` the image into the runner's docker daemon on "
        f"the smoke path (load: {loads}), so the `docker run` smoke step above cannot "
        "see it and would fail on a missing image rather than on a broken one."
    )

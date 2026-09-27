"""Every tool a process reports a version for must exist in the image it runs in.

WHAT A PHANTOM COSTS. lib/ProcessEnvelope.groovy renders each tool as

    <name>: $(python -c "import <tool>; print(<tool>.__version__)" 2>/dev/null || echo "unknown")

so a tool no image installs does not fail the task -- it writes the literal string
`unknown` into versions.yml. bin/generate_qc_report.py's parse_versions_yml is a
hand-rolled two-level split that never validates what it finds, and renders the value
verbatim into the report's Process | Tool | Version table. The published provenance record
then claims the pipeline consulted a tool it never loaded, which is worse than a missing
row: a missing row invites a question and `unknown` reads like a version-probe hiccup.

Two instances were live when this guard was written:
  * lib/SegBackends.groovy gave StarDist versionTools ['deepcell', 'tensorflow'], and no
    image installs deepcell -- StarDist has nothing to do with DeepCell. tests/
    test_seg_backends.py asserted the pair verbatim, pinning the phantom in place.
  * modules/local/convert_image.nf reported 'aicsimageio', which is on
    test_container_harmonisation.FORBIDDEN and installed nowhere. The name it wanted is
    bioio.

SCOPE. Three sources, because a process gets its tool list from one of three places:
a literal list in modules/local/*.nf, lib/SegBackends.groovy's per-backend versionTools
(SEGMENT), and lib/WarpBackends.groovy's (WARP_SEG_QC). A process whose container is an
external image (VALIS) is skipped -- there is no containers/<dir>/Dockerfile to check
against, which is the same boundary test_container_harmonisation._module_container_and_
scripts() draws and for the same reason.
"""

import re
from pathlib import Path

import pytest

from tests import test_container_harmonisation as harmonisation
from tests.nfmodel import strip_comments

REPO = Path(__file__).resolve().parent.parent
MODULES = REPO / "modules" / "local"
LIB = REPO / "lib"

# `ProcessEnvelope.versions(task.process, ['a', 'b'], task.container)` and its Stub twin.
# The third argument is optional in the regex so this guard is indifferent to the arity
# change plan 04 made -- it is the TOOL LIST that is under assertion, nothing else.
_VERSIONS_CALL = re.compile(
    r"ProcessEnvelope\.versions(?:Stub)?\(\s*task\.process\s*,\s*\[([^\]]*)\]"
)
_QUOTED = re.compile(r"'([^']*)'")

# `container 'bolt3x/mirage-<dir>:...'` or `container : 'bolt3x/mirage-<dir>@sha256:...'`.
_FIRST_PARTY_IMAGE = re.compile(r"bolt3x/mirage-([a-z0-9]+(?:-[a-z0-9]+)*)[:@]")

# Names that are not pip distributions and never will be.
#
# `python` is the INTERPRETER: ProcessEnvelope.versions() emits its own `python:` row from
# `python --version`, and the two modules that also pass it in the tools list get a second,
# harmless row. It is not a package and no image "installs" it.
#
# `CellSegmentationEvaluator` is VENDORED source under bin/utils/cse/ (upstream 1.5.19,
# kept byte-identical), staged onto $PATH by Nextflow like the rest of bin/. pip installs
# nothing for it -- which is exactly why test_container_harmonisation._third_party_imports
# derives its local-package set from bin/ rather than hardcoding names.
NOT_A_DISTRIBUTION = {"python", "CellSegmentationEvaluator"}


def _backend_version_tools(groovy_name):
    """{container dir: {tool, ...}} from a lib/*Backends.groovy per-backend table.

    Read off strip_comments (comments blanked, string CONTENTS kept), because both the
    image reference and every tool name live inside string literals -- the
    strip_comments_and_strings view would blank exactly the text under assertion.
    """
    text = strip_comments((LIB / groovy_name).read_text())
    out = {}
    for match in re.finditer(
        r"container\s*:\s*'([^']+)'.*?versionTools\s*:\s*\[([^\]]*)\]", text, re.DOTALL
    ):
        image = _FIRST_PARTY_IMAGE.search(match.group(1))
        if image:
            out.setdefault(image.group(1), set()).update(
                _QUOTED.findall(match.group(2))
            )
    return out


def _module_version_tools():
    """{container dir: {tool, ...}} from modules/local/*.nf's literal versions() calls."""
    out = {}
    for path in sorted(MODULES.glob("*.nf")):
        text = strip_comments(path.read_text())
        image = _FIRST_PARTY_IMAGE.search(text)
        if not image:
            continue  # external image (VALIS) or a backend-dispatched container
        container = image.group(1)
        if not (REPO / "containers" / container / "Dockerfile").is_file():
            continue
        for call in _VERSIONS_CALL.findall(text):
            out.setdefault(container, set()).update(_QUOTED.findall(call))
    return out


def _claims():
    """{container dir: {tool, ...}} over all three sources."""
    merged = {}
    for source in (
        _module_version_tools(),
        _backend_version_tools("SegBackends.groovy"),
        _backend_version_tools("WarpBackends.groovy"),
    ):
        for container, tools in source.items():
            merged.setdefault(container, set()).update(tools)
    return merged


def test_the_scan_finds_all_three_sources():
    """A regex that stops matching reports every image as clean.

    plan 04 changed ProcessEnvelope.versions() from two arguments to three; a pattern
    anchored on the closing paren would have silently matched nothing afterwards, and
    every assertion below would have passed over an empty set.
    """
    claims = _claims()
    assert len(claims) >= 8, (
        f"only {sorted(claims)} images were resolved from versions() calls and the two "
        f"backend tables. The scan is stale, not the repository."
    )
    assert "stardist" in claims, (
        "lib/SegBackends.groovy's per-backend versionTools were not resolved; SEGMENT's "
        "container never appears as a literal in modules/local/segment.nf, so losing that "
        "table means losing three images at once."
    )
    assert "drape" in claims, (
        "lib/WarpBackends.groovy's versionTools were not resolved."
    )
    assert "convert" in claims, (
        "no modules/local/*.nf versions() call was parsed at all."
    )


@pytest.mark.parametrize("container", sorted(_claims()))
def test_every_reported_tool_is_installed_in_that_image(container):
    installed = set(harmonisation._installed(container))
    installed |= {
        p.lower() for p in harmonisation.BASE_IMAGE_PROVIDES.get(container, set())
    }
    phantoms = []
    for tool in sorted(_claims()[container]):
        if tool in NOT_A_DISTRIBUTION:
            continue
        dist = harmonisation._IMPORT_TO_DIST.get(tool, tool).lower()
        if dist not in installed and tool.lower() not in installed:
            phantoms.append(f"{tool} (would come from {dist!r})")
    assert not phantoms, (
        f"containers/{container} never installs {phantoms}, but a process running in it "
        f"reports a version for each. ProcessEnvelope's probe swallows the ImportError and "
        f"writes the literal string 'unknown' into versions.yml, which "
        f"bin/generate_qc_report.py renders verbatim -- so the published provenance record "
        f"claims a tool the run never loaded. Either install it, or take it out of the "
        f"tool list.\nInstalled: {sorted(installed)}"
    )

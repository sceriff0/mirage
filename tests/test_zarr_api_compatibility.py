"""The zarr group check must be spelled the one way that works in both zarr 2 and zarr 3.

``zarr.hierarchy`` is REMOVED in zarr 3 -- zarr's own 3.0 migration guide lists it alongside
``zarr.attrs``, ``zarr.indexing`` and ``zarr.util`` as internal modules that no longer exist. So
code must not use ``isinstance(x, zarr.hierarchy.Group)`` to decide whether a tifffile zarr
view is a pyramid group or a plain array.

Under zarr 3 that raises ``AttributeError``, and a consumer that catches ``Exception`` and
falls back to a whole-array ``imread`` turns it into a silent loss of the lazy read: a green
run using far more memory than it should.

``zarr.Group`` exists in both majors (checked by
``test_zarr_group_is_available_under_the_installed_zarr``), so the check is version-agnostic
and images can migrate independently; zarr 3 needs Python >= 3.12 and most base images are
Python 3.10.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Removed in zarr 3.0. Keyed by module path rather than by file, because the point is that the
# attribute does not exist, not that a particular line uses it.
REMOVED_IN_ZARR_3 = (
    "zarr.hierarchy",
    "zarr.creation",
    "zarr.attrs",
    "zarr.indexing",
    "zarr.util",
    "zarr.meta",
    "zarr.n5",
)


def _python_sources():
    return sorted(p for p in (REPO / "bin").rglob("*.py"))


@pytest.mark.parametrize("removed", REMOVED_IN_ZARR_3)
def test_no_source_uses_a_module_removed_in_zarr_3(removed):
    """Using one of these pins the code to zarr 2 without saying so."""
    pattern = re.compile(rf"\b{re.escape(removed)}\b")
    # Comments are stripped before matching. A comment naming the removed module -- e.g. one
    # explaining why the code avoids it -- cannot raise AttributeError, and a guard
    # that cannot tell prose from a call forces the explanation to be deleted to stay green.
    offenders = [
        f"{p.relative_to(REPO)}:{i}"
        for p in _python_sources()
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line.split("#")[0])
    ]
    assert not offenders, (
        f"{removed} was removed in zarr 3.0 but is still referenced at {offenders}. "
        f"Under zarr 3 this raises AttributeError; where the call is wrapped in a broad "
        f"except, it degrades silently to a non-lazy read instead of failing."
    )


def test_zarr_group_is_available_under_the_installed_zarr():
    """Pins the reason the replacement is safe, rather than assuming it.

    If a future zarr moves ``Group`` again, this fails here rather than at gigapixel scale in
    a container nobody can run locally.
    """
    zarr = pytest.importorskip("zarr")
    assert hasattr(zarr, "Group"), (
        f"zarr {zarr.__version__} has no top-level Group; the version-agnostic group check "
        f"used by tiled_io.open_lazy and merge_channels_pyramid relies on it."
    )


def test_the_group_check_actually_discriminates():
    """A group must test True and a plain array must test False under the installed zarr.

    Asserting only that ``zarr.Group`` exists would pass even if it were an unrelated object,
    which would make every pyramid read take the array branch.
    """
    zarr = pytest.importorskip("zarr")
    group = zarr.group()
    group.create_dataset("0", shape=(4, 4), dtype="uint16")
    assert isinstance(group, zarr.Group)
    assert not isinstance(group["0"], zarr.Group)

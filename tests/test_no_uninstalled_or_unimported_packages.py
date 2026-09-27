"""The reverse of "installs what it imports": an image may not install what nothing reaches.

tests/test_container_harmonisation.py::test_container_installs_what_its_scripts_import
catches the dependency an image is MISSING -- the bioio case, invisible at build time and
fatal at gigapixel scale after the scheduler has already granted the task its memory.
Nothing caught the opposite direction, and the opposite direction had a live instance:
containers/regqc carried tensorflow 2.17.1, stardist 0.9.1, csbdeep, gputools, edt and a
whole Miniconda install with bioconda bftools -- roughly 4 GB pulled on every QC task,
imported by nothing, for an image that runs ONE script whose third-party import set is
{cv2, numpy, skimage, tifffile, zarr}.

tests/test_no_unreachable_container_frameworks.py is the neighbouring guard and it is
deliberately a NAMED DENYLIST: "Most entries in a Dockerfile are legitimately transitive
or runtime-only dependencies that no first-party file imports, so a blanket rule would
produce noise and get disabled." That is true, and it is the reason this file exists in
the shape it does: it takes the blanket rule and pays for it with ALLOWED_UNIMPORTED --
explicit, per-image, each entry naming the distribution that loads the package at run
time, and each entry checked by two meta-tests so it cannot outlive its reason.

"Might be needed later" is not a reason. That is what produced the regqc stack, and
tests/test_no_dead_bin_modules.py already rejects it one layer down for bin/utils/.

NOTHING HERE IS RE-DERIVED. The parser, the container-to-scripts mapping, the import walk
and the import-name-to-distribution table are all imported from
test_container_harmonisation, which is where they are debugged; this repository has been
bitten by seven guards each carrying a private parse of the same files, each with a
different blind spot.
"""

import pytest

from tests import test_container_harmonisation as harmonisation

# (container, distribution) -> why nothing in bin/ imports it and it must stay anyway.
#
# THE RULE FOR ADDING AN ENTRY: the package must be loaded by ANOTHER INSTALLED
# DISTRIBUTION at run time -- a codec backend, a plugin discovered by entry point, a
# required dependency of a package bin/ does import. If the reason you can write is
# "something might need it", delete the pip line instead.
#
# NOTE ON WHAT DOES NOT NEED AN ENTRY HERE: `_unimported()` below counts
# `REQUIRED_RUNTIME_IMPORTS[container]` (test_container_harmonisation.py) as reached, not
# just module-scope imports. A package whose only importer is a LAZY import already
# declared there -- containers/convert's bioio, h5py, scyjava; containers/tiled's torch,
# kornia, zarr, scipy, skimage; containers/segeval's matplotlib; every name in
# containers/spatialdata's REQUIRED_RUNTIME_IMPORTS entry -- is therefore never unimported
# in the first place and must NOT be listed below: an entry here for one of those would
# immediately fail test_every_allowlist_entry_is_still_unimported, because the package is
# reached. Two tables would otherwise carry the same fact with two different reasons that
# can drift -- REQUIRED_RUNTIME_IMPORTS is the one place a lazily-reached runtime import is
# recorded; this dict is for a package NEITHER walker finds an importer for, because it is
# loaded by another INSTALLED DISTRIBUTION rather than by any first-party script at all.
ALLOWED_UNIMPORTED = {
    # --- codec and store backends, everywhere -------------------------------------
    # tifffile does not depend on imagecodecs; it imports it lazily, per codec, on the
    # first LZW/zstd/JPEG read or write. Its absence is therefore never an ImportError at
    # module scope -- it surfaces mid-write, on the largest artifact in the pipeline.
    **{
        (c, "imagecodecs"): (
            "tifffile's codec backend, imported lazily per codec on the first "
            "LZW/zstd/JPEG read or write; not a declared dependency of tifffile, so it "
            "must be installed explicitly or a compressed write fails at run time."
        )
        # spatialdata is NOT in this comprehension: it gets its own entry below, with a
        # reason naming the specific artifact it reads. Listing it in both would make one
        # dict literal silently override the other -- legal Python, and exactly the kind of
        # duplicate that makes an allowlist unreviewable.
        for c in (
            "cellsam",
            "convert",
            "instanseg",
            "merge",
            "preprocess",
            "quantify",
            "regqc",
            "stardist",
            "tiled",
        )
    },
    # zarr 2 delegates every compression to numcodecs, and the two must move together:
    # a newer numcodecs drops blosc.cbuffer_sizes, which zarr 2.18 imports at module
    # scope (requirements/tiled.txt records the same fact for the same reason).
    **{
        (c, "numcodecs"): (
            "zarr 2.18's codec layer. Pinned beside zarr because the pair cannot split: "
            "a newer numcodecs drops blosc.cbuffer_sizes, which zarr 2.18 imports."
        )
        for c in (
            "cellsam",
            "convert",
            "instanseg",
            "merge",
            "preprocess",
            "quantify",
            "regqc",
            "stardist",
            "tiled",
        )
    },
    # --- per-image runtime dependencies -------------------------------------------
    ("instanseg", "zarr"): (
        "tifffile's aszarr store backend. bin/segment_instantseg.py reads through "
        "tifffile, which builds a zarr view for region reads; zarr is not a tifffile "
        "dependency, so it is installed explicitly."
    ),
    ("quantify", "zarr"): (
        "tifffile's aszarr store backend, as for containers/instanseg. The six quantify "
        "entrypoints read masks through tifffile."
    ),
    ("preprocess", "scipy"): (
        "scikit-image's required runtime dependency -- skimage.transform.rescale calls "
        "into scipy.ndimage. Pinned here so the harmonised version is enforced at BUILD "
        "time rather than resolved on build day."
    ),
    ("regqc", "scipy"): (
        "scikit-image's required runtime dependency; bin/utils/qc.py calls "
        "skimage.transform.rescale, which goes through scipy.ndimage."
    ),
    ("stardist", "scipy"): (
        "required by both scikit-image and csbdeep, which bin/segment.py imports "
        "directly. Pinned so the harmonised version is enforced at build time."
    ),
    ("stardist", "gputools"): (
        "StarDist's optional OpenCL accelerator for non-maximum suppression, imported by "
        "the stardist distribution when present -- never by bin/segment.py. The apt block "
        "in this Dockerfile installs the OpenCL ICD loader and headers for it."
    ),
    ("stardist", "edt"): (
        "StarDist's optional Euclidean-distance-transform accelerator, imported by the "
        "stardist distribution when present -- never by bin/segment.py."
    ),
    # --- containers/convert: bioio's plugins are entry points, not imports ---------
    **{
        ("convert", plugin): (
            "a bioio reader PLUGIN. bioio discovers it through its entry point and picks "
            "it by file extension; nothing imports the name. Dropping it removes a "
            "supported format silently instead of raising ImportError."
        )
        for plugin in (
            "bioio-ome-tiff",
            "bioio-tifffile",
            "bioio-nd2",
            "bioio-czi",
            "bioio-lif",
            "bioio-bioformats",
        )
    },
    ("convert", "dask"): (
        "bioio's array backend. bin/convert_image.py's read is BioImage.dask_data and its "
        "write iterates that array's chunks, so dask is on the live path -- but it is "
        "reached through bioio, not imported by name."
    ),
    ("convert", "zarr"): (
        "tifffile's aszarr store backend, which bioio-ome-tiff reads through."
    ),
    # ('convert', 'scyjava') is deliberately ABSENT: REQUIRED_RUNTIME_IMPORTS['convert']
    # ['scyjava'] (test_container_harmonisation.py) already covers it -- bin/utils/ome_io.py's
    # and bin/convert_image.py's calls to jvm_cache.point_jvm_cache_off_readonly_home() reach
    # scyjava's lazy `from scyjava import config` unconditionally on the bioformats route -- so
    # scyjava is REACHED, not unimported, and an entry here would fail
    # test_every_allowlist_entry_is_still_unimported.
    # --- containers/spatialdata: the scverse serialization stack -------------------
    ("spatialdata", "zarr"): (
        "spatialdata's on-disk store. This image is the python:3.12 / zarr>=3 island and "
        "the zarr major version is the reason it is a separate image at all, so the pin "
        "is named here rather than left transitive."
    ),
    ("spatialdata", "ome-zarr"): (
        "the NGFF writer spatialdata serialises images and labels through."
    ),
    ("spatialdata", "pyarrow"): (
        "the GeoParquet writer spatialdata's shapes/ element is backed by."
    ),
    ("spatialdata", "imagecodecs"): (
        "tifffile's codec backend: bin/export_spatialdata.py reads MERGE_AND_PYRAMID's "
        "zstd/LZW-compressed OME-TIFF through tifffile."
    ),
    # --- pending an image rebuild -------------------------------------------------
    **{
        ("tiled", dist): (
            "DRAPE's COARSE anchor stopped importing torch/kornia on 2026-09-27 (DISK + "
            "LightGlue -> NCC rotation sweep + ORB fallback, drape/coarse_align.py). "
            "bolt3x/mirage-tiled:1.0.0 still installs both; dropping requirements/"
            "torch-cpu.txt and requirements/kornia.txt from containers/tiled/Dockerfile "
            "needs a user-approved rebuild + publish. Delete this entry in that change -- "
            "test_every_allowlist_entry_is_still_installed will insist."
        )
        for dist in ("torch", "kornia")
    },
}


def _unimported(container):
    """Distributions this image installs that no script it runs imports.

    "Imports" here means module-scope imports (``_third_party_imports``, walked
    module-scope-only as of Task 8a) UNIONED with ``REQUIRED_RUNTIME_IMPORTS[container]``
    -- the positive table of names a container's scripts reach only through a lazy
    (runtime, not module-scope) import, each proven still genuine by
    ``test_required_runtime_imports_are_actually_reached`` in test_container_harmonisation.
    Without that union, every package a script needs only at runtime -- bioio, h5py and
    scyjava for containers/convert; torch/kornia for containers/tiled; matplotlib for
    containers/segeval -- would read as unreached and demand an ALLOWED_UNIMPORTED entry
    duplicating a reason REQUIRED_RUNTIME_IMPORTS already states, which is exactly the kind
    of second copy of the same fact this repository's guards avoid (see
    test_container_harmonisation's own docstring on re-derivation).
    """
    installed = set(harmonisation._installed(container))
    scripts = harmonisation._module_container_and_scripts().get(container, set())
    reached = set()
    for script in scripts:
        for name in harmonisation._third_party_imports(script):
            reached.add(harmonisation._IMPORT_TO_DIST.get(name, name).lower())
            reached.add(name.lower())
    for name in harmonisation.REQUIRED_RUNTIME_IMPORTS.get(container, {}):
        reached.add(harmonisation._IMPORT_TO_DIST.get(name, name).lower())
        reached.add(name.lower())
    return {p for p in installed if p not in reached}


def test_the_scan_reaches_every_image_that_runs_a_script():
    """A mapping that comes back empty for an image would exempt it silently.

    _module_container_and_scripts() resolves SEGMENT's three backends through
    lib/SegBackends.groovy, so its coverage is wider than a modules/local/ glob; if that
    resolution ever breaks, this is what says so rather than eleven vacuous passes.
    """
    mapping = harmonisation._module_container_and_scripts()
    for expected in (
        "convert",
        "regqc",
        "quantify",
        "stardist",
        "cellsam",
        "instanseg",
        "preprocess",
        "merge",
        "tiled",
        "segeval",
        "spatialdata",
    ):
        assert mapping.get(expected), (
            f"no scripts were resolved for containers/{expected}, so the unimported scan "
            f"below would compare its install list against an empty import set and flag "
            f"everything -- or, with a generous allowlist, nothing. The mapping is what "
            f"changed, not the image."
        )


@pytest.mark.parametrize("container", harmonisation._container_dirs())
def test_no_image_installs_a_package_nothing_reaches(container):
    """An install nothing imports is dead weight pulled on every task, forever.

    It is also a false claim: the Dockerfile says this image depends on the package, and
    the next person reading it to find out what the scripts need is misled.
    """
    offenders = sorted(
        p for p in _unimported(container) if (container, p) not in ALLOWED_UNIMPORTED
    )
    assert not offenders, (
        f"containers/{container} installs {offenders}, which no script it runs imports. "
        f"Delete the pip line -- every install in these Dockerfiles passes "
        f"`-c constraints.txt`, so a package that genuinely arrives transitively is still "
        f"held at the harmonised version and an explicit line buys nothing. If it is "
        f"loaded at RUN TIME by another installed distribution (a codec backend, an "
        f"entry-point plugin), add it to ALLOWED_UNIMPORTED with that distribution named. "
        f"'Might be needed later' is not a reason; that is what put TensorFlow in the "
        f"registration-QC image."
    )


@pytest.mark.parametrize("key", sorted(ALLOWED_UNIMPORTED))
def test_every_allowlist_entry_is_still_installed(key):
    """A stale exemption is worse than none: it reads as a reviewed decision.

    Same argument as test_every_pinned_exception_is_still_doing_something -- an entry for
    a package the image no longer installs is a hole nobody re-examines.
    """
    container, package = key
    assert container in harmonisation._container_dirs(), (
        f"ALLOWED_UNIMPORTED names containers/{container}, which has no Dockerfile."
    )
    assert package in harmonisation._installed(container), (
        f"ALLOWED_UNIMPORTED exempts {package!r} in containers/{container}, but that image "
        f"no longer installs it. Delete the entry."
    )
    assert ALLOWED_UNIMPORTED[key].strip(), (
        "every entry must name the distribution that loads the package at run time"
    )


@pytest.mark.parametrize("key", sorted(ALLOWED_UNIMPORTED))
def test_every_allowlist_entry_is_still_unimported(key):
    """The other direction: once a script imports it, the exemption is not just stale, it
    is actively hiding the package from the FORWARD guard.

    test_container_installs_what_its_scripts_import would then be the thing keeping it
    honest -- but only if the package is genuinely required, and an exemption here does
    not stop that. Failing loudly is still right: it means the reason written above the
    entry is now false.
    """
    container, package = key
    assert package in _unimported(container), (
        f"ALLOWED_UNIMPORTED says nothing in containers/{container} imports {package!r}, "
        f"but a script it runs now does. The exemption's stated reason is false; delete "
        f"the entry and let the forward guard own the dependency."
    )

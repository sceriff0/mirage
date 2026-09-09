"""Every fixture a test opens must actually be there.

tests/testdata/ is gitignored (.gitignore:140,142) with a small tracked
exception, so a fixture is available at all only if
tests/testdata/generate_complete_testdata.py writes it or git carries it. A
fixture that is neither exists on exactly one developer's machine, and the test
that opens it fails everywhere else -- or passes locally and hides a regression
CI cannot see.

Not hypothetical. tests/testdata/seg_eval_P001.json and _P002.json were
referenced by tests/modules/merge_seg_eval.nf.test with `checkIfExists: true`
and produced by nothing: the CSE removal in cea0b47 took the writer away and
left the reader. They were 2 of the 3 nf-test failures standing on this branch,
and had been failing since that commit.

RULING R7, INVERTED. R7 says fixtures go in the generator and are never
committed. This is the other direction -- a fixture the generator never writes
-- and it is the one that actually bit.

WHY THIS CHECKS EXISTENCE RATHER THAN THE GENERATOR'S SOURCE. The first version
of this file grepped the generator for each filename, and it was wrong twice
over: it reported sample_DAPI_intensity.csv as an orphan (the generator writes
it through an f-string, so the literal name never appears) and it would have
missed a producer that had been deleted while its `open(...)` line survived.
Existence is the property that matters and the only one with no blind spot.

It binds in CI, which is where it must. Every job that touches fixtures runs the
generator into a fresh checkout first, so "the file is there" can only mean the
generator wrote it or git carries it. Locally it is weaker -- a stale file from
an earlier branch satisfies it -- and that is an acceptable asymmetry: the
failure being prevented is "green here, red in CI", and CI is where the strong
form runs.
"""

import json
import re
import subprocess
from pathlib import Path

from tests.nfmodel import strip_comments

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "tests" / "testdata"
GENERATOR = TESTDATA / "generate_complete_testdata.py"

# Any tests/testdata/<name> reference in an nf-test file, a Python test or a
# test config. A trailing '.' is prose punctuation, not part of the name.
# `\w` is Python's default Unicode-aware word class (no re.UNICODE needed for a
# str pattern), so this already matches an accented filename like
# P001_ünïcode.ome.tiff -- it does NOT match a space, which is deliberate: this
# regex also runs over prose (docstrings, assertion messages) where a bare
# space would swallow the rest of a sentence as part of the "fixture name".
_REF = re.compile(r"tests/testdata/([\w\-][\w\-.]*[\w\-])")

# A tests/testdata/<name> reference INSIDE A CSV FIXTURE FIELD specifically
# (space_path_input.csv, unicode_input.csv, and friends -- generate_complete_
# testdata.py section 10i). A CSV field's own syntax already gives an
# unambiguous terminator -- a comma (unquoted) or the closing quote (quoted) --
# so unlike _REF above, this can safely allow a space or a `/` subdirectory
# without risking swallowing trailing prose: a fixture CSV has no prose. This
# is what lets "with space/P001 ref.ome.tiff" and a comma-free absolute path
# resolve to the right file instead of just the first path segment.
_REF_CSV_FIELD = re.compile(r'tests/testdata/([^,"\n]+)')

# Fixtures that are DELIBERATELY empty, keyed on the name, with the reason. Each
# entry is verified below to still exist AND still be empty -- a stale exemption is
# a licence nobody is using that the next reader will trust.
EMPTY_ON_PURPOSE = {
    "dummy_singularity.sif": (
        "not an image and never opened. tests/modules/"
        "segment_deepcell_token_singularity.config points SEGMENT's `container` at "
        "this path so Nextflow's Singularity engine treats it as an ALREADY-LOCAL "
        "image and skips `singularity pull` -- there is no singularity binary in "
        "this dev environment or on the CI runner. The task then fails at exec, "
        "AFTER .command.run has been written, and .command.run is the only thing "
        "tests/modules/segment_deepcell_token.nf.test reads. A real .sif would be "
        "hundreds of MB and would change nothing."
    ),
}

# Fixtures that are DELIBERATELY CORRUPT -- non-empty, but built specifically so a
# reader FAILS on them -- keyed on the name, with the reason. A DIFFERENT shape of
# exemption from EMPTY_ON_PURPOSE above: these are not zero bytes and the zero-byte
# check still applies to them (an unexpected truncation to 0 bytes would itself be a
# regression worth catching); only `_openable()`'s "this must parse" assumption is
# wrong for a fixture whose entire point is that it does not parse. Each entry is
# verified below to still exist, still be non-empty, AND still actually fail to open --
# a fixture that became openable again (or vanished) would make the exemption a stale
# licence nobody is using.
DELIBERATELY_UNOPENABLE = {
    "fmt_truncated.ome.tiff": (
        "the first 100 bytes of fmt_bigtiff.ome.tiff -- past the TIFF header, short of "
        "the first IFD (generate_complete_testdata.py section 11f). "
        "tests/integration/formats/test_ome_io_read_info.py's "
        "test_a_truncated_file_raises_rather_than_returning_a_partial_read exists "
        "precisely because a reader that returned an empty array instead of raising on "
        "a damaged slide becomes an all-zero slide four processes later -- a version of "
        "THIS guard that demanded the file be openable would fail the one fixture "
        "whose whole job is proving the pipeline does not swallow that failure."
    ),
}


def _openable(path):
    """(ok, reason) -- can this fixture be opened as what its suffix claims?

    Emptiness alone is not the property that matters: a 12-byte TIFF that
    tifffile cannot parse fails in exactly the way the 0-byte one did
    (`tifffile.TiffFileError: not a TIFF file b''`, which is how
    MIRAGE:READ_SEGMENTED_CHECKPOINT:EXTRACT_NUCLEI_PROPERTIES aborted on the
    nightly runner). So the check is per-type and it actually opens the file.
    """
    if path.suffix in (".tif", ".tiff"):
        import tifffile

        try:
            with tifffile.TiffFile(path) as tif:
                if not tif.series:
                    return False, "TIFF parsed but carries no series"
        except Exception as exc:  # noqa: BLE001 -- any reader failure is the failure
            return False, f"tifffile could not open it: {exc.__class__.__name__}: {exc}"
    elif path.suffix == ".csv":
        lines = path.read_text(errors="ignore").splitlines()
        if not lines or "," not in lines[0]:
            return False, "no comma-separated header line"
    elif path.suffix == ".json":
        try:
            json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            return False, f"not valid JSON: {exc.__class__.__name__}: {exc}"
    return True, ""


def _referenced():
    """fixture path (relative to tests/testdata) -> files that reference it."""
    refs = {}
    for path in sorted(ROOT.glob("tests/**/*")):
        if path.is_dir() or path.suffix not in (
            ".test",
            ".py",
            ".nf",
            ".config",
            ".csv",
        ):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.resolve() == GENERATOR.resolve():
            continue  # the generator naturally names everything it writes
        if TESTDATA in path.parents and path.suffix != ".csv":
            continue
        # A GENERATED CHECKPOINT CSV IS A REFERENCE. (The case that motivated this,
        # a prior_run/csv/postprocessed.csv whose `pyramid` column named a fixture
        # nothing else in tests/ mentioned, belongs to the add_cycle path on the dev
        # branch; the rule stays because any checkpoint CSV can be the sole witness.)
        #
        # Comments blanked, STRINGS INTACT. A fixture named in a `//` comment --
        # this repo's tests explain their fixtures constantly -- is not a
        # reference, and would make this file demand a producer for a path
        # nothing opens. The path itself lives inside a quoted `file('...')`
        # argument, so the strings must survive or the scan finds nothing.
        #
        # A .csv fixture gets the CSV-FIELD pattern, not the prose-safe one: a
        # fixture CSV has no prose to protect against, and its own comma/quote
        # syntax is what lets a space or a `/` subdirectory be captured safely
        # (see _REF_CSV_FIELD's doc above).
        pattern = _REF_CSV_FIELD if path.suffix == ".csv" else _REF
        for name in pattern.findall(strip_comments(path.read_text(errors="ignore"))):
            refs.setdefault(name, set()).add(str(path.relative_to(ROOT)))
    return refs


def test_every_referenced_fixture_exists():
    missing = [
        f"tests/testdata/{name} -- referenced by {', '.join(sorted(users))}"
        for name, users in sorted(_referenced().items())
        if not (TESTDATA / name).exists()
    ]
    assert not missing, "\n".join(missing) + (
        "\n\nRun `python tests/testdata/generate_complete_testdata.py`. If it "
        "still does not appear, the fixture has no producer: add it to that "
        "script (RULING R7 -- fixtures are generated, never committed, because "
        "tests/testdata/ is gitignored and a hand-written one is absent in CI)."
    )


def test_every_referenced_fixture_is_non_empty_and_openable():
    """`.exists()` is satisfied by a 0-byte file, and five of them were tracked.

    tests/testdata/{P001_image.tiff, P001_merged_quant.csv, P001_nuclei_mask.tif,
    P001_pyramid.ome.tiff, dummy_singularity.sif} were committed at zero bytes and
    passed the existence check above for months. The first four have producers now;
    the fifth is a genuine sentinel -- nothing opens it, and it is allowlisted in
    EMPTY_ON_PURPOSE above with the reason -- which is why this test skips that one
    rather than the count dropping to four. The cost was not hypothetical:
    `nf-test --tag real` aborted in EXTRACT_NUCLEI_PROPERTIES with
    `tifffile.TiffFileError: not a TIFF file b''` and the failure was recorded in
    .github/workflows/nightly.yml's header as a property of the SUITE rather than
    of a fixture.
    """
    bad = []
    for name, users in sorted(_referenced().items()):
        path = TESTDATA / name
        if not path.is_file():
            continue  # absence is the check above's business, not this one's
        if name in EMPTY_ON_PURPOSE:
            continue
        if path.stat().st_size == 0:
            bad.append(
                f"tests/testdata/{name} is 0 bytes -- referenced by {', '.join(sorted(users))}"
            )
            continue
        if name in DELIBERATELY_UNOPENABLE:
            continue
        ok, why = _openable(path)
        if not ok:
            bad.append(
                f"tests/testdata/{name}: {why} -- referenced by {', '.join(sorted(users))}"
            )
    assert not bad, "\n".join(bad) + (
        "\n\nA fixture that exists but cannot be opened fails at the READER, in a "
        "process, with a message about the tool rather than about the fixture. Give "
        "it a producer in tests/testdata/generate_complete_testdata.py (RULING R7), "
        "or -- if it is genuinely a sentinel whose content is never read -- add it to "
        "EMPTY_ON_PURPOSE (zero bytes) or DELIBERATELY_UNOPENABLE (non-empty but built "
        "to fail a reader on purpose) in this file with the reason."
    )


def test_the_empty_on_purpose_allowlist_has_no_dead_entries():
    """An exemption for a fixture that is gone, or that now has real content, is a
    licence nobody is using and the next reader will trust.

    A THIRD way an entry goes dead: nothing references it any more. The fixture
    still exists at zero bytes, so the two checks above stay silent, but no test
    opens it -- the exemption is licensing a file the suite no longer looks at.
    """
    stale = []
    refs = _referenced()
    for name, reason in EMPTY_ON_PURPOSE.items():
        assert reason.strip(), f"{name} has an empty reason"
        path = TESTDATA / name
        if not path.is_file():
            stale.append(f"{name}: no such fixture -- delete the exemption")
        elif path.stat().st_size != 0:
            stale.append(
                f"{name}: is {path.stat().st_size} bytes now, not empty -- delete the exemption"
            )
        elif name not in refs:
            stale.append(
                f"{name}: no longer referenced by any test -- delete the exemption"
            )
    assert not stale, "\n".join(stale)


def test_the_deliberately_unopenable_allowlist_has_no_dead_entries():
    """Same shape as the EMPTY_ON_PURPOSE staleness check above, plus a FOURTH way
    this particular exemption class goes dead that the empty one cannot: the fixture
    quietly became openable. A DELIBERATELY_UNOPENABLE entry that _openable() now
    accepts is licensing a claim ("this cannot be read") that is no longer true, and
    the reader-side test the fixture exists for (e.g.
    test_a_truncated_file_raises_rather_than_returning_a_partial_read) would then be
    the only thing standing between a silent regression and the suite -- exactly the
    blind spot this file exists to close.
    """
    stale = []
    refs = _referenced()
    for name, reason in DELIBERATELY_UNOPENABLE.items():
        assert reason.strip(), f"{name} has an empty reason"
        path = TESTDATA / name
        if not path.is_file():
            stale.append(f"{name}: no such fixture -- delete the exemption")
            continue
        if path.stat().st_size == 0:
            stale.append(f"{name}: is 0 bytes -- belongs in EMPTY_ON_PURPOSE, not here")
            continue
        if name not in refs:
            stale.append(
                f"{name}: no longer referenced by any test -- delete the exemption"
            )
            continue
        ok, _why = _openable(path)
        if ok:
            stale.append(
                f"{name}: _openable() now accepts it -- it is no longer corrupt, "
                "delete the exemption"
            )
    assert not stale, "\n".join(stale)


def test_the_scan_finds_real_references():
    """A regex that matched nothing would pass the check above vacuously, and
    the failure it prevents is invisible from the interface."""
    refs = _referenced()
    assert len(refs) >= 15, (
        f"only {len(refs)} fixture reference(s) found across tests/ -- the scan "
        "is stale, not the tests"
    )
    assert GENERATOR.exists(), "the fixture generator is gone"


def test_the_generator_is_what_produces_the_untracked_ones():
    """The check above is satisfied by any file on disk. This pins the claim the
    docstring makes about WHERE those files come from: most referenced fixtures
    are NOT tracked by git, so something else must be writing them, and the
    generator is the only candidate."""
    tracked = {
        Path(p).name
        for p in subprocess.run(
            ["git", "ls-files", "tests/testdata"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    }
    referenced = set(_referenced())
    untracked = {n for n in referenced if Path(n).name not in tracked}
    assert len(untracked) > len(referenced) / 2, (
        "most referenced fixtures are now git-tracked, which contradicts RULING "
        "R7 and means tests/testdata/ has quietly stopped being generated: "
        f"{len(untracked)} of {len(referenced)} are untracked"
    )

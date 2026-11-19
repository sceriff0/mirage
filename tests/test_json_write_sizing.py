"""`f.write(json.dumps(o))` is faster than `json.dump(o, f)` -- and only safe on small outputs.

`json.dump` serialises straight into the file through many small `f.write` calls;
`json.dumps` builds the whole document as one string, about 5x faster but costing roughly
3x the output size in peak memory. Free on a small manifest; wrong on a per-cell artifact
(`contours.json`, `cells.geojson`), which must stream.

This guard pins the boundary in both directions.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Writers whose output is one entry PER CELL. None may serialise the WHOLE document into a
# string before writing it.
#
# `json.dumps` on one feature at a time (what `_stream_collection` does) is fine; `dumps` on
# the whole collection is not. So each file is checked for its specific form, not the token.
PER_CELL_WRITERS = {
    "bin/mask_to_geojson.py": "one polygon per label",
    "bin/extract_cell_properties.py": "contours.json -- one ring per cell",
}

# Writers that stream through the incremental helper instead.
STREAMING_WRITERS = {
    "bin/export_geojson.py": "cells.geojson -- written one feature at a time",
}

# Writers whose output is bounded and small. These should use the fast form.
SMALL_WRITERS = {
    "bin/create_channels_manifest.py": "a channel manifest, a few hundred bytes",
    "bin/warp_seg_qc.py": "one summary record",
    "bin/utils/stage_checkpoint.py": "a stage manifest",
}

_DUMPS_WRITE = re.compile(r"\.write\(\s*json\.dumps\(")
_DUMP = re.compile(r"\bjson\.dump\(")


def test_per_cell_writers_still_stream():
    offenders = []
    for rel, why in PER_CELL_WRITERS.items():
        text = (REPO / rel).read_text()
        if _DUMPS_WRITE.search(text):
            offenders.append(f"{rel} ({why})")

    assert not offenders, (
        "these writers emit one entry per cell, so json.dumps would materialise the whole "
        "document -- measured at ~3.2x the output size in peak RSS. Stream with json.dump, or "
        "stream incrementally (PERF-PLAN measures -1004 MB for EXPORT_GEOJSON):\n  "
        + "\n  ".join(offenders)
    )


def test_the_streaming_writers_do_not_build_the_document_first():
    """export_geojson must go through _stream_collection, not dump a finished collection."""
    offenders = []
    for rel, why in STREAMING_WRITERS.items():
        text = (REPO / rel).read_text()
        if "_stream_collection" not in text:
            offenders.append(f"{rel} ({why}): no _stream_collection")
        if (
            re.search(r'json\.dump\(\s*\{?\s*"type"', text)
            or 'features": features' in text
        ):
            offenders.append(f"{rel} ({why}): serialises the whole FeatureCollection")

    assert not offenders, (
        "the per-cell GeoJSON must be written one feature at a time; building the collection "
        "first is the -1004 MB regression PERF-PLAN C11 removes:\n  "
        + "\n  ".join(offenders)
    )


def test_small_writers_use_the_fast_form():
    offenders = []
    for rel, why in SMALL_WRITERS.items():
        text = (REPO / rel).read_text()
        if not _DUMPS_WRITE.search(text):
            offenders.append(f"{rel} ({why})")

    assert not offenders, (
        "these outputs are small and bounded, so the 5.6x faster f.write(json.dumps(...)) "
        "costs nothing:\n  " + "\n  ".join(offenders)
    )


def test_every_listed_file_exists_and_writes_json():
    """A guard naming a file that moved or stopped writing JSON checks nothing."""
    for rel in {**PER_CELL_WRITERS, **SMALL_WRITERS, **STREAMING_WRITERS}:
        path = REPO / rel
        assert path.exists(), f"{rel} no longer exists -- re-point or remove this entry"
        text = path.read_text()
        assert _DUMP.search(text) or _DUMPS_WRITE.search(text), (
            f"{rel} no longer writes JSON -- remove its entry rather than leaving it to pass "
            "vacuously"
        )


def test_the_two_forms_produce_identical_bytes(tmp_path):
    """The whole premise: this is a speed/memory trade, never an output change."""
    import json

    obj = {
        "cells": [{"id": i, "poly": [[float(i), float(i + 1)]] * 4} for i in range(50)]
    }

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    with open(a, "w") as f:
        json.dump(obj, f)
    with open(b, "w") as f:
        f.write(json.dumps(obj))

    assert a.read_bytes() == b.read_bytes()


def test_the_two_forms_are_identical_with_indent_too(tmp_path):
    """The small writers pass indent=2; the equivalence must hold there as well."""
    import json

    obj = {"stage": "rigid", "files": ["a", "b"], "n": 3}

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    with open(a, "w") as f:
        json.dump(obj, f, indent=2)
    with open(b, "w") as f:
        f.write(json.dumps(obj, indent=2))

    assert a.read_bytes() == b.read_bytes()

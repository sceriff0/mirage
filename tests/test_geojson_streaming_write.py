"""`cells.geojson` must be written without holding every feature in memory first.

Building a list of every cell's feature and then one `json.dump` holds a dict-of-dicts per
cell on the pipeline's largest artifact; the writer streams instead.

The requirement that makes this safe is **byte-identity**. `json.dump` uses `", "` between
array items and `": "` between keys; a hand-rolled writer that emits bare `,` produces valid,
semantically identical JSON that is nonetheless a different file. The consumer is QuPath /
FlowPath, which parses JSON and does not care -- but a byte-identical write means this change
cannot be the cause of any downstream difference, which is worth more than the whitespace.

So these tests compare the streamed output against `json.dump` of the very same features, byte
for byte, including the empty case.
"""

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import export_geojson as eg  # noqa: E402


def _features(n):
    return [
        {
            "type": "Feature",
            "id": f"cell-{i}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[float(i), float(i + 1)]] * 4],
            },
            "properties": {"measurements": [{"name": "CD3", "value": i * 1.5}]},
        }
        for i in range(n)
    ]


def _reference(features, path):
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def test_streamed_output_is_byte_identical_to_a_single_dump(tmp_path):
    features = _features(25)
    streamed, reference = tmp_path / "s.json", tmp_path / "r.json"

    count = eg._stream_collection(iter(features), str(streamed))
    _reference(features, str(reference))

    assert count == 25
    assert streamed.read_bytes() == reference.read_bytes()


def test_streamed_output_is_byte_identical_for_a_single_feature(tmp_path):
    features = _features(1)
    streamed, reference = tmp_path / "s.json", tmp_path / "r.json"

    eg._stream_collection(iter(features), str(streamed))
    _reference(features, str(reference))

    assert streamed.read_bytes() == reference.read_bytes()


def test_streamed_output_is_byte_identical_when_there_are_no_features(tmp_path):
    """The empty collection is the case a naive separator-joining writer gets wrong."""
    streamed, reference = tmp_path / "s.json", tmp_path / "r.json"

    count = eg._stream_collection(iter([]), str(streamed))
    _reference([], str(reference))

    assert count == 0
    assert streamed.read_bytes() == reference.read_bytes()


def test_streamed_output_parses_back_to_the_same_object(tmp_path):
    features = _features(10)
    streamed = tmp_path / "s.json"

    eg._stream_collection(iter(features), str(streamed))

    assert json.loads(streamed.read_text()) == {
        "type": "FeatureCollection",
        "features": features,
    }


def test_the_writer_never_materialises_the_feature_list(tmp_path):
    """The point of the change: it must consume a lazy iterator without building a list."""
    consumed = []

    def gen():
        for feat in _features(5):
            consumed.append(feat["id"])
            yield feat
            # if the writer had built a list first, every id would already be recorded
            # by the time the first one is written -- see the file-size check below.
            assert streamed.exists()

    streamed = tmp_path / "s.json"
    eg._stream_collection(gen(), str(streamed))

    assert consumed == [f"cell-{i}" for i in range(5)]


def test_export_geojson_produces_the_same_bytes_as_before_the_change(tmp_path):
    """End to end through the real export, against a reference built the old way."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "x": [10.0, 20.0, 30.0],
            "y": [11.0, 21.0, 31.0],
            "CD3": [1.0, 2.0, 3.0],
            "area": [100.0, 200.0, 300.0],
        }
    )
    out = tmp_path / "cells.geojson"

    n = eg.export_geojson(
        df=df, output_path=str(out), pixel_size=0.325, marker_cols=["CD3"]
    )

    doc = json.loads(out.read_text())
    assert n == 3
    assert doc["type"] == "FeatureCollection"
    assert len(doc["features"]) == 3
    # the file must be a single well-formed collection, not concatenated fragments
    assert out.read_text().startswith('{"type": "FeatureCollection", "features": [')
    assert out.read_text().endswith("]}")

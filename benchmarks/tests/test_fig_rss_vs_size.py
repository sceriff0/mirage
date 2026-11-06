"""Fig. 2c (benchmarks.figures.fig_rss_vs_size): both VALIS series and STARE are drawn."""

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from benchmarks.figures import fig_rss_vs_size as fig

SIZES = (4096, 16384, 65536)
# peak GB per size: STARE flat, VALIS defaults flat (fixed working size), hires growing
PEAK = {
    "stare": (0.6, 0.7, 2.5),
    "valis120": (21, 23, 25),
    "valis120_hires": (7, 40, 209),
}


def _synthetic(tmp_path):
    rows = []
    for method, peaks in PEAK.items():
        for n, gb in zip(SIZES, peaks):
            for patient in ("p1", "p2"):
                rows.append(
                    dict(
                        dataset="synthetic",
                        method=method,
                        group=f"{patient}_{n}",
                        case_id=f"scale_{n}_{patient}_r1",
                        ok=True,
                        cause="",
                        moving_gpx=n * n / 1e9,
                        peak_rss_gb=gb,
                        unit="",
                        launch_id="",
                    )
                )
    d = tmp_path / "tables"
    d.mkdir()
    pd.DataFrame(rows).to_csv(d / "resources.csv", index=False)
    return d


def test_both_valis_series_and_stare_are_drawn(tmp_path, capsys):
    out = tmp_path / "fig"
    fig.main(
        [
            "--synthetic",
            str(_synthetic(tmp_path)),
            "--node-gb",
            "200",
            "--outdir",
            str(out),
        ]
    )
    data = pd.read_csv(out / "rss_vs_size_source_data.csv")
    assert set(data["label"]) == {"STARE", "VALIS", fig.HIRES}
    assert set(data.loc[data["label"] == fig.HIRES, "variant"]) == {"valis120_hires"}
    caption = (out / "rss_vs_size_caption.txt").read_text()
    assert "VALIS median peak per size:" in caption
    assert f"{fig.HIRES} median peak per size:" in caption
    assert "65536 px: 209.0 GB" in caption  # the hires example crosses the 200 GB line


def test_an_empty_valis_hires_draws_only_the_default(tmp_path):
    out = tmp_path / "fig"
    fig.main(
        [
            "--synthetic",
            str(_synthetic(tmp_path)),
            "--valis-hires",
            "",
            "--outdir",
            str(out),
        ]
    )
    data = pd.read_csv(out / "rss_vs_size_source_data.csv")
    assert set(data["label"]) == {"STARE", "VALIS"}

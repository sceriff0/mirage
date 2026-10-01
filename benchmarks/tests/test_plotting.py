import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np

from benchmarks.analysis.lib import plotting


def test_save_fig_writes_pdf_and_svg(tmp_path):
    import matplotlib.pyplot as plt

    plotting.set_paper_theme()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = plotting.save_fig(fig, tmp_path / "fig1")
    assert (tmp_path / "fig1.pdf").exists() and (tmp_path / "fig1.svg").exists()
    assert (tmp_path / "fig1.pdf").stat().st_size > 0
    assert set(paths) == {tmp_path / "fig1.pdf", tmp_path / "fig1.svg"}


def test_scatter_with_fit_returns_figure(tmp_path):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    fig = plotting.scatter_with_fit(
        x,
        y,
        slope=2.0,
        intercept=1.0,
        xlabel="input (GB)",
        ylabel="peak RSS (GB)",
        title="SEGMENT",
    )
    assert fig is not None
    plotting.save_fig(fig, tmp_path / "scatter")  # must not raise
    assert (tmp_path / "scatter.pdf").exists()


def test_save_fig_formats_arg_writes_only_requested(tmp_path):
    import matplotlib.pyplot as plt

    plotting.set_paper_theme()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    paths = plotting.save_fig(fig, tmp_path / "png_only", formats=("png",))
    assert (tmp_path / "png_only.png").exists()
    assert not (tmp_path / "png_only.pdf").exists()
    assert paths == [tmp_path / "png_only.png"]


def test_save_fig_default_still_pdf_svg(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plotting.save_fig(fig, tmp_path / "d")
    assert (tmp_path / "d.pdf").exists() and (tmp_path / "d.svg").exists()

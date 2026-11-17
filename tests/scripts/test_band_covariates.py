"""Tests for the per-band covariate figure (``*_band_covariates.png``).

One figure per band with the light curve and each covariate on its own axis,
so systematics can be read against flux in physical units (cf. the stacked,
normalized multi-band ``*_covariates.png``).
"""

import numpy as np
import pandas as pd
import pytest

from prose.scripts import run_photometry as rp

N_POINTS = 60


class _FakeDiff:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.time = df["time"].to_numpy()


def _make_df(n: int = N_POINTS, drop: tuple[str, ...] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "time": np.linspace(2460000.5, 2460000.6, n),
            "flux": 1 + 1e-3 * rng.standard_normal(n),
            "airmass": np.linspace(1.2, 1.5, n),
            "dx": rng.standard_normal(n),
            "dy": rng.standard_normal(n),
            "fwhm": np.full(n, 4.0),
            "peak": np.full(n, 15000.0),
            "bkg": np.full(n, 100.0),
        }
    )
    return df.drop(columns=list(drop))


@pytest.fixture
def captured(monkeypatch):
    figs = {}

    def _fake_savefig(fig, path):
        figs["fig"] = fig
        figs["path"] = path

    monkeypatch.setattr(rp, "_savefig", _fake_savefig)
    return figs


def _plot(df, tmp_path, unit="ADU"):
    rp.plot_band_covariates(
        {"diff": _FakeDiff(df)},
        tmp_path / "x_band_covariates.png",
        target_name="TOI-1",
        instrument="muscat4",
        band="rp",
        date="261117",
        target_index=0,
        bin_size_days=10 / 60 / 24,
        unit=unit,
    )


def test_one_axis_per_signal_sharing_time(tmp_path, captured):
    _plot(_make_df(), tmp_path)

    axes = captured["fig"].axes
    ylabels = [ax.get_ylabel() for ax in axes if ax.get_ylabel()]
    assert ylabels == [
        "Flux",
        "Airmass",
        "dx (pix)",
        "dy (pix)",
        "FWHM (pix)",
        "Peak (ADU)",
        "Bkg (ADU)",
    ]
    main_axes = axes[: len(ylabels)]
    assert all(ax.get_shared_x_axes().joined(main_axes[0], ax) for ax in main_axes)
    assert main_axes[-1].get_xlabel() == "time (JD) - 2460000"


def test_every_panel_has_minor_ticks(tmp_path, captured):
    from matplotlib.ticker import NullLocator

    _plot(_make_df(), tmp_path)

    panels = [ax for ax in captured["fig"].axes if ax.get_ylabel()]
    for ax in panels:
        assert not isinstance(ax.xaxis.get_minor_locator(), NullLocator)
        assert not isinstance(ax.yaxis.get_minor_locator(), NullLocator)


def test_pixel_unit_follows_the_csv(tmp_path, captured):
    _plot(_make_df(), tmp_path, unit="e-")

    ylabels = [ax.get_ylabel() for ax in captured["fig"].axes]
    assert "Peak (e-)" in ylabels
    assert "Bkg (e-)" in ylabels


def test_missing_covariate_is_skipped(tmp_path, captured):
    _plot(_make_df(drop=("airmass", "bkg")), tmp_path)

    ylabels = [ax.get_ylabel() for ax in captured["fig"].axes if ax.get_ylabel()]
    assert "Airmass" not in ylabels
    assert "Bkg (ADU)" not in ylabels
    assert ylabels[0] == "Flux"


def test_title_names_band_and_binned_scatter(tmp_path, captured):
    _plot(_make_df(), tmp_path)

    fig = captured["fig"]
    assert "rp" in fig._suptitle.get_text()
    assert "tID=0" in fig._suptitle.get_text()
    flux_title = fig.axes[0].get_title()
    assert "ppt" in flux_title
    assert "10-min bin" in flux_title


def test_writes_png(tmp_path):
    out = tmp_path / "x_band_covariates.png"
    rp.plot_band_covariates(
        {"diff": _FakeDiff(_make_df())},
        out,
        target_name="TOI-1",
        instrument="muscat4",
        band="rp",
        date="261117",
        target_index=0,
    )
    assert out.is_file()
    assert out.stat().st_size > 0

"""Unit tests for the pure helpers in ``prose.scripts.run_photometry``.

The reduction itself requires real FITS data and network catalog access, so
these tests focus on the deterministic, side-effect-free helpers (naming,
header parsing, z-scaling, CSV column mapping).
"""

import numpy as np
import pandas as pd
import pytest

from prose.scripts import run_photometry as rp


@pytest.mark.parametrize(
    "header, expected",
    [
        ({"TELID": "2m0a", "SITEID": "coj"}, "muscat4"),
        ({"TELID": "2m0a", "SITEID": "ogg"}, "muscat3"),
        ({"TELID": "1m0a", "SITEID": "lsc"}, "sinistro"),
        ({"TELID": "0m4a", "SITEID": "cpt"}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_get_instrument(header, expected):
    assert rp.get_instrument(header) == expected


@pytest.mark.parametrize(
    "day_obs, expected",
    [("20250416", "250416"), ("2025-04-16", "250416")],
)
def test_date_from_header(day_obs, expected):
    assert rp.date_from_header({"DAY-OBS": day_obs}) == expected


def test_build_stem_strips_spaces_and_handles_band():
    assert rp.build_stem("TOI 6715", "muscat4", "250416") == "TOI6715_muscat4_250416"
    assert (
        rp.build_stem("TOI 6715", "muscat4", "250416", "gp")
        == "TOI6715_muscat4_gp_250416"
    )


def test_zscale_is_bounded_unit_interval():
    rng = np.random.default_rng(0)
    data = rng.normal(1000, 50, size=(32, 32))
    scaled = rp._zscale(data)
    assert scaled.shape == data.shape
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


def test_zscale_handles_constant_image_without_nan():
    scaled = rp._zscale(np.full((8, 8), 5.0))
    assert np.all(np.isfinite(scaled))


def test_radial_profile_peaks_at_center():
    data = np.zeros((21, 21))
    data[10, 10] = 100.0
    prof = rp._radial_profile(data, center=(10, 10))
    assert prof[0] == pytest.approx(100.0)
    assert prof[1:].sum() == pytest.approx(0.0)


def test_gif_frame_returns_uint8_rgb():
    rng = np.random.default_rng(0)
    data = rng.normal(1000, 50, size=(40, 30))
    frame = rp._gif_frame(data, label="2025-04-16T00:00:00")
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3  # RGB
    # no downsampling below the max-size threshold; rows flipped, cols kept
    assert frame.shape[:2] == (40, 30)


def test_gif_frame_downsamples_large_image_preserving_aspect():
    data = np.zeros((1000, 500))
    frame = rp._gif_frame(data, max_px=100)
    assert max(frame.shape[:2]) == 100  # longest side clamped to max_px
    assert frame.shape[:2] == (100, 50)  # aspect ratio preserved


class _FakeDiff:
    """Minimal stand-in for a differential ``Fluxes`` object."""

    def __init__(self, df, error):
        self._df = df
        self.error = error

    @property
    def df(self):
        return self._df.copy()


def test_photometry_df_renames_columns_and_adds_bjd():
    n = 4
    base = pd.DataFrame(
        {
            "time": np.linspace(2460844.0, 2460844.1, n),
            "flux": np.ones(n),
            "airmass": np.full(n, 1.04),
            "dx": np.zeros(n),
            "dy": np.zeros(n),
            "bkg": np.full(n, 12.0),
            "fwhm": np.full(n, 8.5),
            "peak": np.full(n, 15000.0),
        }
    )
    bjd = base["time"].to_numpy() + 0.0005
    out = rp.photometry_df(_FakeDiff(base, error=np.full(n, 0.001)), bjd)

    for col in (
        "GJD_UTC",
        "BJD_TDB",
        "Flux",
        "Flux_Err",
        "Airmass",
        "Dx(pix)",
        "Dy(pix)",
        "Bkg(ADU)",
        "FWHM(pix)",
        "Peak(ADU)",
    ):
        assert col in out.columns
    np.testing.assert_allclose(out["BJD_TDB"], bjd)
    np.testing.assert_allclose(out["Flux_Err"], 0.001)

    assert out.columns.tolist()[:3] == ["BJD_TDB", "Flux", "Flux_Err"], (
        "BJD_TDB, Flux, Flux_Err must be the three leftmost columns"
    )


# --------------------------- aperture-grid CLI parsing ---------------------------


def test_parse_aper_grid_is_inclusive_of_max():
    np.testing.assert_allclose(rp.parse_aper_grid("10,20,2"), [10, 12, 14, 16, 18, 20])


def test_parse_aper_grid_fractional_step():
    np.testing.assert_allclose(rp.parse_aper_grid("1,2,0.5"), [1.0, 1.5, 2.0])


@pytest.mark.parametrize("bad", ["10,20", "10,20,2,3", "a,b,c", "10,5,2", "10,20,0"])
def test_parse_aper_grid_rejects_bad_input(bad):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        rp.parse_aper_grid(bad)


def test_parse_pair_valid():
    assert rp.parse_pair("24,30") == (24.0, 30.0)


@pytest.mark.parametrize("bad", ["24", "24,30,5", "a,b", "30,24", "24,24"])
def test_parse_pair_rejects_bad_input(bad):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        rp.parse_pair(bad)


def test_aper_radii_pix_pixel_unit_is_identity():
    r = {
        "aper_radii": np.array([10.0, 12.0]),
        "rin": 20.0,
        "rout": 30.0,
        "scale": False,
    }
    radii, rin, rout = rp.aper_radii_pix(r)
    np.testing.assert_allclose(radii, [10.0, 12.0])
    assert (rin, rout) == (20.0, 30.0)


def test_aper_radii_pix_fwhm_unit_scales_by_reference_fwhm():
    class _Ref:
        fwhm = 4.0

    r = {
        "aper_radii": np.array([1.0, 2.0]),
        "rin": 3.0,
        "rout": 5.0,
        "scale": True,
        "ref": _Ref(),
    }
    radii, rin, rout = rp.aper_radii_pix(r)
    np.testing.assert_allclose(radii, [4.0, 8.0])
    assert (rin, rout) == (12.0, 20.0)


# --------------------------- SIMBAD overlay on ref plot ---------------------------


def test_plot_ref_image_handles_empty_simbad(tmp_path, monkeypatch):
    from astropy.io import fits
    from astropy.wcs import WCS

    monkeypatch.setattr(rp, "get_simbad_data", lambda *a, **kw: None)

    w = WCS(naxis=2)
    w.wcs.crpix = [10, 10]
    w.wcs.cdelt = [0.01, 0.01]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    data = np.ones((20, 20))
    hdr = w.to_header()
    hdr["TELESCOP"] = "2m0a"
    hdr["INSTRUME"] = "ep09"
    hdr["SITEID"] = "coj"
    hdr["OBJECT"] = "test"
    hdr["EXPTIME"] = 1
    hdr["FILTER"] = "gp"
    hdr["AIRMASS"] = 1.0
    hdr["JD"] = 2460000.0
    hdr["DATE-OBS"] = "2025-04-16T00:00:00"
    fpath = tmp_path / "test.fits"
    fits.writeto(fpath, data, header=hdr)

    from prose import FITSImage
    from prose.core.source import Sources

    ref = FITSImage(fpath)
    ref.sources = Sources(np.array([[10, 10]]))
    r = {"ref": ref, "band": "gp"}
    target_coord = __import__("astropy").coordinates.SkyCoord(0, 0, unit="deg")
    out = tmp_path / "ref.png"
    rp.plot_ref_image(r, target_coord, "muscat4", out)
    assert out.exists()

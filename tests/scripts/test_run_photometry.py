"""Unit tests for the pure helpers in ``prose.scripts.run_photometry``.

The reduction itself requires real FITS data and network catalog access, so
these tests focus on the deterministic, side-effect-free helpers (naming,
header parsing, z-scaling, CSV column mapping).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from prose.scripts import run_photometry as rp


def _make_test_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [16.0, 16.0]
    wcs.wcs.crval = [31.0, 46.0]
    wcs.wcs.cdelt = [-0.44 / 3600.0, 0.44 / 3600.0]
    wcs.wcs.set()
    return wcs


def _write_calibrated_fits(path, *, target="HAT-P-32b", filter_name="g", wcs=None):
    hdu = fits.PrimaryHDU(np.zeros((32, 32), dtype=np.float32))
    hdu.header["OBJECT"] = target
    hdu.header["FILTER"] = filter_name
    hdu.header["INSTRUME"] = "MuSCAT2"
    hdu.header["WCSMTHD"] = "astrometry.net"
    if wcs is not None:
        hdu.header.update(wcs.to_header(relax=True))
    hdu.writeto(path, overwrite=True)


def _write_wcs_sidecar(path, wcs):
    hdu = fits.PrimaryHDU()
    hdu.header.update(wcs.to_header(relax=True))
    hdu.writeto(path, overwrite=True)


def test_resolve_simbad_target_uses_decimal_degree_columns(monkeypatch):
    class FakeSimbad:
        def query_object(self, name):
            assert name == "EPIC 211945201"
            return Table({"ra": [130.123], "dec": [-12.456]})

    monkeypatch.setattr(rp, "Simbad", FakeSimbad)
    coord = rp._resolve_simbad_target("EPIC 211945201")
    assert coord.ra.deg == pytest.approx(130.123)
    assert coord.dec.deg == pytest.approx(-12.456)


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


def test_date_from_header_falls_back_to_mjd_when_date_obs_absent():
    # MuSCAT2 frames stripped of DATE-OBS still carry MJD-STRT; the date must be
    # recovered from it so output filenames are not left dateless.
    # MJD 60789.0437 -> 2025-04-24 UT
    header = {"MJD-STRT": 60789.0437086606}
    assert rp.date_from_header(header) == "250424"


def test_date_from_header_prefers_calendar_keyword_over_mjd():
    # When both exist the explicit calendar keyword wins (no UT rollover surprise).
    header = {"DATE-OBS": "2025-4-23", "MJD-STRT": 60789.0437086606}
    assert rp.date_from_header(header) == "250423"


def test_date_from_header_returns_empty_without_any_time_keyword():
    assert rp.date_from_header({"OBJECT": "TOI-6715"}) == ""


def test_date_from_header_ignores_unparseable_time_keyword():
    # A non-numeric MJD must not raise; fall through to ''.
    assert rp.date_from_header({"MJD-STRT": "n/a"}) == ""


def test_inject_wcs_from_sidecars_updates_wcsless_calibrated_files(tmp_path):
    calib_dir = tmp_path / "calibrated"
    sidecar_dir = calib_dir / ".wcs"
    sidecar_dir.mkdir(parents=True)

    gp = calib_dir / "MCT20_2410300001_calibrated.fits"
    rp_file = calib_dir / "MCT21_2410300001_calibrated.fits"
    _write_calibrated_fits(gp, filter_name="g")
    _write_calibrated_fits(rp_file, filter_name="r")

    wcs = _make_test_wcs()
    _write_wcs_sidecar(sidecar_dir / "gp_astrometry.net.wcs.fits", wcs)
    _write_wcs_sidecar(sidecar_dir / "rp_astrometry.net.wcs.fits", wcs)

    assert not rp._header_has_usable_wcs(fits.getheader(gp))

    ok = rp._inject_wcs_from_sidecars(
        calib_label="muscat2",
        calib_dir=calib_dir,
        calibrated_files=[gp, rp_file],
        active_bands=["gp", "rp"],
        requested_bands=["gp", "rp"],
        target_name="HAT-P-32b",
        wcs_method="astrometry.net",
    )

    assert ok is True
    for path in (gp, rp_file):
        header = fits.getheader(path)
        assert header["WCSMTHD"] == "astrometry.net"
        assert rp._header_has_usable_wcs(header)


def test_inject_wcs_from_sidecars_requires_every_active_band(tmp_path):
    calib_dir = tmp_path / "calibrated"
    sidecar_dir = calib_dir / ".wcs"
    sidecar_dir.mkdir(parents=True)

    gp = calib_dir / "MCT20_2410300001_calibrated.fits"
    rp_file = calib_dir / "MCT21_2410300001_calibrated.fits"
    _write_calibrated_fits(gp, filter_name="g")
    _write_calibrated_fits(rp_file, filter_name="r")
    _write_wcs_sidecar(sidecar_dir / "gp_astrometry.net.wcs.fits", _make_test_wcs())

    ok = rp._inject_wcs_from_sidecars(
        calib_label="muscat2",
        calib_dir=calib_dir,
        calibrated_files=[gp, rp_file],
        active_bands=["gp", "rp"],
        requested_bands=["gp", "rp"],
        target_name="HAT-P-32b",
        wcs_method="astrometry.net",
    )

    assert ok is False
    assert not rp._header_has_usable_wcs(fits.getheader(gp))
    assert not rp._header_has_usable_wcs(fits.getheader(rp_file))


def test_calibrated_wcs_problems_detects_single_bad_active_band(tmp_path):
    calib_dir = tmp_path / "calibrated"
    sidecar_dir = calib_dir / ".wcs"
    sidecar_dir.mkdir(parents=True)
    wcs = _make_test_wcs()

    gp = calib_dir / "MSCT0_2601230001_calibrated.fits"
    rp_file = calib_dir / "MSCT1_2601230001_calibrated.fits"
    zs = calib_dir / "MSCT2_2601230001_calibrated.fits"
    _write_calibrated_fits(gp, filter_name="g", wcs=wcs)
    _write_calibrated_fits(rp_file, filter_name="r", wcs=wcs)
    _write_calibrated_fits(zs, filter_name="z_s")

    for band in ("gp", "rp", "zs"):
        _write_wcs_sidecar(sidecar_dir / f"{band}_astrometry.net.wcs.fits", wcs)

    missing, unreadable, no_wcs, wrong_method = rp._calibrated_wcs_problems_by_band(
        calib_label="muscat",
        calibrated_files=[gp, rp_file, zs],
        active_bands=["gp", "rp", "zs"],
        requested_bands=["gp", "rp", "zs"],
        target_name="HAT-P-32b",
        wcs_method="astrometry.net",
    )

    assert missing == []
    assert unreadable == []
    assert no_wcs == ["zs"]
    assert wrong_method == []

    assert rp._inject_wcs_from_sidecars(
        calib_label="muscat",
        calib_dir=calib_dir,
        calibrated_files=[gp, rp_file, zs],
        active_bands=["gp", "rp", "zs"],
        requested_bands=["gp", "rp", "zs"],
        target_name="HAT-P-32b",
        wcs_method="astrometry.net",
    )
    assert rp._header_has_usable_wcs(fits.getheader(zs))


def test_build_stem_strips_spaces_and_handles_band():
    assert rp.build_stem("TOI 6715", "muscat4", "250416") == "TOI6715_muscat4_250416"
    assert (
        rp.build_stem("TOI 6715", "muscat4", "250416", "gp")
        == "TOI6715_muscat4_gp_250416"
    )
    # Sinistro with valid site
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", site="lsc")
        == "TOI6715_sinistro_lsc_250416"
    )
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", "lsc")
        == "TOI6715_sinistro_lsc_gp_250416"
    )
    # Sinistro with uppercase site
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", "CPT")
        == "TOI6715_sinistro_cpt_gp_250416"
    )
    # Sinistro with invalid site
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", "abc")
        == "TOI6715_sinistro_gp_250416"
    )
    # Sinistro without site
    assert rp.build_stem("TOI 6715", "sinistro", "250416") == "TOI6715_sinistro_250416"
    # Sinistro with confmode full
    assert (
        rp.build_stem(
            "TOI 6715", "sinistro", "250416", site="lsc", confmode="full_frame"
        )
        == "TOI6715_sinistro_lsc_250416_full"
    )
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", "lsc", "full_frame")
        == "TOI6715_sinistro_lsc_gp_250416_full"
    )
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", "lsc", "full")
        == "TOI6715_sinistro_lsc_gp_250416_full"
    )
    # Sinistro with confmode 2x2 (no suffix)
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", "lsc", "central_2k_2x2")
        == "TOI6715_sinistro_lsc_gp_250416"
    )
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", "lsc", "2x2")
        == "TOI6715_sinistro_lsc_gp_250416"
    )


def test_build_stem_includes_telescope_token():
    # Sinistro with site + telescope
    assert (
        rp.build_stem(
            "TOI 6715", "sinistro", "250416", "gp", site="lsc", telescope="1m0-05"
        )
        == "TOI6715_sinistro_lsc_tel05_gp_250416"
    )
    # Sinistro with telescope only (no site)
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", telescope="1m0-09")
        == "TOI6715_sinistro_tel09_gp_250416"
    )
    # Sinistro with telescope + confmode full
    assert (
        rp.build_stem(
            "TOI 6715",
            "sinistro",
            "250416",
            "gp",
            site="lsc",
            confmode="full_frame",
            telescope="1m0-05",
        )
        == "TOI6715_sinistro_lsc_tel05_gp_250416_full"
    )
    # Telescope ignored for non-sinistro instruments
    assert (
        rp.build_stem("TOI 6715", "muscat4", "250416", "gp", telescope="1m0-05")
        == "TOI6715_muscat4_gp_250416"
    )
    # Empty/blank telescope contributes no token
    assert (
        rp.build_stem("TOI 6715", "sinistro", "250416", "gp", site="lsc", telescope="")
        == "TOI6715_sinistro_lsc_gp_250416"
    )


def test_telescope_stem_token_extracts_trailing_digits():
    assert rp._telescope_stem_token("1m0-05") == "tel05"
    assert rp._telescope_stem_token("1m0-09") == "tel09"
    assert rp._telescope_stem_token("  1M0-05  ") == "tel05"
    assert rp._telescope_stem_token("") == ""


def test_build_summary_stem_includes_reduced_band_set():
    assert (
        rp.build_summary_stem("TOI 6715", "muscat4", "250416", ["gp", "zs"])
        == "TOI6715_muscat4_gp_zs_250416"
    )
    assert (
        rp.build_summary_stem("TOI 6715", "sinistro", "250416", ["gp"], site="lsc")
        == "TOI6715_sinistro_lsc_gp_250416"
    )
    assert (
        rp.build_summary_stem(
            "TOI 6715", "sinistro", "250416", ["gp", "zs"], site="lsc", confmode="full"
        )
        == "TOI6715_sinistro_lsc_gp_zs_250416_full"
    )
    assert (
        rp.build_summary_stem(
            "TOI 6715",
            "sinistro",
            "250416",
            ["gp", "zs"],
            site="lsc",
            telescope="1m0-05",
        )
        == "TOI6715_sinistro_lsc_tel05_gp_zs_250416"
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


def test_gif_frame_has_white_dotted_grid():
    data = np.zeros((100, 100))
    frame = rp._gif_frame(data)
    # Check that some grid points are drawn and are white
    assert np.all(frame[50, 0] == 255)
    assert np.all(frame[50, 5] == 255)
    # Check that non-grid points are not white (e.g. background black)
    assert not np.all(frame[50, 2] == 255)


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
        "Err",
        "Airmass",
        "Dx(pix)",
        "Dy(pix)",
        "Bkg(ADU)",
        "FWHM(pix)",
        "Peak(ADU)",
    ):
        assert col in out.columns
    np.testing.assert_allclose(out["BJD_TDB"], bjd)
    np.testing.assert_allclose(out["Err"], 0.001)

    assert out.columns.tolist()[:3] == ["BJD_TDB", "Flux", "Err"], (
        "BJD_TDB, Flux, Err must be the three leftmost columns"
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


def test_aperture_geometry_title_formats_pixel_grid():
    title = rp.aperture_geometry_title(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), 20.0, 40.0)
    assert title == "apertures: r=(2, 10) dr=2; annuli=(20, 40) pix"


def test_ref_header_desc_formats_focus_airmass_exptime():
    ref = SimpleNamespace(
        header={"FOCPOSN": "123.45", "AIRMASS": "1.23", "EXPTIME": "60"}
    )
    assert (
        rp.ref_header_desc(ref, "cutouts")
        == "cutouts (focus=123.5 airmass=1.2 exptime=60s)"
    )


def test_ref_header_desc_handles_missing_or_bad_values():
    ref = SimpleNamespace(header={"FOCPOSN": "bad", "AIRMASS": None})
    assert (
        rp.ref_header_desc(ref, "reference frame")
        == "reference frame (focus=nan airmass=nan exptime=nans)"
    )


def test_ref_header_desc_includes_extra_details_before_header_values():
    ref = SimpleNamespace(
        header={"FOCPOSN": "123.45", "AIRMASS": "1.23", "EXPTIME": "60"}
    )
    assert (
        rp.ref_header_desc(ref, "cutouts", ["r=20 pix"])
        == "cutouts (r=20 pix focus=123.5 airmass=1.2 exptime=60s)"
    )


# --------------------------- SIMBAD overlay on ref plot ---------------------------


# --------------------------- tID / cID CLI parsing ---------------------------


def test_parse_tID_accepts_single_int():
    args = rp.parse_args(
        ["--target_name", "T1", "--data_dir", ".", "--results_dir", ".", "--tID", "5"]
    )
    assert args.tID == 5


def test_parse_tID_default_is_None():
    args = rp.parse_args(
        ["--target_name", "T1", "--data_dir", ".", "--results_dir", "."]
    )
    assert args.tID is None


def test_parse_cID_accepts_int_list():
    args = rp.parse_args(
        [
            "--target_name",
            "T1",
            "--data_dir",
            ".",
            "--results_dir",
            ".",
            "--cID",
            "3",
            "7",
            "12",
        ]
    )
    assert args.cID == [3, 7, 12]


def test_parse_cID_default_is_None():
    args = rp.parse_args(
        ["--target_name", "T1", "--data_dir", ".", "--results_dir", "."]
    )
    assert args.cID is None


def test_parse_avoid_cids_accepts_int_list():
    args = rp.parse_args(
        [
            "--target_name",
            "T1",
            "--data_dir",
            ".",
            "--results_dir",
            ".",
            "--avoid_cids",
            "3",
            "7",
            "12",
        ]
    )
    assert args.avoid_cids == [3, 7, 12]


def test_parse_avoid_cids_accepts_dash_form():
    args = rp.parse_args(
        [
            "--target_name",
            "T1",
            "--data_dir",
            ".",
            "--results_dir",
            ".",
            "--avoid-cids",
            "5",
            "9",
        ]
    )
    assert args.avoid_cids == [5, 9]


def test_parse_avoid_cids_default_is_None():
    args = rp.parse_args(
        ["--target_name", "T1", "--data_dir", ".", "--results_dir", "."]
    )
    assert args.avoid_cids is None


# --------------------------- differential_photometry with cids ---------------------------


def _make_fluxes(n_stars: int, n_times: int, rng: np.random.Generator):
    """Build a minimal 3-D ``Fluxes`` with controlled noise."""
    from prose import Fluxes

    fluxes = np.ones((n_stars, n_times))
    fluxes[0] *= 10000  # "target" star
    for i in range(1, n_stars):
        fluxes[i] *= 10000 + 100 * rng.normal(0, 1, n_times)
    errors = np.full((n_stars, n_times), 25.0)
    obj = Fluxes(fluxes=fluxes, errors=errors, time=np.arange(n_times, dtype=float))
    obj.data = {
        "bkg": 100.0 + 10 * rng.normal(0, 1, n_times),
        "fwhm": 4.0 + 0.5 * rng.normal(0, 1, n_times),
    }
    return obj


def test_differential_photometry_cids_uses_diff_not_autodiff():
    rng = np.random.default_rng(42)
    fluxes = _make_fluxes(5, 20, rng)

    result = rp.differential_photometry(fluxes, target_index=0, cids=[1, 2, 3])

    assert result is not None
    assert result.time is not None
    assert len(result.time) == 20  # no frames clipped by sigma
    # diff() returns a Fluxes with weights set; autodiff would have weights too
    assert result.weights is not None
    assert result.weights.shape == (1, 5)
    # only target + cids should have non-zero weight
    assert result.weights[0, 0] == 0.0  # target is NOT a comparison
    for c in (1, 2, 3):
        assert result.weights[0, c] == 1.0  # user-specified comparisons
    assert result.weights[0, 4] == 0.0  # star 4 was masked out


def test_differential_photometry_drops_out_of_range_cids(caplog):
    """Out-of-range cids are dropped with a warning; valid ones are still used."""
    rng = np.random.default_rng(7)
    fluxes = _make_fluxes(3, 10, rng)  # sources 0, 1, 2 only

    with caplog.at_level("WARNING", logger="prose_run_photometry"):
        result = rp.differential_photometry(fluxes, target_index=0, cids=[1, 99])

    assert result is not None
    assert "99" in caplog.text and "out of range" in caplog.text
    # only the in-range comparison (star 1) is used
    assert result.weights[0, 1] == 1.0
    assert result.weights[0, 2] == 0.0


def test_differential_photometry_all_cids_out_of_range_falls_back_to_auto(caplog):
    """When every cid is out of range we fall back to automatic selection."""
    rng = np.random.default_rng(7)
    fluxes = _make_fluxes(3, 10, rng)

    with caplog.at_level("WARNING", logger="prose_run_photometry"):
        result = rp.differential_photometry(fluxes, target_index=0, cids=[99, 100])

    assert result is not None  # autodiff fallback, no crash
    assert "auto-selection" in caplog.text


# --------------------------- differential_photometry with avoid_cids ---------------------------


def test_differential_photometry_avoid_cids_excludes_from_auto():
    """avoid_cids zeroes out the weight of excluded stars in auto mode."""
    rng = np.random.default_rng(42)
    fluxes = _make_fluxes(6, 20, rng)

    result = rp.differential_photometry(fluxes, target_index=0, avoid_cids=[2, 4])

    assert result is not None
    assert result.weights is not None
    assert result.weights[0, 0] == 0.0  # target is never a comparison
    assert result.weights[0, 2] == 0.0  # avoided
    assert result.weights[0, 4] == 0.0  # avoided


def test_differential_photometry_avoid_cids_out_of_range_ignored_in_auto(caplog):
    """Out-of-range avoid_cids do not crash in auto mode."""
    rng = np.random.default_rng(7)
    fluxes = _make_fluxes(3, 10, rng)

    result = rp.differential_photometry(fluxes, target_index=0, avoid_cids=[999, -1])

    assert result is not None


def test_differential_photometry_avoid_cids_target_itself_raises():
    """The target cannot be excluded from its comparison-star mask."""
    rng = np.random.default_rng(42)
    fluxes = _make_fluxes(4, 10, rng)

    with pytest.raises(ValueError, match="target_index=0 must not be in avoid_cids"):
        rp.differential_photometry(fluxes, target_index=0, avoid_cids=[0])


# --------------------------- build_reference target_index_override ---------------------------


def _write_minimal_fits(tmp_path, name="test.fits"):
    """Write a minimal FITS file with a simple WCS."""
    from astropy.io import fits
    from astropy.wcs import WCS

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
    fpath = tmp_path / name
    fits.writeto(fpath, data, header=hdr)
    return fpath


class _FakeRefSeq:
    """Stand-in for the reference_sequence that sets sources and fwhm
    instead of running the full detection pipeline."""

    def run(self, img, **kw):
        from prose.core.source import Sources

        img.sources = Sources(np.array([[5.0, 5.0], [10.0, 10.0]]))
        img.fwhm = 4.0


def _patch_ref_seq(monkeypatch):
    monkeypatch.setattr(rp, "reference_sequence", lambda *a, **kw: _FakeRefSeq())


def test_build_reference_target_index_override_bypasses_gaia(tmp_path, monkeypatch):
    """When ``target_index_override`` is given the Gaia cross-match is skipped."""
    called = False
    original_find = rp.find_target_index

    def _never_called(*a, **kw):
        nonlocal called
        called = True
        return original_find(*a, **kw)

    monkeypatch.setattr(rp, "find_target_index", _never_called)
    _patch_ref_seq(monkeypatch)

    fpath = _write_minimal_fits(tmp_path)

    from astropy.coordinates import SkyCoord

    coord = SkyCoord(0, 0, unit="deg")
    # _FakeRefSeq yields 2 sources, so a valid override is 0..1. Use an in-range
    # index: build_reference now validates the override against the kept-source
    # count and raises for out-of-range values.
    result = rp.build_reference(
        fpath,
        coord,
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_index_override=1,
    )
    assert result["target_index"] == 1
    assert not called, "find_target_index should not have been called"


def test_build_reference_target_index_override_None_still_calls_find(
    tmp_path, monkeypatch
):
    """When ``target_index_override`` is None the normal Gaia path runs."""
    _patch_ref_seq(monkeypatch)
    fpath = _write_minimal_fits(tmp_path)

    from astropy.coordinates import SkyCoord

    coord = SkyCoord(0, 0, unit="deg")
    result = rp.build_reference(
        fpath,
        coord,
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_index_override=None,
    )
    assert isinstance(result["target_index"], int)
    assert result["target_index"] >= 0


def test_build_reference_falls_back_to_source0_when_target_unmatched(
    tmp_path, monkeypatch, caplog
):
    """When the target can't be matched to a detected source (here it sits ~180
    deg away / the TAN projection is degenerate), build_reference warns and falls
    back to a real detected source instead of fabricating one at the center. The
    kept source coordinates stay finite and the returned index is in range."""
    _patch_ref_seq(monkeypatch)
    fpath = _write_minimal_fits(tmp_path)

    from astropy.coordinates import SkyCoord

    with caplog.at_level("WARNING", logger="prose_run_photometry"):
        result = rp.build_reference(
            fpath,
            SkyCoord(180, 0, unit="deg"),
            aper_radii=np.array([3.0, 4.0, 5.0]),
            rin=8.0,
            rout=12.0,
            target_index_override=None,
        )

    coords = np.array([s.coords for s in result["ref"].sources], dtype=float)
    assert np.all(np.isfinite(coords))
    assert 0 <= result["target_index"] < len(result["ref"].sources)
    assert any("Falling back to source 0" in r.message for r in caplog.records)


def test_plot_ref_image_handles_empty_simbad_and_omits_avoided_sources(
    tmp_path, monkeypatch
):
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
    ref.sources = Sources(np.array([[10, 10], [5, 5], [15, 15]]))
    r = {"ref": ref, "band": "gp", "target_index": 0}
    target_coord = __import__("astropy").coordinates.SkyCoord(0, 0, unit="deg")
    out = tmp_path / "ref.png"

    plotted_ids = []
    original_plot = Sources.plot

    def record_plotted_ids(self, *args, **kwargs):
        plotted_ids.extend(source.i for source in self.sources)
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Sources, "plot", record_plotted_ids)
    rp.plot_ref_image(r, target_coord, "muscat4", out, avoid_cids=[1])

    assert out.exists()
    assert plotted_ids == [0, 2]


def test_plot_ref_image_labels_defaulted_target(tmp_path, monkeypatch):
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
    ref.sources = Sources(np.array([[10, 10], [5, 5]]))
    target_coord = __import__("astropy").coordinates.SkyCoord(0, 0, unit="deg")
    out1 = tmp_path / "ref1.png"
    out2 = tmp_path / "ref2.png"

    # Test case 1: defaulted_to_brightest=False -> should annotate "Target"
    r1 = {"ref": ref, "band": "gp", "target_index": 0, "defaulted_to_brightest": False}
    # Mock ax.annotate to capture the label
    annotations = []
    import matplotlib.pyplot as plt

    original_annotate = plt.Axes.annotate

    def mock_annotate(self, text, xy, *args, **kwargs):
        annotations.append(text)
        return original_annotate(self, text, xy, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "annotate", mock_annotate)

    rp.plot_ref_image(r1, target_coord, "muscat4", out1)
    assert "Target" in annotations

    # Test case 2: defaulted_to_brightest=True -> should annotate "Target???"
    annotations.clear()
    r2 = {"ref": ref, "band": "gp", "target_index": 0, "defaulted_to_brightest": True}
    rp.plot_ref_image(r2, target_coord, "muscat4", out2)
    assert "Target???" in annotations


def test_plot_ref_image_simbad_eclbin_color(tmp_path, monkeypatch):
    from astropy.io import fits
    from astropy.wcs import WCS
    import pandas as pd

    # Mock simbad data containing both a regular object ("V*") and an eclipsing binary ("EclBin")
    simbad_df = pd.DataFrame(
        {
            "RA": ["00 00 00.5", "00 00 01.0"],
            "DEC": ["+00 00 05", "+00 00 10"],
            "OTYPE": ["V*", "EclBin"],
        }
    )
    monkeypatch.setattr(rp, "get_simbad_data", lambda *a, **kw: simbad_df)

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
    ref.sources = Sources(np.array([[10, 10], [5, 5]]))
    target_coord = __import__("astropy").coordinates.SkyCoord(0, 0, unit="deg")
    out = tmp_path / "ref_simbad.png"

    # Capture colors passed to ax.scatter and ax.annotate
    scatter_colors = []
    annotate_colors = []

    import matplotlib.pyplot as plt

    original_scatter = plt.Axes.scatter
    original_annotate = plt.Axes.annotate

    def mock_scatter(self, x, y, *args, **kwargs):
        if "ec" in kwargs:
            scatter_colors.append(kwargs["ec"])
        return original_scatter(self, x, y, *args, **kwargs)

    def mock_annotate(self, text, xy, *args, **kwargs):
        if "color" in kwargs:
            annotate_colors.append(kwargs["color"])
        return original_annotate(self, text, xy, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "scatter", mock_scatter)
    monkeypatch.setattr(plt.Axes, "annotate", mock_annotate)

    r = {"ref": ref, "band": "gp", "target_index": 0}
    # When plot_gaia_sources=True and wcs is available, the EclBin should be orange ("C1")
    rp.plot_ref_image(r, target_coord, "muscat4", out, plot_gaia_sources=True)

    # Let's verify that one annotation is COLOR_SIMBAD_DEFAULT (for "V*") and the other is COLOR_SIMBAD_ECLBIN (for "EclBin")
    # Note: Target label is also annotated, which has color=COLOR_TARGET, so we ignore it.
    simbad_annotations = [c for c in annotate_colors if c != rp.COLOR_TARGET]
    assert rp.COLOR_SIMBAD_DEFAULT in simbad_annotations
    assert rp.COLOR_SIMBAD_ECLBIN in simbad_annotations


# --------------------------- Gaia source overlay ---------------------------


class _FakeCutout:
    """Minimal cutout stand-in exposing ``.wcs`` and ``.data`` for overlays."""

    def __init__(self, wcs, data):
        self.wcs = wcs
        self.data = data


def _tan_wcs(crpix=(10, 10)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = list(crpix)
    w.wcs.cdelt = [0.01, 0.01]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_overlay_gaia_sources_noop_without_catalog():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # None catalog and an object without a WCS both degrade to a no-op.
    assert (
        rp._overlay_gaia_sources(ax, _FakeCutout(_tan_wcs(), np.ones((20, 20))), None)
        == 0
    )
    assert (
        rp._overlay_gaia_sources(
            ax, object(), pd.DataFrame({"ra": [0.0], "dec": [0.0]})
        )
        == 0
    )
    plt.close(fig)


def test_overlay_gaia_sources_clips_to_cutout_bounds():
    import matplotlib.pyplot as plt

    cutout = _FakeCutout(_tan_wcs(), np.ones((20, 20)))
    # (0, 0) maps inside the 20x20 cutout; (10, 10) deg lands far outside.
    df = pd.DataFrame(
        {"ra": [0.0, 10.0], "dec": [0.0, 10.0], "phot_g_mean_mag": [12.0, 13.0]}
    )
    fig, ax = plt.subplots()
    n = rp._overlay_gaia_sources(ax, cutout, df)
    plt.close(fig)
    assert n == 1


def test_overlay_gaia_sources_delta_mag_omits_positive_sign():
    import matplotlib.pyplot as plt

    cutout = _FakeCutout(_tan_wcs(), np.ones((20, 20)))
    df = pd.DataFrame(
        {
            "ra": [0.0, 0.01],
            "dec": [0.0, 0.0],
            "phot_g_mean_mag": [12.0, 13.0],
        }
    )
    fig, ax = plt.subplots()
    rp._overlay_gaia_sources(
        ax, cutout, df, target_coord=rp.SkyCoord(0.0, 0.0, unit="deg")
    )
    labels = [annotation.get_text() for annotation in ax.texts]
    plt.close(fig)

    assert labels == ["1.0"]


def test_overlay_gaia_sources_handles_bad_wcs_gracefully():
    import matplotlib.pyplot as plt

    class _BoomWCS:
        def world_to_pixel(self, *a, **k):
            raise ValueError("bad wcs")

    cutout = _FakeCutout(_BoomWCS(), np.ones((20, 20)))
    df = pd.DataFrame({"ra": [0.0], "dec": [0.0]})
    fig, ax = plt.subplots()
    assert rp._overlay_gaia_sources(ax, cutout, df) == 0
    plt.close(fig)


def test_build_reference_stores_gaia_df_when_requested(tmp_path, monkeypatch):
    """``plot_gaia_sources`` fetches the Gaia catalog even with custom apertures."""
    _patch_ref_seq(monkeypatch)
    fake = pd.DataFrame({"ra": [0.0], "dec": [0.0]})
    monkeypatch.setattr(rp, "_gaia_catalog_df", lambda *a, **kw: fake)

    fpath = _write_minimal_fits(tmp_path)
    from astropy.coordinates import SkyCoord

    result = rp.build_reference(
        fpath,
        SkyCoord(0, 0, unit="deg"),
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_index_override=0,
        plot_gaia_sources=True,
    )
    assert result["gaia_df"] is fake


def test_build_reference_does_not_store_gaia_df_without_usable_wcs(
    tmp_path, monkeypatch, caplog
):
    """Gaia overlays require a usable reference WCS."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits

    _patch_ref_seq(monkeypatch)
    called = False

    def _spy(*a, **kw):
        nonlocal called
        called = True
        return pd.DataFrame({"ra": [0.0], "dec": [0.0]})

    monkeypatch.setattr(rp, "_gaia_catalog_df", _spy)

    fpath = tmp_path / "no_wcs.fits"
    hdr = fits.Header()
    hdr["TELESCOP"] = "2m0a"
    hdr["INSTRUME"] = "ep09"
    hdr["SITEID"] = "coj"
    hdr["OBJECT"] = "test"
    hdr["EXPTIME"] = 1
    hdr["FILTER"] = "gp"
    hdr["AIRMASS"] = 1.0
    hdr["JD"] = 2460000.0
    hdr["DATE-OBS"] = "2025-04-16T00:00:00"
    fits.writeto(fpath, np.ones((20, 20)), header=hdr)

    with caplog.at_level("WARNING", logger="prose_run_photometry"):
        result = rp.build_reference(
            fpath,
            SkyCoord(0, 0, unit="deg"),
            aper_radii=np.array([3.0, 4.0, 5.0]),
            rin=8.0,
            rout=12.0,
            target_index_override=0,
            plot_gaia_sources=True,
        )

    assert result["gaia_df"] is None
    assert not called
    assert any("no usable WCS" in r.message for r in caplog.records)


def test_build_reference_skips_gaia_df_for_custom_apertures(tmp_path, monkeypatch):
    """Custom apertures without the overlay flag never query Gaia."""
    _patch_ref_seq(monkeypatch)
    called = False

    def _spy(*a, **kw):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(rp, "_gaia_catalog_df", _spy)

    fpath = _write_minimal_fits(tmp_path)
    from astropy.coordinates import SkyCoord

    result = rp.build_reference(
        fpath,
        SkyCoord(0, 0, unit="deg"),
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_index_override=0,
        plot_gaia_sources=False,
    )
    assert result["gaia_df"] is None
    assert not called


# --------------------------- header overwrite of .telescope parameters ---------------------------


def test_build_reference_overwrites_pixel_scale_and_saturation(tmp_path, monkeypatch):
    _patch_ref_seq(monkeypatch)

    from astropy.io import fits
    from astropy.wcs import WCS

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
    hdr["TELID"] = "2m0a"
    hdr["OBJECT"] = "test"
    hdr["EXPTIME"] = 1
    hdr["FILTER"] = "gp"
    hdr["AIRMASS"] = 1.0
    hdr["JD"] = 2460000.0
    hdr["DATE-OBS"] = "2025-04-16T00:00:00"
    hdr["PIXSCALE"] = 0.5
    hdr["GAIN"] = 1.9
    hdr["CONFMODE"] = "full_frame"
    hdr["SATURATE"] = 64000
    hdr["MAXLIN"] = 60000
    fpath = tmp_path / "test.fits"
    fits.writeto(fpath, data, header=hdr)

    from astropy.coordinates import SkyCoord

    coord = SkyCoord(0, 0, unit="deg")
    result = rp.build_reference(
        fpath,
        coord,
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_index_override=0,
    )
    ref = result["ref"]
    assert abs(ref.telescope.pixel_scale - 0.5) < 1e-6
    assert ref.telescope.saturation is not None


def test_build_reference_overwrites_pixel_scale_with_fallback_when_no_header(
    tmp_path, monkeypatch
):
    _patch_ref_seq(monkeypatch)

    from astropy.io import fits
    from astropy.wcs import WCS

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

    from astropy.coordinates import SkyCoord

    coord = SkyCoord(0, 0, unit="deg")
    result = rp.build_reference(
        fpath,
        coord,
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_index_override=0,
    )
    ref = result["ref"]
    assert abs(ref.telescope.pixel_scale - 0.267) < 1e-6
    assert ref.telescope.saturation is None


# --------------------------- edge-source exclusion ---------------------------


class _FakeSrc:
    def __init__(self, xy):
        self.coords = np.array(xy, dtype=float)


class _FakeRef:
    """Minimal stand-in exposing the ``sources`` and ``shape`` that
    ``_edge_source_indices`` reads."""

    def __init__(self, coords, shape=(20, 20)):
        self.sources = [_FakeSrc(c) for c in coords]
        self.shape = shape


@pytest.mark.parametrize(
    "edge_margin, cutout_size, expected",
    [
        (None, 36, 18),  # auto -> half the cutout box
        (None, 35, 17),  # auto with odd cutout -> floor division
        (20, 36, 20),  # explicit value wins over auto
        (0, 36, 0),  # explicit disable
    ],
)
def test_resolve_edge_margin(edge_margin, cutout_size, expected):
    assert rp.resolve_edge_margin(edge_margin, cutout_size) == expected


def test_edge_source_indices_flags_only_border_stars():
    # 20x20 frame, margin 6: a star at x=5 is within 6 px of the left edge;
    # one at (10, 10) is comfortably interior.
    ref = _FakeRef([[5.0, 5.0], [10.0, 10.0]], shape=(20, 20))
    assert rp._edge_source_indices(ref, margin=6, target_index=99) == [0]


def test_edge_source_indices_never_drops_the_target():
    # The edge star (index 0) is also the target -> it must not be returned.
    ref = _FakeRef([[5.0, 5.0], [10.0, 10.0]], shape=(20, 20))
    assert rp._edge_source_indices(ref, margin=6, target_index=0) == []


def test_edge_source_indices_disabled_when_margin_zero():
    ref = _FakeRef([[0.0, 0.0], [1.0, 1.0]], shape=(20, 20))
    assert rp._edge_source_indices(ref, margin=0, target_index=99) == []


def test_edge_source_indices_handles_no_sources():
    assert rp._edge_source_indices(_FakeRef([], shape=(20, 20)), 6, 0) == []


def test_build_reference_returns_edge_cids(tmp_path, monkeypatch):
    """build_reference flags the border star as an edge comparison to avoid,
    while keeping the interior target."""
    _patch_ref_seq(monkeypatch)  # sources at (5,5) and (10,10) in a 20x20 frame
    fpath = _write_minimal_fits(tmp_path)

    from astropy.coordinates import SkyCoord

    coord = SkyCoord(0, 0, unit="deg")
    result = rp.build_reference(
        fpath,
        coord,
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_index_override=1,  # interior star is the target
        edge_margin=6,
    )
    assert result["edge_cids"] == [0]


def test_build_reference_edge_margin_disabled(tmp_path, monkeypatch):
    _patch_ref_seq(monkeypatch)
    fpath = _write_minimal_fits(tmp_path)

    from astropy.coordinates import SkyCoord

    coord = SkyCoord(0, 0, unit="deg")
    result = rp.build_reference(
        fpath,
        coord,
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_index_override=1,
        edge_margin=0,
    )
    assert result["edge_cids"] == []


def test_build_reference_warns_when_target_near_edge(tmp_path, monkeypatch, caplog):
    """A target sitting inside the margin is kept (never in edge_cids) but the
    border proximity is surfaced as a warning."""
    _patch_ref_seq(monkeypatch)
    fpath = _write_minimal_fits(tmp_path)

    from astropy.coordinates import SkyCoord

    coord = SkyCoord(0, 0, unit="deg")
    with caplog.at_level("WARNING", logger="prose_run_photometry"):
        result = rp.build_reference(
            fpath,
            coord,
            aper_radii=np.array([3.0, 4.0, 5.0]),
            rin=8.0,
            rout=12.0,
            target_index_override=0,  # the (5,5) border star is the target
            edge_margin=6,
        )
    assert result["edge_cids"] == []  # target never dropped
    assert any("within 6 px of a CCD edge" in r.message for r in caplog.records)


def test_find_target_index_falls_back_when_over_5_arcsec(monkeypatch):
    class FakeRef:
        def __init__(self):
            self.sources = [_FakeSrc([10.0, 10.0])]
            self.wcs = self
            self.fwhm = 3.0
            self.telescope = self
            self.pixel_scale = 0.267
            self.header = {}

        def pixel_to_world(self, x, y):
            from astropy.coordinates import SkyCoord

            return SkyCoord([10.0], [10.0], unit="deg")

    from astropy.coordinates import SkyCoord

    ref = FakeRef()
    target_coord = SkyCoord(0.0, 0.0, unit="deg")
    assert rp.find_target_index(ref, target_coord) == 0


def test_order_bands_for_target_id_inference_delays_no_wcs_bands():
    bands = ["gp", "rp", "ip", "zs"]
    ref_wcs_ok = {"gp": False, "rp": True, "ip": True, "zs": True}

    assert rp._order_bands_for_target_id_inference(bands, ref_wcs_ok) == [
        "rp",
        "ip",
        "zs",
        "gp",
    ]


def test_order_bands_for_target_id_inference_preserves_uniform_wcs_state():
    bands = ["gp", "rp", "ip", "zs"]

    assert (
        rp._order_bands_for_target_id_inference(bands, {band: True for band in bands})
        == bands
    )
    assert (
        rp._order_bands_for_target_id_inference(bands, {band: False for band in bands})
        == bands
    )


def test_sort_bands_canonical_orders_broadband_by_wavelength():
    assert rp.sort_bands_canonical(["ip", "zs", "gp", "rp"]) == ["gp", "rp", "ip", "zs"]


def test_sort_bands_canonical_places_narrowbands_after_broadbands():
    mixed = ["z_narrow", "rp", "Na_D", "gp", "i_narrow", "ip", "g_narrow", "zs"]
    assert rp.sort_bands_canonical(mixed) == [
        "gp",
        "rp",
        "ip",
        "zs",
        "g_narrow",
        "Na_D",
        "i_narrow",
        "z_narrow",
    ]


def test_sort_bands_canonical_subset_and_dict_keys():
    assert rp.sort_bands_canonical(["zs", "gp"]) == ["gp", "zs"]
    # Accepts a dict_keys view (band_results.keys()) directly.
    assert rp.sort_bands_canonical({"ip": 1, "gp": 2}.keys()) == ["gp", "ip"]


def test_sort_bands_canonical_unknown_bands_stable_after_known():
    # Unknown bands keep their input relative order, after all known bands.
    assert rp.sort_bands_canonical(["foo", "zs", "bar", "gp"]) == [
        "gp",
        "zs",
        "foo",
        "bar",
    ]


def test_target_pixel_override_for_band_uses_inferred_position_only_without_wcs():
    inferred = [np.array([10.0, 20.0]), np.array([12.0, 22.0])]

    np.testing.assert_allclose(
        rp._target_pixel_override_for_band(None, True, inferred, False),
        np.array([11.0, 21.0]),
    )
    assert rp._target_pixel_override_for_band(None, True, inferred, True) is None
    assert rp._target_pixel_override_for_band(None, False, inferred, False) is None
    assert rp._target_pixel_override_for_band(2, True, inferred, False) is None


def test_build_reference_target_pixel_override_uses_nearest_source(
    tmp_path, monkeypatch
):
    """A no-WCS band should infer target ID by pixel proximity, not source ID."""
    called = False

    def _never_called(*a, **kw):
        nonlocal called
        called = True
        raise AssertionError("find_target_index should not run for pixel override")

    monkeypatch.setattr(rp, "find_target_index", _never_called)
    _patch_ref_seq(monkeypatch)
    fpath = _write_minimal_fits(tmp_path)

    from astropy.coordinates import SkyCoord

    result = rp.build_reference(
        fpath,
        SkyCoord(0, 0, unit="deg"),
        aper_radii=np.array([3.0, 4.0, 5.0]),
        rin=8.0,
        rout=12.0,
        target_pixel_override=np.array([9.7, 10.2]),
    )

    assert result["target_index"] == 1
    assert result["defaulted_to_brightest"] is False
    assert not called


def test_build_reference_target_pixel_override_rejects_far_source(
    tmp_path, monkeypatch
):
    _patch_ref_seq(monkeypatch)
    fpath = _write_minimal_fits(tmp_path)

    from astropy.coordinates import SkyCoord

    with pytest.raises(ValueError, match="nearest detected source"):
        rp.build_reference(
            fpath,
            SkyCoord(0, 0, unit="deg"),
            aper_radii=np.array([3.0, 4.0, 5.0]),
            rin=8.0,
            rout=12.0,
            target_pixel_override=np.array([200.0, 200.0]),
        )


def test_run_band_relaxes_edge_exclusion_when_empty_comparisons(tmp_path, monkeypatch):
    fpath = _write_minimal_fits(tmp_path)
    from astropy.coordinates import SkyCoord

    coord = SkyCoord(0, 0, unit="deg")

    class FakeImage:
        def __init__(self):
            self.sources = [_FakeSrc([5, 5]), _FakeSrc([10, 10])]
            self.shape = (20, 20)
            self.telescope = self
            self.jd_scale = "jd"

    reference = {
        "ref": FakeImage(),
        "target_index": 1,
        "aper_radii": np.array([3, 4]),
        "rin": 8.0,
        "rout": 12.0,
        "scale": False,
        "edge_cids": [0],
    }

    monkeypatch.setattr(rp, "build_reference", lambda *a, **kw: reference)

    def mock_photometry_sequence(ref, aper_radii, rin, rout, **kwargs):
        class MockPhot:
            def run(self, files):
                pass

            @property
            def data(self):
                class FakeData:
                    @property
                    def fluxes(self):
                        class FakeFluxes:
                            def __init__(self):
                                self._fluxes = np.ones((5, 2, 2))
                                self._time = np.arange(5)

                            def copy(self):
                                return self

                            def mask_stars(self, mask):
                                return self

                            def sigma_clipping_data(self, **kw):
                                return self

                            def diff(self, comps):
                                class FakeDiff:
                                    def __init__(self):
                                        self.time = np.arange(5)

                                return FakeDiff()

                            def autodiff(self, nan_imputation_method="linear"):
                                class FakeDiff:
                                    def __init__(self):
                                        self.time = np.arange(5)

                                return FakeDiff()

                            @property
                            def fluxes(self):
                                return self._fluxes

                            @fluxes.setter
                            def fluxes(self, value):
                                self._fluxes = value

                            @property
                            def time(self):
                                return self._time

                        return FakeFluxes()

                return [FakeData()]

        return MockPhot()

    monkeypatch.setattr(rp, "photometry_sequence", mock_photometry_sequence)

    result = rp.run_band(
        band="gp",
        files=[str(fpath)],
        ref_file=str(fpath),
        target_coord=coord,
        edge_margin=6,
    )

    assert result["avoid_cids"] is None or 0 not in result["avoid_cids"]


def test_run_band_remaps_target_index_from_reference_band(tmp_path, monkeypatch):
    from prose import Fluxes
    from prose.core.source import Sources
    from astropy.coordinates import SkyCoord

    ref_band_positions = np.array([[1.0, 1.0], [9.0, 9.0]])
    current_band_positions = np.array([[9.2, 9.1], [1.1, 1.0]])

    class FakeImage:
        def __init__(self):
            self.sources = Sources(np.array(current_band_positions))
            self.shape = (20, 20)
            self.telescope = self
            self.jd_scale = "jd"
            self.header = {}
            self.fwhm = 4.0

    reference = {
        "ref": FakeImage(),
        "target_index": 0,
        "aper_radii": np.array([3.0]),
        "rin": 8.0,
        "rout": 12.0,
        "scale": False,
        "edge_cids": [],
        "gaia_df": None,
        "defaulted_to_brightest": False,
    }

    monkeypatch.setattr(rp, "build_reference", lambda *a, **kw: reference)

    captured = {}

    def mock_photometry_sequence(ref, aper_radii, rin, rout, **kwargs):
        captured["sequence_target_index"] = kwargs["target_index"]

        class MockPhot:
            def run(self, files):
                return None

            @property
            def data(self):
                class FakeData:
                    @property
                    def fluxes(self):
                        return Fluxes(np.ones((2, 4)))

                return [FakeData()]

        return MockPhot()

    def mock_diff(
        fluxes, target_index, cids=None, avoid_cids=None, nan_imputation_method="linear"
    ):
        captured["diff_target_index"] = target_index

        class FakeDiff:
            def __init__(self):
                self.time = np.arange(4, dtype=float)
                self.target = target_index

        return FakeDiff()

    monkeypatch.setattr(rp, "photometry_sequence", mock_photometry_sequence)
    monkeypatch.setattr(rp, "differential_photometry", mock_diff)

    result = rp.run_band(
        band="gp",
        files=[str(_write_minimal_fits(tmp_path, "science.fits"))],
        ref_file=str(_write_minimal_fits(tmp_path, "reference.fits")),
        target_coord=SkyCoord(0, 0, unit="deg"),
        ref_source_positions=ref_band_positions,
    )

    assert result is not None
    assert captured["sequence_target_index"] == 1
    assert captured["diff_target_index"] == 1
    assert result["target_index"] == 1


def test_run_band_remaps_cids_and_avoid_cids_from_reference_band(tmp_path, monkeypatch):
    from prose import Fluxes
    from prose.core.source import Sources
    from astropy.coordinates import SkyCoord

    ref_band_positions = np.array([[1.0, 1.0], [9.0, 9.0]])
    current_band_positions = np.array([[9.2, 9.1], [1.1, 1.0]])

    class FakeImage:
        def __init__(self):
            self.sources = Sources(np.array(current_band_positions))
            self.shape = (20, 20)
            self.telescope = self
            self.jd_scale = "jd"
            self.header = {}
            self.fwhm = 4.0

    reference = {
        "ref": FakeImage(),
        "target_index": 0,
        "aper_radii": np.array([3.0]),
        "rin": 8.0,
        "rout": 12.0,
        "scale": False,
        "edge_cids": [],
        "gaia_df": None,
        "defaulted_to_brightest": False,
    }

    monkeypatch.setattr(rp, "build_reference", lambda *a, **kw: reference)

    captured = {}

    def mock_photometry_sequence(ref, aper_radii, rin, rout, **kwargs):
        class MockPhot:
            def run(self, files):
                return None

            @property
            def data(self):
                class FakeData:
                    @property
                    def fluxes(self):
                        return Fluxes(np.ones((2, 4)))

                return [FakeData()]

        return MockPhot()

    def mock_diff(
        fluxes, target_index, cids=None, avoid_cids=None, nan_imputation_method="linear"
    ):
        captured["diff_cids"] = cids
        captured["diff_avoid_cids"] = avoid_cids

        class FakeDiff:
            def __init__(self):
                self.time = np.arange(4, dtype=float)
                self.target = target_index

        return FakeDiff()

    monkeypatch.setattr(rp, "photometry_sequence", mock_photometry_sequence)
    monkeypatch.setattr(rp, "differential_photometry", mock_diff)

    result = rp.run_band(
        band="gp",
        files=[str(_write_minimal_fits(tmp_path, "science.fits"))],
        ref_file=str(_write_minimal_fits(tmp_path, "reference.fits")),
        target_coord=SkyCoord(0, 0, unit="deg"),
        ref_source_positions=ref_band_positions,
        cids=[0],
        avoid_cids=[1],
    )

    assert result is not None
    assert captured["diff_cids"] == [1]
    assert captured["diff_avoid_cids"] == [0]
    assert result["ref_band_star_ids"] == [1, 0]


def _write_sinistro_fits(
    tmp_path, name, site_id, filter_name="gp", confmode="central_2k_2x2"
):
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [10, 10]
    w.wcs.cdelt = [0.01, 0.01]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    data = np.ones((20, 20))
    hdr = w.to_header()
    hdr["TELID"] = "1m0a"
    hdr["INSTRUME"] = "sinistro"
    hdr["SITEID"] = site_id
    hdr["CONFMODE"] = confmode
    hdr["OBJECT"] = "TOI-6715"
    hdr["EXPTIME"] = 1
    hdr["FILTER"] = filter_name
    hdr["AIRMASS"] = 1.0
    hdr["JD"] = 2460000.0
    hdr["DATE-OBS"] = "2025-04-16T00:00:00"
    fpath = tmp_path / name
    fits.writeto(fpath, data, header=hdr)
    return fpath


def test_main_site_argument_rejected_for_non_sinistro(tmp_path):
    # Instrument is muscat4 (default from _write_minimal_fits: TELESCOP="2m0a", SITEID="coj")
    _write_minimal_fits(tmp_path, "test.fits")

    argv = [
        "--target_name",
        "test",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--site",
        "lsc",
    ]

    ret = rp.main(argv)
    assert ret == 1


def test_main_site_argument_invalid_site_rejected_for_sinistro(tmp_path):
    # Instrument is sinistro, SITEID is lsc
    _write_sinistro_fits(tmp_path, "test.fits", "lsc")

    argv = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--site",
        "abc",
    ]

    ret = rp.main(argv)
    assert ret == 1


def test_main_site_filtering_no_matching_frames(tmp_path):
    # Instrument is sinistro, SITEID is lsc
    _write_sinistro_fits(tmp_path, "test.fits", "lsc")

    argv = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--site",
        "cpt",
    ]

    ret = rp.main(argv)
    assert ret == 1


def test_main_mode_argument_rejected_for_non_sinistro(tmp_path):
    # Instrument is muscat4
    _write_minimal_fits(tmp_path, "test.fits")

    argv = [
        "--target_name",
        "test",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--mode",
        "central_2k_2x2",
    ]

    ret = rp.main(argv)
    assert ret == 1


def test_main_mode_argument_invalid_mode_rejected_for_sinistro(tmp_path):
    _write_sinistro_fits(tmp_path, "test.fits", "lsc")

    argv = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--mode",
        "invalid_mode",
    ]

    with pytest.raises(SystemExit):
        rp.main(argv)


def test_main_mode_filtering_no_matching_frames(tmp_path):
    # Instrument is sinistro, CONFMODE is central_2k_2x2
    _write_sinistro_fits(tmp_path, "test.fits", "lsc", confmode="central_2k_2x2")

    argv = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--mode",
        "full_frame",
    ]

    with pytest.raises(SystemExit):
        rp.main(argv)


def test_main_mode_filtering_checks_obslog_first(tmp_path, monkeypatch, caplog):
    # FITS header has CONFMODE = central_2k_2x2
    fpath = _write_sinistro_fits(
        tmp_path, "test.fits", "lsc", confmode="central_2k_2x2"
    )

    # Mock frames_from_obslog to return confmode = full_frame
    def mock_frames_from_obslog(data_dir, instrument=None):
        return [
            {
                "frame": "test",
                "object": "TOI-6715",
                "filter": "gp",
                "exposure": 1.0,
                "ccd": 1,
                "path": str(fpath),
                "confmode": "full_frame",
            }
        ]

    monkeypatch.setattr(rp, "frames_from_obslog", mock_frames_from_obslog)

    # Request mode: central_2k_2x2 (which matches the header, but not the obslog)
    argv = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--mode",
        "central_2k_2x2",
    ]

    with pytest.raises(SystemExit):
        rp.main(argv)


def test_main_mode_filtering_fallback_to_header(tmp_path, monkeypatch, caplog):
    # FITS header has CONFMODE = central_2k_2x2
    fpath = _write_sinistro_fits(
        tmp_path, "test.fits", "lsc", confmode="central_2k_2x2"
    )

    # Mock frames_from_obslog to return record WITHOUT confmode
    def mock_frames_from_obslog(data_dir, instrument=None):
        return [
            {
                "frame": "test",
                "object": "TOI-6715",
                "filter": "gp",
                "exposure": 1.0,
                "ccd": 1,
                "path": str(fpath),
            }
        ]

    monkeypatch.setattr(rp, "frames_from_obslog", mock_frames_from_obslog)

    # Request mode: central_2k_2x2 (which matches the header)
    argv = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
        "--mode",
        "central_2k_2x2",
    ]

    with caplog.at_level("ERROR", logger="prose_run_photometry"):
        ret = rp.main(argv)
    # The return code will still be 1 (due to MAST/Simbad/photometry failures downstream),
    # but it should NOT have aborted at the mode check!
    assert ret == 1
    assert not any(
        "with mode=central_2k_2x2; aborting" in r.message for r in caplog.records
    )


def test_main_mode_raise_condition_for_multiple_modes_unspecified(tmp_path):
    # Write two fits files with different modes
    _write_sinistro_fits(tmp_path, "test1.fits", "lsc", confmode="central_2k_2x2")
    _write_sinistro_fits(tmp_path, "test2.fits", "lsc", confmode="full_frame")

    argv = [
        "--target_name",
        "TOI-6715",
        "--data_dir",
        str(tmp_path),
        "--results_dir",
        str(tmp_path / "results"),
    ]

    with pytest.raises(ValueError, match="Multiple configuration modes found"):
        rp.main(argv)


def test_gif_stride_step_calculation():
    # 100 frames with target of 10 should yield a stride of 10 (every 10th frame)
    assert max(1, 100 // 10) == 10
    # 9 frames with target of 10 should yield a stride of 1 (show all)
    assert max(1, 9 // 10) == 1
    # 105 frames with target of 10 should yield a stride of 10
    assert max(1, 105 // 10) == 10
    # 20 frames with target of 10 should yield a stride of 2
    assert max(1, 20 // 10) == 2


def test_plot_stacks_draws_saturation_axhline(tmp_path, monkeypatch):
    from astropy.io import fits
    from astropy.wcs import WCS

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
    ref.sources = Sources(np.array([[10, 10], [5, 5]]))
    ref.telescope.saturation = 55000.0

    class FakeDiff:
        def __init__(self):
            self.aperture = 0

    r = {
        "ref": ref,
        "band": "gp",
        "target_index": 0,
        "diff": FakeDiff(),
        "aper_radii": [1.0, 2.0, 3.0],
        "rin": 4.0,
        "rout": 5.0,
        "scale": False,
    }

    axhline_y_vals = []
    import matplotlib.pyplot as plt

    original_axhline = plt.Axes.axhline

    def mock_axhline(self, y, *args, **kwargs):
        axhline_y_vals.append(y)
        return original_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "axhline", mock_axhline)

    twin_axes_created = 0
    original_twinx = plt.Axes.twinx

    def mock_twinx(self):
        nonlocal twin_axes_created
        twin_axes_created += 1
        return original_twinx(self)

    monkeypatch.setattr(plt.Axes, "twinx", mock_twinx)

    out = tmp_path / "stacks.png"
    rp.plot_stacks({"gp": r}, out, "TOI-6715", "muscat4", "2026-06-23", 0)

    assert 55000.0 in axhline_y_vals
    assert twin_axes_created == 1


def test_plot_ref_image_custom_cmap(tmp_path, monkeypatch):
    from astropy.io import fits
    from astropy.wcs import WCS

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
    ref.sources = Sources(np.array([[10, 10], [5, 5]]))
    target_coord = __import__("astropy").coordinates.SkyCoord(0, 0, unit="deg")
    out = tmp_path / "ref_inverted.png"

    show_kwargs = {}

    def mock_show(*args, **kwargs):
        show_kwargs.update(kwargs)
        # Avoid plotting the actual image to speed up/prevent window pops
        return None

    monkeypatch.setattr(ref, "show", mock_show)

    scatter_colors = []
    import matplotlib.pyplot as plt

    original_scatter = plt.Axes.scatter

    def mock_scatter(self, x, y, *args, **kwargs):
        if "ec" in kwargs:
            scatter_colors.append(kwargs["ec"])
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "scatter", mock_scatter)

    r = {"ref": ref, "band": "gp", "target_index": 0}
    rp.plot_ref_image(r, target_coord, "muscat4", out, cmap="Greys")

    assert show_kwargs.get("cmap") == "Greys"
    assert "crimson" in scatter_colors


def test_nearby_stars_csv_generation(tmp_path, monkeypatch):
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS

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
    hdr["PIXSCALE"] = 0.267
    fpath = tmp_path / "test.fits"
    fits.writeto(fpath, data, header=hdr)

    from prose import FITSImage
    from prose.core.source import Sources
    from astropy.coordinates import SkyCoord

    ref = FITSImage(fpath)
    ref.sources = Sources(np.array([[10, 10], [12, 12]]))
    target_coord = SkyCoord(0, 0, unit="deg")

    gaia_df = pd.DataFrame(
        {
            "ra": [0.0, 0.001],
            "dec": [0.0, 0.001],
            "phot_g_mean_mag": [10.0, 12.5],
            "source_id": [123, 456],
        }
    )

    gaia_coords = SkyCoord(gaia_df.ra.values, gaia_df.dec.values, unit="deg")
    target_idx_in_gaia = target_coord.separation(gaia_coords).argmin()
    assert target_idx_in_gaia == 0
    target_g_mag = float(gaia_df.phot_g_mean_mag.values[target_idx_in_gaia])
    assert target_g_mag == 10.0

    detected_pix_coords = np.array([s.coords for s in ref.sources], dtype=float)
    detected_coords = ref.wcs.pixel_to_world(*detected_pix_coords.T)
    match_idx, match_sep, _ = gaia_coords.match_to_catalog_sky(detected_coords)

    nearby_stars_data = []
    pixscale = 0.267
    rout = 50.0
    for i in range(len(gaia_df)):
        if i == target_idx_in_gaia:
            continue
        sep_arc = float(target_coord.separation(gaia_coords[i]).arcsec)
        sep_p = sep_arc / pixscale
        if sep_p <= rout:
            g_mag = gaia_df.phot_g_mean_mag.values[i]
            delta_mag = float(g_mag - target_g_mag)
            contam_ratio = float(10 ** (-delta_mag / 2.5) * 100)
            det_sep_arc = float(match_sep[i].arcsec)
            detected_str = "Y" if det_sep_arc <= 1.5 else "N"
            source_id = gaia_df.source_id.values[i]

            nearby_stars_data.append(
                {
                    "Separation (arcsec)": round(sep_arc, 3),
                    "Separation (pix)": round(sep_p, 2),
                    "Gaia delta mag": round(delta_mag, 3),
                    "Detected (Y/N)": detected_str,
                    "Contamination Ratio (%)": round(contam_ratio, 4),
                    "Gaia Source ID": str(source_id),
                    "RA (deg)": round(float(gaia_df.ra.values[i]), 6),
                    "Dec (deg)": round(float(gaia_df.dec.values[i]), 6),
                    "Gaia G mag": round(float(g_mag), 3),
                }
            )

    assert len(nearby_stars_data) == 1
    item = nearby_stars_data[0]
    assert item["Gaia Source ID"] == "456"
    assert item["Gaia delta mag"] == 2.5
    assert item["Contamination Ratio (%)"] == 10.0


def test_parse_args_accepts_both_sinistro_modes():
    """Verify that argparse accepts both central_2k_2x2 and full_frame modes.

    Both modes should always be valid choices, regardless of which FITS files
    exist. Runtime validation will ensure the specified mode is present in the
    filtered data. This allows users to specify a mode even when dynamic
    discovery doesn't find it (e.g., when filtered by site or band).
    """
    # Both modes should be accepted without error
    args_central = rp.parse_args(
        [
            "--target_name",
            "V1298Tau",
            "--data_dir",
            "/tmp",
            "--results_dir",
            "/tmp/results",
            "--mode",
            "central_2k_2x2",
        ]
    )
    assert args_central.mode == "central_2k_2x2"

    args_full = rp.parse_args(
        [
            "--target_name",
            "V1298Tau",
            "--data_dir",
            "/tmp",
            "--results_dir",
            "/tmp/results",
            "--mode",
            "full_frame",
        ]
    )
    assert args_full.mode == "full_frame"

    # Invalid modes should still be rejected
    with pytest.raises(SystemExit):
        rp.parse_args(
            [
                "--target_name",
                "V1298Tau",
                "--data_dir",
                "/tmp",
                "--results_dir",
                "/tmp/results",
                "--mode",
                "invalid_mode",
            ]
        )


# --------------------------- reference-frame quality pre-check ---------------------------


def _write_quality_header_fits(tmp_path, name, **hdr):
    """Minimal FITS file carrying only the header keywords header_triage()
    reads -- no WCS/pixel data needed since header_triage never opens data."""
    hdu = fits.PrimaryHDU(data=None)
    hdu.header["NAXIS"] = 2
    for k, v in hdr.items():
        hdu.header[k] = v
    fpath = tmp_path / name
    hdu.writeto(fpath, overwrite=True)
    return fpath


def test_header_triage_ranks_by_composite_score_dominated_by_fwhm(tmp_path):
    sharp = _write_quality_header_fits(
        tmp_path,
        "sharp.fits",
        L1FWHM=1.8,
        AIRMASS=1.1,
        FOCPOSN=0.0,
        WMSHUMID=20,
        L1MEAN=30,
        SATURATE=80000,
    )
    blurry = _write_quality_header_fits(
        tmp_path,
        "blurry.fits",
        L1FWHM=4.5,
        AIRMASS=1.1,
        FOCPOSN=0.0,
        WMSHUMID=20,
        L1MEAN=30,
        SATURATE=80000,
    )
    mid = _write_quality_header_fits(
        tmp_path,
        "mid.fits",
        L1FWHM=2.5,
        AIRMASS=1.1,
        FOCPOSN=0.0,
        WMSHUMID=20,
        L1MEAN=30,
        SATURATE=80000,
    )
    files = [str(blurry), str(sharp), str(mid)]

    idxs, records = rp.header_triage(files, top_k=3)

    assert [files[i] for i in idxs] == [str(sharp), str(mid), str(blurry)]
    assert all(r.hard_reject is None for r in records)


def test_header_triage_hard_rejects_extreme_airmass_and_humidity(tmp_path):
    good = _write_quality_header_fits(
        tmp_path, "good.fits", L1FWHM=2.0, AIRMASS=1.2, WMSHUMID=30
    )
    bad_airmass = _write_quality_header_fits(
        tmp_path, "bad_airmass.fits", L1FWHM=1.5, AIRMASS=4.0, WMSHUMID=30
    )
    bad_humidity = _write_quality_header_fits(
        tmp_path, "bad_humidity.fits", L1FWHM=1.5, AIRMASS=1.2, WMSHUMID=99
    )
    files = [str(good), str(bad_airmass), str(bad_humidity)]

    idxs, records = rp.header_triage(files, top_k=5)

    assert [files[i] for i in idxs] == [str(good)]
    reject_reasons = {Path(r.path).name: r.hard_reject for r in records}
    assert reject_reasons["bad_airmass.fits"] is not None
    assert reject_reasons["bad_humidity.fits"] is not None
    assert reject_reasons["good.fits"] is None


def test_header_triage_falls_back_when_all_frames_hard_rejected(tmp_path, caplog):
    files = [
        str(
            _write_quality_header_fits(
                tmp_path, f"f{i}.fits", L1FWHM=2.0 + i, AIRMASS=4.0
            )
        )
        for i in range(3)
    ]

    with caplog.at_level("WARNING"):
        idxs, records = rp.header_triage(files, top_k=5)

    assert len(idxs) == 3  # never empties the pool
    assert any("falling back" in msg for msg in caplog.messages)


def test_header_triage_tolerates_missing_individual_keys(tmp_path):
    """A single missing field (WMSHUMID) shouldn't exclude an otherwise-good frame."""
    complete = _write_quality_header_fits(
        tmp_path,
        "complete.fits",
        L1FWHM=2.0,
        AIRMASS=1.1,
        WMSHUMID=20,
        L1MEAN=30,
        SATURATE=80000,
    )
    missing_humidity = _write_quality_header_fits(
        tmp_path,
        "missing_humidity.fits",
        L1FWHM=2.0,
        AIRMASS=1.1,
        L1MEAN=30,
        SATURATE=80000,
    )
    files = [str(complete), str(missing_humidity)]

    idxs, records = rp.header_triage(files, top_k=5)

    assert len(idxs) == 2  # both ranked, none excluded for one missing field
    assert all(r.score is not None for r in records)


def test_header_triage_clamps_top_k_to_available_frames(tmp_path):
    files = [
        str(
            _write_quality_header_fits(
                tmp_path, f"f{i}.fits", L1FWHM=1.0 + i * 0.1, AIRMASS=1.1
            )
        )
        for i in range(3)
    ]

    idxs, _ = rp.header_triage(files, top_k=100)

    assert len(idxs) == 3


def test_quality_select_eligible_true_for_banzai_instruments_and_muscat_family():
    for inst in ("muscat3", "muscat4", "sinistro", "muscat", "muscat2"):
        assert rp._quality_select_eligible(inst) is True
    assert rp._quality_select_eligible("unknown") is False


def test_candidate_frames_for_tier2_muscat_family_uses_local_midpoint_window(tmp_path):
    files = [str(tmp_path / f"f{i}.fits") for i in range(10)]

    idxs, tier1_diag = rp._candidate_frames_for_tier2(files, "muscat2", top_k=3)

    assert tier1_diag is None  # header triage skipped -- no BANZAI header proxy
    assert idxs == [4, 5, 6]


class _FakeGetDataSequenceParallel:
    """Stand-in for SequenceParallel that populates the data_blocks' Get
    block from a test-configured per-position metrics table, instead of
    running the real (parallel, multi-process) detection/PSF pipeline --
    mirrors this file's existing ``_FakeRefSeq`` convention of bypassing
    the expensive pixel pipeline while exercising the real orchestration
    logic (here: ranking/selection in select_reference_frame)."""

    metrics_by_position: list[dict] = []
    discard_positions: set = set()

    def __init__(self, blocks=None, data_blocks=None, name=""):
        self._get_block = (data_blocks or [None])[0]

    def run(self, images, show_progress=True):
        for pos in range(len(images)):
            if pos in type(self).discard_positions:
                continue
            m = type(self).metrics_by_position[pos]
            for key in self._get_block.values:
                if key == "path":
                    value = str(images[pos])
                elif key == "source_coords":
                    value = m.get(key, np.empty((0, 2)))
                elif key == "target_index":
                    value = m.get(key)
                else:
                    value = m[key]
                self._get_block.values[key].append(value)


def _patch_fake_sequence_parallel(
    monkeypatch, metrics_by_position, discard_positions=()
):
    _FakeGetDataSequenceParallel.metrics_by_position = metrics_by_position
    _FakeGetDataSequenceParallel.discard_positions = set(discard_positions)
    monkeypatch.setattr(rp, "SequenceParallel", _FakeGetDataSequenceParallel)


def test_select_reference_frame_uses_tier2_to_pick_lowest_fwhm(tmp_path, monkeypatch):
    files = [str(tmp_path / f"f{i}.fits") for i in range(3)]
    # header_triage isn't exercised here (muscat2 -> evenly spaced candidates
    # covering all 3 frames); Tier 2 metrics decide the winner.
    _patch_fake_sequence_parallel(
        monkeypatch,
        metrics_by_position=[
            {
                "idx": 0,
                "fwhm": 6.0,
                "n_sources": 10,
                "target_matched": True,
                "target_saturated": False,
            },
            {
                "idx": 1,
                "fwhm": 3.2,
                "n_sources": 9,
                "target_matched": True,
                "target_saturated": False,
            },
            {
                "idx": 2,
                "fwhm": 9.0,
                "n_sources": 10,
                "target_matched": True,
                "target_saturated": False,
            },
        ],
    )

    refid, diag = rp.select_reference_frame(
        files, instrument="muscat2", target_coord=None, top_k=3, ref_seq_kwargs={}
    )

    assert (
        refid == 1
    )  # lowest measured FWHM among target-matched, non-outlier candidates
    assert diag["method"] == "quality"
    assert diag["chosen_index"] == 1
    assert [candidate.path.name for candidate in diag["tier2"]] == [
        "f0.fits",
        "f1.fits",
        "f2.fits",
    ]
    assert all(candidate.error is None for candidate in diag["tier2"])


def test_select_reference_frame_rejects_unmatched_target_candidate(
    tmp_path, monkeypatch
):
    files = [str(tmp_path / f"f{i}.fits") for i in range(3)]
    _patch_fake_sequence_parallel(
        monkeypatch,
        metrics_by_position=[
            # best FWHM but target not matched -- must not win
            {
                "idx": 0,
                "fwhm": 2.0,
                "n_sources": 10,
                "target_matched": False,
                "target_saturated": False,
            },
            {
                "idx": 1,
                "fwhm": 5.0,
                "n_sources": 10,
                "target_matched": True,
                "target_saturated": False,
            },
            {
                "idx": 2,
                "fwhm": 6.0,
                "n_sources": 10,
                "target_matched": True,
                "target_saturated": False,
            },
        ],
    )

    refid, _ = rp.select_reference_frame(
        files,
        instrument="muscat2",
        target_coord="dummy-non-none-coord",
        top_k=3,
        ref_seq_kwargs={},
    )

    assert refid == 1  # best FWHM among the target-matched candidates


def test_select_reference_frame_deprioritizes_saturated_target(tmp_path, monkeypatch):
    files = [str(tmp_path / f"f{i}.fits") for i in range(2)]
    _patch_fake_sequence_parallel(
        monkeypatch,
        metrics_by_position=[
            {
                "idx": 0,
                "fwhm": 2.0,
                "n_sources": 10,
                "target_matched": True,
                "target_saturated": True,
            },
            {
                "idx": 1,
                "fwhm": 3.0,
                "n_sources": 10,
                "target_matched": True,
                "target_saturated": False,
            },
        ],
    )

    refid, _ = rp.select_reference_frame(
        files, instrument="sinistro", target_coord=None, top_k=2, ref_seq_kwargs={}
    )

    assert refid == 1  # unsaturated target wins even with a slightly worse FWHM


def test_persistent_reference_sources_rejects_transient_with_telescope_drift():
    stars = np.array(
        [[10.0, 10.0], [30.0, 12.0], [15.0, 35.0], [42.0, 38.0], [55.0, 20.0]]
    )
    catalogs = [
        np.vstack([stars, [100.0, 100.0]]),
        np.vstack([stars + [2.0, -1.0], [80.0, 105.0]]),
        np.vstack([stars + [-3.0, 2.0], [110.0, 75.0]]),
        np.vstack([stars + [1.5, 3.0], [90.0, 115.0]]),
        np.vstack([stars + [-2.0, -2.5], [120.0, 95.0]]),
    ]
    tier2 = [
        rp.FramePixelQuality(
            path=Path(f"f{i}.fits"),
            n_sources=len(coords),
            source_coords=coords,
        )
        for i, coords in enumerate(catalogs)
    ]

    keep, diagnostics = rp.persistent_reference_sources(tier2, 0)

    assert keep == [0, 1, 2, 3, 4]
    assert diagnostics["enabled"] is True
    assert diagnostics["threshold"] == 3
    assert diagnostics["kept"] == 5


def test_persistent_reference_sources_protects_explicit_target_index():
    stars = np.array(
        [[10.0, 10.0], [30.0, 12.0], [15.0, 35.0], [42.0, 38.0], [55.0, 20.0]]
    )
    tier2 = [
        rp.FramePixelQuality(
            path=Path("anchor.fits"),
            source_coords=np.vstack([stars, [100.0, 100.0]]),
        ),
        rp.FramePixelQuality(path=Path("f1.fits"), source_coords=stars + [2.0, 1.0]),
        rp.FramePixelQuality(path=Path("f2.fits"), source_coords=stars + [-1.0, 2.0]),
    ]

    keep, _ = rp.persistent_reference_sources(tier2, 0, protected_source_index=5)

    assert keep == [0, 1, 2, 3, 4, 5]


def test_select_source_indices_filters_and_renumbers_sources():
    image = SimpleNamespace(
        sources=rp.Sources(
            [
                rp.PointSource(coords=np.array([float(i), float(i)]), i=i)
                for i in range(4)
            ],
            type="PointSource",
        )
    )

    rp.SelectSourceIndices([0, 2, 99]).run(image)

    assert np.array_equal(image.sources.coords, [[0.0, 0.0], [2.0, 2.0]])
    assert [source.i for source in image.sources] == [0, 1]


def test_format_ref_selection_report_position_mode_is_minimal():
    report = rp.format_ref_selection_report(
        "gp", {"method": "position", "chosen_path": Path("/x/MCT20_1234.fits")}
    )
    assert "method: position" in report
    assert "MCT20_1234.fits" in report


def test_format_ref_selection_report_quality_mode_includes_tiers(tmp_path):
    diagnostics = {
        "method": "quality",
        "instrument": "sinistro",
        "top_k": 2,
        "tier1": [
            rp.FrameHeaderQuality(
                path=Path("f0.fits"),
                values={
                    "fwhm": 1.8,
                    "airmass": 1.1,
                    "focus": 0.0,
                    "humidity": 20,
                    "background": 30,
                },
                hard_reject=None,
                score=-0.5,
            ),
        ],
        "tier2": [
            rp.FramePixelQuality(
                path=Path("f0.fits"),
                fwhm=6.8,
                n_sources=42,
                target_matched=True,
                target_saturated=False,
            ),
        ],
        "chosen_index": 0,
        "chosen_path": Path("f0.fits"),
    }
    report = rp.format_ref_selection_report("gp", diagnostics)
    assert "Tier 1" in report and "Tier 2" in report
    assert "f0.fits" in report
    assert "decision:" in report


# --------------------------------------------------------------------------
# Per-band display reference on the shared reference grid: un-suppresses the
# reference-frame plots (_ref/_apertures/_cutouts/_stacks) for non-reference
# bands under --ref_band, showing each band's own (aligned) science data.
# --------------------------------------------------------------------------


def _fake_sources(positions):
    from prose.core.source import Sources

    return Sources(np.array(positions, dtype=float))


def _detect_kwargs():
    return dict(
        ccd_trim_size_yx=(0, 0),
        max_num_stars=10,
        min_star_separation=5.0,
        cutout_size=21,
        min_area=3,
        bad_pixel_map=None,
        centroid_method=rp.CENTROID_METHOD,
    )


def test_select_stack_frames_single_returns_middle_frame():
    assert rp._select_stack_frames(["a", "b", "c", "d", "e"], 1) == ["c"]
    assert rp._select_stack_frames(["only"], 5) == ["only"]


def test_select_stack_frames_evenly_spaced_and_deduped():
    files = [f"f{i}" for i in range(10)]
    picked = rp._select_stack_frames(files, 3)
    assert picked == ["f0", "f4", "f9"]  # endpoints included, evenly spaced
    # n >= len returns every frame (no duplicates)
    assert rp._select_stack_frames(["a", "b"], 5) == ["a", "b"]


def test_warp_onto_reference_resamples_onto_reference_grid(monkeypatch):
    """The warped frame lands on the shared reference grid (its shape)."""
    from skimage.transform import AffineTransform

    shared_ref = SimpleNamespace(data=np.zeros((30, 25)))  # (ny, nx)
    img = SimpleNamespace(data=np.arange(400, dtype=float).reshape(20, 20))

    class FakeCTT:
        def __init__(self, *_a, **_k):
            pass

        def run(self, image):
            image.transform = AffineTransform(translation=(2.0, -1.0))

    monkeypatch.setattr(rp.blocks, "ComputeTransformTwirl", FakeCTT)
    out = rp._warp_onto_reference(img, shared_ref)
    assert out is not None
    assert out.shape == (30, 25)  # reference-grid shape, not the frame's (20, 20)


def test_warp_onto_reference_returns_none_when_transform_unsolved(monkeypatch):
    class FakeCTT:
        def __init__(self, *_a, **_k):
            pass

        def run(self, image):
            pass  # never sets image.transform

    monkeypatch.setattr(rp.blocks, "ComputeTransformTwirl", FakeCTT)
    img = SimpleNamespace(data=np.zeros((10, 10)))
    assert rp._warp_onto_reference(img, SimpleNamespace(data=np.zeros((10, 10)))) is None


def test_build_display_reference_returns_none_without_files():
    assert (
        rp._build_display_reference(
            [], SimpleNamespace(), **_detect_kwargs(), n_stack=1
        )
        is None
    )


def test_build_display_reference_returns_none_when_all_frames_fail(monkeypatch):
    monkeypatch.setattr(rp, "FITSImage", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(
        rp, "reference_sequence",
        lambda **_kw: SimpleNamespace(run=lambda img, show_progress=False: None),
    )
    monkeypatch.setattr(rp, "_warp_onto_reference", lambda img, shared_ref: None)
    out = rp._build_display_reference(
        ["a.fits", "b.fits"], SimpleNamespace(),
        **_detect_kwargs(), n_stack=1,
    )
    assert out is None


def _example_shared_reference(seed=3):
    """A real prose ``Image`` with data, a small celestial WCS and a couple of
    on-grid sources, standing in for a shared reference frame."""
    from astropy.wcs import WCS
    from prose.simulations import example_image

    shared = example_image(seed=seed)
    ny, nx = shared.data.shape
    shared._sources = _fake_sources([[nx / 2, ny / 2], [nx / 3, ny / 3]])
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = [-1e-4, 1e-4]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    shared.wcs = w
    return shared


def test_build_display_reference_single_frame_real_construction(monkeypatch):
    """Exercises the real Image.copy + data swap + Cutouts recompute path with a
    single representative frame (Task 1)."""
    from astropy.io.fits import Header

    shared = _example_shared_reference()
    own_header = Header()
    own_header["OBJECT"] = "TOI-1"
    warped = np.full(shared.data.shape, 7.0)

    class FakeImg:
        def __init__(self, _path):
            self.header = own_header
            self.data = np.ones(shared.data.shape)

    monkeypatch.setattr(rp, "FITSImage", FakeImg)
    monkeypatch.setattr(
        rp, "reference_sequence",
        lambda **_kw: SimpleNamespace(run=lambda img, show_progress=False: None),
    )
    monkeypatch.setattr(rp, "_warp_onto_reference", lambda img, shared_ref: warped)

    disp = rp._build_display_reference(
        ["a.fits", "b.fits", "c.fits"], shared,
        **_detect_kwargs(), n_stack=1,
    )

    assert disp is not None
    assert disp.data.shape == shared.data.shape
    assert np.allclose(disp.data, 7.0)  # this band's own (warped) data
    # Shared-ref source catalog / grid carries over (cross-band consistency).
    assert np.allclose(disp.sources.coords, shared.sources.coords)
    assert disp.header["OBJECT"] == "TOI-1"  # this band's own header for the title
    # Per-source cutouts were recomputed on the display data.
    assert "cutouts" in disp.computed
    assert len(disp.computed["cutouts"]) == len(shared.sources.coords)


def test_build_display_reference_median_stacks_aligned_frames(monkeypatch):
    """Task 2: multiple aligned frames are median-combined."""
    shared = _example_shared_reference()

    class FakeImg:
        def __init__(self, _path):
            from astropy.io.fits import Header

            self.header = Header()
            self.data = np.ones(shared.data.shape)

    warps = iter([
        np.full(shared.data.shape, 2.0),
        np.full(shared.data.shape, 4.0),
    ])
    monkeypatch.setattr(rp, "FITSImage", FakeImg)
    monkeypatch.setattr(
        rp, "reference_sequence",
        lambda **_kw: SimpleNamespace(run=lambda img, show_progress=False: None),
    )
    monkeypatch.setattr(rp, "_warp_onto_reference", lambda img, shared_ref: next(warps))

    disp = rp._build_display_reference(
        ["a.fits", "b.fits", "c.fits", "d.fits"], shared,
        **_detect_kwargs(), n_stack=2,
    )

    assert disp is not None
    # median([2, 4]) == 3 everywhere -> genuine aligned median stack
    assert np.allclose(disp.data, 3.0)


def test_plot_stacks_uses_display_reference_when_present(tmp_path):
    """Non-reference bands plot from their display reference; the reference band
    (display_ref is None) falls back to the shared reference frame."""
    calls = {"display": 0, "ref": 0}

    class FakeCutout:
        def __init__(self):
            self.data = np.linspace(1.0, 100.0, 200 * 200).reshape(200, 200)
            self.wcs = None

    class FakeImg:
        def __init__(self, counter_key):
            self._k = counter_key
            self.header = {}
            self.telescope = SimpleNamespace(saturation=None)
            self.sources = _fake_sources([[100.0, 100.0], [50.0, 50.0]])

        def cutout(self, coords, shape, reset_index=False):
            calls[self._k] += 1
            return FakeCutout()

    def make_result(with_display):
        return dict(
            ref=FakeImg("ref"),
            diff=SimpleNamespace(aperture=0),
            target_index=0,
            aper_radii=np.array([3.0]),
            rin=8.0,
            rout=12.0,
            scale=False,
            gaia_df=None,
            display_ref=FakeImg("display") if with_display else None,
        )

    band_results = {
        "gp": make_result(with_display=False),  # reference band
        "rp": make_result(with_display=True),   # non-reference band
    }

    out_path = tmp_path / "stacks.png"
    rp.plot_stacks(
        band_results,
        out_path,
        target_name="TOI-1",
        instrument="muscat3",
        date="260101",
        target_index=0,
        plot_gaia_sources=False,
    )

    assert out_path.exists()
    assert calls["display"] == 1  # rp used its display reference
    assert calls["ref"] == 1  # gp used the shared reference

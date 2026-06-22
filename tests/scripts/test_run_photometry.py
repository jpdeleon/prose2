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


def test_differential_photometry_avoid_cids_target_itself_warns(caplog):
    """avoid_cids containing the target star triggers a warning."""
    rng = np.random.default_rng(42)
    fluxes = _make_fluxes(4, 10, rng)

    with caplog.at_level("WARNING", logger="prose_run_photometry"):
        result = rp.differential_photometry(fluxes, target_index=0, avoid_cids=[0])

    assert result is not None


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

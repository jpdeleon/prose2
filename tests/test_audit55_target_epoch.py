"""#55 A1: the target must be identified as a Gaia source and placed at the
observation epoch, and no step may take "nearest Gaia row" as the target
without a separation limit."""

import logging

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from prose.scripts import run_photometry as rp
from prose.target_epoch import (
    GAIA_DR2_EPOCH,
    RESOLVER_EPOCH,
    gaia_target_row,
    normalize_gaia_columns,
    propagate_linear,
    select_gaia_target,
)

OBS = Time("2026-04-16T00:00:00")
# High-proper-motion target: 1 arcsec/yr due north. Its J2015.5 Gaia position:
TARGET_GAIA = SkyCoord(150.0, 20.0, unit="deg")
PM_DEC = 1000.0  # mas/yr


def _offset(coord, east_arcsec=0.0, north_arcsec=0.0):
    return coord.spherical_offsets_by(east_arcsec * u.arcsec, north_arcsec * u.arcsec)


def _field():
    """Gaia rows (J2015.5): the moving target and a static neighbour that sits
    3 arcsec from where the target was at J2000."""
    j2000_target = _offset(TARGET_GAIA, north_arcsec=-15.5)
    neighbour = _offset(j2000_target, east_arcsec=3.0)
    return pd.DataFrame(
        {
            "source_id": ["target", "neighbour"],
            "ra": [TARGET_GAIA.ra.deg, neighbour.ra.deg],
            "dec": [TARGET_GAIA.dec.deg, neighbour.dec.deg],
            "pmra": [0.0, 0.0],
            "pmdec": [PM_DEC, 0.0],
            "phot_g_mean_mag": [11.0, 9.0],
        }
    ), j2000_target


# ------------------------------- propagation -------------------------------


def test_propagate_linear_moves_along_proper_motion():
    moved = propagate_linear(TARGET_GAIA, 0.0, PM_DEC, GAIA_DR2_EPOCH, OBS)
    years = (OBS - GAIA_DR2_EPOCH).to_value(u.yr)
    dlon, dlat = TARGET_GAIA.spherical_offsets_to(moved)
    assert dlat.to_value(u.arcsec) == pytest.approx(years, abs=1e-3)
    assert dlon.to_value(u.arcsec) == pytest.approx(0.0, abs=1e-3)


def test_propagate_linear_backwards_and_east():
    moved = propagate_linear(TARGET_GAIA, 2000.0, 0.0, GAIA_DR2_EPOCH, RESOLVER_EPOCH)
    dlon, _ = TARGET_GAIA.spherical_offsets_to(moved)
    assert dlon.to_value(u.arcsec) == pytest.approx(-31.0, abs=1e-3)


def test_propagate_linear_treats_missing_proper_motion_as_static():
    moved = propagate_linear(TARGET_GAIA, np.nan, np.nan, GAIA_DR2_EPOCH, OBS)
    assert TARGET_GAIA.separation(moved).arcsec == pytest.approx(0.0, abs=1e-9)


# -------------------------------- selection --------------------------------


def test_stale_j2000_coordinate_selects_moving_target_not_nearer_neighbour():
    gaia, j2000_target = _field()
    # The old rule: nearest row to the J2000 coordinate is the neighbour.
    seps = j2000_target.separation(
        SkyCoord(gaia.ra.values, gaia.dec.values, unit="deg")
    )
    assert int(np.argmin(seps)) == 1

    match = select_gaia_target(j2000_target, gaia, OBS)

    assert match is not None and match.source_id == "target"
    expected = _offset(TARGET_GAIA, north_arcsec=(OBS - GAIA_DR2_EPOCH).to_value(u.yr))
    assert match.at(OBS).separation(expected).arcsec < 0.01
    assert match.pm_total_masyr == pytest.approx(PM_DEC)


def test_observation_epoch_coordinate_also_matches():
    gaia, _ = _field()
    tonight = _offset(TARGET_GAIA, north_arcsec=(OBS - GAIA_DR2_EPOCH).to_value(u.yr))
    match = select_gaia_target(tonight, gaia, OBS)
    assert match is not None and match.source_id == "target"


def test_no_source_within_limit_returns_none():
    gaia, j2000_target = _field()
    far = _offset(j2000_target, east_arcsec=-10.0)
    assert select_gaia_target(far, gaia, OBS) is None


def test_normalize_accepts_vizier_column_names():
    vizier = pd.DataFrame(
        {
            "Source": [1],
            "RA_ICRS": [1.0],
            "DE_ICRS": [2.0],
            "pmRA": [3.0],
            "pmDE": [4.0],
            "Gmag": [5.0],
        }
    )
    out = normalize_gaia_columns(vizier)
    assert out.iloc[0].to_dict() == {
        "source_id": 1,
        "ra": 1.0,
        "dec": 2.0,
        "pmra": 3.0,
        "pmdec": 4.0,
        "phot_g_mean_mag": 5.0,
    }


def test_normalize_requires_coordinates():
    with pytest.raises(ValueError, match="ra"):
        normalize_gaia_columns(pd.DataFrame({"x": [1]}))


def test_gaia_target_row_requires_separation_limit():
    rows = SkyCoord([150.0, 150.001], [20.0, 20.0], unit="deg")
    assert gaia_target_row(SkyCoord(150.0, 20.0001, unit="deg"), rows) == 0
    assert gaia_target_row(_offset(rows[0], north_arcsec=5.0), rows) is None


# ----------------------------- run_photometry ------------------------------


@pytest.mark.gaia_resolution
def test_resolve_target_at_epoch_returns_tonights_position(monkeypatch):
    gaia, j2000_target = _field()
    monkeypatch.setattr(rp, "gaia_query", lambda *a, **kw: Table.from_pandas(gaia))

    coord = rp.resolve_target_at_epoch(j2000_target, OBS)

    expected = _offset(TARGET_GAIA, north_arcsec=(OBS - GAIA_DR2_EPOCH).to_value(u.yr))
    assert coord.separation(expected).arcsec < 0.01
    assert j2000_target.separation(coord).arcsec > 26.0


@pytest.mark.gaia_resolution
@pytest.mark.parametrize("failure", ["network", "no_match", "no_date"])
def test_resolve_target_at_epoch_falls_back_with_warning(monkeypatch, caplog, failure):
    gaia, j2000_target = _field()
    if failure == "network":

        def _raise(*a, **kw):
            raise TimeoutError("offline")

        monkeypatch.setattr(rp, "gaia_query", _raise)
    else:
        monkeypatch.setattr(rp, "gaia_query", lambda *a, **kw: Table.from_pandas(gaia))
    coord = (
        _offset(j2000_target, east_arcsec=-10.0)
        if failure == "no_match"
        else j2000_target
    )
    obs = None if failure == "no_date" else OBS

    with caplog.at_level(logging.WARNING, logger=rp.logger.name):
        result = rp.resolve_target_at_epoch(coord, obs)

    assert result is coord
    assert "propagat" in caplog.text


def _gaia_ref(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS

    from prose import FITSImage

    w = WCS(naxis=2)
    w.wcs.crpix = [50, 50]
    w.wcs.cdelt = [-1 / 3600, 1 / 3600]
    w.wcs.crval = [TARGET_GAIA.ra.deg, TARGET_GAIA.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()
    header["PIXSCALE"] = 1.0
    path = tmp_path / "ref.fits"
    fits.writeto(path, np.ones((100, 100)), header)
    ref = FITSImage(path)
    ref.fwhm = 3.0
    return ref


def test_aperture_sizing_never_uses_a_neighbour_as_the_target(
    tmp_path, monkeypatch, caplog
):
    """The stale coordinate's nearest Gaia row is a bright neighbour; it must
    not supply the target magnitude for the contamination test."""
    ref = _gaia_ref(tmp_path)
    gaia, _ = _field()
    neighbour = SkyCoord(gaia.ra[1], gaia.dec[1], unit="deg")
    stale = _offset(neighbour, east_arcsec=-2.5)  # 2.5" from neighbour only
    monkeypatch.setattr(rp, "_gaia_catalog_df", lambda *a, **kw: gaia)

    with caplog.at_level(logging.WARNING, logger=rp.logger.name):
        rp.gaia_aperture_radii(ref, 0, stale)

    assert "no Gaia source within 2 arcsec of the target" in caplog.text


def test_aperture_sizing_counts_bright_neighbour_once_target_matches(
    tmp_path, monkeypatch
):
    ref = _gaia_ref(tmp_path)
    target = SkyCoord(150.0, 20.0, unit="deg")
    near = _offset(target, east_arcsec=22.0)  # inside nominal (20, 30) annulus
    gaia = pd.DataFrame(
        {
            "ra": [near.ra.deg, target.ra.deg],
            "dec": [near.dec.deg, target.dec.deg],
            "phot_g_mean_mag": [10.0, 11.0],
        }
    )
    monkeypatch.setattr(rp, "_gaia_catalog_df", lambda *a, **kw: gaia)

    _, rin, rout = rp.gaia_aperture_radii(ref, 0, target)

    assert rout < 22.0  # annulus pulled inside the contaminant


def test_gaia_overlay_target_mask_requires_match():
    class _Cutout:
        def __init__(self, wcs):
            self.wcs = wcs
            self.data = np.zeros((40, 40))

    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [20, 20]
    w.wcs.cdelt = [-1 / 3600, 1 / 3600]
    w.wcs.crval = [150.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    star = _offset(SkyCoord(150.0, 20.0, unit="deg"), east_arcsec=8.0)
    gaia = pd.DataFrame(
        {"ra": [star.ra.deg], "dec": [star.dec.deg], "phot_g_mean_mag": [12.0]}
    )

    proj = rp._project_gaia_to_cutout(
        _Cutout(w), gaia, SkyCoord(150.0, 20.0, unit="deg")
    )

    assert proj is not None and not proj.target_mask.any()
    assert np.isnan(proj.target_mag)


def test_gaia_df_at_epoch_moves_raw_query_rows():
    gaia, _ = _field()
    moved = rp._gaia_df_at_epoch(gaia, OBS)
    years = (OBS - GAIA_DR2_EPOCH).to_value(u.yr)
    target = SkyCoord(moved.ra[0], moved.dec[0], unit="deg")
    assert TARGET_GAIA.separation(target).arcsec == pytest.approx(years, abs=1e-3)
    assert moved.ra[1] == pytest.approx(gaia.ra[1])
    assert gaia.dec[0] == TARGET_GAIA.dec.deg  # input left unchanged


# ------------------------------ review follow-up ------------------------------


def _crossing_field():
    """A static target 0.3" from the J2000 input coordinate, and a fast mover
    whose track crossed that coordinate exactly in 2008."""
    coord = SkyCoord(150.0, 20.0, unit="deg")
    target = _offset(coord, east_arcsec=0.3)
    years = GAIA_DR2_EPOCH.jyear - 2008.0
    mover = _offset(coord, north_arcsec=years * 0.5)  # 500 mas/yr north
    gaia = pd.DataFrame(
        {
            "source_id": ["target", "mover"],
            "ra": [target.ra.deg, mover.ra.deg],
            "dec": [target.dec.deg, mover.dec.deg],
            "pmra": [0.0, 0.0],
            "pmdec": [0.0, 500.0],
            "phot_g_mean_mag": [12.0, 10.0],
        }
    )
    return coord, gaia


def test_star_on_the_coordinate_beats_a_mover_that_crossed_it_earlier():
    coord, gaia = _crossing_field()

    match = select_gaia_target(coord, gaia, OBS)

    assert match.source_id == "target"
    assert match.n_candidates == 2


@pytest.mark.gaia_resolution
def test_ambiguous_gaia_match_is_warned(monkeypatch, caplog):
    coord, gaia = _crossing_field()
    monkeypatch.setattr(rp, "gaia_query", lambda *a, **kw: Table.from_pandas(gaia))
    with caplog.at_level(logging.WARNING, logger=rp.logger.name):
        rp.resolve_target_at_epoch(coord, OBS)
    assert "2 Gaia sources pass within 2 arcsec" in caplog.text


@pytest.mark.gaia_resolution
def test_missing_target_coordinate_skips_the_query(monkeypatch):
    monkeypatch.setattr(rp, "gaia_query", pytest.fail)
    assert rp.resolve_target_at_epoch(None, OBS) is None


def test_main_hands_the_propagated_coordinate_to_run_band(tmp_path, monkeypatch):
    """main() must resolve the target at the first frame's date and reduce
    with that position, not the input coordinate."""
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [10, 10]
    w.wcs.cdelt = [0.01, 0.01]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()
    header.update(
        TELID="1m0a",
        INSTRUME="sinistro",
        SITEID="lsc",
        CONFMODE="central_2k_2x2",
        OBJECT="audit",
        EXPTIME=1.0,
        FILTER="gp",
        JD=2460000.0,
        **{"DATE-OBS": "2025-04-16T00:00:00"},
    )
    fits.writeto(tmp_path / "science.fits", np.ones((20, 20)), header)
    tonight = SkyCoord(0.01, 0.01, unit="deg")
    calls = {}

    def spy_resolver(coord, obs_time):
        calls["input"], calls["time"] = coord, obs_time
        return tonight

    def capture_run_band(band, files, ref_file, target_coord, **kw):
        calls["run_band"] = target_coord
        raise rp.TargetNotFoundError("stop here")

    monkeypatch.setattr(rp, "resolve_target_at_epoch", spy_resolver)
    monkeypatch.setattr(rp, "run_band", capture_run_band)

    rp.main(
        [
            "--target_name",
            "audit",
            "--target_coord",
            "0",
            "0",
            "--data_dir",
            str(tmp_path),
            "--results_dir",
            str(tmp_path / "out"),
            "--bands",
            "gp",
            "--glob",
            "*.fits",
        ]
    )

    assert calls["input"].separation(SkyCoord(0, 0, unit="deg")).arcsec < 1e-6
    assert calls["time"].datetime.date().isoformat() == "2025-04-16"
    assert calls["run_band"] is tonight


def test_gaia_cache_key_depends_on_observation_epoch(monkeypatch):
    paths = []
    monkeypatch.setattr(rp, "_gaia_cache", {})
    monkeypatch.setattr(rp, "load_cached_df", lambda p: paths.append(p) or "df")
    coord = SkyCoord(150.0, 20.0, unit="deg")
    for date in ("2020-01-01", "2026-01-01"):
        monkeypatch.setattr(rp, "_observation_time", lambda _ref, d=date: Time(d))
        rp._gaia_catalog_df(object(), 0, coord, 1.0)
    assert len(set(paths)) == 2
    assert "ep2020.0" in paths[0].name and "ep2026.0" in paths[1].name

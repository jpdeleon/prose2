"""Tests for the shared coordinate-keyed catalog cache in ``prose.utils``.

The cache backs both the Gaia (``run_photometry``) and SIMBAD queries: results
are written under ``CACHE_DIR/<subdir>`` and reused when a live query is
unavailable (offline fallback). These tests are deterministic and need no
network -- the SIMBAD query is monkeypatched.
"""

import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from prose import utils as pu


def _coord(ra=10.12345678, dec=-20.87654321):
    return SkyCoord(ra, dec, unit="deg")


# --------------------------- coord_cache_path ---------------------------


def test_coord_cache_path_encodes_coord_and_key_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(pu, "CACHE_DIR", tmp_path)
    path = pu.coord_cache_path("gaia", _coord(), 200)
    assert path.parent == tmp_path / "gaia"
    assert path.name == "ra10.12346_dec-20.87654_200.csv"  # 5 dp + key part


def test_coord_cache_path_distinguishes_key_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(pu, "CACHE_DIR", tmp_path)
    gaia = pu.coord_cache_path("gaia", _coord(), 200)
    simbad = pu.coord_cache_path("simbad", _coord(), "muscat4", "5")
    assert gaia != simbad
    assert simbad.name == "ra10.12346_dec-20.87654_muscat4_5.csv"


# --------------------------- save / load round-trip ---------------------------


def test_save_then_load_df_roundtrip(tmp_path):
    df = pd.DataFrame(
        {"ra": [10.1, 10.2], "dec": [-20.8, -20.9], "phot_g_mean_mag": [12.0, 15.5]}
    )
    path = tmp_path / "nested" / "cat.csv"
    assert pu.save_cached_df(path, df) is True  # creates parent dirs
    assert path.is_file()
    pd.testing.assert_frame_equal(pu.load_cached_df(path), df)


def test_load_missing_returns_none(tmp_path):
    assert pu.load_cached_df(tmp_path / "absent.csv") is None


def test_save_to_unwritable_path_returns_false_without_raising(tmp_path):
    # a path whose parent is an existing *file* cannot be created
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    assert pu.save_cached_df(blocker / "cat.csv", pd.DataFrame({"a": [1]})) is False


# --------------------------- SIMBAD offline fallback ---------------------------


@pytest.fixture
def _isolated_simbad(monkeypatch, tmp_path):
    """Point the cache at tmp, clear the in-memory cache, neutralise votable
    field setup, and hand back the cache dir."""
    monkeypatch.setattr(pu, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pu.Simbad, "add_votable_fields", lambda *a, **k: None)
    pu._simbad_cache.clear()
    return tmp_path


def _raise_offline(*a, **k):
    raise RuntimeError("network unreachable")


def test_get_simbad_data_falls_back_to_cache_when_query_fails(
    _isolated_simbad, monkeypatch
):
    coord = _coord()
    cached = pd.DataFrame({"RA": ["00 40 30"], "DEC": ["+41 00 00"], "OTYPE": ["G"]})
    # seed a previous run's cache at the exact path get_simbad_data will compute
    pu.save_cached_df(pu.coord_cache_path("simbad", coord, "muscat4", "5"), cached)
    monkeypatch.setattr(pu.Simbad, "query_region", _raise_offline)  # offline

    result = pu.get_simbad_data(coord, "muscat4", fov_arcmin=5)
    pd.testing.assert_frame_equal(result, cached)


def test_get_simbad_data_returns_none_when_query_fails_and_no_cache(
    _isolated_simbad, monkeypatch
):
    monkeypatch.setattr(pu.Simbad, "query_region", _raise_offline)
    assert pu.get_simbad_data(_coord(), "muscat4", fov_arcmin=5) is None


def test_get_simbad_data_no_sources_returns_none_without_caching(
    _isolated_simbad, monkeypatch
):
    monkeypatch.setattr(pu.Simbad, "query_region", lambda *a, **k: None)  # empty result
    coord = _coord()
    assert pu.get_simbad_data(coord, "muscat4", fov_arcmin=5) is None
    # a genuine "no sources" answer is not written to disk
    assert not pu.coord_cache_path("simbad", coord, "muscat4", "5").exists()


# --------------------------- get_saturation_from_header ---------------------------


def test_get_saturation_from_header_sinistro_central_2k_2x2():
    # 1. BANZAI-reduced header (GAIN = 1.0, SATURATE in electrons)
    h_reduced = {
        "TELID": "1m0a",
        "SITEID": "coj",
        "CONFMODE": "central_2k_2x2",
        "GAIN": 1.0,
        "SATURATE": 244000.0,
        "MAXLIN": 244000.0,
        "filter": "zs",
    }
    limits_reduced = pu.get_saturation_from_header(h_reduced)
    assert limits_reduced["zs"] == 244000.0

    # 2. Raw header / gain != 1.0 (fallback to hardcoded values)
    h_raw = {
        "TELID": "1m0a",
        "SITEID": "coj",
        "CONFMODE": "central_2k_2x2",
        "GAIN": 6.6,
        "SATURATE": 37000.0,
        "MAXLIN": 37000.0,
        "filter": "zs",
    }
    limits_raw = pu.get_saturation_from_header(h_raw)
    assert limits_raw["zs"] == pytest.approx(340000.0 / 6.6)


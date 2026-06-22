"""Tests for ``prose.scripts.solve_wcs_astrometry``.

All tests are deterministic and offline — they use minimal in-memory FITS
files and mock the ``astroquery.astrometry_net.AstrometryNet`` client so no
network calls are made.  A separate integration marker (``@pytest.mark.net``)
gates the live test for ``muscat2/260310/TOI07475.01``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import prose.scripts.solve_wcs_astrometry as swa


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_wcs(pixscale_arcsec: float = 0.44) -> WCS:
    """Return a minimal TAN WCS with a known pixel scale."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [512.0, 512.0]
    wcs.wcs.crval = [182.038, 0.099]  # ~RA/Dec of TOI-7475
    pscale_deg = pixscale_arcsec / 3600.0
    wcs.wcs.cdelt = [-pscale_deg, pscale_deg]
    wcs.wcs.set()
    return wcs


def _fake_fits(path: Path, ra: str = "12:08:09", dec: str = "+0:05:58") -> None:
    """Write a minimal FITS file with MuSCAT2-like headers."""
    data = np.zeros((32, 32), dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["INSTRUME"] = "MuSCAT2"
    hdu.header["OBJECT"] = "TOI07475.01"
    hdu.header["EXPTIME"] = 0.5
    hdu.header["DATE-OBS"] = "2026-3-10"
    hdu.header["MJD-STRT"] = 61109.9729769826
    hdu.header["FILTER"] = "g"
    hdu.header["RA"] = ra
    hdu.header["DEC"] = dec
    hdu.writeto(str(path), overwrite=True)


@pytest.fixture()
def probe_fits(tmp_path) -> Path:
    """A single minimal MuSCAT2-like science FITS."""
    fp = tmp_path / "MCT20_2603100239_calibrated.fits"
    _fake_fits(fp)
    return fp


@pytest.fixture()
def science_dir(tmp_path) -> Path:
    """Directory of 3 science FITS files (MCT20 prefix)."""
    for i in range(1, 4):
        _fake_fits(tmp_path / f"MCT20_260310000{i}_calibrated.fits")
    return tmp_path


@pytest.fixture()
def valid_wcs() -> WCS:
    return _make_wcs(0.44)


@pytest.fixture()
def invalid_wcs() -> WCS:
    """WCS with pixel scale outside any instrument range (100 arcsec/pix)."""
    return _make_wcs(100.0)


# ---------------------------------------------------------------------------
# _header_hint
# ---------------------------------------------------------------------------


class TestHeaderHint:
    def test_returns_ra_dec_from_header(self, probe_fits):
        hints = swa._header_hint(probe_fits)
        assert "center_ra" in hints
        assert "center_dec" in hints

    def test_ra_in_degrees_from_sexagesimal_hours(self, probe_fits):
        hints = swa._header_hint(probe_fits)
        # "12:08:09" h → 182.0375 deg
        assert abs(hints["center_ra"] - 182.0375) < 0.01

    def test_dec_in_degrees(self, probe_fits):
        hints = swa._header_hint(probe_fits)
        # "+0:05:58" → 0.09944 deg
        assert abs(hints["center_dec"] - 0.0994) < 0.01

    def test_includes_pixel_scale(self, probe_fits):
        hints = swa._header_hint(probe_fits)
        assert "scale_est" in hints
        assert abs(hints["scale_est"] - 0.44) < 0.01
        assert hints["scale_err"] == 20.0

    def test_missing_file_returns_empty(self, tmp_path):
        hints = swa._header_hint(tmp_path / "nonexistent.fits")
        assert hints == {}

    def test_radius_hint_present(self, probe_fits):
        hints = swa._header_hint(probe_fits)
        assert "radius" in hints


# ---------------------------------------------------------------------------
# validate_wcs
# ---------------------------------------------------------------------------


class TestValidateWcs:
    def test_valid_muscat2_wcs(self, valid_wcs):
        assert swa.validate_wcs(valid_wcs, "muscat2") is True

    def test_invalid_pixel_scale_rejected(self, invalid_wcs):
        assert swa.validate_wcs(invalid_wcs, "muscat2") is False

    def test_unknown_instrument_accepts_any_scale(self, valid_wcs):
        assert swa.validate_wcs(valid_wcs, None) is True

    def test_no_celestial_axes_rejected(self):
        wcs = WCS(naxis=2)  # no ctype set
        assert swa.validate_wcs(wcs, "muscat2") is False


# ---------------------------------------------------------------------------
# save_wcs_fits / load_wcs_fits
# ---------------------------------------------------------------------------


class TestSaveLoadWcsFits:
    def test_roundtrip(self, tmp_path, valid_wcs):
        path = tmp_path / "wcs.fits"
        swa.save_wcs_fits(valid_wcs, path)
        assert path.is_file()
        loaded = swa.load_wcs_fits(path)
        assert loaded is not None
        assert getattr(loaded, "has_celestial", False)

    def test_loaded_wcs_matches_original(self, tmp_path, valid_wcs):
        path = tmp_path / "wcs.fits"
        swa.save_wcs_fits(valid_wcs, path)
        loaded = swa.load_wcs_fits(path)
        # pixel_to_world should give ~same coordinates
        orig = valid_wcs.pixel_to_world(512, 512)
        new_ = loaded.pixel_to_world(512, 512)
        assert abs(orig.ra.deg - new_.ra.deg) < 1e-4
        assert abs(orig.dec.deg - new_.dec.deg) < 1e-4

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert swa.load_wcs_fits(tmp_path / "missing.fits") is None

    def test_load_corrupt_returns_none(self, tmp_path):
        path = tmp_path / "bad.fits"
        path.write_text("not a fits file")
        assert swa.load_wcs_fits(path) is None


# ---------------------------------------------------------------------------
# inject_wcs_into_file
# ---------------------------------------------------------------------------


class TestInjectWcs:
    def test_injects_wcs_keywords(self, probe_fits, valid_wcs):
        assert swa.inject_wcs_into_file(probe_fits, valid_wcs) is True
        hdr = fits.getheader(str(probe_fits))
        assert "CRVAL1" in hdr or "NAXIS" in hdr  # WCS keywords present

    def test_injected_wcs_is_valid(self, probe_fits, valid_wcs):
        swa.inject_wcs_into_file(probe_fits, valid_wcs)
        hdr = fits.getheader(str(probe_fits))
        wcs = WCS(hdr)
        assert getattr(wcs, "has_celestial", False)

    def test_preserves_original_data(self, probe_fits, valid_wcs):
        data_before = fits.getdata(str(probe_fits))
        swa.inject_wcs_into_file(probe_fits, valid_wcs)
        data_after = fits.getdata(str(probe_fits))
        assert np.array_equal(data_before, data_after)

    def test_inject_bad_path_returns_false(self, tmp_path, valid_wcs):
        assert swa.inject_wcs_into_file(tmp_path / "ghost.fits", valid_wcs) is False

    def test_replaces_stale_wcs(self, probe_fits, valid_wcs):
        """Injecting twice should overwrite, not accumulate, WCS keywords."""
        swa.inject_wcs_into_file(probe_fits, valid_wcs)
        wcs2 = _make_wcs(0.44)
        wcs2.wcs.crval = [90.0, 45.0]
        wcs2.wcs.set()
        swa.inject_wcs_into_file(probe_fits, wcs2)
        hdr = fits.getheader(str(probe_fits))
        wcs_loaded = WCS(hdr)
        sky = wcs_loaded.pixel_to_world(512, 512)
        assert abs(sky.ra.deg - 90.0) < 1.0

    def test_preserves_observation_date_keywords(self, probe_fits, valid_wcs):
        """astropy reports DATE-OBS/MJD-OBS as WCS aux cards; injection must not
        delete them, or the downstream filename date collapses to empty."""
        before = fits.getheader(str(probe_fits))
        assert before["DATE-OBS"] == "2026-3-10"
        assert "MJD-STRT" in before
        swa.inject_wcs_into_file(probe_fits, valid_wcs)
        after = fits.getheader(str(probe_fits))
        assert after["DATE-OBS"] == "2026-3-10"
        assert after["MJD-STRT"] == before["MJD-STRT"]


# ---------------------------------------------------------------------------
# apply_wcs_to_directory
# ---------------------------------------------------------------------------


class TestApplyWcsToDirectory:
    def test_patches_all_matching_files(self, science_dir, valid_wcs):
        n_ok, n_fail = swa.apply_wcs_to_directory(valid_wcs, science_dir, "MCT20*.fits")
        assert n_ok == 3
        assert n_fail == 0

    def test_files_have_wcs_after_patch(self, science_dir, valid_wcs):
        swa.apply_wcs_to_directory(valid_wcs, science_dir, "MCT20*.fits")
        for fp in science_dir.glob("MCT20*.fits"):
            hdr = fits.getheader(str(fp))
            wcs = WCS(hdr)
            assert getattr(wcs, "has_celestial", False)

    def test_no_matching_files_returns_zeros(self, science_dir, valid_wcs):
        n_ok, n_fail = swa.apply_wcs_to_directory(
            valid_wcs, science_dir, "NOTHING*.fits"
        )
        assert n_ok == 0
        assert n_fail == 0


# ---------------------------------------------------------------------------
# find_science_files
# ---------------------------------------------------------------------------


class TestFindScienceFiles:
    def test_returns_sorted_paths(self, science_dir):
        files = swa.find_science_files(science_dir, "MCT20*.fits")
        assert len(files) == 3
        assert files == sorted(files)

    def test_empty_pattern_returns_empty(self, science_dir):
        assert swa.find_science_files(science_dir, "NONEXIST*.fits") == []


# ---------------------------------------------------------------------------
# upload_and_solve (mocked network)
# ---------------------------------------------------------------------------


class TestUploadAndSolve:
    def _mock_an(self, wcs_header):
        """Return a mock AstrometryNet whose solve_from_image returns wcs_header."""
        mock_an = MagicMock()
        mock_an.return_value.solve_from_image.return_value = wcs_header
        return mock_an

    def test_returns_wcs_on_success(self, probe_fits, valid_wcs):
        wcs_header = valid_wcs.to_header()
        with patch("prose.scripts.solve_wcs_astrometry.AstrometryNet") as mock_cls:
            instance = MagicMock()
            instance.solve_from_image.return_value = wcs_header
            mock_cls.return_value = instance
            result = swa.upload_and_solve(probe_fits, "fake_key")
        assert result is not None
        assert getattr(result, "has_celestial", False)
        instance.solve_from_image.assert_called_once()
        assert instance.solve_from_image.call_args.kwargs["force_image_upload"] is True

    def test_returns_none_on_empty_solution(self, probe_fits):
        with patch("prose.scripts.solve_wcs_astrometry.AstrometryNet") as mock_cls:
            instance = MagicMock()
            instance.solve_from_image.return_value = {}
            mock_cls.return_value = instance
            result = swa.upload_and_solve(probe_fits, "fake_key")
        assert result is None

    def test_returns_none_on_exception(self, probe_fits):
        with patch("prose.scripts.solve_wcs_astrometry.AstrometryNet") as mock_cls:
            instance = MagicMock()
            instance.solve_from_image.side_effect = RuntimeError("network error")
            mock_cls.return_value = instance
            result = swa.upload_and_solve(probe_fits, "fake_key")
        assert result is None


# ---------------------------------------------------------------------------
# solve_and_apply (end-to-end with mocked network)
# ---------------------------------------------------------------------------


class TestSolveAndApply:
    def _patch_upload(self, wcs: WCS):
        """Context manager patching upload_and_solve to return wcs."""
        return patch(
            "prose.scripts.solve_wcs_astrometry.upload_and_solve",
            return_value=wcs,
        )

    def test_e2e_patches_files(self, probe_fits, science_dir, valid_wcs):
        with self._patch_upload(valid_wcs):
            result = swa.solve_and_apply(
                probe_fits,
                api_key="fake",
                out_dir=science_dir,
                pattern="MCT20*.fits",
                instrument="muscat2",
            )
        assert result is not None
        for fp in science_dir.glob("MCT20*.fits"):
            hdr = fits.getheader(str(fp))
            assert getattr(WCS(hdr), "has_celestial", False)

    def test_dry_run_does_not_patch_files(self, probe_fits, science_dir, valid_wcs):
        with self._patch_upload(valid_wcs):
            result = swa.solve_and_apply(
                probe_fits,
                api_key="fake",
                out_dir=science_dir,
                pattern="MCT20*.fits",
                instrument="muscat2",
                dry_run=True,
            )
        assert result is not None
        for fp in science_dir.glob("MCT20*.fits"):
            hdr = fits.getheader(str(fp))
            assert not getattr(WCS(hdr), "has_celestial", False)

    def test_invalid_wcs_returns_none(self, probe_fits, science_dir, invalid_wcs):
        with self._patch_upload(invalid_wcs):
            result = swa.solve_and_apply(
                probe_fits,
                api_key="fake",
                out_dir=science_dir,
                instrument="muscat2",
            )
        assert result is None

    def test_saves_wcs_fits_when_requested(self, probe_fits, tmp_path, valid_wcs):
        wcs_out = tmp_path / "solved_wcs.fits"
        with self._patch_upload(valid_wcs):
            swa.solve_and_apply(
                probe_fits,
                api_key="fake",
                wcs_output=wcs_out,
                instrument="muscat2",
            )
        assert wcs_out.is_file()
        loaded = swa.load_wcs_fits(wcs_out)
        assert loaded is not None

    def test_auto_pattern_uses_ccd_prefix(self, probe_fits, science_dir, valid_wcs):
        """Pattern should default to MCT20*.fits inferred from probe filename."""
        with self._patch_upload(valid_wcs):
            result = swa.solve_and_apply(
                probe_fits,
                api_key="fake",
                out_dir=science_dir,
                instrument="muscat2",
            )
        assert result is not None


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_parse_minimal(self, tmp_path):
        fp = tmp_path / "test.fits"
        fp.touch()
        args = swa.parse_args(["--fits", str(fp)])
        assert args.fits == fp
        assert args.out_dir is None
        assert args.dry_run is False

    def test_parse_all_options(self, tmp_path):
        fp = tmp_path / "test.fits"
        fp.touch()
        out = tmp_path / "out"
        wcs_out = tmp_path / "wcs.fits"
        args = swa.parse_args(
            [
                "--fits", str(fp),
                "--out_dir", str(out),
                "--instrument", "muscat2",
                "--wcs_output", str(wcs_out),
                "--timeout", "120",
                "--publicly_visible", "n",
                "--allow_commercial_use", "n",
                "--dry_run",
                "--verbose",
            ]
        )
        assert args.instrument == "muscat2"
        assert args.timeout == 120.0
        assert args.publicly_visible == "n"
        assert args.dry_run is True
        assert args.verbose is True

    def test_main_missing_fits_returns_1(self, tmp_path):
        rc = swa.main(["--fits", str(tmp_path / "nonexistent.fits")])
        assert rc == 1

    def test_main_solve_success_returns_0(self, probe_fits, valid_wcs):
        with (
            patch("prose.scripts.solve_wcs_astrometry._api_key", return_value="fake"),
            patch(
                "prose.scripts.solve_wcs_astrometry.upload_and_solve",
                return_value=valid_wcs,
            ),
        ):
            rc = swa.main(["--fits", str(probe_fits), "--instrument", "muscat2"])
        assert rc == 0
        assert probe_fits.with_name(f"{probe_fits.stem}_wcs.fits").is_file()

    def test_main_solve_failure_returns_1(self, probe_fits):
        with (
            patch("prose.scripts.solve_wcs_astrometry._api_key", return_value="fake"),
            patch(
                "prose.scripts.solve_wcs_astrometry.upload_and_solve",
                return_value=None,
            ),
        ):
            rc = swa.main(["--fits", str(probe_fits)])
        assert rc == 1


# ---------------------------------------------------------------------------
# Integration test — real data (muscat2/260310/TOI07475.01)
# Skipped unless --run-net or ASTROMETRY_NET_API_KEY is set
# ---------------------------------------------------------------------------


MUSCAT2_DATA_DIR = Path("/mnt_ut3/raid_ut3/data/MuSCAT2/260310")
MUSCAT2_PROBE = MUSCAT2_DATA_DIR / "MCT20_2603100239.fits"


def _has_api_key() -> bool:
    key = os.environ.get("ASTROMETRY_NET_API_KEY", "").strip()
    if key:
        return True
    try:
        swa._api_key()
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(
    not MUSCAT2_PROBE.is_file(),
    reason="Real MuSCAT2 data not available at expected path.",
)
@pytest.mark.skipif(
    os.environ.get("RUN_ASTROMETRY_NET_TESTS") != "1" or not _has_api_key(),
    reason=(
        "Live solve disabled; set RUN_ASTROMETRY_NET_TESTS=1 and "
        "ASTROMETRY_NET_API_KEY to enable it."
    ),
)
class TestIntegrationMuscat2TOI07475:
    """Live integration test: upload, solve, validate, and apply WCS.

    Requires:
    * Real data at /mnt_ut3/raid_ut3/data/MuSCAT2/260310/
    * A valid API key via the ASTROMETRY_NET_API_KEY env variable.

    Run explicitly with::

        uv run pytest tests/scripts/test_solve_wcs_astrometry.py \\
            -k TestIntegrationMuscat2TOI07475 -v
    """

    def test_solve_download_and_apply(self, tmp_path):
        """Upload TOI07475.01, save its WCS, and attach it to a reference."""
        import shutil

        api_key = os.environ.get("ASTROMETRY_NET_API_KEY") or swa._api_key()
        science_files = [
            MUSCAT2_DATA_DIR / f"MCT20_260310{number:04d}.fits"
            for number in (239, 240, 241)
        ]
        for source in science_files:
            assert source.is_file()
            shutil.copy2(source, tmp_path / source.name)
        wcs_path = tmp_path / "TOI07475.01_wcs.fits"
        result = swa.solve_and_apply(
            MUSCAT2_PROBE,
            out_dir=tmp_path,
            instrument="muscat2",
            pattern="MCT20*.fits",
            api_key=api_key,
            wcs_output=wcs_path,
            timeout=300,
        )
        assert result is not None
        assert swa.validate_wcs(result, "muscat2")
        assert wcs_path.is_file()
        for science_file in tmp_path.glob("MCT20*.fits"):
            assert WCS(fits.getheader(science_file)).has_celestial

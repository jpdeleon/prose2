"""Tests for ``prose.scripts.calibrate_muscat2``.

Uses minimal fake FITS files created on-the-fly so tests are deterministic
and need no network or real observation data.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from prose.scripts import calibrate_muscat2 as cm

CCDS = {0: "g", 1: "r", 2: "i", 3: "z_s"}
DATA_TYP = "OBJECT"  # MuSCAT2 FITS convention


def _fake_fits(path: Path, object_val: str, exptime: float = 1.0) -> None:
    """Create a minimal valid FITS file with MuSCAT2-like headers."""
    data = np.random.default_rng().poisson(1000, size=(32, 32)).astype(np.int16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["DATA-TYP"] = DATA_TYP
    hdu.header["OBJECT"] = object_val
    hdu.header["EXPTIME"] = exptime
    hdu.header["MJD-STRT"] = 60000.0
    hdu.header["NAXIS1"] = 32
    hdu.header["NAXIS2"] = 32
    hdu.writeto(path, overwrite=True)


# ---------- helper to build a fake data directory ----------


@pytest.fixture
def fake_data_dir(tmp_path):
    """Build a minimal MuSCAT2-like data dir with known file counts."""
    # Per CCD: 15 darks, 50 flats (same exposure), variable science frames
    flat_exptimes = {0: 12.5, 1: 3.2, 2: 1.1, 3: 0.9}
    science_counts = {0: 5, 1: 7, 2: 6, 3: 4}

    for ccd, _band in CCDS.items():
        prefix = f"MCT2{ccd}_250310"
        seq = 0

        # 50 flats
        for _ in range(50):
            seq += 1
            _fake_fits(
                tmp_path / f"{prefix}{seq:04d}.fits",
                "FLAT",
                exptime=flat_exptimes[ccd],
            )

        # 15 darks (same exptime as flats)
        for _ in range(15):
            seq += 1
            _fake_fits(
                tmp_path / f"{prefix}{seq:04d}.fits",
                "DARK",
                exptime=flat_exptimes[ccd],
            )

        # science frames for TOI00663.02
        for _ in range(science_counts[ccd]):
            seq += 1
            _fake_fits(
                tmp_path / f"{prefix}{seq:04d}.fits",
                "TOI00663.02",
                exptime=20.0,
            )

        # science frames for other target (not our target)
        for _ in range(3):
            seq += 1
            _fake_fits(
                tmp_path / f"{prefix}{seq:04d}.fits",
                "TOI04717.01",
                exptime=20.0,
            )

    return tmp_path


# ---------- find_files ----------


class TestFindFiles:
    def test_returns_three_dicts(self, fake_data_dir):
        darks, flats = cm.find_files(fake_data_dir)
        assert list(darks) == [0, 1, 2, 3]
        assert list(flats) == [0, 1, 2, 3]

    @pytest.mark.parametrize("ccd", [0, 1, 2, 3])
    def test_per_ccd_counts(self, fake_data_dir, ccd):
        darks, flats = cm.find_files(fake_data_dir)
        assert len(darks[ccd]) == 15, f"CCD{ccd} darks"
        assert len(flats[ccd]) == 50, f"CCD{ccd} flats"

    def test_returns_paths_to_existing_files(self, fake_data_dir):
        darks, flats = cm.find_files(fake_data_dir)
        for fp in darks[0] + flats[0]:
            assert Path(fp).is_file()

    def test_empty_directory(self, tmp_path):
        darks, flats = cm.find_files(tmp_path)
        assert all(len(darks[c]) == 0 for c in CCDS)
        assert all(len(flats[c]) == 0 for c in CCDS)

    def test_skips_non_muscat2_files(self, tmp_path):
        _fake_fits(tmp_path / "random.fits", "DARK")
        darks, flats = cm.find_files(tmp_path)
        assert all(len(darks[c]) == 0 for c in CCDS)


# ---------- find_science_files ----------


class TestFindScienceFiles:
    def test_finds_target_per_ccd(self, fake_data_dir):
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        expected = {0: 5, 1: 7, 2: 6, 3: 4}
        for ccd, band in CCDS.items():
            assert len(sciences[ccd]) == expected[ccd], f"{band}"

    def test_other_target_counts(self, fake_data_dir):
        sciences = cm.find_science_files(fake_data_dir, "TOI04717.01")
        for ccd in CCDS:
            assert len(sciences[ccd]) == 3, f"CCD{ccd}"

    def test_nonexistent_target(self, fake_data_dir):
        sciences = cm.find_science_files(fake_data_dir, "NONEXIST")
        for ccd in CCDS:
            assert len(sciences[ccd]) == 0

    def test_excludes_empty_str(self, fake_data_dir):
        sciences = cm.find_science_files(fake_data_dir, "")
        for ccd in CCDS:
            assert len(sciences[ccd]) == 0


# ---------- calibrate_band (integration) ----------


class TestCalibrateBand:
    @pytest.mark.parametrize("ccd,band", [(0, "g"), (1, "r"), (2, "i"), (3, "z_s")])
    def test_calibrates_all_science_frames(self, fake_data_dir, tmp_path, ccd, band):
        darks, flats = cm.find_files(fake_data_dir)
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        cm.calibrate_band(darks[ccd], flats[ccd], sciences[ccd], tmp_path, band)
        out_files = sorted(tmp_path.glob("*_calibrated.fits"))
        assert len(out_files) == len(sciences[ccd])

    @pytest.mark.parametrize("ccd,band", [(0, "g"), (1, "r"), (2, "i"), (3, "z_s")])
    def test_output_valid_fits(self, fake_data_dir, tmp_path, ccd, band):
        darks, flats = cm.find_files(fake_data_dir)
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        cm.calibrate_band(darks[ccd], flats[ccd], sciences[ccd], tmp_path, band)
        out_files = sorted(tmp_path.glob("*_calibrated.fits"))
        for fp in out_files:
            hdr = fits.getheader(fp)
            data = fits.getdata(fp)
            assert data.dtype.kind == "f" and data.dtype.itemsize == 4
            assert data.shape == (32, 32)
            assert np.all(np.isfinite(data))
            assert hdr.get("CALSTAGE") == "calibrated"

    def test_skips_band_without_darks(self, tmp_path):
        cm.calibrate_band([], ["fake.fits"], [], tmp_path, "g")
        assert len(list(tmp_path.glob("*_calibrated.fits"))) == 0

    def test_skips_band_without_sciences(self, fake_data_dir, tmp_path):
        darks = ["fake.fits"]
        flats = ["fake.fits"]
        cm.calibrate_band(darks, flats, [], tmp_path, "g")
        assert len(list(tmp_path.glob("*_calibrated.fits"))) == 0

    def test_output_files_have_calstage(self, fake_data_dir, tmp_path):
        darks, flats = cm.find_files(fake_data_dir)
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        cm.calibrate_band(darks[0], flats[0], sciences[0], tmp_path, "g")
        hdr = fits.getheader(sorted(tmp_path.glob("*_calibrated.fits"))[0])
        assert hdr["CALSTAGE"] == "calibrated"

    def test_solve_wcs_graceful_failure(self, fake_data_dir, tmp_path):
        """solve_wcs=True on fake data (no real stars) must not crash."""
        darks, flats = cm.find_files(fake_data_dir)
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        cm.calibrate_band(
            darks[0], flats[0], sciences[0], tmp_path, "g", solve_wcs=True
        )
        out_files = sorted(tmp_path.glob("*_calibrated.fits"))
        assert len(out_files) == len(sciences[0])


# ---------- CLI argument parsing ----------


class TestCLI:
    def test_parse_args_minimal(self):
        args = cm.parse_args(["--data_dir", "/data", "--output_dir", "/out"])
        assert args.data_dir == Path("/data")
        assert args.output_dir == Path("/out")
        assert args.target is None
        assert args.verbose is False

    def test_parse_args_full(self):
        args = cm.parse_args(
            [
                "--data_dir",
                "/data/MuSCAT2/250310",
                "--target",
                "TOI00663.02",
                "--output_dir",
                "/tmp/out",
                "--verbose",
            ]
        )
        assert args.target == "TOI00663.02"
        assert args.verbose is True

    def test_parse_args_requires_data_dir(self):
        with pytest.raises(SystemExit):
            cm.parse_args(["--output_dir", "/out"])

    def test_solve_wcs_flag(self):
        args = cm.parse_args(["--data_dir", "/d", "--output_dir", "/o", "--solve-wcs"])
        assert args.solve_wcs == "twirl"

    def test_solve_wcs_defaults_to_none(self):
        args = cm.parse_args(["--data_dir", "/d", "--output_dir", "/o"])
        assert args.solve_wcs is None

    def test_solve_wcs_nova(self):
        args = cm.parse_args(
            ["--data_dir", "/d", "--output_dir", "/o", "--solve-wcs", "nova"]
        )
        assert args.solve_wcs == "nova"

    def test_solve_wcs_main_flag(self, fake_data_dir, tmp_path):
        """End-to-end with --solve-wcs: WCS may fail on fake data but must not crash."""
        out_dir = tmp_path / "out"
        rc = cm.main(
            [
                "--data_dir",
                str(fake_data_dir),
                "--target",
                "TOI00663.02",
                "--output_dir",
                str(out_dir),
                "--solve-wcs",
            ]
        )
        assert rc == 0
        calibrated = sorted(out_dir.glob("*_calibrated.fits"))
        assert len(calibrated) > 0
        assert (out_dir / "master_dark.png").is_file()
        assert (out_dir / "master_flat.png").is_file()


# ---------- exposure-matched dark selection ----------


class TestExposureMatching:
    def test_read_exposures_from_headers(self, tmp_path):
        a = tmp_path / "MCT20_0001.fits"
        _fake_fits(a, "DARK", exptime=12.5)
        assert cm.read_exposures([str(a)])[str(a)] == pytest.approx(12.5)

    def test_group_by_exposure(self):
        groups = cm.group_by_exposure(["a", "b"], {"a": 12.5, "b": 20.0})
        assert groups[12.5] == ["a"] and groups[20.0] == ["b"]

    def test_selects_matching_subset(self, tmp_path):
        m = tmp_path / "MCT20_0001.fits"
        x = tmp_path / "MCT20_0002.fits"
        _fake_fits(m, "DARK", exptime=20.0)
        _fake_fits(x, "DARK", exptime=12.5)
        darks, status = cm.select_darks_for_exposure([str(m), str(x)], 20.0)
        assert status == "matched" and darks == [str(m)]

    def test_no_match_falls_back_with_warning(self, tmp_path, caplog):
        d = tmp_path / "MCT20_0001.fits"
        _fake_fits(d, "DARK", exptime=12.5)
        with caplog.at_level("WARNING", logger="calibrate_muscat2"):
            darks, status = cm.select_darks_for_exposure([str(d)], 20.0)
        assert status == "no-match" and darks == [str(d)]
        assert any("no darks match" in r.message.lower() for r in caplog.records)

    def test_fixture_is_no_match_but_still_calibrates(self, fake_data_dir, tmp_path):
        """Fixture darks (12.5s) differ from science (20s): no-match fallback runs."""
        darks, flats = cm.find_files(fake_data_dir)
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        _, status = cm.select_darks_for_exposure(darks[0], 20.0, "g")
        assert status == "no-match"
        cm.calibrate_band(darks[0], flats[0], sciences[0], tmp_path, "g")
        assert len(list(tmp_path.glob("*_calibrated.fits"))) == len(sciences[0])


# ---------- main ----------


class TestMain:
    def test_missing_data_dir_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rc = cm.main(
            [
                "--data_dir",
                str(tmp_path / "nonexistent"),
                "--output_dir",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 1

    def test_end_to_end(self, fake_data_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "out"
        rc = cm.main(
            [
                "--data_dir",
                str(fake_data_dir),
                "--target",
                "TOI00663.02",
                "--output_dir",
                str(out_dir),
                "--verbose",
            ]
        )
        assert rc == 0
        calibrated = sorted(out_dir.glob("*_calibrated.fits"))
        assert len(calibrated) > 0
        assert (out_dir / "master_dark.png").is_file()
        assert (out_dir / "master_flat.png").is_file()

    def test_bands_limits_calibrated_ccds(self, tmp_path, monkeypatch):
        (tmp_path / "MCT20_0001.fits").touch()
        calls = []
        empty_frames = {ccd: [] for ccd in cm.CCD_BANDS}

        monkeypatch.setattr(
            cm,
            "find_frames",
            lambda data_dir, target: (empty_frames, empty_frames, empty_frames),
        )
        monkeypatch.setattr(cm, "save_master_plots", lambda frames, output_dir: None)

        def fake_calibrate_band(darks, flats, sciences, output_dir, band, **kwargs):
            calls.append(band)
            return np.zeros((1, 1)), np.ones((1, 1))

        monkeypatch.setattr(cm, "calibrate_band", fake_calibrate_band)

        rc = cm.main(
            [
                "--data_dir",
                str(tmp_path),
                "--target",
                "TOI00663.02",
                "--output_dir",
                str(tmp_path / "out"),
                "--bands",
                "zs",
            ]
        )

        assert rc == 0
        assert calls == ["zs"]

    def test_main_without_target(self, fake_data_dir, tmp_path):
        out_dir = tmp_path / "out"
        rc = cm.main(
            [
                "--data_dir",
                str(fake_data_dir),
                "--output_dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        assert len(list(out_dir.glob("*.fits"))) == 0
        # master plots are only written when master frames actually exist
        assert not (out_dir / "master_dark.png").is_file()
        assert not (out_dir / "master_flat.png").is_file()

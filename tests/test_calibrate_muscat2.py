"""Tests for ``prose.scripts.calibrate_muscat2``.

Uses minimal fake FITS files created on-the-fly so tests are deterministic
and need no network or real observation data.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from prose.scripts import calibrate_muscat as cm1
from prose.scripts import calibrate_muscat2 as cm
from prose.scripts import solve_wcs_astrometry as swa

CCD_FILTERS = {0: "g", 1: "r", 2: "i", 3: "z_s"}
CCD_BANDS = {0: "gp", 1: "rp", 2: "ip", 3: "zs"}
DATA_TYP = "OBJECT"  # MuSCAT2 FITS convention


def _fake_fits(
    path: Path,
    object_val: str,
    exptime: float = 1.0,
    filter_value: str | None = "g",
) -> None:
    """Create a minimal valid FITS file with MuSCAT2-like headers."""
    data = np.random.default_rng().poisson(1000, size=(32, 32)).astype(np.int16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["DATA-TYP"] = DATA_TYP
    hdu.header["OBJECT"] = object_val
    if filter_value is not None:
        hdu.header["FILTER"] = filter_value
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

    for ccd, raw_filter in CCD_FILTERS.items():
        prefix = f"MCT2{ccd}_250310"
        seq = 0

        # 50 flats
        for _ in range(50):
            seq += 1
            _fake_fits(
                tmp_path / f"{prefix}{seq:04d}.fits",
                "FLAT",
                exptime=flat_exptimes[ccd],
                filter_value=raw_filter,
            )

        # 15 darks (same exptime as flats)
        for _ in range(15):
            seq += 1
            _fake_fits(
                tmp_path / f"{prefix}{seq:04d}.fits",
                "DARK",
                exptime=flat_exptimes[ccd],
                filter_value=raw_filter,
            )

        # science frames for TOI00663.02
        for _ in range(science_counts[ccd]):
            seq += 1
            _fake_fits(
                tmp_path / f"{prefix}{seq:04d}.fits",
                "TOI00663.02",
                exptime=20.0,
                filter_value=raw_filter,
            )

        # science frames for other target (not our target)
        for _ in range(3):
            seq += 1
            _fake_fits(
                tmp_path / f"{prefix}{seq:04d}.fits",
                "TOI04717.01",
                exptime=20.0,
                filter_value=raw_filter,
            )

    return tmp_path


# ---------- find_files ----------


class TestFindFiles:
    def test_returns_three_dicts(self, fake_data_dir):
        darks, flats = cm.find_files(fake_data_dir)
        assert list(darks) == ["gp", "rp", "ip", "zs"]
        assert list(flats) == ["gp", "rp", "ip", "zs"]

    @pytest.mark.parametrize("band", ["gp", "rp", "ip", "zs"])
    def test_per_band_counts(self, fake_data_dir, band):
        darks, flats = cm.find_files(fake_data_dir)
        assert len(darks[band]) == 15, f"{band} darks"
        assert len(flats[band]) == 50, f"{band} flats"

    def test_returns_paths_to_existing_files(self, fake_data_dir):
        darks, flats = cm.find_files(fake_data_dir)
        for fp in darks["gp"] + flats["gp"]:
            assert Path(fp).is_file()

    def test_empty_directory(self, tmp_path):
        darks, flats = cm.find_files(tmp_path)
        assert all(len(darks[b]) == 0 for b in cm.BAND_ORDER)
        assert all(len(flats[b]) == 0 for b in cm.BAND_ORDER)

    def test_skips_non_muscat2_files(self, tmp_path):
        _fake_fits(tmp_path / "random.fits", "DARK")
        darks, flats = cm.find_files(tmp_path)
        assert all(len(darks[b]) == 0 for b in cm.BAND_ORDER)

    def test_filter_header_overrides_ccd_index_when_layout_differs(self, tmp_path):
        # Regression: classification must follow FILTER, not the filename CCD index.
        # This catches nights where CCD1/CCD3 carry a non-standard r/z_s layout.
        z_flat = tmp_path / "MCT21_0001.fits"
        z_dark = tmp_path / "MCT21_0002.fits"
        z_science = tmp_path / "MCT21_0003.fits"
        r_flat = tmp_path / "MCT23_0001.fits"
        r_dark = tmp_path / "MCT23_0002.fits"
        r_science = tmp_path / "MCT23_0003.fits"
        for path, kind in ((z_flat, "FLAT"), (z_dark, "DARK"), (z_science, "TOI")):
            _fake_fits(path, kind, filter_value="z_s")
        for path, kind in ((r_flat, "FLAT"), (r_dark, "DARK"), (r_science, "TOI")):
            _fake_fits(path, kind, filter_value="r")

        darks, flats, sciences = cm.find_frames(tmp_path, "TOI")

        assert flats["zs"] == [str(z_flat)]
        assert darks["zs"] == [str(z_dark)]
        assert sciences["zs"] == [str(z_science)]
        assert flats["rp"] == [str(r_flat)]
        assert darks["rp"] == [str(r_dark)]
        assert sciences["rp"] == [str(r_science)]


# ---------- find_science_files ----------


class TestFindScienceFiles:
    def test_finds_target_per_ccd(self, fake_data_dir):
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        expected = {"gp": 5, "rp": 7, "ip": 6, "zs": 4}
        for band, count in expected.items():
            assert len(sciences[band]) == count, f"{band}"

    def test_other_target_counts(self, fake_data_dir):
        sciences = cm.find_science_files(fake_data_dir, "TOI04717.01")
        for band in cm.BAND_ORDER:
            assert len(sciences[band]) == 3, f"{band}"

    def test_nonexistent_target(self, fake_data_dir):
        sciences = cm.find_science_files(fake_data_dir, "NONEXIST")
        for band in cm.BAND_ORDER:
            assert len(sciences[band]) == 0

    def test_excludes_empty_str(self, fake_data_dir):
        sciences = cm.find_science_files(fake_data_dir, "")
        for band in cm.BAND_ORDER:
            assert len(sciences[band]) == 0


# ---------- calibrate_band (integration) ----------


class TestCalibrateBand:
    @pytest.mark.parametrize("band", ["gp", "rp", "ip", "zs"])
    def test_calibrates_all_science_frames(self, fake_data_dir, tmp_path, band):
        darks, flats = cm.find_files(fake_data_dir)
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        cm.calibrate_band(darks[band], flats[band], sciences[band], tmp_path, band)
        out_files = sorted(tmp_path.glob("*_calibrated.fits"))
        assert len(out_files) == len(sciences[band])

    @pytest.mark.parametrize("band", ["gp", "rp", "ip", "zs"])
    def test_output_valid_fits(self, fake_data_dir, tmp_path, band):
        darks, flats = cm.find_files(fake_data_dir)
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        cm.calibrate_band(darks[band], flats[band], sciences[band], tmp_path, band)
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
        cm.calibrate_band(darks["gp"], flats["gp"], sciences["gp"], tmp_path, "gp")
        hdr = fits.getheader(sorted(tmp_path.glob("*_calibrated.fits"))[0])
        assert hdr["CALSTAGE"] == "calibrated"

    def test_solve_wcs_graceful_failure(self, fake_data_dir, tmp_path):
        """solve_wcs=True on fake data (no real stars) must not crash."""
        darks, flats = cm.find_files(fake_data_dir)
        sciences = cm.find_science_files(fake_data_dir, "TOI00663.02")
        cm.calibrate_band(
            darks["gp"], flats["gp"], sciences["gp"], tmp_path, "gp", solve_wcs=True
        )
        out_files = sorted(tmp_path.glob("*_calibrated.fits"))
        assert len(out_files) == len(sciences["gp"])

    @pytest.mark.parametrize("module,instrument", [(cm1, "muscat"), (cm, "muscat2")])
    def test_astrometry_net_only_updates_current_band(
        self, module, instrument, tmp_path, monkeypatch
    ):
        dark = tmp_path / f"{instrument}_dark.fits"
        flat = tmp_path / f"{instrument}_flat.fits"
        science = tmp_path / f"{instrument}_science.fits"
        for path, kind in ((dark, "DARK"), (flat, "FLAT"), (science, "OBJECT")):
            _fake_fits(path, kind)

        output_dir = tmp_path / instrument
        output_dir.mkdir()
        foreign = output_dir / "other_band_calibrated.fits"
        _fake_fits(foreign, "OBJECT")

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        uploaded = []
        injected = []
        monkeypatch.setattr(swa, "_api_key", lambda: "key")
        monkeypatch.setattr(
            swa, "upload_and_solve", lambda path, key: uploaded.append(path) or wcs
        )
        monkeypatch.setattr(swa, "validate_wcs", lambda value, name: True)
        monkeypatch.setattr(
            swa,
            "inject_wcs_into_file",
            lambda path, value: injected.append(path) or True,
        )

        module.calibrate_band(
            [str(dark)],
            [str(flat)],
            [str(science)],
            output_dir,
            "gp",
            solve_wcs="astrometry.net",
        )

        expected = output_dir / f"{science.stem}_calibrated.fits"
        assert uploaded == [expected]
        assert injected == [expected]
        assert foreign not in injected


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

    def test_solve_wcs_astrometry_net(self):
        args = cm.parse_args(
            ["--data_dir", "/d", "--output_dir", "/o", "--solve-wcs", "astrometry.net"]
        )
        assert args.solve_wcs == "astrometry.net"

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
        _, status = cm.select_darks_for_exposure(darks["gp"], 20.0, "gp")
        assert status == "no-match"
        cm.calibrate_band(darks["gp"], flats["gp"], sciences["gp"], tmp_path, "gp")
        assert len(list(tmp_path.glob("*_calibrated.fits"))) == len(sciences["gp"])


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
        empty_frames = {band: [] for band in cm.BAND_ORDER}

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

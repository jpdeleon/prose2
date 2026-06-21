"""Tests for ``prose.scripts.calibrate_muscat`` exposure-time handling.

Focused on the exposure-matched dark selection added so the master dark is not
built from a mix of exposures (``blocks.Calibration`` rescales the master dark by
exposure assuming a zero-bias pedestal, which is only exact for exposure-matched
darks when no master bias is supplied).

Uses minimal fake FITS files / obslog CSVs so tests are deterministic and need no
network or real observation data.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import prose.blocks.utils as block_utils
from prose import utils
from prose.scripts import calibrate_muscat as cm

CCD_BANDS = {0: "gp", 1: "rp", 2: "zs"}


def _fake_fits(path: Path, object_val: str, exptime: float = 1.0) -> None:
    """Create a minimal valid FITS file with MuSCAT-like headers."""
    data = np.random.default_rng().poisson(1000, size=(16, 16)).astype(np.int16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["OBJECT"] = object_val
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(path, overwrite=True)


# ---------- read_exposures ----------


class TestReadExposures:
    def test_empty_paths(self):
        assert cm.read_exposures([]) == {}

    def test_reads_exptime_from_headers(self, tmp_path):
        a = tmp_path / "MSCT0_0001.fits"
        b = tmp_path / "MSCT0_0002.fits"
        _fake_fits(a, "DARK", exptime=20.0)
        _fake_fits(b, "DARK", exptime=5.0)
        exposures = cm.read_exposures([str(a), str(b)])
        assert exposures[str(a)] == pytest.approx(20.0)
        assert exposures[str(b)] == pytest.approx(5.0)

    def test_missing_exptime_is_none(self, tmp_path):
        p = tmp_path / "MSCT0_0001.fits"
        hdu = fits.PrimaryHDU(np.zeros((4, 4), dtype=np.int16))
        hdu.header["OBJECT"] = "DARK"
        hdu.writeto(p, overwrite=True)
        assert cm.read_exposures([str(p)])[str(p)] is None

    def test_prefers_obslog_over_headers(self, tmp_path, monkeypatch):
        """Obslog exposures win and avoid FITS reads."""
        data_dir = tmp_path / "muscat" / "220514"
        data_dir.mkdir(parents=True)
        # FITS header says 99.0 but obslog says 20.0 -> obslog wins.
        fp = data_dir / "MSCT0_0001.fits"
        _fake_fits(fp, "DARK", exptime=99.0)

        obslog_root = tmp_path / "obslog"
        obslog_dir = obslog_root / "muscat" / "220514"
        obslog_dir.mkdir(parents=True)
        (obslog_dir / "obslog-muscat-220514-ccd0.csv").write_text(
            "FRAME,OBJECT,EXPTIME (s),FILTER\nMSCT0_0001,DARK,20.0,gp\n"
        )
        monkeypatch.setattr(utils, "OBSLOG_ROOT", str(obslog_root))

        assert cm.read_exposures([str(fp)])[str(fp)] == pytest.approx(20.0)


# ---------- group_by_exposure ----------


class TestGroupByExposure:
    def test_groups_by_value(self):
        paths = ["a", "b", "c"]
        exposures = {"a": 20.0, "b": 20.0, "c": 5.0}
        groups = cm.group_by_exposure(paths, exposures)
        assert sorted(groups[20.0]) == ["a", "b"]
        assert groups[5.0] == ["c"]

    def test_unknown_exposure_buckets_under_none(self):
        groups = cm.group_by_exposure(["a"], {"a": None})
        assert groups[None] == ["a"]


# ---------- select_darks_for_exposure ----------


class TestSelectDarksForExposure:
    def test_no_darks(self):
        darks, status = cm.select_darks_for_exposure([], 20.0)
        assert darks == [] and status == "no-darks"

    def test_science_exposure_unknown_returns_all(self, tmp_path):
        a = tmp_path / "MSCT0_0001.fits"
        _fake_fits(a, "DARK", exptime=20.0)
        darks, status = cm.select_darks_for_exposure([str(a)], None)
        assert darks == [str(a)] and status == "science-exposure-unknown"

    def test_selects_matching_subset(self, tmp_path):
        match1 = tmp_path / "MSCT0_0001.fits"
        match2 = tmp_path / "MSCT0_0002.fits"
        other = tmp_path / "MSCT0_0003.fits"
        _fake_fits(match1, "DARK", exptime=20.0)
        _fake_fits(match2, "DARK", exptime=20.0)
        _fake_fits(other, "DARK", exptime=5.0)
        darks, status = cm.select_darks_for_exposure(
            [str(match1), str(match2), str(other)], 20.0
        )
        assert status == "matched"
        assert sorted(darks) == sorted([str(match1), str(match2)])

    def test_no_match_falls_back_to_all_with_warning(self, tmp_path, caplog):
        a = tmp_path / "MSCT0_0001.fits"
        b = tmp_path / "MSCT0_0002.fits"
        _fake_fits(a, "DARK", exptime=5.0)
        _fake_fits(b, "DARK", exptime=3.0)
        with caplog.at_level("WARNING", logger="calibrate_muscat"):
            darks, status = cm.select_darks_for_exposure([str(a), str(b)], 20.0)
        assert status == "no-match"
        assert sorted(darks) == sorted([str(a), str(b)])
        assert any("no darks match" in r.message.lower() for r in caplog.records)

    def test_tolerance_matches_near_equal(self, tmp_path):
        a = tmp_path / "MSCT0_0001.fits"
        _fake_fits(a, "DARK", exptime=20.0005)
        darks, status = cm.select_darks_for_exposure([str(a)], 20.0)
        assert status == "matched" and darks == [str(a)]


# ---------- calibrate_band integration ----------


class TestCalibrateBandExposure:
    def test_uses_only_matching_darks(self, tmp_path, monkeypatch):
        """Mixed-exposure darks: only science-matched darks reach Calibration."""
        captured = {}
        real_select = cm.select_darks_for_exposure

        def spy(darks, science_exposure, band="?"):
            result = real_select(darks, science_exposure, band)
            captured["selected"] = result[0]
            captured["status"] = result[1]
            return result

        monkeypatch.setattr(cm, "select_darks_for_exposure", spy)

        darks, flats, sciences = [], [], []
        for i in range(3):
            d = tmp_path / f"MSCT0_d{i:02d}.fits"
            _fake_fits(d, "DARK", exptime=20.0)
            darks.append(str(d))
        for i in range(2):  # mismatched darks that must be dropped
            d = tmp_path / f"MSCT0_x{i:02d}.fits"
            _fake_fits(d, "DARK", exptime=5.0)
            darks.append(str(d))
        for i in range(4):
            f = tmp_path / f"MSCT0_f{i:02d}.fits"
            _fake_fits(f, "FLAT", exptime=20.0)
            flats.append(str(f))
        for i in range(3):
            s = tmp_path / f"MSCT0_s{i:02d}.fits"
            _fake_fits(s, "TOI126", exptime=20.0)
            sciences.append(str(s))

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        cm.calibrate_band(darks, flats, sciences, out_dir, "gp")

        assert captured["status"] == "matched"
        assert len(captured["selected"]) == 3
        assert len(list(out_dir.glob("*_calibrated.fits"))) == len(sciences)


# ---------- main ----------


class TestMain:
    def test_multiband_calibration_isolates_and_cleans_shared_memmaps(
        self, tmp_path, monkeypatch
    ):
        """Real multi-band calibration must not leave process-global memmaps."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "calibrated"
        work_dir = tmp_path / "work"
        raw_dir.mkdir()
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)
        shared_dirs = []
        real_mkdtemp = block_utils.tempfile.mkdtemp

        def tracked_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            shared_dirs.append(Path(path))
            return path

        monkeypatch.setattr(block_utils.tempfile, "mkdtemp", tracked_mkdtemp)

        darks = {}
        flats = {}
        sciences = {}
        for ccd, band in ((1, "rp"), (2, "zs")):
            darks[band] = []
            flats[band] = []
            sciences[band] = []
            for image_type, destination in (
                ("DARK", darks[band]),
                ("FLAT", flats[band]),
                ("EPIC29937", sciences[band]),
            ):
                for i in range(2):
                    path = raw_dir / f"MSCT{ccd}_{image_type}_{i}.fits"
                    _fake_fits(path, image_type, exptime=30.0)
                    destination.append(str(path))

        monkeypatch.setattr(
            cm,
            "find_frames",
            lambda data_dir, target: (darks, flats, sciences),
        )
        monkeypatch.setattr(cm, "save_master_plots", lambda frames, output_dir: None)

        assert (
            cm.main(
                [
                    "--data_dir",
                    str(raw_dir),
                    "--target",
                    "EPIC29937",
                    "--output_dir",
                    str(output_dir),
                    "--bands",
                    "rp",
                    "zs",
                ]
            )
            == 0
        )
        assert len(list(output_dir.glob("*_calibrated.fits"))) == 4
        assert len(shared_dirs) == 2
        assert len(set(shared_dirs)) == 2
        assert all(not path.exists() for path in shared_dirs)
        assert list(work_dir.glob("__*.array")) == []

    def test_bands_limits_calibrated_ccds(self, tmp_path, monkeypatch):
        (tmp_path / "MSCT0_0001.fits").touch()
        calls = []
        empty_frames = {band: [] for band in cm.CCD_BANDS.values()}

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
                "TOI126",
                "--output_dir",
                str(tmp_path / "out"),
                "--bands",
                "zs",
            ]
        )

        assert rc == 0
        assert calls == ["zs"]

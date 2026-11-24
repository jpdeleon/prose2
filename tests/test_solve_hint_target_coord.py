"""The plate-solve hint uses --target_coord instead of a wrong header pointing (#99).

No network: ``upload_and_solve`` is faked.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import prose.scripts.calibrate_muscat as cm1
import prose.scripts.calibrate_muscat2 as cm2
import prose.scripts.solve_wcs_astrometry as swa
from prose.scripts import run_photometry as rp

# TOI-6109.02 on MuSCAT2 240912: header says Dec +28:17 (Teide's latitude),
# the target is at Dec +42.6, about 14.5 deg away.
WRONG_HEADER = ("3:30:21", "+28:17:12")
TARGET = (50.131921, 42.606956)


def _fits(path: Path, ra: str | None = None, dec: str | None = None) -> Path:
    hdu = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
    hdu.header["INSTRUME"] = "MuSCAT2"
    if ra is not None:
        hdu.header["RA"] = ra
        hdu.header["DEC"] = dec
    hdu.writeto(path, overwrite=True)
    return path


def _wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


class TestSolveHints:
    def test_target_coord_replaces_wrong_header_pointing(self, tmp_path, caplog):
        fp = _fits(tmp_path / "a.fits", *WRONG_HEADER)
        with caplog.at_level(logging.WARNING, logger="solve_wcs_astrometry"):
            hints, source = swa.solve_hints(fp, TARGET)
        assert source == "target_coord"
        assert hints["center_ra"] == pytest.approx(TARGET[0])
        assert hints["center_dec"] == pytest.approx(TARGET[1])
        assert hints["radius"] == swa.HINT_RADIUS_DEG
        assert hints["scale_est"] == pytest.approx(0.44)
        assert "14.5 deg" in caplog.text

    def test_target_coord_close_to_header_logs_no_warning(self, tmp_path, caplog):
        fp = _fits(tmp_path / "a.fits", "3:20:31.66", "+42:36:25")
        with caplog.at_level(logging.WARNING, logger="solve_wcs_astrometry"):
            hints, source = swa.solve_hints(fp, TARGET)
        assert source == "target_coord"
        assert caplog.text == ""

    def test_target_coord_used_when_header_has_no_pointing(self, tmp_path):
        hints, source = swa.solve_hints(_fits(tmp_path / "a.fits"), TARGET)
        assert source == "target_coord"
        assert hints["center_dec"] == pytest.approx(TARGET[1])

    def test_negative_decimal_header_matches_target(self, tmp_path, caplog):
        fp = _fits(tmp_path / "south.fits", "50", "-42.5")
        header, _ = swa.solve_hints(fp)
        assert header["center_dec"] == -42.5
        with caplog.at_level(logging.WARNING, logger="solve_wcs_astrometry"):
            swa.solve_hints(fp, (50.0, -42.5))
        assert caplog.text == ""

    def test_header_pointing_without_target_coord(self, tmp_path):
        hints, source = swa.solve_hints(_fits(tmp_path / "a.fits", *WRONG_HEADER))
        assert source == "header"
        assert hints["center_dec"] == pytest.approx(28.2867, abs=1e-3)

    def test_no_position_hint_at_all(self, tmp_path):
        hints, source = swa.solve_hints(_fits(tmp_path / "a.fits"))
        assert source is None
        assert "center_ra" not in hints and "scale_est" in hints


class TestWithoutPosition:
    def test_keeps_scale_hints_only(self):
        hints = {
            "center_ra": 1.0,
            "center_dec": 2.0,
            "radius": 0.5,
            "scale_est": 0.44,
            "scale_err": 20.0,
            "scale_units": "arcsecperpix",
        }
        stripped = swa.without_position(hints)
        assert stripped == {
            "scale_est": 0.44,
            "scale_err": 20.0,
            "scale_units": "arcsecperpix",
        }
        assert "center_ra" in hints  # input is not mutated


class TestSolveFirstValidHints:
    def _patch(self, monkeypatch, results):
        calls = []

        def fake_upload(path, key, *, hints=None):
            calls.append(hints)
            return results[len(calls) - 1]

        monkeypatch.setattr(swa, "upload_and_solve", fake_upload)
        monkeypatch.setattr(swa, "validate_wcs", lambda wcs, inst: True)
        return calls

    def _files(self, tmp_path, n=3):
        return [_fits(tmp_path / f"f{i}.fits", *WRONG_HEADER) for i in range(n)]

    def test_every_attempt_uses_target_coord(self, tmp_path, monkeypatch):
        calls = self._patch(monkeypatch, [None, None, None])
        assert (
            swa.solve_first_valid(self._files(tmp_path), "k", "muscat2", TARGET) is None
        )
        assert len(calls) == 3
        assert all(h["center_dec"] == pytest.approx(TARGET[1]) for h in calls)

    def test_failed_header_hint_is_dropped_for_later_frames(
        self, tmp_path, monkeypatch
    ):
        wcs = _wcs()
        calls = self._patch(monkeypatch, [None, wcs])
        assert swa.solve_first_valid(self._files(tmp_path), "k", "muscat2") is wcs
        assert "center_ra" in calls[0]
        assert "center_ra" not in calls[1] and "scale_est" in calls[1]


class TestCalibrationForwardsTarget:
    @pytest.mark.parametrize("module", [cm1, cm2])
    def test_parse_args_accepts_target_coord(self, module, tmp_path):
        args = module.parse_args(
            [
                "--data_dir",
                str(tmp_path),
                "--target",
                "T",
                "--output_dir",
                str(tmp_path),
                "--target_coord",
                "50.131921",
                "42.606956",
            ]
        )
        assert args.target_coord == list(TARGET)

    @pytest.mark.parametrize("module", [cm1, cm2])
    def test_target_coord_defaults_to_none(self, module, tmp_path):
        args = module.parse_args(
            [
                "--data_dir",
                str(tmp_path),
                "--target",
                "T",
                "--output_dir",
                str(tmp_path),
            ]
        )
        assert args.target_coord is None

    @pytest.mark.parametrize("module", [cm1, cm2])
    def test_solve_receives_target(self, module, tmp_path, monkeypatch):
        files = [_fits(tmp_path / "MCT20_0_calibrated.fits", *WRONG_HEADER)]
        seen = []
        monkeypatch.setattr(swa, "_api_key", lambda: "key")
        monkeypatch.setattr(
            swa,
            "solve_first_valid",
            lambda f, key, inst, target_radec=None: seen.append(target_radec),
        )
        module._solve_astrometry_net("gp", tmp_path, files, target_radec=TARGET)
        assert seen == [TARGET]


class TestRunPhotometryForwardsTarget:
    def _args(self, target_coord):
        return Namespace(
            data_dir=Path("/d"),
            target_name="TOI06109.02",
            wcs_method="astrometry.net",
            test_run=False,
            test_run_frames=10,
            refid=None,
            verbose=False,
            target_coord=target_coord,
        )

    def _forwarded(self, calib_args):
        i = calib_args.index("--target_coord")
        return float(calib_args[i + 1]), float(calib_args[i + 2])

    def test_decimal_degrees(self):
        calib_args = rp._calibration_args(
            self._args(["50.131921", "42.606956"]), Path("/c"), None
        )
        assert self._forwarded(calib_args) == pytest.approx(TARGET)

    def test_sexagesimal_converted_to_degrees(self):
        calib_args = rp._calibration_args(
            self._args(["3:20:31.66", "+42:36:25.0"]), Path("/c"), None
        )
        assert self._forwarded(calib_args) == pytest.approx(TARGET, abs=1e-4)

    def test_omitted_without_target_coord(self):
        calib_args = rp._calibration_args(self._args(None), Path("/c"), None)
        assert "--target_coord" not in calib_args

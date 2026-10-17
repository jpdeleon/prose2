"""Tests for astrometry.net job-id logging and retry on alternative frames.

No network: ``AstrometryNet`` and ``upload_and_solve`` are faked.
"""

import logging
from pathlib import Path

import pytest
from astropy.io import fits
from astropy.wcs import WCS

from prose.scripts import calibrate_muscat as cm1
from prose.scripts import calibrate_muscat2 as cm2
from prose.scripts import solve_wcs_astrometry as swa


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _paths(n):
    return [Path(f"frame_{i}.fits") for i in range(n)]


class TestCandidateFrames:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, []),
            (1, [0]),
            (2, [0, 1]),
            (3, [0, 1, 2]),
            (5, [0, 2, 4]),
            (10, [0, 5, 9]),
        ],
    )
    def test_first_middle_last_without_duplicates(self, n, expected):
        files = _paths(n)
        assert swa.candidate_frames(files) == [files[i] for i in expected]

    def test_limit_is_respected(self):
        assert len(swa.candidate_frames(_paths(100), n=2)) == 2


class TestSolveFirstValid:
    def _patch(self, monkeypatch, results, valid=True):
        calls = []

        def fake_upload(path, key):
            calls.append(path)
            return results[len(calls) - 1]

        monkeypatch.setattr(swa, "upload_and_solve", fake_upload)
        monkeypatch.setattr(swa, "validate_wcs", lambda wcs, inst: valid)
        return calls

    def test_retries_next_frame_after_failure(self, monkeypatch):
        files = _paths(5)
        wcs = _wcs()
        calls = self._patch(monkeypatch, [None, wcs])
        assert swa.solve_first_valid(files, "key", "muscat2") is wcs
        assert calls == [files[0], files[2]]  # stops at the first success

    def test_first_frame_success_makes_one_attempt(self, monkeypatch):
        files = _paths(5)
        calls = self._patch(monkeypatch, [_wcs()])
        assert swa.solve_first_valid(files, "key", "muscat2") is not None
        assert calls == [files[0]]

    def test_all_attempts_fail_returns_none(self, monkeypatch):
        files = _paths(10)
        calls = self._patch(monkeypatch, [None, None, None])
        assert swa.solve_first_valid(files, "key", "muscat2") is None
        assert calls == [files[0], files[5], files[9]]

    def test_invalid_wcs_counts_as_failure(self, monkeypatch):
        files = _paths(3)
        calls = self._patch(monkeypatch, [_wcs(), _wcs(), _wcs()], valid=False)
        assert swa.solve_first_valid(files, "key", "muscat2") is None
        assert len(calls) == 3

    def test_no_files_returns_none(self, monkeypatch):
        calls = self._patch(monkeypatch, [])
        assert swa.solve_first_valid([], "key", "muscat2") is None
        assert calls == []


class _FakeAN:
    outcome = None

    def __init__(self):
        self.api_key = None

    def solve_from_image(self, path, **kwargs):
        assert kwargs["return_submission_id"] is True
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class TestUploadAndSolveLogging:
    @pytest.fixture
    def fits_file(self, tmp_path):
        p = tmp_path / "f.fits"
        fits.PrimaryHDU().writeto(p)
        return p

    def _run(self, monkeypatch, fits_file, outcome, caplog):
        _FakeAN.outcome = outcome
        monkeypatch.setattr(swa, "AstrometryNet", _FakeAN)
        with caplog.at_level(logging.INFO, logger="solve_wcs_astrometry"):
            return swa.upload_and_solve(fits_file, "key")

    def test_solver_failure_logs_submission_id_and_url(
        self, monkeypatch, fits_file, caplog
    ):
        assert self._run(monkeypatch, fits_file, ({}, 4242), caplog) is None
        assert "https://nova.astrometry.net/status/4242" in caplog.text
        assert "solver reported failure" in caplog.text

    def test_timeout_logs_submission_id(self, monkeypatch, fits_file, caplog):
        err = TimeoutError("Solve timed out without success or failure", 777)
        assert self._run(monkeypatch, fits_file, err, caplog) is None
        assert "https://nova.astrometry.net/status/777" in caplog.text
        assert "timed out" in caplog.text

    def test_success_returns_wcs(self, monkeypatch, fits_file, caplog):
        header = _wcs().to_header()
        wcs = self._run(monkeypatch, fits_file, (header, 99), caplog)
        assert wcs is not None and wcs.has_celestial


@pytest.mark.parametrize("module,instrument", [(cm1, "muscat"), (cm2, "muscat2")])
def test_calibrate_scripts_fall_back_to_another_frame(
    module, instrument, tmp_path, monkeypatch
):
    files = []
    for i in range(3):
        p = tmp_path / f"MCT20_{i}_calibrated.fits"
        fits.PrimaryHDU().writeto(p)
        files.append(p)
    wcs = _wcs()
    uploaded, injected = [], []
    results = iter([None, wcs])
    monkeypatch.setattr(swa, "_api_key", lambda: "key")
    monkeypatch.setattr(
        swa,
        "upload_and_solve",
        lambda path, key: uploaded.append(path) or next(results),
    )
    monkeypatch.setattr(swa, "validate_wcs", lambda value, name: True)
    monkeypatch.setattr(
        swa, "inject_wcs_into_file", lambda path, value: injected.append(path) or True
    )

    module._solve_astrometry_net("gp", tmp_path, files)

    assert uploaded == [files[0], files[1]]
    assert injected == files
    assert (tmp_path / ".wcs" / "gp_astrometry.net.wcs.fits").is_file()

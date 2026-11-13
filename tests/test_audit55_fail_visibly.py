"""#55 A2/B3: runs that cannot locate the target, or that lose a band, must
fail visibly instead of logging success and exiting 0."""

import logging

import numpy as np
import pytest

from prose.scripts import run_photometry as rp


def _write_sinistro_frame(path):
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [10, 10]
    w.wcs.cdelt = [0.01, 0.01]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()
    for key, value in {
        "TELID": "1m0a",
        "INSTRUME": "sinistro",
        "SITEID": "lsc",
        "CONFMODE": "central_2k_2x2",
        "OBJECT": "audit",
        "EXPTIME": 1.0,
        "FILTER": "gp",
        "JD": 2460000.0,
        "DATE-OBS": "2025-04-16T00:00:00",
    }.items():
        header[key] = value
    fits.writeto(path, np.ones((20, 20)), header)


def _argv(tmp_path):
    return [
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


def test_unlocated_target_fails_run_and_names_tid(tmp_path, monkeypatch, caplog):
    _write_sinistro_frame(tmp_path / "science.fits")

    def _raise(*_a, **_kw):
        raise rp.TargetNotFoundError("target not located; set --tID")

    monkeypatch.setattr(rp, "run_band", _raise)
    with caplog.at_level(logging.INFO, logger="prose_run_photometry"):
        ret = rp.main(_argv(tmp_path))

    assert ret == 1
    messages = [r.getMessage() for r in caplog.records]
    assert any("[gp]" in m and "--tID" in m for m in messages)
    assert any("photometry FAILED" in m for m in messages)
    assert not any("photometry SUCCEEDED" in m for m in messages)
    # An expected user-actionable failure is logged without a traceback.
    target_records = [r for r in caplog.records if "--tID" in r.getMessage()]
    assert all(r.exc_info is None for r in target_records)


@pytest.mark.parametrize(
    "n_ok,failed,expected",
    [
        (2, [], 0),
        (1, ["rp"], rp.EXIT_PARTIAL_FAILURE),
    ],
)
def test_final_status_exit_code(caplog, n_ok, failed, expected):
    band_results = {f"b{i}": {} for i in range(n_ok)}
    with caplog.at_level(logging.INFO, logger="prose_run_photometry"):
        ret = rp._final_status(band_results, failed, n_bands=2, elapsed=1.0)

    assert ret == expected
    messages = " ".join(r.getMessage() for r in caplog.records)
    if failed:
        assert "photometry PARTIAL FAILURE" in messages
        assert "photometry SUCCEEDED" not in messages
    else:
        assert "photometry SUCCEEDED" in messages


def test_partial_failure_exit_code_is_distinct():
    assert rp.EXIT_PARTIAL_FAILURE not in (0, 1, 2)

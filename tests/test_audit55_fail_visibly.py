"""#55 A2/B3: runs that cannot locate the target, or that lose a band, must
fail visibly instead of logging success and exiting 0."""

import logging

import numpy as np
import pytest

from prose.scripts import run_photometry as rp


def _write_sinistro_frame(path, band="gp"):
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
        "FILTER": band,
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
        (2, {}, 0),
        (1, {"rp": "no output"}, rp.EXIT_PARTIAL_FAILURE),
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


def test_final_status_names_reason_per_band_grouped(caplog):
    failures = {
        "gp": rp.TARGET_NOT_LOCATED_REASON,
        "rp": rp.TARGET_NOT_LOCATED_REASON,
        "zs": "no output",
    }
    with caplog.at_level(logging.INFO, logger="prose_run_photometry"):
        rp._final_status({"ip": {}}, failures, n_bands=4, elapsed=1.0)

    line = next(
        r.getMessage() for r in caplog.records if "PARTIAL FAILURE" in r.getMessage()
    )
    assert f"{rp.TARGET_NOT_LOCATED_REASON}: gp, rp" in line
    assert "no output: zs" in line
    assert "--tID" in line
    # muscat-db shows this line as the job's error, clipped at 200 characters.
    assert len(line) <= 200


class _Stop(Exception):
    pass


class _Src:
    def __init__(self, xy):
        self.coords = np.asarray(xy, dtype=float)


class _Ref:
    sources = [_Src([12.0, 8.0])]


def _run_two_band_main(tmp_path, monkeypatch, run_band):
    _write_sinistro_frame(tmp_path / "g.fits", "gp")
    _write_sinistro_frame(tmp_path / "i.fits", "ip")
    monkeypatch.setattr(rp, "_reference_file_has_usable_wcs", lambda *a: True)
    monkeypatch.setattr(rp, "run_band", run_band)

    def _stop(*a, **kw):
        raise _Stop

    monkeypatch.setattr(rp, "build_summary_stem", _stop)
    argv = _argv(tmp_path)
    argv[argv.index("--bands") + 1 : argv.index("--bands") + 2] = ["gp", "ip"]
    return [*argv, "--plots", "none"]


def test_band_missing_target_is_retried_with_later_band_position(
    tmp_path, monkeypatch, caplog
):
    calls = []

    def run_band(band, *a, **kw):
        calls.append((band, kw.get("target_pixel_fallback")))
        if band == "gp" and kw.get("target_pixel_fallback") is None:
            raise rp.TargetNotFoundError("target not located; set --tID")
        return {"ref": _Ref(), "target_index": 0}

    argv = _run_two_band_main(tmp_path, monkeypatch, run_band)
    with caplog.at_level(logging.INFO, logger="prose_run_photometry"):
        with pytest.raises(_Stop):  # reached the export step: no band failed
            rp.main(argv)

    assert [band for band, _ in calls] == ["gp", "ip", "gp"]
    np.testing.assert_allclose(calls[2][1], [12.0, 8.0])
    assert any(
        "retrying after the other bands" in r.getMessage().lower()
        for r in caplog.records
    )


def test_band_missing_target_fails_when_no_band_locates_it(tmp_path, monkeypatch):
    calls = []

    def run_band(band, *a, **kw):
        calls.append(band)
        raise rp.TargetNotFoundError("target not located; set --tID")

    argv = _run_two_band_main(tmp_path, monkeypatch, run_band)
    assert rp.main(argv) == 1
    # Nothing located the target, so the deferred bands are not re-run.
    assert calls == ["gp", "ip"]


def test_no_retry_when_tid_is_given(tmp_path, monkeypatch):
    calls = []

    def run_band(band, *a, **kw):
        calls.append(band)
        if band == "gp":
            raise rp.TargetNotFoundError("target not located; set --tID")
        return {"ref": _Ref(), "target_index": 0}

    argv = _run_two_band_main(tmp_path, monkeypatch, run_band)
    with pytest.raises(_Stop):
        rp.main([*argv, "--tID", "0"])
    assert calls == ["gp", "ip"]

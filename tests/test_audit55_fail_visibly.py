"""#55 A2/B3: runs that cannot locate the target, or that lose a band, must
fail visibly instead of logging success and exiting 0."""

import logging
from pathlib import Path

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
    data = np.arange(400).reshape(20, 20)
    computed = {}
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


@pytest.mark.parametrize("header_has_wcs", [True, False], ids=["header", "gaia_refit"])
def test_band_missing_target_is_retried_with_later_band_position(
    tmp_path, monkeypatch, caplog, header_has_wcs
):
    calls = []

    def run_band(band, *a, **kw):
        calls.append((band, kw.get("target_pixel_fallback")))
        if band == "gp" and kw.get("target_pixel_fallback") is None:
            rp.find_target_index(_Ref(), None)
        kw["outcome"]["identification_method"] = (
            "wcs_match" if band == "ip" else "other_band_pixel"
        )
        return {"ref": _Ref(), "target_index": 0}

    argv = _run_two_band_main(tmp_path, monkeypatch, run_band)
    monkeypatch.setattr(rp, "_reference_file_has_usable_wcs", lambda *a: header_has_wcs)
    with caplog.at_level(logging.INFO, logger="prose_run_photometry"):
        with pytest.raises(_Stop):  # reached the export step: no band failed
            rp.main(argv)

    assert not list((tmp_path / "out").glob("*_unmatched_ref.png"))
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
        rp.find_target_index(_Ref(), None)

    argv = _run_two_band_main(tmp_path, monkeypatch, run_band)
    assert rp.main(argv) == 1
    # Nothing located the target, so the deferred bands are not re-run.
    assert calls == ["gp", "ip"]
    assert len(list((tmp_path / "out").glob("*_unmatched_ref.png"))) == 2


def test_rerun_failing_otherwise_removes_previous_unmatched_charts(
    tmp_path, monkeypatch
):
    def unmatched(band, *a, **kw):
        rp.find_target_index(_Ref(), None)

    argv = _run_two_band_main(tmp_path, monkeypatch, unmatched)
    assert rp.main(argv) == 1
    assert len(list((tmp_path / "out").glob("*_unmatched_ref.png"))) == 2

    def broken(band, *a, **kw):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(rp, "run_band", broken)
    assert rp.main(argv) == 1
    assert not list((tmp_path / "out").glob("*_unmatched_ref.png"))


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


def test_unmatched_chart_uses_anchor_ids_and_frame_number(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from prose import Image
    from prose.core.source import PointSource, Sources

    ref = Image(np.arange(400).reshape(20, 20))
    ref.sources = Sources([PointSource(coords=np.array([i + 2, 8])) for i in range(5)])
    rp.SelectSourceIndices([1, 4]).run(ref)
    labels = []
    titles = []
    annotate = Axes.annotate
    title = Axes.set_title

    def record_label(ax, text, xy, **kwargs):
        labels.append((text, tuple(xy)))
        return annotate(ax, text, xy, **kwargs)

    def record_title(ax, text, **kwargs):
        titles.append(text)
        return title(ax, text, **kwargs)

    monkeypatch.setattr(Axes, "annotate", record_label)
    monkeypatch.setattr(Axes, "set_title", record_title)
    path = tmp_path / "chart.png"
    rp.save_unmatched_reference(ref, "MCT20_1911191480.fits", path, "gray")
    assert labels == [("1", (3, 8)), ("4", (6, 8))]
    assert "--refid 1480" in titles[-1]
    assert path.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("explicit,expected", [(False, [1, 2, 3]), (True, [1])])
def test_main_retries_sparse_automatic_reference_only(
    tmp_path, monkeypatch, caplog, explicit, expected
):
    for number in (1, 2, 3):
        _write_sinistro_frame(tmp_path / f"science-20250416-{number:04}-e91.fits")
    seen = []

    def reduce_band(band, files, ref_file, *args, **kwargs):
        seen.append(rp._fits_file_number(ref_file))
        raise rp.SparseReferenceError("reference has 2 sources")

    monkeypatch.setattr(rp, "run_band", reduce_band)
    monkeypatch.setattr(rp, "resolve_target_at_epoch", lambda coord, _: coord)
    args = _argv(tmp_path) + (["--refid", "1"] if explicit else [])
    assert rp.main(args) == 1
    assert seen == expected
    assert "insufficient reference sources" in caplog.text


def test_alignment_catalog_requirement_precedes_psf_fit():
    from prose import Image
    from prose.core import Sources

    for count in (0, 1, 2, 3):
        image = Image(_sources=Sources(np.arange(count * 2).reshape((count, 2))))
        block = rp.RequireAlignmentSources()
        if count < 3:
            with pytest.raises(
                rp.SparseReferenceError, match=f"reference has {count} sources"
            ):
                block.run(image)
        else:
            block.run(image)


def test_shared_sparse_reference_replacement_is_used_for_all_bands(
    tmp_path, monkeypatch
):
    for band in ("gp", "rp"):
        for number in (1, 2, 3):
            _write_sinistro_frame(
                tmp_path / f"{band}-20250416-{number:04}-e91.fits", band
            )
    seen = []

    def reduce_band(band, files, ref_file, *args, **kwargs):
        seen.append((band, Path(ref_file).name))
        if rp._fits_file_number(ref_file) == 2:
            raise rp.SparseReferenceError("reference has 2 sources")
        return None

    monkeypatch.setattr(rp, "run_band", reduce_band)
    monkeypatch.setattr(rp, "resolve_target_at_epoch", lambda coord, _: coord)
    rp.main(_argv(tmp_path) + ["--bands", "gp", "rp", "--ref_band", "gp"])
    assert seen == [
        ("gp", "gp-20250416-0002-e91.fits"),
        ("gp", "gp-20250416-0001-e91.fits"),
        ("rp", "gp-20250416-0001-e91.fits"),
    ]


@pytest.mark.parametrize("n,code", [(4, 0), (3, rp.EXIT_PARTIAL_FAILURE), (0, 1)])
def test_requested_band_outcomes(tmp_path, monkeypatch, n, code):
    import json

    from prose import Fluxes

    bands = ["gp", "rp", "ip", "zs"]
    for band in bands[:n]:
        _write_sinistro_frame(tmp_path / f"{band}.fits", band)

    def reduce(band, files, *args, outcome, **kwargs):
        outcome.update(
            identification_method="explicit_index",
            comparison_stars=[1],
            aperture_index=0,
            aperture_radius_px=3.0,
        )
        diff = Fluxes(np.ones(3), time=np.arange(3.0), errors=np.ones(3))
        return dict(
            target_index=0, diff=diff, ref=rp.SimpleNamespace(header={}), files=files
        )

    monkeypatch.setattr(rp, "run_band", reduce)
    monkeypatch.setattr(rp, "resolve_target_at_epoch", lambda coord, _: coord)
    monkeypatch.setattr(rp, "compute_bjd_tdb", lambda diff, *a: diff.time)
    monkeypatch.setattr(rp, "save_all_bands_npz", lambda *a, **kw: None)
    argv = _argv(tmp_path)
    argv[argv.index("gp") : argv.index("gp") + 1] = bands
    assert rp.main([*argv, "--tID", "0", "--plots", "none"]) == code
    records = json.loads((tmp_path / "out" / "band_outcomes.json").read_text())["bands"]
    assert list(records) == bands
    for band in bands[:n]:
        assert records[band] == dict(
            outcome="usable",
            target_index=0,
            identification_method="explicit_index",
            comparison_stars=[1],
            aperture_index=0,
            aperture_radius_px=3.0,
            failure_reason=None,
        )
    for band in bands[n:]:
        assert records[band]["outcome"] == "failed"
        assert records[band]["failure_reason"]
        assert records[band]["target_index"] is None

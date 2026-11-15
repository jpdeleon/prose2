"""#55 A10: comparison stars are screened for saturation in every frame, not
only the reference frame, and the target's saturated fraction is reported.
#55 A11 (last item): a saturation screen that would empty the comparison
pool fails the band instead of re-admitting saturated stars."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from prose.scripts import run_photometry as rp

LIMIT = 60000.0


def _write_lco_frame(path, saturate=LIMIT):
    from astropy.io import fits

    header = fits.Header()
    for key, value in {
        "TELID": "1m0a",
        "INSTRUME": "fa01",
        "SITEID": "lsc",
        "SATURATE": saturate,
        "GAIN": 1.0,
        "DATE-OBS": "2026-04-16T00:00:00",
    }.items():
        header[key] = value
    fits.writeto(path, np.ones((10, 10)), header)
    return path


def _phot(peaks):
    return SimpleNamespace(data=[SimpleNamespace(values={"_peaks": np.asarray(peaks)})])


def _peaks(n_frames=100, n_sources=4, saturated=None):
    """Peaks well below LIMIT; ``saturated`` maps source -> number of frames
    (the latest ones) at the limit."""
    peaks = np.full((n_frames, n_sources), 20000.0)
    for source, n in (saturated or {}).items():
        peaks[n_frames - n :, source] = LIMIT + 1
    return peaks


# ------------------------------ fraction rule ------------------------------


def test_star_saturating_late_in_the_night_is_excluded():
    excluded, target_fraction = rp._saturated_over_time(
        _peaks(saturated={2: 30}), LIMIT, target_index=0
    )
    assert excluded == [2]
    assert target_fraction == 0.0


def test_single_saturated_frame_is_tolerated():
    # one frame in 20 is 5% (> 2%), but a single cosmic ray must not reject a star
    excluded, _ = rp._saturated_over_time(
        _peaks(n_frames=20, saturated={1: 1}), LIMIT, 0
    )
    assert excluded == []


def test_fraction_threshold_is_two_percent():
    assert rp._saturated_over_time(_peaks(saturated={1: 2}), LIMIT, 0)[0] == []
    assert rp._saturated_over_time(_peaks(saturated={1: 3}), LIMIT, 0)[0] == [1]


def test_target_is_never_excluded_and_its_fraction_is_reported():
    excluded, target_fraction = rp._saturated_over_time(
        _peaks(saturated={0: 25}), LIMIT, target_index=0
    )
    assert excluded == []
    assert target_fraction == pytest.approx(0.25)


def test_nan_peaks_do_not_count_as_frames():
    peaks = _peaks(n_frames=10, saturated={1: 2})
    peaks[:5, 1] = np.nan  # 2 saturated of 5 measured = 40%
    excluded, _ = rp._saturated_over_time(peaks, LIMIT, 0)
    assert excluded == [1]


def test_frame_peaks_falls_back_to_nan_when_missing_or_mismatched():
    im = SimpleNamespace(sources=[0, 1, 2], computed={})
    assert np.isnan(rp._frame_peaks(im)).all() and len(rp._frame_peaks(im)) == 3
    im.computed["peaks"] = np.array([1.0, 2.0])
    assert np.isnan(rp._frame_peaks(im)).all()
    im.computed["peaks"] = np.array([1.0, 2.0, 3.0])
    assert rp._frame_peaks(im).tolist() == [1.0, 2.0, 3.0]


# ------------------------------- exclusion --------------------------------


def test_exclusion_strips_explicit_cids_and_falls_back_to_auto(caplog):
    avoid, cids = rp._apply_saturation_exclusion("gp", [2], [3], [2], 0, 5)
    assert avoid == [2, 3]
    assert cids is None
    assert "all explicit cIDs are saturated" in caplog.text


def test_exclusion_that_empties_the_pool_fails_the_band():
    with pytest.raises(ValueError, match="no usable comparison stars"):
        rp._apply_saturation_exclusion("gp", [1, 2], [3], None, 0, 4)


# ------------------------------- screening --------------------------------


def test_screen_excludes_late_saturating_comparison(tmp_path, caplog):
    path = _write_lco_frame(tmp_path / "f.fits")
    phot = _phot(_peaks(saturated={2: 30, 0: 10}))

    with caplog.at_level(logging.INFO, logger=rp.logger.name):
        avoid, cids, saturated, target_fraction = rp._screen_saturation_over_time(
            "gp", [path], phot, None, [1, 2], 0, 4
        )

    assert saturated == [2]
    assert avoid == [2]
    assert cids == [1]
    assert target_fraction == pytest.approx(0.10)
    assert "target 0 reaches the saturation limit" in caplog.text
    assert "in 10.0% of frames" in caplog.text


@pytest.mark.parametrize("case", ["unknown_limit", "no_peaks"])
def test_screen_skips_with_warning(tmp_path, caplog, case):
    from astropy.io import fits

    path = tmp_path / "f.fits"
    if case == "unknown_limit":
        fits.writeto(path, np.ones((10, 10)))
        phot = _phot(_peaks(saturated={2: 30}))
    else:
        _write_lco_frame(path)
        phot = SimpleNamespace(data=[SimpleNamespace()])

    with caplog.at_level(logging.WARNING, logger=rp.logger.name):
        avoid, cids, saturated, fraction = rp._screen_saturation_over_time(
            "gp", [path], phot, [3], None, 0, 4
        )

    assert (avoid, cids, saturated) == ([3], None, [])
    assert np.isnan(fraction)
    assert "per-frame saturation not screened" in caplog.text


def test_parallel_sequence_returns_frames_by_sources_peaks_outside_csv():
    """The per-source peaks survive the worker round trip as an
    (n_frames, n_sources) matrix and stay out of ``Fluxes.data``."""
    from prose import blocks, example_image
    from prose.core.sequence import SequenceParallel

    images = [example_image() for _ in range(4)]
    for i, im in enumerate(images):
        im.metadata["jd"] = 2460000.0 + i
    seq = SequenceParallel(
        [
            blocks.PointSourceDetection(n=6),
            blocks.AperturePhotometry(radii=[3, 5], scale=False),
            blocks.AnnulusBackground(rin=8, rout=12, scale=False),
            rp.MeasurePeaks(radius=5.0),
        ],
        data_blocks=[blocks.GetFluxes(_peaks=rp._frame_peaks)],
    )
    seq.run(images, show_progress=False)

    peaks = seq.data[0].values["_peaks"]
    fluxes = seq.data[0].fluxes
    assert peaks.shape == (4, fluxes.fluxes.shape[1])
    assert np.isfinite(peaks).all()
    assert "_peaks" not in fluxes.data


def test_photometry_sequence_wires_peak_getter(monkeypatch):
    # Gaussian2D needs a fitted reference ePSF; replace it, keep the rest real.
    monkeypatch.setattr(rp.blocks, "Gaussian2D", lambda ref: rp.blocks.Trim(0))
    ref = SimpleNamespace(telescope=SimpleNamespace(keyword_airmass="AIRMASS"))
    seq = rp.photometry_sequence(ref, np.array([3.0]), 5.0, 8.0, n_stars_align=0)
    assert seq.data[0].getters["_peaks"] is rp._frame_peaks


# ------------------------------ run_band wiring ------------------------------


def _run_band_with_peaks(tmp_path, monkeypatch, peaks, cids=None):
    """Run ``run_band`` (self-reference) with stubbed reference/photometry and
    return the avoid list and cIDs handed to ``differential_photometry``."""
    from astropy.coordinates import SkyCoord

    from prose import Fluxes
    from prose.core.source import Sources

    n_sources = peaks.shape[1]
    ref = SimpleNamespace(
        sources=Sources(np.array([[10.0 * i + 5, 5.0] for i in range(n_sources)])),
        shape=(20, 60),
        telescope=SimpleNamespace(jd_scale="jd"),
        header={},
        fwhm=3.0,
    )
    reference = {
        "ref": ref,
        "target_index": 0,
        "aper_radii": np.array([3.0]),
        "rin": 8.0,
        "rout": 12.0,
        "scale": False,
        "edge_cids": [],
        "gaia_df": None,
    }
    monkeypatch.setattr(rp, "build_reference", lambda *a, **kw: reference)
    data = SimpleNamespace(
        values={"_peaks": peaks},
        fluxes=Fluxes(np.ones((1, n_sources, len(peaks)))),
    )
    monkeypatch.setattr(
        rp,
        "photometry_sequence",
        lambda *a, **kw: SimpleNamespace(run=lambda files: None, data=[data]),
    )
    captured = {}

    def fake_diff(fluxes, target_index, cids=None, avoid_cids=None, **kw):
        captured.update(cids=cids, avoid=avoid_cids)
        return SimpleNamespace(time=np.arange(len(peaks), dtype=float))

    monkeypatch.setattr(rp, "differential_photometry", fake_diff)
    frame = str(_write_lco_frame(tmp_path / "science.fits"))
    result = rp.run_band("gp", [frame], frame, SkyCoord(0, 0, unit="deg"), cids=cids)
    return result, captured


def test_run_band_feeds_per_frame_saturation_into_differential_photometry(
    tmp_path, monkeypatch
):
    result, captured = _run_band_with_peaks(
        tmp_path, monkeypatch, _peaks(saturated={2: 30, 0: 5}), cids=[1, 2]
    )
    assert captured["avoid"] == [2]
    assert captured["cids"] == [1]
    assert result["saturated_cids"] == [2]
    assert result["target_saturated_fraction"] == pytest.approx(0.05)


def test_run_band_fails_when_every_comparison_saturates_over_time(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="no usable comparison stars"):
        _run_band_with_peaks(
            tmp_path, monkeypatch, _peaks(saturated={1: 30, 2: 30, 3: 30})
        )

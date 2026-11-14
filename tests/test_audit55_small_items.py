"""#55 A11 smaller items: BJD offset sign, per-band tID in summary titles,
negative source indices, pixel-scale guess, and the noise model behind Err.
(The saturation case of A11's last item is in test_audit55_saturation_frames.)"""

import argparse
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import TimeDelta

from prose.scripts import run_photometry as rp

# ------------------------------- BJD offset --------------------------------


@pytest.mark.parametrize(
    ("lttd_min", "warns"), [(-30.0, True), (30.0, True), (0.0, False)]
)
def test_bjd_offset_warns_on_magnitude_not_sign(monkeypatch, caplog, lttd_min, warns):
    times = np.array([2460000.5, 2460000.6])
    monkeypatch.setattr(
        rp.Time,
        "light_travel_time",
        lambda self, *a, **kw: TimeDelta(
            np.full(len(times), lttd_min / 1440), format="jd"
        ),
    )
    diff = SimpleNamespace(time=times)

    with caplog.at_level(logging.INFO, logger=rp.logger.name):
        rp.compute_bjd_tdb(diff, {}, SkyCoord(0, 0, unit="deg"), use_barycorrpy=False)

    warned = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "GJD->BJD offset" in r.message
    ]
    assert bool(warned) is warns


# ------------------------------ tID in titles ------------------------------


def test_target_id_label_shared_index():
    assert rp.target_id_label({"g": {"target_index": 4}, "r": {"target_index": 4}}) == 4


def test_target_id_label_names_each_band_when_indices_differ():
    label = rp.target_id_label({"g": {"target_index": 5}, "r": {"target_index": 7}})
    assert label == "g:5 r:7"


# ----------------------------- source indices ------------------------------


@pytest.mark.parametrize("flag", ["--tID", "--cID", "--avoid_cids"])
def test_negative_source_index_is_rejected_by_the_parser(flag, capsys):
    with pytest.raises(SystemExit):
        rp.parse_args(["--target_name", "x", "--data_dir", ".", flag, "-1"])
    assert "must be >= 0" in capsys.readouterr().err


def test_parse_source_index():
    assert rp.parse_source_index("0") == 0
    with pytest.raises(argparse.ArgumentTypeError, match="integer"):
        rp.parse_source_index("a")


def test_negative_avoid_cid_does_not_mask_the_last_star():
    """With explicit cIDs, a negative avoid ID must not wrap to the last star."""
    from prose import Fluxes

    rng = np.random.default_rng(0)
    raw = 1000 + rng.normal(0, 1, size=(1, 4, 50))
    fluxes = Fluxes(time=np.linspace(0, 1, 50), fluxes=raw, errors=np.ones_like(raw))

    diff = rp.differential_photometry(fluxes, 0, cids=[3], avoid_cids=[-1])

    assert list(diff.comparisons) == [3]


# ------------------------------- pixel scale -------------------------------


def test_pixel_scale_sources():
    assert rp.resolve_pixel_scale({"PIXSCALE": 0.39}, "x") == (0.39, "header PIXSCALE")
    scale, source = rp.resolve_pixel_scale({"PIXSCALE": "n/a"}, "muscat3")
    assert source == "instrument table (muscat3)"
    scale, source = rp.resolve_pixel_scale({}, "unknown")
    assert scale == rp.PIXSCALE_GUESS_ARCSEC and source.startswith("guess")


# ------------------------------- noise model -------------------------------


def _image(header, gain, read_noise, default):
    telescope = SimpleNamespace(
        name="t", keyword_gain="GAIN", keyword_read_noise="RDNOISE", _default=default
    )
    return SimpleNamespace(
        header=header, gain=gain, read_noise=read_noise, telescope=telescope
    )


def test_noise_model_from_header_is_info(caplog):
    image = _image({"GAIN": 1.5, "RDNOISE": 7.0}, 1.5, 7.0, default=True)
    with caplog.at_level(logging.INFO, logger=rp.logger.name):
        rp._log_noise_model(image, "f.fits")
    record = next(r for r in caplog.records if "Err noise model" in r.message)
    assert record.levelno == logging.INFO
    assert "gain=1.5" in record.message and "(header GAIN)" in record.message
    assert "read_noise=7 e- (header RDNOISE)" in record.message


def test_noise_model_from_generic_default_warns(caplog):
    image = _image({}, 1.0, 9.0, default=True)
    with caplog.at_level(logging.INFO, logger=rp.logger.name):
        rp._log_noise_model(image, "f.fits")
    record = next(r for r in caplog.records if "Err noise model" in r.message)
    assert record.levelno == logging.WARNING
    assert "generic default, not measured" in record.message


def test_noise_model_from_telescope_file_is_info(caplog):
    image = _image({}, 2.0, 5.0, default=False)
    with caplog.at_level(logging.INFO, logger=rp.logger.name):
        rp._log_noise_model(image, "f.fits")
    record = next(r for r in caplog.records if "Err noise model" in r.message)
    assert record.levelno == logging.INFO
    assert "t telescope file" in record.message

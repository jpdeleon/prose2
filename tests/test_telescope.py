from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits import Header

from prose import CONFIG, FITSImage, Image, Telescope


def test_creation(name="test_telescope"):
    Telescope(name=name, save=True)
    assert name in CONFIG.build_telescopes_dict().keys()


def test_custom_header_date(tmp_path):
    im = Image()
    im.header = Header()

    keyword = "OBSTIME"
    value = "2023:02:16:19:38:54.250"

    im.header[keyword] = value
    im.writeto(tmp_path / "test.fits")

    telescope = Telescope(keyword_observation_date=keyword)
    telescope.date_string_format = "%Y:%m:%d:%H:%M:%S.%f"

    im = FITSImage(tmp_path / "test.fits", load_data=False, telescope=telescope)

    assert im.date == datetime(2023, 2, 16, 19, 38, 54, 250000)


def test_fitsimage_resolves_telescope_from_header(tmp_path):
    telescope_name = "test_telescope_resolve"
    Telescope(name=telescope_name, pixel_scale=0.42, save=True)

    data = np.ones((20, 20))
    hdr = fits.Header()
    hdr["INSTRUME"] = telescope_name
    hdr["TELESCOP"] = telescope_name
    hdr["EXPTIME"] = 1.0
    hdr["DATE-OBS"] = "2025-01-01T00:00:00"
    hdr["FILTER"] = "gp"
    hdr["JD"] = 2460000.0
    hdr["RA"] = 0.0
    hdr["DEC"] = 0.0
    fpath = tmp_path / "test.fits"
    fits.writeto(fpath, data, header=hdr)

    im = FITSImage(fpath, load_data=True)
    assert im.telescope.name == telescope_name
    assert abs(im.telescope.pixel_scale - 0.42) < 1e-6


def test_fitsimage_header_keywords_do_not_modify_telescope(tmp_path):
    telescope = Telescope(
        name="test_telescope_immutable",
        pixel_scale=0.42,
        gain=2.0,
        read_noise=10.0,
        saturation=50000,
    )

    data = np.ones((20, 20))
    hdr = fits.Header()
    hdr["INSTRUME"] = "some_instrument"
    hdr["TELESCOP"] = "some_telescope"
    hdr["EXPTIME"] = 99.0
    hdr["DATE-OBS"] = "2025-06-01T12:00:00"
    hdr["FILTER"] = "ip"
    hdr["JD"] = 2460000.0
    hdr["GAIN"] = 5.0
    fpath = tmp_path / "test.fits"
    fits.writeto(fpath, data, header=hdr)

    im = FITSImage(fpath, load_data=True, telescope=telescope)
    assert im.metadata["exposure"] == 99.0
    assert im.metadata["filter"] == "ip"
    assert im.telescope.name == "test_telescope_immutable"
    assert abs(im.telescope.pixel_scale - 0.42) < 1e-6
    assert abs(im.telescope.gain - 2.0) < 1e-6
    assert abs(im.telescope.read_noise - 10.0) < 1e-6
    assert abs(im.telescope.saturation - 50000) < 1e-6


def _write_fits(tmp_path, name="f.fits", **extra):
    hdr = fits.Header()
    hdr["INSTRUME"] = "some_instrument"
    hdr["TELESCOP"] = "some_telescope"
    hdr["EXPTIME"] = 10.0
    hdr["DATE-OBS"] = "2025-06-01T12:00:00"
    hdr["FILTER"] = "ip"
    hdr["JD"] = 2460000.0
    for key, value in extra.items():
        hdr[key] = value
    fpath = tmp_path / name
    fits.writeto(fpath, np.ones((20, 20)), header=hdr)
    return fpath


def test_header_gain_and_read_noise_override_when_keywords_are_set(tmp_path):
    # BANZAI headers carry per-frame GAIN (e-/ADU, 1.0 once calibrated) and
    # RDNOISE (e-), which depend on camera unit and readout mode; a single
    # .telescope value cannot represent that.
    telescope = Telescope(
        name="test_header_noise",
        gain=1.8,
        read_noise=6.0,
        keyword_gain="GAIN",
        keyword_read_noise="RDNOISE",
    )
    fpath = _write_fits(tmp_path, GAIN=1.0, RDNOISE=12.0)

    im = FITSImage(fpath, load_data=True, telescope=telescope)

    assert im.metadata["gain"] == 1.0
    assert im.metadata["read_noise"] == 12.0
    # the telescope definition itself is never mutated by header values
    assert im.telescope.gain == 1.8
    assert im.telescope.read_noise == 6.0


def test_header_read_noise_is_per_frame(tmp_path):
    telescope = Telescope(
        name="test_header_noise_per_frame",
        read_noise=6.0,
        keyword_read_noise="RDNOISE",
    )
    a = FITSImage(_write_fits(tmp_path, "a.fits", RDNOISE=7.8), telescope=telescope)
    b = FITSImage(_write_fits(tmp_path, "b.fits", RDNOISE=12.0), telescope=telescope)

    assert a.metadata["read_noise"] == 7.8
    assert b.metadata["read_noise"] == 12.0


def test_header_noise_keywords_are_opt_in(tmp_path):
    # without keyword_gain / keyword_read_noise the header is ignored, so
    # existing .telescope definitions behave exactly as before
    telescope = Telescope(name="test_header_noise_optin", gain=2.0, read_noise=10.0)
    fpath = _write_fits(tmp_path, GAIN=1.0, RDNOISE=12.0)

    im = FITSImage(fpath, telescope=telescope)

    assert im.metadata["gain"] == 2.0
    assert im.metadata["read_noise"] == 10.0


@pytest.mark.parametrize("bad", [None, "UNKNOWN", "N/A", 0.0, -1.0])
def test_header_noise_falls_back_to_telescope_when_value_unusable(tmp_path, bad):
    telescope = Telescope(
        name="test_header_noise_fallback",
        gain=1.8,
        read_noise=6.0,
        keyword_gain="GAIN",
        keyword_read_noise="RDNOISE",
    )
    extra = {} if bad is None else {"GAIN": bad, "RDNOISE": bad}
    fpath = _write_fits(tmp_path, **extra)

    im = FITSImage(fpath, telescope=telescope)

    assert im.metadata["gain"] == 1.8
    assert im.metadata["read_noise"] == 6.0


# LCO BANZAI-reduced instruments: headers carry per-frame GAIN (e-/ADU, 1.0) and
# RDNOISE (e-), verified on real archive frames of each instrument.
BANZAI_TELESCOPE_FILES = (
    ["sinistro", "sbig", "qhy600"]
    + [f"muscat3_{b}" for b in ("gp", "rp", "ip", "zs")]
    + [f"muscat4_{b}" for b in ("gp", "rp", "ip", "zs")]
)


@pytest.mark.parametrize("name", BANZAI_TELESCOPE_FILES)
def test_banzai_telescope_files_read_noise_and_gain_from_header(name):
    import yaml
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / f"{name}.telescope"
    spec = yaml.safe_load(path.read_text())
    assert spec["keyword_gain"] == "GAIN"
    assert spec["keyword_read_noise"] == "RDNOISE"


def test_header_positive_float_rejects_nan_and_inf():
    # FITS headers cannot store NaN, but dict-like headers can
    from prose.core.image import _header_positive_float

    assert _header_positive_float({"RDNOISE": float("nan")}, "RDNOISE", 6.0) == 6.0
    assert _header_positive_float({"RDNOISE": float("inf")}, "RDNOISE", 6.0) == 6.0
    assert _header_positive_float({"RDNOISE": 7.8}, "RDNOISE", 6.0) == 7.8
    assert _header_positive_float({"RDNOISE": 7.8}, None, 6.0) == 6.0

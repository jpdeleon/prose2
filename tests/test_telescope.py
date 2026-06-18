from datetime import datetime

import numpy as np
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

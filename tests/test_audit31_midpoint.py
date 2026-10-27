import numpy as np
import pytest
from astropy.io import fits
from prose import FITSImage, Telescope, Image, blocks


@pytest.mark.parametrize(
    "keyword, value", [("MJD-OBS", 60000.0), ("MJD-STRT", 60000.0), ("JD", 2460000.5)]
)
@pytest.mark.parametrize("reference", ["auto", "start", "midpoint", "end"])
def test_midpoint_conventions(tmp_path, keyword, value, reference):
    header = fits.Header(
        {keyword: value, "EXPTIME": 120.0, "DATE-OBS": "2023-01-01T00:00:00"}
    )
    path = tmp_path / "time.fits"
    fits.writeto(path, np.ones((5, 5)), header)
    image = FITSImage(
        path, telescope=Telescope(keyword_jd=keyword, time_reference=reference)
    )
    offset = {
        "start": 60.0,
        "midpoint": 0.0,
        "end": -60.0,
        "auto": 0.0 if keyword == "JD" else 60.0,
    }[reference]
    assert image.jd == value
    assert (image.midpoint_jd - value) * 86400 == pytest.approx(offset, abs=3e-5)


def test_date_fallback_timepixr_and_getfluxes(tmp_path):
    path = tmp_path / "time.fits"
    fits.writeto(
        path,
        np.ones((5, 5)),
        fits.Header({"DATE-OBS": "2023-01-01T00:00:00", "EXPTIME": 120.0}),
    )
    image = FITSImage(path, telescope=Telescope())
    assert (image.midpoint_jd - image.jd) * 86400 == pytest.approx(60, abs=3e-5)
    image.header["TIMEPIXR"] = 0.5
    assert image.midpoint_jd == image.jd
    image.aperture = {"fluxes": np.array([[100.0]]), "radii": np.array([1.0])}
    image.annulus = {"median": np.array([1.0])}
    get = blocks.GetFluxes(time="midpoint_jd")
    get.run(image)
    get.terminate()
    assert get.fluxes.time[0] == image.jd


def test_missing_exposure_is_not_silently_assumed():
    image = Image(metadata={"jd": 60000.0, "time_keyword": "MJD-STRT"})
    with pytest.raises(ValueError, match="exposure"):
        _ = image.midpoint_jd

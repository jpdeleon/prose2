from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from prose import Image
from prose.core.source import PointSource, Sources
from prose.scripts import run_photometry as rp


def _field(sigma, ring=0):
    y, x = np.indices((240, 240))
    r = np.hypot(x - 120, y - 120)
    image = Image(3000 + 1500 * np.exp(-0.5 * ((r - ring) / sigma) ** 2))
    image.sources = Sources(
        [PointSource(coords=np.array([120, 120]), a=ring + 2 * sigma)]
    )
    return image


@pytest.mark.parametrize("sigma,ring,expected", [(12, 0, 37), (2, 12, 17)])
def test_growth_radius_and_fit_window(sigma, ring, expected):
    # For a circular Gaussian, r99 = sigma * sqrt(2 log 100).
    # A thin Gaussian ring encloses 99% just beyond its outer 2-sigma wing.
    image = _field(sigma, ring)
    assert rp._growth_radius(image, 35) == pytest.approx(expected, abs=1)
    rp.ProfileCutouts(shape=35).run(image)
    assert image.cutouts[0].data.shape[0] > 2 * expected


def test_native_widest_frame_controls_annulus(tmp_path, monkeypatch):
    ref = _field(6)
    ref.fwhm = 1800.0  # deliberately unreliable reference fit
    ref.header["PIXSCALE"] = 1.0
    ref.telescope = SimpleNamespace(pixel_scale=1)
    path = tmp_path / "wider.fits"
    fits.writeto(path, _field(12).data)
    widest = rp._widest_growth_radius(
        [path],
        "other-band.fits",
        35,
        (0, 0),
        None,
        rp.blocks.PointSourceDetection(n=5, min_area=10),
    )
    assert widest == pytest.approx(37, abs=1)
    monkeypatch.setattr(rp, "_gaia_catalog_df", lambda *a: None)
    r99 = rp._growth_radius(ref, 35)
    grid, rin, rout = rp.gaia_aperture_radii(ref, 0, None, r99=r99, widest=widest)
    assert rin >= widest + 2
    assert len(grid) > 2 and np.all(np.diff(grid) > 0)
    assert grid[0] < r99 < grid[-1] < rin < rout
